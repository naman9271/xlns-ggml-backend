# xlns-ggml-backend

GSoC 2026 code for **"Support for Logarithmic Number Systems in Large Language Models"**
(xlnsresearch / ggml + llama.cpp).

Two ggml backends built on the [xlnscpp](https://github.com/xlnsresearch/xlnscpp) library:

- **LNS16** — 16-bit LNS (xlns16) arithmetic, with a compile-time choice of
  summation algorithm (`base` / `monte` / `lpvip`) for the reduction-heavy ops
  (softmax, rms_norm, and the vec_dot inside mat-mul).
- **LNS32** — 32-bit LNS (xlns32) arithmetic, structural parity port of the
  same op set.

---

## Reproducing the end-to-end inference result

See [SETUP.md](SETUP.md) for full instructions.

Tested models: SmolLM2-135M and Llama-3.2-1B are coherent on LNS16/LNS32 at
temp 0 (continuations diverge from the pure-CPU reference within a few tokens
— xlns16 rounding flips some argmaxes — but stay coherent).

Larger qwen2-family models (qwen2.5-1.5B, DeepSeek-R1-Distill-Qwen-1.5B)
degenerate on LNS16 with the default `base` summation, and the fix is the
**summation mode, not the storage format**: with `-DGGML_LNS16_SUM=sum32` (a
pure xlns32 accumulator inside the three reduction ops, every value still
stored as 7-fraction-bit xlns16) both become coherent, and Llama-3.2-1B
becomes token-identical to the CPU reference. Measured on
DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M, first-token logits vs the pure-CPU
reference (prompt "The capital of France is"):

| LNS16 mode | logit rms | corr | rank of `" Paris"` | argmax |
|---|---|---|---|---|
| `base` | 2.099 | 0.447 | 2807 | wrong |
| `sum32` | **0.603** | **0.953** | **0** | `" Paris"` |
| (LNS32 backend) | 0.167 | 0.998 | 0 | `" Paris"` |

Graded across 3 prompts x 2 models rather than on a single token, sum32 is
coherent in **6/6** prompt-model combinations where `base` is degenerate in
**0/6** (`results/2026-08-20/robust-sum32.txt`). It is not a free lunch: the
residual error is non-zero, and where the reference has a narrow margin the
16-bit *storage* grid can still collapse it — on "The largest planet in our
solar system is" the reference separates `" Jupiter"` from `" Earth"` by 0.82
logits, while on LNS16 both land on the same xlns16 code (gap 0.000) and the
tie-break emits `" Earth"`. No accumulator width can fix that; it is the
7-fraction-bit format itself. sum32 also costs ~1.7x in throughput.

Artefacts and the grading instruments are in `results/2026-08-20/` and
`scripts/rc-logits.cpp` + `scripts/logit-compare.py`. See
[SETUP.md](SETUP.md#model-compatibility-the-deepseek-r1-distill-qwen-15b-case)
for the full evidence matrix, the earlier `base`-mode bisection, and a warning
about what counts as a true CPU reference.

---

## Repository structure

```
lns-backend/
  include/      Public API headers (ggml-lns.h, ggml-lns32.h)
  src/          LNS16 backend  (ggml-lns.cpp, lns-ops.h, lns-ops.cpp, CMakeLists.txt)
  src32/        LNS32 backend  (ggml-lns32.cpp, lns32-ops.h, lns32-ops.cpp, CMakeLists.txt)
  test-lns-backend.cpp   Validation tests (runs against either backend)
```

---

## Dependencies

| Dependency | Purpose |
|---|---|
| [xlnscpp](https://github.com/xlnsresearch/xlnscpp) | C++ LNS library — included as a **git submodule** |
| [ggml](https://github.com/ggml-org/ggml) | Tensor runtime for the backend and unit test |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | End-to-end LLM inference validation |

xlnscpp is included as a git submodule for the initial checkout.  `setup.sh`
initialises it if needed, then fetches latest `origin/main`.  If the build
fails against that HEAD, it falls back to the verified-good commit in
`patches/xlnscpp-base-commit.txt`.

For manual builds without `setup.sh`, populate the submodule first:

```bash
git submodule update --init
```

This ensures `#include "xlns16.cpp"` resolves correctly in the backend.

---

## LNS16 backend

A proper `ggml_backend` (taking inspiration from the BLAS and CPU backends) that performs all
supported ops in **xlns16** arithmetic.  It registers as the `"LNS16"` device
with `GGML_BACKEND_DEVICE_TYPE_ACCEL`.

Compilation uses `xlns16_table` (O(1) float↔xlns16 via a 65536-entry lookup
table) and `xlns16_alt` (reduced-branching addition path).  `xlns16_ideal` is
not defined — this matches a realistic hardware target rather than ideal
arithmetic.

### Summation variants (`GGML_LNS16_SUM`)

Long xlns16 summations "stick": once the partial sum dwarfs each new addend,
`xlns16_add` cannot represent the increment and the result is systematically
biased (see `xlnscpp/summation16bit.MD`).  The reduction inside softmax
(normalizer), rms_norm (sum of squares), and mat-mul vec_dot can therefore be
switched between four implementations at build time:

```bash
cmake .. -DGGML_LNS=ON -DGGML_LNS16_SUM=sum32   # base (default xlns16) | monte | lpvip | sum32
```

| Mode | Implementation | Character |
|---|---|---|
| `base` | sequential `xlns16_add` (`xlns16.cpp`) | pure 16-bit; ~4% bias observed on an n=576 sum of squares |
| `monte` | Monte-Carlo LNS dither (`xlns16monte.cpp`) | pure 16-bit; unbiased in expectation, individual reductions can err more than base |
| `lpvip` | 32-bit LPVIP accumulator (`xlns32lpvip.cpp`) | ~5× better than base on same-sign long sums (0.77% on the same n=576 case), cheaper than a full xlns32 accumulator — but `xlns32_add_lpvip` is inaccurate on mixed-sign cancellation, so it is *worse* than base inside mat-mul (57% vs 4.4% on a K=1536 dot) |
| `sum32` | pure xlns32 accumulator, rounded back to xlns16 (`lns-backend/src/lns16-sum32.h`) | the accuracy ceiling: two-term sums come out correctly rounded and the ~370-term stagnation disappears (K=1536 dot 0.31%, n=576 sum of squares 0.45%). Also the slowest — see the throughput note below |

`sum32` is the option `xlnscpp/summation16bit.MD` describes but does not ship
("one could sum such `xlns16` values using an `xlns32` accumulator, but that is
very expensive").  Each term is widened to xlns32 — exact, the formats are
bit-aligned — accumulated with `xlns32_add`, and the accumulator is rounded
back to xlns16 with round-to-nearest at the end.  `xlns32_add` is the *only*
xlns32 arithmetic involved: operands, results and storage stay xlns16
throughout, and each reduction narrows exactly once.  It uses the same `xlns32_alt` configuration as the
LNS32 backend (table + interpolation `sb`, cotransformation `db`; no
`xlns32_ideal`).  Measured cost: roughly 1.7x slower than `base`
end-to-end on SmolLM2-135M.

Numbers from `test-lns-backend` (the `mul mat long` case, K=1536 mixed-sign —
the only case in the suite that can see an accumulator at all; every mode
scores an identical 0.59% on the K=2 `mul mat`):

| | base | monte | lpvip | sum32 | LNS32 |
|---|---|---|---|---|---|
| `mul mat long` NRMSE | 4.39% | 4.72% | 57.21% | **0.31%** | 0.012% |
| `rms_norm 576` | 4.07% | 5.68% | 0.77% | **0.45%** | 0.0012% |

`monte` has a second compile-time knob, `-DGGML_LNS16_MONTE_BYLPVIP=ON`, which
swaps the pure-16-bit LFSR implementation for the `xlns32_add_lpvip`-based one
(`xlns16monte_bylpvip.cpp`).  The two define identical symbols, so this choice
is inherently compile-time.

Only these three ops contain reductions; every other implemented op is
elementwise.  If layernorm (`GGML_OP_NORM`) or `SUM`/`SUM_ROWS`/`MEAN` are
added later, xlnscpp already ships `_monte`/`_lpvip32` versions of them, and
`lns16-sum32.h` ships `xlns16_layernorm_sum32` / `xlns16_sum_sum32`.

The `sum32` family lives in `lns-backend/src/lns16-sum32.h` rather than in
xlnscpp for now; it is written to move to `xlnscpp/xlns16sum32.cpp` verbatim
once upstreamed.  Its invariants (exact widen/narrow round-trip, no
stagnation, long mixed-sign dots) are checked by a standalone program that
needs no ggml:

```bash
g++ -O2 -std=c++17 -I xlnscpp -I lns-backend/src lns-backend/test-lns16-sum32.cpp -o /tmp/test-lns16-sum32 && /tmp/test-lns16-sum32
```

## LNS32 backend

A parallel backend (`"LNS32"` device, option `GGML_LNS32`) porting the same op
set to **xlns32** (1 sign / 8 int / 23 frac log bits — F32-class accuracy,
unit tests pass at 0.1% tolerance).  Built with `xlns32_alt`; `xlns32_ideal`
is not defined.

Two structural caveats:

- **No conversion table exists for xlns32** (`fp2xlns32` is a libm `log()` per
  element), and weights are converted per-op, so inference is much slower than
  LNS16.  A per-tensor weight cache / custom buffer type is documented future
  work.
- There is no `GGML_TYPE_LNS32`; all tensors flow as F32.

### Backend selection at runtime

Both backends register as ACCEL devices sharing the CPU buffer type, so the
scheduler would give every op to whichever registered first.  The
`GGML_LNS_BACKEND` env var (`16` default, or `32`) decides which backend
claims ops; the other declines with audit reason `disabled_by_env`
(observable with `LNS_OP_AUDIT=1`).

### Core ggml changes (applied automatically by `setup.sh`)

| File | Change |
|---|---|
| `ggml/include/ggml.h` | Insert `GGML_TYPE_LNS16` at the current `GGML_TYPE_COUNT` value, then bump `GGML_TYPE_COUNT` by 1 |
| `ggml/src/ggml.c` | Register LNS16 type traits (`blck_size=1`, `type_size=2`, not quantized) |
| `ggml/CMakeLists.txt` | Add `GGML_LNS` and `GGML_LNS32` CMake options (default OFF) |
| `ggml/src/CMakeLists.txt` | Add `ggml_add_backend(LNS)` and `ggml_add_backend(LNS32)` |
| `ggml/src/ggml-backend-reg.cpp` | Add `#ifdef GGML_USE_LNS` / `#ifdef GGML_USE_LNS32` includes and registrations |

`setup.sh` fetches latest upstream for xlnscpp (`origin/main`), ggml, and
llama.cpp (`origin/master`).  For ggml and llama.cpp it applies LNS hooks via
regex (`scripts/integrate-lns.sh`), falling back to `patches/*.patch` if needed.
Verified-good commits for all three are recorded in `patches/*-base-commit.txt`.

### Build

```bash
cd ggml/build
cmake .. -DGGML_LNS=ON -DGGML_LNS32=ON -DGGML_LNS16_SUM=base
cmake --build . -j4

# Validate (per backend; second arg = tolerance profile, which MUST match the
# GGML_LNS16_SUM the binary was built with — a mismatch is refused, not graded)
GGML_LNS_BACKEND=16 ./bin/test-lns-backend LNS16 base
GGML_LNS_BACKEND=32 ./bin/test-lns-backend LNS32
```

### End-to-end inference (SmolLM2-135M-Instruct, Q4_K_M)

```bash
# This assumes SmolLM2 .gguf is in the repo root; setup.sh uses it by default
./setup.sh                       # LNS16 (base) + a short LNS32 run
./setup.sh --sum-mode=lpvip      # LNS16 with LPVIP summation
./setup.sh --sum-mode=sum32      # LNS16 with a pure xlns32 accumulator

# Or with Llama 3.2 1B as well
MODEL_LLAMA32=./Llama-3.2-1B-Instruct-Q4_K_M.gguf ./setup.sh

# After setup: re-run without rebuilding (see SETUP.md for flags)
./run.sh
./run.sh --backend=32
```

`scripts/validate-latest.sh` runs the full gate against latest upstream
ggml/llama.cpp: unit tests for all four sum modes plus LNS32, SmolLM2
inference in every configuration, then Llama 3.2 1B (LNS16). Both GGUFs are
required — see [SETUP.md](SETUP.md).

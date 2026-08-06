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
switched between three implementations at build time:

```bash
cmake .. -DGGML_LNS=ON -DGGML_LNS16_SUM=lpvip   # base (default xlns16) | monte | lpvip
```

| Mode | Implementation | Character |
|---|---|---|
| `base` | sequential `xlns16_add` (`xlns16.cpp`) | pure 16-bit; ~4% bias observed on an n=576 sum of squares |
| `monte` | Monte-Carlo LNS dither (`xlns16monte.cpp`) | pure 16-bit; unbiased in expectation, individual reductions can err more than base |
| `lpvip` | 32-bit LPVIP accumulator (`xlns32lpvip.cpp`) | ~5× better than base on long sums (0.77% on the same n=576 case), cheaper than a full xlns32 accumulator |

`monte` has a second compile-time knob, `-DGGML_LNS16_MONTE_BYLPVIP=ON`, which
swaps the pure-16-bit LFSR implementation for the `xlns32_add_lpvip`-based one
(`xlns16monte_bylpvip.cpp`).  The two define identical symbols, so this choice
is inherently compile-time.

Only these three ops contain reductions; every other implemented op is
elementwise.  If layernorm (`GGML_OP_NORM`) or `SUM`/`SUM_ROWS`/`MEAN` are
added later, xlnscpp already ships `_monte`/`_lpvip32` versions of them.

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

# Validate (per backend; second arg = tolerance profile matching GGML_LNS16_SUM)
GGML_LNS_BACKEND=16 ./bin/test-lns-backend LNS16 base
GGML_LNS_BACKEND=32 ./bin/test-lns-backend LNS32
```

### End-to-end inference (SmolLM2-135M-Instruct, Q4_K_M)

```bash
# This assumes SmolLM2 .gguf is in the repo root; setup.sh uses it by default
./setup.sh                       # LNS16 (base) + a short LNS32 run
./setup.sh --sum-mode=lpvip      # LNS16 with LPVIP summation

# Or with Llama 3.2 1B as well
MODEL_LLAMA32=./Llama-3.2-1B-Instruct-Q4_K_M.gguf ./setup.sh

# After setup: re-run without rebuilding (see SETUP.md for flags)
./run.sh
./run.sh --backend=32
```

`scripts/validate-latest.sh` runs the full gate against latest upstream
ggml/llama.cpp: unit tests for all three sum modes plus LNS32, SmolLM2
inference in every configuration, then Llama 3.2 1B (LNS16). Both GGUFs are
required — see [SETUP.md](SETUP.md).

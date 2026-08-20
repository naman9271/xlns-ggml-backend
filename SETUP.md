# Reproducing the Results

## Prerequisites

- `git`, `cmake`, `g++`, `make`, `perl`
- ~2 GB free disk space (llama.cpp clone + build)

## Steps

**1. Clone and initialise**
```bash
git clone https://github.com/abracodeabra1/xlns-ggml-backend-final
cd xlns-ggml-backend-final
git submodule update --init   # optional — as setup.sh also initialises xlnscpp
```

`setup.sh` will then advance `xlnscpp/` to latest `origin/main` (falling back to the pinned commit in `patches/xlnscpp-base-commit.txt` if the build fails).

**2. Download the model**

Download `SmolLM2-135M-Instruct-Q4_K_M.gguf`:

```bash
# Using huggingface-cli, in the newer versions, use the hf command instead of huggingface-cli (pip install huggingface_hub if needed)
huggingface-cli download bartowski/SmolLM2-135M-Instruct-GGUF \
    SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir .
```

**3. Run**
```bash
# End-to-end LNS inference on SmolLM2 (default model path is the repo-root .gguf)
# Runs LNS16 (n=20 tokens) plus a short LNS32 run (n=4; disable with LNS32_INFERENCE=0)
./setup.sh

# Unit test + SmolLM2 inference (Llama 3.2 only if MODEL_LLAMA32 is set)
./setup.sh --unit-test

# Unit test only (no model needed) — runs the LNS16 suite and the LNS32 suite
./setup.sh --unit-test --skip-inference

# Select the xlns16 summation variant (compile-time): base (default) | monte | lpvip | sum32
./setup.sh --sum-mode=lpvip

# Explicit model paths (MODEL_LLAMA32 enables the Llama 3.2 1B inference run)
MODEL_SMOLLM=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf \
MODEL_LLAMA32=/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
./setup.sh
```

`MODEL=/path/to/model.gguf` remains supported as an alias for `MODEL_SMOLLM`.

At runtime, `GGML_LNS_BACKEND=16|32` (default `16`) selects which of the two
backends claims ops — both are always built. `LNS_OP_AUDIT=1` prints one JSON
line per `supports_op` query (including which backend answered and, for LNS16,
the compiled sum mode).

## Re-run inference

After a successful `./setup.sh`, re-run the model without rebuilding:

```bash
./run.sh                          # LNS16, default SmolLM2 + prompt
./run.sh --backend=32             # LNS32; default -n 4
./run.sh -m ./other_model_file.gguf -n 40
./run.sh -p "Hello" -- --temp 0.2 # extra args after -- go to llama-completion
```

`run.sh` uses `build/llama.cpp/build-lns/bin/llama-completion`. If that binary
is missing, it tells you to run `./setup.sh` first. Summation mode
(`base` / `monte` / `lpvip` / `sum32`) is compile-time only — change it with
`./setup.sh --sum-mode=...`, not `./run.sh`.

### `run.sh` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--backend=16\|32` | `16` (or `$GGML_LNS_BACKEND`) | Which LNS backend claims ops |
| `-m`, `--model PATH` | `$MODEL` / `$MODEL_SMOLLM` / repo-root SmolLM2 GGUF | Model file |
| `-p`, `--prompt TEXT` | `The capital of France is` | Prompt |
| `-n`, `--tokens N` | `20` (LNS16) or `4` (LNS32) | Tokens to predict |
| `-h`, `--help` | — | Show usage |
| `-- …` | — | Forward remaining args to `llama-completion` |

Always passed to `llama-completion`: `-fa off -no-cnv`.

Useful env vars: `GGML_LNS_BACKEND`, `MODEL` / `MODEL_SMOLLM`, `LNS_OP_AUDIT=1`.

### Important `llama-completion` flags

Verified from `build/llama.cpp/build-lns/bin/llama-completion --help`. Pass any
of these after `--` on `./run.sh`, or call the binary directly. Full list:
`llama-completion --help`.

| Flag | Meaning |
|------|---------|
| `-m`, `--model FNAME` | Model path to load |
| `-p`, `--prompt PROMPT` | Prompt to start generation with |
| `-n`, `--predict`, `--n-predict N` | Tokens to predict (`-1` = infinity) |
| `-c`, `--ctx-size N` | Prompt context size (`0` = from model) |
| `-t`, `--threads N` | CPU threads for generation |
| `-fa`, `--flash-attn [on\|off\|auto]` | Flash Attention (`run.sh` sets `off`) |
| `-cnv` / `-no-cnv` | Conversation mode on/off (`run.sh` forces `-no-cnv`) |
| `-sys`, `--system-prompt PROMPT` | System prompt (chat template) |
| `-f`, `--file FNAME` | Read prompt from a file |
| `-s`, `--seed SEED` | RNG seed (`-1` = random) |
| `--temp N` | Temperature (default `0.80`) |
| `--top-k N` / `--top-p N` / `--min-p N` | Sampling controls |

## Model compatibility: the DeepSeek-R1-Distill-Qwen-1.5B case

`DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` (qwen2 architecture) produces
incoherent output on LNS16 **with the default `base` summation** while
SmolLM2-135M and Llama-3.2-1B stay coherent on the same backend.

> **Update (2026-08-20): this is a summation problem, and
> `-DGGML_LNS16_SUM=sum32` fixes it.** With a pure xlns32 accumulator inside
> the three reduction ops — every operand and every result still stored as
> 7-fraction-bit xlns16 — DeepSeek is correct: `" Paris"` is the argmax
> (rank 0), logit rms vs the CPU reference falls 2.099 → 0.603 and correlation
> rises 0.447 → 0.953. qwen2.5-1.5B is likewise rescued from degenerate to
> coherent, and Llama-3.2-1B becomes token-identical to the CPU reference.
> Graded across 3 prompts x 2 models rather than on one token (the mistake this
> document warns about): **sum32 is coherent in 6/6 combinations, `base` in
> 0/6** — see `results/2026-08-20/robust-sum32.txt`. Residual error remains, and
> narrow reference margins can still be lost to the 16-bit *storage* grid rather
> than to summation: on "The largest planet…" the reference separates
> `" Jupiter"` from `" Earth"` by 0.82 logits while LNS16 puts both on the same
> xlns16 code, so the tie-break emits `" Earth"`. Cost: 1.66x slower than `base`.
> All artefacts are in `results/2026-08-20/`.
>
> The text below is the 2026-08-18 analysis of the `base` mode and is kept
> because its bisection method and its warnings still apply. Its conclusion
> that operand quantization is the limiting factor was **too pessimistic**:
> the earlier "a wide accumulator is not enough" experiment
> (`LNS_MM_ROUNDTRIP=1`) only widened `lns_mul_mat`, re-snapped each dot to
> the xlns16 grid, and had to route `RMS_NORM`/`SOFT_MAX` to CPU rather than
> accumulate them accurately in-format. `sum32` widens all three reductions at
> once and clears the bar.

Conclusion of the original `base`-mode bisection: **the cause is LNS16's
operand quantization (~0.3–0.5% relative error per matmul operand from
xlns16's 7 fraction bits), which this model cannot tolerate. It is a precision
limit of the 16-bit format on this particular model, not a bug in any LNS
kernel.**

> **Warning: what counts as a "CPU reference".** Inside the LNS-enabled
> `llama-completion` binary, neither `--no-op-offload` nor `--device none`
> gives you CPU-only execution. `--no-op-offload` only disables the
> weight-buffer offload path in `ggml-backend.cpp`, and `--device none` only
> affects layer offload; the scheduler still assigns ops to LNS16/LNS32/BLAS
> (verify with `GGML_SCHED_DEBUG=2 ... -v` and look at the `SPLIT #n: BACKEND`
> lines). The only trustworthy CPU reference is a separate build with
> `-DGGML_LNS=OFF -DGGML_LNS32=OFF -DGGML_BLAS=OFF` (this repo keeps one at
> `build/llama.cpp/build-ref`; `--list-devices` on that binary prints
> "(none)"). An earlier version of this section compared LNS16 against a
> contaminated "CPU" run that was itself mostly LNS16 — the two matched
> bit-for-bit precisely because both were the same backend, and the wrong
> conclusion ("not an LNS bug") followed.

Evidence (all runs greedy: `--temp 0 --seed 1`, `-fa off`, raw prompt
"The capital of France is"; CPU = build-ref):

| Model / config | CPU reference (build-ref) | LNS16 | LNS32 |
|---|---|---|---|
| SmolLM2-135M, n=20 | "Paris. Paris is a city that is known for ..." | "Paris. Paris is the political, ..." — coherent, diverges at token ~6 | — |
| Llama-3.2-1B, n=20 | "Paris. The Eiffel Tower is ..." | "Paris, the capital of Belgium ..." — coherent, diverges early | — |
| DeepSeek-Qwen-1.5B, n=40 | "Paris, the capital of Italy is Rome, ..." (fully coherent) | "not the only one\" feature ..." (incoherent) | "Paris, the capital" (correct) |
| DeepSeek, chat template (`-cnv --jinja`), n=60 | "\<think>\</think> The capital of France is Paris." (correct) | degenerate loop | "The capital of France is Paris." (correct) |

Bisection (LNS16, DeepSeek, which op or operand breaks it):

| Experiment | Result |
|---|---|
| `LNS_DISABLE_OPS=ADD` / `RMS_NORM` / `SOFT_MAX` / `ROPE` | still incoherent |
| `LNS_DISABLE_OPS=MUL_MAT` (matmuls on CPU, all else LNS16) | **coherent** |
| `LNS_DISABLE_OPS=<everything except MUL_MAT>` (only matmuls on LNS16) | degenerate |
| Only `result_output` / `ffn_down` / `Qcur,Kcur,Vcur,kqv_out` / `ffn_up,ffn_gate` matmuls on CPU | still incoherent (no single matmul group is responsible) |
| `GGML_LNS16_SUM=lpvip` or `=monte` build | still incoherent |
| `GGML_LNS16_SUM=sum32` build (2026-08-20) | **coherent** — "Paris, the capital of Spain is Madrid, the capital of Italy is Rome," |
| `LNS_MM_ROUNDTRIP=1` (operands xlns16-rounded, dot accumulated in f32) | degraded |
| `LNS_MM_ROUNDTRIP=2` (only weights rounded to xlns16, f32 dot) | degraded |
| `LNS_MM_ROUNDTRIP=3` (only activations rounded to xlns16, f32 dot) | degraded |

Interpretation:

- `lns_mul_mat` is the culprit class, and no single matmul group (attn / FFN /
  logits) or accumulation mode is decisive — the damage is the ~0.3–0.5%
  per-value rounding that xlns16 (1 sign + 8 exp + 7 fraction bits) applies to
  **both** operands of every matmul, compounding over 28 layers. A standalone
  roundtrip check of `fp2xlns16`/`xlns162fp` over [1e-4, 10] shows max rel
  error 0.54%, mean 0.27%, no underflow/saturation/sign bugs — the conversion
  itself is sound; the model is simply that sensitive.

Why does DeepSeek break while SmolLM2/Llama survive? A decisive control
experiment, completely outside the LNS backend: rewrite the GGUF with the
weights rounded through an emulated xlns16 grid (or simply converted by
llama.cpp itself) and run on the pure-CPU build-ref:

| Model variant (all on pure CPU) | Result |
|---|---|
| SmolLM2, weights xlns16-rounded → F16 | "Paris. Paris is a city known for ..." (coherent, ≈ CPU reference) |
| Llama-3.2-1B, weights xlns16-rounded → F16 | **byte-identical to CPU reference** |
| DeepSeek, weights xlns16-rounded → F16 | "Paris,@@@@@@..." (degenerate), PPL → NaN |
| DeepSeek, Q4_K→F16 via `llama-quantize --allow-requantize` (no extra rounding) | same degenerate output, PPL → NaN |
| DeepSeek, Q4_K→Q8_0 requantize (finer grid, even with `--imatrix`) | same degenerate output, PPL → NaN |

So the difference is in the models, not the kernels: the shipped
DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M occupies a measure-zero region of weight
space — *any* re-representation of its weights (even to the strictly finer
F16/Q8_0) destroys it, while SmolLM2 and Llama-3.2-1B tolerate the same
treatment without blinking. Q4_K's per-block min/max structure apparently
preserves exactly the directions this distilled model depends on; xlns16's
~0.3% per-value grid does not. Llama survives because its function is
genuinely smooth under weight perturbation (F16 conversion is byte-identical
in output); DeepSeek's is not.

- LNS32 (xlns32, ~24 fraction bits) is near-f32 and is fully coherent and
  correct on this model. Use LNS32 for precision-sensitive models.

Debug/bisection knobs (compiled into the LNS16 backend):

- `LNS_DISABLE_OPS="ADD,ROPE,..."` — force listed ops (by ggml op name, or by
  substring of the tensor name, e.g. `ffn_down`, `result_output`) to fall back
  to CPU. No rebuild needed.
- `LNS_MM_ROUNDTRIP=1|2|3` — in `lns_mul_mat`, round operands to xlns16 but
  accumulate the dot product in IEEE float (1=both operands, 2=weights only,
  3=activations only). Isolates operand quantization from LNS accumulation.
- `LNS_OP_AUDIT=1` — log every `supports_op` query as JSON on stderr.
- `GGML_SCHED_DEBUG=2` + `-v` — ggml prints which backend every graph node is
  assigned to; the only reliable way to check what actually ran where.

Practical guidance:

- For DeepSeek-class (precision-sensitive) models, run `GGML_LNS_BACKEND=32`.
- Always run reasoning-distill models with their chat template
  (`-cnv --jinja`); raw-completion prompts are off-distribution.
- To verify a suspected LNS regression quantitatively, use the quality gate
  (uses the build-ref pure-CPU binaries as reference):

```bash
# greedy divergence report + perplexity delta vs the CPU reference
./scripts/lns-vs-cpu-quality.sh -m ./DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
# quick SmolLM2-only check (default model), divergence only:
./scripts/lns-vs-cpu-quality.sh --skip-ppl
```

Note: perplexity on the LNS backends is single-threaded — a 256-token chunk
on a 1.5B model takes on the order of 20 minutes on LNS16.

## Upstream integration

`setup.sh` always fetches latest upstream for all three dependencies:

| Dependency | Branch | Integration |
|---|---|---|
| [xlnscpp](https://github.com/xlnsresearch/xlnscpp) | `origin/main` | submodule init + `git reset --hard` |
| [ggml](https://github.com/ggml-org/ggml) | `origin/master` | regex edits in [`scripts/integrate-lns.sh`](scripts/integrate-lns.sh) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `origin/master` | regex edits in [`scripts/integrate-lns.sh`](scripts/integrate-lns.sh) |

If regex integration (ggml/llama.cpp) or the subsequent build fails, it falls back to the pinned patches in `patches/` (verified-good commits in `patches/*-base-commit.txt`). If the build fails against latest xlnscpp, it falls back to `patches/xlnscpp-base-commit.txt`.

To re-validate all three against latest upstream before refreshing the fallback pins
(requires both GGUFs — defaults to the repo-root SmolLM2 and Llama 3.2 1B paths):
```bash
MODEL_SMOLLM=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf \
MODEL_LLAMA32=/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
./scripts/validate-latest.sh
```

The gate runs unit tests for all four LNS16 sum modes plus LNS32, SmolLM2
inference in every configuration, then a Llama 3.2 1B LNS16 run.

## What is in this repo

| Path | Contents |
|------|----------|
| `lns-backend/` | LNS backend sources — `src/` (LNS16: `ggml-lns.cpp`, `lns-ops.cpp`), `src32/` (LNS32: `ggml-lns32.cpp`, `lns32-ops.cpp`), `include/`, and the unit test |
| `xlnscpp/` | xlnscpp submodule (16-bit and 32-bit LNS library); `setup.sh` advances to latest `main` |
| `patches/` | Fallback patches + pinned base commits for xlnscpp, ggml, and llama.cpp |
| `scripts/` | `integrate-lns.sh` (regex integration), `validate-latest.sh` (compatibility gate), `lns-vs-cpu-quality.sh` (LNS-vs-CPU token-match + perplexity gate) |
| `setup.sh` | Automated build and run script |
| `run.sh` | Re-run inference after a successful setup (no rebuild) |

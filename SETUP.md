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

# Select the xlns16 summation variant (compile-time): base (default) | monte | lpvip
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
(`base` / `monte` / `lpvip`) is compile-time only — change it with
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

The gate runs unit tests for all three LNS16 sum modes plus LNS32, SmolLM2
inference in every configuration, then a Llama 3.2 1B LNS16 run.

## What is in this repo

| Path | Contents |
|------|----------|
| `lns-backend/` | LNS backend sources — `src/` (LNS16: `ggml-lns.cpp`, `lns-ops.cpp`), `src32/` (LNS32: `ggml-lns32.cpp`, `lns32-ops.cpp`), `include/`, and the unit test |
| `xlnscpp/` | xlnscpp submodule (16-bit and 32-bit LNS library); `setup.sh` advances to latest `main` |
| `patches/` | Fallback patches + pinned base commits for xlnscpp, ggml, and llama.cpp |
| `scripts/` | `integrate-lns.sh` (regex integration), `validate-latest.sh` (compatibility gate) |
| `setup.sh` | Automated build and run script |
| `run.sh` | Re-run inference after a successful setup (no rebuild) |

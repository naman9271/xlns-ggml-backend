#!/usr/bin/env bash
# run.sh — re-run LNS inference after a successful ./setup.sh build.
#
# Usage:
#   ./run.sh                          # LNS16, base sum mode, default model/prompt
#   ./run.sh --sum-mode=sum32         # pick a summation-variant binary
#   ./run.sh --sum-mode=all           # run the same prompt across all four variants
#   ./run.sh --backend=32             # LNS32; default -n 4
#   ./run.sh -m ./other.gguf -n 40
#   ./run.sh -p "Hello" -- --temp 0.2 # extra args after -- go to llama-completion
#
# Summation mode is compile-time: ./setup.sh builds one binary per mode under
# build/llama.cpp/build-lns-{base,monte,lpvip,sum32}; this script only selects
# among them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LLAMA_BUILD_ROOT="$REPO_ROOT/build/llama.cpp"
DEFAULT_MODEL="$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf"
ALL_MODES=(base monte lpvip sum32)

BACKEND="${GGML_LNS_BACKEND:-16}"
MODEL="${MODEL:-${MODEL_SMOLLM:-$DEFAULT_MODEL}}"
SUM_MODE="${LNS16_SUM:-base}"
PROMPT="The capital of France is"
N_TOKENS=""
N_TOKENS_SET=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./run.sh [options] [-- llama-completion-args...]

Re-run LNS inference using the binaries built by ./setup.sh.
Does not rebuild; each summation mode is a separate compile-time binary.

Options:
  --sum-mode=MODE     base|monte|lpvip|sum32|all (default: base, or $LNS16_SUM)
                      'all' runs the same prompt through every variant in turn
  --backend=16|32     LNS backend (default: 16, or $GGML_LNS_BACKEND)
  -m, --model PATH    GGUF model path (default: $MODEL / $MODEL_SMOLLM /
                      SmolLM2-135M-Instruct-Q4_K_M.gguf in repo root)
  -p, --prompt TEXT   Prompt (default: "The capital of France is")
  -n, --tokens N      Tokens to predict (default: 20 for LNS16, 4 for LNS32)
  -h, --help          Show this help

Environment:
  LNS16_SUM           Same as --sum-mode
  GGML_LNS_BACKEND    Same as --backend (16 or 32)
  MODEL / MODEL_SMOLLM
                      Default model path
  LNS_OP_AUDIT=1      Print one JSON line per supports_op query

Examples:
  ./run.sh
  ./run.sh --sum-mode=sum32
  ./run.sh --sum-mode=all -n 8
  ./run.sh --backend=32
  ./run.sh -m ./Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 40
  ./run.sh -p "Hello" -- --temp 0.2 --seed 1

Pass-through: anything after -- is forwarded to llama-completion.
See SETUP.md for important llama-completion flags, or run:
  ./build/llama.cpp/build-lns/bin/llama-completion --help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --sum-mode=*)
            SUM_MODE="${1#*=}"
            shift
            ;;
        --sum-mode)
            if [ $# -lt 2 ]; then
                echo "ERROR: --sum-mode requires base|monte|lpvip|sum32|all" >&2
                exit 1
            fi
            SUM_MODE="$2"
            shift 2
            ;;
        --backend=*)
            BACKEND="${1#*=}"
            shift
            ;;
        --backend)
            if [ $# -lt 2 ]; then
                echo "ERROR: --backend requires 16 or 32" >&2
                exit 1
            fi
            BACKEND="$2"
            shift 2
            ;;
        -m|--model)
            if [ $# -lt 2 ]; then
                echo "ERROR: $1 requires a path" >&2
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -p|--prompt)
            if [ $# -lt 2 ]; then
                echo "ERROR: $1 requires a prompt string" >&2
                exit 1
            fi
            PROMPT="$2"
            shift 2
            ;;
        -n|--tokens)
            if [ $# -lt 2 ]; then
                echo "ERROR: $1 requires a number" >&2
                exit 1
            fi
            N_TOKENS="$2"
            N_TOKENS_SET=1
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        -*)
            echo "ERROR: unknown option '$1' (use -- to forward args to llama-completion)" >&2
            echo "Run ./run.sh --help for usage." >&2
            exit 1
            ;;
        *)
            echo "ERROR: unexpected argument '$1'" >&2
            echo "Run ./run.sh --help for usage." >&2
            exit 1
            ;;
    esac
done

case "$SUM_MODE" in
    base|monte|lpvip|sum32|all) ;;
    *)
        echo "ERROR: --sum-mode must be base, monte, lpvip, sum32, or all (got '$SUM_MODE')" >&2
        exit 1
        ;;
esac

case "$BACKEND" in
    16|32) ;;
    *)
        echo "ERROR: --backend must be 16 or 32 (got '$BACKEND')" >&2
        exit 1
        ;;
esac

if [ "$N_TOKENS_SET" -eq 0 ]; then
    if [ "$BACKEND" = "32" ]; then
        N_TOKENS=4
    else
        N_TOKENS=20
    fi
fi

# Locate the binary for one sum mode; falls back to the legacy single-build
# path (build-lns) when the per-mode directory does not exist yet.
bin_for_mode() {
    local mode="$1"
    local bin="$LLAMA_BUILD_ROOT/build-lns-$mode/bin/llama-completion"
    if [ -x "$bin" ]; then
        echo "$bin"
        return 0
    fi
    local legacy="$LLAMA_BUILD_ROOT/build-lns/bin/llama-completion"
    if [ -x "$legacy" ]; then
        echo "NOTE: build-lns-$mode not built; using legacy build-lns binary" >&2
        echo "$legacy"
        return 0
    fi
    echo "ERROR: llama-completion not found at:" >&2
    echo "  $bin" >&2
    echo "Run ./setup.sh first to build it (all modes), or" >&2
    echo "  ./setup.sh --sum-mode=$mode  for just this variant." >&2
    return 1
}

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 1
fi

MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"

run_one() {
    local mode="$1"
    local llama_bin
    llama_bin="$(bin_for_mode "$mode")" || exit 1

    echo "==> Running LNS inference (backend: LNS$BACKEND, sum mode: $mode, n=$N_TOKENS)"
    echo "    model: $MODEL"
    echo "    bin:   $llama_bin"

    GGML_LNS_BACKEND="$BACKEND" "$llama_bin" \
        -m "$MODEL" \
        -p "$PROMPT" \
        -n "$N_TOKENS" \
        -fa off \
        -no-cnv \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

if [ "$SUM_MODE" = "all" ]; then
    for mode in "${ALL_MODES[@]}"; do
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  sum mode: $mode"
        echo "════════════════════════════════════════════════════════════════"
        run_one "$mode"
    done
else
    run_one "$SUM_MODE"
fi

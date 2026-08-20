#!/usr/bin/env bash
# run.sh — re-run LNS inference after a successful ./setup.sh build.
#
# Usage:
#   ./run.sh                          # LNS16, default model/prompt
#   ./run.sh --backend=32             # LNS32; default -n 4
#   ./run.sh -m ./other.gguf -n 40
#   ./run.sh -p "Hello" -- --temp 0.2 # extra args after -- go to llama-completion
#
# Summation mode (base|monte|lpvip|sum32) is compile-time; rebuild with
# ./setup.sh --sum-mode=... to change it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LLAMA_BIN="$REPO_ROOT/build/llama.cpp/build-lns/bin/llama-completion"
DEFAULT_MODEL="$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf"

BACKEND="${GGML_LNS_BACKEND:-16}"
MODEL="${MODEL:-${MODEL_SMOLLM:-$DEFAULT_MODEL}}"
PROMPT="The capital of France is"
N_TOKENS=""
N_TOKENS_SET=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./run.sh [options] [-- llama-completion-args...]

Re-run LNS inference using the binary built by ./setup.sh.
Does not rebuild; summation mode is fixed at compile time.

Options:
  --backend=16|32     LNS backend (default: 16, or $GGML_LNS_BACKEND)
  -m, --model PATH    GGUF model path (default: $MODEL / $MODEL_SMOLLM /
                      SmolLM2-135M-Instruct-Q4_K_M.gguf in repo root)
  -p, --prompt TEXT   Prompt (default: "The capital of France is")
  -n, --tokens N      Tokens to predict (default: 20 for LNS16, 4 for LNS32)
  -h, --help          Show this help

Environment:
  GGML_LNS_BACKEND    Same as --backend (16 or 32)
  MODEL / MODEL_SMOLLM
                      Default model path
  LNS_OP_AUDIT=1      Print one JSON line per supports_op query

Examples:
  ./run.sh
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

if [ ! -x "$LLAMA_BIN" ]; then
    echo "ERROR: llama-completion not found at:" >&2
    echo "  $LLAMA_BIN" >&2
    echo "Run ./setup.sh first to build it." >&2
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 1
fi

MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"

echo "==> Running LNS inference (backend: LNS$BACKEND, n=$N_TOKENS)"
echo "    model: $MODEL"
echo "    bin:   $LLAMA_BIN"

export GGML_LNS_BACKEND="$BACKEND"
exec "$LLAMA_BIN" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n "$N_TOKENS" \
    -fa off \
    -no-cnv \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

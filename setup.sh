#!/usr/bin/env bash
# setup.sh — reproduce the LNS backend results from scratch.
#
# Usage:
#   ./setup.sh                    # end-to-end inference (requires at least one model)
#   ./setup.sh --unit-test        # also build and run test-lns-backend (standalone ggml)
#   ./setup.sh --unit-test --skip-inference
#   ./setup.sh --sum-mode=monte   # xlns16 summation variant: base|monte|lpvip
#                                 # (compile-time; also settable via LNS16_SUM env)
#
# Backends: both LNS16 (xlns16) and LNS32 (xlns32) are built. At runtime the
# GGML_LNS_BACKEND env var (16|32, default 16) selects which backend claims ops.
# The LNS32 SmolLM2 inference run can be skipped with LNS32_INFERENCE=0
# (it is slow: float<->xlns32 conversion has no lookup table).
#
# Models (at least one required for inference):
#   MODEL_SMOLLM  — SmolLM2-135M-Instruct-Q4_K_M.gguf (in repo root by default)
#   MODEL_LLAMA32   — Llama-3.2-1B-Instruct-Q4_K_M.gguf
#   MODEL           — alias for MODEL_SMOLLM (backward compatible)
#
# Upstream integration: always fetches latest ggml, llama.cpp, and xlnscpp,
# applies regex edits first for ggml/llama.cpp, falls back to pinned commits
# in patches/ if integration or build fails.

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
XLNSCPP_DIR="$REPO_ROOT/xlnscpp"
BUILD_DIR="$REPO_ROOT/build"
LLAMA_BASE_COMMIT="$(cat "$REPO_ROOT/patches/llama-base-commit.txt")"
GGML_BASE_COMMIT="$(cat "$REPO_ROOT/patches/ggml-base-commit.txt")"
XLNSCPP_BASE_COMMIT="$(cat "$REPO_ROOT/patches/xlnscpp-base-commit.txt")"

MODEL_SMOLLM="${MODEL_SMOLLM:-${MODEL:-$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf}}"
MODEL_LLAMA32="${MODEL_LLAMA32:-}"

RUN_UNIT_TEST=0
RUN_INFERENCE=1
LNS16_SUM="${LNS16_SUM:-base}"
LNS32_INFERENCE="${LNS32_INFERENCE:-1}"

for arg in "$@"; do
    case "$arg" in
        --unit-test)         RUN_UNIT_TEST=1 ;;
        --skip-inference)    RUN_INFERENCE=0 ;;
        --sum-mode=*)        LNS16_SUM="${arg#*=}" ;;
        --sum-mode)          echo "ERROR: use --sum-mode=<base|monte|lpvip> (with '=')" >&2; exit 1 ;;
        *)                   echo "WARNING: ignoring unknown argument '$arg'" >&2 ;;
    esac
done

case "$LNS16_SUM" in
    base|monte|lpvip) ;;
    *) echo "ERROR: --sum-mode must be base, monte, or lpvip (got '$LNS16_SUM')" >&2; exit 1 ;;
esac

resolve_model_path() {
    local var_name="$1"
    local path="$2"
    if [ -n "$path" ]; then
        if [ ! -f "$path" ]; then
            echo "ERROR: $var_name model not found: $path" >&2
            exit 1
        fi
        echo "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
    fi
}

MODEL_SMOLLM="$(resolve_model_path MODEL_SMOLLM "$MODEL_SMOLLM")"
MODEL_LLAMA32="$(resolve_model_path MODEL_LLAMA32 "$MODEL_LLAMA32")"

if [ "$RUN_INFERENCE" -eq 1 ] && [ -z "$MODEL_SMOLLM" ] && [ -z "$MODEL_LLAMA32" ]; then
    echo "ERROR: At least one model path required for inference." >&2
    echo "  MODEL_SMOLLM=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf ./setup.sh" >&2
    echo "  MODEL_LLAMA32=/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf ./setup.sh" >&2
    echo "" >&2
    echo "SmolLM2 is included in the repo root:" >&2
    echo "  MODEL_SMOLLM=$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf ./setup.sh" >&2
    exit 1
fi

# Detect parallel job count
if command -v nproc &>/dev/null; then
    JOBS=$(nproc)
else
    JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
fi

# shellcheck source=scripts/integrate-lns.sh
source "$REPO_ROOT/scripts/integrate-lns.sh"

check_deps() {
    for cmd in git cmake g++ make perl; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "ERROR: '$cmd' not found. Please install it and retry." >&2
            exit 1
        fi
    done
}

# NOTE: functions below are invoked from contexts where `set -e` is suppressed
# (the `if ! run_build_and_test` retry wrapper disables errexit inside the
# whole call tree — a bash rule), so failures must be propagated explicitly
# with `|| return 1`.
copy_backend() {
    local dest_src="$1"
    local dest_inc="$2"
    local dest_src32="${dest_src%/ggml-lns}/ggml-lns32"
    mkdir -p "$dest_src" "$dest_src32" || return 1
    cp "$REPO_ROOT/lns-backend/src/ggml-lns.cpp" "$dest_src/" || return 1
    cp "$REPO_ROOT/lns-backend/src/lns-ops.cpp"  "$dest_src/" || return 1
    cp "$REPO_ROOT/lns-backend/src/lns-ops.h"    "$dest_src/" || return 1
    cp "$REPO_ROOT/lns-backend/src/CMakeLists.txt" "$dest_src/" || return 1
    cp "$REPO_ROOT/lns-backend/include/ggml-lns.h" "$dest_inc/" || return 1
    cp "$REPO_ROOT/lns-backend/src32/ggml-lns32.cpp" "$dest_src32/" || return 1
    cp "$REPO_ROOT/lns-backend/src32/lns32-ops.cpp"  "$dest_src32/" || return 1
    cp "$REPO_ROOT/lns-backend/src32/lns32-ops.h"    "$dest_src32/" || return 1
    cp "$REPO_ROOT/lns-backend/src32/CMakeLists.txt" "$dest_src32/" || return 1
    cp "$REPO_ROOT/lns-backend/include/ggml-lns32.h" "$dest_inc/" || return 1
}

fetch_latest() {
    local repo_dir="$1"
    local branch="${2:-master}"
    cd "$repo_dir" || return 1
    git fetch origin || return 1
    git reset --hard "origin/$branch" || return 1
    echo "    at $(git rev-parse --short HEAD)"
}

prepare_xlnscpp() {
    if [ ! -d "$XLNSCPP_DIR/.git" ]; then
        echo "    Initialising xlnscpp submodule..."
        git -C "$REPO_ROOT" submodule update --init
    fi
    echo "    Fetching latest xlnscpp"
    fetch_latest "$XLNSCPP_DIR" main
}

fallback_xlnscpp() {
    echo "    xlnscpp latest failed; using pinned commit"
    cd "$XLNSCPP_DIR"
    git reset --hard "$XLNSCPP_BASE_COMMIT"
    echo "    at $(git rev-parse --short HEAD)"
}

prepare_ggml_test_files() {
    local ggml_dir="$1"
    cp "$REPO_ROOT/lns-backend/test-lns-backend.cpp" "$ggml_dir/examples/simple/" || return 1
    if ! grep -q 'test-lns-backend' "$ggml_dir/examples/simple/CMakeLists.txt"; then
        cat >> "$ggml_dir/examples/simple/CMakeLists.txt" << 'EOF'

if (GGML_LNS)
    add_executable(test-lns-backend test-lns-backend.cpp)
    target_link_libraries(test-lns-backend PRIVATE ggml)
endif()
EOF
    fi
}

prepare_ggml() {
    local ggml_dir="$1"
    if [ ! -d "$ggml_dir/.git" ]; then
        echo "    Cloning ggml..."
        git clone https://github.com/ggml-org/ggml.git "$ggml_dir" || return 1
    fi
    echo "    Fetching latest ggml"
    fetch_latest "$ggml_dir" || return 1
    integrate_lns_or_fallback "$ggml_dir" \
        "$REPO_ROOT/patches/ggml-core.patch" "$GGML_BASE_COMMIT" || return 1
    copy_backend "$ggml_dir/src/ggml-lns" "$ggml_dir/include" || return 1
}

prepare_llama() {
    local llama_dir="$1"
    if [ ! -d "$llama_dir/.git" ]; then
        echo "    Cloning llama.cpp (this may take a few minutes)..."
        git clone https://github.com/ggml-org/llama.cpp.git "$llama_dir" || return 1
    fi
    echo "    Fetching latest llama.cpp"
    fetch_latest "$llama_dir" || return 1
    integrate_lns_or_fallback "$llama_dir" \
        "$REPO_ROOT/patches/llama-core.patch" "$LLAMA_BASE_COMMIT" || return 1
    copy_backend "$llama_dir/ggml/src/ggml-lns" "$llama_dir/ggml/include" || return 1
}

run_inference() {
    local label="$1"
    local model_path="$2"
    local llama_bin="$3"
    local backend="${4:-16}"   # GGML_LNS_BACKEND: 16 or 32
    local n_tokens="${5:-20}"

    echo ""
    echo "==> Running LNS inference: $label (backend: LNS$backend, sum mode: $LNS16_SUM)"
    GGML_LNS_BACKEND="$backend" "$llama_bin" \
        -m "$model_path" \
        -p "The capital of France is" \
        -n "$n_tokens" \
        -fa 0 \
        -no-cnv || return 1
}

run_build_and_test() {
    # ── Unit test (standalone ggml) ───────────────────────────────────────────

    if [ "$RUN_UNIT_TEST" -eq 1 ]; then
        echo ""
        echo "==> Setting up standalone ggml for unit test"

        GGML_DIR="$BUILD_DIR/ggml"
        prepare_ggml "$GGML_DIR" || return 1
        prepare_ggml_test_files "$GGML_DIR" || return 1

        echo "    Building (sum mode: $LNS16_SUM)"
        mkdir -p "$GGML_DIR/build-lns" || return 1
        cd "$GGML_DIR/build-lns" || return 1
        cmake .. \
            -DGGML_LNS=ON \
            -DGGML_LNS32=ON \
            -DGGML_LNS16_SUM="$LNS16_SUM" \
            -DGGML_LNS16_MONTE_BYLPVIP=OFF \
            -DGGML_BLAS=OFF \
            -DGGML_CPU_REPACK=OFF \
            -DGGML_METAL=OFF \
            -DCMAKE_BUILD_TYPE=Release \
            -DXLNSCPP_DIR="$XLNSCPP_DIR" || return 1
        cmake --build . --target test-lns-backend -j"$JOBS" || return 1

        echo ""
        echo "==> Running test-lns-backend (LNS16, tolerance profile: $LNS16_SUM)"
        GGML_LNS_BACKEND=16 ./bin/test-lns-backend LNS16 "$LNS16_SUM" || return 1

        echo ""
        echo "==> Running test-lns-backend (LNS32)"
        GGML_LNS_BACKEND=32 ./bin/test-lns-backend LNS32 || return 1
    fi

    # ── End-to-end inference (llama.cpp) ─────────────────────────────────────

    if [ "$RUN_INFERENCE" -eq 1 ]; then
        echo ""
        echo "==> Setting up llama.cpp for end-to-end LNS inference"

        LLAMA_DIR="$BUILD_DIR/llama.cpp"
        prepare_llama "$LLAMA_DIR" || return 1

        echo "    Building llama-completion (sum mode: $LNS16_SUM)"
        mkdir -p "$LLAMA_DIR/build-lns" || return 1
        cd "$LLAMA_DIR/build-lns" || return 1
        cmake .. \
            -DGGML_LNS=ON \
            -DGGML_LNS32=ON \
            -DGGML_LNS16_SUM="$LNS16_SUM" \
            -DGGML_LNS16_MONTE_BYLPVIP=OFF \
            -DGGML_BLAS=OFF \
            -DGGML_METAL=OFF \
            -DGGML_CPU_REPACK=OFF \
            -DCMAKE_BUILD_TYPE=Release \
            -DXLNSCPP_DIR="$XLNSCPP_DIR" || return 1
        cmake --build . --target llama-completion -j"$JOBS" || return 1

        LLAMA_BIN="$LLAMA_DIR/build-lns/bin/llama-completion"

        if [ -n "$MODEL_SMOLLM" ]; then
            run_inference "SmolLM2-135M-Instruct" "$MODEL_SMOLLM" "$LLAMA_BIN" 16 20 || return 1
            if [ "$LNS32_INFERENCE" -eq 1 ]; then
                # Short run: the LNS32 backend converts every weight element
                # through libm log() per matmul (no lookup table exists).
                run_inference "SmolLM2-135M-Instruct" "$MODEL_SMOLLM" "$LLAMA_BIN" 32 4 || return 1
            fi
        fi
        if [ -n "$MODEL_LLAMA32" ]; then
            run_inference "Llama-3.2-1B-Instruct" "$MODEL_LLAMA32" "$LLAMA_BIN" 16 20 || return 1
        fi
    fi
}

# ── Prerequisites ─────────────────────────────────────────────────────────────

check_deps

echo "==> Preparing xlnscpp"
prepare_xlnscpp

mkdir -p "$BUILD_DIR"

if ! run_build_and_test; then
    fallback_xlnscpp
    run_build_and_test
fi

echo ""
echo "==> Done."

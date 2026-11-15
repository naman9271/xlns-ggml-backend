#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../lib.sh"
lock_experiment op-sensitivity

setup_build
if [ "${SKIP_SETUP:-0}" != 1 ]; then
    echo "==> build-lns-base: adding the base/lpvip/sum32 reductions for LNS16_SUM_POLICY (GGML_LNSCUDA16_MIXED=ON)"
    lib_run bash -c 'cd "$1" && cmake .. -DGGML_LNSCUDA16_MIXED=ON > /dev/null && cmake --build . --target llama-completion -j "$2"' \
        _ "$REPO_ROOT/build/llama.cpp/build-lns-base" "$JOBS"
fi
ensure_models

lib_run python3 "$HERE/run.py" -m "$MODEL_SMOLLM" -m "$MODEL_QWEN" "$@"

for d in "$HERE"/results/*/; do
    if [ -f "$d/runs.jsonl" ]; then
        write_report "$HERE/report.py" "$d" "$d/REPORT.md"
    fi
done
finish_work "$REPO_ROOT/build/exp-op"

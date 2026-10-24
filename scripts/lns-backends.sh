#!/usr/bin/env bash
# lns-backends.sh — the list of LNS backends and how each plugs into ggml.
#
# Sourced by scripts/integrate-lns.sh (and through it by setup.sh and
# scripts/validate-latest.sh). Everything that hooks, copies or validates a
# backend iterates over this list, so adding a backend is one row here plus
# its sources; no script logic changes.
#
# Row format (fields separated by '|'):
#   NAME        ggml_add_backend() name. ggml derives the rest from it: CMake
#               option GGML_<NAME>, source dir src/ggml-<name in lowercase>,
#               compile definition GGML_USE_<NAME>.
#   SOURCES     repo-relative directory copied to src/ggml-<name>/
#   HEADER      public header in lns-backend/include/, copied to include/
#   REG_FNS     registration function(s), space-separated, called in order
#   DESCRIPTION help text of the CMake option
#
# Row order is registration order, which is scheduler priority among the LNS
# devices: GPU backends come first so they win the ops they support, and the
# CPU backend selected alongside them (see GGML_LNS_BACKEND in
# lns-backend/common/lns-backend-config.h) picks up the rest.

LNS_BACKENDS=(
    "LNSCUDA|lns-backend/cuda|ggml-lns-cuda.h|ggml_backend_lns_cuda16_reg ggml_backend_lns_cuda32_reg|ggml: use LNS on CUDA GPUs (xlnscuda)"
    "LNS|lns-backend/src|ggml-lns.h|ggml_backend_lns_reg|ggml: use LNS (Logarithmic Number System)"
    "LNS32|lns-backend/src32|ggml-lns32.h|ggml_backend_lns32_reg|ggml: use LNS32 (32-bit Logarithmic Number System)"
)

# Shared headers and CMake helpers used by every backend, copied to
# src/ggml-lns-common/.
LNS_COMMON_SOURCES="lns-backend/common"

# lns_backend_row <row> — splits one LNS_BACKENDS row into the variables
# LNS_NAME, LNS_SOURCES, LNS_HEADER, LNS_REG_FNS, LNS_DESCRIPTION.
lns_backend_row() {
    IFS='|' read -r LNS_NAME LNS_SOURCES LNS_HEADER LNS_REG_FNS LNS_DESCRIPTION <<< "$1"
}

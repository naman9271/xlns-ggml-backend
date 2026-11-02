#pragma once

// The ops the LNSCUDA backends run on the GPU: one table row per op, pairing
// the support predicate with the handler that marshals the ggml node into a
// kernel_set call. Both LNSCUDA devices share this table; only their
// kernel_set (xlns16 or xlns32) differs.
//
// Ops missing here (MUL_MAT_ID, GET_ROWS, ARGSORT, CLAMP, DIAG_MASK_INF, LNS16
// copies) are declined, so with GGML_LNS_BACKEND=cuda16 / cuda32 they fall to
// the CPU LNS backend selected alongside.

#include "lns-backend-framework.h"
#include "lns-cuda-context.h"

namespace lns_cuda {

lns::op_table<context> op_table();

} // namespace lns_cuda

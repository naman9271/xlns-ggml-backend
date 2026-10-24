// LNS32 backend: every supported op in xlns32 arithmetic on the CPU.
//
// Same op set as LNS16, but there is no GGML_TYPE_LNS32: all tensors flow as
// F32 (or a weight type with to_float) and are converted per op. The
// backend/device/registry plumbing is shared (lns-backend-framework.h); this
// file only says what LNS32 is and which kernel (lns32-ops.cpp) computes
// which op.

#include "ggml-lns32.h"

#include "lns-backend-framework.h"
#include "lns32-ops.h"

#include <cstdio>

namespace {

using lns::any_subop;
using lns::no_context;
using lns::op_kernel;
using lns::stateless;

// No CPY / CONT / DUP: without an LNS32 tensor type every copy is a plain
// F32 / F16 copy, which the CPU backend handles.
const op_kernel<no_context> k_kernels[] = {
    { GGML_OP_MUL_MAT,       any_subop,          lns::check_mul_mat,          stateless<lns32_mul_mat>       },
    { GGML_OP_MUL_MAT_ID,    any_subop,          lns::check_mul_mat_id,       stateless<lns32_mul_mat_id>    },
    { GGML_OP_ADD,           any_subop,          lns::check_binary_f32,       stateless<lns32_add>           },
    { GGML_OP_MUL,           any_subop,          lns::check_binary_f32,       stateless<lns32_mul>           },
    { GGML_OP_DIV,           any_subop,          lns::check_binary_f32,       stateless<lns32_div>           },
    { GGML_OP_SCALE,         any_subop,          lns::check_contiguous_f32,   stateless<lns32_scale>         },
    { GGML_OP_CLAMP,         any_subop,          lns::check_contiguous_f32,   stateless<lns32_clamp>         },
    { GGML_OP_SOFT_MAX,      any_subop,          lns::check_soft_max,         stateless<lns32_soft_max>      },
    { GGML_OP_RMS_NORM,      any_subop,          lns::check_unary_like_f32,   stateless<lns32_rms_norm>      },
    { GGML_OP_DIAG_MASK_INF, any_subop,          lns::check_contiguous_f32,   stateless<lns32_diag_mask_inf> },
    { GGML_OP_UNARY,         GGML_UNARY_OP_SILU, lns::check_unary_activation, stateless<lns32_silu>          },
    { GGML_OP_UNARY,         GGML_UNARY_OP_GELU, lns::check_unary_activation, stateless<lns32_gelu>          },
    { GGML_OP_UNARY,         GGML_UNARY_OP_RELU, lns::check_unary_activation, stateless<lns32_relu>          },
    { GGML_OP_UNARY,         GGML_UNARY_OP_TANH, lns::check_unary_activation, stateless<lns32_tanh>          },
    { GGML_OP_GLU,           GGML_GLU_OP_SWIGLU, lns::check_glu,              stateless<lns32_swiglu>        },
    { GGML_OP_GLU,           GGML_GLU_OP_GEGLU,  lns::check_glu,              stateless<lns32_geglu>         },
    { GGML_OP_GET_ROWS,      any_subop,          lns::check_get_rows_f32,     stateless<lns32_get_rows>      },
    { GGML_OP_ROPE,          any_subop,          lns::check_rope,             stateless<lns32_rope>          },
    { GGML_OP_ARGSORT,       any_subop,          lns::check_argsort,          stateless<lns32_argsort>       },
};

struct lns32_traits : lns::backend_traits_base {
    static constexpr const char * name = "LNS32";

    static const char * description() {
        return "Logarithmic Number System (xlns32)";
    }

    static ggml_guid_t guid() {
        // ASCII "LNS32-bkend-ggml" (distinct from the LNS16 backend's GUID)
        static ggml_guid guid = {
            0x4c, 0x4e, 0x53, 0x33, 0x32, 0x2d, 0x62, 0x6b,
            0x65, 0x6e, 0x64, 0x2d, 0x67, 0x67, 0x6d, 0x6c
        };
        return &guid;
    }

    static lns::op_table<context> kernels() {
        return lns::make_op_table(k_kernels);
    }

    static void log_init() {
        std::fprintf(stderr, "ggml-lns32: LNS32 backend initialized\n");
    }
};

using lns32_backend = lns::host_backend<lns32_traits>;

} // namespace

ggml_backend_t ggml_backend_lns32_init(void) {
    return lns32_backend::init();
}

bool ggml_backend_is_lns32(ggml_backend_t backend) {
    return lns32_backend::is(backend);
}

ggml_backend_reg_t ggml_backend_lns32_reg(void) {
    return lns32_backend::reg();
}

GGML_BACKEND_DL_IMPL(ggml_backend_lns32_reg)

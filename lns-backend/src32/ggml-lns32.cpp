#include "ggml-impl.h"
#include "ggml-lns32.h"
#include "ggml-backend-impl.h"

#include "lns32-ops.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <string>

// ============================================================
// Backend context
// ============================================================

struct ggml_backend_lns32_context {
    // placeholder for future state (threading, profiling, etc.)
};

// ============================================================
// Backend interface
// ============================================================

static const char * ggml_backend_lns32_get_name(ggml_backend_t backend) {
    return "LNS32";

    GGML_UNUSED(backend);
}

static void ggml_backend_lns32_free(ggml_backend_t backend) {
    ggml_backend_lns32_context * ctx = (ggml_backend_lns32_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_lns32_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                lns32_mul_mat(node);
                break;

            case GGML_OP_MUL_MAT_ID:
                lns32_mul_mat_id(node);
                break;

            case GGML_OP_ADD:
                lns32_add(node);
                break;

            case GGML_OP_MUL:
                lns32_mul(node);
                break;

            case GGML_OP_DIV:
                lns32_div(node);
                break;

            case GGML_OP_SCALE:
                lns32_scale(node);
                break;

            case GGML_OP_CLAMP:
                lns32_clamp(node);
                break;

            case GGML_OP_SOFT_MAX:
                lns32_soft_max(node);
                break;

            case GGML_OP_RMS_NORM:
                lns32_rms_norm(node);
                break;

            case GGML_OP_DIAG_MASK_INF:
                lns32_diag_mask_inf(node);
                break;

            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(node)) {
                    case GGML_UNARY_OP_SILU: lns32_silu(node); break;
                    case GGML_UNARY_OP_GELU: lns32_gelu(node); break;
                    case GGML_UNARY_OP_RELU: lns32_relu(node); break;
                    case GGML_UNARY_OP_TANH: lns32_tanh(node); break;
                    default:
                        GGML_ABORT("%s: unsupported unary op\n", __func__);
                }
                break;

            case GGML_OP_GLU:
                switch (ggml_get_glu_op(node)) {
                    case GGML_GLU_OP_SWIGLU: lns32_swiglu(node); break;
                    case GGML_GLU_OP_GEGLU:  lns32_geglu(node);  break;
                    default:
                        GGML_ABORT("%s: unsupported GLU op\n", __func__);
                }
                break;

            case GGML_OP_GET_ROWS:
                lns32_get_rows(node);
                break;

            case GGML_OP_ROPE:
                lns32_rope(node);
                break;

            case GGML_OP_ARGSORT:
                lns32_argsort(node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }
    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i lns32_backend_i = {
    /* .get_name                = */ ggml_backend_lns32_get_name,
    /* .free                    = */ ggml_backend_lns32_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_lns32_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

// ============================================================
// GUID
// ============================================================

static ggml_guid_t ggml_backend_lns32_guid(void) {
    // ASCII "LNS32-bkend-ggml" (distinct from the LNS16 backend's GUID)
    static ggml_guid guid = {
        0x4c, 0x4e, 0x53, 0x33, 0x32, 0x2d, 0x62, 0x6b,
        0x65, 0x6e, 0x64, 0x2d, 0x67, 0x67, 0x6d, 0x6c
    };
    return &guid;
}

// ============================================================
// Public API
// ============================================================

ggml_backend_t ggml_backend_lns32_init(void) {
    static bool logged = false;
    if (!logged) {
        logged = true;
        std::fprintf(stderr, "ggml-lns32: LNS32 backend initialized\n");
    }

    ggml_backend_lns32_context * ctx = new ggml_backend_lns32_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_lns32_guid(),
        /* .iface   = */ lns32_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_lns32_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_lns32(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_lns32_guid());
}

// ============================================================
// Device interface
// ============================================================

static const char * ggml_backend_lns32_device_get_name(ggml_backend_dev_t dev) {
    return "LNS32";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_lns32_device_get_description(ggml_backend_dev_t dev) {
    return "Logarithmic Number System (xlns32)";

    GGML_UNUSED(dev);
}

static void ggml_backend_lns32_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_lns32_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_lns32_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_lns32_device_get_name(dev);
    props->description = ggml_backend_lns32_device_get_description(dev);
    props->type        = ggml_backend_lns32_device_get_type(dev);
    ggml_backend_lns32_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_lns32_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_lns32_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_lns32_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_lns32_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

struct ggml_backend_lns32_op_support {
    bool supported;
    const char * reason;
};

static const char * ggml_backend_lns32_tensor_type_name(const struct ggml_tensor * tensor) {
    return tensor ? ggml_type_name(tensor->type) : "null";
}

static bool ggml_backend_lns32_audit_enabled(void) {
    static const bool enabled = []() {
        const char * env = std::getenv("LNS_OP_AUDIT");
        return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
    }();
    return enabled;
}

// Arbitration between the LNS16 and LNS32 backends. Both are ACCEL backends
// sharing the CPU buffer type, so the scheduler would hand every mutually
// supported op to whichever registered first — the env var decides which one
// claims ops for this run. Default: 16.
static bool lns_backend_enabled_by_env(bool is_lns32) {
    static const int sel = []() {
        const char * env = std::getenv("GGML_LNS_BACKEND");
        return (env != nullptr && strcmp(env, "32") == 0) ? 32 : 16;
    }();
    return is_lns32 == (sel == 32);
}

static ggml_backend_lns32_op_support ggml_backend_lns32_check_op(const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return { true, "supported_view_or_noop" };

        case GGML_OP_MUL_MAT:
            return {
                op->src[1]->type == GGML_TYPE_F32 &&
                    (src0->type == GGML_TYPE_F32 ||
                     ggml_get_type_traits(src0->type)->to_float != NULL),
                "mul_mat_requires_src1_f32_and_convertible_src0"
            };

        case GGML_OP_MUL_MAT_ID:
            return {
                op->src[1]->type == GGML_TYPE_F32 &&
                    op->src[2] != NULL && op->src[2]->type == GGML_TYPE_I32 &&
                    (src0->type == GGML_TYPE_F32 ||
                     ggml_get_type_traits(src0->type)->to_float != NULL),
                "mul_mat_id_requires_src1_f32_and_i32_ids_and_convertible_src0"
            };

        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return {
                src0->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32,
                "binary_requires_f32_srcs"
            };

        case GGML_OP_SCALE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
            return { src0->type == GGML_TYPE_F32, "unary_like_requires_src0_f32" };

        case GGML_OP_CLAMP:
            return { src0->type == GGML_TYPE_F32, "clamp_requires_src0_f32" };

        case GGML_OP_ARGSORT:
            return { src0->type == GGML_TYPE_F32, "argsort_requires_src0_f32" };

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_TANH:
                    return { src0->type == GGML_TYPE_F32, "unary_activation_requires_src0_f32" };
                default:
                    return { false, "unsupported_unary_op" };
            }

        case GGML_OP_GLU: {
            const struct ggml_tensor * src1 = op->src[1];
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU:
                    break;
                default:
                    return { false, "unsupported_glu_op" };
            }
            if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                return { false, "glu_requires_f32_src0_and_dst" };
            }
            if (!ggml_is_contiguous_1(src0) || !ggml_is_contiguous_1(op)) {
                return { false, "glu_requires_contiguous_rows" };
            }
            if (src1) {
                return {
                    src1->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous_1(src1) &&
                        src1->ne[0] == src0->ne[0] &&
                        ggml_nrows(src1) == ggml_nrows(src0) &&
                        op->ne[0] == src0->ne[0] &&
                        ggml_nrows(op) == ggml_nrows(src0),
                    "glu_split_requires_matching_f32_contiguous_inputs"
                };
            }
            return {
                src0->ne[0] % 2 == 0 &&
                    op->ne[0] == src0->ne[0] / 2 &&
                    ggml_nrows(op) == ggml_nrows(src0),
                "glu_fused_requires_even_src0_and_half_width_dst"
            };
        }

        case GGML_OP_GET_ROWS:
            // lns32_get_rows writes F32 and dequantizes src0 via to_float —
            // decline anything else (e.g. I32->I32 state-copy gathers in
            // recurrent models) so it falls back to CPU instead of hitting
            // the kernel's asserts.
            return {
                op->type == GGML_TYPE_F32 &&
                    (src0->type == GGML_TYPE_F32 ||
                     ggml_get_type_traits(src0->type)->to_float != NULL),
                "get_rows_requires_f32_dst_and_convertible_src0"
            };

        // CPY/CONT/DUP: never claimed. There is no GGML_TYPE_LNS32, so every
        // copy in the graph is a plain F32/F16 copy that the CPU backend
        // handles correctly (the LNS16 backend only claims LNS16-typed copies
        // for the same reason).
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return { false, "copy_left_to_cpu_backend" };

        default:
            return { false, "unsupported_op" };
    }
}

static void ggml_backend_lns32_audit_op(const struct ggml_tensor * op, ggml_backend_lns32_op_support support) {
    if (!ggml_backend_lns32_audit_enabled()) {
        return;
    }

    std::fprintf(stderr,
        "{\"event\":\"lns_supports_op\","
        "\"backend\":\"LNS32\","
        "\"op\":\"%s\","
        "\"desc\":\"%s\","
        "\"supported\":%s,"
        "\"dst_type\":\"%s\","
        "\"src_types\":[\"%s\",\"%s\",\"%s\",\"%s\"],"
        "\"shape\":[%lld,%lld,%lld,%lld],"
        "\"reason\":\"%s\"}\n",
        ggml_op_name(op->op),
        ggml_op_desc(op),
        support.supported ? "true" : "false",
        ggml_type_name(op->type),
        ggml_backend_lns32_tensor_type_name(op->src[0]),
        ggml_backend_lns32_tensor_type_name(op->src[1]),
        ggml_backend_lns32_tensor_type_name(op->src[2]),
        ggml_backend_lns32_tensor_type_name(op->src[3]),
        (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3],
        support.reason);
}

static bool ggml_backend_lns32_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);

    if (!lns_backend_enabled_by_env(/*is_lns32=*/true)) {
        ggml_backend_lns32_audit_op(op, { false, "disabled_by_env" });
        return false;
    }

    const ggml_backend_lns32_op_support support = ggml_backend_lns32_check_op(op);
    ggml_backend_lns32_audit_op(op, support);
    if (!support.supported) {
        return false;
    }

    // LNS_DISABLE_OPS="ADD,ROPE,..." — force the named ops to fall back to CPU.
    // Bisection aid, same semantics as the LNS16 backend: tokens match either
    // the ggml op name (case-insensitive) or a substring of the tensor name.
    static const std::string disabled = []() {
        const char * env = std::getenv("LNS_DISABLE_OPS");
        return env ? std::string(env) : std::string();
    }();
    if (!disabled.empty()) {
        const char * name = ggml_op_name(op->op);
        size_t pos = 0;
        while (pos <= disabled.size()) {
            const size_t comma = disabled.find(',', pos);
            const std::string tok = disabled.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (strcasecmp(tok.c_str(), name) == 0 || (op->name[0] != '\0' && strstr(op->name, tok.c_str()) != NULL)) {
                return false;
            }
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
    }
    return true;
}

static bool ggml_backend_lns32_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static bool ggml_backend_lns32_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // Offload all ops we support — this tells the scheduler to prefer LNS32
    // over CPU for these ops even though both use host buffers
    return ggml_backend_lns32_device_supports_op(dev, op);
}

static const struct ggml_backend_device_i ggml_backend_lns32_device_i = {
    /* .get_name             = */ ggml_backend_lns32_device_get_name,
    /* .get_description      = */ ggml_backend_lns32_device_get_description,
    /* .get_memory           = */ ggml_backend_lns32_device_get_memory,
    /* .get_type             = */ ggml_backend_lns32_device_get_type,
    /* .get_props            = */ ggml_backend_lns32_device_get_props,
    /* .init_backend         = */ ggml_backend_lns32_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_lns32_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_lns32_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_lns32_device_supports_op,
    /* .supports_buft        = */ ggml_backend_lns32_device_supports_buft,
    /* .offload_op           = */ ggml_backend_lns32_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ============================================================
// Registry interface
// ============================================================

static const char * ggml_backend_lns32_reg_get_name(ggml_backend_reg_t reg) {
    return "LNS32";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_lns32_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_lns32_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_lns32_device = {
        /* .iface   = */ ggml_backend_lns32_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_lns32_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_lns32_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_lns32_reg_i = {
    /* .get_name         = */ ggml_backend_lns32_reg_get_name,
    /* .get_device_count = */ ggml_backend_lns32_reg_get_device_count,
    /* .get_device       = */ ggml_backend_lns32_reg_get_device,
    /* .get_proc_address = */ ggml_backend_lns32_get_proc_address,
};

ggml_backend_reg_t ggml_backend_lns32_reg(void) {
    static struct ggml_backend_reg ggml_backend_lns32_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_lns32_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_lns32_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_lns32_reg)

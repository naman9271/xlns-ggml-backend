#pragma once

// Op-support predicates shared by the LNS backends.
//
// Each predicate answers "can an LNS kernel compute this node exactly as
// ggml defines it?" and says why not when it cannot; the reason string shows
// up in the LNS_OP_AUDIT log. Backends list predicates next to their kernels
// in an op table (lns-backend-framework.h) and may wrap them to add their own
// constraints, so the CPU and GPU backends cannot drift apart on what an op
// means.

#include "ggml.h"

#include <cstring>

namespace lns {

struct op_support {
    bool         supported;
    const char * reason;
};

// F32 itself, or any type ggml can dequantize to F32 (F16, Q4_K, ...).
inline bool convertible_to_f32(ggml_type type) {
    return type == GGML_TYPE_F32 || ggml_get_type_traits(type)->to_float != nullptr;
}

// The kernels read each operand row as one contiguous run of ne[0] elements.
inline bool rows_contiguous(const ggml_tensor * t) {
    return t->nb[0] == ggml_type_size(t->type);
}

inline op_support check_mul_mat(const ggml_tensor * op) {
    if (op->src[1]->type != GGML_TYPE_F32 || !convertible_to_f32(op->src[0]->type)) {
        return { false, "mul_mat_requires_src1_f32_and_convertible_src0" };
    }
    return {
        rows_contiguous(op->src[0]) && rows_contiguous(op->src[1]),
        "mul_mat_requires_contiguous_rows"
    };
}

inline op_support check_mul_mat_id(const ggml_tensor * op) {
    return {
        op->src[1]->type == GGML_TYPE_F32 &&
            op->src[2] != nullptr && op->src[2]->type == GGML_TYPE_I32 &&
            convertible_to_f32(op->src[0]->type),
        "mul_mat_id_requires_src1_f32_and_i32_ids_and_convertible_src0"
    };
}

// ADD / MUL / DIV on F32 operands.
inline op_support check_binary_f32(const ggml_tensor * op) {
    return {
        op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32,
        "binary_requires_f32_srcs"
    };
}

// ADD / MUL / DIV where the second operand may already be LNS16.
inline op_support check_binary_f32_or_lns16(const ggml_tensor * op) {
    return {
        op->src[0]->type == GGML_TYPE_F32 &&
            (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_LNS16),
        "binary_requires_src0_f32_and_src1_f32_or_lns16"
    };
}

// RMS_NORM reads a single F32 input, row by row through its strides.
inline op_support check_unary_like_f32(const ggml_tensor * op) {
    return { op->src[0]->type == GGML_TYPE_F32, "unary_like_requires_src0_f32" };
}

inline op_support check_argsort(const ggml_tensor * op) {
    return { op->src[0]->type == GGML_TYPE_F32, "argsort_requires_src0_f32" };
}

// SCALE, CLAMP and DIAG_MASK_INF walk src0 and dst as flat arrays.
inline op_support check_contiguous_f32(const ggml_tensor * op) {
    return {
        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op),
        "requires_contiguous_f32_src0_and_dst"
    };
}

// SILU / GELU / RELU / TANH, which also walk src0 and dst as flat arrays.
inline op_support check_unary_activation(const ggml_tensor * op) {
    return {
        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op),
        "unary_activation_requires_contiguous_f32_src0_and_dst"
    };
}

// The LNS RoPE kernels rotate pairs (normal or NEOX layout) by
// theta = pos * freq_base^(-2i/n_dims) / freq_factor[i]: ggml's RoPE with
// optional frequency factors (e.g. Llama 3 scaling). YaRN (ext_factor),
// linear scaling (freq_scale), attention scaling and the multi-section /
// vision modes change the result and are not implemented, so such nodes
// stay on the CPU backend.
inline op_support check_rope(const ggml_tensor * op) {
    if (op->src[0]->type != GGML_TYPE_F32) {
        return { false, "rope_requires_src0_f32" };
    }
    const int mode = ((const int32_t *) op->op_params)[2];
    if (mode != 0 && mode != GGML_ROPE_TYPE_NEOX) {
        return { false, "rope_mode_unsupported" };
    }
    float freq_scale, ext_factor, attn_factor;
    memcpy(&freq_scale,  (const int32_t *) op->op_params + 6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params + 8, sizeof(float));
    if (freq_scale != 1.0f || ext_factor != 0.0f || attn_factor != 1.0f) {
        return { false, "rope_scaling_unsupported" };
    }
    const ggml_tensor * ff = op->src[2];
    const int n_dims = ((const int32_t *) op->op_params)[1];
    if (ff != nullptr && (ff->type != GGML_TYPE_F32 || !ggml_is_contiguous(ff) || ff->ne[0] < n_dims / 2)) {
        return { false, "rope_freq_factors_require_contiguous_f32" };
    }
    return { true, "rope_normal_or_neox" };
}

// The LNS softmax kernels implement softmax(x * scale + mask) with an F32
// mask. Attention sinks (src[2]) and ALiBi (max_bias != 0) change the result
// and are not implemented, so such nodes must stay on the CPU backend.
inline op_support check_soft_max(const ggml_tensor * op) {
    const ggml_tensor * mask = op->src[1];
    float max_bias;
    memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
    if (op->src[0]->type != GGML_TYPE_F32 || (mask != nullptr && mask->type != GGML_TYPE_F32)) {
        return { false, "soft_max_requires_f32_src0_and_f32_mask" };
    }
    if (op->src[2] != nullptr) {
        return { false, "soft_max_sinks_unsupported" };
    }
    if (max_bias != 0.0f) {
        return { false, "soft_max_alibi_unsupported" };
    }
    return { true, "soft_max_f32" };
}

// SWIGLU / GEGLU, in both the split (src1 present) and fused layouts.
inline op_support check_glu(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
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

// The gather kernels dequantize src0 rows through to_float and write F32
// rows. Anything else (e.g. the I32 -> I32 state-copy gathers of recurrent
// models) must fall back to the CPU backend instead of tripping an assert.
inline op_support check_get_rows_f32(const ggml_tensor * op) {
    return {
        op->type == GGML_TYPE_F32 && convertible_to_f32(op->src[0]->type),
        "get_rows_requires_f32_dst_and_convertible_src0"
    };
}

inline op_support check_get_rows_f32_or_lns16(const ggml_tensor * op) {
    return {
        (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_LNS16) && convertible_to_f32(op->src[0]->type),
        "get_rows_requires_f32_or_lns16_dst_and_convertible_src0"
    };
}

// CPY / CONT / DUP: claimed only when an LNS16 tensor is involved. Plain
// F32 / F16 copies involve no LNS arithmetic and the CPU backend handles them
// correctly (and more safely).
inline op_support check_lns16_copy(const ggml_tensor * op) {
    const ggml_type s = op->src[0]->type;
    const ggml_type d = op->type;
    return {
        (s == GGML_TYPE_LNS16 && d == GGML_TYPE_F32) ||
            (s == GGML_TYPE_F32 && d == GGML_TYPE_LNS16) ||
            (s == GGML_TYPE_LNS16 && d == GGML_TYPE_LNS16),
        "copy_requires_lns16_src_or_dst"
    };
}

} // namespace lns

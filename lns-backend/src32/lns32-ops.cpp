#include "lns32-ops.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================
// xlnscpp inclusion
// ============================================================
// System headers are hoisted to global scope here; the nested includes inside
// the anonymous namespace below are then no-ops (include guards).
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

// The xlnscpp .cpp files carry non-inline definitions and mutable global
// tables, so they may appear in exactly one translation unit per link unit.
// The anonymous namespace gives this TU a private (internal-linkage) copy of
// xlns32.cpp configured with xlns32_alt, isolated from the ggml-lns (16-bit)
// backend's copy, which the lpvip bootstrap configures differently
// (xlns32_ideal + xlns32_aicasb). Same symbol names, different definitions —
// without this namespace the link would silently merge them (ODR violation).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
namespace {

// xlnscpp declares xlns32_float overloads of these libm names inside this
// namespace, which would otherwise hide the global libm functions from the
// library's own internal calls (unqualified lookup stops at the nearest scope
// that has the name). Import the global overload sets so both participate in
// overload resolution.
using ::sin; using ::cos; using ::exp; using ::log; using ::atan;
using ::sqrt; using ::pow; using ::fabs; using ::abs;

// xlns32 defines — must be set before including xlns32.cpp
// (also set via CMake target_compile_definitions; guarded here for clarity).
// xlns32_alt: reduced-branching sb/db addition path.
// xlns32_ideal is NOT defined (matches a realistic hardware target).
// NOTE (thread safety): the non-ideal xlns32_sb is a macro that writes the
// globals xlns32_z / xlns32_zh. Safe while graph_compute is single-threaded;
// those globals must become thread_local if the backend is ever parallelized.
#ifndef xlns32_alt
#define xlns32_alt
#endif

#include "xlns32.cpp"

} // anonymous namespace
#pragma GCC diagnostic pop

// ============================================================
// Helper: convert a row from any ggml type to xlns32
// Handles F32 and any quantized type with to_float.
// NOTE: fp2xlns32 has no table fast path (unlike fp2xlns16 with
// xlns16_table) — each element costs a libm log(). This is the
// dominant cost of the LNS32 backend.
// ============================================================

static void row_to_xlns32(const void * src, enum ggml_type type, xlns32 * dst, float * f32_scratch, int64_t n) {
    if (type == GGML_TYPE_F32) {
        xlns32_batch_from_float((const float *)src, dst, (size_t)n);
    } else {
        const auto * traits = ggml_get_type_traits(type);
        GGML_ASSERT(traits->to_float != NULL);
        traits->to_float(src, f32_scratch, n);
        xlns32_batch_from_float(f32_scratch, dst, (size_t)n);
    }
}

// ============================================================
// Element-wise typed read/write helpers
// These use array indexing within a row — strides across rows
// are handled by the caller via the nb[] byte offsets.
// ============================================================

static inline xlns32 read_elem_xlns32(const void * row, int64_t i, enum ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_F32);
    return fp2xlns32(((const float *)row)[i]);
}

static inline void write_elem_xlns32(void * row, int64_t i, enum ggml_type type, xlns32 val) {
    GGML_ASSERT(type == GGML_TYPE_F32);
    ((float *)row)[i] = xlns322fp(val);
}

// ============================================================
// MUL_MAT: dst = src0^T * src1
// src0: weights (any type with to_float, or F32)
// src1: activations (F32)
// dst:  F32
// ============================================================

void lns32_mul_mat(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type0 = src0->type;
    const int64_t K = ne00;
    GGML_ASSERT(ne10 == K);

    std::vector<float>  f32_scratch(K);
    std::vector<xlns32> a_row_lns(K);
    // Pre-converted activation rows for one (i12,i13) batch slice: ne11 rows of K.
    // Converting src1 once per slice (rather than once per weight row) so this avoids
    // re-converting each weight row ne11 times.
    std::vector<xlns32> b_lns((size_t)ne11 * (size_t)K);

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            // Convert all src1 (activation) rows for this slice just once.
            for (int64_t i1 = 0; i1 < ne11; i1++) {
                const void * b_row_raw = (const char *)src1->data
                    + i1*nb11 + i12*nb12 + i13*nb13;
                row_to_xlns32(b_row_raw, src1->type, b_lns.data() + (size_t)i1 * (size_t)K,
                              f32_scratch.data(), K);
            }

            for (int64_t i0 = 0; i0 < ne01; i0++) {
                // src0 (weights): any supported type — convert each weight row only once per slice.
                const void * a_row_raw = (const char *)src0->data
                    + i0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
                row_to_xlns32(a_row_raw, type0, a_row_lns.data(), f32_scratch.data(), K);

                for (int64_t i1 = 0; i1 < ne11; i1++) {
                    xlns32 dot = xlns32_vec_dot(a_row_lns.data(),
                                                b_lns.data() + (size_t)i1 * (size_t)K, K);

                    void * dst_ptr = (char *)dst->data + i0*nb0 + i1*nb1 + i12*nb2 + i13*nb3;
                    *(float *)dst_ptr = xlns322fp(dot);
                }
            }
        }
    }
}

// ============================================================
// MUL_MAT_ID: batched/grouped MUL_MAT for MoE expert FFNs
// src0: expert weights [n_embd, n_ff, n_expert]
// src1: activations    [n_embd, n_expert_used_bcast, n_tokens] (F32)
// src2: ids            [n_expert_used, n_tokens] (I32) — selected expert per slot
// dst:  [n_ff, n_expert_used, n_tokens]
// ============================================================

void lns32_mul_mat_id(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const enum ggml_type type0 = src0->type;
    const int64_t K = src0->ne[0]; // n_embd
    const int64_t N = src0->ne[1]; // n_ff
    const int64_t n_expert = src0->ne[2];
    GGML_ASSERT(src1->ne[0] == K);

    // Match ggml_mul_mat_id / CPU kernel invariants
    GGML_ASSERT(src0->nb[0] == ggml_type_size(type0));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(ids->ne[1] == src1->ne[2]); // one id list per activation token

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t ne11          = src1->ne[1]; // activation broadcast rows per token

    GGML_ASSERT(ne11 > 0);
    GGML_ASSERT(n_expert_used % ne11 == 0); // can broadcast activations over slots
    GGML_ASSERT(dst->ne[0] == N);
    GGML_ASSERT(dst->ne[1] == n_expert_used);
    GGML_ASSERT(dst->ne[2] == n_tokens);

    std::vector<float>  f32_scratch(K);
    std::vector<xlns32> w_row_lns(K);
    // Activation rows for the current token, converted once and reused across
    // every expert slot that broadcasts to the same row (ne11 is 1 in practice).
    std::vector<xlns32> act_lns((size_t)ne11 * (size_t)K);

    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t i11 = 0; i11 < ne11; i11++) {
            const void * act_row_raw = (const char *)src1->data + i11*src1->nb[1] + t*src1->nb[2];
            row_to_xlns32(act_row_raw, src1->type, act_lns.data() + (size_t)i11 * (size_t)K,
                          f32_scratch.data(), K);
        }

        for (int64_t s = 0; s < n_expert_used; s++) {
            const int32_t expert = *(const int32_t *)((const char *)ids->data
                + t*ids->nb[1] + s*ids->nb[0]);
            GGML_ASSERT(expert >= 0 && expert < n_expert);

            // Slot-based broadcast (matches CPU one_chunk): act row = s % ne11, not expert % ne11.
            const int64_t i11 = s % ne11;
            const xlns32 * act_row_lns = act_lns.data() + (size_t)i11 * (size_t)K;

            const char * expert_base = (const char *)src0->data + expert*src0->nb[2];
            char * dst_col = (char *)dst->data + s*dst->nb[1] + t*dst->nb[2];

            for (int64_t i0 = 0; i0 < N; i0++) {
                const void * w_row_raw = expert_base + i0*src0->nb[1];
                row_to_xlns32(w_row_raw, type0, w_row_lns.data(), f32_scratch.data(), K);

                xlns32 dot = xlns32_vec_dot(w_row_lns.data(), act_row_lns, K);

                void * dst_ptr = dst_col + i0*dst->nb[0];
                *(float *)dst_ptr = xlns322fp(dot);
            }
        }
    }
}

// ============================================================
// ADD: dst = src0 + src1 (with broadcasting)
// ============================================================

void lns32_add(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    // Match ggml-cpu/binary-ops.cpp apply_binary_op(): full repeat-compatibility
    // across all dims (not just dim 0), plus dst must match src0's shape.
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0 = (const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                const void * s1 = (const char *)src1->data
                    + (i1 % ne11)*nb11 + (i2 % ne12)*nb12 + (i3 % ne13)*nb13;
                void * d = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3;

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns32 a = read_elem_xlns32(s0, i0, src0->type);
                    xlns32 b = read_elem_xlns32(s1, i0 % ne10, src1->type);
                    write_elem_xlns32(d, i0, dst->type, xlns32_add(a, b));
                }
            }
        }
    }
}

// ============================================================
// MUL: dst = src0 * src1 (element-wise with broadcasting)
// ============================================================

void lns32_mul(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    // Match ggml-cpu/binary-ops.cpp apply_binary_op(): full repeat-compatibility
    // across all dims (not just dim 0), plus dst must match src0's shape.
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0 = (const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                const void * s1 = (const char *)src1->data
                    + (i1 % ne11)*nb11 + (i2 % ne12)*nb12 + (i3 % ne13)*nb13;
                void * d = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3;

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns32 a = read_elem_xlns32(s0, i0, src0->type);
                    xlns32 b = read_elem_xlns32(s1, i0 % ne10, src1->type);
                    write_elem_xlns32(d, i0, dst->type, xlns32_mul(a, b));
                }
            }
        }
    }
}

// ============================================================
// DIV: dst = src0 / src1 (element-wise with broadcasting)
// ============================================================

void lns32_div(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0 = (const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                const void * s1 = (const char *)src1->data
                    + (i1 % ne11)*nb11 + (i2 % ne12)*nb12 + (i3 % ne13)*nb13;
                void * d = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3;

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns32 a = read_elem_xlns32(s0, i0, src0->type);
                    xlns32 b = read_elem_xlns32(s1, i0 % ne10, src1->type);
                    write_elem_xlns32(d, i0, dst->type, xlns32_div(a, b));
                }
            }
        }
    }
}

// ============================================================
// SCALE: dst = src0 * s + b
// ============================================================

void lns32_scale(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    float s, b;
    memcpy(&s, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&b, (float *) dst->op_params + 1, sizeof(float));

    const xlns32 s_lns = fp2xlns32(s);
    const xlns32 b_lns = fp2xlns32(b);

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns32 val = fp2xlns32(((const float *)src0->data)[i]);
        val = xlns32_mul(val, s_lns);
        if (b != 0.0f) {
            val = xlns32_add(val, b_lns);
        }
        ((float *)dst->data)[i] = xlns322fp(val);
    }
}

// ============================================================
// CLAMP: dst = max(min(x, max), min)
// ============================================================

void lns32_clamp(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    float min_f, max_f;
    memcpy(&min_f, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_f, (float *) dst->op_params + 1, sizeof(float));

    const xlns32 min_lns = fp2xlns32(min_f);
    const xlns32 max_lns = fp2xlns32(max_f);

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns32 v = fp2xlns32(((const float *)src0->data)[i]);
        xlns32 r = xlns32_max(xlns32_min(v, max_lns), min_lns);
        ((float *)dst->data)[i] = xlns322fp(r);
    }
}

// Convert one F32 mask entry to xlns32 (mask tensor is always F32 in ggml).
// ggml represents "masked out" as a very large negative float (its own
// -infinity convention), not xlns32's own -inf encoding, so this sentinel
// mapping belongs here (ggml-tensor extraction) rather than in xlns32.cpp.
static inline xlns32 f32_mask_to_lns32(float m) {
    if (m <= -3.0e38f) return xlns32_neg_inf;
    return fp2xlns32(m);
}

// ============================================================
// SOFT_MAX: dst = softmax(src0 * scale + mask)
// ============================================================

void lns32_soft_max(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1]; // mask (optional)

    // Same limitations as the LNS16 backend: max_bias/ALiBi and sinks are not
    // implemented (not required by the validated models).

    float scale, max_bias;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    (void)max_bias; // ignored (see above)

    const int64_t ne00 = src0->ne[0]; // sequence length
    const int64_t ne01 = src0->ne[1]; // num queries
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const xlns32 scale_lns = fp2xlns32(scale);

    std::vector<xlns32> row(ne00);
    std::vector<xlns32> mask_lns(src1 ? (size_t)ne00 : 0);

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0_row = (const char *)src0->data
                    + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                void * d_row = (char *)dst->data
                    + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    row[i0] = read_elem_xlns32(s0_row, i0, src0->type);
                }

                if (src1) {
                    const int64_t ne12 = src1->ne[2];
                    const float * mask = (const float *)((const char *)src1->data
                        + i1*src1->nb[1] + (i2 % ne12)*src1->nb[2] + i3*src1->nb[3]);
                    for (int64_t i0 = 0; i0 < ne00; i0++) {
                        mask_lns[(size_t)i0] = f32_mask_to_lns32(mask[i0]);
                    }
                    xlns32_softmax_masked(row.data(), mask_lns.data(), row.data(), (size_t)ne00, scale_lns);
                } else {
                    xlns32_softmax(row.data(), row.data(), (size_t)ne00, scale_lns);
                }

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    write_elem_xlns32(d_row, i0, dst->type, row[i0]);
                }
            }
        }
    }
}

// ============================================================
// RMS_NORM: dst = x / sqrt(mean(x^2) + eps)
// ============================================================

void lns32_rms_norm(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const xlns32 eps_lns = fp2xlns32(eps);

    std::vector<xlns32> row((size_t)ne00);

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s = (const char *)src0->data
                    + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                void * d = (char *)dst->data
                    + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    row[(size_t)i0] = read_elem_xlns32(s, i0, src0->type);
                }

                xlns32_rms_norm(row.data(), row.data(), (size_t)ne00, eps_lns);

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    write_elem_xlns32(d, i0, dst->type, row[(size_t)i0]);
                }
            }
        }
    }
}

// ============================================================
// DIAG_MASK_INF: causal attention mask
// ============================================================

void lns32_diag_mask_inf(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    int n_past;
    memcpy(&n_past, dst->op_params, sizeof(int32_t));

    const int64_t nc = src0->ne[0]; // columns
    const int64_t nr = src0->ne[1]; // rows
    const int64_t n  = ggml_nrows(src0);
    const int64_t nz = n / nr;

    const bool inplace = (src0->data == dst->data);
    if (!inplace) {
        GGML_ASSERT(ggml_nbytes(dst) <= ggml_nbytes(src0));
        memcpy(dst->data, src0->data, ggml_nbytes(dst));
    }

    // dst is F32: write ggml's own -infinity convention (softmax's mask
    // conversion maps it to the xlns32 sentinel later).
    float * data = (float *)dst->data;
    for (int64_t k = 0; k < nz; k++) {
        for (int64_t j = 0; j < nr; j++) {
            for (int64_t i = n_past + j + 1; i < nc; i++) {
                data[k*nr*nc + j*nc + i] = -INFINITY;
            }
        }
    }
}

// ============================================================
// SILU: dst = x * sigmoid(x)
// ============================================================

void lns32_silu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns32 v = fp2xlns32(((const float *)src0->data)[i]);
        ((float *)dst->data)[i] = xlns322fp(xlns32_silu(v));
    }
}

// ============================================================
// SWIGLU: dst = silu(gate) * up
// ============================================================

void lns32_swiglu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    } else {
        GGML_ASSERT(src0->ne[0] % 2 == 0);
    }

    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);
    if (src1) {
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(ggml_nrows(src1) == nr);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const size_t src0_elem_size = ggml_type_size(src0->type);
    const size_t src1_elem_size = src1 ? ggml_type_size(src1->type) : src0_elem_size;

    const char * src0_data = (const char *)src0->data;
    const char * src1_data = src1 ? (const char *)src1->data : src0_data;
    char * dst_data = (char *)dst->data;

    for (int64_t i1 = 0; i1 < nr; i1++) {
        const char * gate_row = src0_data + i1*src0->nb[1];
        const char * up_row   = src1_data + i1*(src1 ? src1->nb[1] : src0->nb[1]);
        char * dst_row = dst_data + i1*dst->nb[1];

        if (!src1) {
            gate_row += (swapped ? nc : 0) * src0_elem_size;
            up_row   += (swapped ? 0 : nc) * src1_elem_size;
        }

        for (int64_t k = 0; k < nc; k++) {
            const xlns32 gate = read_elem_xlns32(gate_row, k, src0->type);
            const xlns32 up   = read_elem_xlns32(up_row,   k, src1 ? src1->type : src0->type);
            write_elem_xlns32(dst_row, k, dst->type, xlns32_mul(xlns32_silu(gate), up));
        }
    }
}

// ============================================================
// GEGLU: dst = gelu(gate) * up
// Same structure as SWIGLU (lns32_swiglu above), just gelu instead of silu.
// ============================================================

void lns32_geglu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    } else {
        GGML_ASSERT(src0->ne[0] % 2 == 0);
    }

    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);
    if (src1) {
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(ggml_nrows(src1) == nr);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const size_t src0_elem_size = ggml_type_size(src0->type);
    const size_t src1_elem_size = src1 ? ggml_type_size(src1->type) : src0_elem_size;

    const char * src0_data = (const char *)src0->data;
    const char * src1_data = src1 ? (const char *)src1->data : src0_data;
    char * dst_data = (char *)dst->data;

    for (int64_t i1 = 0; i1 < nr; i1++) {
        const char * gate_row = src0_data + i1*src0->nb[1];
        const char * up_row   = src1_data + i1*(src1 ? src1->nb[1] : src0->nb[1]);
        char * dst_row = dst_data + i1*dst->nb[1];

        if (!src1) {
            gate_row += (swapped ? nc : 0) * src0_elem_size;
            up_row   += (swapped ? 0 : nc) * src1_elem_size;
        }

        for (int64_t k = 0; k < nc; k++) {
            const xlns32 gate = read_elem_xlns32(gate_row, k, src0->type);
            const xlns32 up   = read_elem_xlns32(up_row,   k, src1 ? src1->type : src0->type);
            write_elem_xlns32(dst_row, k, dst->type, xlns32_mul(xlns32_gelu(gate), up));
        }
    }
}

// ============================================================
// GELU: dst = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// ============================================================

void lns32_gelu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns32 v = fp2xlns32(((const float *)src0->data)[i]);
        ((float *)dst->data)[i] = xlns322fp(xlns32_gelu(v));
    }
}

// ============================================================
// RELU: dst = max(0, x)
// ============================================================

void lns32_relu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns32 v = fp2xlns32(((const float *)src0->data)[i]);
        ((float *)dst->data)[i] = xlns322fp(xlns32_relu(v));
    }
}

// ============================================================
// TANH: dst = tanh(x)
// ============================================================

void lns32_tanh(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns32 v = fp2xlns32(((const float *)src0->data)[i]);
        ((float *)dst->data)[i] = xlns322fp(xlns32_tanh(v));
    }
}

// ============================================================
// GET_ROWS: gather rows from src0 using indices in src1
// Pure dequantization to F32 (no LNS arithmetic) — implemented so the
// embedding lookup stays on this backend and avoids a graph split.
// ============================================================

void lns32_get_rows(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // embedding table
    const struct ggml_tensor * src1 = dst->src[1]; // indices (I32)

    const int64_t nc = src0->ne[0]; // embedding dim
    const enum ggml_type type = src0->type;
    const auto * traits = ggml_get_type_traits(type);

    const int64_t nr = ggml_nelements(src1); // number of rows to gather

    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    for (int64_t i = 0; i < nr; i++) {
        // Compute multi-dimensional index into src1
        const int64_t ne10 = src1->ne[0];
        const int64_t ne11 = src1->ne[1];
        const int64_t i12 = i / (ne11 * ne10);
        const int64_t i11 = (i - i12*ne11*ne10) / ne10;
        const int64_t i10 = i - i12*ne11*ne10 - i11*ne10;

        // Get row index
        const int32_t row_idx = ((const int32_t *)((const char *)src1->data
            + i10*src1->nb[0] + i11*src1->nb[1] + i12*src1->nb[2]))[0];
        GGML_ASSERT(row_idx >= 0 && row_idx < src0->ne[1]);

        // Source row (embedding table — always a weight type)
        const void * src_row = (const char *)src0->data
            + row_idx*src0->nb[1] + i11*src0->nb[2] + i12*src0->nb[3];

        float * dst_row = (float *)((char *)dst->data
            + i10*dst->nb[1] + i11*dst->nb[2] + i12*dst->nb[3]);
        if (type == GGML_TYPE_F32) {
            memcpy(dst_row, src_row, (size_t)nc * sizeof(float));
        } else {
            GGML_ASSERT(traits->to_float != NULL);
            traits->to_float(src_row, dst_row, nc);
        }
    }
}

// ============================================================
// ARGSORT: dst = permutation of indices sorting src0's rows
// Reference: ggml-cpu/ops.cpp ggml_compute_forward_argsort_f32()
// Pure comparison, no arithmetic — xlns32_lt/xlns32_gt (via xlns32_canon)
// ============================================================

void lns32_argsort(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    const int64_t nc = src0->ne[0];
    const int64_t nr = ggml_nrows(src0);

    const enum ggml_sort_order order = (enum ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    std::vector<xlns32> row((size_t)nc);

    for (int64_t i = 0; i < nr; i++) {
        const void * src_row = (const char *)src0->data + i*src0->nb[1];
        int32_t * dst_row = (int32_t *)((char *)dst->data + i*dst->nb[1]);

        for (int64_t j = 0; j < nc; j++) {
            row[(size_t)j] = read_elem_xlns32(src_row, j, src0->type);
            dst_row[j] = (int32_t)j;
        }

        switch (order) {
            case GGML_SORT_ORDER_ASC:
                std::sort(dst_row, dst_row + nc, [&row](int32_t a, int32_t b) {
                    return xlns32_lt(row[(size_t)a], row[(size_t)b]);
                });
                break;
            case GGML_SORT_ORDER_DESC:
                std::sort(dst_row, dst_row + nc, [&row](int32_t a, int32_t b) {
                    return xlns32_gt(row[(size_t)a], row[(size_t)b]);
                });
                break;
            default:
                GGML_ABORT("lns32_argsort: invalid sort order");
        }
    }
}

// ============================================================
// ROPE: Rotary Position Embeddings
// ============================================================

void lns32_rope(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // Q or K tensor
    const struct ggml_tensor * src1 = dst->src[1]; // position indices (I32)

    const int64_t ne0 = src0->ne[0]; // head dim
    const int64_t ne1 = src0->ne[1]; // num heads
    const int64_t ne2 = src0->ne[2]; // seq len
    const int64_t ne3 = src0->ne[3]; // batch

    const int32_t n_dims     = ((int32_t *) dst->op_params)[1];
    const int32_t mode       = ((int32_t *) dst->op_params)[2];
    const int32_t n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    const bool is_neox = (mode & 2) != 0; // NEOX-style: pairs at distance n_dims/2

    // Same limitations as the LNS16 backend: freq_scale/ext_factor/attn_factor/
    // beta_fast/beta_slow/n_ctx_orig are ignored (not required by the
    // validated models).
    (void)n_ctx_orig; (void)ext_factor; (void)attn_factor;
    (void)beta_fast; (void)beta_slow; (void)freq_scale;

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int32_t pos = ((const int32_t *)src1->data)[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const void * src_row = (const char *)src0->data
                    + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                void * dst_row = (char *)dst->data
                    + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

                float t = (float)pos;
                for (int64_t j = 0; j < n_dims; j += 2) {
                    float cos_t = cosf(t);
                    float sin_t = sinf(t);

                    // Convert cos/sin to xlns32 for the rotation
                    xlns32 c     = fp2xlns32(cos_t);
                    xlns32 s_val = fp2xlns32(sin_t);

                    int64_t i0_0, i0_1;
                    if (is_neox) {
                        // NEOX: pairs at (j/2, j/2 + n_dims/2)
                        i0_0 = j / 2;
                        i0_1 = j / 2 + n_dims / 2;
                    } else {
                        // Normal: adjacent pairs (j, j+1)
                        i0_0 = j;
                        i0_1 = j + 1;
                    }

                    xlns32 x0 = read_elem_xlns32(src_row, i0_0, src0->type);
                    xlns32 x1 = read_elem_xlns32(src_row, i0_1, src0->type);

                    // Rotation: [cos -sin; sin cos] * [x0; x1]
                    xlns32 r0 = xlns32_sub(xlns32_mul(x0, c), xlns32_mul(x1, s_val));
                    xlns32 r1 = xlns32_add(xlns32_mul(x0, s_val), xlns32_mul(x1, c));

                    write_elem_xlns32(dst_row, i0_0, dst->type, r0);
                    write_elem_xlns32(dst_row, i0_1, dst->type, r1);

                    t *= theta_scale;
                }

                // Copy non-rotated dimensions (passthrough)
                for (int64_t j = n_dims; j < ne0; j++) {
                    xlns32 v = read_elem_xlns32(src_row, j, src0->type);
                    write_elem_xlns32(dst_row, j, dst->type, v);
                }
            }
        }
    }
}

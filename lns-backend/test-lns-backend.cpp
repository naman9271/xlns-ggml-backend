#include "ggml.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Test the LNS backends with the same matrix multiply as simple-backend.cpp
//
// Usage: test-lns-backend [DEVICE] [PROFILE]
//   DEVICE  — backend device name to test: LNS16 (default) or LNS32
//   PROFILE — tolerance profile: base (default) | monte | lpvip
//             (must match the GGML_LNS16_SUM mode the backend was built with;
//             ignored for LNS32, which always uses its own tight profile)

// Per-mode tolerances for the summation-dependent tests. The elementwise
// tests (div/swiglu/geglu/tanh/clamp/argsort) keep their fixed tolerances —
// no summation variant touches them.
struct tol_profile {
    const char * name;
    float mul_mat;     // scaled error, MUL_MAT reference case
    float mul_mat_id;  // scaled error, MUL_MAT_ID cases
    float softmax;     // absolute error on probabilities
    float rms_norm;    // scaled error
};

// base:  plain xlns16 accumulation (matches the historical 0.006 matmul gate).
//        rms_norm at n=576 shows the known accumulator-sticking bias (~4%
//        observed): the partial sum of squares dwarfs later addends, which is
//        precisely the phenomenon the monte/lpvip variants address.
// monte: stochastic dither — unbiased in EXPECTATION but any single reduction
//        can err more than base (5.7% observed on rms_norm 576 with the
//        default LFSR seed; the draw depends on the LFSR phase, i.e. on how
//        many monte adds ran earlier in the process).
// lpvip: 32-bit LPVIP accumulator — slightly looser than base on short sums
//        (Mitchell-approximated adds, 1.6% observed on the K=2 matmul),
//        dramatically better on long sums (0.77% observed on rms_norm 576,
//        vs base's 4%) — which is its purpose.
// lns32: 23-bit log fraction, F32-class accuracy
static const tol_profile PROFILES[] = {
    { "base",  0.006f, 0.02f,  0.02f, 0.06f },
    { "monte", 0.03f,  0.05f,  0.05f, 0.08f },
    { "lpvip", 0.02f,  0.03f,  0.02f, 0.02f },
    { "lns32", 0.001f, 0.001f, 0.001f, 0.001f },
};

static tol_profile g_tol = PROFILES[0];
// True when testing the 16-bit backend (device LNS16); the LNS16-typed-dst
// mul_mat_id case only applies there.
static bool g_device_is_lns16 = true;

const int rows_A = 4, cols_A = 2;
float matrix_A[rows_A * cols_A] = {
    2, 8,
    5, 1,
    4, 2,
    8, 6
};

const int rows_B = 3, cols_B = 2;
float matrix_B[rows_B * cols_B] = {
    10, 5,
    9, 9,
    5, 4
};

// Expected result (A * B^T):
// [ 60  90  42
//   55  54  29
//   50  54  28
//  110 126  64 ]
// Transposed (as ggml outputs): 4 cols x 3 rows
float expected[12] = {
    60, 55, 50, 110,
    90, 54, 54, 126,
    42, 29, 28,  64
};

static float silu_ref(float x) {
    return x / (1.0f + expf(-x));
}

static float gelu_ref(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

static bool check_close(const char * name, const std::vector<float> & out_data, const float * expected_data, float tol) {
    float max_scaled_err = 0.0f;
    bool pass = true;

    for (size_t i = 0; i < out_data.size(); i++) {
        const float abs_err = fabsf(out_data[i] - expected_data[i]);
        const float scaled_err = abs_err / fmaxf(fabsf(expected_data[i]), 1.0f);
        if (scaled_err > max_scaled_err) {
            max_scaled_err = scaled_err;
        }
        if (scaled_err > tol) {
            printf("FAIL %s: element %zu: got %.6f, expected %.6f (scaled err %.4f%%)\n",
                   name, i, out_data[i], expected_data[i], scaled_err * 100.0f);
            pass = false;
        }
    }

    printf("%s max scaled error: %.4f%%\n", name, max_scaled_err * 100.0f);
    printf("%s result: %s\n\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

static ggml_context * make_context(std::vector<uint8_t> & buf) {
    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    buf.resize(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    return ggml_init(params0);
}

static bool run_mul_mat_test(ggml_backend_sched_t sched) {
    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);

    struct ggml_tensor * result = ggml_mul_mat(ctx, a, b);
    ggml_build_forward_expand(gf, result);

    // Compute
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);

    ggml_backend_tensor_set(a, matrix_A, 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, matrix_B, 0, ggml_nbytes(b));

    ggml_backend_sched_graph_compute(sched, gf);

    // Read results
    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    // Print results
    printf("LNS mul mat (%d x %d):\n[", (int)out->ne[0], (int)out->ne[1]);
    for (int j = 0; j < out->ne[1]; j++) {
        if (j > 0) printf("\n");
        for (int i = 0; i < out->ne[0]; i++) {
            printf(" %.2f", out_data[j * out->ne[0] + i]);
        }
    }
    printf(" ]\n\n");

    bool pass = check_close("LNS mul mat", out_data, expected, g_tol.mul_mat);
    ggml_free(ctx);
    return pass;
}

// ============================================================
// MUL_MAT_ID: batched/grouped matmul for MoE expert FFNs.
// Reference: dst[:, s, t] = src0[:, :, ids[s,t]] @ src1[:, s % ne11, t]
// (mirrors lns_mul_mat_id in lns-backend/src/lns-ops.cpp and the naive
// float oracle in scripts/test-xlns16-mul-mat-id.cpp).
// ============================================================

static void fill_ramp(std::vector<float> & v, float start, float step) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = start + step * (float)i;
    }
}

static void mul_mat_id_ref(
        const std::vector<float> & as, const std::vector<float> & b,
        const std::vector<int32_t> & ids, std::vector<float> & dst,
        int64_t K, int64_t N, int64_t ne11, int64_t n_expert_used, int64_t n_tokens) {
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t s = 0; s < n_expert_used; s++) {
            const int32_t expert = ids[(size_t)(t * n_expert_used + s)];
            const int64_t i11 = s % ne11;
            const float * act    = b.data() + (size_t)((t * ne11 + i11) * K);
            const float * w_base = as.data() + (size_t)((int64_t)expert * N * K);
            float * dst_col = dst.data() + (size_t)((t * n_expert_used + s) * N);
            for (int64_t i0 = 0; i0 < N; i0++) {
                const float * w = w_base + i0 * K;
                float sum = 0.0f;
                for (int64_t k = 0; k < K; k++) {
                    sum += w[k] * act[k];
                }
                dst_col[i0] = sum;
            }
        }
    }
}

static bool run_mul_mat_id_case(ggml_backend_sched_t sched, ggml_backend_t lns_backend, const char * name,
                                int64_t K, int64_t N, int64_t n_expert, int64_t ne11,
                                int64_t n_expert_used, int64_t n_tokens,
                                const std::vector<int32_t> & ids, float tol, bool lns16_dst) {
    std::vector<float> as((size_t)(K * N * n_expert));
    std::vector<float> b((size_t)(K * ne11 * n_tokens));
    fill_ramp(as, -1.5f, 0.07f);
    fill_ramp(b, 0.3f, -0.05f);

    // Perturb a few entries so experts / broadcast rows are distinguishable.
    for (int64_t e = 0; e < n_expert; e++) {
        as[(size_t)(e * N * K)] += 0.25f * (float)e;
    }
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t i11 = 0; i11 < ne11; i11++) {
            b[(size_t)((t * ne11 + i11) * K)] += 0.1f * (float)(t + i11);
        }
    }

    std::vector<float> expected((size_t)(N * n_expert_used * n_tokens));
    mul_mat_id_ref(as, b, ids, expected, K, N, ne11, n_expert_used, n_tokens);

    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * t_as  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, n_expert);
    struct ggml_tensor * t_b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, ne11, n_tokens);
    struct ggml_tensor * t_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);

    struct ggml_tensor * result;
    if (lns16_dst) {
        // ggml_mul_mat_id() always allocates an F32 result; build the node
        // manually (same op/src wiring) to exercise the LNS16 dst write path.
        const int64_t ne[4] = { N, n_expert_used, n_tokens, 1 };
        result = ggml_new_tensor(ctx, GGML_TYPE_LNS16, 4, ne);
        result->op     = GGML_OP_MUL_MAT_ID;
        result->src[0] = t_as;
        result->src[1] = t_b;
        result->src[2] = t_ids;
        // GGML_TYPE_LNS16 has no to_float in ggml's core type-traits table
        // (only the LNS backend's own lns_cpy knows how to convert it) —
        // so route the LNS16 result through a cast node to read it back.
        // NOTE: ggml_backend_sched's generic tie-break heuristic routes this
        // plain CPY node to the CPU backend by default even though the LNS
        // backend also declares support for it (see lns_supports_op's
        // GGML_OP_CPY case) — and CPU's dup implementation aborts on the
        // unknown LNS16 type. Pin it to the LNS backend explicitly so this
        // test isolates lns_mul_mat_id's own LNS16 write-path correctness
        // rather than that separate scheduler/CPY interaction gap.
        struct ggml_tensor * mul_mat_id_result = result;
        result = ggml_cast(ctx, result, GGML_TYPE_F32);
        ggml_build_forward_expand(gf, result);

        // ggml_backend_sched_reset() wipes any prior tensor->backend
        // assignment, so the manual pin below must come AFTER reset and
        // BEFORE alloc — reversing this order silently drops the override.
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_set_tensor_backend(sched, mul_mat_id_result, lns_backend);
        ggml_backend_sched_set_tensor_backend(sched, result, lns_backend);
        ggml_backend_sched_alloc_graph(sched, gf);
    } else {
        result = ggml_mul_mat_id(ctx, t_as, t_b, t_ids);
        ggml_build_forward_expand(gf, result);

        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
    }

    ggml_backend_tensor_set(t_as, as.data(), 0, ggml_nbytes(t_as));
    ggml_backend_tensor_set(t_b, b.data(), 0, ggml_nbytes(t_b));
    ggml_backend_tensor_set(t_ids, ids.data(), 0, ggml_nbytes(t_ids));

    ggml_backend_sched_graph_compute(sched, gf);

    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data((size_t)ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    bool pass = check_close(name, out_data, expected.data(), tol);
    ggml_free(ctx);
    return pass;
}

static bool run_mul_mat_id_test(ggml_backend_sched_t sched, ggml_backend_t lns_backend) {
    bool pass = true;

    // Non-broadcast: ne11 == n_expert_used, includes the same expert selected
    // twice in one token's slot list (must produce independent, correct columns).
    {
        const int64_t K = 8, N = 4, n_expert = 6, n_expert_used = 3, n_tokens = 4;
        const int64_t ne11 = n_expert_used;
        std::vector<int32_t> ids = {
            0, 2, 5, // t=0
            1, 3, 4, // t=1
            5, 0, 1, // t=2
            2, 2, 3, // t=3 duplicate expert 2 in two slots
        };
        pass = run_mul_mat_id_case(sched, lns_backend, "LNS mul_mat_id non-broadcast",
                K, N, n_expert, ne11, n_expert_used, n_tokens, ids, g_tol.mul_mat_id, false) && pass;
    }

    // Broadcast: ne11 == 1, n_expert_used > 1 (the common real-world MoE shape).
    {
        const int64_t K = 8, N = 5, n_expert = 8, n_expert_used = 4, n_tokens = 3;
        const int64_t ne11 = 1;
        std::vector<int32_t> ids = {
            0, 1, 2, 7, // t=0
            3, 4, 5, 6, // t=1
            1, 1, 0, 7, // t=2 duplicate expert 1 in two slots
        };
        pass = run_mul_mat_id_case(sched, lns_backend, "LNS mul_mat_id broadcast ne11=1",
                K, N, n_expert, ne11, n_expert_used, n_tokens, ids, g_tol.mul_mat_id, false) && pass;
    }

    // Partial broadcast: ne11 == 2, n_expert_used == 4 (slots 0,2 share row 0; 1,3 share row 1).
    {
        const int64_t K = 6, N = 3, n_expert = 5, n_expert_used = 4, n_tokens = 2;
        const int64_t ne11 = 2;
        std::vector<int32_t> ids = {
            0, 4, 2, 1, // t=0
            3, 0, 4, 2, // t=1
        };
        pass = run_mul_mat_id_case(sched, lns_backend, "LNS mul_mat_id partial broadcast ne11=2",
                K, N, n_expert, ne11, n_expert_used, n_tokens, ids, g_tol.mul_mat_id, false) && pass;
    }

    // Same broadcast shape as above, but with an LNS16-typed dst to exercise
    // the xlns16 write path instead of the F32 conversion path.
    // Only meaningful (and only supported) on the 16-bit backend: LNS32 has
    // no LNS-typed ggml tensors at all.
    if (g_device_is_lns16) {
        const int64_t K = 8, N = 5, n_expert = 8, n_expert_used = 4, n_tokens = 3;
        const int64_t ne11 = 1;
        std::vector<int32_t> ids = {
            0, 1, 2, 7,
            3, 4, 5, 6,
            1, 1, 0, 7,
        };
        pass = run_mul_mat_id_case(sched, lns_backend, "LNS mul_mat_id dst=LNS16",
                K, N, n_expert, ne11, n_expert_used, n_tokens, ids, g_tol.mul_mat_id, true) && pass;
    }

    return pass;
}

static bool run_div_case(ggml_backend_sched_t sched, const char * name,
                         const float * a, int64_t ne00, int64_t ne01,
                         const float * b, int64_t ne10, int64_t ne11,
                         const float * expected, float tol) {
    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * ta = (ne01 == 1)
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne00)
        : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ne01);
    struct ggml_tensor * tb = (ne11 == 1 && ne10 == 1)
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
        : (ne11 == 1)
            ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne10)
            : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne10, ne11);

    struct ggml_tensor * result = ggml_div(ctx, ta, tb);
    ggml_build_forward_expand(gf, result);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(ta, a, 0, ggml_nbytes(ta));
    ggml_backend_tensor_set(tb, b, 0, ggml_nbytes(tb));
    ggml_backend_sched_graph_compute(sched, gf);

    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    printf("%s: [", name);
    for (size_t i = 0; i < out_data.size(); i++) printf(" %.6f", out_data[i]);
    printf(" ]\n\n");

    bool pass = check_close(name, out_data, expected, tol);
    ggml_free(ctx);
    return pass;
}

static bool run_div_test(ggml_backend_sched_t sched) {
    bool pass = true;

    // Same-shape 1D
    {
        constexpr int n = 8;
        const float a[n] = { 8.0f, -6.0f, 5.0f,  1.0f, 0.5f, -0.25f, 100.0f, -3.0f };
        const float b[n] = { 2.0f,  3.0f, -1.0f, 4.0f, 0.5f,  0.1f,   7.0f,  -6.0f };
        float expected_div[n];
        for (int i = 0; i < n; i++) expected_div[i] = a[i] / b[i];
        pass = run_div_case(sched, "LNS div", a, n, 1, b, n, 1, expected_div, 0.01f) && pass;
    }

    // Scalar broadcast: ne10 == 1
    {
        constexpr int n = 6;
        const float a[n] = { 8.0f, -6.0f, 5.0f, 1.0f, 0.5f, -0.25f };
        const float b[1] = { 2.0f };
        float expected_div[n];
        for (int i = 0; i < n; i++) expected_div[i] = a[i] / b[0];
        pass = run_div_case(sched, "LNS div scalar broadcast", a, n, 1, b, 1, 1, expected_div, 0.01f) && pass;
    }

    // 2D with tiled dim0 (ne10=2, ne00=4) and row broadcast (ne11=1, ne01=3)
    {
        constexpr int ne00 = 4, ne01 = 3, ne10 = 2, ne11 = 1;
        const float a[ne00 * ne01] = {
            8.0f, -6.0f, 5.0f, 1.0f,
            4.0f, -2.0f, 9.0f, 3.0f,
            1.0f,  2.0f, 6.0f, -4.0f,
        };
        const float b[ne10 * ne11] = { 2.0f, -1.0f };
        float expected_div[ne00 * ne01];
        for (int i1 = 0; i1 < ne01; i1++) {
            for (int i0 = 0; i0 < ne00; i0++) {
                expected_div[i1 * ne00 + i0] = a[i1 * ne00 + i0] / b[i0 % ne10];
            }
        }
        pass = run_div_case(sched, "LNS div 2d broadcast",
                            a, ne00, ne01, b, ne10, ne11, expected_div, 0.01f) && pass;
    }

    return pass;
}

static bool run_swiglu_test(ggml_backend_sched_t sched, const char * name, bool split, bool swapped) {
    constexpr int rows = 2;
    constexpr int cols = 4;

    const float gate[rows * cols] = {
        -3.0f, -1.0f,  0.0f,  2.0f,
         1.5f, -0.5f,  4.0f, -2.0f,
    };
    const float up[rows * cols] = {
         0.5f, -2.0f,  3.0f,  1.0f,
        -1.0f,  4.0f,  0.25f, 2.0f,
    };
    float expected_swiglu[rows * cols];
    float fused[rows * cols * 2];

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const int src_i = r*cols + c;
            const int fused_row = r*cols*2;
            expected_swiglu[src_i] = silu_ref(gate[src_i]) * up[src_i];
            if (swapped) {
                fused[fused_row + c]        = up[src_i];
                fused[fused_row + cols + c] = gate[src_i];
            } else {
                fused[fused_row + c]        = gate[src_i];
                fused[fused_row + cols + c] = up[src_i];
            }
        }
    }

    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * result = nullptr;
    struct ggml_tensor * gate_tensor = nullptr;
    struct ggml_tensor * up_tensor = nullptr;
    struct ggml_tensor * fused_tensor = nullptr;

    if (split) {
        gate_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        up_tensor   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        result = ggml_swiglu_split(ctx, gate_tensor, up_tensor);
    } else {
        fused_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols*2, rows);
        result = swapped ? ggml_swiglu_swapped(ctx, fused_tensor) : ggml_swiglu(ctx, fused_tensor);
    }

    ggml_build_forward_expand(gf, result);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);

    if (split) {
        ggml_backend_tensor_set(gate_tensor, gate, 0, ggml_nbytes(gate_tensor));
        ggml_backend_tensor_set(up_tensor, up, 0, ggml_nbytes(up_tensor));
    } else {
        ggml_backend_tensor_set(fused_tensor, fused, 0, ggml_nbytes(fused_tensor));
    }

    ggml_backend_sched_graph_compute(sched, gf);

    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    printf("%s (%d x %d):\n[", name, (int)out->ne[0], (int)out->ne[1]);
    for (int j = 0; j < out->ne[1]; j++) {
        if (j > 0) printf("\n");
        for (int i = 0; i < out->ne[0]; i++) {
            printf(" %.6f", out_data[j * out->ne[0] + i]);
        }
    }
    printf(" ]\n\n");

    bool pass = check_close(name, out_data, expected_swiglu, 0.01f);
    ggml_free(ctx);
    return pass;
}

static bool run_geglu_test(ggml_backend_sched_t sched, const char * name, bool split, bool swapped) {
    constexpr int rows = 2;
    constexpr int cols = 4;

    const float gate[rows * cols] = {
        -3.0f, -1.0f,  0.0f,  2.0f,
         1.5f, -0.5f,  4.0f, -2.0f,
    };
    const float up[rows * cols] = {
         0.5f, -2.0f,  3.0f,  1.0f,
        -1.0f,  4.0f,  0.25f, 2.0f,
    };
    float expected_geglu[rows * cols];
    float fused[rows * cols * 2];

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const int src_i = r*cols + c;
            const int fused_row = r*cols*2;
            expected_geglu[src_i] = gelu_ref(gate[src_i]) * up[src_i];
            if (swapped) {
                fused[fused_row + c]        = up[src_i];
                fused[fused_row + cols + c] = gate[src_i];
            } else {
                fused[fused_row + c]        = gate[src_i];
                fused[fused_row + cols + c] = up[src_i];
            }
        }
    }

    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * result = nullptr;
    struct ggml_tensor * gate_tensor = nullptr;
    struct ggml_tensor * up_tensor = nullptr;
    struct ggml_tensor * fused_tensor = nullptr;

    if (split) {
        gate_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        up_tensor   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        result = ggml_geglu_split(ctx, gate_tensor, up_tensor);
    } else {
        fused_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols*2, rows);
        result = swapped ? ggml_geglu_swapped(ctx, fused_tensor) : ggml_geglu(ctx, fused_tensor);
    }

    ggml_build_forward_expand(gf, result);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);

    if (split) {
        ggml_backend_tensor_set(gate_tensor, gate, 0, ggml_nbytes(gate_tensor));
        ggml_backend_tensor_set(up_tensor, up, 0, ggml_nbytes(up_tensor));
    } else {
        ggml_backend_tensor_set(fused_tensor, fused, 0, ggml_nbytes(fused_tensor));
    }

    ggml_backend_sched_graph_compute(sched, gf);

    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    printf("%s (%d x %d):\n[", name, (int)out->ne[0], (int)out->ne[1]);
    for (int j = 0; j < out->ne[1]; j++) {
        if (j > 0) printf("\n");
        for (int i = 0; i < out->ne[0]; i++) {
            printf(" %.6f", out_data[j * out->ne[0] + i]);
        }
    }
    printf(" ]\n\n");

    bool pass = check_close(name, out_data, expected_geglu, 0.01f);
    ggml_free(ctx);
    return pass;
}

static bool run_tanh_test(ggml_backend_sched_t sched) {
    constexpr int n = 8;
    const float a[n] = { -4.0f, -1.5f, -0.5f, -0.01f, 0.0f, 0.5f, 1.5f, 4.0f };
    float expected_tanh[n];
    for (int i = 0; i < n; i++) expected_tanh[i] = tanhf(a[i]);

    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * ta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * result = ggml_tanh(ctx, ta);
    ggml_build_forward_expand(gf, result);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(ta, a, 0, ggml_nbytes(ta));
    ggml_backend_sched_graph_compute(sched, gf);

    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    printf("LNS tanh: [");
    for (int i = 0; i < n; i++) printf(" %.6f", out_data[i]);
    printf(" ]\n\n");

    bool pass = check_close("LNS tanh", out_data, expected_tanh, 0.01f);
    ggml_free(ctx);
    return pass;
}

static bool run_one_clamp_case(ggml_backend_sched_t sched, const char * name,
                               const float * a, int64_t ne0, int64_t ne1,
                               float clamp_min, float clamp_max) {
    const int64_t n = ne0 * ne1;
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; i++) {
        expected[i] = fmaxf(fminf(a[i], clamp_max), clamp_min);
    }

    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * ta = (ne1 == 1)
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0)
        : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    struct ggml_tensor * result = ggml_clamp(ctx, ta, clamp_min, clamp_max);
    ggml_build_forward_expand(gf, result);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(ta, a, 0, ggml_nbytes(ta));
    ggml_backend_sched_graph_compute(sched, gf);

    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    printf("%s: [", name);
    for (int64_t i = 0; i < n; i++) printf(" %.6f", out_data[i]);
    printf(" ]\n\n");

    bool pass = check_close(name, out_data, expected.data(), 0.01f);
    ggml_free(ctx);
    return pass;
}

static bool run_clamp_test(ggml_backend_sched_t sched) {
    constexpr int n = 6;
    const float a[n] = { -5.0f, -1.0f, 0.0f, 0.5f, 2.0f, 10.0f };

    // 2x3 contiguous layout of the same values
    constexpr int ne0 = 3, ne1 = 2;
    const float a2d[ne0 * ne1] = {
        -5.0f, -1.0f, 0.0f,
         0.5f,  2.0f, 10.0f,
    };

    bool pass = true;
    pass = run_one_clamp_case(sched, "LNS clamp", a, n, 1, -1.0f, 2.0f) && pass;
    pass = run_one_clamp_case(sched, "LNS clamp +inf max", a, n, 1, -1.0f, INFINITY) && pass;
    pass = run_one_clamp_case(sched, "LNS clamp -inf min", a, n, 1, -INFINITY, 2.0f) && pass;
    pass = run_one_clamp_case(sched, "LNS clamp min>max", a, n, 1, 2.0f, -1.0f) && pass;
    pass = run_one_clamp_case(sched, "LNS clamp 2d", a2d, ne0, ne1, -1.0f, 2.0f) && pass;
    return pass;
}

// Soft check for argsort with possible ties: each row must be a permutation of
// 0..ne0-1 whose values are sorted in the requested order (equal values OK).
static bool check_argsort_valid(const char * name, const float * a, int64_t ne0, int64_t ne1,
                                enum ggml_sort_order order, const std::vector<int32_t> & out) {
    bool pass = true;
    for (int64_t r = 0; r < ne1; r++) {
        const float * row = a + r * ne0;
        const int32_t * idx = out.data() + r * ne0;

        std::vector<bool> seen((size_t)ne0, false);
        for (int64_t j = 0; j < ne0; j++) {
            const int32_t k = idx[j];
            if (k < 0 || k >= ne0 || seen[(size_t)k]) {
                printf("FAIL %s: row %lld: invalid index %d at position %lld\n",
                       name, (long long)r, k, (long long)j);
                pass = false;
                break;
            }
            seen[(size_t)k] = true;
        }
        if (!pass) continue;

        for (int64_t j = 0; j + 1 < ne0; j++) {
            const float left  = row[idx[j]];
            const float right = row[idx[j + 1]];
            const bool ok = (order == GGML_SORT_ORDER_ASC) ? (left <= right) : (left >= right);
            if (!ok) {
                printf("FAIL %s: row %lld: values not sorted at %lld..%lld (%g vs %g)\n",
                       name, (long long)r, (long long)j, (long long)(j + 1), left, right);
                pass = false;
            }
        }
    }
    return pass;
}

static bool run_one_argsort_case(ggml_backend_sched_t sched, const char * name,
                                  const float * a, int64_t ne0, int64_t ne1,
                                  enum ggml_sort_order order,
                                  const int32_t * expected) {
    const int64_t n = ne0 * ne1;
    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * ta = (ne1 == 1)
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0)
        : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    struct ggml_tensor * result = ggml_argsort(ctx, ta, order);
    ggml_build_forward_expand(gf, result);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(ta, a, 0, ggml_nbytes(ta));
    ggml_backend_sched_graph_compute(sched, gf);

    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<int32_t> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    printf("%s: [", name);
    for (int64_t i = 0; i < n; i++) printf(" %d", out_data[i]);
    printf(" ]\n\n");

    bool pass = true;
    if (expected) {
        for (int64_t i = 0; i < n; i++) {
            if (out_data[i] != expected[i]) {
                printf("FAIL %s: element %lld: got %d, expected %d\n",
                       name, (long long)i, out_data[i], expected[i]);
                pass = false;
            }
        }
    } else {
        pass = check_argsort_valid(name, a, ne0, ne1, order, out_data);
    }
    printf("%s result: %s\n\n", name, pass ? "PASS" : "FAIL");

    ggml_free(ctx);
    return pass;
}

// ============================================================
// SOFT_MAX: validates the summation-variant dispatch (the
// normalization sum is the reduction that base/monte/lpvip differ on).
// ============================================================

static void soft_max_ref(const float * x, const float * mask, float scale,
                         float * out, int64_t n) {
    std::vector<float> t((size_t)n);
    float maxv = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        t[(size_t)i] = x[i] * scale + (mask ? mask[i] : 0.0f);
        if (t[(size_t)i] > maxv) maxv = t[(size_t)i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        t[(size_t)i] = expf(t[(size_t)i] - maxv);
        sum += t[(size_t)i];
    }
    for (int64_t i = 0; i < n; i++) {
        out[i] = t[(size_t)i] / sum;
    }
}

static bool run_one_soft_max_case(ggml_backend_sched_t sched, const char * name,
                                  int64_t ne00, int64_t ne01, float scale, bool masked) {
    std::vector<float> a((size_t)(ne00 * ne01));
    // Mixed-magnitude logits: a deterministic pseudo-pattern with both signs.
    for (int64_t r = 0; r < ne01; r++) {
        for (int64_t i = 0; i < ne00; i++) {
            const float u = (float)((i * 37 + r * 101) % 97) / 97.0f; // [0,1)
            a[(size_t)(r * ne00 + i)] = 8.0f * (u - 0.5f);           // [-4,4)
        }
    }

    std::vector<float> mask;
    if (masked) {
        mask.assign((size_t)(ne00 * ne01), 0.0f);
        // Causal-style tail mask plus a few interior holes (exercises the
        // f32_mask_to_lns sentinel path).
        for (int64_t r = 0; r < ne01; r++) {
            for (int64_t i = 0; i < ne00; i++) {
                const bool tail = i >= ne00 - 1 - r;
                const bool hole = (i % 13) == 5;
                if (tail || hole) mask[(size_t)(r * ne00 + i)] = -INFINITY;
            }
        }
    }

    std::vector<float> expected((size_t)(ne00 * ne01));
    for (int64_t r = 0; r < ne01; r++) {
        soft_max_ref(a.data() + r * ne00,
                     masked ? mask.data() + r * ne00 : nullptr,
                     scale, expected.data() + r * ne00, ne00);
    }

    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ne01);
    struct ggml_tensor * tm = masked ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ne01) : nullptr;
    struct ggml_tensor * result = ggml_soft_max_ext(ctx, ta, tm, scale, 0.0f);
    ggml_build_forward_expand(gf, result);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(ta, a.data(), 0, ggml_nbytes(ta));
    if (masked) {
        ggml_backend_tensor_set(tm, mask.data(), 0, ggml_nbytes(tm));
    }
    ggml_backend_sched_graph_compute(sched, gf);

    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data((size_t)ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    // Probabilities: check absolute error per element, plus each row's sum ≈ 1
    // (informational bias diagnostics are printed for the monte profile).
    bool pass = true;
    float max_abs_err = 0.0f;
    for (size_t i = 0; i < out_data.size(); i++) {
        const float err = fabsf(out_data[i] - expected[i]);
        if (err > max_abs_err) max_abs_err = err;
        if (err > g_tol.softmax) {
            printf("FAIL %s: element %zu: got %.6f, expected %.6f (abs err %.6f)\n",
                   name, i, out_data[i], expected[i], err);
            pass = false;
        }
    }
    for (int64_t r = 0; r < ne01; r++) {
        float row_sum = 0.0f;
        for (int64_t i = 0; i < ne00; i++) row_sum += out_data[(size_t)(r * ne00 + i)];
        if (fabsf(row_sum - 1.0f) > g_tol.softmax * 4.0f) {
            printf("FAIL %s: row %lld sums to %.6f (expected ~1)\n", name, (long long)r, row_sum);
            pass = false;
        }
    }
    printf("%s max abs error: %.6f\n", name, max_abs_err);
    printf("%s result: %s\n\n", name, pass ? "PASS" : "FAIL");

    ggml_free(ctx);
    return pass;
}

static bool run_soft_max_test(ggml_backend_sched_t sched) {
    bool pass = true;
    pass = run_one_soft_max_case(sched, "LNS softmax short",         8,   4, 1.0f,   false) && pass;
    pass = run_one_soft_max_case(sched, "LNS softmax long",          512, 4, 1.0f,   false) && pass;
    pass = run_one_soft_max_case(sched, "LNS softmax scaled",        512, 4, 0.125f, false) && pass;
    pass = run_one_soft_max_case(sched, "LNS softmax masked",        64,  8, 1.0f,   true)  && pass;
    pass = run_one_soft_max_case(sched, "LNS softmax masked scaled", 256, 4, 0.125f, true)  && pass;
    return pass;
}

// ============================================================
// RMS_NORM: validates the sum-of-squares reduction variant.
// ============================================================

static bool run_one_rms_norm_case(ggml_backend_sched_t sched, const char * name,
                                  int64_t ne00, int64_t ne01, float eps) {
    std::vector<float> a((size_t)(ne00 * ne01));
    for (int64_t r = 0; r < ne01; r++) {
        for (int64_t i = 0; i < ne00; i++) {
            const float u = (float)((i * 53 + r * 19) % 89) / 89.0f;  // [0,1)
            a[(size_t)(r * ne00 + i)] = 6.0f * (u - 0.5f);            // [-3,3)
        }
    }

    std::vector<float> expected((size_t)(ne00 * ne01));
    for (int64_t r = 0; r < ne01; r++) {
        double sum_sq = 0.0;
        for (int64_t i = 0; i < ne00; i++) {
            const double v = a[(size_t)(r * ne00 + i)];
            sum_sq += v * v;
        }
        const float inv_rms = 1.0f / sqrtf((float)(sum_sq / (double)ne00) + eps);
        for (int64_t i = 0; i < ne00; i++) {
            expected[(size_t)(r * ne00 + i)] = a[(size_t)(r * ne00 + i)] * inv_rms;
        }
    }

    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ne01);
    struct ggml_tensor * result = ggml_rms_norm(ctx, ta, eps);
    ggml_build_forward_expand(gf, result);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(ta, a.data(), 0, ggml_nbytes(ta));
    ggml_backend_sched_graph_compute(sched, gf);

    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data((size_t)ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    return check_close(name, out_data, expected.data(), g_tol.rms_norm);
}

static bool run_rms_norm_test(ggml_backend_sched_t sched) {
    bool pass = true;
    pass = run_one_rms_norm_case(sched, "LNS rms_norm short", 16,  3, 1e-5f) && pass;
    pass = run_one_rms_norm_case(sched, "LNS rms_norm 576",   576, 3, 1e-5f) && pass;
    return pass;
}

static bool run_argsort_test(ggml_backend_sched_t sched) {
    constexpr int n = 5;
    const float a[n] = { 3.0f, -1.0f, 4.0f, 1.5f, -5.0f };
    const int32_t expected_asc[n]  = { 4, 1, 3, 0, 2 };
    const int32_t expected_desc[n] = { 2, 0, 3, 1, 4 };

    // zeros + mixed signs (single zero — no ties)
    const float zeros_mixed[n] = { -2.0f, 0.0f, 3.0f, -5.0f, 1.0f };
    const int32_t zeros_asc[n]  = { 3, 0, 1, 4, 2 };
    const int32_t zeros_desc[n] = { 2, 4, 1, 0, 3 };

    // 2x3 — exercises nrows / nb[1] row loop
    constexpr int ne0 = 3, ne1 = 2;
    const float a2d[ne0 * ne1] = {
        3.0f, -1.0f, 4.0f,
        1.5f, -5.0f, 0.0f,
    };
    const int32_t expected_2d_asc[ne0 * ne1]  = { 1, 0, 2,  1, 2, 0 };
    const int32_t expected_2d_desc[ne0 * ne1] = { 2, 0, 1,  0, 2, 1 };

    // duplicates — soft-check valid permutation (std::sort is unstable on ties)
    const float dups[n] = { 1.0f, 2.0f, 1.0f, 3.0f, 2.0f };

    bool pass = true;
    pass = run_one_argsort_case(sched, "LNS argsort asc",  a, n, 1, GGML_SORT_ORDER_ASC,  expected_asc)  && pass;
    pass = run_one_argsort_case(sched, "LNS argsort desc", a, n, 1, GGML_SORT_ORDER_DESC, expected_desc) && pass;
    pass = run_one_argsort_case(sched, "LNS argsort zeros asc",  zeros_mixed, n, 1, GGML_SORT_ORDER_ASC,  zeros_asc)  && pass;
    pass = run_one_argsort_case(sched, "LNS argsort zeros desc", zeros_mixed, n, 1, GGML_SORT_ORDER_DESC, zeros_desc) && pass;
    pass = run_one_argsort_case(sched, "LNS argsort 2d asc",  a2d, ne0, ne1, GGML_SORT_ORDER_ASC,  expected_2d_asc)  && pass;
    pass = run_one_argsort_case(sched, "LNS argsort 2d desc", a2d, ne0, ne1, GGML_SORT_ORDER_DESC, expected_2d_desc) && pass;
    pass = run_one_argsort_case(sched, "LNS argsort dups asc",  dups, n, 1, GGML_SORT_ORDER_ASC,  nullptr) && pass;
    pass = run_one_argsort_case(sched, "LNS argsort dups desc", dups, n, 1, GGML_SORT_ORDER_DESC, nullptr) && pass;
    return pass;
}

int main(int argc, char ** argv) {
    const char * dev_name = argc > 1 ? argv[1] : "LNS16";
    const bool is_lns32 = strcmp(dev_name, "LNS32") == 0;
    g_device_is_lns16 = !is_lns32;

    // Tolerance profile: LNS32 always uses its own tight profile; for LNS16
    // argv[2] must match the GGML_LNS16_SUM mode the backend was built with.
    const char * profile_name = is_lns32 ? "lns32" : (argc > 2 ? argv[2] : "base");
    if (!is_lns32 && strcmp(profile_name, "lns32") == 0) {
        fprintf(stderr, "ERROR: profile 'lns32' is reserved for the LNS32 device (use base|monte|lpvip)\n");
        return 1;
    }
    bool profile_found = false;
    for (const tol_profile & p : PROFILES) {
        if (strcmp(p.name, profile_name) == 0) {
            g_tol = p;
            profile_found = true;
            break;
        }
    }
    if (!profile_found) {
        fprintf(stderr, "ERROR: unknown tolerance profile '%s' (expected base|monte|lpvip)\n", profile_name);
        return 1;
    }

    // Route ops to the device under test. setenv with overwrite=0 respects an
    // externally set GGML_LNS_BACKEND; must happen before any ggml call (the
    // backends cache the selection on first use).
    setenv("GGML_LNS_BACKEND", is_lns32 ? "32" : "16", 0);

    ggml_time_init();

    ggml_backend_load_all();

    // Try to init the requested LNS backend by name
    ggml_backend_t lns_backend = ggml_backend_init_by_name(dev_name, nullptr);
    if (!lns_backend) {
        fprintf(stderr, "ERROR: %s backend not found! Make sure ggml was built with -DGGML_LNS=ON -DGGML_LNS32=ON\n", dev_name);
        return 1;
    }

    printf("Using backend: %s (tolerance profile: %s)\n",
           ggml_backend_name(lns_backend), g_tol.name);

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    ggml_backend_t backends[2] = { lns_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);

    bool pass = true;
    pass = run_mul_mat_test(sched) && pass;
    pass = run_mul_mat_id_test(sched, lns_backend) && pass;
    pass = run_div_test(sched) && pass;
    pass = run_swiglu_test(sched, "LNS swiglu split", true, false) && pass;
    pass = run_swiglu_test(sched, "LNS swiglu fused", false, false) && pass;
    pass = run_swiglu_test(sched, "LNS swiglu fused swapped", false, true) && pass;
    pass = run_geglu_test(sched, "LNS geglu split", true, false) && pass;
    pass = run_geglu_test(sched, "LNS geglu fused", false, false) && pass;
    pass = run_geglu_test(sched, "LNS geglu fused swapped", false, true) && pass;
    pass = run_tanh_test(sched) && pass;
    pass = run_clamp_test(sched) && pass;
    pass = run_soft_max_test(sched) && pass;
    pass = run_rms_norm_test(sched) && pass;
    pass = run_argsort_test(sched) && pass;

    ggml_backend_sched_free(sched);
    ggml_backend_free(lns_backend);
    ggml_backend_free(cpu_backend);

    return pass ? 0 : 1;
}

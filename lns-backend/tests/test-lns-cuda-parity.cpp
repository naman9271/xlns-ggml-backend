#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Parity of the LNSCUDA backends with the CPU LNS backends.
//
// Usage: test-lns-cuda-parity
//
// LNSCUDA16 is meant to be bit-identical to LNS16 built with
// GGML_LNS16_SUM=base, and LNSCUDA32 to LNS32: the GPU kernels perform the
// same LNS operations in the same order, with the same lookup tables. Every
// case below builds a small graph on host buffers, computes it on the GPU
// backend and on its CPU counterpart, and compares the outputs bit for bit.
// LNSCUDA32 is held to a small scaled-error tolerance instead (see compare()):
// xlns32's activations (silu, gelu, tanh, and exp inside softmax) evaluate
// expf / tanhf on floats, and CUDA's and glibc's implementations may disagree
// in the last ulp, which grows a little through a chain of ops. Its
// integer-only ops (mul_mat, add / mul / div, rms_norm, rope) come out
// bit-identical, and every difference is reported.
//
// Exit codes: 0 pass, 1 fail, 77 skipped (no CUDA device).

static uint32_t g_rng = 0x2545F491u;

static float rand_unit() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float) (g_rng >> 8) / 16777216.0f * 2.0f - 1.0f;
}

// Inputs are filled by name: "pos" gets token positions, "mask" a causal
// mask with -INFINITY above the diagonal, "freq_factors" RoPE frequency
// factors in [1, 8); everything else gets values in [-2, 2), quantized for
// quantized types.
static void fill_input(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> v((size_t) n);
        for (int64_t i = 0; i < n; i++) v[(size_t) i] = (int32_t) (i * 7 + 3);
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        return;
    }
    std::vector<float> v((size_t) n);
    if (std::strcmp(t->name, "mask") == 0) {
        for (int64_t r = 0; r < n / t->ne[0]; r++) {
            for (int64_t c = 0; c < t->ne[0]; c++) {
                v[(size_t) (r * t->ne[0] + c)] = c > (r % t->ne[1]) + 2 ? -INFINITY : 0.25f * rand_unit();
            }
        }
    } else if (std::strcmp(t->name, "freq_factors") == 0) {
        for (float & x : v) x = 4.5f + 3.5f * rand_unit();
    } else {
        for (float & x : v) x = 2.0f * rand_unit();
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, v.data(), q.data(), 0, n / t->ne[0], t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

// A case builds its graph from inputs created in `weights` (allocated in a
// buffer marked as model weights, i.e. eligible for the GPU weight cache) and
// `inputs`, and returns the node whose output is compared.
using build_fn = std::function<ggml_tensor *(ggml_context * weights, ggml_context * inputs)>;

struct test_case {
    std::string name;
    build_fn    build;
};

struct comparison {
    size_t mismatches = 0;
    float  max_err    = 0.0f;
};

static comparison compare(const std::vector<float> & ref, const std::vector<float> & got) {
    comparison c;
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::memcmp(&ref[i], &got[i], sizeof(float)) == 0) {
            continue;
        }
        c.mismatches++;
        // scaled error, as in test-lns-backend: relative for |ref| >= 1,
        // absolute below, so values near zero do not inflate it
        const float err = std::fabs(ref[i] - got[i]) / std::fmax(std::fabs(ref[i]), 1.0f);
        c.max_err = std::isnan(err) ? INFINITY : std::fmax(c.max_err, err);
    }
    return c;
}

class graph_runner {
public:
    explicit graph_runner(const test_case & tc) {
        const ggml_init_params params = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        wctx = ggml_init(params);
        ictx = ggml_init(params);
        out  = tc.build(wctx, ictx);
        graph = ggml_new_graph(ictx);
        ggml_build_forward_expand(graph, out);

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (ggml_get_first_tensor(wctx) != nullptr) {
            wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
            ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ictx, buft);
        for (ggml_context * c : { wctx, ictx }) {
            for (ggml_tensor * t = ggml_get_first_tensor(c); t != nullptr; t = ggml_get_next_tensor(c, t)) {
                if (t->op == GGML_OP_NONE && t->view_src == nullptr) {
                    fill_input(t);
                }
            }
        }
    }

    ~graph_runner() {
        ggml_backend_buffer_free(ibuf);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        ggml_free(ictx);
        ggml_free(wctx);
    }

    // Names of the nodes `backend` would not run (it must run all of them).
    std::string unsupported(ggml_backend_t backend) const {
        std::string names;
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                names += std::string(names.empty() ? "" : ", ") + ggml_op_desc(node);
            }
        }
        return names;
    }

    std::vector<float> run(ggml_backend_t backend) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "graph_compute failed on %s\n", ggml_backend_name(backend));
            std::exit(1);
        }
        return result();
    }

    std::vector<float> run(ggml_backend_sched_t sched) {
        ggml_backend_sched_reset(sched);
        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "sched graph_compute failed\n");
            std::exit(1);
        }
        return result();
    }

    ggml_cgraph * graph = nullptr;

private:
    std::vector<float> result() const {
        std::vector<float> v((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, v.data(), 0, ggml_nbytes(out));
        return v;
    }

    ggml_context *        wctx = nullptr;
    ggml_context *        ictx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    ggml_tensor *         out  = nullptr;
};

static ggml_tensor * named(ggml_tensor * t, const char * name) {
    ggml_set_name(t, name);
    return t;
}

static std::vector<test_case> make_cases() {
    std::vector<test_case> cases;
    auto add = [&](const std::string & name, build_fn fn) { cases.push_back({ name, std::move(fn) }); };

    for (ggml_type wtype : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K }) {
        add(std::string("mul_mat weights ") + ggml_type_name(wtype), [wtype](ggml_context * w, ggml_context * in) {
            ggml_tensor * a = ggml_new_tensor_2d(w, wtype, 256, 67);
            ggml_tensor * b = ggml_new_tensor_2d(in, GGML_TYPE_F32, 256, 5); // N not a multiple of 4
            return ggml_mul_mat(in, a, b);
        });
    }
    add("mul_mat long K=1536", [](ggml_context * w, ggml_context * in) {
        return ggml_mul_mat(in, ggml_new_tensor_2d(w, GGML_TYPE_F32, 1536, 40),
                                ggml_new_tensor_2d(in, GGML_TYPE_F32, 1536, 3));
    });
    add("mul_mat GQA broadcast, F16 KV view", [](ggml_context *, ggml_context * in) {
        // K cache [head_dim, n_kv, n_head_kv] viewed with permuted strides
        ggml_tensor * kv = ggml_new_tensor_3d(in, GGML_TYPE_F16, 64, 2, 24);
        ggml_tensor * k  = ggml_permute(in, kv, 0, 2, 1, 3); // [64, 24, 2]
        ggml_tensor * q  = ggml_new_tensor_3d(in, GGML_TYPE_F32, 64, 3, 6);
        return ggml_mul_mat(in, k, q);
    });
    add("add bias broadcast", [](ggml_context *, ggml_context * in) {
        return ggml_add(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 96, 7), ggml_new_tensor_1d(in, GGML_TYPE_F32, 96));
    });
    add("mul broadcast rows", [](ggml_context *, ggml_context * in) {
        return ggml_mul(in, ggml_new_tensor_3d(in, GGML_TYPE_F32, 32, 5, 3), ggml_new_tensor_3d(in, GGML_TYPE_F32, 32, 1, 3));
    });
    add("div", [](ggml_context *, ggml_context * in) {
        return ggml_div(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 50, 4), ggml_new_tensor_2d(in, GGML_TYPE_F32, 50, 4));
    });
    add("scale", [](ggml_context *, ggml_context * in) {
        return ggml_scale(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 77, 3), 0.125f);
    });
    add("scale_bias", [](ggml_context *, ggml_context * in) {
        return ggml_scale_bias(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 77, 3), -1.5f, 0.75f);
    });
    add("silu", [](ggml_context *, ggml_context * in) { return ggml_silu(in, ggml_new_tensor_1d(in, GGML_TYPE_F32, 1000)); });
    add("gelu", [](ggml_context *, ggml_context * in) { return ggml_gelu(in, ggml_new_tensor_1d(in, GGML_TYPE_F32, 1000)); });
    add("relu", [](ggml_context *, ggml_context * in) { return ggml_relu(in, ggml_new_tensor_1d(in, GGML_TYPE_F32, 1000)); });
    add("tanh", [](ggml_context *, ggml_context * in) { return ggml_tanh(in, ggml_new_tensor_1d(in, GGML_TYPE_F32, 1000)); });
    add("swiglu split", [](ggml_context *, ggml_context * in) {
        return ggml_swiglu_split(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 48, 5), ggml_new_tensor_2d(in, GGML_TYPE_F32, 48, 5));
    });
    add("swiglu fused", [](ggml_context *, ggml_context * in) {
        return ggml_swiglu(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 96, 5));
    });
    add("swiglu fused swapped", [](ggml_context *, ggml_context * in) {
        return ggml_swiglu_swapped(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 96, 5));
    });
    add("geglu fused", [](ggml_context *, ggml_context * in) {
        return ggml_geglu(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 96, 5));
    });
    add("rms_norm 576", [](ggml_context *, ggml_context * in) {
        return ggml_rms_norm(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 576, 4), 1e-5f);
    });
    add("soft_max", [](ggml_context *, ggml_context * in) {
        return ggml_soft_max(in, ggml_new_tensor_3d(in, GGML_TYPE_F32, 300, 3, 2));
    });
    add("soft_max masked, padded mask", [](ggml_context *, ggml_context * in) {
        ggml_tensor * kq   = ggml_new_tensor_3d(in, GGML_TYPE_F32, 40, 5, 4);
        ggml_tensor * mask = named(ggml_new_tensor_2d(in, GGML_TYPE_F32, 40, 8), "mask");
        return ggml_soft_max_ext(in, kq, mask, 0.125f, 0.0f);
    });
    for (int mode : { 0, GGML_ROPE_TYPE_NEOX }) {
        add(mode ? "rope neox, partial dims" : "rope normal, partial dims", [mode](ggml_context *, ggml_context * in) {
            ggml_tensor * x   = ggml_new_tensor_3d(in, GGML_TYPE_F32, 64, 4, 6);
            ggml_tensor * pos = named(ggml_new_tensor_1d(in, GGML_TYPE_I32, 6), "pos");
            return ggml_rope_ext(in, x, pos, nullptr, 48, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        });
    }
    add("rope neox, frequency factors", [](ggml_context *, ggml_context * in) {
        ggml_tensor * x   = ggml_new_tensor_3d(in, GGML_TYPE_F32, 64, 4, 6);
        ggml_tensor * pos = named(ggml_new_tensor_1d(in, GGML_TYPE_I32, 6), "pos");
        ggml_tensor * ff  = named(ggml_new_tensor_1d(in, GGML_TYPE_F32, 32), "freq_factors");
        return ggml_rope_ext(in, x, pos, ff, 64, GGML_ROPE_TYPE_NEOX, 0, 500000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    });
    return cases;
}

// A miniature attention + FFN block. CLAMP is not a GPU op (and CONT is not an
// LNS op at all), so under the scheduler they must land on the CPU LNS / CPU
// backends: the chain [GPU LNS, CPU LNS, CPU] has to produce exactly what
// [CPU LNS, CPU] produces.
static test_case make_chain_case() {
    return { "block with CPU-LNS fallback (sched)", [](ggml_context * w, ggml_context * in) {
        ggml_tensor * x     = ggml_new_tensor_2d(in, GGML_TYPE_F32, 128, 6);
        ggml_tensor * pos   = named(ggml_new_tensor_1d(in, GGML_TYPE_I32, 6), "pos");
        ggml_tensor * mask  = named(ggml_new_tensor_2d(in, GGML_TYPE_F32, 6, 8), "mask");
        ggml_tensor * wq    = ggml_new_tensor_2d(w, GGML_TYPE_Q8_0, 128, 128);
        ggml_tensor * wup   = ggml_new_tensor_2d(w, GGML_TYPE_Q4_0, 128, 192);
        ggml_tensor * wdown = ggml_new_tensor_2d(w, GGML_TYPE_Q4_K, 256, 128);

        // attention: q = rope(Wq norm(x)), softmax(q q^T) q, per head
        ggml_tensor * h  = ggml_rms_norm(in, x, 1e-6f);
        ggml_tensor * q  = ggml_reshape_3d(in, ggml_mul_mat(in, wq, h), 32, 4, 6);
        q = ggml_rope_ext(in, q, pos, nullptr, 32, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * qp  = ggml_cont(in, ggml_permute(in, q, 0, 2, 1, 3));       // [32, 6, 4]
        ggml_tensor * qt  = ggml_cont(in, ggml_transpose(in, qp));                // [6, 32, 4]
        ggml_tensor * kq  = ggml_soft_max_ext(in, ggml_mul_mat(in, qp, qp), mask, 0.17f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(in, qt, kq);                             // [32, 6, 4]
        ggml_tensor * att = ggml_cont_2d(in, ggml_permute(in, kqv, 0, 2, 1, 3), 128, 6);

        // FFN: Wdown clamp(swiglu(Wup norm(x)))
        ggml_tensor * up  = ggml_clamp(in, ggml_swiglu(in, ggml_mul_mat(in, wup, h)), -0.5f, 0.5f); // [96, 6]
        ggml_tensor * up2 = ggml_concat(in, up, up, 0);                           // [192, 6] -> pad to 256
        ggml_tensor * ffn = ggml_mul_mat(in, wdown, ggml_pad(in, up2, 64, 0, 0, 0));

        return ggml_scale(in, ggml_add(in, ggml_add(in, att, x), ffn), 0.5f);
    } };
}

struct pair_result {
    int run     = 0;
    int failed  = 0;
};

// See the header comment for why LNSCUDA32 gets a tolerance.
static constexpr float k_lns32_tol = 1e-5f;

static bool grade(const char * label, const comparison & c, bool exact, pair_result & r) {
    const bool ok = exact ? c.mismatches == 0 : c.max_err <= k_lns32_tol;
    r.run++;
    r.failed += ok ? 0 : 1;
    if (c.mismatches == 0) {
        std::printf("  %-44s ok (bit-identical)\n", label);
    } else {
        std::printf("  %-44s %s (%zu values differ, max scaled err %.3g)\n", label, ok ? "ok" : "FAIL", c.mismatches, c.max_err);
    }
    return ok;
}

static void run_pair(const char * gpu_name, const char * cpu_name, bool exact, pair_result & r) {
    ggml_backend_t gpu = ggml_backend_init_by_name(gpu_name, nullptr);
    ggml_backend_t cpu_lns = ggml_backend_init_by_name(cpu_name, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (gpu == nullptr || cpu_lns == nullptr || cpu == nullptr) {
        std::printf("ERROR: could not initialize %s / %s / CPU\n", gpu_name, cpu_name);
        r.run++;
        r.failed++;
        return;
    }
    std::printf("\n%s vs %s (%s)\n", gpu_name, cpu_name, exact ? "bit-exact" : "max scaled err 1e-5");

    // The GPU weight cache is keyed by tensor address and lives as long as the
    // backend; in llama.cpp the backend dies before the model's buffers do.
    // Keep every case's buffers alive until the backend is freed, so no case
    // can reuse a previous case's weight address.
    std::vector<std::unique_ptr<graph_runner>> runners;
    for (const test_case & tc : make_cases()) {
        runners.push_back(std::make_unique<graph_runner>(tc));
        graph_runner & g = *runners.back();
        const std::string missing = g.unsupported(gpu) + g.unsupported(cpu_lns);
        if (!missing.empty()) {
            std::printf("  %-44s FAIL (not claimed: %s)\n", tc.name.c_str(), missing.c_str());
            r.run++;
            r.failed++;
            continue;
        }
        const std::vector<float> ref  = g.run(cpu_lns);
        const std::vector<float> cold = g.run(gpu); // converts (and caches) weights
        const std::vector<float> warm = g.run(gpu); // reuses the cached weights
        grade(tc.name.c_str(), compare(ref, cold), exact, r);
        if (compare(cold, warm).mismatches != 0) {
            std::printf("  %-44s FAIL (cached weights changed the result)\n", tc.name.c_str());
            r.failed++;
        }
    }

    {
        const test_case tc = make_chain_case();
        graph_runner g(tc);
        ggml_backend_t chain[] = { gpu, cpu_lns, cpu };
        ggml_backend_t plain[] = { cpu_lns, cpu };
        ggml_backend_sched_t s_chain = ggml_backend_sched_new(chain, nullptr, 3, GGML_DEFAULT_GRAPH_SIZE, false, true);
        ggml_backend_sched_t s_plain = ggml_backend_sched_new(plain, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        const std::vector<float> ref = g.run(s_plain);
        const std::vector<float> got = g.run(s_chain);
        int on_gpu = 0;
        for (int i = 0; i < ggml_graph_n_nodes(g.graph); i++) {
            on_gpu += ggml_backend_sched_get_tensor_backend(s_chain, ggml_graph_node(g.graph, i)) == gpu;
        }
        std::printf("  (%d of %d nodes ran on %s)\n", on_gpu, ggml_graph_n_nodes(g.graph), gpu_name);
        grade(tc.name.c_str(), compare(ref, got), exact, r);
        if (on_gpu == 0) {
            std::printf("  %-44s FAIL (nothing was scheduled on %s)\n", tc.name.c_str(), gpu_name);
            r.failed++;
        }
        ggml_backend_sched_free(s_chain);
        ggml_backend_sched_free(s_plain);
    }

    ggml_backend_free(gpu);
    ggml_backend_free(cpu_lns);
    ggml_backend_free(cpu);
}

int main() {
    // Every LNS device may claim ops; each comparison picks its backends
    // explicitly. Must be set before the first ggml call (the selection is
    // cached on first use).
    setenv("GGML_LNS_BACKEND", "LNSCUDA16,LNSCUDA32,LNS16,LNS32", 1);
    ggml_backend_load_all();

    if (ggml_backend_dev_by_name("LNSCUDA16") == nullptr) {
        std::printf("SKIP: no LNSCUDA device (no CUDA GPU, or built without -DGGML_LNSCUDA=ON)\n");
        return 77;
    }
    ggml_backend_dev_t lns16 = ggml_backend_dev_by_name("LNS16");
    if (lns16 == nullptr || std::strstr(ggml_backend_dev_description(lns16), "sum=base") == nullptr) {
        std::printf("ERROR: LNSCUDA16 parity is defined against LNS16 built with -DGGML_LNS16_SUM=base (found: %s)\n",
                    lns16 ? ggml_backend_dev_description(lns16) : "no LNS16 device");
        return 1;
    }

    pair_result r;
    run_pair("LNSCUDA16", "LNS16", /*exact=*/true,  r);
    run_pair("LNSCUDA32", "LNS32", /*exact=*/false, r);

    std::printf("\n%s: %d of %d checks passed\n", r.failed ? "FAIL" : "PASS", r.run - r.failed, r.run);
    return r.failed ? 1 : 0;
}

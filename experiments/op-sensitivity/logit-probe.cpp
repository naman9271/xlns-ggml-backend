#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct graph_log {
    FILE * f = nullptr;
    bool done = false;
};

static bool graph_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * g = static_cast<graph_log *>(user_data);
    if (ask && !g->done) {
        const long long k = t->op == GGML_OP_MUL_MAT && t->src[0] ? (long long) t->src[0]->ne[0] : 0;
        std::fprintf(g->f, "%s\t%s\t%s\t%s\t%lld\t%lld\n", t->name, ggml_op_name(t->op), ggml_op_desc(t),
                     t->src[1] ? t->src[1]->name : "", k, (long long) ggml_nelements(t));
    }
    return false;
}

static std::vector<std::string> read_prompts(const char * path) {
    std::vector<std::string> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            out.push_back(line);
        }
    }
    return out;
}

static void log_softmax(const float * x, int n, std::vector<double> & out) {
    double mx = x[0];
    for (int i = 1; i < n; i++) mx = std::max(mx, (double) x[i]);
    double s = 0;
    for (int i = 0; i < n; i++) s += std::exp((double) x[i] - mx);
    const double lse = mx + std::log(s);
    out.resize(n);
    for (int i = 0; i < n; i++) out[i] = (double) x[i] - lse;
}

static int argmax(const float * x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <prompts.txt> <out.json> [--save ref.bin] [--ref ref.bin] [--graph graph.tsv]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path = argv[3];
    const char * save_path = nullptr;
    const char * ref_path = nullptr;
    const char * graph_path = nullptr;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--save")) save_path = argv[i + 1];
        else if (!std::strcmp(argv[i], "--ref")) ref_path = argv[i + 1];
        else if (!std::strcmp(argv[i], "--graph")) graph_path = argv[i + 1];
        else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    const std::vector<std::string> prompts = read_prompts(argv[2]);
    if (prompts.empty()) { std::fprintf(stderr, "no prompts\n"); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    graph_log glog;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = cp.n_ubatch = 1024;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (graph_path) {
        glog.f = std::fopen(graph_path, "w");
        std::fprintf(glog.f, "name\top\tdesc\tsrc1\tK\tnelements\n");
        cp.cb_eval = graph_cb;
        cp.cb_eval_user_data = &glog;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "ctx failed\n"); return 3; }

    FILE * fsave = save_path ? std::fopen(save_path, "wb") : nullptr;
    FILE * fref = ref_path ? std::fopen(ref_path, "rb") : nullptr;
    if ((save_path && !fsave) || (ref_path && !fref)) { std::fprintf(stderr, "cannot open logits file\n"); return 1; }

    double kl_sum = 0, rel_num = 0, rel_den = 0, nll_run = 0, nll_ref = 0, kl_max = 0;
    long long n_pos = 0, n_top1 = 0, n_next = 0, n_flip_last = 0;
    std::vector<double> lp_run, lp_ref;
    std::vector<float> ref(n_vocab);
    std::vector<float> kl_per_prompt;

    for (size_t p = 0; p < prompts.size(); p++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<llama_token> toks(1024);
        const int n = llama_tokenize(vocab, prompts[p].c_str(), (int) prompts[p].size(), toks.data(), 1024, true, false);
        if (n <= 0) { std::fprintf(stderr, "tokenize failed\n"); return 4; }
        toks.resize(n);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) {
            batch.token[i] = toks[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        batch.n_tokens = n;
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 5; }
        glog.done = true;
        double kl_prompt = 0;
        for (int i = 0; i < n; i++) {
            const float * lg = llama_get_logits_ith(ctx, i);
            if (fsave) std::fwrite(lg, sizeof(float), n_vocab, fsave);
            if (!fref) continue;
            if (std::fread(ref.data(), sizeof(float), n_vocab, fref) != (size_t) n_vocab) {
                std::fprintf(stderr, "reference logits too short (different prompts or model?)\n");
                return 6;
            }
            log_softmax(lg, n_vocab, lp_run);
            log_softmax(ref.data(), n_vocab, lp_ref);
            double kl = 0;
            for (int v = 0; v < n_vocab; v++) {
                const double pr = std::exp(lp_ref[v]);
                if (pr > 0) kl += pr * (lp_ref[v] - lp_run[v]);
                const double d = (double) lg[v] - ref[v];
                rel_num += d * d;
                rel_den += (double) ref[v] * ref[v];
            }
            kl_sum += kl;
            kl_prompt += kl;
            kl_max = std::max(kl_max, kl);
            const int a = argmax(lg, n_vocab), b = argmax(ref.data(), n_vocab);
            n_top1 += a == b;
            if (i == n - 1) n_flip_last += a != b;
            if (i + 1 < n) {
                nll_run -= lp_run[toks[i + 1]];
                nll_ref -= lp_ref[toks[i + 1]];
                n_next++;
            }
            n_pos++;
        }
        kl_per_prompt.push_back((float) (kl_prompt / n));
        llama_batch_free(batch);
    }
    if (fsave) std::fclose(fsave);
    if (glog.f) std::fclose(glog.f);

    FILE * fo = std::fopen(out_path, "w");
    if (fref) {
        std::fclose(fref);
        std::fprintf(fo, "{\"positions\":%lld,\"kl\":%.6g,\"kl_max\":%.6g,\"top1\":%.6g,\"logit_rel\":%.6g,"
                         "\"ppl\":%.6g,\"ppl_ref\":%.6g,\"last_flips\":%lld,\"prompts\":%zu,\"kl_per_prompt\":[",
                     n_pos, kl_sum / n_pos, kl_max, (double) n_top1 / n_pos, std::sqrt(rel_num / rel_den),
                     std::exp(nll_run / n_next), std::exp(nll_ref / n_next), n_flip_last, prompts.size());
        for (size_t i = 0; i < kl_per_prompt.size(); i++) std::fprintf(fo, "%s%.6g", i ? "," : "", kl_per_prompt[i]);
        std::fprintf(fo, "]}\n");
    } else {
        std::fprintf(fo, "{\"prompts\":%zu}\n", prompts.size());
    }
    std::fclose(fo);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// rc-logits: run ONE forward pass over a prompt, dump the full final-token logit
// vector to a binary file, and print the top-K.  Links against whichever
// libllama we point it at, so the same source is built once per backend
// flavour (build-ref, build-lns-base, build-lns-sum32, ...).
//
// A single generated token is far too noisy to judge a numerics change by, so
// compare whole logit vectors instead -- rms, correlation, and the rank of the
// expected token -- with scripts/logit-compare.py.
//
// Build one binary per llama.cpp tree, e.g.
//   T=build/llama.cpp/build-lns-sum32
//   g++ -O2 -std=c++17 -I build/llama.cpp/include -I build/llama.cpp/ggml/include \
//       scripts/rc-logits.cpp -o /tmp/rc-logits-sum32 \
//       -L $T/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/$T/bin
//
// Run (greedy, one forward pass; set GGML_LNS_BACKEND for the LNS trees):
//   GGML_LNS_BACKEND=16 /tmp/rc-logits-sum32 model.gguf "The capital of France is" /tmp/sum32.bin 20
//   python3 scripts/logit-compare.py /tmp/ref.bin /tmp/sum32.bin --token 12095
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> <out.bin> [topk] [kvtype]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * prompt     = argv[2];
    const char * out_path   = argv[3];
    int topk = argc > 4 ? atoi(argv[4]) : 20;
    const char * kvtype = argc > 5 ? argv[5] : "f16";

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 512;
    cp.n_batch = 512;
    cp.n_ubatch= 512;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (strcmp(kvtype, "f32") == 0) { cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 3; }

    std::vector<llama_token> toks(512);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), 512, true, true);
    if (n < 0) { fprintf(stderr, "tokenize failed %d\n", n); return 4; }
    toks.resize(n);
    fprintf(stderr, "[rc] n_tokens=%d n_vocab=%d kv=%s\n", n, n_vocab, kvtype);
    for (int i = 0; i < n; i++) {
        char buf[256];
        int m = llama_token_to_piece(vocab, toks[i], buf, sizeof(buf), 0, true);
        fprintf(stderr, "[rc] tok[%d]=%d '%.*s'\n", i, toks[i], m>0?m:0, buf);
    }

    llama_batch batch = llama_batch_get_one(toks.data(), n);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 5; }

    float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) { fprintf(stderr, "no logits\n"); return 6; }

    FILE * f = fopen(out_path, "wb");
    fwrite(logits, sizeof(float), n_vocab, f);
    fclose(f);

    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin()+topk, idx.end(),
        [&](int a, int b){ return logits[a] > logits[b]; });

    printf("TOPK\n");
    for (int i = 0; i < topk; i++) {
        char buf[256];
        int m = llama_token_to_piece(vocab, idx[i], buf, sizeof(buf), 0, true);
        printf("%2d  id=%-8d logit=%12.6f  '%.*s'\n", i, idx[i], logits[idx[i]], m>0?m:0, buf);
    }
    printf("GAP_TOP1_TOP2 %.6f\n", logits[idx[0]] - logits[idx[1]]);
    return 0;
}

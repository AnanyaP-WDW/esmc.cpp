#include "esmc-arch.h"
#include "esmc-internal.h"
#include "esmc.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// Public C API implementations (must use C linkage for esmc.h)
extern "C" {

static uint32_t gguf_get_u32_or(const gguf_context * ctx, const char * key, uint32_t def) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        return def;
    }
    return gguf_get_val_u32(ctx, idx);
}

static float gguf_get_f32_or(const gguf_context * ctx, const char * key, float def) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        return def;
    }
    return gguf_get_val_f32(ctx, idx);
}

static ggml_tensor * esmc_get_tensor(esmc_model * model, const char * name, bool required = true) {
    struct ggml_tensor * t = ggml_get_tensor(model->ctx_weights, name);
    if (!t && required) {
        fprintf(stderr, "esmc: missing tensor %s\n", name);
    }
    return t;
}

// In-place scale of an F16/F32 weight tensor (used to fold residue_scale).
static void esmc_scale_weights_inplace(struct ggml_tensor * t, float s) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        float * d = (float *) t->data;
        for (int64_t i = 0; i < n; i++) {
            d[i] *= s;
        }
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t * d = (ggml_fp16_t *) t->data;
        for (int64_t i = 0; i < n; i++) {
            d[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(d[i]) * s);
        }
    }
}

void esmc_model_print_tensors(const esmc_model * model) {
    if (!model || !model->ctx_weights) {
        return;
    }

    const auto & p = model->hparams;
    printf("ESM-C tensors (layers=%u embd=%u ffn=%u heads=%u vocab=%u):\n",
           p.n_layer, p.n_embd, p.n_intermediate, p.n_head, p.n_vocab);

    auto print_t = [&](const char * name, struct ggml_tensor * t) {
        if (!t) {
            printf("  %-40s MISSING\n", name);
            return;
        }
        printf("  %-40s [%5lld, %5lld, %5lld, %5lld] type=%s\n",
               name,
               (long long) t->ne[0],
               (long long) t->ne[1],
               (long long) t->ne[2],
               (long long) t->ne[3],
               ggml_type_name(t->type));
    };

    print_t("token_embd.weight", model->tok_embd);
    print_t("output_norm.weight", model->output_norm);
    print_t("output.weight", model->lm_head);

    for (uint32_t i = 0; i < p.n_layer; i++) {
        char name[128];
        const esmc_layer & layer = model->layers[i];

        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", i);
        print_t(name, layer.attn_norm);
        snprintf(name, sizeof(name), "blk.%u.attn_norm.bias", i);
        print_t(name, layer.attn_norm_bias);
        snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", i);
        print_t(name, layer.q_norm);
        snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", i);
        print_t(name, layer.k_norm);
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", i);
        print_t(name, layer.wq);
        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", i);
        print_t(name, layer.wk);
        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", i);
        print_t(name, layer.wv);
        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", i);
        print_t(name, layer.wo);
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", i);
        print_t(name, layer.ffn_norm);
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", i);
        print_t(name, layer.ffn_gate);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", i);
        print_t(name, layer.ffn_up);
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", i);
        print_t(name, layer.ffn_down);
    }
}

static bool esmc_load_tensors(esmc_model * model) {
    auto & p = model->hparams;

    model->tok_embd    = esmc_get_tensor(model, "token_embd.weight");
    model->output_norm = esmc_get_tensor(model, "output_norm.weight");
    model->lm_head     = esmc_get_tensor(model, "output.weight", false);

    if (!model->tok_embd || !model->output_norm) {
        return false;
    }

    model->layers.resize(p.n_layer);
    for (uint32_t i = 0; i < p.n_layer; i++) {
        esmc_layer & layer = model->layers[i];
        char name[128];

#define GET(field, fmt)                                   \
    do {                                                  \
        snprintf(name, sizeof(name), fmt, i);             \
        layer.field = esmc_get_tensor(model, name);       \
        if (!layer.field) {                               \
            return false;                                 \
        }                                                 \
    } while (0)

        GET(attn_norm, "blk.%u.attn_norm.weight");
        GET(attn_norm_bias, "blk.%u.attn_norm.bias");
        GET(q_norm, "blk.%u.attn_q_norm.weight");
        GET(k_norm, "blk.%u.attn_k_norm.weight");
        GET(wq, "blk.%u.attn_q.weight");
        GET(wk, "blk.%u.attn_k.weight");
        GET(wv, "blk.%u.attn_v.weight");
        GET(wo, "blk.%u.attn_output.weight");
        GET(ffn_norm, "blk.%u.ffn_norm.weight");
        GET(ffn_gate, "blk.%u.ffn_gate.weight");
        GET(ffn_up, "blk.%u.ffn_up.weight");
        GET(ffn_down, "blk.%u.ffn_down.weight");
#undef GET
    }

    return true;
}

esmc_model * esmc_load_model(const char * path, esmc_model_params params) {
    auto * model = new esmc_model();

    struct gguf_init_params gguf_params = {
        .no_alloc = true,
        .ctx      = nullptr,
    };
    model->gguf_ctx = gguf_init_from_file(path, gguf_params);
    if (!model->gguf_ctx) {
        fprintf(stderr, "esmc: failed to open GGUF %s\n", path);
        delete model;
        return nullptr;
    }

    const int arch_idx = gguf_find_key(model->gguf_ctx, "general.architecture");
    if (arch_idx < 0) {
        fprintf(stderr, "esmc: missing general.architecture\n");
        esmc_free_model(model);
        return nullptr;
    }
    const char * arch = gguf_get_val_str(model->gguf_ctx, arch_idx);
    if (strcmp(arch, "esmc") != 0) {
        fprintf(stderr, "esmc: expected architecture 'esmc', got '%s'\n", arch);
        esmc_free_model(model);
        return nullptr;
    }

    gguf_context * g = model->gguf_ctx;
    esmc_hparams & p = model->hparams;
    p.n_vocab        = gguf_get_u32_or(g, "esmc.vocab_size", 33);
    p.n_ctx          = gguf_get_u32_or(g, "esmc.context_length", 2048);
    p.n_embd         = gguf_get_u32_or(g, "esmc.embedding_length", 960);
    p.n_intermediate = gguf_get_u32_or(g, "esmc.feed_forward_length", 2560);
    p.n_head         = gguf_get_u32_or(g, "esmc.attention.head_count", 15);
    p.n_head_kv      = gguf_get_u32_or(g, "esmc.attention.head_count_kv", 15);
    p.n_layer        = gguf_get_u32_or(g, "esmc.block_count", 30);
    p.norm_eps       = gguf_get_f32_or(g, "esmc.attention.layer_norm_epsilon", 1e-5f);
    p.rope_theta     = gguf_get_f32_or(g, "esmc.rope.freq_base", 10000.0f);

    if (p.n_embd % p.n_head != 0) {
        fprintf(stderr, "esmc: n_embd (%u) must be divisible by n_head (%u)\n", p.n_embd, p.n_head);
        esmc_free_model(model);
        return nullptr;
    }
    p.residue_scale = std::sqrt((float) p.n_layer / 36.0f);

    const int vocab_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (vocab_idx >= 0) {
        const int n = (int) gguf_get_arr_n(g, vocab_idx);
        model->vocab.resize(n);
        for (int i = 0; i < n; i++) {
            model->vocab[i] = gguf_get_arr_str(g, vocab_idx, i);
        }
    }
    model->bos_id  = (int) gguf_get_u32_or(g, "tokenizer.ggml.bos_token_id", 0);
    model->eos_id  = (int) gguf_get_u32_or(g, "tokenizer.ggml.eos_token_id", 2);
    model->pad_id  = (int) gguf_get_u32_or(g, "tokenizer.ggml.padding_token_id", 1);
    model->mask_id = (int) gguf_get_u32_or(g, "tokenizer.ggml.mask_token_id", 32);

    printf("ESM-C: layers=%u embd=%u ffn=%u heads=%u head_dim=%u ctx=%u vocab=%u\n",
           p.n_layer,
           p.n_embd,
           p.n_intermediate,
           p.n_head,
           p.n_embd / p.n_head,
           p.n_ctx,
           p.n_vocab);

    struct gguf_init_params load_params = {
        .no_alloc = false,
        .ctx      = &model->ctx_weights,
    };
    gguf_free(model->gguf_ctx);
    model->gguf_ctx = gguf_init_from_file(path, load_params);
    if (!model->gguf_ctx || !model->ctx_weights) {
        fprintf(stderr, "esmc: failed to load tensor data from %s\n", path);
        esmc_free_model(model);
        return nullptr;
    }

    if (!esmc_load_tensors(model)) {
        esmc_free_model(model);
        return nullptr;
    }

    if (params.use_metal) {
        model->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!model->backend && params.require_metal) {
            fprintf(stderr, "esmc: Metal requested but unavailable\n");
            esmc_free_model(model);
            return nullptr;
        }
        if (!model->backend) {
            fprintf(stderr, "esmc: Metal unavailable, falling back to CPU backend\n");
        }
    }

    if (!model->backend) {
        model->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!model->backend) {
        fprintf(stderr, "esmc: failed to init CPU backend\n");
        esmc_free_model(model);
        return nullptr;
    }
    printf("ESM-C backend: %s\n", ggml_backend_name(model->backend));

    // Create a separate CPU backend for the scheduler (required as fallback).
    if (!ggml_backend_is_cpu(model->backend)) {
        model->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!model->backend_cpu) {
            fprintf(stderr, "esmc: failed to init CPU backend for scheduler\n");
            esmc_free_model(model);
            return nullptr;
        }
    }

    // M1 — Upload weights to persistent backend buffer once at load time.
    // Create a scratch ggml context (no_alloc), duplicate weight tensors,
    // and let ggml_backend_alloc_ctx_tensors allocate them into model->buf.
    // Then copy the CPU-loaded data into the backend buffer and swap pointers.
    {
        const int64_t n_t = gguf_get_n_tensors(model->gguf_ctx);
        struct ggml_init_params wparams = {
            .mem_size = (size_t)n_t * ggml_tensor_overhead() + 1024,
            .mem_buffer = nullptr,
            .no_alloc = true,
        };
        struct ggml_context * ctx_w = ggml_init(wparams);
        if (!ctx_w) {
            fprintf(stderr, "esmc: failed to create scratch weight context\n");
            esmc_free_model(model);
            return nullptr;
        }

        // Duplicate every weight tensor into the scratch context (metadata only)
        for (int64_t i = 0; i < n_t; i++) {
            const char * name = gguf_get_tensor_name(model->gguf_ctx, i);
            struct ggml_tensor * old_t = ggml_get_tensor(model->ctx_weights, name);
            if (!old_t) continue;
            struct ggml_tensor * new_t = ggml_dup_tensor(ctx_w, old_t);
            ggml_set_name(new_t, name);
        }

        // Allocate backend buffer for all weight tensors
        model->buf = ggml_backend_alloc_ctx_tensors(ctx_w, model->backend);
        if (!model->buf) {
            fprintf(stderr, "esmc: failed to allocate persistent weight buffer\n");
            ggml_free(ctx_w);
            esmc_free_model(model);
            return nullptr;
        }

        // M-F: pre-fold residue_scale into wo / ffn_down at load time. Mathematically
        // exact (linear), and only valid for non-quantized tensors; quantized models
        // keep the explicit graph scale node.
        const bool fold_residue =
            model->layers[0].wo && !ggml_is_quantized(model->layers[0].wo->type) &&
            model->layers[0].ffn_down && !ggml_is_quantized(model->layers[0].ffn_down->type);
        const float inv_residue = 1.0f / p.residue_scale;

        // Copy data from CPU to backend buffer and update model pointers
        for (int64_t i = 0; i < n_t; i++) {
            const char * name = gguf_get_tensor_name(model->gguf_ctx, i);
            struct ggml_tensor * old_t = ggml_get_tensor(model->ctx_weights, name);
            struct ggml_tensor * new_t = ggml_get_tensor(ctx_w, name);
            if (!old_t || !new_t) continue;
            if (fold_residue &&
                (strstr(name, "attn_output.weight") || strstr(name, "ffn_down.weight"))) {
                esmc_scale_weights_inplace(old_t, inv_residue);
            }
            ggml_backend_tensor_set(new_t, old_t->data, 0, ggml_nbytes(old_t));

            // Replace every occurrence of old_t with new_t in the model struct.
            // Walk all known weight pointers to find the match.
            #define REPLACE(ptr) do { \
                if ((void*)(ptr) == (void*)old_t) { (ptr) = new_t; } \
            } while(0)

            REPLACE(model->tok_embd);
            REPLACE(model->output_norm);
            REPLACE(model->lm_head);
            for (auto & layer : model->layers) {
                REPLACE(layer.attn_norm);
                REPLACE(layer.attn_norm_bias);
                REPLACE(layer.q_norm);
                REPLACE(layer.k_norm);
                REPLACE(layer.wq);
                REPLACE(layer.wk);
                REPLACE(layer.wv);
                REPLACE(layer.wo);
                REPLACE(layer.ffn_norm);
                REPLACE(layer.ffn_gate);
                REPLACE(layer.ffn_up);
                REPLACE(layer.ffn_down);
            }
            #undef REPLACE
        }

        model->ctx_weights_backend = ctx_w;
        model->residue_folded = fold_residue;
    }
    printf("ESM-C: weight buffer allocated (%zu MB)\n",
           ggml_backend_buffer_get_size(model->buf) / (1024 * 1024));

    (void) params.use_mmap;
    return model;
}

void esmc_free_model(esmc_model * model) {
    if (!model) {
        return;
    }
    if (model->buf) {
        ggml_backend_buffer_free(model->buf);
        model->buf = nullptr;
    }
    if (model->backend) {
        ggml_backend_free(model->backend);
        model->backend = nullptr;
    }
    if (model->backend_cpu) {
        ggml_backend_free(model->backend_cpu);
        model->backend_cpu = nullptr;
    }
    if (model->ctx_weights_backend) {
        ggml_free(model->ctx_weights_backend);
        model->ctx_weights_backend = nullptr;
    }
    if (model->gguf_ctx) {
        gguf_free(model->gguf_ctx);
        model->gguf_ctx = nullptr;
    }
    model->ctx_weights = nullptr;
    delete model;
}

esmc_model_params esmc_default_model_params(void) {
    esmc_model_params p = {};
    p.use_mmap  = true;
    p.use_metal = true;
    p.require_metal = false;
    p.n_threads = 4;
    return p;
}

esmc_context * esmc_new_context(esmc_model * model) {
    if (!model) {
        return nullptr;
    }
    auto * ctx = new esmc_context{};
    ctx->model = model;
    // M-E: use performance-core count on Apple Silicon (hardware_concurrency()
    // includes efficiency cores and oversubscribes compute-bound work).
#if defined(__APPLE__)
    {
        int n_pcores = 0;
        size_t sz = sizeof(n_pcores);
        if (sysctlbyname("hw.perflevel0.physicalcpu", &n_pcores, &sz, nullptr, 0) == 0 && n_pcores > 0) {
            ctx->n_threads = n_pcores;
        } else {
            ctx->n_threads = (int) std::thread::hardware_concurrency();
        }
    }
#else
    ctx->n_threads = (int) std::thread::hardware_concurrency();
#endif
    return ctx;
}

void esmc_free_context(esmc_context * ctx) {
    if (!ctx) {
        return;
    }
    esmc_reset_compute(ctx);
    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }
    delete ctx;
}

void esmc_context_set_max_layers(esmc_context * ctx, int n_layers) {
    if (ctx && ctx->n_layers_max != n_layers) {
        ctx->n_layers_max = n_layers;
        // Graph structure changed — drop cached graphs.
        esmc_reset_compute(ctx);
    }
}

void esmc_context_set_flash_attn(esmc_context * ctx, bool enabled) {
    if (ctx && ctx->use_flash_attn != enabled) {
        ctx->use_flash_attn = enabled;
        // Invalidate cached graphs — flash vs dense produce different graphs
        esmc_reset_compute(ctx);
    }
}

void esmc_context_set_buckets(esmc_context * ctx, bool enabled) {
    if (ctx) {
        ctx->use_buckets = enabled;
        esmc_reset_compute(ctx);
    }
}

int esmc_layer0_qk_norms(
    esmc_context * ctx,
    const int32_t * tokens,
    int32_t n_tokens,
    float * q_norm_out,
    float * k_norm_out) {
    if (!ctx || !tokens || !q_norm_out || !k_norm_out || n_tokens <= 0) {
        return -1;
    }

    const int saved_max = ctx->n_layers_max;
    ctx->n_layers_max   = 1;

    struct ggml_cgraph * gf = esmc_build_graph(ctx, tokens, n_tokens, true);
    ctx->n_layers_max       = saved_max;

    if (!gf || esmc_run_graph(ctx, gf, tokens, n_tokens) != 0) {
        return -1;
    }

    struct ggml_tensor * Q = ggml_graph_get_tensor(gf, "debug_Q");
    struct ggml_tensor * K = ggml_graph_get_tensor(gf, "debug_K");
    if (!Q || !K) {
        return -1;
    }

    *q_norm_out = esmc_tensor_norm_l2(Q);
    *k_norm_out = esmc_tensor_norm_l2(K);
    return 0;
}

int esmc_n_embd(const esmc_model * model) {
    return model ? (int) model->hparams.n_embd : 0;
}

int esmc_n_vocab(const esmc_model * model) {
    return model ? (int) model->hparams.n_vocab : 0;
}

// M-B: map a token count to the smallest cached graph bucket >= it. Buckets are
// dense at short lengths (cheap graphs) and coarser at long lengths, keeping
// padding waste below ~20-25% across realistic length distributions.
int esmc_pad_to_bucket(int n_tokens) {
    static const int32_t buckets[] = {
        16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256,
        320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048,
    };
    for (int32_t b : buckets) {
        if (n_tokens <= b) {
            return b;
        }
    }
    return n_tokens;
}

int esmc_n_ctx(const esmc_model * model) {
    return model ? (int) model->hparams.n_ctx : 0;
}

const char * esmc_backend_name(const esmc_model * model) {
    if (!model || !model->backend) {
        return nullptr;
    }
    return ggml_backend_name(model->backend);
}

const char * esmc_token_to_str(const esmc_model * model, int token_id) {
    if (!model || token_id < 0 || token_id >= (int) model->vocab.size()) {
        return nullptr;
    }
    return model->vocab[token_id].c_str();
}

int esmc_save_npy(const char * path, const float * data, int n_rows, int n_cols) {
    if (!path || !data || n_rows <= 0 || n_cols <= 0) {
        return -1;
    }

    FILE * fp = fopen(path, "wb");
    if (!fp) {
        return -1;
    }

    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, sizeof(magic) - 1, fp);

    char header[256];
    const int n = snprintf(
        header,
        sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }",
        n_rows,
        n_cols);
    if (n < 0 || n >= (int) sizeof(header)) {
        fclose(fp);
        return -1;
    }

    // .npy v1: total of magic(6) + version(2) + HEADER_LEN(2) + HEADER must be
    // a multiple of 16. HEADER itself is the dict text + space padding + '\n'.
    const size_t header_len = strlen(header);
    size_t padding          = 16 - ((10 + header_len + 1) % 16);
    if (padding == 16) {
        padding = 0;
    }
    const uint16_t header_size = (uint16_t) (header_len + padding + 1);

    fputc(1, fp);
    fputc(0, fp);
    fwrite(&header_size, sizeof(header_size), 1, fp);
    fwrite(header, 1, header_len, fp);
    for (size_t i = 0; i < padding; i++) {
        fputc(' ', fp);
    }
    fputc('\n', fp);

    fwrite(data, sizeof(float), (size_t) n_rows * (size_t) n_cols, fp);
    fclose(fp);
    return 0;
}

int esmc_embed(
    esmc_context * ctx,
    const int32_t * tokens,
    int32_t n_tokens,
    float * embeddings_out) {
    if (!ctx || !tokens || !embeddings_out || n_tokens <= 0) {
        return -1;
    }

    // M-B (opt-in): pad to the smallest bucket and reuse a cached batch graph.
    if (ctx->use_buckets) {
        const int32_t max_len = esmc_pad_to_bucket(n_tokens);
        if ((int) ctx->pad_scratch.size() < max_len) {
            ctx->pad_scratch.resize(max_len);
        }
        for (int32_t i = 0; i < n_tokens; i++) {
            ctx->pad_scratch[i] = tokens[i];
        }
        const int32_t pad_id = ctx->model->pad_id;
        for (int32_t i = n_tokens; i < max_len; i++) {
            ctx->pad_scratch[i] = pad_id;
        }
        const int32_t lengths[1] = { n_tokens };
        return esmc_embed_batch(ctx, ctx->pad_scratch.data(), lengths, 1, max_len, embeddings_out);
    }

    using clk = std::chrono::steady_clock;
    const bool profile = getenv("ESMC_PROFILE") && getenv("ESMC_PROFILE")[0] == '1';
    const int n_embd = (int) ctx->model->hparams.n_embd;

    ctx->profile_alloc_us = 0;
    ctx->profile_uploads  = 0;
    const int64_t builds_before = ctx->profile_builds;
    auto t0 = clk::now();

    struct ggml_cgraph * gf = esmc_build_graph(ctx, tokens, n_tokens, false);
    if (!gf) {
        return -1;
    }
    auto t1 = clk::now();

    if (esmc_run_graph(ctx, gf, tokens, n_tokens) != 0) {
        return -1;
    }
    auto t2 = clk::now();

    struct ggml_tensor * output = ggml_graph_get_tensor(gf, "output");
    if (!output) {
        return -1;
    }

    // Output is ggml-native [n_embd, n_tokens] contiguous, bytewise identical to
    // the caller's row-major [n_tokens, n_embd] buffer.
    const size_t n_out = (size_t) n_embd * (size_t) n_tokens;
    if (output->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(output, embeddings_out, 0, n_out * sizeof(float));
    } else if (output->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(n_out);
        ggml_backend_tensor_get(output, raw.data(), 0, raw.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < raw.size(); i++) {
            embeddings_out[i] = ggml_fp16_to_fp32(raw[i]);
        }
    } else {
        return -1;
    }

    auto t3 = clk::now();

    if (profile) {
        auto build_total_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        auto compute_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto readback_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        auto alloc_us = ctx->profile_alloc_us;
        fprintf(stderr, "ESMC_PROFILE: build=%.1fms alloc=%.1fms compute=%.1fms readback=%.1fms total=%.1fms tokens=%d nodes=%d builds=%lld upload_bytes=%lld\n",
                (build_total_us - alloc_us) / 1000.0, alloc_us / 1000.0, compute_us / 1000.0,
                readback_us / 1000.0, (build_total_us + compute_us + readback_us) / 1000.0,
                n_tokens, ggml_graph_n_nodes(gf), (long long) (ctx->profile_builds - builds_before),
                (long long) ctx->profile_uploads);
    }

    return 0;
}

int esmc_embed_batch(
    esmc_context * ctx,
    const int32_t * tokens,
    const int32_t * lengths,
    int32_t n_seq,
    int32_t max_len,
    float * embeddings_out) {
    if (!ctx || !tokens || !lengths || !embeddings_out || n_seq <= 0 || max_len <= 0) {
        return -1;
    }

    using clk = std::chrono::steady_clock;
    const bool profile = getenv("ESMC_PROFILE") && getenv("ESMC_PROFILE")[0] == '1';
    const int n_embd = (int) ctx->model->hparams.n_embd;

    ctx->profile_alloc_us = 0;
    ctx->profile_uploads  = 0;
    const int64_t builds_before = ctx->profile_builds;

    // M-D: uniform-length batches need no mask (non-causal encoder).
    bool uniform = true;
    for (int s = 0; s < n_seq; s++) {
        if (lengths[s] != max_len) {
            uniform = false;
            break;
        }
    }
    const bool use_mask = !uniform;

    auto t0 = clk::now();

    struct ggml_cgraph * gf = esmc_build_graph_batch(ctx, max_len, n_seq, use_mask);
    if (!gf) {
        return -1;
    }
    auto t1 = clk::now();

    // Fill inp_tokens: [max_len, n_seq], row-major: element (t, s) = tokens[s * max_len + t]
    struct ggml_tensor * inp_t = ggml_graph_get_tensor(gf, "inp_tokens");
    if (!inp_t) return -1;
    const size_t tok_bytes = (size_t) max_len * (size_t) n_seq * sizeof(int32_t);
    ggml_backend_tensor_set(inp_t, tokens, 0, tok_bytes);
    ctx->profile_uploads += (int64_t) tok_bytes;

    // Positions depend only on max_len; upload once per backend tensor.
    struct ggml_tensor * pos_t = ggml_graph_get_tensor(gf, "pos");
    if (pos_t && pos_t != ctx->cached_pos_t) {
        if ((int) ctx->positions.size() < max_len) {
            ctx->positions.resize(max_len);
        }
        for (int i = 0; i < max_len; i++) {
            ctx->positions[i] = i;
        }
        const size_t pos_bytes = (size_t) max_len * sizeof(int32_t);
        ggml_backend_tensor_set(pos_t, ctx->positions.data(), 0, pos_bytes);
        ctx->profile_uploads += (int64_t) pos_bytes;
        ctx->cached_pos_t = pos_t;
    }

    // Mask: [max_len, max_len, 1, n_seq] (F16), rebuilt/uploaded only when the
    // length signature changes. mask[t_kv][t_q][0][s] = 0 if both < length[s] else -INF.
    struct ggml_tensor * mask_t = use_mask ? ggml_graph_get_tensor(gf, "mask") : nullptr;
    if (mask_t) {
        bool content_changed = (ctx->cached_mask_max_len != max_len ||
                                ctx->cached_mask_n_seq != n_seq ||
                                (int) ctx->cached_mask_lengths.size() != n_seq);
        if (!content_changed) {
            for (int s = 0; s < n_seq; s++) {
                if (ctx->cached_mask_lengths[s] != lengths[s]) {
                    content_changed = true;
                    break;
                }
            }
        }
        if (content_changed) {
            const size_t mask_size = (size_t) max_len * (size_t) max_len * (size_t) n_seq;
            ctx->cached_mask.assign(mask_size, ggml_fp32_to_fp16(0.0f));
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            for (int s = 0; s < n_seq; s++) {
                const int len = lengths[s];
                for (int t_kv = 0; t_kv < max_len; t_kv++) {
                    for (int t_q = 0; t_q < max_len; t_q++) {
                        if (t_kv >= len || t_q >= len) {
                            const size_t idx = (size_t) t_kv
                                             + (size_t) max_len * (size_t) t_q
                                             + (size_t) max_len * (size_t) max_len * (size_t) s;
                            ctx->cached_mask[idx] = neg_inf;
                        }
                    }
                }
            }
            ctx->cached_mask_lengths.assign(lengths, lengths + n_seq);
            ctx->cached_mask_max_len = max_len;
            ctx->cached_mask_n_seq   = n_seq;
        }
        if (content_changed || mask_t != ctx->cached_mask_t) {
            const size_t mask_bytes = (size_t) max_len * (size_t) max_len * (size_t) n_seq * sizeof(ggml_fp16_t);
            ggml_backend_tensor_set(mask_t, ctx->cached_mask.data(), 0, mask_bytes);
            ctx->profile_uploads += (int64_t) mask_bytes;
            ctx->cached_mask_t = mask_t;
        }
    }

    if (ggml_backend_is_cpu(ctx->model->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->model->backend, ctx->n_threads);
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        return -1;
    }
    auto t2 = clk::now();

    // Read back output: [n_embd, max_len, n_seq]
    struct ggml_tensor * output = ggml_graph_get_tensor(gf, "output");
    if (!output) {
        return -1;
    }

    std::vector<float> tmp((size_t) n_embd * (size_t) max_len * (size_t) n_seq);
    if (output->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(output, tmp.data(), 0, tmp.size() * sizeof(float));
    } else if (output->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw((size_t) n_embd * (size_t) max_len * (size_t) n_seq);
        ggml_backend_tensor_get(output, raw.data(), 0, raw.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < raw.size(); i++) {
            tmp[i] = ggml_fp16_to_fp32(raw[i]);
        }
    } else {
        return -1;
    }

    auto t3 = clk::now();

    // Extract per-sequence embeddings.
    // Output layout: tmp[d + n_embd*(t + max_len*s)] = element at (d, t, s).
    // Caller wants embeddings concatenated: each sequence's tokens in order,
    // each token as n_embd floats.
    size_t offset = 0;
    for (int s = 0; s < n_seq; s++) {
        const int len = lengths[s];
        for (int t = 0; t < len; t++) {
            const float * src = tmp.data() + (size_t) t * (size_t) n_embd + (size_t) max_len * (size_t) n_embd * (size_t) s;
            std::memcpy(embeddings_out + offset, src, (size_t) n_embd * sizeof(float));
            offset += n_embd;
        }
    }

    if (profile) {
        auto build_total_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        auto compute_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto readback_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        auto alloc_us = ctx->profile_alloc_us;
        double build_ms = (build_total_us - alloc_us) / 1000.0;
        double alloc_ms = alloc_us / 1000.0;
        double compute_ms = compute_us / 1000.0;
        double readback_ms = readback_us / 1000.0;
        double total_ms = (build_total_us + compute_us + readback_us) / 1000.0;
        fprintf(stderr,
                "ESMC_PROFILE: build=%.1fms alloc=%.1fms compute=%.1fms readback=%.1fms total=%.1fms "
                "batch=%d max_len=%d total_tokens=%zu use_mask=%d builds=%lld upload_bytes=%lld\n",
                build_ms, alloc_ms, compute_ms, readback_ms, total_ms,
                n_seq, max_len, offset / n_embd, (int) use_mask,
                (long long) (ctx->profile_builds - builds_before),
                (long long) ctx->profile_uploads);
    }

    return 0;
}

int esmc_embed_mean(esmc_context * /*ctx*/,
                    const int32_t * /*tokens*/,
                    int32_t /*n_tokens*/,
                    float * /*embedding_out*/) {
    return -1;
}

int esmc_score_position(esmc_context * /*ctx*/,
                        const int32_t * /*tokens*/,
                        int32_t /*n_tokens*/,
                        int32_t /*mask_position*/,
                        float * /*logits_out*/) {
    return -1;
}

} // extern "C"

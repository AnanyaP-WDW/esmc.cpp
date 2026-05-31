#include "esmc-arch.h"
#include "esmc-internal.h"
#include "esmc.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>

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

    (void) params.use_mmap;
    return model;
}

void esmc_free_model(esmc_model * model) {
    if (!model) {
        return;
    }
    if (model->backend) {
        ggml_backend_free(model->backend);
        model->backend = nullptr;
    }
    if (model->buf) {
        ggml_backend_buffer_free(model->buf);
        model->buf = nullptr;
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
    ctx->model     = model;
    ctx->n_threads = 4;
    return ctx;
}

void esmc_free_context(esmc_context * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->buf_compute) {
        ggml_backend_buffer_free(ctx->buf_compute);
        ctx->buf_compute = nullptr;
    }
    if (ctx->ctx_compute) {
        ggml_free(ctx->ctx_compute);
        ctx->ctx_compute = nullptr;
    }
    delete ctx;
}

void esmc_context_set_max_layers(esmc_context * ctx, int n_layers) {
    if (ctx) {
        ctx->n_layers_max = n_layers;
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

    const int n_embd = (int) ctx->model->hparams.n_embd;

    struct ggml_cgraph * gf = esmc_build_graph(ctx, tokens, n_tokens, false);
    if (!gf || esmc_run_graph(ctx, gf, tokens, n_tokens) != 0) {
        return -1;
    }

    struct ggml_tensor * output = ggml_graph_get_tensor(gf, "output");
    if (!output) {
        return -1;
    }

    std::vector<float> tmp((size_t) n_embd * (size_t) n_tokens);
    if (output->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(output, tmp.data(), 0, tmp.size() * sizeof(float));
    } else if (output->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw((size_t) n_embd * (size_t) n_tokens);
        ggml_backend_tensor_get(output, raw.data(), 0, raw.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < raw.size(); i++) {
            tmp[i] = ggml_fp16_to_fp32(raw[i]);
        }
    } else {
        return -1;
    }

    // Output tensor is ggml-native (ne[0]=n_embd, ne[1]=n_tokens, contiguous).
    // Memory layout: tmp[d + t*n_embd] = element (d, t). The desired output is
    // row-major [n_tokens, n_embd]: embeddings_out[t*n_embd + d] = element (d, t).
    // These coincide bytewise so a single memcpy is correct.
    std::memcpy(embeddings_out, tmp.data(), tmp.size() * sizeof(float));
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

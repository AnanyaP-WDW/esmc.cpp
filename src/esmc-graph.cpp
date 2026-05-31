#include "esmc-internal.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

ggml_tensor * esmc_import_tensor(esmc_context * ectx, const ggml_tensor * src) {
    if (!ectx || !src || !ectx->ctx_compute) {
        return nullptr;
    }

    auto it = ectx->weight_map.find(src);
    if (it != ectx->weight_map.end()) {
        return it->second;
    }

    struct ggml_context * ctx = ectx->ctx_compute;
    struct ggml_tensor * dst = ggml_dup_tensor(ctx, src);
    ggml_set_name(dst, src->name);

    ectx->weight_map[src] = dst;
    return dst;
}

bool esmc_prepare_compute(esmc_context * ectx) {
    if (ectx->ctx_compute) {
        return true;
    }

    struct ggml_init_params params = {
        .mem_size   = 512ull * 1024ull * 1024ull,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ectx->ctx_compute = ggml_init(params);
    if (!ectx->ctx_compute) {
        fprintf(stderr, "esmc: failed to create compute context\n");
        return false;
    }
    return true;
}

static bool esmc_alloc_compute(esmc_context * ectx) {
    ectx->buf_compute = ggml_backend_alloc_ctx_tensors(ectx->ctx_compute, ectx->model->backend);
    if (!ectx->buf_compute) {
        fprintf(stderr, "esmc: failed to allocate compute buffer\n");
        return false;
    }

    for (const auto & kv : ectx->weight_map) {
        const ggml_tensor * src = kv.first;
        ggml_tensor *       dst = kv.second;
        if (src->data) {
            ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
        }
    }

    return true;
}

static ggml_tensor * esmc_wt(esmc_context * ectx, const ggml_tensor * src) {
    return esmc_import_tensor(ectx, src);
}

static ggml_tensor * esmc_layernorm(
    esmc_context * ectx,
    struct ggml_context * ctx,
    ggml_tensor * x,
    const ggml_tensor * weight,
    const ggml_tensor * bias,
    float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, esmc_wt(ectx, weight));
    if (bias) {
        struct ggml_tensor * b = esmc_wt(ectx, bias);
        b                      = ggml_reshape_2d(ctx, b, (int) b->ne[0], 1);
        x                      = ggml_add(ctx, x, ggml_repeat(ctx, b, x));
    }
    return x;
}

struct ggml_cgraph * esmc_build_graph(
    esmc_context * ectx,
    const int32_t * tokens,
    int32_t n_tokens,
    bool stop_after_layer0_qk) {
    esmc_reset_compute(ectx);
    if (!esmc_prepare_compute(ectx)) {
        return nullptr;
    }

    esmc_model & model = *ectx->model;
    const esmc_hparams & p = model.hparams;

    const int n_embd   = (int) p.n_embd;
    const int n_heads  = (int) p.n_head;
    const int n_layers = ectx->n_layers_max > 0
        ? std::min(ectx->n_layers_max, (int) p.n_layer)
        : (int) p.n_layer;
    const int head_dim = n_embd / n_heads;

    struct ggml_context * ctx = ectx->ctx_compute;
    struct ggml_cgraph  * gf  = ggml_new_graph(ctx);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");

    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(pos);
    ggml_set_name(pos, "pos");

    struct ggml_tensor * cur = ggml_get_rows(ctx, esmc_wt(ectx, model.tok_embd), inp_tokens);
    cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
    ggml_set_name(cur, "embeddings");

    for (int il = 0; il < n_layers; il++) {
        const esmc_layer & layer = model.layers[il];
        struct ggml_tensor * residual = cur;

        cur = esmc_layernorm(ectx, ctx, cur, layer.attn_norm, layer.attn_norm_bias, p.norm_eps);

        struct ggml_tensor * Q = ggml_mul_mat(ctx, esmc_wt(ectx, layer.wq), cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, esmc_wt(ectx, layer.wk), cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, esmc_wt(ectx, layer.wv), cur);

        Q = esmc_layernorm(ectx, ctx, Q, layer.q_norm, nullptr, p.norm_eps);
        K = esmc_layernorm(ectx, ctx, K, layer.k_norm, nullptr, p.norm_eps);

        Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, n_tokens);
        K = ggml_reshape_3d(ctx, K, head_dim, n_heads, n_tokens);
        V = ggml_reshape_3d(ctx, V, head_dim, n_heads, n_tokens);

        // Scale Q by 1/sqrt(head_dim) BEFORE RoPE (per plan §7.1 / Appendix).
        // Equivalent to scaling K·Q^T at softmax, but lets us reuse the same
        // Q tensor for both attention and milestone 5's debug L2 check.
        Q = ggml_scale(ctx, Q, 1.0f / std::sqrt((float) head_dim));

        Q = ggml_rope_ext(
            ctx, Q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, (int) p.n_ctx, p.rope_theta,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(
            ctx, K, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, (int) p.n_ctx, p.rope_theta,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        if (stop_after_layer0_qk && il == 0) {
            ggml_set_name(Q, "debug_Q");
            ggml_set_name(K, "debug_K");
            ggml_build_forward_expand(gf, Q);
            ggml_build_forward_expand(gf, K);
            if (!esmc_alloc_compute(ectx)) {
                return nullptr;
            }
            return gf;
        }

        // ggml_permute moves old axis i to new position p_i. Inputs are
        // (head_dim, n_heads, n_tokens). After these permutes:
        //   Q, K -> (head_dim, n_tokens, n_heads)  (matmul contracts head_dim)
        //   V    -> (n_tokens, head_dim, n_heads)  (matmul contracts n_tokens)
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));

        // Q already carries the 1/sqrt(head_dim) factor; softmax scale = 1.0.
        struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
        ggml_mul_mat_set_prec(KQ, GGML_PREC_F32);
        KQ = ggml_soft_max_ext(ctx, KQ, nullptr, 1.0f, 0.0f);

        struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
        ggml_mul_mat_set_prec(KQV, GGML_PREC_F32);
        KQV = ggml_cont(ctx, ggml_permute(ctx, KQV, 0, 2, 1, 3));
        cur = ggml_cont_2d(ctx, KQV, n_embd, n_tokens);

        cur = ggml_mul_mat(ctx, esmc_wt(ectx, layer.wo), cur);
        cur = ggml_scale(ctx, cur, 1.0f / p.residue_scale);
        cur = ggml_add(ctx, cur, residual);

        residual = cur;
        cur = esmc_layernorm(ectx, ctx, cur, layer.ffn_norm, nullptr, p.norm_eps);

        struct ggml_tensor * gate = ggml_mul_mat(ctx, esmc_wt(ectx, layer.ffn_gate), cur);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, esmc_wt(ectx, layer.ffn_up), cur);
        gate = ggml_silu(ctx, gate);
        cur  = ggml_mul(ctx, gate, up);
        cur  = ggml_mul_mat(ctx, esmc_wt(ectx, layer.ffn_down), cur);
        cur  = ggml_scale(ctx, cur, 1.0f / p.residue_scale);
        cur  = ggml_add(ctx, cur, residual);
    }

    cur = esmc_layernorm(ectx, ctx, cur, model.output_norm, nullptr, p.norm_eps);
    cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
    ggml_set_name(cur, "output");

    ggml_build_forward_expand(gf, cur);

    if (!esmc_alloc_compute(ectx)) {
        return nullptr;
    }
    return gf;
}

int esmc_run_graph(
    esmc_context * ectx,
    struct ggml_cgraph * gf,
    const int32_t * tokens,
    int32_t n_tokens) {
    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (!inp_tokens) {
        return -1;
    }
    ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));

    struct ggml_tensor * pos = ggml_graph_get_tensor(gf, "pos");
    if (pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            positions[i] = i;
        }
        ggml_backend_tensor_set(pos, positions.data(), 0, n_tokens * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(ectx->model->backend, gf) != GGML_STATUS_SUCCESS) {
        return -1;
    }
    return 0;
}

void esmc_reset_compute(esmc_context * ectx) {
    if (!ectx) {
        return;
    }
    if (ectx->buf_compute) {
        ggml_backend_buffer_free(ectx->buf_compute);
        ectx->buf_compute = nullptr;
    }
    if (ectx->ctx_compute) {
        ggml_free(ectx->ctx_compute);
        ectx->ctx_compute = nullptr;
    }
    ectx->weight_map.clear();
}

float esmc_tensor_norm_l2(const ggml_tensor * t) {
    if (!t) {
        return 0.0f;
    }

    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());

    double sum = 0.0;
    if (t->type == GGML_TYPE_F32) {
        const float * x = (const float *) raw.data();
        for (int64_t i = 0; i < n; i++) {
            sum += (double) x[i] * (double) x[i];
        }
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * x = (const ggml_fp16_t *) raw.data();
        for (int64_t i = 0; i < n; i++) {
            const float v = ggml_fp16_to_fp32(x[i]);
            sum += (double) v * (double) v;
        }
    }
    return (float) std::sqrt(sum);
}

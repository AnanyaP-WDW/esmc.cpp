#include "esmc-internal.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool esmc_profile_enabled() {
    static bool enabled = getenv("ESMC_PROFILE") && getenv("ESMC_PROFILE")[0] == '1';
    return enabled;
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

    if (!ectx->sched) {
        // Scheduler requires CPU backend as the last entry (fallback for ops the
        // primary backend doesn't support).
        ggml_backend_t backends[2];
        int n_backends = 0;
        backends[n_backends++] = ectx->model->backend;
        if (ectx->model->backend_cpu) {
            backends[n_backends++] = ectx->model->backend_cpu;
        }
        ectx->sched = ggml_backend_sched_new(backends, NULL, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, false);
        if (!ectx->sched) {
            fprintf(stderr, "esmc: failed to create scheduler\n");
            ggml_free(ectx->ctx_compute);
            ectx->ctx_compute = nullptr;
            return false;
        }
    }

    return true;
}

static bool esmc_alloc_compute(esmc_context * ectx, struct ggml_cgraph * gf) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    if (!ggml_backend_sched_alloc_graph(ectx->sched, gf)) {
        fprintf(stderr, "esmc: failed to allocate graph via scheduler\n");
        return false;
    }

    if (esmc_profile_enabled()) {
        ectx->profile_alloc_us = std::chrono::duration_cast<std::chrono::microseconds>(
            clk::now() - t0).count();
    }
    return true;
}

static ggml_tensor * esmc_wt(esmc_context * /*ectx*/, const ggml_tensor * src) {
    // M1: weights are already in the persistent model->buf at load time.
    // Return the original tensor directly — no duplication needed.
    return const_cast<ggml_tensor *>(src);
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
    // Debug/partial graph: always rebuild, never cache.
    if (!stop_after_layer0_qk &&
        ectx->cached_gf && ectx->cached_n_tokens == n_tokens) {
        return ectx->cached_gf;
    }

    esmc_reset_compute(ectx);
    if (!esmc_prepare_compute(ectx)) {
        return nullptr;
    }
    ectx->profile_builds++;

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
            if (!esmc_alloc_compute(ectx, gf)) {
                return nullptr;
            }
            return gf;  // NOT cached — debug-only partial graph
        }

        if (ectx->use_flash_attn) {
            // All inputs are (head_dim, n_heads, n_tokens). Permute to
            // (head_dim, n_tokens, n_heads) — the layout flash expects, *including* V
            // (the "!! not transposed !!" comment in ggml.h).
            Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
            V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

            // Q already carries the 1/sqrt(head_dim) factor; flash scale = 1.0.
            // Mask is nullptr because ESM-C is non-causal (encoder) — all tokens attend
            // to all tokens.
            struct ggml_tensor * attn = ggml_flash_attn_ext(
                ctx, Q, K, V, nullptr, 1.0f, 0.0f, 0.0f);
            cur = ggml_cont_2d(ctx, attn, n_embd, n_tokens);
        } else {
            // Dense fallback: Q/K -> (head_dim, n_tokens, n_heads) for KQ matmul
            // (contracts head_dim), V -> (n_tokens, head_dim, n_heads) for KV matmul
            // (contracts n_tokens).
            Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
            V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));

            struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
            KQ = ggml_soft_max_ext(ctx, KQ, nullptr, 1.0f, 0.0f);

            struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
            KQV = ggml_cont(ctx, ggml_permute(ctx, KQV, 0, 2, 1, 3));
            cur = ggml_cont_2d(ctx, KQV, n_embd, n_tokens);
        }

        cur = ggml_mul_mat(ctx, esmc_wt(ectx, layer.wo), cur);
        if (!model.residue_folded) {
            cur = ggml_scale(ctx, cur, 1.0f / p.residue_scale);
        }
        cur = ggml_add(ctx, cur, residual);

        residual = cur;
        cur = esmc_layernorm(ectx, ctx, cur, layer.ffn_norm, nullptr, p.norm_eps);

        struct ggml_tensor * gate = ggml_mul_mat(ctx, esmc_wt(ectx, layer.ffn_gate), cur);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, esmc_wt(ectx, layer.ffn_up), cur);
        // Fused SwiGLU (silu(gate) * up) — one op instead of silu + mul.
        cur  = ggml_swiglu_split(ctx, gate, up);
        cur  = ggml_mul_mat(ctx, esmc_wt(ectx, layer.ffn_down), cur);
        if (!model.residue_folded) {
            cur = ggml_scale(ctx, cur, 1.0f / p.residue_scale);
        }
        cur  = ggml_add(ctx, cur, residual);
    }

    cur = esmc_layernorm(ectx, ctx, cur, model.output_norm, nullptr, p.norm_eps);
    cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
    ggml_set_name(cur, "output");

    ggml_build_forward_expand(gf, cur);

    if (!esmc_alloc_compute(ectx, gf)) {
        return nullptr;
    }
    if (!stop_after_layer0_qk) {
        ectx->cached_n_tokens = n_tokens;
        ectx->cached_gf       = gf;
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
    ectx->profile_uploads += (int64_t) n_tokens * (int64_t) sizeof(int32_t);

    struct ggml_tensor * pos = ggml_graph_get_tensor(gf, "pos");
    if (pos) {
        if ((int) ectx->positions.size() < n_tokens) {
            ectx->positions.resize(n_tokens);
        }
        for (int i = 0; i < n_tokens; i++) {
            ectx->positions[i] = i;
        }
        ggml_backend_tensor_set(pos, ectx->positions.data(), 0, n_tokens * sizeof(int32_t));
        ectx->profile_uploads += (int64_t) n_tokens * (int64_t) sizeof(int32_t);
    }

    if (ggml_backend_is_cpu(ectx->model->backend)) {
        ggml_backend_cpu_set_n_threads(ectx->model->backend, ectx->n_threads);
    }

    if (ggml_backend_sched_graph_compute(ectx->sched, gf) != GGML_STATUS_SUCCESS) {
        return -1;
    }
    return 0;
}

void esmc_reset_compute(esmc_context * ectx) {
    if (!ectx) {
        return;
    }
    if (ectx->sched) {
        ggml_backend_sched_reset(ectx->sched);
    }
    if (ectx->ctx_compute) {
        ggml_free(ectx->ctx_compute);
        ectx->ctx_compute = nullptr;
    }
    ectx->cached_n_tokens = -1;
    ectx->cached_gf       = nullptr;
    ectx->cached_max_len  = -1;
    ectx->cached_n_seq    = -1;
    ectx->cached_use_mask = false;
    ectx->cached_gf_batch = nullptr;
    // Backend input tensors are invalidated when the compute context is freed.
    ectx->cached_pos_t    = nullptr;
    ectx->cached_mask_t   = nullptr;
}

struct ggml_cgraph * esmc_build_graph_batch(
    esmc_context * ectx,
    int32_t max_len,
    int32_t n_seq,
    bool use_mask) {
    if (ectx->cached_gf_batch && ectx->cached_max_len == max_len &&
        ectx->cached_n_seq == n_seq && ectx->cached_use_mask == use_mask) {
        return ectx->cached_gf_batch;
    }

    esmc_reset_compute(ectx);
    if (!esmc_prepare_compute(ectx)) {
        return nullptr;
    }
    ectx->profile_builds++;

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

    // Input tokens: flattened [max_len * n_seq] — ggml_get_rows requires 1D indices
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_len * n_seq);
    ggml_set_input(inp_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");

    // Shared position IDs: [max_len]
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_len);
    ggml_set_input(pos);
    ggml_set_name(pos, "pos");

    // Attention mask: optional. Uniform-length batches are non-causal encoders,
    // so no mask is needed at all (and the maskless flash path is faster).
    struct ggml_tensor * mask = nullptr;
    if (use_mask) {
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, max_len, max_len, 1, n_seq);
        ggml_set_input(mask);
        ggml_set_name(mask, "mask");
    }

    // Embedding lookup: tok_embd [n_embd, n_vocab] x tokens [max_len*n_seq] -> [n_embd, max_len*n_seq]
    // Reshape to 3D: [n_embd, max_len, n_seq]
    struct ggml_tensor * cur = ggml_get_rows(ctx, esmc_wt(ectx, model.tok_embd), inp_tokens);
    cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
    cur = ggml_reshape_3d(ctx, cur, n_embd, max_len, n_seq);
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

        // Reshape to 4D: [head_dim, n_heads, max_len, n_seq]
        Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads, max_len, n_seq);
        K = ggml_reshape_4d(ctx, K, head_dim, n_heads, max_len, n_seq);
        V = ggml_reshape_4d(ctx, V, head_dim, n_heads, max_len, n_seq);

        Q = ggml_scale(ctx, Q, 1.0f / std::sqrt((float) head_dim));

        Q = ggml_rope_ext(
            ctx, Q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, (int) p.n_ctx, p.rope_theta,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(
            ctx, K, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, (int) p.n_ctx, p.rope_theta,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        if (ectx->use_flash_attn) {
            // Flash layout: [head_dim, max_len, n_heads, n_seq]
            Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
            V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

            struct ggml_tensor * attn = ggml_flash_attn_ext(
                ctx, Q, K, V, mask, 1.0f, 0.0f, 0.0f);

            // Match single-seq: cont_2d flattens heads into embedding dimension
            cur = ggml_cont_2d(ctx, attn, n_embd, max_len * n_seq);
            cur = ggml_reshape_3d(ctx, cur, n_embd, max_len, n_seq);
        } else {
            // Dense: Q/K -> [head_dim, max_len, n_heads, n_seq], V -> [n_heads, max_len, head_dim, n_seq]

            Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
            V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));

            struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
            KQ = ggml_soft_max_ext(ctx, KQ, mask, 1.0f, 0.0f);

            struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
            KQV = ggml_cont(ctx, ggml_permute(ctx, KQV, 0, 2, 1, 3));
            cur = ggml_reshape_3d(ctx, KQV, n_embd, max_len, n_seq);
        }

        cur = ggml_mul_mat(ctx, esmc_wt(ectx, layer.wo), cur);
        if (!model.residue_folded) {
            cur = ggml_scale(ctx, cur, 1.0f / p.residue_scale);
        }
        cur = ggml_add(ctx, cur, residual);

        residual = cur;
        cur = esmc_layernorm(ectx, ctx, cur, layer.ffn_norm, nullptr, p.norm_eps);

        struct ggml_tensor * gate = ggml_mul_mat(ctx, esmc_wt(ectx, layer.ffn_gate), cur);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, esmc_wt(ectx, layer.ffn_up), cur);
        // Fused SwiGLU (silu(gate) * up) — one op instead of silu + mul.
        cur  = ggml_swiglu_split(ctx, gate, up);
        cur  = ggml_mul_mat(ctx, esmc_wt(ectx, layer.ffn_down), cur);
        if (!model.residue_folded) {
            cur = ggml_scale(ctx, cur, 1.0f / p.residue_scale);
        }
        cur  = ggml_add(ctx, cur, residual);
    }

    cur = esmc_layernorm(ectx, ctx, cur, model.output_norm, nullptr, p.norm_eps);
    cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
    ggml_set_name(cur, "output");

    ggml_build_forward_expand(gf, cur);

    if (!esmc_alloc_compute(ectx, gf)) {
        return nullptr;
    }
    ectx->cached_max_len  = max_len;
    ectx->cached_n_seq    = n_seq;
    ectx->cached_use_mask = use_mask;
    ectx->cached_gf_batch = gf;
    return gf;
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

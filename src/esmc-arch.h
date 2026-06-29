#pragma once

#include "ggml.h"
#include <cstdint>
#include <string>
#include <vector>

struct esmc_hparams {
    uint32_t n_vocab        = 33;
    uint32_t n_ctx          = 2048;
    uint32_t n_embd         = 960;
    uint32_t n_intermediate = 2560;
    uint32_t n_head         = 15;
    uint32_t n_head_kv      = 15;
    uint32_t n_layer        = 30;
    float    norm_eps       = 1e-5f;
    float    rope_theta     = 10000.0f;
    float    residue_scale  = 1.0f;  // sqrt(n_layer / 36), Biohub TransformerStack
};

struct esmc_layer {
    struct ggml_tensor * attn_norm      = nullptr;
    struct ggml_tensor * attn_norm_bias = nullptr;
    struct ggml_tensor * q_norm         = nullptr;
    struct ggml_tensor * k_norm         = nullptr;
    struct ggml_tensor * wq         = nullptr;
    struct ggml_tensor * wk         = nullptr;
    struct ggml_tensor * wv         = nullptr;
    struct ggml_tensor * wo         = nullptr;
    struct ggml_tensor * ffn_norm   = nullptr;
    struct ggml_tensor * ffn_gate   = nullptr;
    struct ggml_tensor * ffn_up     = nullptr;
    struct ggml_tensor * ffn_down   = nullptr;
};

struct esmc_model {
    esmc_hparams hparams;

    struct ggml_tensor * tok_embd    = nullptr;
    struct ggml_tensor * output_norm = nullptr;
    struct ggml_tensor * lm_head     = nullptr;

    std::vector<esmc_layer> layers;

    struct ggml_context      * ctx_meta    = nullptr;
    struct ggml_context      * ctx_weights = nullptr;
    struct ggml_context      * ctx_weights_backend = nullptr; // scratch context for backend-resident weight tensors
    struct ggml_backend      * backend     = nullptr;
    struct ggml_backend      * backend_cpu = nullptr; // CPU backend (required by scheduler as fallback)
    struct ggml_backend_buffer * buf       = nullptr;
    struct gguf_context      * gguf_ctx    = nullptr;

    std::vector<std::string> vocab;
    int bos_id  = 0;
    int eos_id  = 2;
    int pad_id  = 1;
    int mask_id = 32;
};

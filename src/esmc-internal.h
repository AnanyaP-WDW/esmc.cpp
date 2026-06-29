#pragma once

#include "esmc-arch.h"

#include "ggml.h"

#include <cstdint>

struct ggml_cgraph;
struct ggml_backend_sched;

struct esmc_context {
    esmc_model * model = nullptr;

    struct ggml_context       * ctx_compute  = nullptr;
    struct ggml_backend_sched * sched        = nullptr;

    int n_threads    = 4;
    int n_layers_max = -1;

    bool use_flash_attn = true;

    // graph cache (M1: skip rebuild for repeated n_tokens)
    int32_t           cached_n_tokens = 0;
    struct ggml_cgraph * cached_gf   = nullptr;

    // graph cache for batch path (M5)
    int32_t           cached_max_len  = 0;
    int32_t           cached_n_seq    = 0;
    struct ggml_cgraph * cached_gf_batch = nullptr;

    // profiling (ESMC_PROFILE=1)
    int64_t profile_alloc_us = 0;
};

void esmc_reset_compute(esmc_context * ectx);

bool esmc_prepare_compute(esmc_context * ectx);

struct ggml_cgraph * esmc_build_graph(
    esmc_context * ectx,
    const int32_t * tokens,
    int32_t n_tokens,
    bool stop_after_layer0_qk = false);

struct ggml_cgraph * esmc_build_graph_batch(
    esmc_context * ectx,
    int32_t max_len,
    int32_t n_seq);

int esmc_run_graph(
    esmc_context * ectx,
    struct ggml_cgraph * gf,
    const int32_t * tokens,
    int32_t n_tokens);

float esmc_tensor_norm_l2(const ggml_tensor * t);

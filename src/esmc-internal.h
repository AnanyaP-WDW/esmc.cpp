#pragma once

#include "esmc-arch.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

struct ggml_cgraph;
struct ggml_backend_sched;

struct esmc_context {
    esmc_model * model = nullptr;

    struct ggml_context       * ctx_compute  = nullptr;
    struct ggml_backend_sched * sched        = nullptr;

    int n_threads    = 4;
    int n_layers_max = -1;

    bool use_flash_attn = true;
    bool use_buckets    = false;  // M-B: pad single sequences to length buckets

    // Single-slot graph caches. M-B/M-C feed length-sorted work, so successive
    // calls mostly reuse one graph; a shape change rebuilds (at most once per
    // distinct bucket). Multiple simultaneously-allocated graphs are not
    // supported by one ggml_backend_sched, hence single-slot.
    int32_t           cached_n_tokens = -1;
    struct ggml_cgraph * cached_gf   = nullptr;

    int32_t           cached_max_len  = -1;
    int32_t           cached_n_seq    = -1;
    bool              cached_use_mask = false;
    struct ggml_cgraph * cached_gf_batch = nullptr;

    // M-D: cached batch mask/positions (skip O(L^2*B) rebuild + upload).
    // Keyed by the backend tensor pointer so a graph switch (masked/maskless or
    // different bucket) forces a re-upload; cleared whenever graphs are freed.
    struct ggml_tensor *        cached_pos_t         = nullptr;
    struct ggml_tensor *        cached_mask_t        = nullptr;
    int32_t                     cached_pos_max_len  = -1;
    int32_t                     cached_mask_max_len = -1;
    int32_t                     cached_mask_n_seq   = -1;
    std::vector<int32_t>        cached_mask_lengths;
    std::vector<ggml_fp16_t>    cached_mask;

    // profiling (ESMC_PROFILE=1)
    int64_t profile_alloc_us = 0;
    int64_t profile_builds   = 0;   // graph constructions (cache misses)
    int64_t profile_uploads  = 0;   // host->device bytes uploaded per call

    // scratch buffers reused across calls (avoid per-call heap churn)
    std::vector<int32_t> positions;
    std::vector<int32_t> pad_scratch;   // M-B padded single-sequence tokens
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
    int32_t n_seq,
    bool use_mask);

int esmc_run_graph(
    esmc_context * ectx,
    struct ggml_cgraph * gf,
    const int32_t * tokens,
    int32_t n_tokens);

float esmc_tensor_norm_l2(const ggml_tensor * t);

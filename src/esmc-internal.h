#pragma once

#include "esmc-arch.h"

#include "ggml.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

struct ggml_cgraph;

struct esmc_context {
    esmc_model * model = nullptr;

    struct ggml_context       * ctx_compute  = nullptr;
    struct ggml_backend_buffer * buf_compute = nullptr;

    std::unordered_map<const ggml_tensor *, ggml_tensor *> weight_map;

    std::vector<float> embd_out;
    std::vector<uint8_t> graph_work;

    int n_threads    = 4;
    int n_layers_max = -1;
};

ggml_tensor * esmc_import_tensor(esmc_context * ectx, const ggml_tensor * src);

void esmc_reset_compute(esmc_context * ectx);

bool esmc_prepare_compute(esmc_context * ectx);

struct ggml_cgraph * esmc_build_graph(
    esmc_context * ectx,
    const int32_t * tokens,
    int32_t n_tokens,
    bool stop_after_layer0_qk = false);

int esmc_run_graph(
    esmc_context * ectx,
    struct ggml_cgraph * gf,
    const int32_t * tokens,
    int32_t n_tokens);

float esmc_tensor_norm_l2(const ggml_tensor * t);

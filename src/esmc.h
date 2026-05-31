#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

typedef struct esmc_model   esmc_model;
typedef struct esmc_context esmc_context;

typedef struct {
    bool use_mmap;
    bool use_metal;
    bool require_metal;
    int  n_threads;
} esmc_model_params;

esmc_model_params esmc_default_model_params(void);

esmc_model   * esmc_load_model(const char * path, esmc_model_params params);
void           esmc_free_model(esmc_model * model);
void           esmc_model_print_tensors(const esmc_model * model);
esmc_context * esmc_new_context(esmc_model * model);
void           esmc_free_context(esmc_context * ctx);
void           esmc_context_set_max_layers(esmc_context * ctx, int n_layers);

int esmc_layer0_qk_norms(
    esmc_context * ctx,
    const int32_t * tokens,
    int32_t n_tokens,
    float * q_norm_out,
    float * k_norm_out);

int   esmc_n_embd(const esmc_model * model);
int   esmc_n_vocab(const esmc_model * model);
int   esmc_n_ctx(const esmc_model * model);
const char * esmc_backend_name(const esmc_model * model);
const char * esmc_token_to_str(const esmc_model * model, int token_id);

int esmc_tokenize(const esmc_model * model,
                  const char * sequence,
                  int32_t * tokens_out,
                  int32_t max_tokens);

int esmc_embed(esmc_context * ctx,
               const int32_t * tokens,
               int32_t n_tokens,
               float * embeddings_out);

/** Write float32 array [n_rows, n_cols] in NumPy .npy format. */
int esmc_save_npy(const char * path, const float * data, int n_rows, int n_cols);

int esmc_embed_mean(esmc_context * ctx,
                    const int32_t * tokens,
                    int32_t n_tokens,
                    float * embedding_out);

int esmc_score_position(esmc_context * ctx,
                        const int32_t * tokens,
                        int32_t n_tokens,
                        int32_t mask_position,
                        float * logits_out);

#ifdef __cplusplus
}
#endif

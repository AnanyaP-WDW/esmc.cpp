#include "esmc.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s -m MODEL.gguf [options]\n"
            "\n"
            "Options:\n"
            "  -m, --model PATH          GGUF model file\n"
            "  -s, --sequence STR        Amino acid sequence\n"
            "  -o, --output PATH         Write per-residue embeddings (.npy)\n"
            "  --pool STR                Pooling: none (default), mean\n"
            "  --verify-load             Print tensor shapes (milestone 3)\n"
            "  --test-tokenizer          Print token IDs for -s (milestone 4)\n"
            "  --check-layer0-qk         Print layer-0 Q/K norms (milestone 5)\n"
            "  --layers N                Max transformer layers (default: all)\n"
            "  --no-metal                CPU backend only\n"
            "  --require-metal           Fail instead of falling back to CPU\n"
            "  -h, --help                Show help\n",
            prog);
}

static int run_embed(
    esmc_model * model,
    esmc_context * ctx,
    const char * sequence,
    const char * output_path,
    const char * pool) {
    std::vector<int32_t> tokens(strlen(sequence) + 2);
    const int n = esmc_tokenize(model, sequence, tokens.data(), (int32_t) tokens.size());
    if (n < 0) {
        return 1;
    }

    const int n_embd = esmc_n_embd(model);
    std::vector<float> emb((size_t) n * (size_t) n_embd);
    if (esmc_embed(ctx, tokens.data(), n, emb.data()) != 0) {
        fprintf(stderr, "embed failed\n");
        return 1;
    }

    const int n_res = n - 2;
    if (n_res <= 0) {
        fprintf(stderr, "sequence too short after special tokens\n");
        return 1;
    }

    if (strcmp(pool, "mean") == 0) {
        std::vector<float> mean((size_t) n_embd, 0.0f);
        for (int t = 1; t < n - 1; t++) {
            for (int d = 0; d < n_embd; d++) {
                mean[d] += emb[(size_t) t * (size_t) n_embd + (size_t) d];
            }
        }
        const float inv = 1.0f / (float) n_res;
        for (int d = 0; d < n_embd; d++) {
            mean[d] *= inv;
        }
        if (output_path) {
            if (esmc_save_npy(output_path, mean.data(), 1, n_embd) != 0) {
                fprintf(stderr, "failed to write %s\n", output_path);
                return 1;
            }
        }
        printf("mean-pool OK: %d dims\n", n_embd);
        return 0;
    }

    std::vector<float> residues((size_t) n_res * (size_t) n_embd);
    for (int t = 0; t < n_res; t++) {
        const int src = t + 1;
        for (int d = 0; d < n_embd; d++) {
            residues[(size_t) t * (size_t) n_embd + (size_t) d] =
                emb[(size_t) src * (size_t) n_embd + (size_t) d];
        }
    }

    if (output_path) {
        if (esmc_save_npy(output_path, residues.data(), n_res, n_embd) != 0) {
            fprintf(stderr, "failed to write %s\n", output_path);
            return 1;
        }
    }
    printf("embed OK: %d residues x %d dims\n", n_res, n_embd);
    return 0;
}

int main(int argc, char ** argv) {
    const char * model_path       = nullptr;
    const char * sequence         = nullptr;
    const char * output_path      = nullptr;
    const char * pool             = "none";
    bool         verify_load      = false;
    bool         test_tokenizer   = false;
    bool         check_layer0_qk  = false;
    bool         no_metal         = false;
    bool         require_metal    = false;
    int          max_layers       = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i >= argc) {
                return 1;
            }
            model_path = argv[i];
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sequence") == 0) {
            if (++i >= argc) {
                return 1;
            }
            sequence = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i >= argc) {
                return 1;
            }
            output_path = argv[i];
        } else if (strcmp(argv[i], "--pool") == 0) {
            if (++i >= argc) {
                return 1;
            }
            pool = argv[i];
        } else if (strcmp(argv[i], "--verify-load") == 0) {
            verify_load = true;
        } else if (strcmp(argv[i], "--test-tokenizer") == 0) {
            test_tokenizer = true;
        } else if (strcmp(argv[i], "--check-layer0-qk") == 0) {
            check_layer0_qk = true;
        } else if (strcmp(argv[i], "--layers") == 0) {
            if (++i >= argc) {
                return 1;
            }
            max_layers = atoi(argv[i]);
        } else if (strcmp(argv[i], "--no-metal") == 0) {
            no_metal = true;
        } else if (strcmp(argv[i], "--require-metal") == 0) {
            require_metal = true;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path && (verify_load || test_tokenizer || check_layer0_qk)) {
        fprintf(stderr, "missing -m MODEL.gguf\n");
        return 1;
    }

    if (verify_load) {
        esmc_model_params params = esmc_default_model_params();
        params.use_metal = !no_metal;
        params.require_metal = require_metal;
        esmc_model * model = esmc_load_model(model_path, params);
        if (!model) {
            return 1;
        }
        esmc_model_print_tensors(model);
        esmc_free_model(model);
        printf("verify-load: OK (all required tensors present)\n");
        return 0;
    }

    if (test_tokenizer || check_layer0_qk || (model_path && sequence)) {
        if (!sequence) {
            fprintf(stderr, "-s SEQUENCE is required\n");
            return 1;
        }

        esmc_model_params params = esmc_default_model_params();
        params.use_metal = !no_metal;
        params.require_metal = require_metal;
        esmc_model * model = esmc_load_model(model_path, params);
        if (!model) {
            return 1;
        }

        if (test_tokenizer) {
            std::vector<int32_t> tokens(strlen(sequence) + 2);
            const int n = esmc_tokenize(model, sequence, tokens.data(), (int32_t) tokens.size());
            if (n < 0) {
                esmc_free_model(model);
                return 1;
            }
            printf("tokens:");
            for (int i = 0; i < n; i++) {
                printf("%s%d", i ? "," : "", tokens[i]);
            }
            printf("\n");
            esmc_free_model(model);
            return 0;
        }

        esmc_context * ctx = esmc_new_context(model);
        if (!ctx) {
            esmc_free_model(model);
            return 1;
        }
        if (max_layers > 0) {
            esmc_context_set_max_layers(ctx, max_layers);
        }

        int rc = 0;
        if (check_layer0_qk) {
            std::vector<int32_t> tokens(strlen(sequence) + 2);
            const int n = esmc_tokenize(model, sequence, tokens.data(), (int32_t) tokens.size());
            float q_norm = 0.0f;
            float k_norm = 0.0f;
            if (n < 0 || esmc_layer0_qk_norms(ctx, tokens.data(), n, &q_norm, &k_norm) != 0) {
                fprintf(stderr, "layer0 Q/K check failed\n");
                rc = 1;
            } else {
                printf("q_norm:%g\n", q_norm);
                printf("k_norm:%g\n", k_norm);
            }
        } else {
            rc = run_embed(model, ctx, sequence, output_path, pool);
        }

        esmc_free_context(ctx);
        esmc_free_model(model);
        return rc;
    }

    if (model_path) {
        esmc_model_params params = esmc_default_model_params();
        params.use_metal = !no_metal;
        params.require_metal = require_metal;
        esmc_model * model = esmc_load_model(model_path, params);
        if (!model) {
            return 1;
        }
        printf("Loaded %s (embd=%d vocab=%d)\n",
               model_path,
               esmc_n_embd(model),
               esmc_n_vocab(model));
        esmc_free_model(model);
        return 0;
    }

    printf("esmc-embed: build OK (ggml %s)\n", ggml_version());
    return 0;
}

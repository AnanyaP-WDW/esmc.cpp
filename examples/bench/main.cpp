#include "esmc.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s -m MODEL.gguf -s SEQUENCE [options]\n"
            "\n"
            "Options:\n"
            "  -m, --model PATH          GGUF model file\n"
            "  -s, --sequence STR        Amino acid sequence\n"
            "  --warmup N                Warmup iterations (default: 3)\n"
            "  --iterations N            Measured iterations (default: 10)\n"
            "  --batch N                 Batch N copies of the sequence (default: 1)\n"
            "  --no-metal                CPU backend only\n"
            "  --require-metal           Fail instead of falling back to CPU\n"
            "  --no-flash                Use dense attention (disable flash)\n"
            "  -h, --help                Show help\n",
            prog);
}

static int parse_positive_int(const char * value, const char * name) {
    char * end = nullptr;
    const long parsed = strtol(value, &end, 10);
    if (!value[0] || (end && *end) || parsed <= 0 || parsed > 1000000000L) {
        fprintf(stderr, "invalid %s: %s\n", name, value);
        return -1;
    }
    return (int) parsed;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * sequence = nullptr;
    int warmup = 3;
    int iterations = 10;
    int batch_size = 1;
    bool no_metal = false;
    bool require_metal = false;
    bool no_flash = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i >= argc) { print_usage(argv[0]); return 1; }
            model_path = argv[i];
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sequence") == 0) {
            if (++i >= argc) { print_usage(argv[0]); return 1; }
            sequence = argv[i];
        } else if (strcmp(argv[i], "--warmup") == 0) {
            if (++i >= argc) { print_usage(argv[0]); return 1; }
            warmup = parse_positive_int(argv[i], "warmup");
            if (warmup < 0) return 1;
        } else if (strcmp(argv[i], "--iterations") == 0) {
            if (++i >= argc) { print_usage(argv[0]); return 1; }
            iterations = parse_positive_int(argv[i], "iterations");
            if (iterations < 0) return 1;
        } else if (strcmp(argv[i], "--batch") == 0) {
            if (++i >= argc) { print_usage(argv[0]); return 1; }
            batch_size = parse_positive_int(argv[i], "batch");
            if (batch_size < 0) return 1;
        } else if (strcmp(argv[i], "--no-metal") == 0) {
            no_metal = true;
        } else if (strcmp(argv[i], "--require-metal") == 0) {
            require_metal = true;
        } else if (strcmp(argv[i], "--no-flash") == 0) {
            no_flash = true;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !sequence) {
        print_usage(argv[0]);
        return 1;
    }

    esmc_model_params params = esmc_default_model_params();
    params.use_metal = !no_metal;
    params.require_metal = require_metal;

    esmc_model * model = esmc_load_model(model_path, params);
    if (!model) return 1;

    esmc_context * ctx = esmc_new_context(model);
    if (!ctx) { esmc_free_model(model); return 1; }
    esmc_context_set_flash_attn(ctx, !no_flash);

    std::vector<int32_t> tokens(strlen(sequence) + 2);
    const int n_tokens = esmc_tokenize(model, sequence, tokens.data(), (int32_t) tokens.size());
    if (n_tokens < 0) {
        esmc_free_context(ctx);
        esmc_free_model(model);
        return 1;
    }

    const int n_embd = esmc_n_embd(model);

    if (batch_size > 1) {
        // Batched mode
        const int max_len = n_tokens;
        std::vector<int32_t> padded((size_t) max_len * (size_t) batch_size);
        std::vector<int32_t> lengths(batch_size, n_tokens);
        for (int s = 0; s < batch_size; s++) {
            std::memcpy(&padded[(size_t) s * (size_t) max_len], tokens.data(),
                        (size_t) n_tokens * sizeof(int32_t));
        }

        size_t total_tokens = (size_t) n_tokens * (size_t) batch_size;
        std::vector<float> embeddings(total_tokens * (size_t) n_embd);

        for (int i = 0; i < warmup; i++) {
            if (esmc_embed_batch(ctx, padded.data(), lengths.data(), batch_size, max_len,
                                  embeddings.data()) != 0) {
                fprintf(stderr, "warmup batch embed failed at iteration %d\n", i + 1);
                esmc_free_context(ctx);
                esmc_free_model(model);
                return 1;
            }
        }

        printf("batch_size max_len total_tokens latency_ms seq_s\n");
        for (int i = 0; i < iterations; i++) {
            const auto start = std::chrono::steady_clock::now();
            if (esmc_embed_batch(ctx, padded.data(), lengths.data(), batch_size, max_len,
                                  embeddings.data()) != 0) {
                fprintf(stderr, "batch embed failed at iteration %d\n", i + 1);
                esmc_free_context(ctx);
                esmc_free_model(model);
                return 1;
            }
            const auto end = std::chrono::steady_clock::now();
            const std::chrono::duration<double, std::milli> elapsed = end - start;
            const double seq_s = (double) batch_size / (elapsed.count() / 1000.0);
            printf("%d %d %zu %.6f %.2f\n",
                   batch_size, max_len, total_tokens, elapsed.count(), seq_s);
        }
    } else {
        // Single-seq mode (original behavior)
        std::vector<float> embeddings((size_t) n_tokens * (size_t) n_embd);

        for (int i = 0; i < warmup; i++) {
            if (esmc_embed(ctx, tokens.data(), n_tokens, embeddings.data()) != 0) {
                fprintf(stderr, "warmup embed failed at iteration %d\n", i + 1);
                esmc_free_context(ctx);
                esmc_free_model(model);
                return 1;
            }
        }

        printf("latency_ms\n");
        for (int i = 0; i < iterations; i++) {
            const auto start = std::chrono::steady_clock::now();
            if (esmc_embed(ctx, tokens.data(), n_tokens, embeddings.data()) != 0) {
                fprintf(stderr, "embed failed at iteration %d\n", i + 1);
                esmc_free_context(ctx);
                esmc_free_model(model);
                return 1;
            }
            const auto end = std::chrono::steady_clock::now();
            const std::chrono::duration<double, std::milli> elapsed = end - start;
            printf("%.6f\n", elapsed.count());
        }
    }

    esmc_free_context(ctx);
    esmc_free_model(model);
    return 0;
}

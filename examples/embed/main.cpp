#include "esmc.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

struct FastaRecord {
    std::string name;
    std::string seq;
};

static std::vector<FastaRecord> read_fasta(const char * path) {
    std::vector<FastaRecord> out;
    std::ifstream in(path);
    if (!in) {
        return out;
    }
    std::string line, cur_name, cur_seq;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (line[0] == '>') {
            if (!cur_name.empty()) {
                out.push_back({cur_name, cur_seq});
            }
            const size_t sp = line.find_first_of(" \t");
            cur_name = line.substr(1, sp == std::string::npos ? std::string::npos : sp - 1);
            cur_seq.clear();
        } else {
            cur_seq += line;
        }
    }
    if (!cur_name.empty()) {
        out.push_back({cur_name, cur_seq});
    }
    return out;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s -m MODEL.gguf [options]\n"
            "\n"
            "Options:\n"
            "  -m, --model PATH          GGUF model file\n"
            "  -s, --sequence STR        Amino acid sequence\n"
            "  -o, --output PATH         Write per-residue embeddings (.npy)\n"
            "  --pool STR                Pooling: none (default), mean\n"
            "  --fasta PATH              Embed every sequence in a FASTA (load model once)\n"
            "  --output-dir DIR          With --fasta: write one .npy per sequence\n"
            "  --max-batch N             Cap sequences per batch (0 = length-aware, default)\n"
            "  --max-tokens N            Token budget per batch, n_seq*max_len (default: 8192)\n"
            "  --single-bucketed         With --fasta: one seq/call via bucketed esmc_embed (M-B)\n"
            "  --single-exact            With --fasta: one seq/call, exact length (M-B baseline)\n"
            "  --single-plain            With --fasta: one seq/call via esmc_embed single graph\n"
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

struct FastaEmbedOptions {
    enum Mode { BATCH, SINGLE_BUCKETED, SINGLE_EXACT, SINGLE_PLAIN };
    const char * fasta      = nullptr;
    const char * output_dir = nullptr;  // nullptr => no writes (benchmark mode)
    const char * pool       = "none";
    int          max_batch  = 0;        // 0 => length-aware schedule; >0 => fixed cap
    long         max_tokens = 8192;     // n_seq * max_len budget per batch
    Mode         mode       = BATCH;    // SINGLE_* are for M-B measurement
};

// Length-aware batch-size schedule. Measured per-residue optimum on M4 Max Metal
// F16 (README / lab_manual EXP-023): short ~16, medium ~4, long ~1. The schedule
// approximates the observed inverse relation between length and optimal batch
// (roughly a constant token budget of ~1k). Overridden by a fixed --max-batch.
static int esmc_auto_batch_size(int max_len) {
    if (max_len <= 64)  return 16;
    if (max_len <= 128) return 8;
    if (max_len <= 256) return 4;
    if (max_len <= 512) return 2;
    return 1;
}

// M-C: read a whole FASTA, load the model once, length-sort, and embed in
// token-budget batches. Outputs are written per sequence in input order.
static int run_fasta(esmc_model * model, esmc_context * ctx, const FastaEmbedOptions & opt) {
    std::vector<FastaRecord> records = read_fasta(opt.fasta);
    if (records.empty()) {
        fprintf(stderr, "no sequences read from %s\n", opt.fasta);
        return 1;
    }

    const int n_embd = esmc_n_embd(model);
    const int n      = (int) records.size();
    const bool mean_pool = strcmp(opt.pool, "mean") == 0;

    std::vector<std::vector<int32_t>> toks(n);
    std::vector<int> lens(n);
    for (int i = 0; i < n; i++) {
        toks[i].resize(records[i].seq.size() + 2);
        const int nt = esmc_tokenize(model, records[i].seq.c_str(), toks[i].data(), (int) toks[i].size());
        if (nt < 0) {
            fprintf(stderr, "tokenization failed for %s\n", records[i].name.c_str());
            return 1;
        }
        toks[i].resize(nt);
        lens[i] = nt;
    }

    // Length-sort so batches pack similar lengths (minimizes padding).
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return lens[a] < lens[b]; });

    std::vector<float> pooled;
    if (mean_pool) {
        pooled.assign((size_t) n * (size_t) n_embd, 0.0f);
    }

    // Emit one sequence's embeddings (mean-pool or per-residue write).
    auto emit = [&](int idx, const float * src, int len) -> bool {
        if (mean_pool) {
            const int n_res = len - 2;
            float * dst = pooled.data() + (size_t) idx * n_embd;
            if (n_res > 0) {
                for (int t = 1; t < len - 1; t++) {
                    const float * row = src + (size_t) t * n_embd;
                    for (int d = 0; d < n_embd; d++) {
                        dst[d] += row[d];
                    }
                }
                const float inv = 1.0f / (float) n_res;
                for (int d = 0; d < n_embd; d++) {
                    dst[d] *= inv;
                }
            }
        } else if (opt.output_dir) {
            const int n_res = len - 2;
            if (n_res <= 0) {
                return true;
            }
            std::string path = std::string(opt.output_dir) + "/" + records[idx].name + ".npy";
            std::vector<float> residues((size_t) n_res * (size_t) n_embd);
            for (int t = 0; t < n_res; t++) {
                std::memcpy(&residues[(size_t) t * n_embd],
                            src + (size_t) (t + 1) * n_embd,
                            (size_t) n_embd * sizeof(float));
            }
            if (esmc_save_npy(path.c_str(), residues.data(), n_res, n_embd) != 0) {
                fprintf(stderr, "failed to write %s\n", path.c_str());
                return false;
            }
        }
        return true;
    };

    const int32_t pad_id = 1;
    size_t total_tokens = 0;
    size_t total_padded = 0;
    int    n_batches    = 0;

    if (opt.mode != FastaEmbedOptions::BATCH) {
        // M-B measurement modes: one sequence per call, model loaded once.
        // Iterate length-sorted so bucket switches are monotone (matches the
        // realistic pipeline and bounds graph rebuilds to one per bucket).
        for (int oi = 0; oi < n; oi++) {
            const int idx = order[oi];
            const int len = lens[idx];
            std::vector<float> emb((size_t) len * (size_t) n_embd);
            int rc;
            if (opt.mode == FastaEmbedOptions::SINGLE_BUCKETED ||
                opt.mode == FastaEmbedOptions::SINGLE_PLAIN) {
                rc = esmc_embed(ctx, toks[idx].data(), len, emb.data());
            } else {
                const int32_t l1[1] = { len };
                rc = esmc_embed_batch(ctx, toks[idx].data(), l1, 1, len, emb.data());
            }
            if (rc != 0) {
                fprintf(stderr, "single embed failed at index %d\n", idx);
                return 1;
            }
            if (!emit(idx, emb.data(), len)) {
                return 1;
            }
            total_tokens += (size_t) len;
            total_padded += (size_t) len;
            n_batches++;
        }
    } else {
    int i = 0;
    while (i < n) {
        int j       = i;
        int max_len = lens[order[i]];
        while (j + 1 < n) {
            const int n_seq = (j + 1 - i) + 1;
            const int ml    = std::max(max_len, lens[order[j + 1]]);
            // Length-aware cap: the optimal batch shrinks as sequences get longer.
            const int cap = opt.max_batch > 0 ? opt.max_batch : esmc_auto_batch_size(ml);
            if (n_seq > cap) break;
            if ((long) n_seq * (long) ml > opt.max_tokens) break;
            max_len = ml;
            j++;
        }

        const int n_seq = j - i + 1;
        std::vector<int32_t> flat((size_t) max_len * (size_t) n_seq);
        std::vector<int32_t> lengths(n_seq);
        size_t batch_total = 0;
        for (int s = 0; s < n_seq; s++) {
            const int idx = order[i + s];
            lengths[s] = lens[idx];
            batch_total += (size_t) lens[idx];
            for (int t = 0; t < max_len; t++) {
                flat[(size_t) s * max_len + t] = (t < lens[idx]) ? toks[idx][t] : pad_id;
            }
        }

        std::vector<float> emb(batch_total * (size_t) n_embd);
        if (esmc_embed_batch(ctx, flat.data(), lengths.data(), n_seq, max_len, emb.data()) != 0) {
            fprintf(stderr, "batch embed failed at sequence index %d\n", i);
            return 1;
        }
        total_tokens += batch_total;
        total_padded += (size_t) max_len * (size_t) n_seq;

        size_t off = 0;
        for (int s = 0; s < n_seq; s++) {
            const int idx = order[i + s];
            const int len = lengths[s];
            const float * src = emb.data() + off;
            off += (size_t) len * (size_t) n_embd;
            if (!emit(idx, src, len)) {
                return 1;
            }
        }
        n_batches++;
        i = j + 1;
    }
    }  // end batch mode

    if (mean_pool && opt.output_dir) {
        for (int k = 0; k < n; k++) {
            std::string path = std::string(opt.output_dir) + "/" + records[k].name + ".npy";
            if (esmc_save_npy(path.c_str(), pooled.data() + (size_t) k * n_embd, 1, n_embd) != 0) {
                fprintf(stderr, "failed to write %s\n", path.c_str());
                return 1;
            }
        }
    }

    printf("fasta OK: %d sequences, %zu tokens, %d batches (%s)\n",
           n, total_tokens, n_batches, mean_pool ? "mean" : "per-residue");
    if (total_tokens > 0) {
        printf("padding waste: %.1f%% (%zu real -> %zu padded tokens)\n",
               100.0 * (double) (total_padded - total_tokens) / (double) total_tokens,
               total_tokens, total_padded);
    }
    return 0;
}

int main(int argc, char ** argv) {
    const char * model_path       = nullptr;
    const char * sequence         = nullptr;
    const char * output_path      = nullptr;
    const char * pool             = "none";
    const char * fasta_path       = nullptr;
    const char * output_dir       = nullptr;
    int          fasta_max_batch  = 0;
    long         fasta_max_tokens = 8192;
    FastaEmbedOptions::Mode fasta_mode = FastaEmbedOptions::BATCH;
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
        } else if (strcmp(argv[i], "--fasta") == 0) {
            if (++i >= argc) {
                return 1;
            }
            fasta_path = argv[i];
        } else if (strcmp(argv[i], "--output-dir") == 0) {
            if (++i >= argc) {
                return 1;
            }
            output_dir = argv[i];
        } else if (strcmp(argv[i], "--max-batch") == 0) {
            if (++i >= argc) {
                return 1;
            }
            fasta_max_batch = atoi(argv[i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0) {
            if (++i >= argc) {
                return 1;
            }
            fasta_max_tokens = atol(argv[i]);
        } else if (strcmp(argv[i], "--single-bucketed") == 0) {
            fasta_mode = FastaEmbedOptions::SINGLE_BUCKETED;
        } else if (strcmp(argv[i], "--single-exact") == 0) {
            fasta_mode = FastaEmbedOptions::SINGLE_EXACT;
        } else if (strcmp(argv[i], "--single-plain") == 0) {
            fasta_mode = FastaEmbedOptions::SINGLE_PLAIN;
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

    if (fasta_path) {
        if (!model_path) {
            fprintf(stderr, "--fasta requires -m MODEL.gguf\n");
            return 1;
        }
        esmc_model_params params = esmc_default_model_params();
        params.use_metal     = !no_metal;
        params.require_metal = require_metal;
        esmc_model * model = esmc_load_model(model_path, params);
        if (!model) {
            return 1;
        }
        esmc_context * ctx = esmc_new_context(model);
        if (!ctx) {
            esmc_free_model(model);
            return 1;
        }
        if (max_layers > 0) {
            esmc_context_set_max_layers(ctx, max_layers);
        }
        if (fasta_mode == FastaEmbedOptions::SINGLE_BUCKETED) {
            esmc_context_set_buckets(ctx, true);
        }
        FastaEmbedOptions opt;
        opt.fasta      = fasta_path;
        opt.output_dir = output_dir;
        opt.pool       = pool;
        opt.max_batch  = fasta_max_batch;
        opt.max_tokens = fasta_max_tokens > 0 ? fasta_max_tokens : 8192;
        opt.mode       = fasta_mode;
        const int rc = run_fasta(model, ctx, opt);
        esmc_free_context(ctx);
        esmc_free_model(model);
        return rc;
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

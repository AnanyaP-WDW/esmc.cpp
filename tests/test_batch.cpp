#include "esmc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool approx_equal(float a, float b, float rtol = 1e-4f, float atol = 5e-4f) {
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}

// Documented F16 noise ceiling for batched (masked) vs single (maskless) flash:
// EXP-021 measured max_diff 4.2e-4 before residue-scale folding; M-F folding
// raises the masked/unmasked gap to ~5.5e-4 (~1 ULP at |x|~1). Guard at 1e-3.
static constexpr float kMaxDiff = 1.0e-3f;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s MODEL.gguf\n", argv[0]);
        return 1;
    }

    esmc_model_params params = esmc_default_model_params();
    params.use_metal = true;
    params.require_metal = false;

    esmc_model * model = esmc_load_model(argv[1], params);
    if (!model) return 1;

    esmc_context * ctx = esmc_new_context(model);
    if (!ctx) { esmc_free_model(model); return 1; }

    const char * seqs[] = {
        "MKLLVLFVFLVAAQ",
        "MKTVRQERLKSIVRILERSKEPVSGA",
        "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    };
    const int n_seq = 3;
    const int n_embd = esmc_n_embd(model);

    // Tokenize all sequences
    std::vector<std::vector<int32_t>> all_tokens(n_seq);
    std::vector<int> lengths(n_seq);
    int max_len = 0;
    for (int s = 0; s < n_seq; s++) {
        all_tokens[s].resize(strlen(seqs[s]) + 4);
        int n = esmc_tokenize(model, seqs[s], all_tokens[s].data(), (int) all_tokens[s].size());
        if (n < 0) {
            fprintf(stderr, "tokenization failed for seq %d\n", s);
            return 1;
        }
        all_tokens[s].resize(n);
        lengths[s] = n;
        if (n > max_len) max_len = n;
    }

    // Per-sequence embeddings
    std::vector<std::vector<float>> single_embeds(n_seq);
    for (int s = 0; s < n_seq; s++) {
        single_embeds[s].resize((size_t) lengths[s] * (size_t) n_embd);
        if (esmc_embed(ctx, all_tokens[s].data(), lengths[s], single_embeds[s].data()) != 0) {
            fprintf(stderr, "single embed failed for seq %d\n", s);
            return 1;
        }
    }

    // Batched embeddings
    std::vector<int32_t> padded_tokens((size_t) max_len * (size_t) n_seq);
    std::vector<int32_t> batch_lengths(n_seq);
    for (int s = 0; s < n_seq; s++) {
        batch_lengths[s] = lengths[s];
        std::memcpy(&padded_tokens[(size_t) s * (size_t) max_len], all_tokens[s].data(),
                    (size_t) lengths[s] * sizeof(int32_t));
        // Pad with pad_id (1 = <pad> in ESM tokenizer)
        for (int t = lengths[s]; t < max_len; t++) {
            padded_tokens[(size_t) s * (size_t) max_len + (size_t) t] = 1;
        }
    }

    size_t total_tokens = 0;
    for (int s = 0; s < n_seq; s++) total_tokens += lengths[s];
    std::vector<float> batch_embeds(total_tokens * (size_t) n_embd);

    if (esmc_embed_batch(ctx, padded_tokens.data(), batch_lengths.data(), n_seq, max_len,
                          batch_embeds.data()) != 0) {
        fprintf(stderr, "batch embed failed\n");
        return 1;
    }

    // Compare
    int n_fail = 0;
    float max_diff = 0.0f;
    size_t batch_off = 0;
    for (int s = 0; s < n_seq; s++) {
        for (int t = 0; t < lengths[s]; t++) {
            for (int d = 0; d < n_embd; d++) {
                float a = single_embeds[s][(size_t) t * (size_t) n_embd + (size_t) d];
                float b = batch_embeds[batch_off + (size_t) d];
                float diff = fabsf(a - b);
                if (diff > max_diff) max_diff = diff;
                if (!approx_equal(a, b)) {
                    if (n_fail < 10) {
                        fprintf(stderr, "MISMATCH seq=%d token=%d dim=%d: single=%f batch=%f\n",
                                s, t, d, a, b);
                    }
                    n_fail++;
                }
            }
            batch_off += n_embd;
        }
    }

    if (max_diff <= kMaxDiff) {
        printf("PASS: all %zu tokens matched per-sequence results (max_diff=%g, %d dims over 5e-4)\n",
               total_tokens, max_diff, n_fail);
    } else {
        printf("FAIL: max_diff=%g exceeds limit %g (%d dims over 5e-4)\n",
               max_diff, kMaxDiff, n_fail);
    }

    int result = (n_fail > 0 || max_diff > kMaxDiff) ? 1 : 0;

    // Uniform-length (maskless) path: same sequence repeated, all lengths == max_len.
    {
        const int u_len = lengths[1];
        std::vector<int32_t> u_tokens((size_t) u_len * (size_t) n_seq);
        std::vector<int32_t> u_lengths(n_seq, u_len);
        for (int s = 0; s < n_seq; s++) {
            std::memcpy(&u_tokens[(size_t) s * u_len], all_tokens[1].data(),
                        (size_t) u_len * sizeof(int32_t));
        }
        std::vector<float> u_out((size_t) u_len * (size_t) n_seq * (size_t) n_embd);
        if (esmc_embed_batch(ctx, u_tokens.data(), u_lengths.data(), n_seq, u_len, u_out.data()) != 0) {
            fprintf(stderr, "uniform batch embed failed\n");
            result = 1;
        } else {
            float umax = 0.0f;
            for (int s = 0; s < n_seq; s++) {
                for (int t = 0; t < u_len; t++) {
                    for (int d = 0; d < n_embd; d++) {
                        const float a = single_embeds[1][(size_t) t * (size_t) n_embd + (size_t) d];
                        const float b = u_out[((size_t) s * u_len + (size_t) t) * (size_t) n_embd + (size_t) d];
                        umax = fmaxf(umax, fabsf(a - b));
                    }
                }
            }
            printf("%s uniform maskless max_diff=%g\n", umax <= kMaxDiff ? "PASS" : "FAIL", umax);
            if (umax > kMaxDiff) result = 1;
        }
    }

    esmc_free_context(ctx);
    esmc_free_model(model);
    return result;
}

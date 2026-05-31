// esmc-quantize: convert F32/F16 ESM-C GGUF to Q8_0 / Q4_K_M / Q4_K_S.
//
// Uses ggml + gguf APIs directly (no llama.cpp architecture validation), so
// custom "esmc" GGUFs are quantizable without registering the architecture in
// llama-arch.cpp.
//
// Mixing rules (mirror llama.cpp spirit, simplified):
//   - All "*norm*" tensors and biases stay F32.
//   - token_embd.weight and output.weight (if present) stay F16.
//   - Per-block weight matrices are quantized to the target type. If the row
//     length is not divisible by the k-quant block size (QK_K = 256), fall
//     back to a legacy 32-block quant.
//
//     Q8_0     : all weight matrices -> Q8_0
//     Q4_K_S   : heavy (attn_v, ffn_down) -> Q5_K (fallback Q8_0),
//                others -> Q4_K (fallback Q5_0)
//     Q4_K_M   : heavy (attn_v, ffn_down) -> Q6_K (fallback Q8_0),
//                others -> Q4_K (fallback Q5_0)
//
// The legacy fallbacks are bumped to Q5_0/Q8_0 (rather than Q4_0) because the
// ESM-C 300M embedding dim (960) is not a multiple of QK_K=256, so most rows
// take the fallback path; using a stronger legacy quant is necessary to hit
// the paper threshold of mean cosine > 0.995 vs PyTorch reference.

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// llama_ftype values written into general.file_type (matches llama.h enum)
enum esmc_file_type : uint32_t {
    ESMC_FTYPE_MOSTLY_F16    = 1,
    ESMC_FTYPE_MOSTLY_Q4_0   = 2,
    ESMC_FTYPE_MOSTLY_Q8_0   = 7,
    ESMC_FTYPE_MOSTLY_Q4_K_S = 14,
    ESMC_FTYPE_MOSTLY_Q4_K_M = 15,
};

struct esmc_quant_scheme {
    const char *    name;
    esmc_file_type  file_type;
    ggml_type       legacy_fallback;   // used when n_per_row % 256 != 0 for k-quants
    ggml_type       default_type;      // base k-quant for most weights
    ggml_type       heavy_type;        // optional bumped type for attn_v / ffn_down
    ggml_type       heavy_fallback;    // legacy fallback when heavy is unavailable
};

static const esmc_quant_scheme SCHEMES[] = {
    { "Q8_0",   ESMC_FTYPE_MOSTLY_Q8_0,   GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 },
    { "Q4_K_S", ESMC_FTYPE_MOSTLY_Q4_K_S, GGML_TYPE_Q5_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0 },
    { "Q4_K_M", ESMC_FTYPE_MOSTLY_Q4_K_M, GGML_TYPE_Q5_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 },
};

static const esmc_quant_scheme * find_scheme(const std::string & name) {
    for (const auto & s : SCHEMES) {
        if (name == s.name) {
            return &s;
        }
    }
    return nullptr;
}

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool is_keep_full_precision(const std::string & name) {
    // Norms and biases stay F32 to preserve numerical stability.
    if (name.find("norm") != std::string::npos) {
        return true;
    }
    if (ends_with(name, ".bias")) {
        return true;
    }
    return false;
}

static bool is_special_high_precision(const std::string & name) {
    // Embedding and output projection are kept at F16 even when quantizing.
    return name == "token_embd.weight" || name == "output.weight";
}

static bool is_heavy_tensor(const std::string & name) {
    return ends_with(name, ".attn_v.weight") ||
           ends_with(name, ".ffn_down.weight");
}

static ggml_type pick_target_type(
    const std::string &        name,
    enum ggml_type             src_type,
    int64_t                    n_per_row,
    const esmc_quant_scheme &  scheme)
{
    if (is_keep_full_precision(name)) {
        return GGML_TYPE_F32;
    }
    if (is_special_high_precision(name)) {
        return GGML_TYPE_F16;
    }

    const bool k_quant_ok = (n_per_row % 256) == 0;
    if (is_heavy_tensor(name)) {
        return k_quant_ok ? scheme.heavy_type : scheme.heavy_fallback;
    }

    if (scheme.default_type == GGML_TYPE_Q8_0) {
        return GGML_TYPE_Q8_0;
    }
    return k_quant_ok ? scheme.default_type : scheme.legacy_fallback;

    (void) src_type; // currently unused; reserved for future per-source decisions
}

static void f16_to_f32(const ggml_fp16_t * src, float * dst, int64_t n) {
    ggml_fp16_to_fp32_row(src, dst, n);
}

static void f32_to_f16(const float * src, ggml_fp16_t * dst, int64_t n) {
    ggml_fp32_to_fp16_row(src, dst, n);
}

static void dequantize_to_f32(
    enum ggml_type      src_type,
    const void *        src,
    std::vector<float> & dst_f32,
    int64_t              n_elements)
{
    dst_f32.resize((size_t) n_elements);
    if (src_type == GGML_TYPE_F32) {
        std::memcpy(dst_f32.data(), src, n_elements * sizeof(float));
        return;
    }
    if (src_type == GGML_TYPE_F16) {
        f16_to_f32((const ggml_fp16_t *) src, dst_f32.data(), n_elements);
        return;
    }
    fprintf(stderr, "esmc-quantize: cannot dequantize source type %s (must be F32 or F16)\n",
            ggml_type_name(src_type));
    std::exit(1);
}

struct quant_stats {
    int64_t kept_f32  = 0;
    int64_t kept_f16  = 0;
    int64_t quantized = 0;
    size_t  bytes_in  = 0;
    size_t  bytes_out = 0;
};

static int quantize_gguf(
    const std::string &       fname_in,
    const std::string &       fname_out,
    const esmc_quant_scheme & scheme)
{
    struct ggml_context * ctx_in = nullptr;
    struct gguf_init_params gp = { /*no_alloc=*/false, /*ctx=*/&ctx_in };
    struct gguf_context * src = gguf_init_from_file(fname_in.c_str(), gp);
    if (!src || !ctx_in) {
        fprintf(stderr, "esmc-quantize: failed to open %s\n", fname_in.c_str());
        return 1;
    }

    const int arch_idx = gguf_find_key(src, "general.architecture");
    if (arch_idx < 0 || std::strcmp(gguf_get_val_str(src, arch_idx), "esmc") != 0) {
        fprintf(stderr, "esmc-quantize: input is not an ESM-C GGUF\n");
        gguf_free(src);
        return 1;
    }

    struct gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    gguf_set_val_u32(dst, "general.file_type", scheme.file_type);
    gguf_set_val_u32(dst, "general.quantization_version", GGML_QNT_VERSION);

    const int64_t n_tensors = gguf_get_n_tensors(src);
    printf("esmc-quantize: %s -> %s  (scheme=%s, tensors=%lld)\n",
           fname_in.c_str(), fname_out.c_str(), scheme.name, (long long) n_tensors);

    std::vector<std::vector<uint8_t>> dst_blobs;
    dst_blobs.reserve((size_t) n_tensors);

    quant_stats stats;
    std::vector<float> scratch_f32;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(src, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_in, name);
        if (!t) {
            fprintf(stderr, "esmc-quantize: tensor %s missing in context\n", name);
            gguf_free(src);
            gguf_free(dst);
            return 1;
        }

        const enum ggml_type src_type = t->type;
        const int64_t n_per_row       = t->ne[0];
        const int64_t nrows           = ggml_nrows(t);
        const int64_t n_elements      = n_per_row * nrows;
        const size_t  bytes_in        = ggml_nbytes(t);

        const enum ggml_type dst_type = pick_target_type(
            name, src_type, n_per_row, scheme);

        stats.bytes_in += bytes_in;

        std::vector<uint8_t> dst_data;

        if (dst_type == src_type) {
            dst_data.assign((const uint8_t *) t->data,
                            (const uint8_t *) t->data + bytes_in);
        } else if (dst_type == GGML_TYPE_F32 && src_type == GGML_TYPE_F16) {
            dst_data.resize((size_t) n_elements * sizeof(float));
            f16_to_f32((const ggml_fp16_t *) t->data,
                       (float *) dst_data.data(),
                       n_elements);
        } else if (dst_type == GGML_TYPE_F16 && src_type == GGML_TYPE_F32) {
            dst_data.resize((size_t) n_elements * sizeof(ggml_fp16_t));
            f32_to_f16((const float *) t->data,
                       (ggml_fp16_t *) dst_data.data(),
                       n_elements);
        } else {
            dequantize_to_f32(src_type, t->data, scratch_f32, n_elements);

            if (n_per_row % ggml_blck_size(dst_type) != 0) {
                fprintf(stderr,
                        "esmc-quantize: tensor %s n_per_row=%lld not divisible by block size %lld of %s\n",
                        name,
                        (long long) n_per_row,
                        (long long) ggml_blck_size(dst_type),
                        ggml_type_name(dst_type));
                gguf_free(src);
                gguf_free(dst);
                return 1;
            }

            const size_t row_bytes = ggml_row_size(dst_type, n_per_row);
            dst_data.resize(row_bytes * (size_t) nrows);

            const size_t written = ggml_quantize_chunk(
                dst_type,
                scratch_f32.data(),
                dst_data.data(),
                /*start=*/   0,
                /*nrows=*/   nrows,
                /*n_per_row=*/n_per_row,
                /*imatrix=*/ nullptr);

            if (written != dst_data.size()) {
                fprintf(stderr,
                        "esmc-quantize: %s expected %zu bytes, ggml_quantize_chunk wrote %zu\n",
                        name, dst_data.size(), written);
                gguf_free(src);
                gguf_free(dst);
                return 1;
            }
        }

        stats.bytes_out += dst_data.size();

        // Construct a metadata-only tensor on the destination side so we can
        // hand it to gguf_add_tensor, then patch the data via gguf_set_tensor_data.
        struct ggml_init_params meta_params = {
            /*.mem_size   =*/ ggml_tensor_overhead() + 128,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * meta_ctx = ggml_init(meta_params);
        struct ggml_tensor * meta = ggml_new_tensor(
            meta_ctx, dst_type, GGML_MAX_DIMS, t->ne);
        ggml_set_name(meta, name);

        gguf_add_tensor(dst, meta);

        dst_blobs.emplace_back(std::move(dst_data));
        gguf_set_tensor_data(dst, name, dst_blobs.back().data());

        ggml_free(meta_ctx);

        if (dst_type == GGML_TYPE_F32) {
            stats.kept_f32++;
        } else if (dst_type == GGML_TYPE_F16) {
            stats.kept_f16++;
        } else {
            stats.quantized++;
        }

        printf("  %-40s %-7s -> %-7s  rows=%6lld n_per_row=%6lld  %8zu -> %8zu bytes\n",
               name,
               ggml_type_name(src_type),
               ggml_type_name(dst_type),
               (long long) nrows,
               (long long) n_per_row,
               bytes_in,
               dst_blobs.back().size());
    }

    if (!gguf_write_to_file(dst, fname_out.c_str(), /*only_meta=*/false)) {
        fprintf(stderr, "esmc-quantize: failed to write %s\n", fname_out.c_str());
        gguf_free(src);
        gguf_free(dst);
        ggml_free(ctx_in);
        return 1;
    }

    printf("esmc-quantize: wrote %s\n", fname_out.c_str());
    printf("  tensors: %lld kept F32 + %lld kept F16 + %lld quantized\n",
           (long long) stats.kept_f32,
           (long long) stats.kept_f16,
           (long long) stats.quantized);
    printf("  size:    %.2f MiB -> %.2f MiB (ratio %.3f)\n",
           (double) stats.bytes_in  / (1024.0 * 1024.0),
           (double) stats.bytes_out / (1024.0 * 1024.0),
           stats.bytes_in ? (double) stats.bytes_out / (double) stats.bytes_in : 0.0);

    gguf_free(src);
    gguf_free(dst);
    ggml_free(ctx_in);
    ggml_quantize_free();
    return 0;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s INPUT.gguf OUTPUT.gguf TYPE\n"
            "\n"
            "TYPE:\n"
            "  Q8_0     8-bit, block 32   (mean cos > 0.999 expected)\n"
            "  Q4_K_S   k-quant small mix (mean cos > 0.995 expected)\n"
            "  Q4_K_M   k-quant medium mix (mean cos > 0.995 expected)\n",
            prog);
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string fname_in  = argv[1];
    const std::string fname_out = argv[2];
    const std::string type_name = argv[3];

    const esmc_quant_scheme * scheme = find_scheme(type_name);
    if (!scheme) {
        fprintf(stderr, "esmc-quantize: unknown TYPE '%s'\n", type_name.c_str());
        print_usage(argv[0]);
        return 1;
    }

    return quantize_gguf(fname_in, fname_out, *scheme);
}

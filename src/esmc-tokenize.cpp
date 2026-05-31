#include "esmc.h"
#include "esmc-arch.h"

#include <cctype>
#include <cstring>
#include <unordered_map>

// Fallback token alphabet when the model has no embedded vocab.
// MUST match tools/convert_esmc_to_gguf.py and esmc-300m/tokenizer.json.
static const char * ESMC_TOKENS[] = {
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",
    "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
    "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>",
};

static std::unordered_map<std::string, int32_t> esmc_build_token_map(const esmc_model * model) {
    std::unordered_map<std::string, int32_t> m;
    if (model && !model->vocab.empty()) {
        for (size_t i = 0; i < model->vocab.size(); i++) {
            m[model->vocab[i]] = (int32_t) i;
        }
        return m;
    }
    for (size_t i = 0; i < sizeof(ESMC_TOKENS) / sizeof(ESMC_TOKENS[0]); i++) {
        m[ESMC_TOKENS[i]] = (int32_t) i;
    }
    return m;
}

static std::unordered_map<char, int32_t> esmc_build_aa_map(const esmc_model * model) {
    std::unordered_map<char, int32_t> m;
    const auto token_map = esmc_build_token_map(model);
    for (const auto & kv : token_map) {
        if (kv.first.size() == 1) {
            m[kv.first[0]] = kv.second;
        }
    }
    return m;
}

extern "C" int esmc_tokenize(const esmc_model * model,
                             const char * sequence,
                             int32_t * tokens_out,
                             int32_t max_tokens) {
    if (!model || !sequence || !tokens_out) {
        return -1;
    }

    static thread_local std::unordered_map<char, int32_t> aa_cache;
    static thread_local const esmc_model * cached_model = nullptr;
    if (cached_model != model) {
        aa_cache      = esmc_build_aa_map(model);
        cached_model  = model;
    }

    const int unk_id = model->vocab.empty() ? 3 : (int) esmc_build_token_map(model)["<unk>"];

    const int seq_len = (int) std::strlen(sequence);
    if (seq_len + 2 > max_tokens) {
        return -1;
    }

    int pos = 0;
    tokens_out[pos++] = model->bos_id;

    for (int i = 0; i < seq_len; i++) {
        const char c = (char) std::toupper((unsigned char) sequence[i]);
        const auto it = aa_cache.find(c);
        tokens_out[pos++] = (it == aa_cache.end()) ? unk_id : it->second;
    }

    tokens_out[pos++] = model->eos_id;
    return pos;
}

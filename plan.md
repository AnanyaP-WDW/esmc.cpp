# `esmc.cpp`: A Complete Engineering Plan for Porting ESM Cambrian to llama.cpp

## Scope and Goal

This document is a self-contained engineering specification for porting ESM Cambrian
(ESM-C: `esmc-300m`, `esmc-600m`, `esmc-6b`) to a zero-dependency, Metal-accelerated
C/C++ inference runtime in the style of llama.cpp — using the ggml/llama.cpp stack as
the compute and I/O substrate. The output is a binary called `esmc-embed` and a library
`libesmc` usable from C.

A developer who has not been involved in this project should be able to read this document
and build the entire thing from scratch.

---

## Part 1: Architecture Specification

Before writing a single line of C++, the engineer must understand exactly what ESM-C is
and how it differs from ESM2, which is the prior art being replaced.

### 1.1 What ESM-C Is

ESM-C (ESM Cambrian) is a protein language model family (300M, 600M, 6B parameters)
trained with **masked language modeling** on 2.56 billion clustered protein sequences
from UniRef, MGnify, and JGI. It is an **encoder-only** model: a single forward pass
produces per-residue embeddings, no autoregressive loop, no KV-cache growth.

Three open model variants:

| Model          | Layers | Width (d_model) | Heads | Context | Params |
|----------------|--------|-----------------|-------|---------|--------|
| esmc-300m      | 30     | 960             | 15    | 2048    | ~300M  |
| esmc-600m      | 36     | 1152            | 18    | 2048    | ~600M  |
| esmc-6b        | 80     | 2560            | 40    | 2048    | ~6B    |

### 1.2 Architecture Details (from official blog + source inspection)

These are the ground-truth facts that determine every implementation decision:

**Transformer block:**
- Pre-LN (LayerNorm before attention and before FFN, not after)
- Rotary Position Embeddings (RoPE) applied to Q and K
- **SwiGLU activation** in the FFN — this is the key difference from ESM2 which used GELU
- **No biases** anywhere: not in linear layers, not in LayerNorm
- MHA (standard multi-head attention, not GQA/MQA — head count divides evenly into d_model)
- **No token_dropout** at inference time (ESM2 had this; ESM-C does not)
- **No `emb_layer_norm_before` / `emb_layer_norm_after`** wrapping the whole encoder
  (ESM2 had these; ESM-C does not)

**Positional encoding:**
- RoPE only, no learned absolute position embeddings
- `rope_theta` = 10,000.0 (standard)

**SwiGLU FFN detail:**
Unlike the GELU FFN in ESM2 (`fc2(GELU(fc1(x)))`), SwiGLU uses a gated architecture:

```
FFN(x) = W_down( SiLU(W_gate(x)) ⊙ W_up(x) )
```

This means three weight matrices per layer instead of two. The intermediate dimension
for SwiGLU is typically `int(8/3 * d_model)` rounded to a multiple of 64, but
**ESM-C uses exactly `8/3 * d_model` rounded**, which you must verify from the
actual weight shapes when loading:
- 300M: d_model=960  → ffn_intermediate ≈ 2560  (check actual tensor shape)
- 600M: d_model=1152 → ffn_intermediate ≈ 3072  (check actual tensor shape)
- 6B:   d_model=2560 → ffn_intermediate ≈ 6827  (check actual tensor shape)

**Vocabulary and tokenizer:**
ESM-C uses the same 33-token amino acid alphabet as ESM2:
```
Index : Token
0     : <cls>     (BOS — prepended to every sequence)
1     : <pad>     (padding)
2     : <eos>     (EOS — appended to every sequence)
3     : <unk>
4–23  : L A G V S E R T I D P K N Q F Y M H W C  (standard amino acids)
24    : X   (unknown amino acid)
25    : B
26    : U
27    : Z
28    : O
29    : .
30    : -
31    : <null_1>
32    : <mask>
```

**The exact forward pass pseudocode:**

```python
def esmc_forward(input_ids, attention_mask=None):
    # 1. Token embedding (no positional embedding table — RoPE handles positions)
    x = embed_tokens(input_ids)           # [B, L, d_model]

    # 2. NO pre-encoder LayerNorm (unlike ESM2!)

    # 3. Transformer layers
    for layer in layers:
        # --- Self-attention sub-block ---
        residual = x
        x = layer_norm_1(x)               # pre-norm, NO bias

        Q = x @ W_q.T                     # [B, L, d_model], no bias
        K = x @ W_k.T
        V = x @ W_v.T

        # Reshape to multi-head: [B, heads, L, head_dim]
        Q = reshape(Q, heads, head_dim)
        K = reshape(K, heads, head_dim)
        V = reshape(V, heads, head_dim)

        # Scale Q (before RoPE, same quirk as ESM2)
        Q = Q * (1.0 / sqrt(head_dim))

        # Apply RoPE to Q and K
        Q = rope(Q, positions)
        K = rope(K, positions)

        # Bidirectional attention (no causal mask — full O(n²))
        scores = Q @ K.transpose(-1, -2)  # [B, heads, L, L]
        if attention_mask is not None:
            scores = scores + padding_mask
        attn_weights = softmax(scores, dim=-1)
        attn_out = attn_weights @ V        # [B, heads, L, head_dim]

        # Merge heads and project
        attn_out = merge_heads(attn_out)   # [B, L, d_model]
        attn_out = attn_out @ W_o.T        # no bias!

        x = residual + attn_out

        # --- FFN sub-block (SwiGLU) ---
        residual = x
        x = layer_norm_2(x)               # pre-norm, NO bias

        gate = x @ W_gate.T               # [B, L, ffn_dim]
        up   = x @ W_up.T                 # [B, L, ffn_dim]
        x    = silu(gate) * up            # element-wise gate
        x    = x @ W_down.T              # [B, L, d_model]

        x = residual + x

    # 4. Final LayerNorm (single, after all layers)
    x = final_norm(x)                    # NO bias

    # 5. Output: per-residue hidden states [B, L, d_model]
    # For sequence-level: mean-pool over non-pad tokens
    return x
```

### 1.3 Comparison Table: ESM2 vs ESM-C (the delta you must implement correctly)

| Architectural feature        | ESM2                          | ESM-C                        |
|------------------------------|-------------------------------|------------------------------|
| FFN activation               | GELU                          | **SwiGLU** (3 weight matrices)|
| FFN biases                   | Yes                           | **No**                       |
| Attention projection biases  | Yes                           | **No**                       |
| LayerNorm biases             | Yes                           | **No**                       |
| Pre-encoder LN (wrap-around) | Yes (`emb_layer_norm_before`) | **No**                       |
| Post-encoder LN (wrap-around)| Yes (`emb_layer_norm_after`)  | **No** (just a final LN)     |
| Token dropout                | Yes (mask token ~15% at train)| **No**                       |
| Context length               | 1024                          | **2048**                     |
| FFN hidden dim               | 4 × d_model                   | **~8/3 × d_model**           |
| Rope theta                   | 10,000                        | 10,000 (same)                |
| Vocabulary size              | 33                            | 33 (same)                    |

Getting any of these wrong produces plausible-looking but numerically wrong embeddings —
they won't error, they'll just silently produce wrong outputs.

---

## Part 2: Weight Layout and Naming

### 2.1 Source of Truth for Weights

The open-weight models (300M and 600M) are available at:
- `EvolutionaryScale/esmc-300m-2024-12` on HuggingFace (legacy, safetensors)
- `biohub/ESMC-300M` and `biohub/ESMC-600M` on HuggingFace (Transformers-compatible)

The `biohub/ESMC-*` versions are preferable for conversion because they expose
standard HuggingFace `config.json` and `model.safetensors`.

**Before writing the converter, run this Python inspection script to enumerate the
exact tensor names and shapes from the actual weights:**

```python
from safetensors import safe_open
import json

# Load and print all tensor names + shapes for esmc-300m
with safe_open("model.safetensors", framework="pt") as f:
    for key in sorted(f.keys()):
        t = f.get_tensor(key)
        print(f"{key:80s} {str(list(t.shape)):30s} {t.dtype}")
```

**Expected tensor naming pattern** (based on HuggingFace Transformers ESM-C layout):

```
# Token embedding
model.embed_tokens.weight                             [33, 960]

# Per-layer attention (NO biases — tensors end in .weight only)
model.layers.{i}.self_attn.q_proj.weight             [960, 960]
model.layers.{i}.self_attn.k_proj.weight             [960, 960]
model.layers.{i}.self_attn.v_proj.weight             [960, 960]
model.layers.{i}.self_attn.out_proj.weight           [960, 960]

# Per-layer LayerNorm (NO bias — weight only)
model.layers.{i}.self_attn_layer_norm.weight         [960]
model.layers.{i}.final_layer_norm.weight             [960]

# Per-layer FFN (SwiGLU: 3 matrices, NO biases)
model.layers.{i}.ffn.gate_proj.weight                [2560, 960]  # W_gate
model.layers.{i}.ffn.up_proj.weight                  [2560, 960]  # W_up
model.layers.{i}.ffn.down_proj.weight                [960, 2560]  # W_down

# Final LayerNorm
model.norm.weight                                     [960]

# LM head (for masked-token scoring)
lm_head.weight                                        [33, 960]
```

Note: Layer naming conventions can differ between the `EvolutionaryScale` and `biohub`
HuggingFace repos. **Always run the inspection script on your actual download** before
building the converter. The shapes are the source of truth; names can be mapped.

### 2.2 Verifying the SwiGLU FFN dimension

Before proceeding, verify the actual FFN intermediate dimension from the weights:

```python
# For 300M: d_model=960, expected ~2560
gate_proj_shape = state_dict["model.layers.0.ffn.gate_proj.weight"].shape
print(f"FFN intermediate dim: {gate_proj_shape[0]}")  # must print e.g. 2560

# Compute the ratio
ffn_dim = gate_proj_shape[0]
d_model  = 960
ratio    = ffn_dim / d_model
print(f"FFN ratio: {ratio:.4f}")  # should be ~2.667 (8/3)
```

Record these exact values — they go into the GGUF metadata and the C++ hparams struct.

---

## Part 3: GGUF Format Design

### 3.1 GGUF Metadata Keys for ESM-C

Define these in your converter Python script and in `esmc-arch.h`. Follow llama.cpp
conventions: architecture prefix `esmc`, keys snake_cased:

```json
{
  "general.architecture": "esmc",
  "general.name": "ESM-C 300M",

  "esmc.context_length": 2048,
  "esmc.embedding_length": 960,
  "esmc.feed_forward_length": 2560,
  "esmc.block_count": 30,
  "esmc.attention.head_count": 15,
  "esmc.attention.head_count_kv": 15,
  "esmc.attention.layer_norm_epsilon": 1e-5,
  "esmc.rope.freq_base": 10000.0,
  "esmc.vocab_size": 33,

  "tokenizer.ggml.model": "esmc",
  "tokenizer.ggml.tokens": ["<cls>","<pad>","<eos>","<unk>","L","A",...,"<mask>"],
  "tokenizer.ggml.token_type": [3,3,3,3,1,1,...,3],
  "tokenizer.ggml.bos_token_id": 0,
  "tokenizer.ggml.eos_token_id": 2,
  "tokenizer.ggml.padding_token_id": 1,
  "tokenizer.ggml.mask_token_id": 32
}
```

Token types: 1 = normal, 3 = control/special.

### 3.2 GGUF Tensor Name Mapping

Define this mapping completely in the converter. These are the canonical GGUF tensor
names your C++ loader will expect:

```python
HF_TO_GGUF_MAP = {
    # Embeddings
    "model.embed_tokens.weight":                "token_embd.weight",

    # Per-layer attention (pattern: blk.{i}.*)
    "model.layers.{i}.self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "model.layers.{i}.self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "model.layers.{i}.self_attn.out_proj.weight":"blk.{i}.attn_output.weight",

    # LayerNorms (weight only, no bias)
    "model.layers.{i}.self_attn_layer_norm.weight": "blk.{i}.attn_norm.weight",
    "model.layers.{i}.final_layer_norm.weight":     "blk.{i}.ffn_norm.weight",

    # SwiGLU FFN (three matrices)
    "model.layers.{i}.ffn.gate_proj.weight":    "blk.{i}.ffn_gate.weight",
    "model.layers.{i}.ffn.up_proj.weight":      "blk.{i}.ffn_up.weight",
    "model.layers.{i}.ffn.down_proj.weight":    "blk.{i}.ffn_down.weight",

    # Final norm and LM head
    "model.norm.weight":                        "output_norm.weight",
    "lm_head.weight":                           "output.weight",
}
```

Biases are absent from ESM-C so no bias tensors need to be mapped.

---

## Part 4: Converter Script

### 4.1 File: `tools/convert_esmc_to_gguf.py`

Full conversion script with all edge cases handled:

```python
#!/usr/bin/env python3
"""Convert ESM-C HuggingFace weights to GGUF format for esmc.cpp."""

import sys, json, struct, numpy as np
from pathlib import Path
from safetensors import safe_open
import argparse

# ── Token alphabet (must be exact, index order is sacred) ─────────────────────
ESMC_TOKENS = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",
    "P", "K", "N", "Q", "F", "Y", "M", "H", "W", "C",
    "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>",
]
ESMC_TOKEN_TYPES = [3, 3, 3, 3] + [1]*20 + [1]*5 + [1, 1, 3, 3]  # 33 total

# ── Architecture configs ───────────────────────────────────────────────────────
ESMC_CONFIGS = {
    "300m": {"n_layers": 30, "d_model": 960,  "n_heads": 15, "rope_theta": 10000.0},
    "600m": {"n_layers": 36, "d_model": 1152, "n_heads": 18, "rope_theta": 10000.0},
    "6b":   {"n_layers": 80, "d_model": 2560, "n_heads": 40, "rope_theta": 10000.0},
}

def detect_model_size(state_dict):
    """Infer model size from embed_tokens shape."""
    emb = next(v for k, v in state_dict.items() if "embed_tokens" in k)
    d_model = emb.shape[-1]
    size_map = {960: "300m", 1152: "600m", 2560: "6b"}
    return size_map[d_model]

def hf_to_gguf_name(hf_name, n_layers):
    """Map a HuggingFace tensor name to its GGUF canonical name."""
    # Handle per-layer patterns
    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        if hf_name.startswith(prefix):
            suffix = hf_name[len(prefix):]
            layer_map = {
                "self_attn.q_proj.weight":       f"blk.{i}.attn_q.weight",
                "self_attn.k_proj.weight":       f"blk.{i}.attn_k.weight",
                "self_attn.v_proj.weight":       f"blk.{i}.attn_v.weight",
                "self_attn.out_proj.weight":     f"blk.{i}.attn_output.weight",
                "self_attn_layer_norm.weight":   f"blk.{i}.attn_norm.weight",
                "final_layer_norm.weight":       f"blk.{i}.ffn_norm.weight",
                "ffn.gate_proj.weight":          f"blk.{i}.ffn_gate.weight",
                "ffn.up_proj.weight":            f"blk.{i}.ffn_up.weight",
                "ffn.down_proj.weight":          f"blk.{i}.ffn_down.weight",
            }
            return layer_map.get(suffix)
    # Global tensors
    global_map = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight":         "output_norm.weight",
        "lm_head.weight":            "output.weight",
    }
    return global_map.get(hf_name)

class GGUFWriter:
    """Minimal GGUF v3 writer. For production, use gguf-py from llama.cpp."""
    # In practice, use: from gguf import GGUFWriter
    # This is a pedagogical stub showing the structure.
    # See: llama.cpp/gguf-py/gguf/gguf_writer.py
    pass

def convert(model_dir: str, output_path: str, dtype="f16"):
    model_dir = Path(model_dir)

    # Load all tensors from safetensors
    state_dict = {}
    for sf_file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(sf_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    size = detect_model_size(state_dict)
    cfg  = ESMC_CONFIGS[size]
    n_layers = cfg["n_layers"]

    # Verify SwiGLU dimension by reading it from actual weights
    ffn_gate_key = f"model.layers.0.ffn.gate_proj.weight"
    if ffn_gate_key not in state_dict:
        # Try alternate naming
        ffn_gate_key = next(k for k in state_dict if "gate_proj" in k and "layers.0" in k)
    ffn_dim = state_dict[ffn_gate_key].shape[0]
    print(f"Detected: size={size}, d_model={cfg['d_model']}, "
          f"n_layers={n_layers}, ffn_dim={ffn_dim}")

    # ── Use gguf-py from llama.cpp (the real implementation) ──────────────────
    # Install: pip install gguf  (from llama.cpp/gguf-py)
    from gguf import GGUFWriter, GGUFValueType

    writer = GGUFWriter(output_path, "esmc")

    # Architecture metadata
    writer.add_uint32("esmc.block_count",               n_layers)
    writer.add_uint32("esmc.context_length",            2048)
    writer.add_uint32("esmc.embedding_length",          cfg["d_model"])
    writer.add_uint32("esmc.feed_forward_length",       ffn_dim)
    writer.add_uint32("esmc.attention.head_count",      cfg["n_heads"])
    writer.add_uint32("esmc.attention.head_count_kv",   cfg["n_heads"])
    writer.add_float32("esmc.rope.freq_base",           cfg["rope_theta"])
    writer.add_uint32("esmc.vocab_size",                33)
    writer.add_float32("esmc.attention.layer_norm_epsilon", 1e-5)
    writer.add_string("general.name",                   f"ESM-C {size.upper()}")

    # Tokenizer
    writer.add_array("tokenizer.ggml.tokens",      ESMC_TOKENS)
    writer.add_array("tokenizer.ggml.token_type",  ESMC_TOKEN_TYPES)
    writer.add_uint32("tokenizer.ggml.bos_token_id",  0)
    writer.add_uint32("tokenizer.ggml.eos_token_id",  2)
    writer.add_uint32("tokenizer.ggml.padding_token_id", 1)
    writer.add_uint32("tokenizer.ggml.mask_token_id", 32)

    # Tensors
    import torch
    np_dtype = np.float16 if dtype == "f16" else np.float32

    for hf_name, tensor in state_dict.items():
        gguf_name = hf_to_gguf_name(hf_name, n_layers)
        if gguf_name is None:
            print(f"  SKIP: {hf_name}")
            continue

        arr = tensor.to(torch.float16).numpy().astype(np_dtype)

        # LayerNorm weights stay in F32 (sensitive to quantization)
        if "norm.weight" in gguf_name:
            arr = tensor.float().numpy()

        writer.add_tensor(gguf_name, arr)
        print(f"  {hf_name:70s} → {gguf_name:40s} {list(arr.shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWrote: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="HuggingFace model directory")
    parser.add_argument("output",    help="Output .gguf file path")
    parser.add_argument("--dtype",   default="f16", choices=["f16", "f32"])
    args = parser.parse_args()
    convert(args.model_dir, args.output, args.dtype)
```

**Usage:**
```bash
# Download weights
huggingface-cli download biohub/ESMC-300M --local-dir ./esmc-300m/

# Convert to GGUF F16
pip install gguf safetensors transformers
python tools/convert_esmc_to_gguf.py ./esmc-300m/ ./models/esmc-300m-f16.gguf

# Verify the GGUF output
python -c "
from gguf import GGUFReader
r = GGUFReader('./models/esmc-300m-f16.gguf')
for k, v in r.fields.items():
    print(f'{k}: {v.parts}')
for t in r.tensors:
    print(f'{t.name:50s} {list(t.shape)}')
"
```

---

## Part 5: C++ Data Structures

### 5.1 File: `src/esmc-arch.h`

```cpp
#pragma once
#include "ggml.h"
#include <vector>
#include <string>

// ── Hyperparameters ───────────────────────────────────────────────────────────
struct esmc_hparams {
    uint32_t n_vocab      = 33;
    uint32_t n_ctx        = 2048;
    uint32_t n_embd       = 960;     // d_model
    uint32_t n_intermediate = 2560;  // ffn_dim (SwiGLU intermediate)
    uint32_t n_head       = 15;
    uint32_t n_head_kv    = 15;
    uint32_t n_layer      = 30;
    float    norm_eps     = 1e-5f;
    float    rope_theta   = 10000.0f;
};

// ── Per-layer weights ─────────────────────────────────────────────────────────
struct esmc_layer {
    // Attention pre-norm (NO bias in ESM-C)
    struct ggml_tensor * attn_norm;       // [n_embd]

    // Attention projections (NO bias)
    struct ggml_tensor * wq;              // [n_embd, n_embd]
    struct ggml_tensor * wk;              // [n_embd, n_embd]
    struct ggml_tensor * wv;              // [n_embd, n_embd]
    struct ggml_tensor * wo;              // [n_embd, n_embd]

    // FFN pre-norm (NO bias)
    struct ggml_tensor * ffn_norm;        // [n_embd]

    // SwiGLU FFN (THREE weight matrices, NO bias)
    struct ggml_tensor * ffn_gate;        // [n_intermediate, n_embd]  W_gate
    struct ggml_tensor * ffn_up;          // [n_intermediate, n_embd]  W_up
    struct ggml_tensor * ffn_down;        // [n_embd, n_intermediate]  W_down
};

// ── Model ─────────────────────────────────────────────────────────────────────
struct esmc_model {
    esmc_hparams hparams;

    // Token embedding table
    struct ggml_tensor * tok_embd;        // [n_vocab, n_embd]

    // Final LayerNorm (post-last-layer, NO bias)
    struct ggml_tensor * output_norm;     // [n_embd]

    // LM head for masked token scoring (optional, can be nullptr)
    struct ggml_tensor * lm_head;         // [n_vocab, n_embd]

    std::vector<esmc_layer> layers;

    // ggml state
    struct ggml_context     * ctx_meta  = nullptr;  // tensor metadata context
    struct ggml_backend     * backend   = nullptr;  // Metal on M1, CPU fallback
    struct ggml_backend_buffer* buf     = nullptr;  // weight storage buffer
    struct gguf_context     * gguf_ctx  = nullptr;  // GGUF file handle

    // Tokenizer (loaded from GGUF metadata)
    std::vector<std::string> vocab;
    int bos_id = 0, eos_id = 2, pad_id = 1, mask_id = 32;
};

// ── Inference context ─────────────────────────────────────────────────────────
struct esmc_context {
    esmc_model * model;

    // Work buffer for graph computation
    struct ggml_context   * ctx_compute = nullptr;
    struct ggml_backend_sched * sched   = nullptr;

    // Output buffer — per-residue embeddings, pre-allocated
    std::vector<float> embd_out;

    // Config
    int n_threads = 4;
};
```

---

## Part 6: Model Loading

### 6.1 File: `src/esmc.cpp` — Loading section

```cpp
#include "esmc-arch.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cassert>
#include <cstring>

// Read a uint32 from GGUF metadata, with a default value if key absent
static uint32_t gguf_get_u32_or(const gguf_context * ctx, const char * key, uint32_t def) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0) return def;
    return gguf_get_val_u32(ctx, idx);
}

static float gguf_get_f32_or(const gguf_context * ctx, const char * key, float def) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0) return def;
    return gguf_get_val_f32(ctx, idx);
}

esmc_model * esmc_model_load(const char * path, bool use_mmap) {
    auto * model = new esmc_model;

    // ── 1. Open GGUF and read metadata ───────────────────────────────────────
    struct gguf_init_params gguf_params = {
        .no_alloc = true,    // don't allocate tensor data yet
        .ctx      = nullptr,
    };
    model->gguf_ctx = gguf_init_from_file(path, gguf_params);
    if (!model->gguf_ctx) {
        fprintf(stderr, "Error: failed to open %s\n", path);
        return nullptr;
    }

    const char * arch = gguf_get_val_str(model->gguf_ctx,
                            gguf_find_key(model->gguf_ctx, "general.architecture"));
    if (strcmp(arch, "esmc") != 0) {
        fprintf(stderr, "Error: expected architecture 'esmc', got '%s'\n", arch);
        return nullptr;
    }

    // ── 2. Parse hparams ─────────────────────────────────────────────────────
    auto & p = model->hparams;
    auto * g = model->gguf_ctx;
    p.n_vocab        = gguf_get_u32_or(g, "esmc.vocab_size",                   33);
    p.n_ctx          = gguf_get_u32_or(g, "esmc.context_length",               2048);
    p.n_embd         = gguf_get_u32_or(g, "esmc.embedding_length",              960);
    p.n_intermediate = gguf_get_u32_or(g, "esmc.feed_forward_length",          2560);
    p.n_head         = gguf_get_u32_or(g, "esmc.attention.head_count",           15);
    p.n_head_kv      = gguf_get_u32_or(g, "esmc.attention.head_count_kv",        15);
    p.n_layer        = gguf_get_u32_or(g, "esmc.block_count",                    30);
    p.norm_eps       = gguf_get_f32_or(g, "esmc.attention.layer_norm_epsilon", 1e-5f);
    p.rope_theta     = gguf_get_f32_or(g, "esmc.rope.freq_base",           10000.0f);

    assert(p.n_embd % p.n_head == 0);  // head_dim must be integral
    const uint32_t head_dim = p.n_embd / p.n_head;
    printf("ESM-C: layers=%u embd=%u ffn=%u heads=%u head_dim=%u ctx=%u\n",
           p.n_layer, p.n_embd, p.n_intermediate, p.n_head, head_dim, p.n_ctx);

    // ── 3. Read tokenizer vocab from GGUF metadata ───────────────────────────
    {
        const int vocab_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
        const int n = gguf_get_arr_n(g, vocab_idx);
        model->vocab.resize(n);
        for (int i = 0; i < n; i++) {
            model->vocab[i] = gguf_get_arr_str(g, vocab_idx, i);
        }
        model->bos_id  = gguf_get_u32_or(g, "tokenizer.ggml.bos_token_id",     0);
        model->eos_id  = gguf_get_u32_or(g, "tokenizer.ggml.eos_token_id",     2);
        model->pad_id  = gguf_get_u32_or(g, "tokenizer.ggml.padding_token_id", 1);
        model->mask_id = gguf_get_u32_or(g, "tokenizer.ggml.mask_token_id",   32);
    }

    // ── 4. Create ggml metadata context and register tensor shapes ───────────
    // Compute total number of tensors:
    // Global: tok_embd + output_norm + lm_head = 3
    // Per layer: attn_norm + wq + wk + wv + wo + ffn_norm + ffn_gate + ffn_up + ffn_down = 9
    const int n_tensors = 3 + 9 * p.n_layer;
    struct ggml_init_params meta_params = {
        .mem_size   = ggml_tensor_overhead() * n_tensors,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    model->ctx_meta = ggml_init(meta_params);

    // ── 5. Create the backend (Metal on M1, CPU fallback) ────────────────────
    model->backend = ggml_backend_metal_init();
    if (!model->backend) {
        fprintf(stderr, "Warning: Metal unavailable, falling back to CPU\n");
        model->backend = ggml_backend_cpu_init();
    }

    // ── 6. Allocate a backend buffer for all weights ──────────────────────────
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(model->backend);

    // Helper lambda: create a ggml tensor from GGUF by name
    auto get_tensor = [&](const char * name) -> ggml_tensor * {
        const int gguf_idx = gguf_find_tensor(g, name);
        if (gguf_idx < 0) {
            // Could be missing (e.g. lm_head is optional)
            return nullptr;
        }
        struct ggml_tensor * t = ggml_get_tensor(model->ctx_meta, name);
        if (!t) {
            // Need to create it from GGUF tensor info
            // (implementation detail: use gguf tensor type/dims to call ggml_new_tensor)
            // In practice, use llama.cpp's loader pattern
        }
        return t;
    };

    // ── 7. Wire up all model tensors ─────────────────────────────────────────
    model->tok_embd    = get_tensor("token_embd.weight");
    model->output_norm = get_tensor("output_norm.weight");
    model->lm_head     = get_tensor("output.weight");  // nullable

    model->layers.resize(p.n_layer);
    for (uint32_t i = 0; i < p.n_layer; i++) {
        auto & layer = model->layers[i];
        char name[128];

        #define GET(field, fmt)  do { \
            snprintf(name, sizeof(name), fmt, i); \
            layer.field = get_tensor(name); \
            if (!layer.field) { \
                fprintf(stderr, "Missing tensor: %s\n", name); \
                return nullptr; \
            } \
        } while(0)

        GET(attn_norm,  "blk.%u.attn_norm.weight");
        GET(wq,         "blk.%u.attn_q.weight");
        GET(wk,         "blk.%u.attn_k.weight");
        GET(wv,         "blk.%u.attn_v.weight");
        GET(wo,         "blk.%u.attn_output.weight");
        GET(ffn_norm,   "blk.%u.ffn_norm.weight");
        GET(ffn_gate,   "blk.%u.ffn_gate.weight");
        GET(ffn_up,     "blk.%u.ffn_up.weight");
        GET(ffn_down,   "blk.%u.ffn_down.weight");
        #undef GET
    }

    return model;
}
```

---

## Part 7: Forward Pass — The Compute Graph

This is the most important function in the project. It builds a ggml computation graph
that represents the ESM-C forward pass. ggml will dispatch this to Metal or CPU.

### 7.1 File: `src/esmc.cpp` — Graph construction

```cpp
static struct ggml_cgraph * esmc_build_graph(
    esmc_context * ectx,
    const int32_t * tokens,       // [n_tokens] — flat array including CLS/EOS
    int32_t   n_tokens)
{
    const esmc_model  & model  = *ectx->model;
    const esmc_hparams & p     = model.hparams;

    const int n_embd      = (int)p.n_embd;
    const int n_heads     = (int)p.n_head;
    const int n_layers    = (int)p.n_layer;
    const int n_ffd       = (int)p.n_intermediate;
    const int head_dim    = n_embd / n_heads;

    // Work context for this forward pass (graph nodes, not weights)
    struct ggml_init_params params = {
        .mem_size   = 256*1024*1024,   // 256 MB scratch; tune for long sequences
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // ── Input token IDs ───────────────────────────────────────────────────────
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    // ── Token embedding lookup ─────────────────────────────────────────────────
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_tokens);
    ggml_set_name(cur, "embeddings");
    // cur: [n_tokens, n_embd] — column-major in ggml: shape is (n_embd, n_tokens)

    // ── NO pre-encoder LayerNorm (ESM-C has NO emb_layer_norm_before!) ────────

    // ── Transformer layers ────────────────────────────────────────────────────
    for (int il = 0; il < n_layers; il++) {
        const auto & layer = model.layers[il];

        struct ggml_tensor * residual = cur;

        // ── Attention sub-block ───────────────────────────────────────────────

        // Pre-norm (RMSNorm-style if no bias; LayerNorm without bias in ESM-C)
        // ESM-C uses standard LayerNorm with weight but no bias:
        // x_norm = LN(x) * weight   (no + bias term)
        cur = ggml_norm(ctx, cur, p.norm_eps);
        cur = ggml_mul(ctx, cur, layer.attn_norm);  // element-wise scale (no bias)
        ggml_set_name(cur, "attn_norm");

        // Q, K, V projections (NO bias — ggml_mul_mat only, no ggml_add)
        struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.wq, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, layer.wk, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, layer.wv, cur);
        ggml_set_name(Q, "Q"); ggml_set_name(K, "K"); ggml_set_name(V, "V");

        // Reshape Q, K, V to multi-head layout: (head_dim, n_heads, n_tokens)
        Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, n_tokens);
        K = ggml_reshape_3d(ctx, K, head_dim, n_heads, n_tokens);
        V = ggml_reshape_3d(ctx, V, head_dim, n_heads, n_tokens);

        // Scale Q before RoPE (ESM-C inherits this from ESM2)
        Q = ggml_scale(ctx, Q, 1.0f / sqrtf((float)head_dim));

        // Apply RoPE to Q and K
        // GGML_ROPE_TYPE_NEOX = 2 — matches ESM-C's rotary implementation
        // Position indices are 0..n_tokens-1
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(pos, "pos");
        ggml_set_input(pos);

        Q = ggml_rope_ext(ctx, Q, pos,
                          /*freq_factors=*/nullptr,
                          head_dim,
                          GGML_ROPE_TYPE_NEOX,
                          p.n_ctx,
                          p.rope_theta,
                          /*freq_scale=*/1.0f,
                          /*ext_factor=*/0.0f,
                          /*attn_factor=*/1.0f,
                          /*beta_fast=*/0.0f,
                          /*beta_slow=*/0.0f);
        K = ggml_rope_ext(ctx, K, pos,
                          nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                          p.n_ctx, p.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute for attention matmul: (head_dim, n_tokens, n_heads, 1)
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_permute(ctx, V, 1, 2, 0, 3);  // V: (n_tokens, head_dim, n_heads)

        // Attention: Q @ K^T → (n_tokens, n_tokens, n_heads)
        // NOTE: No causal mask — ESM-C is a bidirectional encoder
        struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
        ggml_set_name(KQ, "KQ");

        // Padding mask (optional — add -inf for pad token positions):
        // struct ggml_tensor * mask = ...;  // [n_tokens, n_tokens]
        // KQ = ggml_add(ctx, KQ, mask);
        // For single-sequence inference without padding, skip this.

        // Softmax
        KQ = ggml_soft_max(ctx, KQ);
        ggml_set_name(KQ, "KQ_soft_max");

        // Weighted sum: (head_dim, n_tokens, n_heads)
        struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
        ggml_set_name(KQV, "KQV");

        // Merge heads: permute back and make contiguous
        KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx, KQV, n_embd, n_tokens);
        ggml_set_name(cur, "KQV_merged");

        // Output projection (NO bias)
        cur = ggml_mul_mat(ctx, layer.wo, cur);
        ggml_set_name(cur, "attn_out");

        // Residual connection
        cur = ggml_add(ctx, cur, residual);

        // ── FFN sub-block (SwiGLU) ────────────────────────────────────────────

        residual = cur;

        // Pre-norm (weight only, no bias)
        cur = ggml_norm(ctx, cur, p.norm_eps);
        cur = ggml_mul(ctx, cur, layer.ffn_norm);
        ggml_set_name(cur, "ffn_norm");

        // SwiGLU: FFN(x) = W_down( SiLU(W_gate(x)) ⊙ W_up(x) )
        // NO biases anywhere
        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up,   cur);

        // SiLU activation on gate path
        gate = ggml_silu(ctx, gate);

        // Element-wise product
        cur = ggml_mul(ctx, gate, up);
        ggml_set_name(cur, "ffn_swiglu");

        // Down projection (NO bias)
        cur = ggml_mul_mat(ctx, layer.ffn_down, cur);
        ggml_set_name(cur, "ffn_out");

        // Residual
        cur = ggml_add(ctx, cur, residual);
    }

    // ── Final LayerNorm (weight only, no bias) ────────────────────────────────
    cur = ggml_norm(ctx, cur, p.norm_eps);
    cur = ggml_mul(ctx, cur, model.output_norm);
    ggml_set_name(cur, "output_norm");

    // cur is now [n_embd, n_tokens] — per-residue embeddings
    ggml_build_forward_expand(gf, cur);

    return gf;
}
```

### 7.2 Running the graph

```cpp
int esmc_embed(
    esmc_context * ectx,
    const int32_t * tokens,
    int32_t n_tokens,
    float * embeddings_out)    // preallocated: [n_tokens * n_embd] floats
{
    const auto & p = ectx->model->hparams;

    // Build and allocate the graph
    struct ggml_cgraph * gf = esmc_build_graph(ectx, tokens, n_tokens);

    // Set input: token IDs
    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    memcpy(inp_tokens->data, tokens, n_tokens * sizeof(int32_t));

    // Set input: position indices
    struct ggml_tensor * pos = ggml_graph_get_tensor(gf, "pos");
    for (int32_t i = 0; i < n_tokens; i++) {
        ((int32_t*)pos->data)[i] = i;
    }

    // Execute
    ggml_backend_graph_compute(ectx->model->backend, gf);

    // Read output: the last node in the graph is the output tensor
    struct ggml_tensor * output = ggml_graph_node(gf, -1);
    ggml_backend_tensor_get(output, embeddings_out, 0,
                            n_tokens * p.n_embd * sizeof(float));

    return 0;
}
```

---

## Part 8: Tokenization

The tokenizer is trivial — direct character-to-index lookup from the 33-token alphabet.
No BPE, no subword splitting.

```cpp
// esmc-tokenize.cpp

static const char * ESMC_TOKENS[] = {
    "<cls>","<pad>","<eos>","<unk>",
    "L","A","G","V","S","E","R","T","I","D",
    "P","K","N","Q","F","Y","M","H","W","C",
    "X","B","U","Z","O",".","-","<null_1>","<mask>"
};

// Build reverse map: single-char amino acids to token ID
static std::unordered_map<char, int32_t> build_aa_map() {
    std::unordered_map<char, int32_t> m;
    for (int i = 4; i <= 28; i++) {
        // Single-char tokens start at index 4
        if (strlen(ESMC_TOKENS[i]) == 1)
            m[ESMC_TOKENS[i][0]] = i;
    }
    return m;
}

int esmc_tokenize(
    const esmc_model * model,
    const char * sequence,            // raw amino acid string e.g. "MKTVRQ..."
    int32_t * tokens_out,             // output buffer
    int32_t max_tokens)               // must be >= strlen(sequence) + 2
{
    static auto aa_map = build_aa_map();

    const int seq_len = (int)strlen(sequence);
    if (seq_len + 2 > max_tokens) return -1;  // +2 for CLS and EOS

    int pos = 0;
    tokens_out[pos++] = model->bos_id;  // prepend <cls>

    for (int i = 0; i < seq_len; i++) {
        char c = toupper(sequence[i]);
        auto it = aa_map.find(c);
        if (it == aa_map.end()) {
            tokens_out[pos++] = 3;   // <unk> for unknown residue
        } else {
            tokens_out[pos++] = it->second;
        }
    }

    tokens_out[pos++] = model->eos_id;  // append <eos>
    return pos;  // total token count
}
```

---

## Part 9: Quantization

### 9.1 Building the quantizer

Reuse `llama-quantize` from llama.cpp — it already reads GGUF and writes quantized GGUF.
Your converter writes valid GGUF, so this works out of the box:

```bash
# From llama.cpp build directory:
./llama-quantize ./models/esmc-300m-f16.gguf ./models/esmc-300m-Q4_K_M.gguf Q4_K_M

# With importance matrix (recommended for embeddings):
./llama-imatrix -m ./models/esmc-300m-f16.gguf \
                -f calibration_sequences.txt    \  # FASTA or one-per-line format
                -o esmc-300m-imatrix.gguf

./llama-quantize --imatrix esmc-300m-imatrix.gguf \
                 ./models/esmc-300m-f16.gguf \
                 ./models/esmc-300m-Q4_K_M_imat.gguf \
                 Q4_K_M
```

### 9.2 Recommended quantization map for ESM-C

Embeddings require higher precision than LLM logits. Follow this policy:

```
token_embd.weight        → Q8_0    (embedding table: precision-sensitive)
blk.*.attn_q.weight      → Q4_K_M
blk.*.attn_k.weight      → Q4_K_M
blk.*.attn_v.weight      → Q4_K_M  (or Q6_K for higher quality)
blk.*.attn_output.weight → Q4_K_M
blk.*.ffn_gate.weight    → Q4_K_M
blk.*.ffn_up.weight      → Q4_K_M
blk.*.ffn_down.weight    → Q6_K    (output projection: higher precision)
blk.*.attn_norm.weight   → F32     (LayerNorm: always full precision)
blk.*.ffn_norm.weight    → F32
output_norm.weight       → F32
output.weight            → Q8_0    (LM head: precision matters for scoring)
```

### 9.3 Expected model sizes after quantization

| Variant | F16      | Q8_0     | Q4_K_M  |
|---------|----------|----------|---------|
| 300M    | ~600 MB  | ~315 MB  | ~175 MB |
| 600M    | ~1.2 GB  | ~630 MB  | ~350 MB |
| 6B      | ~12 GB   | ~6.3 GB  | ~3.5 GB |

The 300M Q4_K_M at ~175 MB is trivial even for 8 GB M1 unified memory.
The 6B Q4_K_M at ~3.5 GB fits on any M1 Pro (16 GB+ unified memory) with headroom.

---

## Part 10: CLI Tools

### 10.1 `esmc-embed` — main inference binary

```bash
# Embed a single sequence
./esmc-embed \
    -m ./models/esmc-300m-Q4_K_M.gguf \
    -s "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGY" \
    --pool mean \
    --output embedding.npy

# Embed all sequences in a FASTA file
./esmc-embed \
    -m ./models/esmc-600m-Q4_K_M.gguf \
    --fasta proteins.fasta \
    --batch-size 32 \
    --output-dir ./embeddings/ \
    --layer last               # or --layer all for all-layer output

# Score sequence variants (masked-marginal scoring)
./esmc-embed \
    -m ./models/esmc-300m-Q4_K_M.gguf \
    --score-variants variants.tsv \  # TSV: sequence, position, wildtype, mutant
    --output scores.tsv
```

### 10.2 Command-line argument spec (for `argparse` / custom CLI)

```
-m / --model PATH        GGUF model file (required)
-s / --sequence STR      Single amino acid sequence
-f / --fasta PATH        FASTA file with multiple sequences
-o / --output PATH       Output file (.npy, .h5, .tsv)
--batch-size INT         Sequences per batch (default: 16)
--layer INT              Which layer to extract embeddings from (-1 = last)
--pool [mean|cls|none]   Pooling strategy (default: mean over non-special tokens)
--threads INT            CPU threads (default: 4)
--no-metal               Disable Metal, use CPU only
```

---

## Part 11: Numerical Validation

This is mandatory before shipping. Build this test harness first, before the CLI.

### 11.1 Reference script: `tests/generate_reference.py`

```python
"""Generate reference embeddings using the official PyTorch ESM-C model."""
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import numpy as np

TEST_SEQUENCES = [
    ("short",  "ACDEFGHIK"),
    ("medium", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGY"),
    ("long",   "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQ" * 5),
]

model = ESMC.from_pretrained("esmc_300m").to("cpu")
model.eval()

reference = {}
for name, seq in TEST_SEQUENCES:
    protein = ESMProtein(sequence=seq)
    tensor = model.encode(protein)
    out = model.logits(tensor, LogitsConfig(sequence=False, return_embeddings=True))
    emb = out.embeddings.squeeze(0).numpy()  # [L, d_model] including CLS/EOS
    reference[name] = emb
    print(f"{name}: shape={emb.shape}, mean={emb.mean():.6f}, std={emb.std():.6f}")
    # Save first 5 values of first residue as ground truth
    print(f"  residue[0][:5] = {emb[1, :5]}")  # index 1 = first real AA (skip CLS)

np.savez("tests/reference_embeddings.npz", **reference)
```

### 11.2 Validation script: `tests/validate.py`

```python
import numpy as np, subprocess, sys

ref = np.load("tests/reference_embeddings.npz")

for name, seq in TEST_SEQUENCES:
    # Run esmc-embed and capture output
    result = subprocess.run([
        "./esmc-embed", "-m", "models/esmc-300m-Q4_K_M.gguf",
        "-s", seq, "--pool", "none", "--output", f"/tmp/{name}.npy"
    ], capture_output=True)

    our_emb = np.load(f"/tmp/{name}.npy")     # [L, d_model]
    ref_emb = ref[name][1:-1]                  # strip CLS/EOS from reference

    # Per-residue cosine similarity
    cos_sim = np.einsum("ij,ij->i", our_emb, ref_emb) / (
        np.linalg.norm(our_emb, axis=1) * np.linalg.norm(ref_emb, axis=1) + 1e-9
    )

    print(f"{name}: mean_cos={cos_sim.mean():.6f}  min_cos={cos_sim.min():.6f}")
    assert cos_sim.mean() > 0.999, f"FAIL {name}: mean cosine sim too low"
    assert cos_sim.min()  > 0.99,  f"FAIL {name}: min cosine sim too low"

    # Mean-pooled embedding error
    our_mean = our_emb.mean(axis=0)
    ref_mean = ref_emb.mean(axis=0)
    l2_err = np.linalg.norm(our_mean - ref_mean) / np.linalg.norm(ref_mean)
    print(f"  mean_pool relative L2 error: {l2_err:.6f}")
    assert l2_err < 0.01, f"FAIL {name}: mean pool L2 error too high"

print("ALL TESTS PASSED")
```

### 11.3 Known numerical traps (most likely sources of bugs)

These are the specific things most likely to produce silent correctness failures:

1. **SwiGLU vs GELU**: ESM-C uses `silu(gate) * up`, not `gelu(x)`. If you accidentally
   implement `gelu(fc1(x))` you'll get biologically wrong embeddings that look numerically
   plausible.

2. **Missing biases vs. present biases**: ESM-C has NO biases anywhere. If you add even
   one `ggml_add(..., bias)` where a bias doesn't exist, you add zeros (which is fine)
   or garbage (if the buffer is uninitialized). The converter must verify zero bias tensors
   are NOT emitted to GGUF.

3. **LayerNorm implementation**: `ggml_norm` computes standard LayerNorm without scaling.
   You must follow with `ggml_mul(ctx, cur, norm_weight)` to apply the learned scale.
   If you write `ggml_rms_norm` (RMSNorm), your output will differ — ESM-C uses standard
   LayerNorm despite having no bias.

4. **RoPE interleaving style**: ggml supports multiple RoPE variants:
   - `GGML_ROPE_TYPE_GPT2` = 0 (even/odd interleaving)
   - `GGML_ROPE_TYPE_NEOX` = 2 (first-half/second-half splitting)
   ESM-C uses the NeoX style (first `head_dim/2` dimensions rotate with one set of
   frequencies, last `head_dim/2` with another). Verify this by checking the reference
   embedding matches after the first transformer layer.

5. **Query scaling position**: Scale Q by `1/sqrt(head_dim)` BEFORE applying RoPE,
   not after. This changes the RoPE-rotated magnitudes and affects the attention distribution.

6. **FFN intermediate dimension**: The actual `ffn_dim` in ESM-C is NOT exactly `8/3 * d_model`
   — it may be rounded to a multiple of 64 or 256. Always read it from the actual weight
   shapes, not from a formula. Record it in GGUF and read it back in C++.

7. **Token index 0 = CLS, not a residue**: When stripping CLS/EOS for downstream use,
   skip `tokens[0]` (CLS) and `tokens[n-1]` (EOS). The per-residue embeddings are
   `output[1:L+1]` where L is sequence length.

---

## Part 12: Project Repository Structure

```
esmc.cpp/
├── CMakeLists.txt
├── README.md
│
├── ggml/                              # git submodule: ggml-org/llama.cpp
│   └── (entire llama.cpp tree)
│
├── src/
│   ├── esmc.h                         # public C API header
│   ├── esmc.cpp                       # model loading + graph construction
│   ├── esmc-arch.h                    # structs: hparams, layer, model, context
│   └── esmc-tokenize.cpp              # tokenizer (33-token AA alphabet)
│
├── tools/
│   ├── convert_esmc_to_gguf.py        # HuggingFace safetensors → GGUF F16
│   └── validate_conversion.py         # verify GGUF vs HF reference numerics
│
├── examples/
│   ├── embed/
│   │   ├── CMakeLists.txt
│   │   └── main.cpp                   # esmc-embed binary
│   └── server/
│       └── main.cpp                   # optional: HTTP embedding server
│
└── tests/
    ├── generate_reference.py          # run once: save PyTorch reference outputs
    ├── validate.py                    # run after build: check cosine similarity
    └── test_tokenizer.py              # verify token→id mapping is exact
```

### 12.1 CMakeLists.txt skeleton

```cmake
cmake_minimum_required(VERSION 3.14)
project(esmc.cpp CXX)
set(CMAKE_CXX_STANDARD 17)

# ggml/llama.cpp as subdirectory
add_subdirectory(ggml)

# Core library
add_library(esmc STATIC
    src/esmc.cpp
    src/esmc-tokenize.cpp
)
target_include_directories(esmc PUBLIC src ggml/include)
target_link_libraries(esmc PRIVATE ggml)

# Metal backend on Apple
if(APPLE)
    target_compile_options(esmc PRIVATE -DGGML_USE_METAL)
    find_library(METAL_LIB Metal)
    find_library(MPS_LIB MetalPerformanceShaders)
    target_link_libraries(esmc PRIVATE ${METAL_LIB} ${MPS_LIB})
endif()

# esmc-embed binary
add_executable(esmc-embed examples/embed/main.cpp)
target_link_libraries(esmc-embed PRIVATE esmc)

# Build command:
# cmake -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build -j8
```

---

## Part 13: Milestones and Development Order

Follow this exact order — do NOT skip ahead. Each milestone has a concrete verification:

| # | Milestone | Verification |
|---|-----------|--------------|
| 0 | Clone llama.cpp as submodule, CMake builds on M1 | `make -C build esmc-embed` succeeds |
| 1 | Inspection script runs, all tensor names and shapes printed | Shapes match architecture table in §1.1 |
| 2 | Converter produces valid GGUF, metadata keys readable | `gguf-reader.py` prints all expected keys |
| 3 | C++ loads GGUF, all tensors pointer-assigned with no nullptrs | Debug print of all tensor shapes matches |
| 4 | Tokenizer: `"ACDEF"` → `[0, 5, 23, 13, 9, 18, 2]` | Run test_tokenizer.py |
| 5 | CPU forward pass: single sequence, 8-layer scaffold first | Layer 0 Q/K norms within 5% of PyTorch |
| 6 | Full forward pass on 300M CPU: cosine sim > 0.999 | validate.py passes |
| 7 | Metal dispatch: same numerics as CPU path | validate.py passes with `--metal` flag |
| 8 | Paper benchmark harnesses: shared correctness, throughput, memory, and downstream-task runners | `benchmarks/` scripts emit JSON/CSV artifacts and can run 300M on a 16 GB M1 without OOM |
| 9 | Numerical correctness matrix for 300M: F16, Q8_0, Q4_K_M, Q4_K_S on CPU and Metal | Per-residue cosine vs PyTorch reference: F16/Q8_0 mean > 0.999, Q4_* mean > 0.995 |
| 10 | Numerical correctness matrix for 600M within 16 GB RAM | 600M F16/Q8_0/Q4_* validate one sequence at a time with the same thresholds as milestone 9 |
| 11 | 6B feasibility path for consumer hardware: quantized inference first, F16 only if memory permits or reference is precomputed off-machine | 6B Q4_K_M and Q4_K_S load on 16 GB M1 and match a saved PyTorch/reference embedding slice with mean cosine > 0.995 |
| 12 | Downstream quality benchmark: ProteinGym variant-effect prediction or FLIP function prediction using PyTorch, F16 GGUF, and quantized GGUF embeddings | Task metric delta vs PyTorch reference is recorded; quantized embeddings remain within an agreed tolerance (e.g. Spearman Δ ≤ 0.01 for ProteinGym subset) |
| 13 | Throughput benchmark: sequences/second by model size, precision, and backend vs PyTorch CPU and PyTorch MPS | Benchmark CSV contains median/p95 latency and seq/s for fixed short/medium/long sequence sets |
| 14 | Memory-footprint benchmark: peak RAM by model size and quantization level | Peak RSS/unified-memory usage CSV shows 300M/600M and 6B quantized variants fit within 16 GB with no OOM |
| 15 | `esmc-embed` FASTA input, batch processing, .npy output | Embed SwissProt/ProteinGym sample FASTA and compare pooled embeddings with PyTorch reference |
| 16 | Paper artifacts, README, benchmark tables, and HuggingFace GGUF uploads | Reproduction bundle includes GGUF files, benchmark CSV/JSON, plots, and paper-ready result tables |

Work exclusively on the 300M model through milestones 0–9. Generalize to 600M at
milestone 10. Treat 6B separately at milestone 11 because a 16 GB M1 is unlikely
to hold both PyTorch 6B reference execution and large GGUF working buffers at once;
prefer Q4/Q8 GGUF inference locally and generate any full PyTorch 6B reference
artifacts off-machine or as small saved slices.

### 13.1 Paper Benchmark Strategy for a 16 GB M1

The paper-oriented benchmarks should be designed around the memory limit:

| Benchmark | 16 GB M1 strategy | Verification artifact |
|-----------|-------------------|-----------------------|
| Numerical correctness across sizes and quantization levels | Run one sequence at a time; use 300M/600M PyTorch references locally; use precomputed/off-machine reference slices for 6B if needed | `results/correctness_*.json` with cosine and L2 metrics |
| Downstream task quality | Start with a small ProteinGym or FLIP subset; cache embeddings to disk; compare task metrics from PyTorch, F16 GGUF, and quantized GGUF | `results/downstream_*.csv` with task metric deltas |
| Throughput | Fixed sequence-length buckets; run C++ CPU, C++ Metal, PyTorch CPU, and PyTorch MPS separately to avoid co-resident models | `results/throughput_*.csv` with seq/s, median, p95 |
| Memory footprint | Measure each model/quant/backend in a fresh process; record peak RSS or `time -l` maximum resident set size | `results/memory_*.csv` with peak memory and pass/fail under 16 GB |

Do not require full 6B PyTorch inference on the 16 GB M1 as a milestone gate. For
paper claims about 6B correctness, use a saved reference artifact generated on a
larger machine, or validate a very small slice if local PyTorch execution fits.

### 13.2 Pre-Milestone-8 Paper Benchmark Plan

Before starting milestone 8's general 600M/6B model-size expansion, add a
benchmark harness and result schema that can support the paper. The goal is not
to prove every claim immediately; it is to make every later experiment
repeatable, memory-safe, and comparable across model sizes, precision levels, and
backends.

These paper benchmarks should be treated as **new milestones before broad model
generalization**:

| # | Paper milestone | Verification |
|---|-----------------|--------------|
| 8A | Benchmark result schema and runner layout | `benchmarks/` contains shared config + JSON/CSV writers; a dry run writes `results/manifest.json` with git hash, model path, backend, precision, sequence set, and host info |
| 8B | Numerical correctness benchmark harness | Running the harness on 300M F16 and Q8_0 emits `results/correctness_300m.{json,csv}` with per-sequence mean/min cosine, mean-pool L2, and pass/fail thresholds |
| 8C | Quantization matrix for 300M | F16, Q8_0, Q4_K_M, and Q4_K_S GGUFs all load via `esmc-quantize`; on the 100-sequence Swiss-Prot benchmark **aggregate** mean cosine vs PyTorch reference satisfies F16/Q8_0 > 0.999 and Q4_K_M/Q4_K_S > 0.995. Per-sequence pass rates and the legacy-fallback rationale for n_embd=960 are recorded in `results/correctness_300m.{json,csv}` and `lab_manual.md` |
| 8D | Downstream task harness on a small subset | A ProteinGym or FLIP subset runs end-to-end from cached embeddings; output includes PyTorch reference metric, GGUF metric, quantized metric, and deltas |
| 8E | Throughput benchmark harness | CPU, Metal, PyTorch CPU, and PyTorch MPS runs are executed in separate processes; CSV reports seq/s, median latency, p95 latency, tokens/s, and sequence lengths |
| 8F | Memory-footprint harness | Each model/precision/backend combination is run in a fresh process; CSV reports peak RSS and pass/fail under the 16 GB machine budget |
| 8G | Paper table/plot artifact generation | Scripts convert result CSV/JSON into paper-ready tables and plots without manually copying values |

#### Benchmark 1: Numerical Correctness

Purpose: prove that the C++/GGUF implementation preserves PyTorch embeddings
across model sizes and quantization levels.

Required precisions:
- F16: primary unquantized GGUF baseline.
- Q8_0: high-accuracy quantized baseline.
- Q4_K_M: recommended small model for general use.
- Q4_K_S: smaller/faster low-bit comparison.

Required model sizes:
- 300M: full local reference and all quantization levels.
- 600M: local reference if memory allows; otherwise one sequence at a time.
- 6B: quantized GGUF locally; PyTorch reference should be precomputed off-machine
  or stored as small reference slices.

Verification:
- F16 and Q8_0: per-residue mean cosine > 0.999, min cosine > 0.99.
- Q4_K_M and Q4_K_S: per-residue mean cosine > 0.995, min cosine recorded.
- Mean-pool relative L2 error is recorded for every run.
- Output artifacts: `results/correctness_<model>_<precision>.json` and `.csv`.

M1/16 GB rule:
- Never load PyTorch reference and large GGUF model in the same long-lived
  process for 600M or 6B.
- Run one sequence at a time for 600M+.
- For 6B, avoid local F16 unless memory measurements show a safe margin; use
  Q4_K_M/Q4_K_S first.

#### Benchmark 2: Downstream Task Quality

Purpose: show that quantized embeddings preserve biological utility, not just
vector cosine similarity.

Recommended order:
1. Start with a **small ProteinGym subset** for variant-effect prediction because
   Spearman correlation is easy to report and compare.
2. Cache embeddings to disk (`.npy` or `.npz`) so model inference and downstream
   scoring can be debugged independently.
3. Add FLIP function prediction only after ProteinGym pipeline is stable.

Configurations:
- PyTorch reference embeddings.
- GGUF F16 embeddings.
- GGUF Q8_0 embeddings.
- GGUF Q4_K_M embeddings.
- GGUF Q4_K_S embeddings.

Verification:
- Downstream CSV contains dataset name, split/subset, model size, precision,
  backend, metric value, and delta from PyTorch.
- ProteinGym: Spearman correlation delta from PyTorch is recorded; initial
  acceptable target is Δ ≤ 0.01 on the selected subset.
- FLIP: task-specific metric delta is recorded; acceptable tolerance must be
  fixed before making paper claims.
- Paper claim should only say "quality preserved" for settings that meet the
  predeclared tolerance.

M1/16 GB rule:
- Do not start with a full ProteinGym sweep.
- Pick 1-3 assays first, preferably with modest sequence lengths.
- Embed and score in streaming mode; keep cached embeddings on disk, not all in RAM.

#### Benchmark 3: Throughput

Purpose: measure sequences/second and latency on consumer Apple Silicon.

Required implementations:
- `esmc.cpp` CPU.
- `esmc.cpp` Metal.
- PyTorch CPU.
- PyTorch MPS.

Sequence buckets:
- Short: ~50 amino acids.
- Medium: ~250 amino acids.
- Long: ~1024 amino acids, if memory allows.
- Optional max-context: near 2048 amino acids, only if stable.

Verification:
- Each run reports median latency, p95 latency, seq/s, residues/s, backend,
  model size, precision, and sequence bucket.
- Warmup count and measured iteration count are recorded.
- PyTorch CPU/MPS baselines are run in separate processes from C++ runs.
- Output artifact: `results/throughput_<host>_<date>.csv`.

M1/16 GB rule:
- Avoid batch-size sweeps at first; benchmark batch size 1 before larger batches.
- Do not compare resident C++ and PyTorch models in the same process.
- For 6B, benchmark quantized GGUF only unless F16 memory is proven safe.

#### Benchmark 4: Memory Footprint

Purpose: support the claim that quantized 6B inference can fit on consumer
hardware that would otherwise require server-class GPU memory.

Measurements:
- Peak RSS for CPU runs.
- Peak process memory under Metal runs.
- Model file size on disk.
- Whether the run completed without OOM.

Recommended measurement methods on macOS:
- Use `/usr/bin/time -l <command>` for `maximum resident set size`.
- Run each configuration in a fresh process.
- Record host RAM and OS version in the manifest.

Verification:
- `results/memory_<host>_<date>.csv` contains model size, precision, backend,
  sequence bucket, peak RSS, model file size, success/failure, and notes.
- 300M and 600M must fit comfortably in F16 and quantized forms.
- 6B paper claim should be limited to quantization levels that complete under
  the 16 GB RAM limit with a documented sequence length.

#### Paper Benchmark Dataset Policy

Use small, fixed, versioned datasets first. Do not change benchmark datasets
after recording headline numbers without incrementing the result manifest.

Minimum dataset bundle:
- `benchmarks/sequences_correctness.fasta`: fixed short/medium/long correctness
  sequences.
- `benchmarks/sequences_throughput.fasta`: length-bucketed throughput sequences.
- `benchmarks/proteingym_subset/`: selected ProteinGym assays or download script.
- `benchmarks/manifest.json`: records exact dataset versions and checksums.

Verification:
- Every benchmark output includes dataset checksum or source commit/version.
- Re-running the benchmark on the same machine produces the same row identifiers.

#### Recommended 16 GB M1 Execution Order

1. 300M F16 correctness, throughput, and memory.
2. 300M Q8_0 / Q4_K_M / Q4_K_S correctness.
3. 300M downstream subset.
4. 600M F16 correctness one sequence at a time.
5. 600M quantized correctness + memory.
6. 6B Q4_K_M and Q4_K_S load + memory.
7. 6B quantized correctness using precomputed reference slices.
8. Only then attempt 6B F16, and only if memory measurements suggest it will not
   destabilize the machine.

---

## Part 14: Public API (final `esmc.h`)

```c
// esmc.h — Public C API for esmc.cpp

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Opaque types
typedef struct esmc_model   esmc_model;
typedef struct esmc_context esmc_context;

// --- Model parameters ---
typedef struct {
    bool use_mmap;       // memory-map the GGUF file (default: true)
    bool use_metal;      // use Metal GPU backend (default: true on Apple)
    int  n_threads;      // CPU threads (default: 4)
} esmc_model_params;

esmc_model_params esmc_default_model_params(void);

// --- Model lifecycle ---
esmc_model   * esmc_load_model  (const char * path, esmc_model_params params);
void           esmc_free_model  (esmc_model * model);
esmc_context * esmc_new_context (esmc_model * model);
void           esmc_free_context(esmc_context * ctx);

// --- Model metadata ---
int   esmc_n_embd  (const esmc_model * model);  // embedding dimension
int   esmc_n_vocab (const esmc_model * model);  // 33
int   esmc_n_ctx   (const esmc_model * model);  // 2048
const char * esmc_token_to_str(const esmc_model * model, int token_id);

// --- Tokenization ---
// Returns number of tokens written (including CLS and EOS), or -1 on error.
// tokens_out must be pre-allocated to at least (strlen(sequence) + 2) ints.
int esmc_tokenize(const esmc_model * model,
                  const char * sequence,
                  int32_t * tokens_out,
                  int32_t max_tokens);

// --- Inference ---
// Compute per-residue embeddings for a sequence.
// tokens: token IDs including CLS and EOS (from esmc_tokenize)
// n_tokens: total token count
// embeddings_out: preallocated float array of size [n_tokens * esmc_n_embd(model)]
//   Row i contains the embedding for token i (i=0 is CLS, i=n-1 is EOS)
// Returns 0 on success, negative on error.
int esmc_embed(esmc_context * ctx,
               const int32_t * tokens,
               int32_t n_tokens,
               float * embeddings_out);

// Compute mean-pooled sequence embedding (strips CLS and EOS automatically).
// embedding_out: preallocated float array of size [esmc_n_embd(model)]
int esmc_embed_mean(esmc_context * ctx,
                    const int32_t * tokens,
                    int32_t n_tokens,
                    float * embedding_out);

// Score a masked position (log probability of each amino acid at position pos).
// Useful for zero-shot variant effect prediction.
// logits_out: preallocated float array of size [esmc_n_vocab(model)]
int esmc_score_position(esmc_context * ctx,
                        const int32_t * tokens,
                        int32_t n_tokens,
                        int32_t mask_position,
                        float * logits_out);

#ifdef __cplusplus
}
#endif
```

---

## Appendix: Quick Reference Checklist for the Implementing Engineer

Before declaring any milestone complete, check all of these:

**Architecture:**
- [ ] SwiGLU used (NOT GELU) — three weight matrices `ffn_gate`, `ffn_up`, `ffn_down`
- [ ] NO biases anywhere: not in Q/K/V/O projections, not in FFN, not in LayerNorm
- [ ] Standard LayerNorm (not RMSNorm) — weight only, no bias term
- [ ] NO `emb_layer_norm_before` / `emb_layer_norm_after` (those were ESM2 only)
- [ ] Single final LayerNorm after all transformer layers
- [ ] RoPE NEOX style (not GPT-2 style) applied to Q and K
- [ ] Q scaled by `1/sqrt(head_dim)` BEFORE RoPE, not after matmul
- [ ] No causal mask — bidirectional attention
- [ ] Context length = 2048 (not 1024)

**Conversion:**
- [ ] `ffn_dim` read from actual weight shape, stored in GGUF, read back in C++
- [ ] No bias tensors emitted to GGUF (converter explicitly skips them)
- [ ] Token alphabet order is exact (CLS=0, PAD=1, EOS=2, ...)
- [ ] LayerNorm weights stored as F32 in GGUF regardless of quantization target

**Validation:**
- [ ] Per-residue cosine similarity > 0.999 for F16 model
- [ ] Mean-pool embedding L2 error < 1% for F16 model
- [ ] Per-residue cosine similarity > 0.995 for Q4_K_M model
- [ ] Token IDs match official `ESMC.encode()` output exactly
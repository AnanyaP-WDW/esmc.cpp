#!/usr/bin/env python3
"""Convert ESM-C weights (native esmc.* or HuggingFace) to GGUF for esmc.cpp."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("Install: pip install safetensors numpy", file=sys.stderr)
    sys.exit(1)

# gguf-py from llama.cpp submodule
GGUF_PY = Path(__file__).resolve().parent.parent / "ggml" / "gguf-py"
if GGUF_PY.is_dir():
    sys.path.insert(0, str(GGUF_PY))

from gguf import GGUFWriter  # noqa: E402

# Token alphabet — must match the order in esmc-300m/tokenizer.json exactly,
# because the embedding table rows are indexed by these IDs.
# Note: Q (16) and N (17) are NOT in alphabetical order — this matches the
# official ESM-C tokenizer; flipping them silently breaks embeddings for any
# sequence containing N or Q.
ESMC_TOKENS = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",
    "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
    "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>",
]
ESMC_TOKEN_TYPES = [3, 3, 3, 3] + [1] * 20 + [1] * 5 + [1, 1, 3]

ESMC_CONFIGS = {
    "300m": {"n_layers": 30, "d_model": 960, "n_heads": 15, "rope_theta": 10000.0},
    "600m": {"n_layers": 36, "d_model": 1152, "n_heads": 18, "rope_theta": 10000.0},
    "6b": {"n_layers": 80, "d_model": 2560, "n_heads": 40, "rope_theta": 10000.0},
}

D_MODEL_TO_SIZE = {v["d_model"]: k for k, v in ESMC_CONFIGS.items()}

def load_state_dict(model_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            for key in f.keys():
                out[key] = f.get_tensor(key)
    return out


def detect_layout(tensors: dict[str, np.ndarray]) -> str:
    if any(k.startswith("esmc.") for k in tensors):
        return "esmc_native"
    if any("embed_tokens" in k for k in tensors):
        return "hf_transformers"
    raise ValueError("Unknown checkpoint layout")


def detect_size(tensors: dict[str, np.ndarray], layout: str, config: dict) -> str:
    if "d_model" in config and config["d_model"] in D_MODEL_TO_SIZE:
        return D_MODEL_TO_SIZE[config["d_model"]]
    if layout == "esmc_native":
        d = tensors["esmc.embed.weight"].shape[1]
    else:
        emb = next(v for k, v in tensors.items() if "embed_tokens" in k)
        d = emb.shape[1]
    return D_MODEL_TO_SIZE[d]


def to_ggml_linear(w: np.ndarray) -> np.ndarray:
    """PyTorch Linear [out, in] -> ggml [in, out]."""
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight, got {w.shape}")
    return np.ascontiguousarray(w.T)


def to_dtype(arr: np.ndarray, dtype: str, force_f32: bool = False) -> np.ndarray:
    if force_f32:
        return np.ascontiguousarray(arr.astype(np.float32))
    if dtype == "f16":
        return np.ascontiguousarray(arr.astype(np.float16))
    return np.ascontiguousarray(arr.astype(np.float32))


def convert_native(
    tensors: dict[str, np.ndarray],
    cfg: dict,
    ffn_dim: int,
    vocab_size: int,
    dtype: str,
) -> dict[str, np.ndarray]:
    """Map native esmc.* checkpoint to canonical GGUF tensor names."""
    d = cfg["d_model"]
    n_layers = cfg["n_layers"]
    out: dict[str, np.ndarray] = {}

    embed = tensors["esmc.embed.weight"][:vocab_size]
    # Keep [vocab, d_model]: GGUF stores dims reversed into ggml ne (ne[0]=d_model, ne[1]=vocab)
    out["token_embd.weight"] = to_dtype(np.ascontiguousarray(embed), dtype)
    norm_key = "esmc.transformer.norm.weight"
    if norm_key not in tensors:
        norm_key = "esmc.norm.weight"
    out["output_norm.weight"] = tensors[norm_key].astype(np.float32)

    lm_key = "lm_head.3.weight"
    if lm_key in tensors:
        out["output.weight"] = to_dtype(
            np.ascontiguousarray(tensors[lm_key][:vocab_size]), dtype
        )

    for i in range(n_layers):
        p = f"esmc.transformer.blocks.{i}"
        out[f"blk.{i}.attn_norm.weight"] = tensors[f"{p}.attn.layernorm_qkv.layer_norm_weight"].astype(
            np.float32
        )
        out[f"blk.{i}.attn_norm.bias"] = tensors[f"{p}.attn.layernorm_qkv.layer_norm_bias"].astype(
            np.float32
        )
        out[f"blk.{i}.attn_q_norm.weight"] = tensors[f"{p}.attn.q_ln.weight"].astype(np.float32)
        out[f"blk.{i}.attn_k_norm.weight"] = tensors[f"{p}.attn.k_ln.weight"].astype(np.float32)
        out[f"blk.{i}.ffn_norm.weight"] = tensors[f"{p}.ffn.layer_norm_weight"].astype(np.float32)

        qkv = tensors[f"{p}.attn.layernorm_qkv.weight"]
        wq, wk, wv = np.split(qkv, 3, axis=0)
        out[f"blk.{i}.attn_q.weight"] = to_dtype(np.ascontiguousarray(wq), dtype)
        out[f"blk.{i}.attn_k.weight"] = to_dtype(np.ascontiguousarray(wk), dtype)
        out[f"blk.{i}.attn_v.weight"] = to_dtype(np.ascontiguousarray(wv), dtype)
        out[f"blk.{i}.attn_output.weight"] = to_dtype(
            np.ascontiguousarray(tensors[f"{p}.attn.out_proj.weight"]), dtype
        )

        fc1 = tensors[f"{p}.ffn.fc1_weight"]
        gate, up = np.split(fc1, 2, axis=0)
        out[f"blk.{i}.ffn_gate.weight"] = to_dtype(np.ascontiguousarray(gate), dtype)
        out[f"blk.{i}.ffn_up.weight"] = to_dtype(np.ascontiguousarray(up), dtype)
        out[f"blk.{i}.ffn_down.weight"] = to_dtype(
            np.ascontiguousarray(tensors[f"{p}.ffn.fc2_weight"]), dtype
        )

    return {k: to_dtype(v, dtype, force_f32="norm" in k) for k, v in out.items()}


def hf_to_gguf_name(hf_name: str, n_layers: int) -> str | None:
    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        if hf_name.startswith(prefix):
            suffix = hf_name[len(prefix) :]
            return {
                "self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
                "self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
                "self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
                "self_attn.out_proj.weight": f"blk.{i}.attn_output.weight",
                "self_attn_layer_norm.weight": f"blk.{i}.attn_norm.weight",
                "final_layer_norm.weight": f"blk.{i}.ffn_norm.weight",
                "ffn.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
                "ffn.up_proj.weight": f"blk.{i}.ffn_up.weight",
                "ffn.down_proj.weight": f"blk.{i}.ffn_down.weight",
            }.get(suffix)
    return {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }.get(hf_name)


def convert_hf(
    tensors: dict[str, np.ndarray],
    cfg: dict,
    dtype: str,
) -> dict[str, np.ndarray]:
    n_layers = cfg["n_layers"]
    out: dict[str, np.ndarray] = {}
    for hf_name, arr in tensors.items():
        gguf_name = hf_to_gguf_name(hf_name, n_layers)
        if gguf_name is None:
            continue
        if arr.ndim == 2 and "norm" not in gguf_name:
            arr = np.ascontiguousarray(arr)
        force_f32 = "norm" in gguf_name
        out[gguf_name] = to_dtype(arr, dtype, force_f32=force_f32)
    return out


def native_ffn_dim(tensors: dict[str, np.ndarray]) -> int:
    return int(tensors["esmc.transformer.blocks.0.ffn.fc2_weight"].shape[1])


def hf_ffn_dim(tensors: dict[str, np.ndarray], n_layers: int) -> int:
    key = f"model.layers.0.ffn.gate_proj.weight"
    if key not in tensors:
        key = next(k for k in tensors if "gate_proj" in k and "layers.0" in k)
    return int(tensors[key].shape[0])


def write_gguf(
    gguf_tensors: dict[str, np.ndarray],
    output_path: Path,
    cfg: dict,
    size: str,
    ffn_dim: int,
    vocab_size: int,
) -> None:
    writer = GGUFWriter(str(output_path), "esmc")
    n_layers = cfg["n_layers"]

    writer.add_uint32("esmc.block_count", n_layers)
    writer.add_uint32("esmc.context_length", 2048)
    writer.add_uint32("esmc.embedding_length", cfg["d_model"])
    writer.add_uint32("esmc.feed_forward_length", ffn_dim)
    writer.add_uint32("esmc.attention.head_count", cfg["n_heads"])
    writer.add_uint32("esmc.attention.head_count_kv", cfg["n_heads"])
    writer.add_float32("esmc.rope.freq_base", cfg["rope_theta"])
    writer.add_uint32("esmc.vocab_size", vocab_size)
    writer.add_float32("esmc.attention.layer_norm_epsilon", 1e-5)
    writer.add_string("general.name", f"ESM-C {size.upper()}")

    writer.add_array("tokenizer.ggml.tokens", ESMC_TOKENS[:vocab_size])
    writer.add_array("tokenizer.ggml.token_type", ESMC_TOKEN_TYPES[:vocab_size])
    writer.add_uint32("tokenizer.ggml.bos_token_id", 0)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 2)
    writer.add_uint32("tokenizer.ggml.padding_token_id", 1)
    writer.add_uint32("tokenizer.ggml.mask_token_id", vocab_size - 1)

    for name in sorted(gguf_tensors.keys()):
        writer.add_tensor(name, gguf_tensors[name])
        print(f"  {name:45s} {list(gguf_tensors[name].shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def convert(model_dir: Path, output_path: Path, dtype: str = "f16") -> None:
    config: dict = {}
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    tensors = load_state_dict(model_dir)
    layout = detect_layout(tensors)
    size = detect_size(tensors, layout, config)
    cfg = ESMC_CONFIGS[size]
    vocab_size = len(ESMC_TOKENS)

    if layout == "esmc_native":
        ffn_dim = native_ffn_dim(tensors)
        gguf_tensors = convert_native(tensors, cfg, ffn_dim, vocab_size, dtype)
    else:
        ffn_dim = hf_ffn_dim(tensors, cfg["n_layers"])
        gguf_tensors = convert_hf(tensors, cfg, dtype)

    print(
        f"Converting layout={layout} size={size} "
        f"d_model={cfg['d_model']} n_layers={cfg['n_layers']} "
        f"ffn_dim={ffn_dim} vocab={vocab_size} -> {output_path}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(gguf_tensors, output_path, cfg, size, ffn_dim, vocab_size)
    print(f"\nWrote {len(gguf_tensors)} tensors to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dtype", default="f16", choices=["f16", "f32"])
    args = parser.parse_args()
    convert(args.model_dir.resolve(), args.output.resolve(), args.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())

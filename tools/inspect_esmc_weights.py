#!/usr/bin/env python3
"""Inspect ESM-C weights and verify shapes against the architecture spec (plan §1.1)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("Install: pip install -r tools/requirements.txt", file=sys.stderr)
    sys.exit(1)

# §1.1 — structural hparams (ffn_dim read from weights)
ESMC_ARCH = {
    "300m": {"n_layers": 30, "d_model": 960, "n_heads": 15, "n_ctx": 2048},
    "600m": {"n_layers": 36, "d_model": 1152, "n_heads": 18, "n_ctx": 2048},
    "6b": {"n_layers": 80, "d_model": 2560, "n_heads": 40, "n_ctx": 2048},
}

D_MODEL_TO_SIZE = {v["d_model"]: k for k, v in ESMC_ARCH.items()}


def load_state_dict(model_dir: Path) -> dict[str, tuple[list[int], str]]:
    tensors: dict[str, tuple[list[int], str]] = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                tensors[key] = (list(t.shape), str(t.dtype))
    return tensors


def detect_layout(tensors: dict[str, tuple]) -> str:
    if any(k.startswith("model.layers.") or k.startswith("encoder.layers.") for k in tensors):
        return "hf_transformers"
    if any(k.startswith("esmc.") for k in tensors):
        return "esmc_native"
    if any("embed_tokens" in k for k in tensors):
        return "hf_transformers"
    raise ValueError("Unknown weight layout (expected HuggingFace or esmc.* native keys)")


def detect_size_from_config(config: dict) -> str | None:
    d_model = config.get("d_model") or config.get("hidden_size")
    if d_model in D_MODEL_TO_SIZE:
        return D_MODEL_TO_SIZE[d_model]
    return None


def detect_size_from_tensors(tensors: dict[str, tuple], layout: str) -> str:
    if layout == "hf_transformers":
        for key, (shape, _) in tensors.items():
            if "embed_tokens" in key and len(shape) == 2:
                d_model = shape[1]
                if d_model in D_MODEL_TO_SIZE:
                    return D_MODEL_TO_SIZE[d_model]
    else:
        for key, (shape, _) in tensors.items():
            if key.endswith("embed.weight") and len(shape) == 2:
                d_model = shape[1]
                if d_model in D_MODEL_TO_SIZE:
                    return D_MODEL_TO_SIZE[d_model]
    raise ValueError("Could not detect model size from embedding table")


def block_indices_native(tensors: dict[str, tuple]) -> list[int]:
    indices = set()
    for key in tensors:
        m = re.search(r"esmc\.transformer\.blocks\.(\d+)\.", key)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def validate_hf_transformers(
    tensors: dict[str, tuple], cfg: dict, ffn_dim: int, n_layers: int
) -> list[str]:
    errors: list[str] = []

    def classify(hf_name: str) -> str | None:
        if hf_name in ("model.embed_tokens.weight", "encoder.embed_tokens.weight"):
            return "embed"
        if hf_name in ("model.norm.weight", "encoder.norm.weight"):
            return "final_norm"
        if hf_name == "lm_head.weight":
            return "lm_head"
        for i in range(n_layers):
            for prefix in (f"model.layers.{i}.", f"encoder.layers.{i}."):
                if hf_name.startswith(prefix):
                    suffix = hf_name[len(prefix) :]
                    return {
                        "self_attn.q_proj.weight": "attn_q",
                        "self_attn.k_proj.weight": "attn_k",
                        "self_attn.v_proj.weight": "attn_v",
                        "self_attn.out_proj.weight": "attn_out",
                        "self_attn_layer_norm.weight": "attn_norm",
                        "final_layer_norm.weight": "ffn_norm",
                        "ffn.gate_proj.weight": "ffn_gate",
                        "ffn.up_proj.weight": "ffn_up",
                        "ffn.down_proj.weight": "ffn_down",
                    }.get(suffix)
        return None

    def expected_shape(kind: str) -> list[int]:
        d, v = cfg["d_model"], 33
        table = {
            "embed": [v, d],
            "final_norm": [d],
            "lm_head": [v, d],
            "attn_q": [d, d],
            "attn_k": [d, d],
            "attn_v": [d, d],
            "attn_out": [d, d],
            "attn_norm": [d],
            "ffn_norm": [d],
            "ffn_gate": [ffn_dim, d],
            "ffn_up": [ffn_dim, d],
            "ffn_down": [d, ffn_dim],
        }
        return table[kind]

    expected_counts = {
        "embed": 1,
        "final_norm": 1,
        "lm_head": 1,
        **{k: n_layers for k in (
            "attn_q", "attn_k", "attn_v", "attn_out",
            "attn_norm", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down",
        )},
    }
    counts: dict[str, int] = {}
    for hf_name in tensors:
        kind = classify(hf_name)
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
        elif "bias" in hf_name:
            errors.append(f"unexpected bias (ESM-C has no biases): {hf_name}")
        else:
            errors.append(f"unrecognized tensor: {hf_name}")

    for kind, exp in expected_counts.items():
        if counts.get(kind, 0) != exp:
            errors.append(f"count {kind}: expected {exp}, got {counts.get(kind, 0)}")

    for hf_name, (shape, _) in tensors.items():
        kind = classify(hf_name)
        if kind and shape != expected_shape(kind):
            errors.append(f"{hf_name}: {shape} != {expected_shape(kind)}")

    return errors


def validate_esmc_native(tensors: dict[str, tuple], cfg: dict) -> tuple[list[str], int]:
    """Validate EvolutionaryScale native checkpoint (biohub / legacy). Returns (errors, ffn_dim)."""
    errors: list[str] = []
    d = cfg["d_model"]
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    head_dim = d // n_heads

    blocks = block_indices_native(tensors)
    if len(blocks) != n_layers:
        errors.append(f"n_layers: expected {n_layers} blocks, found {len(blocks)}")
    if blocks and max(blocks) != n_layers - 1:
        errors.append(f"block indices: expected 0..{n_layers - 1}, got max {max(blocks)}")

    embed_key = "esmc.embed.weight"
    if embed_key not in tensors:
        errors.append(f"missing {embed_key}")
    else:
        vocab, d_model = tensors[embed_key][0]
        if d_model != d:
            errors.append(f"{embed_key}: d_model {d_model} != {d}")

    ffn_dim = None
    fc2_key = "esmc.transformer.blocks.0.ffn.fc2_weight"
    if fc2_key in tensors:
        ffn_dim = tensors[fc2_key][0][1]
    else:
        errors.append(f"missing {fc2_key}")

    fc1_key = "esmc.transformer.blocks.0.ffn.fc1_weight"
    if fc1_key in tensors and ffn_dim is not None:
        fc1_rows = tensors[fc1_key][0][0]
        if fc1_rows != 2 * ffn_dim:
            errors.append(
                f"{fc1_key}: rows {fc1_rows} != 2*ffn_dim ({2 * ffn_dim}); "
                "expected fused gate+up for SwiGLU"
            )

    qkv_key = "esmc.transformer.blocks.0.attn.layernorm_qkv.weight"
    if qkv_key in tensors:
        rows, cols = tensors[qkv_key][0]
        if rows != 3 * d or cols != d:
            errors.append(f"{qkv_key}: shape {tensors[qkv_key][0]} != [{3 * d}, {d}] (fused QKV)")

    per_layer_required = [
        "attn.layernorm_qkv.weight",
        "attn.layernorm_qkv.layer_norm_weight",
        "attn.out_proj.weight",
        "attn.q_ln.weight",
        "attn.k_ln.weight",
        "ffn.layer_norm_weight",
        "ffn.fc1_weight",
        "ffn.fc2_weight",
    ]
    for i in range(n_layers):
        for suffix in per_layer_required:
            key = f"esmc.transformer.blocks.{i}.{suffix}"
            if key not in tensors:
                errors.append(f"missing {key}")

    for suffix, exp_shape in [
        ("attn.out_proj.weight", [d, d]),
        ("attn.q_ln.weight", [d]),
        ("attn.k_ln.weight", [d]),
        ("ffn.layer_norm_weight", [d]),
        ("ffn.fc2_weight", [d, ffn_dim] if ffn_dim else None),
    ]:
        key = f"esmc.transformer.blocks.0.{suffix}"
        if key in tensors and exp_shape and tensors[key][0] != exp_shape:
            errors.append(f"{key}: {tensors[key][0]} != {exp_shape}")

    if head_dim * n_heads != d:
        errors.append(f"head_dim: {d} not divisible by n_heads={n_heads}")

    return errors, ffn_dim or 0


def hf_ffn_dim(tensors: dict[str, tuple], n_layers: int) -> int:
    for i in range(n_layers):
        for pattern in (
            f"model.layers.{i}.ffn.gate_proj.weight",
            f"encoder.layers.{i}.ffn.gate_proj.weight",
        ):
            if pattern in tensors:
                return tensors[pattern][0][0]
    for key in tensors:
        if "gate_proj" in key and "layers.0" in key:
            return tensors[key][0][0]
    raise ValueError("Could not find gate_proj.weight for ffn_dim")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Directory with config.json and *.safetensors")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()

    if not model_dir.is_dir():
        print(f"Error: not a directory: {model_dir}", file=sys.stderr)
        return 1

    config: dict = {}
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print("config.json:")
        for k in (
            "model_type",
            "d_model",
            "hidden_size",
            "n_layers",
            "num_hidden_layers",
            "n_heads",
            "num_attention_heads",
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
        ):
            if k in config:
                print(f"  {k}: {config[k]}")
        print()

    sf_files = list(model_dir.glob("*.safetensors"))
    if not sf_files:
        print(f"Error: no *.safetensors in {model_dir}", file=sys.stderr)
        return 1

    tensors = load_state_dict(model_dir)
    layout = detect_layout(tensors)
    print(f"Layout: {layout}")

    size = detect_size_from_config(config) or detect_size_from_tensors(tensors, layout)
    cfg = ESMC_ARCH[size]

    if config.get("n_layers"):
        if int(config["n_layers"]) != cfg["n_layers"]:
            print(f"Warning: config n_layers={config['n_layers']} != spec {cfg['n_layers']}")
    if config.get("n_heads"):
        if int(config["n_heads"]) != cfg["n_heads"]:
            print(f"Warning: config n_heads={config['n_heads']} != spec {cfg['n_heads']}")

    if layout == "hf_transformers":
        ffn_dim = hf_ffn_dim(tensors, cfg["n_layers"])
        errors = validate_hf_transformers(tensors, cfg, ffn_dim, cfg["n_layers"])
    else:
        errors, ffn_dim = validate_esmc_native(tensors, cfg)

    ratio = ffn_dim / cfg["d_model"] if ffn_dim else 0.0
    print(f"Detected model: ESM-C {size}")
    print(f"  n_layers={cfg['n_layers']}  d_model={cfg['d_model']}  n_heads={cfg['n_heads']}")
    print(f"  head_dim={cfg['d_model'] // cfg['n_heads']}  ffn_dim={ffn_dim}  (ratio {ratio:.4f}, ~8/3)")
    print(f"  n_tensors={len(tensors)}")
    print()

    print("All tensors:")
    print(f"{'name':<80} {'shape':<30} dtype")
    print("-" * 120)
    for key in sorted(tensors.keys()):
        shape, dtype = tensors[key]
        print(f"{key:<80} {str(shape):<30} {dtype}")
    print()

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    if layout == "esmc_native":
        print(
            "VALIDATION OK: native esmc.* layout matches §1.1 hparams "
            "(30L / 960d / 15 heads / ffn≈2560). "
            "Converter must map fused QKV + fc1, not HuggingFace model.layers.* names."
        )
    else:
        print("VALIDATION OK: HuggingFace layout matches §1.1 / §2.1 tensor shapes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

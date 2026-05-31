"""Reference ESM-C forward pass from native safetensors (Biohub layout)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "esmc-300m" / "model.safetensors"
DEFAULT_TOKENIZER = ROOT / "esmc-300m" / "tokenizer.json"

TEST_SEQUENCES: list[tuple[str, str]] = [
    ("short", "ACDEFGHIK"),
    ("medium", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGY"),
    ("long", "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQ" * 5),
]


def tokenize(sequence: str, tokenizer_path: Path = DEFAULT_TOKENIZER) -> list[int]:
    with open(tokenizer_path) as f:
        vocab = json.load(f)["model"]["vocab"]
    ids = [vocab["<cls>"]]
    for c in sequence.upper():
        ids.append(vocab.get(c, vocab["<unk>"]))
    ids.append(vocab["<eos>"])
    return ids


def load_state_dict(weights_path: Path = DEFAULT_WEIGHTS) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(weights_path), framework="pt") as f:
        for key in f.keys():
            sd[key] = f.get_tensor(key)
    return sd


def _rope_neox(q: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """q: [batch, heads, seq, head_dim]."""
    _, _, _, dh = q.shape
    half = dh // 2
    inv = 1.0 / (theta ** (torch.arange(0, dh, 2, device=q.device, dtype=torch.float32) / dh))
    t = torch.arange(q.shape[2], device=q.device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    cos = torch.cos(freqs).to(dtype=q.dtype)[None, None, :, :]
    sin = torch.sin(freqs).to(dtype=q.dtype)[None, None, :, :]
    q1, q2 = q[..., :half], q[..., half:]
    return torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)


def forward(
    sd: dict[str, torch.Tensor],
    token_ids: list[int] | torch.Tensor,
    *,
    n_layers: int = 30,
    d_model: int = 960,
    n_heads: int = 15,
    eps: float = 1e-5,
    rope_theta: float = 10000.0,
    residue_scale: float | None = None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Returns embeddings [seq_len, d_model] including CLS/EOS positions."""
    if residue_scale is None:
        residue_scale = math.sqrt(n_layers / 36.0)
    if device is None:
        first_tensor = next(iter(sd.values()), None)
        device = first_tensor.device if first_tensor is not None else None

    tokens = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    x = sd["esmc.embed.weight"][tokens].unsqueeze(0)  # [1, L, D]
    head_dim = d_model // n_heads

    for i in range(n_layers):
        p = f"esmc.transformer.blocks.{i}"
        residual = x

        h = F.layer_norm(
            x,
            (d_model,),
            sd[f"{p}.attn.layernorm_qkv.layer_norm_weight"],
            sd[f"{p}.attn.layernorm_qkv.layer_norm_bias"],
            eps,
        )
        qkv = F.linear(h, sd[f"{p}.attn.layernorm_qkv.weight"])
        q, k, v = qkv.chunk(3, dim=-1)
        q = F.layer_norm(q, (d_model,), sd[f"{p}.attn.q_ln.weight"], None, eps)
        k = F.layer_norm(k, (d_model,), sd[f"{p}.attn.k_ln.weight"], None, eps)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(1, -1, n_heads, head_dim).transpose(1, 2)

        q, k, v = map(to_heads, (q, k, v))
        q, k = _rope_neox(q, rope_theta), _rope_neox(k, rope_theta)
        ctx = F.scaled_dot_product_attention(q, k, v)
        ctx = ctx.transpose(1, 2).reshape(1, -1, d_model)
        attn_out = F.linear(ctx, sd[f"{p}.attn.out_proj.weight"])
        x = residual + attn_out / residue_scale

        residual = x
        h = F.layer_norm(
            x,
            (d_model,),
            sd[f"{p}.ffn.layer_norm_weight"],
            None,
            eps,
        )
        fc1 = sd[f"{p}.ffn.fc1_weight"]
        mid = F.linear(h, fc1)
        h1, h2 = mid.chunk(2, dim=-1)
        mid = F.silu(h1) * h2
        ffn_out = F.linear(mid, sd[f"{p}.ffn.fc2_weight"])
        x = residual + ffn_out / residue_scale

    x = F.layer_norm(
        x,
        (d_model,),
        sd["esmc.transformer.norm.weight"],
        None,
        eps,
    )
    return x.squeeze(0).detach().cpu().numpy().astype(np.float32)

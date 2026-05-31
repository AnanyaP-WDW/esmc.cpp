"""NumPy forward pass using GGUF weights (mirrors esmc-graph.cpp)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GGUF_PY = ROOT / "ggml" / "gguf-py"
import sys

if GGUF_PY.is_dir():
    sys.path.insert(0, str(GGUF_PY))

from gguf import GGUFReader  # noqa: E402


def load_f32(reader: GGUFReader, name: str) -> np.ndarray:
    for t in reader.tensors:
        if t.name == name:
            return t.data.astype(np.float32)
    raise KeyError(name)


def ggml_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None, eps: float
) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    x = x * weight[:, None]
    if bias is not None:
        x = x + bias[:, None]
    return x


def ggml_mul(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """PyTorch [out, in] @ [in, seq] (x is [d_model, n_tokens])."""
    if w.shape[1] == x.shape[0]:
        return w @ x
    if w.shape[0] == x.shape[0]:
        return w @ x
    raise ValueError(f"incompatible shapes w={w.shape} x={x.shape}")


def rope_neox(q: np.ndarray, head_dim: int, n_heads: int, theta: float) -> np.ndarray:
    _, _, n_tokens = q.shape
    half = head_dim // 2
    out = q.copy()
    for pos in range(n_tokens):
        for h in range(n_heads):
            for i in range(half):
                freq = 1.0 / (theta ** (2.0 * i / head_dim))
                angle = pos * freq
                c, s = np.cos(angle), np.sin(angle)
                a0, a1 = out[i, h, pos], out[i + half, h, pos]
                out[i, h, pos] = a0 * c - a1 * s
                out[i + half, h, pos] = a0 * s + a1 * c
    return out


def forward_gguf(reader: GGUFReader, tokens: list[int]) -> np.ndarray:
    eps = float(reader.fields["esmc.attention.layer_norm_epsilon"].contents())
    theta = float(reader.fields["esmc.rope.freq_base"].contents())
    n_heads = int(reader.fields["esmc.attention.head_count"].contents())
    d_model = int(reader.fields["esmc.embedding_length"].contents())
    n_layers = int(reader.fields["esmc.block_count"].contents())
    head_dim = d_model // n_heads
    scale = math.sqrt(n_layers / 36.0)

    emb = load_f32(reader, "token_embd.weight")
    if emb.shape[0] == d_model:
        x = emb[:, tokens].astype(np.float32)
    else:
        x = emb[tokens, :].T.astype(np.float32)

    for il in range(n_layers):
        residual = x
        p = f"blk.{il}"
        x = ggml_norm(
            x,
            load_f32(reader, f"{p}.attn_norm.weight"),
            load_f32(reader, f"{p}.attn_norm.bias"),
            eps,
        )
        Q = ggml_mul(load_f32(reader, f"{p}.attn_q.weight"), x)
        K = ggml_mul(load_f32(reader, f"{p}.attn_k.weight"), x)
        V = ggml_mul(load_f32(reader, f"{p}.attn_v.weight"), x)
        Q = ggml_norm(Q, load_f32(reader, f"{p}.attn_q_norm.weight"), None, eps)
        K = ggml_norm(K, load_f32(reader, f"{p}.attn_k_norm.weight"), None, eps)

        Q = Q.reshape(head_dim, n_heads, len(tokens)) / np.sqrt(head_dim)
        K = K.reshape(head_dim, n_heads, len(tokens))
        V = V.reshape(head_dim, n_heads, len(tokens))
        Q = rope_neox(Q, head_dim, n_heads, theta)
        K = rope_neox(K, head_dim, n_heads, theta)

        # attention: K @ Q -> softmax -> V @ sm
        for h in range(n_heads):
            kq = K[:, h, :].T @ Q[:, h, :]  # [L,L]
            sm = np.exp(kq - kq.max(axis=-1, keepdims=True))
            sm /= sm.sum(axis=-1, keepdims=True)
            V[:, h, :] = V[:, h, :] @ sm

        cur = V.transpose(0, 2, 1).reshape(d_model, len(tokens))
        cur = ggml_mul(load_f32(reader, f"{p}.attn_output.weight"), cur)
        x = residual + cur / scale

        residual = x
        x = ggml_norm(x, load_f32(reader, f"{p}.ffn_norm.weight"), None, eps)
        gate = ggml_mul(load_f32(reader, f"{p}.ffn_gate.weight"), x)
        up = ggml_mul(load_f32(reader, f"{p}.ffn_up.weight"), x)
        mid = (gate / (1 + np.exp(-gate))) * up
        cur = ggml_mul(load_f32(reader, f"{p}.ffn_down.weight"), mid)
        x = residual + cur / scale

    x = ggml_norm(x, load_f32(reader, "output_norm.weight"), None, eps)
    return x.T.astype(np.float32)

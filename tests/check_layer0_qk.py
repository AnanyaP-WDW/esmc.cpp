#!/usr/bin/env python3
"""Milestone 5: layer-0 Q/K L2 norms within 5% of numpy reference from GGUF weights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GGUF_PY = ROOT / "ggml" / "gguf-py"
if GGUF_PY.is_dir():
    sys.path.insert(0, str(GGUF_PY))

from gguf import GGUFReader  # noqa: E402

EMBED = ROOT / "build" / "esmc-embed"
MODEL = ROOT / "models" / "esmc-300m-f16.gguf"
SEQUENCE = "ACDEF"
TOLERANCE = 0.05


def load_f32(reader: GGUFReader, name: str) -> np.ndarray:
    for t in reader.tensors:
        if t.name == name:
            return t.data.astype(np.float32)
    raise KeyError(name)


def ggml_norm(
    x: np.ndarray, eps: float, weight: np.ndarray, bias: np.ndarray | None = None
) -> np.ndarray:
    """LayerNorm over dim 0 for x shape [d_model, n_tokens]."""
    mean = x.mean(axis=0, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    x = x * weight[:, None]
    if bias is not None:
        x = x + bias[:, None]
    return x


def ggml_mul(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """PyTorch [out, in] @ [in, seq] with x shaped [d_model, n_tokens]."""
    if w.shape[1] == x.shape[0]:
        return w @ x
    if w.shape[0] == x.shape[0]:
        return w @ x
    raise ValueError(f"incompatible shapes w={w.shape} x={x.shape}")


def rope_neox(q: np.ndarray, head_dim: int, n_heads: int, theta: float) -> np.ndarray:
    """q shape [head_dim, n_heads, n_tokens]."""
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


def reference_qk_norms(reader: GGUFReader, tokens: list[int]) -> tuple[float, float]:
    eps = float(reader.fields["esmc.attention.layer_norm_epsilon"].contents())
    theta = float(reader.fields["esmc.rope.freq_base"].contents())
    n_heads = int(reader.fields["esmc.attention.head_count"].contents())
    d_model = int(reader.fields["esmc.embedding_length"].contents())
    head_dim = d_model // n_heads

    emb = load_f32(reader, "token_embd.weight")
    attn_norm = load_f32(reader, "blk.0.attn_norm.weight")
    attn_bias = load_f32(reader, "blk.0.attn_norm.bias")
    q_ln = load_f32(reader, "blk.0.attn_q_norm.weight")
    k_ln = load_f32(reader, "blk.0.attn_k_norm.weight")
    wq = load_f32(reader, "blk.0.attn_q.weight")
    wk = load_f32(reader, "blk.0.attn_k.weight")

    if emb.shape[0] == d_model:
        x = emb[:, tokens].astype(np.float32)
    else:
        x = emb[tokens, :].T.astype(np.float32)
    x = ggml_norm(x, eps, attn_norm, attn_bias)
    Q = ggml_mul(wq, x)
    K = ggml_mul(wk, x)
    Q = ggml_norm(Q, eps, q_ln)
    K = ggml_norm(K, eps, k_ln)
    scale = 1.0 / np.sqrt(head_dim)
    Q = Q.reshape(head_dim, n_heads, len(tokens)) * scale
    K = K.reshape(head_dim, n_heads, len(tokens))
    Q = rope_neox(Q, head_dim, n_heads, theta)
    K = rope_neox(K, head_dim, n_heads, theta)

    q_norm = float(np.linalg.norm(Q))
    k_norm = float(np.linalg.norm(K))
    return q_norm, k_norm


def cpp_qk_norms(sequence: str) -> tuple[float, float]:
    proc = subprocess.run(
        [
            str(EMBED),
            "-m",
            str(MODEL),
            "--check-layer0-qk",
            "-s",
            sequence,
            "--no-metal",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(proc.stdout + proc.stderr)

    q_norm = k_norm = None
    for line in proc.stdout.splitlines():
        if line.startswith("q_norm:"):
            q_norm = float(line.split(":")[1])
        if line.startswith("k_norm:"):
            k_norm = float(line.split(":")[1])
    if q_norm is None or k_norm is None:
        raise RuntimeError(f"missing norms in:\n{proc.stdout}")
    return q_norm, k_norm


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / (abs(b) + 1e-9)


def main() -> int:
    if not MODEL.exists():
        print(f"Missing {MODEL}", file=sys.stderr)
        return 1
    if not EMBED.exists():
        print(f"Missing {EMBED} — run cmake --build build", file=sys.stderr)
        return 1

    with open(ROOT / "esmc-300m" / "tokenizer.json") as f:
        import json

        vocab = json.load(f)["model"]["vocab"]
    tokens = [vocab["<cls>"]] + [vocab[c] for c in SEQUENCE] + [vocab["<eos>"]]

    reader = GGUFReader(str(MODEL), "r")
    ref_q, ref_k = reference_qk_norms(reader, tokens)
    cpp_q, cpp_k = cpp_qk_norms(SEQUENCE)

    q_err = rel_err(cpp_q, ref_q)
    k_err = rel_err(cpp_k, ref_k)

    print(f"reference: q_norm={ref_q:.6f} k_norm={ref_k:.6f}")
    print(f"cpp:       q_norm={cpp_q:.6f} k_norm={cpp_k:.6f}")
    print(f"rel_err:   q={q_err:.4%} k={k_err:.4%}")

    errors = []
    if q_err > TOLERANCE:
        errors.append(f"Q norm rel err {q_err:.4%} > {TOLERANCE:.0%}")
    if k_err > TOLERANCE:
        errors.append(f"K norm rel err {k_err:.4%} > {TOLERANCE:.0%}")

    if errors:
        print("FAILED (milestone 5):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("VALIDATION OK (milestone 5): layer-0 Q/K norms within 5%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

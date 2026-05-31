#!/usr/bin/env python3
"""Generate reference_embeddings.npz for milestone 6 validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.common import parse_fasta  # noqa: E402
from ref_forward import (  # noqa: E402
    DEFAULT_WEIGHTS,
    forward,
    load_state_dict,
    tokenize,
)

DEFAULT_SMOKE_FASTA = ROOT / "tests" / "sequences_smoke.fasta"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Native safetensors checkpoint",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=DEFAULT_SMOKE_FASTA,
        help="FASTA file of sequences to embed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "reference_embeddings_smoke.npz",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        print(f"Missing weights: {args.weights}", file=sys.stderr)
        return 1
    if not args.fasta.is_file():
        print(f"Missing FASTA: {args.fasta}", file=sys.stderr)
        return 1

    sd = load_state_dict(args.weights)
    reference: dict[str, np.ndarray] = {}
    sequences = parse_fasta(args.fasta)

    for name, seq in sequences:
        ids = tokenize(seq)
        emb = forward(sd, ids)
        reference[name] = emb
        print(f"{name}: shape={emb.shape} mean={emb.mean():.6f} std={emb.std():.6f}")
        print(f"  residue[1][:5] = {emb[1, :5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **reference)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Milestone 4: verify esmc_tokenize matches the HF ESM-C tokenizer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBED = ROOT / "build" / "esmc-embed"
MODEL = ROOT / "models" / "esmc-300m-f16.gguf"


def hf_tokenize(sequence: str) -> list[int]:
    with open(ROOT / "esmc-300m" / "tokenizer.json") as f:
        vocab = json.load(f)["model"]["vocab"]
    ids = [vocab["<cls>"]]
    for c in sequence.upper():
        ids.append(vocab.get(c, vocab["<unk>"]))
    ids.append(vocab["<eos>"])
    return ids


def cpp_tokenize(sequence: str) -> list[int]:
    if not EMBED.exists():
        raise FileNotFoundError(f"build esmc-embed first: {EMBED}")
    if not MODEL.exists():
        raise FileNotFoundError(f"GGUF model missing: {MODEL}")

    proc = subprocess.run(
        [str(EMBED), "-m", str(MODEL), "--test-tokenizer", "-s", sequence],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"esmc-embed failed: {proc.returncode}")

    for line in proc.stdout.splitlines():
        if line.startswith("tokens:"):
            inner = line.split(":", 1)[1].strip()
            return [int(x) for x in inner.strip("[]").split(",") if x.strip()]
    raise RuntimeError(f"no tokens line in output:\n{proc.stdout}")


def main() -> int:
    # Include all 20 standard amino acids (especially Q and N which differ in
    # alphabetical vs ESM-C tokenizer order — easy to swap by mistake).
    cases = {
        "ACDEF": [0, 5, 23, 13, 9, 18, 2],
        "ACDEFGHIK": [0, 5, 23, 13, 9, 18, 6, 21, 12, 15, 2],
        "MKT": [0, 20, 15, 11, 2],
        "QN": [0, 16, 17, 2],
        "ACDEFGHIKLMNPQRSTVWY": [
            0, 5, 23, 13, 9, 18, 6, 21, 12, 15,
            4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19, 2,
        ],
    }

    errors: list[str] = []
    for seq, expected_hf in cases.items():
        hf = hf_tokenize(seq)
        if hf != expected_hf:
            errors.append(f"{seq}: HF tokenizer.json mismatch {hf} vs {expected_hf}")
            continue

        try:
            cpp = cpp_tokenize(seq)
        except FileNotFoundError as e:
            print(f"SKIP cpp: {e}", file=sys.stderr)
            return 0

        if cpp != expected_hf:
            errors.append(f"{seq}: cpp {cpp} != expected {expected_hf}")

    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("VALIDATION OK (milestone 4): tokenizer matches HF vocab")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Milestone 2 verification: GGUF metadata keys and tensor inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

GGUF_PY = Path(__file__).resolve().parent.parent / "ggml" / "gguf-py"
if GGUF_PY.is_dir():
    sys.path.insert(0, str(GGUF_PY))

from gguf import GGUFReader  # noqa: E402

EXPECTED_METADATA_KEYS = [
    "general.architecture",
    "general.name",
    "esmc.block_count",
    "esmc.context_length",
    "esmc.embedding_length",
    "esmc.feed_forward_length",
    "esmc.attention.head_count",
    "esmc.attention.head_count_kv",
    "esmc.rope.freq_base",
    "esmc.vocab_size",
    "esmc.attention.layer_norm_epsilon",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.mask_token_id",
]

REQUIRED_TENSORS = [
    "token_embd.weight",
    "output_norm.weight",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path", type=Path)
    args = parser.parse_args()

    reader = GGUFReader(str(args.gguf_path), "r")

    print("=== Metadata (KV) ===")
    field_names = {f.name for f in reader.fields.values()}
    errors: list[str] = []
    for key in EXPECTED_METADATA_KEYS:
        if key not in field_names:
            errors.append(f"missing metadata key: {key}")
        else:
            f = reader.fields[key]
            print(f"  {key}: {f.contents()}")

    arch = reader.fields.get("general.architecture")
    if arch and arch.contents() != "esmc":
        errors.append(f"general.architecture must be 'esmc', got {arch.contents()}")

    print("\n=== Tensors ===")
    tensor_names = [t.name for t in reader.tensors]
    for t in reader.tensors:
        print(f"  {t.name:45s} {list(t.shape)} {t.tensor_type.name}")

    for req in REQUIRED_TENSORS:
        if req not in tensor_names:
            errors.append(f"missing tensor: {req}")

    n_layer = int(reader.fields["esmc.block_count"].contents())
    for i in range(n_layer):
        for suffix in (
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ):
            name = f"blk.{i}.{suffix}"
            if name not in tensor_names:
                errors.append(f"missing tensor: {name}")

    print(f"\nTotal tensors: {len(tensor_names)} (expected {2 + 9 * n_layer}+)")
    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nVALIDATION OK (milestone 2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

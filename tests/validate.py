#!/usr/bin/env python3
"""Milestone 6/7: validate esmc-embed against reference and CPU/Metal parity."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

EMBED = ROOT / "build" / "esmc-embed"
REF_NPZ = ROOT / "tests" / "reference_embeddings_smoke.npz"
SMOKE_FASTA = ROOT / "tests" / "sequences_smoke.fasta"
DEFAULT_MODEL = ROOT / "models" / "esmc-300m-f16.gguf"

sys.path.insert(0, str(ROOT / "tests"))
from benchmarks.common import cosine_by_row, parse_fasta, run_embed  # noqa: E402

MEAN_COS_MIN = 0.999
MIN_COS_MIN = 0.99
MEAN_POOL_L2_MAX = 0.01
CPU_METAL_MEAN_COS_MIN = 0.999
CPU_METAL_MIN_COS_MIN = 0.99


def load_smoke_sequences() -> list[tuple[str, str]]:
    return parse_fasta(SMOKE_FASTA)


def validate_reference(model: Path, reference: Path, backend: str) -> list[str]:
    ref = np.load(reference)
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name, seq in load_smoke_sequences():
            if name not in ref:
                errors.append(f"{name}: missing in reference npz")
                continue

            our_emb, err = run_embed(model, seq, tmp / f"{name}.{backend}.npy", backend)
            if err:
                errors.append(f"{name}: {err}")
                continue
            assert our_emb is not None

            ref_emb = ref[name][1:-1]

            if our_emb.shape != ref_emb.shape:
                errors.append(f"{name}: shape {our_emb.shape} != {ref_emb.shape}")
                continue

            if not np.isfinite(our_emb).all():
                errors.append(f"{name}: esmc-embed produced non-finite values")
                continue

            cos_sim = cosine_by_row(our_emb, ref_emb)
            mean_cos = float(cos_sim.mean())
            min_cos = float(cos_sim.min())
            print(f"{backend}/{name}: mean_cos={mean_cos:.6f}  min_cos={min_cos:.6f}")

            if not np.isfinite(mean_cos) or mean_cos <= MEAN_COS_MIN:
                errors.append(f"{name}: mean cosine {mean_cos:.6f} <= {MEAN_COS_MIN}")
            if not np.isfinite(min_cos) or min_cos <= MIN_COS_MIN:
                errors.append(f"{name}: min cosine {min_cos:.6f} <= {MIN_COS_MIN}")

            our_mean = our_emb.mean(axis=0)
            ref_mean = ref_emb.mean(axis=0)
            l2_err = float(
                np.linalg.norm(our_mean - ref_mean) / (np.linalg.norm(ref_mean) + 1e-9)
            )
            print(f"  mean_pool relative L2 error: {l2_err:.6f}")
            if not np.isfinite(l2_err) or l2_err >= MEAN_POOL_L2_MAX:
                errors.append(
                    f"{name}: mean pool L2 error {l2_err:.6f} >= {MEAN_POOL_L2_MAX}"
                )

    return errors


def validate_cpu_metal(model: Path) -> list[str]:
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name, seq in load_smoke_sequences():
            cpu_emb, err = run_embed(model, seq, tmp / f"{name}.cpu.npy", "cpu")
            if err:
                errors.append(f"{name}: CPU {err}")
                continue
            metal_emb, err = run_embed(model, seq, tmp / f"{name}.metal.npy", "metal")
            if err:
                errors.append(f"{name}: Metal {err}")
                continue
            assert cpu_emb is not None and metal_emb is not None

            if cpu_emb.shape != metal_emb.shape:
                errors.append(
                    f"{name}: CPU shape {cpu_emb.shape} != Metal shape {metal_emb.shape}"
                )
                continue
            if not np.isfinite(cpu_emb).all() or not np.isfinite(metal_emb).all():
                errors.append(f"{name}: CPU or Metal produced non-finite values")
                continue

            cos_sim = cosine_by_row(cpu_emb, metal_emb)
            mean_cos = float(cos_sim.mean())
            min_cos = float(cos_sim.min())
            print(
                f"cpu_vs_metal/{name}: mean_cos={mean_cos:.6f}  min_cos={min_cos:.6f}"
            )

            if not np.isfinite(mean_cos) or mean_cos <= CPU_METAL_MEAN_COS_MIN:
                errors.append(
                    f"{name}: CPU/Metal mean cosine {mean_cos:.6f} <= {CPU_METAL_MEAN_COS_MIN}"
                )
            if not np.isfinite(min_cos) or min_cos <= CPU_METAL_MIN_COS_MIN:
                errors.append(
                    f"{name}: CPU/Metal min cosine {min_cos:.6f} <= {CPU_METAL_MIN_COS_MIN}"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference", type=Path, default=REF_NPZ)
    parser.add_argument("--no-metal", action="store_true")
    parser.add_argument("--metal", action="store_true", help="require Metal backend")
    parser.add_argument(
        "--compare-cpu-metal",
        action="store_true",
        help="also compare CPU outputs directly against required-Metal outputs",
    )
    args = parser.parse_args()

    if args.no_metal and args.metal:
        print("Choose only one of --no-metal or --metal", file=sys.stderr)
        return 1

    if not EMBED.is_file():
        print(f"Build esmc-embed first: {EMBED}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"Missing model: {args.model}", file=sys.stderr)
        return 1
    if not args.reference.is_file():
        print(
            f"Missing reference: {args.reference}\n"
            "Run: .venv/bin/python tests/generate_reference.py "
            "--fasta tests/sequences_smoke.fasta "
            "--output tests/reference_embeddings_smoke.npz",
            file=sys.stderr,
        )
        return 1

    backend = "cpu" if args.no_metal else "metal" if args.metal else "auto"
    errors = validate_reference(args.model, args.reference, backend)
    if args.compare_cpu_metal:
        errors.extend(validate_cpu_metal(args.model))

    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    if args.metal or args.compare_cpu_metal:
        print("VALIDATION OK (milestone 7): Metal numerics match reference/CPU")
    else:
        print("VALIDATION OK (milestone 6): cosine similarity > 0.999")
    return 0


if __name__ == "__main__":
    sys.exit(main())

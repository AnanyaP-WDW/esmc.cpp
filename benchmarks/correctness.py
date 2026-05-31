#!/usr/bin/env python3
"""Numerical correctness benchmark for esmc.cpp GGUF embeddings.

Milestone 8A:
  --dry-run writes results/manifest.json with host, git, model, backend,
  precision, sequence-set, and reference metadata.

Milestone 8B:
  Running on 300M F16/Q8_0 emits correctness JSON/CSV with per-sequence cosine
  and mean-pool L2 metrics.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import (  # noqa: E402
    RESULTS_DIR,
    base_manifest,
    embedding_metrics,
    load_config,
    models_from_config,
    parse_fasta,
    parse_model_arg,
    resolve_path,
    run_embed,
    write_csv,
    write_json,
)

DEFAULT_SEQUENCES = ROOT / "benchmarks" / "sequences_correctness.fasta"
DEFAULT_REFERENCE = ROOT / "tests" / "reference_embeddings.npz"
DEFAULT_MODELS = {
    "f16": ROOT / "models" / "esmc-300m-f16.gguf",
    "q8_0": ROOT / "models" / "esmc-300m-Q8_0.gguf",
}

CSV_FIELDS = [
    "model_size",
    "precision",
    "backend",
    "sequence_name",
    "sequence_length",
    "reference_shape",
    "output_shape",
    "mean_cosine",
    "min_cosine",
    "mean_pool_l2",
    "mean_cosine_threshold",
    "min_cosine_threshold",
    "mean_pool_l2_threshold",
    "passed",
    "error",
]

def thresholds_for_precision(precision: str) -> dict[str, float]:
    # Thresholds follow plan §13.2 "Benchmark 1: Numerical Correctness":
    #   F16 / Q8_0  -> mean cos > 0.999, min cos > 0.99
    #   Q4_K_M / S  -> mean cos > 0.995, min cos recorded
    # mean_pool_l2 is recorded for every run; the gate is generous since the
    # plan only requires it to be reported, not to define pass/fail.
    if precision in {"f16", "f32", "q8_0"}:
        return {
            "mean_cosine": 0.999,
            "min_cosine": 0.99,
            "mean_pool_l2": 0.05,
        }
    if precision.startswith("q4"):
        return {
            "mean_cosine": 0.995,
            "min_cosine": 0.0,
            "mean_pool_l2": 0.20,
        }
    return {
        "mean_cosine": 0.995,
        "min_cosine": 0.0,
        "mean_pool_l2": 0.20,
    }


def metric_row_passed(metrics: dict[str, Any], thresholds: dict[str, float]) -> bool:
    if metrics.get("error"):
        return False
    return (
        bool(metrics.get("finite"))
        and bool(metrics.get("shape_match"))
        and float(metrics["mean_cosine"]) > thresholds["mean_cosine"]
        and float(metrics["min_cosine"]) > thresholds["min_cosine"]
        and float(metrics["mean_pool_l2"]) < thresholds["mean_pool_l2"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "benchmarks" / "config_300m.json")
    parser.add_argument("--model-size")
    parser.add_argument("--backend", choices=["cpu", "metal", "auto"])
    parser.add_argument("--sequences", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_arg,
        help="Model entry as PRECISION=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-models",
        action="store_true",
        help="Record missing models in artifacts but do not fail the whole benchmark.",
    )
    args = parser.parse_args()

    config = load_config(resolve_path(args.config)) if args.config else {}
    model_size = args.model_size or str(config.get("model_size", "300m"))
    backend = args.backend or str(config.get("backend", "cpu"))
    sequences_path = resolve_path(args.sequences or config.get("sequences", DEFAULT_SEQUENCES))
    reference_path = resolve_path(args.reference or config.get("reference", DEFAULT_REFERENCE))
    output_prefix = resolve_path(
        args.output_prefix or config.get("output_prefix", RESULTS_DIR / "correctness_300m")
    )

    if not sequences_path.is_file():
        print(f"Missing sequence FASTA: {sequences_path}", file=sys.stderr)
        return 1
    if not reference_path.is_file():
        print(f"Missing reference NPZ: {reference_path}", file=sys.stderr)
        return 1

    models = dict(args.model) if args.model else (models_from_config(config) or dict(DEFAULT_MODELS))
    records = parse_fasta(sequences_path)
    manifest = base_manifest(
        benchmark="correctness",
        sequence_path=sequences_path,
        sequences=records,
        models=models,
        backend=backend,
        reference_path=reference_path,
    )

    if args.dry_run:
        write_json(RESULTS_DIR / "manifest.json", manifest)
        print(f"Wrote {RESULTS_DIR / 'manifest.json'}")
        return 0

    ref = np.load(reference_path)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for precision, model_path in models.items():
            thresholds = thresholds_for_precision(precision)
            if not model_path.is_file():
                msg = f"missing model for {precision}: {model_path}"
                errors.append(msg)
                rows.append(
                    {
                        "model_size": model_size,
                        "precision": precision,
                        "backend": backend,
                        "passed": False,
                        "error": msg,
                        "mean_cosine_threshold": thresholds["mean_cosine"],
                        "min_cosine_threshold": thresholds["min_cosine"],
                        "mean_pool_l2_threshold": thresholds["mean_pool_l2"],
                    }
                )
                continue

            for name, seq in records:
                if name not in ref:
                    msg = f"{name}: missing reference embedding"
                    errors.append(msg)
                    rows.append(
                        {
                            "model_size": model_size,
                            "precision": precision,
                            "backend": backend,
                            "sequence_name": name,
                            "sequence_length": len(seq),
                            "passed": False,
                            "error": msg,
                        }
                    )
                    continue

                out_path = tmp / f"{model_size}.{precision}.{backend}.{name}.npy"
                emb, err = run_embed(model_path, seq, out_path, backend)
                ref_emb = ref[name][1:-1]
                if err:
                    metrics: dict[str, Any] = {"error": err, "finite": False, "shape_match": False}
                else:
                    assert emb is not None
                    metrics = embedding_metrics(emb, ref_emb)

                passed = metric_row_passed(metrics, thresholds)
                if not passed:
                    errors.append(f"{precision}/{name}: {metrics.get('error', 'threshold failure')}")

                row = {
                    "model_size": model_size,
                    "precision": precision,
                    "backend": backend,
                    "sequence_name": name,
                    "sequence_length": len(seq),
                    "reference_shape": list(ref_emb.shape),
                    "output_shape": list(emb.shape) if err is None and emb is not None else None,
                    "mean_cosine": metrics.get("mean_cosine"),
                    "min_cosine": metrics.get("min_cosine"),
                    "mean_pool_l2": metrics.get("mean_pool_l2"),
                    "mean_cosine_threshold": thresholds["mean_cosine"],
                    "min_cosine_threshold": thresholds["min_cosine"],
                    "mean_pool_l2_threshold": thresholds["mean_pool_l2"],
                    "passed": passed,
                    "error": metrics.get("error"),
                }
                rows.append(row)
                status = "PASS" if passed else "FAIL"
                print(
                    f"{status} {precision}/{backend}/{name}: "
                    f"mean_cos={row['mean_cosine']} min_cos={row['min_cosine']} "
                    f"mean_pool_l2={row['mean_pool_l2']}"
                )

    summary = {
        "rows": len(rows),
        "passed": sum(1 for row in rows if row.get("passed")),
        "failed": sum(1 for row in rows if not row.get("passed")),
        "allow_missing_models": args.allow_missing_models,
    }
    result = {
        "manifest": manifest,
        "thresholds_by_precision": {
            precision: thresholds_for_precision(precision) for precision in models
        },
        "summary": summary,
        "rows": rows,
    }

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    write_json(json_path, result)
    write_csv(csv_path, rows, CSV_FIELDS)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")

    blocking_errors = [
        e for e in errors if not (args.allow_missing_models and e.startswith("missing model for "))
    ]
    return 1 if blocking_errors else 0


if __name__ == "__main__":
    sys.exit(main())

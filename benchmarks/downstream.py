#!/usr/bin/env python3
"""ProteinGym downstream-quality benchmark for cached ESMC embeddings.

Milestone 8D:
  Run a small ProteinGym substitution subset end-to-end from cached PyTorch and
  GGUF embeddings. Report Spearman correlation and delta from PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests"
for path in (ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.common import (  # noqa: E402
    RESULTS_DIR,
    base_manifest,
    is_standard_sequence,
    load_config,
    mean_pool_residues,
    models_from_config,
    parse_model_arg,
    resolve_path,
    run_embed,
    sha256_file,
    stable_sequence_key,
    write_csv,
    write_json,
)
from benchmarks.downstream_scoring import METRIC_FUNCTIONS, compute_metrics, cosine_to_wildtype  # noqa: E402
from ref_forward import DEFAULT_WEIGHTS, forward, load_state_dict, tokenize  # noqa: E402

DEFAULT_CONFIG = ROOT / "benchmarks" / "config_downstream_300m.json"
DEFAULT_DATASET_DIR = ROOT / "benchmarks" / "proteingym_subset"
DEFAULT_DATASET_MANIFEST = ROOT / "benchmarks" / "datasets" / "proteingym_subset_manifest.json"
DEFAULT_CACHE_DIR = RESULTS_DIR / "downstream_cache" / "300m"
DEFAULT_METRICS = ["spearman", "pearson", "kendall_tau_b", "top10_overlap", "bottom10_overlap"]

CSV_FIELDS = [
    "dataset",
    "subset",
    "assay",
    "model_size",
    "precision",
    "backend",
    "metric_name",
    "metric_value",
    "pytorch_metric",
    "delta_from_pytorch",
    "threshold",
    "passed",
    "variants",
    "cache_dir",
    "error",
]


def load_dataset_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return load_config(path)


def load_assays(dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    assays: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(dataset_dir.glob("*.csv")):
        rows: list[dict[str, Any]] = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            required = {"assay", "target_sequence", "mutated_sequence", "DMS_score"}
            if not required.issubset(fields):
                raise ValueError(f"{path} missing required columns: {sorted(required - fields)}")
            for row in reader:
                target = row["target_sequence"].strip().upper()
                mutated = row["mutated_sequence"].strip().upper()
                if not (
                    is_standard_sequence(target)
                    and is_standard_sequence(mutated)
                    and len(target) == len(mutated)
                ):
                    continue
                rows.append(
                    {
                        "assay": row["assay"].strip() or path.stem,
                        "mutant": row.get("mutant", "").strip(),
                        "target_sequence": target,
                        "mutated_sequence": mutated,
                        "DMS_score": float(row["DMS_score"]),
                    }
                )
        if rows:
            assays[path.stem] = rows
    if not assays:
        raise ValueError(
            f"no ProteinGym subset CSVs found in {dataset_dir}; run benchmarks/fetch_proteingym_subset.py"
        )
    return assays


def unique_sequences(assays: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for rows in assays.values():
        for row in rows:
            for seq in (row["target_sequence"], row["mutated_sequence"]):
                sequences[stable_sequence_key(seq)] = seq
    return sequences


def cache_path(cache_dir: Path, source: str, seq: str) -> Path:
    return cache_dir / source / f"{stable_sequence_key(seq)}.npy"


def load_cached(cache_dir: Path, source: str, seq: str) -> np.ndarray:
    path = cache_path(cache_dir, source, seq)
    if not path.is_file():
        raise FileNotFoundError(f"missing cached embedding: {path}")
    return np.load(path).reshape(-1).astype(np.float32)


def write_cached(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embedding.reshape(-1).astype(np.float32))


def embed_pytorch_sequences(
    sequences: dict[str, str],
    *,
    weights_path: Path,
    cache_dir: Path,
    refresh_cache: bool,
) -> list[str]:
    errors: list[str] = []
    missing = [
        seq for seq in sequences.values() if refresh_cache or not cache_path(cache_dir, "pytorch", seq).is_file()
    ]
    if not missing:
        return errors
    if not weights_path.is_file():
        return [f"missing PyTorch reference weights: {weights_path}"]

    sd = load_state_dict(weights_path)
    total = len(missing)
    for idx, seq in enumerate(missing, start=1):
        path = cache_path(cache_dir, "pytorch", seq)
        try:
            emb = forward(sd, tokenize(seq))[1:-1]
            write_cached(path, mean_pool_residues(emb))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pytorch/{stable_sequence_key(seq)}: {exc}")
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"pytorch embed progress: {idx}/{total}")
    return errors


def embed_gguf_sequences(
    sequences: dict[str, str],
    *,
    models: dict[str, Path],
    backend: str,
    cache_dir: Path,
    refresh_cache: bool,
    allow_missing_models: bool,
) -> list[str]:
    errors: list[str] = []
    for precision, model_path in models.items():
        source = f"{precision}_{backend}"
        if not model_path.is_file():
            msg = f"missing model for {precision}: {model_path}"
            errors.append(msg)
            if allow_missing_models:
                continue
            continue

        pending = [
            seq
            for seq in sequences.values()
            if refresh_cache or not cache_path(cache_dir, source, seq).is_file()
        ]
        total = len(pending)
        for idx, seq in enumerate(pending, start=1):
            path = cache_path(cache_dir, source, seq)
            emb, err = run_embed(model_path, seq, path, backend, pool="mean")
            if err:
                errors.append(f"{source}/{stable_sequence_key(seq)}: {err}")
            elif emb is None:
                errors.append(f"{source}/{stable_sequence_key(seq)}: no embedding returned")
            if total and (idx == 1 or idx % 25 == 0 or idx == total):
                print(f"{source} embed progress: {idx}/{total}")
    return errors


def score_assay(
    assay_name: str,
    rows: list[dict[str, Any]],
    *,
    source: str,
    precision: str,
    backend: str,
    model_size: str,
    cache_dir: Path,
    pytorch_metrics: dict[str, float] | None,
    threshold: float,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    experimental: list[float] = []
    predicted: list[float] = []
    for row in rows:
        wt = load_cached(cache_dir, source, row["target_sequence"])
        mut = load_cached(cache_dir, source, row["mutated_sequence"])
        predicted.append(cosine_to_wildtype(mut, wt))
        experimental.append(float(row["DMS_score"]))

    metrics = compute_metrics(predicted, experimental, metric_names)
    metric_rows: list[dict[str, Any]] = []
    for metric_name, metric in metrics.items():
        pytorch_metric = metric if pytorch_metrics is None else pytorch_metrics[metric_name]
        delta = 0.0 if pytorch_metrics is None else metric - pytorch_metric
        metric_rows.append(
            {
                "dataset": "ProteinGym",
                "subset": "DMS substitutions",
                "assay": assay_name,
                "model_size": model_size,
                "precision": precision,
                "backend": backend,
                "metric_name": metric_name,
                "metric_value": metric,
                "pytorch_metric": pytorch_metric,
                "delta_from_pytorch": delta,
                "threshold": threshold,
                "passed": abs(delta) <= threshold,
                "variants": len(rows),
                "cache_dir": str(cache_dir / source),
                "error": None,
            }
        )
    return metric_rows


def score_all(
    assays: dict[str, list[dict[str, Any]]],
    *,
    models: dict[str, Path],
    backend: str,
    model_size: str,
    cache_dir: Path,
    threshold: float,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for assay_name, assay_rows in assays.items():
        pytorch_rows = score_assay(
            assay_name,
            assay_rows,
            source="pytorch",
            precision="pytorch",
            backend="pytorch",
            model_size=model_size,
            cache_dir=cache_dir,
            pytorch_metrics=None,
            threshold=threshold,
            metric_names=metric_names,
        )
        rows.extend(pytorch_rows)
        pytorch_metrics = {
            str(row["metric_name"]): float(row["metric_value"]) for row in pytorch_rows
        }

        for precision in models:
            source = f"{precision}_{backend}"
            try:
                rows.extend(
                    score_assay(
                        assay_name,
                        assay_rows,
                        source=source,
                        precision=precision,
                        backend=backend,
                        model_size=model_size,
                        cache_dir=cache_dir,
                        pytorch_metrics=pytorch_metrics,
                        threshold=threshold,
                        metric_names=metric_names,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                for metric_name in metric_names:
                    rows.append(
                        {
                            "dataset": "ProteinGym",
                            "subset": "DMS substitutions",
                            "assay": assay_name,
                            "model_size": model_size,
                            "precision": precision,
                            "backend": backend,
                            "metric_name": metric_name,
                            "threshold": threshold,
                            "passed": False,
                            "variants": len(assay_rows),
                            "cache_dir": str(cache_dir / source),
                            "error": str(exc),
                        }
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-size")
    parser.add_argument("--backend", choices=["cpu", "metal", "auto"])
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--metrics",
        help=(
            "Comma-separated metric names. Available: "
            + ", ".join(sorted(METRIC_FUNCTIONS))
        ),
    )
    parser.add_argument("--model", action="append", type=parse_model_arg)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--allow-missing-models", action="store_true")
    args = parser.parse_args()

    config = load_config(resolve_path(args.config)) if args.config else {}
    downstream_config = config.get("downstream", {})
    model_size = args.model_size or str(config.get("model_size", "300m"))
    backend = args.backend or str(config.get("backend", "cpu"))
    dataset_dir = resolve_path(
        args.dataset_dir or downstream_config.get("dataset_dir", DEFAULT_DATASET_DIR)
    )
    dataset_manifest_path = resolve_path(
        args.dataset_manifest
        or downstream_config.get("dataset_manifest", DEFAULT_DATASET_MANIFEST)
    )
    weights_path = resolve_path(args.weights or downstream_config.get("weights", DEFAULT_WEIGHTS))
    cache_dir = resolve_path(args.cache_dir or downstream_config.get("cache_dir", DEFAULT_CACHE_DIR))
    output_prefix = resolve_path(
        args.output_prefix or downstream_config.get("output_prefix", RESULTS_DIR / "downstream_300m")
    )
    threshold = float(args.threshold or downstream_config.get("threshold", 0.01))
    metric_names = [
        name.strip()
        for name in str(args.metrics or downstream_config.get("metrics", ",".join(DEFAULT_METRICS))).split(",")
        if name.strip()
    ]
    unknown_metrics = sorted(set(metric_names) - set(METRIC_FUNCTIONS))
    if unknown_metrics:
        print(f"Unknown downstream metrics: {', '.join(unknown_metrics)}", file=sys.stderr)
        print(f"Available metrics: {', '.join(sorted(METRIC_FUNCTIONS))}", file=sys.stderr)
        return 1
    if not metric_names:
        print("At least one downstream metric is required.", file=sys.stderr)
        return 1
    models = dict(args.model) if args.model else models_from_config(config)

    assays = load_assays(dataset_dir)
    sequences = unique_sequences(assays)
    sequence_records = [(key, seq) for key, seq in sorted(sequences.items())]
    dataset_manifest = load_dataset_manifest(dataset_manifest_path)
    manifest = base_manifest(
        benchmark="downstream",
        sequence_path=dataset_dir,
        sequences=sequence_records,
        models=models,
        backend=backend,
        reference_path=weights_path,
    )
    manifest["dataset"] = {
        "name": "ProteinGym",
        "subset": "DMS substitutions",
        "path": str(dataset_dir),
        "manifest": str(dataset_manifest_path),
        "manifest_exists": dataset_manifest_path.is_file(),
        "manifest_sha256": sha256_file(dataset_manifest_path),
        "manifest_data": dataset_manifest,
        "assays": {name: len(rows) for name, rows in assays.items()},
    }
    manifest["cache_dir"] = str(cache_dir)
    manifest["metrics"] = metric_names

    if args.dry_run:
        write_json(RESULTS_DIR / "downstream_manifest.json", manifest)
        print(f"Wrote {RESULTS_DIR / 'downstream_manifest.json'}")
        return 0

    errors: list[str] = []
    if not args.score_only:
        errors.extend(
            embed_pytorch_sequences(
                sequences,
                weights_path=weights_path,
                cache_dir=cache_dir,
                refresh_cache=args.refresh_cache,
            )
        )
        errors.extend(
            embed_gguf_sequences(
                sequences,
                models=models,
                backend=backend,
                cache_dir=cache_dir,
                refresh_cache=args.refresh_cache,
                allow_missing_models=args.allow_missing_models,
            )
        )

    rows: list[dict[str, Any]] = []
    if not args.embed_only:
        try:
            rows = score_all(
                assays,
                models=models,
                backend=backend,
                model_size=model_size,
                cache_dir=cache_dir,
                threshold=threshold,
                metric_names=metric_names,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    result = {
        "manifest": manifest,
        "summary": {
            "assays": len(assays),
            "variants": sum(len(rows_) for rows_ in assays.values()),
            "unique_sequences": len(sequences),
            "rows": len(rows),
            "passed": sum(1 for row in rows if row.get("passed")),
            "failed": sum(1 for row in rows if not row.get("passed")),
            "errors": errors,
        },
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

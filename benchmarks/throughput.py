#!/usr/bin/env python3
"""Milestone 8E throughput benchmark.

Runs each backend in a fresh worker process and writes CSV/JSON artifacts with
latency percentiles, seq/s, residues/s, tokens/s, warmups, measured iterations,
and sequence bucket metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests"
for path in (ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.common import (  # noqa: E402
    RESULTS_DIR,
    base_manifest,
    load_config,
    models_from_config,
    parse_fasta,
    parse_model_arg,
    resolve_path,
    write_csv,
    write_json,
)

DEFAULT_CONFIG = ROOT / "benchmarks" / "config_throughput_300m.json"
DEFAULT_SEQUENCES = ROOT / "benchmarks" / "sequences_throughput.fasta"
DEFAULT_CPP_BENCH = ROOT / "build" / "esmc-bench"
DEFAULT_WEIGHTS = ROOT / "esmc-300m" / "model.safetensors"
BACKENDS = ("cpu", "metal", "pytorch_cpu", "pytorch_mps")
PYTORCH_DTYPES = ("f32", "f16", "bf16")

CSV_FIELDS = [
    "model_size",
    "precision",
    "implementation",
    "backend",
    "sequence_name",
    "sequence_bucket",
    "sequence_length",
    "token_count",
    "warmup",
    "iterations",
    "median_latency_ms",
    "p95_latency_ms",
    "seq_per_s",
    "residues_per_s",
    "tokens_per_s",
    "total_time_s",
    "error",
]


def default_output_prefix() -> Path:
    host = socket.gethostname().split(".")[0] or "host"
    date = datetime.now().strftime("%Y%m%d")
    return RESULTS_DIR / f"throughput_{host}_{date}"


def sequence_bucket(name: str, seq: str) -> str:
    prefix = name.split("_", 1)[0].lower()
    if prefix in {"short", "medium", "long", "max", "max-context"}:
        return prefix
    length = len(seq)
    if length <= 100:
        return "short"
    if length <= 512:
        return "medium"
    if length <= 1536:
        return "long"
    return "max-context"


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    idx = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[idx]


def summarize_latencies(
    latencies_ms: list[float],
    *,
    model_size: str,
    precision: str,
    implementation: str,
    backend: str,
    name: str,
    seq: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    total_time_s = sum(latencies_ms) / 1000.0
    seq_per_s = len(latencies_ms) / total_time_s if total_time_s > 0 else 0.0
    residues = len(seq) * len(latencies_ms)
    tokens = (len(seq) + 2) * len(latencies_ms)
    return {
        "model_size": model_size,
        "precision": precision,
        "implementation": implementation,
        "backend": backend,
        "sequence_name": name,
        "sequence_bucket": sequence_bucket(name, seq),
        "sequence_length": len(seq),
        "token_count": len(seq) + 2,
        "warmup": warmup,
        "iterations": iterations,
        "median_latency_ms": median(latencies_ms),
        "p95_latency_ms": p95(latencies_ms),
        "seq_per_s": seq_per_s,
        "residues_per_s": residues / total_time_s if total_time_s > 0 else 0.0,
        "tokens_per_s": tokens / total_time_s if total_time_s > 0 else 0.0,
        "total_time_s": total_time_s,
        "error": None,
    }


def error_row(
    *,
    model_size: str,
    precision: str,
    implementation: str,
    backend: str,
    name: str,
    seq: str,
    warmup: int,
    iterations: int,
    error: str,
) -> dict[str, Any]:
    return {
        "model_size": model_size,
        "precision": precision,
        "implementation": implementation,
        "backend": backend,
        "sequence_name": name,
        "sequence_bucket": sequence_bucket(name, seq),
        "sequence_length": len(seq),
        "token_count": len(seq) + 2,
        "warmup": warmup,
        "iterations": iterations,
        "error": error,
    }


def run_cpp_sequence(
    *,
    bench: Path,
    model: Path,
    backend: str,
    seq: str,
    warmup: int,
    iterations: int,
) -> tuple[list[float], str | None]:
    if not bench.is_file():
        return [], f"missing esmc-bench binary: {bench}"
    if not model.is_file():
        return [], f"missing model: {model}"

    cmd = [
        str(bench),
        "-m",
        str(model),
        "-s",
        seq,
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    if backend == "cpu":
        cmd.append("--no-metal")
    elif backend == "metal":
        cmd.append("--require-metal")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return [], f"esmc-bench failed ({proc.returncode})\n{proc.stderr.strip()}"

    latencies: list[float] = []
    for line in proc.stdout.splitlines():
        if line.strip() == "latency_ms":
            continue
        try:
            latencies.append(float(line))
        except ValueError:
            continue
    if len(latencies) != iterations:
        return latencies, f"expected {iterations} latencies, got {len(latencies)}"
    return latencies, None


def run_pytorch_worker(
    *,
    backend: str,
    weights: Path,
    records: list[tuple[str, str]],
    model_size: str,
    pytorch_dtype: str,
    warmup: int,
    iterations: int,
) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415
    from ref_forward import forward, load_state_dict, tokenize  # noqa: PLC0415

    dtype_by_name = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_by_name[pytorch_dtype]
    device = "cpu"
    if backend == "pytorch_mps":
        if not torch.backends.mps.is_available():
            return [
                error_row(
                    model_size=model_size,
                    precision=pytorch_dtype,
                    implementation="pytorch",
                    backend=backend,
                    name=name,
                    seq=seq,
                    warmup=warmup,
                    iterations=iterations,
                    error="torch MPS is not available",
                )
                for name, seq in records
            ]
        device = "mps"

    if not weights.is_file():
        return [
            error_row(
                model_size=model_size,
                precision=pytorch_dtype,
                implementation="pytorch",
                backend=backend,
                name=name,
                seq=seq,
                warmup=warmup,
                iterations=iterations,
                error=f"missing PyTorch weights: {weights}",
            )
            for name, seq in records
        ]

    sd = {key: tensor.to(device=device, dtype=dtype) for key, tensor in load_state_dict(weights).items()}
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for name, seq in records:
            token_ids = tokenize(seq)
            try:
                for _ in range(warmup):
                    forward(sd, token_ids, device=device)
                    if device == "mps":
                        torch.mps.synchronize()

                latencies: list[float] = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    forward(sd, token_ids, device=device)
                    if device == "mps":
                        torch.mps.synchronize()
                    latencies.append((time.perf_counter() - start) * 1000.0)
                rows.append(
                    summarize_latencies(
                        latencies,
                        model_size=model_size,
                        precision=pytorch_dtype,
                        implementation="pytorch",
                        backend=backend,
                        name=name,
                        seq=seq,
                        warmup=warmup,
                        iterations=iterations,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    error_row(
                        model_size=model_size,
                        precision=pytorch_dtype,
                        implementation="pytorch",
                        backend=backend,
                        name=name,
                        seq=seq,
                        warmup=warmup,
                        iterations=iterations,
                        error=str(exc),
                    )
                )
    return rows


def run_worker(args: argparse.Namespace) -> int:
    records = parse_fasta(resolve_path(args.sequences))
    rows: list[dict[str, Any]] = []

    if args.backend.startswith("pytorch"):
        rows = run_pytorch_worker(
            backend=args.backend,
            weights=resolve_path(args.weights),
            records=records,
            model_size=args.model_size,
            pytorch_dtype=args.pytorch_dtype,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    else:
        model = resolve_path(args.model_path)
        bench = resolve_path(args.cpp_bench)
        for name, seq in records:
            latencies, err = run_cpp_sequence(
                bench=bench,
                model=model,
                backend=args.backend,
                seq=seq,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            if err:
                rows.append(
                    error_row(
                        model_size=args.model_size,
                        precision=args.precision,
                        implementation="esmc.cpp",
                        backend=args.backend,
                        name=name,
                        seq=seq,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        error=err,
                    )
                )
                continue
            rows.append(
                summarize_latencies(
                    latencies,
                    model_size=args.model_size,
                    precision=args.precision,
                    implementation="esmc.cpp",
                    backend=args.backend,
                    name=name,
                    seq=seq,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )

    write_json(resolve_path(args.worker_output), {"rows": rows})
    return 0 if all(not row.get("error") for row in rows) else 1


def run_parent(args: argparse.Namespace) -> int:
    config = load_config(resolve_path(args.config)) if args.config else {}
    model_size = args.model_size or str(config.get("model_size", "300m"))
    sequences_path = resolve_path(args.sequences or config.get("sequences", DEFAULT_SEQUENCES))
    weights = resolve_path(args.weights or config.get("pytorch_weights", DEFAULT_WEIGHTS))
    pytorch_dtype = str(args.pytorch_dtype or config.get("pytorch_dtype", "f32")).lower()
    cpp_bench = resolve_path(args.cpp_bench or config.get("cpp_bench", DEFAULT_CPP_BENCH))
    warmup = args.warmup if args.warmup is not None else int(config.get("warmup", 3))
    iterations = args.iterations if args.iterations is not None else int(config.get("iterations", 10))
    output_prefix = resolve_path(args.output_prefix or config.get("output_prefix", default_output_prefix()))
    backends = args.backend or list(config.get("backends", BACKENDS))
    models = dict(args.model) if args.model else models_from_config(config)

    unknown_backends = sorted(set(backends) - set(BACKENDS))
    if unknown_backends:
        print(f"unknown backend(s): {', '.join(unknown_backends)}", file=sys.stderr)
        return 1
    if pytorch_dtype not in PYTORCH_DTYPES:
        print(f"unknown PyTorch dtype: {pytorch_dtype}", file=sys.stderr)
        return 1
    if not sequences_path.is_file():
        print(f"missing sequence FASTA: {sequences_path}", file=sys.stderr)
        return 1
    records = parse_fasta(sequences_path)

    rows: list[dict[str, Any]] = []
    worker_failures = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        jobs: list[dict[str, Any]] = []
        for backend in backends:
            if backend.startswith("pytorch"):
                jobs.append({"backend": backend, "precision": pytorch_dtype, "model_path": ""})
            else:
                for precision, model_path in models.items():
                    jobs.append({"backend": backend, "precision": precision, "model_path": str(model_path)})

        for idx, job in enumerate(jobs, start=1):
            worker_output = tmp / f"worker_{idx}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--model-size",
                model_size,
                "--backend",
                job["backend"],
                "--precision",
                job["precision"],
                "--model-path",
                job["model_path"],
                "--sequences",
                str(sequences_path),
                "--weights",
                str(weights),
                "--cpp-bench",
                str(cpp_bench),
                "--pytorch-dtype",
                pytorch_dtype,
                "--warmup",
                str(warmup),
                "--iterations",
                str(iterations),
                "--worker-output",
                str(worker_output),
            ]
            print(f"Running {job['precision']}/{job['backend']} in a fresh process")
            proc = subprocess.run(cmd, text=True, check=False)
            if proc.returncode != 0:
                worker_failures += 1
            if worker_output.is_file():
                rows.extend(load_config(worker_output)["rows"])
            else:
                worker_failures += 1
                rows.append(
                    {
                        "model_size": model_size,
                        "precision": job["precision"],
                        "implementation": "pytorch" if job["backend"].startswith("pytorch") else "esmc.cpp",
                        "backend": job["backend"],
                        "warmup": warmup,
                        "iterations": iterations,
                        "error": "worker did not write output",
                    }
                )

    manifest = base_manifest(
        benchmark="throughput",
        sequence_path=sequences_path,
        sequences=records,
        models=models,
        backend=",".join(backends),
        reference_path=weights,
    )
    result = {
        "manifest": manifest,
        "summary": {
            "rows": len(rows),
            "failed_rows": sum(1 for row in rows if row.get("error")),
            "worker_failures": worker_failures,
            "warmup": warmup,
            "iterations": iterations,
            "pytorch_dtype": pytorch_dtype,
        },
        "rows": rows,
    }

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    write_json(json_path, result)
    write_csv(csv_path, rows, CSV_FIELDS)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0 if worker_failures == 0 and all(not row.get("error") for row in rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-size")
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--sequences", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--cpp-bench", type=Path)
    parser.add_argument("--pytorch-dtype", choices=PYTORCH_DTYPES)
    parser.add_argument("--model", action="append", type=parse_model_arg)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--precision", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if isinstance(args.backend, list):
            args.backend = args.backend[-1] if args.backend else None
        required = [args.model_size, args.backend, args.precision, args.sequences, args.worker_output]
        if any(value is None for value in required):
            print("missing worker arguments", file=sys.stderr)
            return 1
        args.pytorch_dtype = args.pytorch_dtype or "f32"
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

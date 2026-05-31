#!/usr/bin/env python3
"""Milestone 8F memory-footprint benchmark.

Runs each model / backend / sequence bucket in a fresh process under
`/usr/bin/time -l` and records peak resident set size plus model file size.
"""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
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

DEFAULT_CONFIG = ROOT / "benchmarks" / "config_memory_300m.json"
DEFAULT_SEQUENCES = ROOT / "benchmarks" / "sequences_throughput.fasta"
DEFAULT_CPP_EMBED = ROOT / "build" / "esmc-embed"
DEFAULT_WEIGHTS = ROOT / "esmc-300m" / "model.safetensors"
BACKENDS = ("cpu", "metal", "pytorch_cpu", "pytorch_mps")
PYTORCH_DTYPES = ("f32", "f16", "bf16")
TIME_BIN = Path("/usr/bin/time")
TIME_RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)

CSV_FIELDS = [
    "model_size",
    "precision",
    "implementation",
    "backend",
    "sequence_name",
    "sequence_bucket",
    "sequence_length",
    "token_count",
    "model_path",
    "model_file_size_bytes",
    "peak_rss_bytes",
    "peak_rss_mib",
    "machine_budget_bytes",
    "budget_pass",
    "success",
    "returncode",
    "elapsed_s",
    "notes",
    "error",
]


def default_output_prefix() -> Path:
    host = socket.gethostname().split(".")[0] or "host"
    date = datetime.now().strftime("%Y%m%d")
    return RESULTS_DIR / f"memory_{host}_{date}"


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


def model_file_size(path: Path | None) -> int | None:
    if path and path.is_file():
        return path.stat().st_size
    return None


def parse_peak_rss(stderr: str) -> int | None:
    match = TIME_RSS_RE.search(stderr)
    return int(match.group(1)) if match else None


def run_timed_command(cmd: list[str]) -> tuple[int, float, int | None, str, str]:
    if not TIME_BIN.is_file():
        raise RuntimeError(f"missing required timer: {TIME_BIN}")
    timed_cmd = [str(TIME_BIN), "-l", *cmd]
    start = time.perf_counter()
    proc = subprocess.run(timed_cmd, capture_output=True, text=True, check=False)
    elapsed_s = time.perf_counter() - start
    return proc.returncode, elapsed_s, parse_peak_rss(proc.stderr), proc.stdout, proc.stderr


def row_for_result(
    *,
    model_size: str,
    precision: str,
    implementation: str,
    backend: str,
    name: str,
    seq: str,
    model_path: Path | None,
    budget_bytes: int,
    returncode: int,
    elapsed_s: float,
    peak_rss_bytes: int | None,
    notes: str,
    error: str | None,
) -> dict[str, Any]:
    effective_error = error
    if returncode == 0 and peak_rss_bytes is None:
        effective_error = "could not parse maximum resident set size from /usr/bin/time -l"
    success = returncode == 0 and effective_error is None and peak_rss_bytes is not None
    peak_rss_mib = (peak_rss_bytes / (1024.0 * 1024.0)) if peak_rss_bytes is not None else None
    return {
        "model_size": model_size,
        "precision": precision,
        "implementation": implementation,
        "backend": backend,
        "sequence_name": name,
        "sequence_bucket": sequence_bucket(name, seq),
        "sequence_length": len(seq),
        "token_count": len(seq) + 2,
        "model_path": str(model_path) if model_path else "",
        "model_file_size_bytes": model_file_size(model_path),
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_mib": peak_rss_mib,
        "machine_budget_bytes": budget_bytes,
        "budget_pass": bool(success and peak_rss_bytes is not None and peak_rss_bytes <= budget_bytes),
        "success": success,
        "returncode": returncode,
        "elapsed_s": elapsed_s,
        "notes": notes,
        "error": effective_error,
    }


def run_cpp_job(
    *,
    embed: Path,
    model: Path,
    backend: str,
    seq: str,
) -> tuple[int, float, int | None, str | None, str]:
    if not embed.is_file():
        return 127, 0.0, None, f"missing esmc-embed binary: {embed}", ""
    if not model.is_file():
        return 127, 0.0, None, f"missing model: {model}", ""

    cmd = [
        str(embed),
        "-m",
        str(model),
        "-s",
        seq,
        "--pool",
        "mean",
    ]
    if backend == "cpu":
        cmd.append("--no-metal")
    elif backend == "metal":
        cmd.append("--require-metal")

    returncode, elapsed_s, peak_rss, _stdout, stderr = run_timed_command(cmd)
    error = None if returncode == 0 else stderr.strip() or f"esmc-embed failed ({returncode})"
    return returncode, elapsed_s, peak_rss, error, "single mean-pooled embed"


def run_pytorch_worker(args: argparse.Namespace) -> int:
    import torch  # noqa: PLC0415
    from ref_forward import forward, load_state_dict, tokenize  # noqa: PLC0415

    dtype_by_name = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }
    device_name = "mps" if args.backend == "pytorch_mps" else "cpu"
    device = torch.device(device_name)
    dtype = dtype_by_name[args.pytorch_dtype]

    sd = load_state_dict(resolve_path(args.weights))
    sd = {key: value.to(device=device, dtype=dtype) for key, value in sd.items()}
    token_ids = tokenize(args.sequence)

    with torch.inference_mode():
        output = forward(sd, token_ids, device=device)
        if device_name == "mps":
            torch.mps.synchronize()

    print(f"pytorch embed OK: {output.shape[0]} tokens x {output.shape[1]} dims")
    return 0


def run_pytorch_job(
    *,
    weights: Path,
    backend: str,
    pytorch_dtype: str,
    seq: str,
) -> tuple[int, float, int | None, str | None, str]:
    if not weights.is_file():
        return 127, 0.0, None, f"missing PyTorch weights: {weights}", ""

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--pytorch-worker",
        "--backend",
        backend,
        "--weights",
        str(weights),
        "--pytorch-dtype",
        pytorch_dtype,
        "--sequence",
        seq,
    ]
    returncode, elapsed_s, peak_rss, _stdout, stderr = run_timed_command(cmd)
    error = None if returncode == 0 else stderr.strip() or f"PyTorch worker failed ({returncode})"
    return returncode, elapsed_s, peak_rss, error, "single PyTorch forward"


def run_parent(args: argparse.Namespace) -> int:
    config = load_config(resolve_path(args.config)) if args.config else {}
    model_size = args.model_size or str(config.get("model_size", "300m"))
    sequences_path = resolve_path(args.sequences or config.get("sequences", DEFAULT_SEQUENCES))
    weights = resolve_path(args.weights or config.get("pytorch_weights", DEFAULT_WEIGHTS))
    pytorch_dtype = str(args.pytorch_dtype or config.get("pytorch_dtype", "f32")).lower()
    cpp_embed = resolve_path(args.cpp_embed or config.get("cpp_embed", DEFAULT_CPP_EMBED))
    output_prefix = resolve_path(args.output_prefix or config.get("output_prefix", default_output_prefix()))
    backends = args.backend or list(config.get("backends", BACKENDS))
    models = dict(args.model) if args.model else models_from_config(config)
    budget_gb = args.budget_gb if args.budget_gb is not None else float(config.get("budget_gb", 16.0))
    budget_bytes = int(budget_gb * 1024 * 1024 * 1024)

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
    cpp_backends = [backend for backend in backends if not backend.startswith("pytorch")]
    pytorch_backends = [backend for backend in backends if backend.startswith("pytorch")]
    expected_rows = (len(cpp_backends) * len(models) + len(pytorch_backends)) * len(records)

    print(
        "Memory benchmark job matrix: "
        f"{len(models)} GGUF model(s) × {len(cpp_backends)} esmc.cpp backend(s) + "
        f"{len(pytorch_backends)} PyTorch backend(s), across {len(records)} sequence bucket(s) "
        f"= {expected_rows} fresh process run(s)"
    )
    if models:
        print("GGUF models: " + ", ".join(f"{precision}={path}" for precision, path in models.items()))
    if pytorch_backends:
        print(f"PyTorch baseline: {pytorch_dtype}={weights}")

    rows: list[dict[str, Any]] = []
    for backend in backends:
        if backend.startswith("pytorch"):
            for name, seq in records:
                print(f"Measuring {pytorch_dtype}/{backend}/{name} in a fresh process")
                returncode, elapsed_s, peak_rss, error, notes = run_pytorch_job(
                    weights=weights,
                    backend=backend,
                    pytorch_dtype=pytorch_dtype,
                    seq=seq,
                )
                rows.append(
                    row_for_result(
                        model_size=model_size,
                        precision=pytorch_dtype,
                        implementation="pytorch",
                        backend=backend,
                        name=name,
                        seq=seq,
                        model_path=weights,
                        budget_bytes=budget_bytes,
                        returncode=returncode,
                        elapsed_s=elapsed_s,
                        peak_rss_bytes=peak_rss,
                        notes=notes,
                        error=error,
                    )
                )
            continue

        for precision, model_path in models.items():
            model = resolve_path(model_path)
            for name, seq in records:
                print(f"Measuring {precision}/{backend}/{name} in a fresh process")
                returncode, elapsed_s, peak_rss, error, notes = run_cpp_job(
                    embed=cpp_embed,
                    model=model,
                    backend=backend,
                    seq=seq,
                )
                rows.append(
                    row_for_result(
                        model_size=model_size,
                        precision=precision,
                        implementation="esmc.cpp",
                        backend=backend,
                        name=name,
                        seq=seq,
                        model_path=model,
                        budget_bytes=budget_bytes,
                        returncode=returncode,
                        elapsed_s=elapsed_s,
                        peak_rss_bytes=peak_rss,
                        notes=notes,
                        error=error,
                    )
                )

    manifest = base_manifest(
        benchmark="memory",
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
            "failed_rows": sum(1 for row in rows if not row.get("success")),
            "budget_gb": budget_gb,
            "budget_bytes": budget_bytes,
            "pytorch_dtype": pytorch_dtype,
            "timer": str(TIME_BIN),
        },
        "rows": rows,
    }

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    write_json(json_path, result)
    write_csv(csv_path, rows, CSV_FIELDS)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0 if all(row.get("success") for row in rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-size")
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--sequences", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--cpp-embed", type=Path)
    parser.add_argument("--pytorch-dtype", choices=PYTORCH_DTYPES)
    parser.add_argument("--model", action="append", type=parse_model_arg)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--budget-gb", type=float)
    parser.add_argument("--pytorch-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sequence", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.pytorch_worker:
        if isinstance(args.backend, list):
            args.backend = args.backend[-1] if args.backend else None
        if not args.backend or not args.weights or not args.sequence:
            print("missing PyTorch worker arguments", file=sys.stderr)
            return 1
        args.pytorch_dtype = args.pytorch_dtype or "f32"
        return run_pytorch_worker(args)

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

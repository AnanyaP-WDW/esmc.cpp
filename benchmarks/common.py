#!/usr/bin/env python3
"""Shared utilities for esmc.cpp benchmark harnesses."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBED = ROOT / "build" / "esmc-embed"
RESULTS_DIR = ROOT / "results"
BENCHMARK_SCHEMA_VERSION = 1
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str]) -> str | None:
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"expected JSON object in {path}")
    return config


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model entries must be PRECISION=PATH")
    precision, path = value.split("=", 1)
    precision = precision.strip().lower()
    if not precision:
        raise argparse.ArgumentTypeError("precision cannot be empty")
    return precision, Path(path).expanduser()


def models_from_config(config: dict[str, Any]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for entry in config.get("models", []):
        precision = str(entry["precision"]).strip().lower()
        models[precision] = resolve_path(entry["path"])
    return models


def git_info() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"])
    status = run_command(["git", "status", "--short"])
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def host_info() -> dict[str, Any]:
    sw_vers = run_command(["sw_vers"])
    mem_bytes: int | None = None
    try:
        if hasattr(os, "sysconf"):
            mem_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        mem_bytes = None

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "macos": sw_vers.splitlines() if sw_vers else None,
        "memory_bytes": mem_bytes,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    parts: list[str] = []

    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(parts)))
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)

    if name is not None:
        records.append((name, "".join(parts)))
    if not records:
        raise ValueError(f"no FASTA records in {path}")
    return records


def is_standard_sequence(seq: str) -> bool:
    return bool(seq) and set(seq.upper()).issubset(STANDARD_AA)


def stable_sequence_key(seq: str) -> str:
    return hashlib.sha256(seq.upper().encode()).hexdigest()[:16]


def sequence_manifest(path: Path, records: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "count": len(records),
        "records": [
            {"name": name, "length": len(seq), "sha256": hashlib.sha256(seq.encode()).hexdigest()}
            for name, seq in records
        ],
    }


def run_embed(
    model: Path,
    seq: str,
    out_path: Path,
    backend: str,
    *,
    embed: Path = DEFAULT_EMBED,
    pool: str = "none",
) -> tuple[np.ndarray | None, str | None]:
    if pool not in {"none", "mean"}:
        return None, f"unsupported pooling mode: {pool}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(embed),
        "-m",
        str(model),
        "-s",
        seq,
        "--pool",
        pool,
        "--output",
        str(out_path),
    ]
    if backend == "cpu":
        cmd.append("--no-metal")
    elif backend == "metal":
        cmd.append("--require-metal")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None, f"esmc-embed failed ({proc.returncode})\n{proc.stderr}"
    if not out_path.is_file():
        return None, f"no output file {out_path}"
    return np.load(out_path), None


def mean_pool_residues(emb: np.ndarray) -> np.ndarray:
    if emb.ndim != 2:
        raise ValueError(f"expected 2D embedding, got shape {emb.shape}")
    if emb.shape[0] == 0:
        raise ValueError("cannot mean-pool an empty embedding")
    return emb.mean(axis=0).astype(np.float32)


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def cosine_by_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    )


def embedding_metrics(ours: np.ndarray, ref: np.ndarray) -> dict[str, float | bool | str]:
    if ours.shape != ref.shape:
        return {
            "finite": bool(np.isfinite(ours).all()),
            "shape_match": False,
            "error": f"shape {ours.shape} != {ref.shape}",
        }
    if not np.isfinite(ours).all():
        return {
            "finite": False,
            "shape_match": True,
            "error": "non-finite output values",
        }

    cos = cosine_by_row(ours, ref)
    ours_mean = ours.mean(axis=0)
    ref_mean = ref.mean(axis=0)
    return {
        "finite": True,
        "shape_match": True,
        "mean_cosine": float(cos.mean()),
        "min_cosine": float(cos.min()),
        "mean_pool_l2": float(
            np.linalg.norm(ours_mean - ref_mean) / (np.linalg.norm(ref_mean) + 1e-9)
        ),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def base_manifest(
    *,
    benchmark: str,
    sequence_path: Path,
    sequences: list[tuple[str, str]],
    models: dict[str, Path],
    backend: str,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": benchmark,
        "created_at": utc_now_iso(),
        "host": host_info(),
        "git": git_info(),
        "backend": backend,
        "models": {
            precision: {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path),
            }
            for precision, path in models.items()
        },
        "reference": {
            "path": str(reference_path) if reference_path else None,
            "exists": reference_path.is_file() if reference_path else None,
            "sha256": sha256_file(reference_path) if reference_path else None,
        },
        "sequence_set": sequence_manifest(sequence_path, sequences),
    }

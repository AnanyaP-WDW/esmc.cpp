#!/usr/bin/env python3
"""Download a fixed correctness benchmark FASTA from UniProt Swiss-Prot.

Source: https://www.uniprot.org/ (CC BY 4.0)
API docs: https://rest.uniprot.org/help/api

The script selects reviewed (Swiss-Prot) entries in three length buckets,
filters to the 20 standard amino acids, and writes a reproducible FASTA plus
a JSON manifest recording the UniProt query and selected accessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "benchmarks" / "sequences_correctness.fasta"
DEFAULT_MANIFEST = ROOT / "benchmarks" / "datasets" / "uniprot_correctness_manifest.json"

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

BUCKETS: list[dict[str, object]] = [
    {
        "name": "small",
        "min_len": 10,
        "max_len": 50,
        "count": 34,
    },
    {
        "name": "medium",
        "min_len": 100,
        "max_len": 300,
        "count": 33,
    },
    {
        "name": "large",
        "min_len": 500,
        "max_len": 1500,
        "count": 33,
    },
]


def parse_fasta_text(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header = line[1:].split()[0]
            parts = []
        else:
            parts.append(line)

    if header is not None:
        records.append((header, "".join(parts)))
    return records


def accession_from_header(header: str) -> str:
    token = header.split("|")
    if len(token) >= 2 and re.fullmatch(r"[A-Z0-9]+", token[1]):
        return token[1]
    return header.split()[0]


def is_standard_sequence(seq: str) -> bool:
    return bool(seq) and set(seq).issubset(STANDARD_AA)


def fetch_bucket(bucket: dict[str, object], oversample: int = 3) -> list[tuple[str, str]]:
    name = str(bucket["name"])
    min_len = int(bucket["min_len"])
    max_len = int(bucket["max_len"])
    count = int(bucket["count"])
    fetch_size = count * oversample

    query = (
        f"(reviewed:true) AND (length:[{min_len} TO {max_len}]) "
        f"AND (organism_id:9606)"
    )
    params = {
        "format": "fasta",
        "query": query,
        "size": str(fetch_size),
        "sort": "accession asc",
    }
    url = f"{UNIPROT_STREAM}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        text = resp.read().decode("utf-8")

    selected: list[tuple[str, str]] = []
    for header, seq in parse_fasta_text(text):
        if not is_standard_sequence(seq):
            continue
        acc = accession_from_header(header)
        record_name = f"{name}_{acc}"
        selected.append((record_name, seq))
        if len(selected) >= count:
            break

    if len(selected) < count:
        raise RuntimeError(
            f"bucket {name}: requested {count} sequences, got {len(selected)} "
            f"(query={query!r})"
        )
    return selected


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for name, seq in records:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    all_records: list[tuple[str, str]] = []
    bucket_manifest: list[dict[str, object]] = []

    for bucket in BUCKETS:
        records = fetch_bucket(bucket)
        all_records.extend(records)
        bucket_manifest.append(
            {
                "bucket": bucket["name"],
                "min_len": bucket["min_len"],
                "max_len": bucket["max_len"],
                "count": bucket["count"],
                "query": (
                    f"(reviewed:true) AND (length:[{bucket['min_len']} TO {bucket['max_len']}]) "
                    f"AND (organism_id:9606)"
                ),
                "records": [
                    {"name": name, "length": len(seq), "accession": name.split("_", 1)[1]}
                    for name, seq in records
                ],
            }
        )
        print(f"{bucket['name']}: selected {len(records)} sequences")

    write_fasta(args.output, all_records)
    manifest = {
        "source": "UniProt Swiss-Prot",
        "source_url": "https://www.uniprot.org/",
        "license": "CC BY 4.0",
        "api": UNIPROT_STREAM,
        "selection": {
            "reviewed_only": True,
            "organism": "Homo sapiens (9606)",
            "alphabet": "20 standard amino acids only",
            "sort": "accession asc",
        },
        "total_sequences": len(all_records),
        "buckets": bucket_manifest,
        "output": {
            "fasta": str(args.output),
            "sha256": sha256_file(args.output),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {len(all_records)} sequences to {args.output}")
    print(f"Wrote manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

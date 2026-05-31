#!/usr/bin/env python3
"""Extract a small real ProteinGym DMS substitution subset for Milestone 8D.

ProteinGym distributes the processed substitution benchmark as a large zip. This
script either consumes a local copy of that archive or downloads it when
--download is passed, then writes a filtered, versioned subset plus a manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import is_standard_sequence, sha256_file  # noqa: E402

PROTEINGYM_VERSION = "v1.3"
ARCHIVE_NAME = "DMS_ProteinGym_substitutions.zip"
DEFAULT_URL = f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{PROTEINGYM_VERSION}/{ARCHIVE_NAME}"
DEFAULT_ARCHIVE = ROOT / "benchmarks" / "downloads" / ARCHIVE_NAME
DEFAULT_OUTPUT_DIR = ROOT / "benchmarks" / "proteingym_subset"
DEFAULT_MANIFEST = ROOT / "benchmarks" / "datasets" / "proteingym_subset_manifest.json"

CSV_FIELDS = [
    "assay",
    "mutant",
    "target_sequence",
    "mutated_sequence",
    "DMS_score",
    "DMS_score_bin",
]


def download_archive(url: str, path: Path, *, insecure: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    context = ssl._create_unverified_context() if insecure else None
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120, context=context) as resp, tmp.open(
        "wb"
    ) as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(path)
    print(f"Wrote {path}")


def row_score(row: dict[str, str]) -> float | None:
    try:
        return float(row["DMS_score"])
    except (KeyError, TypeError, ValueError):
        return None


def normalise_row(
    assay: str, target_sequence: str, row: dict[str, str]
) -> dict[str, str] | None:
    seq = row.get("mutated_sequence", "").strip().upper()
    score = row_score(row)
    if not seq or score is None:
        return None
    return {
        "assay": assay,
        "mutant": row.get("mutant", "").strip(),
        "target_sequence": target_sequence,
        "mutated_sequence": seq,
        "DMS_score": f"{score:.8g}",
        "DMS_score_bin": row.get("DMS_score_bin", "").strip(),
    }


def infer_target_sequence(rows: list[dict[str, str]]) -> str | None:
    pattern = re.compile(r"^([A-Z])(\d+)([A-Z])$")
    for row in rows:
        mutant = row.get("mutant", "").strip()
        seq = row.get("mutated_sequence", "").strip().upper()
        if not mutant or mutant.upper() == "WT" or not seq:
            continue
        candidate = list(seq)
        ok = True
        for part in mutant.split(":"):
            match = pattern.match(part)
            if not match:
                ok = False
                break
            wt, pos_text, mut = match.groups()
            pos = int(pos_text) - 1
            if pos < 0 or pos >= len(candidate) or candidate[pos] != mut:
                ok = False
                break
            candidate[pos] = wt
        if ok:
            target = "".join(candidate)
            if is_standard_sequence(target):
                return target
    return None


def read_filtered_rows(
    zf: zipfile.ZipFile,
    member: str,
    *,
    target_sequence: str | None,
    max_rows: int,
    max_length: int,
) -> list[dict[str, str]]:
    assay = Path(member).stem
    raw_rows: list[dict[str, str]] = []
    seen_sequences: set[str] = set()

    with zf.open(member) as raw:
        text = (line.decode("utf-8") for line in raw)
        reader = csv.DictReader(text)
        if not reader.fieldnames:
            return []
        required = {"mutated_sequence", "DMS_score"}
        if not required.issubset(set(reader.fieldnames)):
            return []

        for row in reader:
            seq = row.get("mutated_sequence", "").strip().upper()
            score = row_score(row)
            if not seq or score is None:
                continue
            if len(seq) > max_length or not is_standard_sequence(seq):
                continue
            if seq in seen_sequences:
                continue
            raw_rows.append(row | {"mutated_sequence": seq})
            seen_sequences.add(seq)
            if len(raw_rows) >= max_rows:
                break

    target = target_sequence or infer_target_sequence(raw_rows)
    if not target or len(target) > max_length:
        return []
    rows: list[dict[str, str]] = []
    for row in raw_rows:
        out = normalise_row(assay, target, row)
        if out is not None:
            rows.append(out)
    return rows


def load_target_sequences(zf: zipfile.ZipFile) -> dict[str, str]:
    targets: dict[str, str] = {}
    for member in zf.namelist():
        if not member.lower().endswith(".csv"):
            continue
        with zf.open(member) as raw:
            text = (line.decode("utf-8") for line in raw)
            reader = csv.DictReader(text)
            if not reader.fieldnames or "target_seq" not in reader.fieldnames:
                continue
            for row in reader:
                target = row.get("target_seq", "").strip().upper()
                if not target or not is_standard_sequence(target):
                    continue
                keys = {
                    row.get("DMS_id", "").strip(),
                    row.get("DMS_filename", "").strip(),
                    row.get("assay", "").strip(),
                    row.get("assay_id", "").strip(),
                }
                for key in keys:
                    if not key:
                        continue
                    targets[key] = target
                    targets[Path(key).stem] = target
    return targets


def assay_matches(member: str, assays: set[str]) -> bool:
    if not assays:
        return True
    stem = Path(member).stem
    name = Path(member).name
    return stem in assays or name in assays or member in assays


def write_subset(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assay_entry(
    assay: str,
    member: str,
    rows: list[dict[str, str]],
    output_dir: Path,
) -> dict[str, Any]:
    target_sequence = rows[0]["target_sequence"]
    output_path = output_dir / f"{assay}.csv"
    write_subset(output_path, rows)
    return {
        "assay": assay,
        "source_member": member,
        "target_sequence_length": len(target_sequence),
        "target_sequence_sha256": hashlib.sha256(target_sequence.encode()).hexdigest(),
        "rows": len(rows),
        "max_sequence_length": max(len(row["mutated_sequence"]) for row in rows),
        "output": str(output_path),
        "sha256": sha256_file(output_path),
    }


def extract_assay_rows(
    zf: zipfile.ZipFile,
    member: str,
    *,
    target_sequences: dict[str, str],
    max_rows: int,
    max_length: int,
) -> list[dict[str, str]] | None:
    target_sequence = target_sequences.get(Path(member).name) or target_sequences.get(
        Path(member).stem
    )
    if target_sequence and (
        len(target_sequence) > max_length or not is_standard_sequence(target_sequence)
    ):
        return None

    rows = read_filtered_rows(
        zf,
        member,
        target_sequence=target_sequence,
        max_rows=max_rows,
        max_length=max_length,
    )
    return rows or None


def select_assays(
    archive: Path,
    *,
    requested_assays: set[str],
    max_assays: int,
    max_rows: int,
    min_rows: int,
    max_length: int,
    target_total_rows: int | None,
    selection_mode: str,
    prefer_short_length: bool,
    output_dir: Path,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as zf:
        target_sequences = load_target_sequences(zf)
        members = sorted(
            name
            for name in zf.namelist()
            if name.lower().endswith(".csv") and assay_matches(name, requested_assays)
        )
        if not members:
            raise RuntimeError("no ProteinGym CSV assay files matched the selection")

        if target_total_rows is not None and selection_mode == "single-assay":
            best: tuple[int, str, list[dict[str, str]]] | None = None
            for member in members:
                rows = extract_assay_rows(
                    zf,
                    member,
                    target_sequences=target_sequences,
                    max_rows=target_total_rows,
                    max_length=max_length,
                )
                if rows is None or len(rows) < target_total_rows:
                    continue
                target_len = len(rows[0]["target_sequence"])
                if best is None or target_len < best[0]:
                    best = (target_len, member, rows[:target_total_rows])

            if best is None:
                print(
                    "No single assay reached "
                    f"{target_total_rows} filtered variants; falling back to multi-assay.",
                    file=sys.stderr,
                )
                selection_mode = "multi-assay"
            else:
                target_len, member, rows = best
                assay = Path(member).stem
                entry = assay_entry(assay, member, rows, output_dir)
                selected.append(entry)
                print(
                    f"Selected single assay {assay}: {len(rows)} variants "
                    f"(target length {target_len}) -> {entry['output']}"
                )
                return selected

        candidates: list[tuple[int, int, str, list[dict[str, str]]]] = []
        for member in members:
            row_cap = max_rows
            if target_total_rows is not None and selection_mode == "multi-assay":
                row_cap = max(max_rows, target_total_rows)
            rows = extract_assay_rows(
                zf,
                member,
                target_sequences=target_sequences,
                max_rows=row_cap,
                max_length=max_length,
            )
            if rows is None or len(rows) < min_rows:
                continue
            target_len = len(rows[0]["target_sequence"])
            candidates.append((target_len, -len(rows), member, rows))

        if prefer_short_length:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        else:
            candidates.sort(key=lambda item: (item[1], item[0], item[2]))

        total_rows = 0
        for target_len, _neg_count, member, rows in candidates:
            if len(selected) >= max_assays:
                break
            if target_total_rows is not None and total_rows >= target_total_rows:
                break

            if target_total_rows is not None and selection_mode == "multi-assay":
                remaining = target_total_rows - total_rows
                rows = rows[:remaining]

            assay = Path(member).stem
            entry = assay_entry(assay, member, rows, output_dir)
            selected.append(entry)
            total_rows += len(rows)
            print(
                f"Selected {assay}: {len(rows)} variants "
                f"(target length {target_len}) -> {entry['output']}"
            )

    if not selected:
        raise RuntimeError(
            f"no assays had at least {min_rows} standard-AA rows with length <= {max_length}"
        )
    if target_total_rows is not None and total_rows < target_total_rows:
        raise RuntimeError(
            f"only collected {total_rows} variants across {len(selected)} assays; "
            f"target was {target_total_rows}"
        )
    return selected


def write_manifest(
    path: Path,
    *,
    archive: Path,
    url: str,
    selected: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "source": "ProteinGym DMS substitution benchmark",
        "source_url": url,
        "version": PROTEINGYM_VERSION,
        "archive": {
            "path": str(archive),
            "sha256": sha256_file(archive),
        },
        "license_note": "ProteinGym dataset terms are inherited from the upstream benchmark.",
        "selection": {
            "requested_assays": args.assay or [],
            "max_assays": args.max_assays,
            "max_rows_per_assay": args.max_rows,
            "min_rows_per_assay": args.min_rows,
            "max_sequence_length": args.max_length,
            "target_total_rows": args.target_total_rows,
            "selection_mode": args.selection_mode,
            "prefer_short_length": args.prefer_short_length,
            "total_variants": sum(item["rows"] for item in selected),
            "alphabet": "20 standard amino acids only",
        },
        "assays": selected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote manifest to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--insecure-download",
        action="store_true",
        help="Disable TLS certificate verification for hosts with broken local CA chains.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--assay", action="append", help="Assay CSV filename or stem to extract"
    )
    parser.add_argument("--max-assays", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument(
        "--target-total-rows",
        type=int,
        help=(
            "Collect at least this many variants total. Default selection uses one "
            "short assay when possible so Spearman remains assay-local."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=["single-assay", "multi-assay"],
        default="single-assay",
        help="How to satisfy --target-total-rows.",
    )
    parser.add_argument(
        "--prefer-short-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer shorter target sequences for faster local embedding.",
    )
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    archive = args.archive.expanduser()
    if not archive.is_absolute():
        archive = ROOT / archive

    if not archive.is_file():
        if not args.download:
            print(
                "Missing ProteinGym archive. Re-run with --download or pass "
                f"--archive PATH. Expected: {archive}",
                file=sys.stderr,
            )
            print(f"Download URL: {args.url}", file=sys.stderr)
            return 1
        download_archive(args.url, archive, insecure=args.insecure_download)

    requested = set(args.assay or [])
    max_assays = args.max_assays
    if args.target_total_rows is not None and args.selection_mode == "multi-assay":
        max_assays = max(max_assays, 50)

    selected = select_assays(
        archive,
        requested_assays=requested,
        max_assays=max_assays,
        max_rows=args.max_rows,
        min_rows=args.min_rows,
        max_length=args.max_length,
        target_total_rows=args.target_total_rows,
        selection_mode=args.selection_mode,
        prefer_short_length=args.prefer_short_length,
        output_dir=args.output_dir,
    )
    write_manifest(
        args.manifest, archive=archive, url=args.url, selected=selected, args=args
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

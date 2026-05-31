#!/usr/bin/env python3
"""Milestone 16: upload ESM-C GGUF models + model card to HuggingFace.

Uploads every `*.gguf` in a models directory (plus an optional model card) to a
HuggingFace model repo using `huggingface_hub`. Supports a `--dry-run` that
prints the full upload plan with file sizes and checksums without touching the
network, so the pipeline can be verified offline.

Usage:
    # Offline plan (no network, no token needed)
    python tools/upload_to_hf.py \\
        --repo-id <user>/esmc-300m-gguf \\
        --models-dir ./models \\
        --model-card results/reproduction_bundle/model_card/README.md \\
        --dry-run

    # Real upload (requires HF_TOKEN env var or `huggingface-cli login`)
    python tools/upload_to_hf.py \\
        --repo-id <user>/esmc-300m-gguf \\
        --models-dir ./models \\
        --model-card results/reproduction_bundle/model_card/README.md \\
        --create
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def human_mib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MiB"


def collect_gguf(models_dir: Path, only: list[str] | None) -> list[Path]:
    files = sorted(models_dir.glob("*.gguf"))
    if only:
        wanted = {name.lower() for name in only}
        files = [f for f in files if f.name.lower() in wanted]
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo, e.g. user/esmc-300m-gguf")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--gguf", action="append", default=None,
                        help="Specific GGUF filename to upload (repeatable). Default: all *.gguf")
    parser.add_argument("--model-card", type=Path, default=None,
                        help="Markdown file uploaded as the repo README.md")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HF token (default: $HF_TOKEN or cached login)")
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument("--create", action="store_true", help="Create the repo if it does not exist")
    parser.add_argument("--commit-message", default="Upload ESM-C GGUF models (esmc.cpp)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the upload plan without any network calls")
    args = parser.parse_args()

    models_dir = args.models_dir.expanduser().resolve()
    if not models_dir.is_dir():
        print(f"error: models dir not found: {models_dir}", file=sys.stderr)
        return 1

    gguf_files = collect_gguf(models_dir, args.gguf)
    if not gguf_files:
        print(f"error: no GGUF files matched in {models_dir}", file=sys.stderr)
        return 1

    uploads: list[tuple[Path, str]] = [(f, f.name) for f in gguf_files]
    if args.model_card is not None:
        card = args.model_card.expanduser().resolve()
        if not card.is_file():
            print(f"error: model card not found: {card}", file=sys.stderr)
            return 1
        uploads.append((card, "README.md"))

    total_bytes = sum(f.stat().st_size for f, _ in uploads)
    print(f"Repo: {args.repo_id} ({'private' if args.private else 'public'})")
    print(f"Files: {len(uploads)} ({human_mib(total_bytes)} total)")
    for src, dest in uploads:
        size = src.stat().st_size
        print(f"  {dest:32s} {human_mib(size):>12s}  sha256={sha256_file(src)[:16]}  <- {src}")

    if args.dry_run:
        print("\n[dry-run] no files uploaded. Re-run without --dry-run to upload.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: huggingface_hub not installed (pip install huggingface_hub)", file=sys.stderr)
        return 1

    api = HfApi(token=args.token)
    if args.create:
        api.create_repo(repo_id=args.repo_id, repo_type="model",
                        private=args.private, exist_ok=True)
        print(f"Ensured repo exists: {args.repo_id}")

    for src, dest in uploads:
        print(f"Uploading {dest} ({human_mib(src.stat().st_size)}) ...")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"{args.commit_message}: {dest}",
        )
    print(f"\nDone. https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

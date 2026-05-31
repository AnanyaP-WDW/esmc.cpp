# ProteinGym Subset

This directory is populated by `benchmarks/fetch_proteingym_subset.py` for
Milestone 8D. The script extracts a small real ProteinGym DMS substitution assay
subset from the official upstream archive and writes one filtered CSV per assay.

The generated CSV files are intentionally small enough for local M1 benchmarking
and include:

- `target_sequence`: wild-type assay sequence
- `mutated_sequence`: full mutant sequence
- `DMS_score`: experimental fitness score
- `mutant`: ProteinGym substitution label when present

Run:

```bash
.venv/bin/python benchmarks/fetch_proteingym_subset.py --download
```

Use `--insecure-download` only if the upstream host fails TLS verification on
the local Python install.

or pass a local archive:

```bash
.venv/bin/python benchmarks/fetch_proteingym_subset.py \
  --archive /path/to/DMS_ProteinGym_substitutions.zip
```

The provenance manifest is written to
`benchmarks/datasets/proteingym_subset_manifest.json`.

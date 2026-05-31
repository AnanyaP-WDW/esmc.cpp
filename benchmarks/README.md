# esmc.cpp Benchmarks

This directory contains paper-oriented benchmark harnesses introduced for
milestones 8A and 8B.

Shared 300M configuration lives in:

```text
benchmarks/config_300m.json
```

## Correctness dataset (UniProt Swiss-Prot)

`benchmarks/sequences_correctness.fasta` contains 100 reviewed human Swiss-Prot
sequences from [UniProt](https://www.uniprot.org/) (CC BY 4.0), split into:

| Bucket | Length range | Count |
|--------|--------------|-------|
| small  | 10–50 aa     | 34    |
| medium | 100–300 aa   | 33    |
| large  | 500–1500 aa  | 33    |

Provenance and selected accessions are recorded in:

```text
benchmarks/datasets/uniprot_correctness_manifest.json
```

To refresh the FASTA from UniProt (requires network):

```bash
.venv/bin/python benchmarks/fetch_uniprot_sequences.py
```

Regenerate benchmark reference embeddings after updating the FASTA:

```bash
.venv/bin/python tests/generate_reference.py \
  --fasta benchmarks/sequences_correctness.fasta \
  --output tests/reference_embeddings.npz
```

Quick milestone 6/7 smoke tests still use `tests/sequences_smoke.fasta` (3
fixed sequences) and `tests/reference_embeddings_smoke.npz`.

## Milestone 8A: Result Schema and Runner Layout

Dry-run the correctness harness to write a manifest with host, git, model,
backend, precision, sequence-set, and reference metadata:

```bash
.venv/bin/python benchmarks/correctness.py --config benchmarks/config_300m.json --dry-run
```

Expected artifact:

```text
results/manifest.json
```

## Milestone 8B: Numerical Correctness Harness

Run the available 300M F16 model:

```bash
.venv/bin/python benchmarks/correctness.py \
  --model f16=models/esmc-300m-f16.gguf \
  --backend cpu
```

Expected artifacts:

```text
results/correctness_300m.json
results/correctness_300m.csv
```

When a Q8_0 GGUF is available, include it in the same run:

```bash
.venv/bin/python benchmarks/correctness.py \
  --model f16=models/esmc-300m-f16.gguf \
  --model q8_0=models/esmc-300m-Q8_0.gguf \
  --backend cpu
```

The default model matrix includes `f16` and `q8_0`, and fails if either model is
missing. Use `--allow-missing-models` only for harness smoke tests.

## Milestone 8C: 300M Quantization Matrix

Quantize the 300M F32 GGUF into Q8_0, Q4_K_M, Q4_K_S using the in-tree
`esmc-quantize` binary (CMake target `esmc-quantize`):

```bash
cmake --build build --target esmc-quantize

./build/esmc-quantize models/esmc-300m-f32.gguf models/esmc-300m-Q8_0.gguf   Q8_0
./build/esmc-quantize models/esmc-300m-f32.gguf models/esmc-300m-Q4_K_M.gguf Q4_K_M
./build/esmc-quantize models/esmc-300m-f32.gguf models/esmc-300m-Q4_K_S.gguf Q4_K_S
```

`esmc-quantize` is a small ggml-only tool (see `examples/quantize/main.cpp`).
It calls `ggml_quantize_chunk` directly so the custom `general.architecture =
"esmc"` GGUF is quantizable without registering the architecture in llama.cpp.

ESMC-300M has `n_embd = 960`, which is not a multiple of `QK_K = 256`. As a
result, only `ffn_down.weight` (`n_per_row = 2560`) is eligible for k-quant
block quantization. The other weight matrices use legacy 32-block fallbacks
(Q5_0/Q8_0) chosen to preserve >0.995 aggregate mean cosine. See
`lab_manual.md` §5.7 for per-tensor mix and pass-rate results.

Run the full matrix:

```bash
.venv/bin/python benchmarks/correctness.py
```

Outputs `results/correctness_300m.{json,csv}` with one row per
`(precision, sequence)`. Thresholds in `benchmarks/correctness.py` follow plan
§13.2: F16/Q8_0 mean cosine > 0.999, Q4_K_* mean cosine > 0.995.

## Milestone 8D: ProteinGym Downstream Harness

The downstream harness uses a small real ProteinGym DMS substitution subset,
cached mean-pooled embeddings, and Spearman correlation against `DMS_score`.
PyTorch embeddings are generated from `tests/ref_forward.py`; GGUF embeddings
reuse the same `esmc-embed` subprocess path as the correctness harness.

First extract a subset from the official ProteinGym v1.3 substitution archive:

```bash
# Download the large upstream archive only when needed.
.venv/bin/python benchmarks/fetch_proteingym_subset.py --download \
  --max-assays 1 \
  --max-rows 64 \
  --max-length 512
```

If the upstream host fails TLS verification on a local Python install, retry with
the explicit `--insecure-download` flag and keep the manifest checksum.

If the archive already exists locally, pass it explicitly:

```bash
.venv/bin/python benchmarks/fetch_proteingym_subset.py \
  --archive /path/to/DMS_ProteinGym_substitutions.zip \
  --max-assays 1 \
  --max-rows 64 \
  --max-length 512
```

Expected subset artifacts:

```text
benchmarks/proteingym_subset/*.csv
benchmarks/datasets/proteingym_subset_manifest.json
```

Run the benchmark:

```bash
.venv/bin/python benchmarks/downstream.py \
  --config benchmarks/config_downstream_300m.json
```

Expected result artifacts:

```text
results/downstream_300m.json
results/downstream_300m.csv
results/downstream_cache/300m/
```

The CSV reports `pytorch`, `f16`, `q8_0`, `q4_k_m`, and `q4_k_s` Spearman
metrics per assay plus `delta_from_pytorch`. The initial ProteinGym tolerance is
`abs(delta_from_pytorch) <= 0.01`.

### 1000-variant run (recommended for stronger Spearman)

Use one short assay with 1000 variants so Spearman stays assay-local:

```bash
.venv/bin/python benchmarks/fetch_proteingym_subset.py \
  --archive benchmarks/downloads/DMS_ProteinGym_substitutions.zip \
  --target-total-rows 1000 \
  --selection-mode single-assay \
  --output-dir benchmarks/proteingym_subset_1k \
  --manifest benchmarks/datasets/proteingym_subset_1k_manifest.json

.venv/bin/python benchmarks/downstream.py \
  --config benchmarks/config_downstream_300m_1k.json
```

Expected artifacts:

```text
benchmarks/proteingym_subset_1k/*.csv
results/downstream_300m_1k.{json,csv}
results/downstream_cache/300m_1k/
```

### 10-assay / 10k-variant run

Select 10 assays with 1000 variants each. Metrics remain assay-local; the summary
CSV has one row per `(assay, precision, metric)`.

```bash
.venv/bin/python benchmarks/fetch_proteingym_subset.py \
  --archive benchmarks/downloads/DMS_ProteinGym_substitutions.zip \
  --selection-mode multi-assay \
  --max-assays 10 \
  --max-rows 1000 \
  --min-rows 1000 \
  --max-length 512 \
  --output-dir benchmarks/proteingym_subset_10k \
  --manifest benchmarks/datasets/proteingym_subset_10k_manifest.json

.venv/bin/python benchmarks/downstream.py \
  --config benchmarks/config_downstream_300m_10k.json
```

`benchmarks/config_downstream_300m_10k.json` emits:

```text
spearman, pearson, kendall_tau_b, top10_overlap, bottom10_overlap
```

Expected artifacts:

```text
benchmarks/proteingym_subset_10k/*.csv
benchmarks/datasets/proteingym_subset_10k_manifest.json
results/downstream_300m_10k.{json,csv}
results/downstream_cache/300m_10k/
```

## Milestone 8E: Throughput Harness

Build the dedicated C++ timing binary, then run the Python orchestrator. The
orchestrator launches CPU, Metal, PyTorch CPU, and PyTorch MPS in separate worker
processes and writes CSV/JSON results. The default 300M config runs `f16`,
`q8_0`, `q4_k_m`, and `q4_k_s` GGUFs for `esmc.cpp`; PyTorch defaults to `f32`
unless `pytorch_dtype` or `--pytorch-dtype` is changed.

```bash
cmake --build build --target esmc-bench

.venv/bin/python benchmarks/throughput.py \
  --config benchmarks/config_throughput_300m.json
```

For a 1000-iteration run with a timestamped console log:

```bash
mkdir -p results/logs && \
RUN_ID="$(hostname -s)_$(date +%Y%m%d_%H%M%S)" && \
.venv/bin/python benchmarks/throughput.py \
  --config benchmarks/config_throughput_300m.json \
  --iterations 1000 \
  --output-prefix "results/throughput_${RUN_ID}" \
  2>&1 | tee "results/logs/throughput_${RUN_ID}.log"
```

Expected artifacts:

```text
results/throughput_<host>_<date>.json
results/throughput_<host>_<date>.csv
results/logs/throughput_<host>_<date>.log
```

To experiment with a PyTorch FP16 baseline, add `--pytorch-dtype f16` to the
same command. This is most useful for `pytorch_mps`; CPU FP16 is often slower
than CPU FP32 because many operations are not accelerated for half precision.

## Milestone 8F: Memory-Footprint Harness

Build `esmc-embed`, then run the memory orchestrator. The default 300M config
measures every local GGUF precision (`f32`, `f16`, `q8_0`, `q4_k_m`, `q4_k_s`)
on CPU and Metal, plus PyTorch CPU/MPS baselines. Each model / backend /
sequence bucket runs in a fresh process under `/usr/bin/time -l`; the harness
records peak resident set size, model file size, success/failure, and the 16 GB
machine-budget pass/fail flag.

```bash
cmake --build build --target esmc-embed

.venv/bin/python benchmarks/memory.py \
  --config benchmarks/config_memory_300m.json
```

For a timestamped console log:

```bash
mkdir -p results/logs && \
RUN_ID="$(hostname -s)_$(date +%Y%m%d_%H%M%S)" && \
.venv/bin/python benchmarks/memory.py \
  --config benchmarks/config_memory_300m.json \
  --output-prefix "results/memory_${RUN_ID}" \
  2>&1 | tee "results/logs/memory_${RUN_ID}.log"
```

Expected artifacts:

```text
results/memory_<host>_<date>.json
results/memory_<host>_<date>.csv
results/logs/memory_<host>_<date>.log
```


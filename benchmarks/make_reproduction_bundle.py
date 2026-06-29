#!/usr/bin/env python3
"""Milestone 16: assemble the esmc.cpp reproduction bundle.

The bundle is a single self-describing directory that gathers everything a
reader of the paper needs to reproduce the headline numbers:

- GGUF model files (linked, copied, or referenced by checksum manifest)
- benchmark CSV/JSON artifacts (correctness, throughput, memory, downstream)
- plots (SVG; PNG too if matplotlib is available)
- paper-ready result tables (Markdown + LaTeX)
- a HuggingFace model card
- a top-level MANIFEST.json (git hash, host, file inventory + checksums)
- a bundle README explaining the contents and reproduction steps

Verification target (plan.md milestone 16): the reproduction bundle includes
GGUF files, benchmark CSV/JSON, plots, and paper-ready result tables.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from common import git_info, host_info, sha256_file, utc_now_iso, write_json  # noqa: E402
from paper_artifacts import write_svg_grouped_bar_chart  # noqa: E402

RESULTS = ROOT / "results"
MODELS_DIR = ROOT / "models"

# Default inputs (latest measured artifacts from the experiment log).
DEFAULT_CORRECTNESS = RESULTS / "correctness_300m.csv"
DEFAULT_THROUGHPUT = RESULTS / "throughput_m4_max.csv"
DEFAULT_MEMORY = RESULTS / "memory_m4_max_20260626_153823.csv"
DEFAULT_DOWNSTREAM = RESULTS / "downstream_300m_10k.csv"
DEFAULT_OUT_DIR = RESULTS / "reproduction_bundle"

PRECISION_ORDER = ["f16", "q8_0", "q4_k_m", "q4_k_s"]
PRECISION_LABEL = {"f16": "F16", "q8_0": "Q8_0", "q4_k_m": "Q4_K_M", "q4_k_s": "Q4_K_S", "f32": "F32"}
HF_DEFAULT_REPO = "AnanyaPathak/esmc-300m-gguf"
GITHUB_REPO = "https://github.com/AnanyaP-WDW/esmc.cpp"
UPSTREAM_MODEL = "https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12"
CAMBRIAN_OPEN_LICENSE = "https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement"
ACCEPTABLE_USE_POLICY = "https://www.evolutionaryscale.ai/policies/acceptable-use-policy"


# ── small helpers ────────────────────────────────────────────────────────────


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def fmt(v: float, ndigits: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{ndigits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep, *body])


def latex_table(caption: str, label: str, headers: list[str], rows: list[list[str]]) -> str:
    col_spec = "l" + "r" * (len(headers) - 1)

    def esc(s: str) -> str:
        return s.replace("_", r"\_").replace("%", r"\%")

    out = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{esc(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(esc(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    out += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def write_table(out_dir: Path, name: str, caption: str, label: str,
                headers: list[str], rows: list[list[str]]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{name}.md"
    tex_path = out_dir / f"{name}.tex"
    md_path.write_text(f"### {caption}\n\n{md_table(headers, rows)}\n")
    tex_path.write_text(latex_table(caption, label, headers, rows) + "\n")
    return {"md": str(md_path.relative_to(ROOT)), "tex": str(tex_path.relative_to(ROOT))}


# ── correctness aggregation ──────────────────────────────────────────────────


def correctness_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = [r for r in rows if not r.get("error")]
    summary: list[dict[str, Any]] = []
    for p in PRECISION_ORDER:
        pr = [r for r in rows if r["precision"] == p]
        if not pr:
            continue
        mean_cos = [to_float(r, "mean_cosine") for r in pr]
        min_cos = [to_float(r, "min_cosine") for r in pr]
        l2 = [to_float(r, "mean_pool_l2") for r in pr]
        passed = sum(1 for r in pr if r.get("passed") == "True")
        summary.append({
            "precision": p,
            "sequences": len(pr),
            "agg_mean_cosine": statistics.fmean(mean_cos),
            "min_mean_cosine": min(mean_cos),
            "min_min_cosine": min(min_cos),
            "agg_mean_pool_l2": statistics.fmean(l2),
            "max_mean_pool_l2": max(l2),
            "passed": passed,
            "total": len(pr),
            "pass_rate": passed / len(pr),
        })
    return summary


def correctness_table(summary: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    headers = [
        "Precision", "Seqs", "Aggregate mean cos", "Worst mean cos",
        "Worst min cos", "Mean pool L2", "Pass rate",
    ]
    rows = [[
        PRECISION_LABEL.get(s["precision"], s["precision"]),
        str(s["sequences"]),
        fmt(s["agg_mean_cosine"], 5),
        fmt(s["min_mean_cosine"], 5),
        fmt(s["min_min_cosine"], 5),
        fmt(s["agg_mean_pool_l2"], 4),
        f"{s['passed']}/{s['total']}",
    ] for s in summary]
    return write_table(
        out_dir, "correctness_summary",
        "Numerical correctness vs PyTorch (300M, 100-sequence Swiss-Prot set)",
        "tab:correctness", headers, rows,
    )


# ── throughput aggregation ───────────────────────────────────────────────────


def throughput_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    buckets = ["short", "medium", "long"]

    def find(pred: Callable[[dict[str, str]], bool], bucket: str) -> dict[str, str] | None:
        return next((r for r in rows if r["sequence_bucket"] == bucket and pred(r)), None)

    esmc = [r for r in rows if r["implementation"] == "esmc.cpp"]
    out: list[dict[str, Any]] = []
    for b in buckets:
        candidates = [r for r in esmc if r["sequence_bucket"] == b]
        best = max(candidates, key=lambda r: to_float(r, "seq_per_s")) if candidates else None
        pt_cpu = find(lambda r: r["backend"] == "pytorch_cpu", b)
        pt_mps = find(lambda r: r["backend"] == "pytorch_mps", b)
        s_best = to_float(best, "seq_per_s") if best else math.nan
        s_cpu = to_float(pt_cpu, "seq_per_s") if pt_cpu else math.nan
        s_mps = to_float(pt_mps, "seq_per_s") if pt_mps else math.nan
        out.append({
            "bucket": b,
            "tokens": int(to_float(best, "token_count", 0)) if best else 0,
            "best_config": f"{best['backend']}/{best['precision']}" if best else "n/a",
            "best_seq_per_s": s_best,
            "pt_cpu_seq_per_s": s_cpu,
            "pt_mps_seq_per_s": s_mps,
            "vs_pt_cpu": s_best / s_cpu if s_cpu and not math.isnan(s_cpu) and s_cpu > 0 else math.nan,
            "vs_pt_mps": s_best / s_mps if s_mps and not math.isnan(s_mps) and s_mps > 0 else math.nan,
        })
    return out


def throughput_table(summary: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    headers = ["Bucket", "Tokens", "Best esmc.cpp", "seq/s"]
    rows = [[
        s["bucket"], str(s["tokens"]), s["best_config"],
        fmt(s["best_seq_per_s"], 2),
    ] for s in summary]
    return write_table(
        out_dir, "throughput_summary",
        "Throughput by sequence bucket (300M, seq/s)",
        "tab:throughput", headers, rows,
    )


# ── memory aggregation ───────────────────────────────────────────────────────


def memory_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    long_rows = [r for r in rows if r["sequence_bucket"] == "long"]
    long_rows.sort(key=lambda r: to_float(r, "peak_rss_mib"), reverse=True)
    out: list[dict[str, Any]] = []
    for r in long_rows:
        out.append({
            "config": f"{r['implementation']}/{r['backend']}/{r['precision']}",
            "peak_rss_mib": to_float(r, "peak_rss_mib"),
            "model_file_mib": to_float(r, "model_file_size_bytes") / (1024 * 1024),
            "budget_pass": r.get("budget_pass", ""),
        })
    return out


def memory_table(summary: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    headers = ["Config (long bucket)", "Peak RSS (MiB)", "Model file (MiB)", "<=36 GiB"]
    rows = [[
        s["config"], fmt(s["peak_rss_mib"], 0), fmt(s["model_file_mib"], 0),
        "yes" if s["budget_pass"] == "True" else "no",
    ] for s in summary]
    return write_table(
        out_dir, "memory_summary",
        "Peak resident memory on a 36 GB M4 Max (300M, long bucket)",
        "tab:memory", headers, rows,
    )


# ── downstream aggregation ───────────────────────────────────────────────────


def downstream_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = [r for r in rows if not r.get("error")]
    spearman = [r for r in rows if r["metric_name"] == "spearman"]
    out: list[dict[str, Any]] = []
    for p in PRECISION_ORDER:
        pr = [r for r in spearman if r["precision"] == p]
        if not pr:
            continue
        deltas = [abs(to_float(r, "delta_from_pytorch")) for r in pr]
        passed_all = sum(
            1 for r in rows if r["precision"] == p and r.get("passed") == "True"
        )
        total_all = sum(1 for r in rows if r["precision"] == p)
        out.append({
            "precision": p,
            "assays": len(pr),
            "mean_abs_spearman_delta": statistics.fmean(deltas),
            "max_abs_spearman_delta": max(deltas),
            "metric_pass": passed_all,
            "metric_total": total_all,
        })
    return out


def downstream_table(summary: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    headers = [
        "Precision", "Assays", "Mean abs Spearman delta", "Max abs Spearman delta",
        "Metric rows pass",
    ]
    rows = [[
        PRECISION_LABEL.get(s["precision"], s["precision"]),
        str(s["assays"]),
        fmt(s["mean_abs_spearman_delta"], 4),
        fmt(s["max_abs_spearman_delta"], 4),
        f"{s['metric_pass']}/{s['metric_total']}",
    ] for s in summary]
    return write_table(
        out_dir, "downstream_summary",
        "ProteinGym downstream Spearman delta vs PyTorch (300M, 10 assays / 10k variants)",
        "tab:downstream", headers, rows,
    )


# ── plots ────────────────────────────────────────────────────────────────────


def make_plots(
    correctness: list[dict[str, Any]],
    throughput: list[dict[str, Any]],
    memory: list[dict[str, Any]],
    downstream: list[dict[str, Any]],
    out_dir: Path,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    # Correctness aggregate mean cosine by precision.
    p = out_dir / "correctness_mean_cosine.svg"
    write_svg_grouped_bar_chart(
        path=p,
        title="Aggregate mean cosine vs PyTorch by precision (300M)",
        x_labels=[PRECISION_LABEL.get(s["precision"], s["precision"]) for s in correctness],
        series=[("aggregate mean cosine", [s["agg_mean_cosine"] for s in correctness])],
        y_label="cosine",
    )
    paths.append(str(p.relative_to(ROOT)))

    # Throughput best esmc.cpp (M4 Max; PyTorch baselines not re-benchmarked).
    p = out_dir / "throughput_seqps.svg"
    write_svg_grouped_bar_chart(
        path=p,
        title="Throughput by bucket (300M, seq/s)",
        x_labels=[s["bucket"] for s in throughput],
        series=[
            ("best esmc.cpp", [s["best_seq_per_s"] for s in throughput]),
        ],
        y_label="seq/s",
    )
    paths.append(str(p.relative_to(ROOT)))

    # Memory peak RSS (long bucket).
    p = out_dir / "memory_long_rss.svg"
    write_svg_grouped_bar_chart(
        path=p,
        title="Peak RSS on 36 GB M4 Max (300M, long bucket)",
        x_labels=[s["config"] for s in memory],
        series=[("peak RSS (MiB)", [s["peak_rss_mib"] for s in memory])],
        y_label="MiB",
    )
    paths.append(str(p.relative_to(ROOT)))

    # Downstream mean |Spearman delta|.
    p = out_dir / "downstream_spearman_delta.svg"
    write_svg_grouped_bar_chart(
        path=p,
        title="Mean |Spearman delta| vs PyTorch by precision (ProteinGym 10k)",
        x_labels=[PRECISION_LABEL.get(s["precision"], s["precision"]) for s in downstream],
        series=[("mean |delta|", [s["mean_abs_spearman_delta"] for s in downstream])],
        y_label="|delta|",
    )
    paths.append(str(p.relative_to(ROOT)))
    return paths


# ── GGUF gathering ───────────────────────────────────────────────────────────


def gather_gguf(models_dir: Path, dest_dir: Path, mode: str) -> list[dict[str, Any]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for gguf in sorted(models_dir.glob("*.gguf")):
        size = gguf.stat().st_size
        digest = sha256_file(gguf)
        target = dest_dir / gguf.name
        included = False
        if mode == "copy":
            shutil.copy2(gguf, target)
            included = True
        elif mode == "link":
            if target.exists() or target.is_symlink():
                target.unlink()
            os.symlink(gguf.resolve(), target)
            included = True
        # mode == "manifest": reference only.
        entries.append({
            "name": gguf.name,
            "size_bytes": size,
            "size_mib": round(size / (1024 * 1024), 1),
            "sha256": digest,
            "included": included,
            "include_mode": mode,
            "source_path": str(gguf),
        })
    return entries


def write_models_doc(entries: list[dict[str, Any]], dest_dir: Path, hf_repo: str) -> None:
    headers = ["File", "Size (MiB)", "sha256 (first 16)", "In bundle"]
    rows = [[
        e["name"], f"{e['size_mib']:.1f}", (e["sha256"] or "")[:16],
        e["include_mode"] if e["included"] else "manifest-only",
    ] for e in entries]
    lines = [
        "# ESM-C 300M GGUF models",
        "",
        f"HuggingFace repo (suggested): `{hf_repo}`",
        "",
        md_table(headers, rows),
        "",
        "## Download from HuggingFace",
        "",
        "```bash",
        f"huggingface-cli download {hf_repo} --local-dir ./models",
        "```",
        "",
        "## Verify checksums",
        "",
        "```bash",
        "shasum -a 256 models/*.gguf",
        "```",
        "",
        "Expected sha256:",
        "",
        "```",
        *[f"{e['sha256']}  {e['name']}" for e in entries],
        "```",
    ]
    (dest_dir / "MODELS.md").write_text("\n".join(lines) + "\n")


USE_HINT = {
    "esmc-300m-f16.gguf": "Highest fidelity; numerical reference.",
    "esmc-300m-f32.gguf": "Full precision; mainly the quantization source (largest).",
    "esmc-300m-q8_0.gguf": "**Recommended default** — near-F16 quality at ~half the size.",
    "esmc-300m-q4_k_m.gguf": "Smallest with good quality; best 4-bit choice.",
    "esmc-300m-q4_k_s.gguf": "Smallest footprint; lowest peak RAM.",
}


def write_model_card(entries: list[dict[str, Any]], dest_dir: Path, hf_repo: str,
                     correctness: list[dict[str, Any]],
                     throughput: list[dict[str, Any]],
                     memory: list[dict[str, Any]],
                     downstream: list[dict[str, Any]]) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_rows = [[
        e["name"], f"{e['size_mib']:.1f}", (e["sha256"] or "")[:16],
        USE_HINT.get(e["name"].lower(), ""),
    ] for e in entries]

    cos_rows = [[
        PRECISION_LABEL.get(s["precision"], s["precision"]),
        fmt(s["agg_mean_cosine"], 5),
        fmt(s["min_min_cosine"], 4),
        fmt(s["max_mean_pool_l2"], 4),
        f"{s['passed']}/{s['total']}",
    ] for s in correctness]

    thr_rows = [[
        s["bucket"], str(s["tokens"]), s["best_config"],
        fmt(s["best_seq_per_s"], 2),
    ] for s in throughput]

    down_rows = [[
        PRECISION_LABEL.get(s["precision"], s["precision"]),
        str(s["assays"]),
        fmt(s["mean_abs_spearman_delta"], 4),
        fmt(s["max_abs_spearman_delta"], 4),
        f"{s['metric_pass']}/{s['metric_total']}",
    ] for s in downstream]

    mem_lines: list[str] = []
    if memory:
        best_mem = min(memory, key=lambda s: s["peak_rss_mib"])
        worst_mem = max(memory, key=lambda s: s["peak_rss_mib"])
        n_pass = sum(1 for s in memory if s["budget_pass"] == "True")
        mem_lines = [
            f"- Lowest peak RAM: `{best_mem['config']}` at "
            f"{fmt(best_mem['peak_rss_mib'], 0)} MiB (long sequences).",
            f"- Highest peak RAM: `{worst_mem['config']}` at "
            f"{fmt(worst_mem['peak_rss_mib'], 0)} MiB.",
            f"- All {n_pass}/{len(memory)} measured configurations fit within a 36 GB machine.",
        ]

    example_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGY"
    card = [
        "---",
        "license: other",
        "license_name: cambrian-open-license-agreement",
        f"license_link: {CAMBRIAN_OPEN_LICENSE}",
        "library_name: gguf",
        "pipeline_tag: feature-extraction",
        "base_model: EvolutionaryScale/esmc-300m-2024-12",
        "tags:",
        "  - protein-language-model",
        "  - esm",
        "  - esmc",
        "  - embeddings",
        "  - protein-embeddings",
        "  - bioinformatics",
        "  - ggml",
        "  - gguf",
        "  - llama.cpp",
        "---",
        "",
        "# ESM-C 300M — GGUF (esmc.cpp)",
        "",
        "GGUF conversions of [ESM Cambrian](https://www.evolutionaryscale.ai/blog/esm-cambrian) "
        "(ESM-C) 300M, an encoder-only protein language model, for fast, low-memory "
        "**per-residue and per-sequence embeddings** on CPU and Apple Metal — with no "
        "Python or PyTorch needed at inference time.",
        "",
        f"- **Runtime:** [`esmc.cpp`]({GITHUB_REPO}) (C/C++ on ggml / llama.cpp)",
        f"- **Upstream model:** [EvolutionaryScale/esmc-300m-2024-12]({UPSTREAM_MODEL})",
        "- **Task:** feature extraction (protein embeddings)",
        "",
        "> [!IMPORTANT]",
        "> These files use a custom GGUF architecture (`general.architecture = \"esmc\"`) "
        "and are **not** loadable by stock `llama.cpp` / `llama-cli`. Use the "
        f"[`esmc.cpp`]({GITHUB_REPO}) runtime (the `esmc-embed` tool) shown below.",
        "",
        "## Which file should I download?",
        "",
        md_table(["File", "Size (MiB)", "sha256 (first 16)", "When to use"], files_rows),
        "",
        "If unsure, start with **`esmc-300m-Q8_0.gguf`** (near-identical to PyTorch at "
        "~half the size). Use **Q4_K_M** for the smallest deployment with good quality, "
        "or **F16** when you want the closest possible match to the reference.",
        "",
        "## Quick start",
        "",
        "### 1. Build the esmc.cpp runtime",
        "",
        "```bash",
        f"git clone --recursive {GITHUB_REPO}",
        "cd esmc.cpp",
        "cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8",
        "```",
        "",
        "### 2. Download a model",
        "",
        "```bash",
        "pip install -U huggingface_hub",
        f"huggingface-cli download {hf_repo} esmc-300m-Q8_0.gguf --local-dir ./models",
        "```",
        "",
        "### 3. Embed a protein sequence",
        "",
        "```bash",
        "# Mean-pooled sequence embedding -> one vector per sequence ([n_embd])",
        "./build/esmc-embed -m ./models/esmc-300m-Q8_0.gguf \\",
        f'    -s "{example_seq}" \\',
        "    --pool mean --output embedding.npy",
        "",
        "# Per-residue embeddings -> matrix ([n_tokens, n_embd])",
        "./build/esmc-embed -m ./models/esmc-300m-Q8_0.gguf \\",
        f'    -s "{example_seq}" \\',
        "    --pool none --output residues.npy",
        "",
        "# Force CPU (skip the Metal/GPU backend)",
        "./build/esmc-embed -m ./models/esmc-300m-Q8_0.gguf -s \"...\" --pool mean --no-metal",
        "```",
        "",
        "Outputs are NumPy `.npy` arrays. Mean pooling strips the `<cls>`/`<eos>` tokens.",
        "",
        "### 4. Load the embedding in Python",
        "",
        "```python",
        "import numpy as np",
        "",
        "emb = np.load(\"embedding.npy\")   # mean pool: shape (960,)",
        "res = np.load(\"residues.npy\")    # per-residue: shape (n_tokens, 960)",
        "print(emb.shape, res.shape)",
        "```",
        "",
        "## Benchmarks (300M)",
        "",
        "Measured on an Apple M4 Max (36 GB) against the official PyTorch ESM-C 300M. Full "
        f"methodology and per-sequence data are in the [esmc.cpp repository]({GITHUB_REPO}).",
        "",
        "### Numerical fidelity vs PyTorch (per-residue cosine, 100 Swiss-Prot sequences)",
        "",
        md_table(
            ["Precision", "Aggregate mean cosine", "Worst min cosine",
             "Max mean-pool L2", "Pass rate"],
            cos_rows,
        ),
        "",
        "F16 and Q8_0 clear per-sequence mean cosine > 0.999; Q4_K_M / Q4_K_S clear the "
        "aggregate > 0.995 (4-bit misses concentrate in very short sequences).",
        "",
        "### Throughput (seq/s, best esmc.cpp config)",
        "",
        md_table(
            ["Bucket", "Tokens", "Best esmc.cpp", "seq/s"],
            thr_rows,
        ),
        "",
        "### Peak memory (long sequences, 36 GB budget)",
        "",
        *mem_lines,
        "",
        "### Downstream variant-effect preservation (ProteinGym, 10 assays x 1000 variants)",
        "",
        md_table(
            ["Precision", "Assays", "Mean abs Spearman delta",
             "Max abs Spearman delta", "Metric rows pass"],
            down_rows,
        ),
        "",
        "Variants are scored by the cosine between mean-pooled mutant and wild-type "
        "embeddings; deltas are versus the PyTorch reference (preservation probe).",
        "",
        "## Model details",
        "",
        "- **Architecture:** encoder-only transformer; 30 layers, d_model 960, 15 heads "
        "(head dim 64), SwiGLU FFN (width 2560), pre-LayerNorm, RoPE-NeoX "
        "(theta 10000), query/key LayerNorm, no biases, context length 2048.",
        "- **Tokenizer:** 33-token amino-acid alphabet; `<cls>` prepended and `<eos>` "
        "appended (direct character lookup, no subword splitting).",
        "- **Provenance:** converted from the upstream safetensors checkpoint to GGUF "
        "(fused QKV and SwiGLU projections split); quantized variants use ggml block "
        "quantization. Weight values are otherwise unchanged from the upstream release.",
        "",
        "## Verify downloads",
        "",
        "```bash",
        "shasum -a 256 models/*.gguf   # compare against the sha256 column above",
        "```",
        "",
        "## Reproduce",
        "",
        f"The full replication guide (convert, quantize, validate, benchmark) is in the "
        f"[esmc.cpp README]({GITHUB_REPO}#reproduce-the-paper-results-300m-end-to-end). "
        f"The [lab manual]({GITHUB_REPO}/blob/main/lab_manual.md) documents every "
        f"experiment (EXP-001 through EXP-022) with commands, raw results, and run logs.",
        "",
        "## License",
        "",
        "Built with ESM.",
        "",
        "These GGUF files are Derivative Works of the ESM-C 300M Open Model and are "
        f"distributed under the [EvolutionaryScale Cambrian Open License Agreement]"
        f"({CAMBRIAN_OPEN_LICENSE}) (the permissive license that governs ESM-C 300M), "
        f"subject to the [Acceptable Use Policy]({ACCEPTABLE_USE_POLICY}). "
        "The ESMC 300M Model is licensed under the EvolutionaryScale Cambrian Open "
        "License Agreement.",
        "",
        "## Citation",
        "",
        "If you use these models, please cite the ESM Cambrian work by EvolutionaryScale "
        f"and link the [esmc.cpp runtime]({GITHUB_REPO}).",
    ]
    text = "\n".join(card) + "\n"
    (dest_dir / "README.md").write_text(text)
    return str((dest_dir / "README.md").relative_to(ROOT))


# ── benchmark artifact copying ───────────────────────────────────────────────


def copy_artifacts(paths: list[Path], dest_dir: Path) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in paths:
        if not src.is_file():
            continue
        # Also copy the .json sibling of any .csv when present.
        for candidate in {src, src.with_suffix(".json"), src.with_suffix(".csv")}:
            if candidate.is_file():
                shutil.copy2(candidate, dest_dir / candidate.name)
                copied.append(str((dest_dir / candidate.name).relative_to(ROOT)))
    return sorted(set(copied))


def checksum_tree(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(root))
            inventory[rel] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        elif path.is_symlink():
            rel = str(path.relative_to(root))
            inventory[rel] = {"symlink_to": str(path.resolve())}
    return inventory


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--correctness-csv", type=Path, default=DEFAULT_CORRECTNESS)
    parser.add_argument("--throughput-csv", type=Path, default=DEFAULT_THROUGHPUT)
    parser.add_argument("--memory-csv", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--downstream-csv", type=Path, default=DEFAULT_DOWNSTREAM)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hf-repo", default=HF_DEFAULT_REPO)
    parser.add_argument(
        "--gguf-mode", choices=["link", "copy", "manifest"], default="link",
        help="link (symlink, default), copy (duplicate bytes), or manifest (checksum only)",
    )
    args = parser.parse_args()

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else ROOT / p

    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    correctness_rows = read_csv(resolve(args.correctness_csv))
    throughput_rows = read_csv(resolve(args.throughput_csv))
    memory_rows = read_csv(resolve(args.memory_csv))
    downstream_rows = read_csv(resolve(args.downstream_csv))

    corr_sum = correctness_summary(correctness_rows)
    thr_sum = throughput_summary(throughput_rows)
    mem_sum = memory_summary(memory_rows)
    down_sum = downstream_summary(downstream_rows)

    # Tables (md + tex).
    tables_dir = out_dir / "tables"
    table_files = [
        correctness_table(corr_sum, tables_dir),
        throughput_table(thr_sum, tables_dir),
        memory_table(mem_sum, tables_dir),
        downstream_table(down_sum, tables_dir),
    ]
    combined = ["# Paper-ready result tables (milestone 16)", ""]
    for tf in table_files:
        combined.append((ROOT / tf["md"]).read_text().rstrip())
        combined.append("")
    (tables_dir / "all_tables.md").write_text("\n".join(combined) + "\n")

    # Plots.
    plots_dir = out_dir / "plots"
    plot_files = make_plots(corr_sum, thr_sum, mem_sum, down_sum, plots_dir)

    # Benchmark artifacts.
    bench_dir = out_dir / "benchmarks"
    artifact_sources = [
        resolve(args.correctness_csv),
        resolve(args.throughput_csv),
        resolve(args.memory_csv),
        resolve(args.downstream_csv),
        RESULTS / "manifest.json",
        RESULTS / "downstream_manifest.json",
    ]
    bench_files = copy_artifacts(artifact_sources, bench_dir)

    # GGUF models + docs + card.
    models_out = out_dir / "models"
    gguf_entries = gather_gguf(resolve(args.models_dir), models_out, args.gguf_mode)
    write_models_doc(gguf_entries, models_out, args.hf_repo)
    card_path = write_model_card(
        gguf_entries, out_dir / "model_card", args.hf_repo,
        corr_sum, thr_sum, mem_sum, down_sum,
    )

    # Bundle README.
    gguf_total_mib = sum(e["size_mib"] for e in gguf_entries)
    worst_mem = mem_sum[0] if mem_sum else None
    readme = [
        "# esmc.cpp reproduction bundle",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "Self-contained bundle for reproducing the esmc.cpp (ESM-C 300M) paper results.",
        "Verification target (plan.md milestone 16): includes GGUF files, benchmark "
        "CSV/JSON, plots, and paper-ready result tables.",
        "",
        "## Contents",
        "",
        f"- `models/` — {len(gguf_entries)} GGUF files "
        f"({args.gguf_mode}, {gguf_total_mib:.0f} MiB total) + `MODELS.md` checksum manifest",
        "- `benchmarks/` — correctness, throughput, memory, and downstream CSV/JSON artifacts",
        "- `plots/` — SVG figures for correctness, throughput, memory, and downstream",
        "- `tables/` — paper-ready Markdown + LaTeX result tables (`all_tables.md` combines them)",
        "- `model_card/README.md` — HuggingFace model card",
        "- `MANIFEST.json` — git hash, host info, and a checksummed file inventory",
        "",
        "## Headline numbers (300M)",
        "",
    ]
    for s in corr_sum:
        readme.append(
            f"- Correctness {PRECISION_LABEL.get(s['precision'], s['precision'])}: "
            f"aggregate mean cosine {fmt(s['agg_mean_cosine'], 5)}, "
            f"pass {s['passed']}/{s['total']}"
        )
    if worst_mem:
        readme.append(
            f"- Worst-case peak RSS (long bucket): {worst_mem['config']} = "
            f"{fmt(worst_mem['peak_rss_mib'], 0)} MiB (36 GB budget)"
        )
    readme += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "# 1. Build",
        "cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8",
        "# 2. Download GGUF models (see models/MODELS.md)",
        f"huggingface-cli download {args.hf_repo} --local-dir ./models",
        "# 3. Re-run benchmarks (see lab_manual.md section 11)",
        ".venv/bin/python benchmarks/correctness.py --config benchmarks/config_300m.json",
        "# 4. Regenerate this bundle",
        ".venv/bin/python benchmarks/make_reproduction_bundle.py",
        "```",
        "",
        "## Upload models to HuggingFace",
        "",
        "```bash",
        f".venv/bin/python tools/upload_to_hf.py --repo-id {args.hf_repo} \\",
        "    --models-dir ./models --model-card model_card/README.md --dry-run",
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n")

    # MANIFEST.json (checksum inventory excludes the manifest itself).
    manifest = {
        "schema_version": 1,
        "milestone": 16,
        "created_at": utc_now_iso(),
        "git": git_info(),
        "host": host_info(),
        "hf_repo": args.hf_repo,
        "gguf_mode": args.gguf_mode,
        "inputs": {
            "correctness_csv": str(resolve(args.correctness_csv).relative_to(ROOT)),
            "throughput_csv": str(resolve(args.throughput_csv).relative_to(ROOT)),
            "memory_csv": str(resolve(args.memory_csv).relative_to(ROOT)),
            "downstream_csv": str(resolve(args.downstream_csv).relative_to(ROOT)),
        },
        "models": gguf_entries,
        "tables": table_files,
        "plots": plot_files,
        "benchmarks": bench_files,
        "model_card": card_path,
        "summaries": {
            "correctness": corr_sum,
            "throughput": thr_sum,
            "memory": mem_sum,
            "downstream": down_sum,
        },
        "file_inventory": checksum_tree(out_dir),
    }
    write_json(out_dir / "MANIFEST.json", manifest)

    # Console report.
    print(f"Reproduction bundle written to {out_dir}")
    print(f"  models:     {len(gguf_entries)} GGUF ({args.gguf_mode})")
    print(f"  tables:     {len(table_files)} (md+tex) -> {tables_dir.relative_to(ROOT)}")
    print(f"  plots:      {len(plot_files)} -> {plots_dir.relative_to(ROOT)}")
    print(f"  benchmarks: {len(bench_files)} artifacts -> {bench_dir.relative_to(ROOT)}")
    print(f"  model card: {card_path}")
    print(f"  manifest:   {(out_dir / 'MANIFEST.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

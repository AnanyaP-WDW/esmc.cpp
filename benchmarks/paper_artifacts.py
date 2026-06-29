#!/usr/bin/env python3
"""Milestone 8G paper artifact generation.

Builds paper-ready summary tables and plots from:
- throughput benchmark CSV (M8E)
- memory benchmark CSV (M8F)
- downstream 10k benchmark CSV (M8D)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THROUGHPUT = ROOT / "results" / "throughput_Ananyas-MacBook-Pro_20260530_140618.csv"
DEFAULT_MEMORY = ROOT / "results" / "memory_Ananyas-MacBook-Pro_20260531_013718.csv"
DEFAULT_DOWNSTREAM = ROOT / "results" / "downstream_300m_10k.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "paper_artifacts_300m"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str] | None, key: str, default: float = math.nan) -> float:
    if row is None:
        return default
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: float, ndigits: int = 2) -> str:
    if v is None or math.isnan(v):
        return "n/a"
    return f"{v:.{ndigits}f}"


def throughput_artifacts(rows: list[dict[str, str]], out_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    buckets = ["short", "medium", "long"]

    def by_backend(backend: str, bucket: str) -> dict[str, str] | None:
        return next((r for r in rows if r["backend"] == backend and r["sequence_bucket"] == bucket), None)

    pt_cpu = {b: by_backend("pytorch_cpu", b) for b in buckets}
    pt_mps = {b: by_backend("pytorch_mps", b) for b in buckets}

    esmc_rows = [r for r in rows if r["implementation"] == "esmc.cpp"]
    best_by_bucket = {}
    for b in buckets:
        candidates = [r for r in esmc_rows if r["sequence_bucket"] == b]
        best_by_bucket[b] = max(candidates, key=lambda r: to_float(r, "seq_per_s"))

    summary_rows: list[dict[str, Any]] = []
    for b in buckets:
        best = best_by_bucket[b]
        s_best = to_float(best, "seq_per_s")
        pt_cpu_row = pt_cpu.get(b)
        pt_mps_row = pt_mps.get(b)
        s_cpu = to_float(pt_cpu_row, "seq_per_s") if pt_cpu_row else math.nan
        s_mps = to_float(pt_mps_row, "seq_per_s") if pt_mps_row else math.nan
        summary_rows.append(
            {
                "bucket": b,
                "sequence_tokens": int(to_float(best, "token_count")),
                "best_config": f"{best['backend']}/{best['precision']}",
                "best_seq_per_s": s_best,
                "vs_pt_cpu_ratio": s_best / s_cpu if s_cpu > 0 else math.nan,
                "vs_pt_mps_ratio": s_best / s_mps if s_mps > 0 else math.nan,
                "pt_cpu_seq_per_s": s_cpu,
                "pt_mps_seq_per_s": s_mps,
            }
        )

    summary_csv = out_dir / "throughput_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        [
            "bucket",
            "sequence_tokens",
            "best_config",
            "best_seq_per_s",
            "vs_pt_cpu_ratio",
            "vs_pt_mps_ratio",
            "pt_cpu_seq_per_s",
            "pt_mps_seq_per_s",
        ],
    )

    lines = [
        "### Throughput (M8E)",
        "",
        f"- Source: `{DEFAULT_THROUGHPUT.relative_to(ROOT)}`",
        "- Best esmc.cpp configuration by bucket:",
    ]
    for row in summary_rows:
        lines.append(
            f"  - `{row['bucket']}`: `{row['best_config']}` = {fmt(row['best_seq_per_s'])} seq/s "
            f"({fmt(row['vs_pt_cpu_ratio'])}x PT CPU, {fmt(row['vs_pt_mps_ratio'])}x PT MPS)"
        )

    return summary_rows, lines, [str(summary_csv.relative_to(ROOT))]


def memory_artifacts(rows: list[dict[str, str]], out_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    long_rows = [r for r in rows if r["sequence_bucket"] == "long"]
    for r in long_rows:
        r["_peak"] = to_float(r, "peak_rss_mib")

    long_rows.sort(key=lambda r: r["_peak"], reverse=True)

    summary_rows: list[dict[str, Any]] = []
    for r in long_rows:
        summary_rows.append(
            {
                "config": f"{r['implementation']}/{r['backend']}/{r['precision']}",
                "peak_rss_mib": to_float(r, "peak_rss_mib"),
                "model_file_mib": to_float(r, "model_file_size_bytes") / (1024 * 1024),
                "budget_pass": r["budget_pass"],
            }
        )

    summary_csv = out_dir / "memory_long_summary.csv"
    write_csv(summary_csv, summary_rows, ["config", "peak_rss_mib", "model_file_mib", "budget_pass"])

    worst = summary_rows[0]
    lines = [
        "### Memory (M8F)",
        "",
        f"- Source: `{DEFAULT_MEMORY.relative_to(ROOT)}`",
        f"- Rows: {len(rows)}; budget pass rows: {sum(1 for r in rows if r.get('budget_pass') == 'True')}/{len(rows)}",
        f"- Worst-case peak RSS: `{worst['config']}` = {fmt(worst['peak_rss_mib'])} MiB",
    ]
    return summary_rows, lines, [str(summary_csv.relative_to(ROOT))]


def downstream_artifacts(rows: list[dict[str, str]], out_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows = [r for r in rows if not r.get("error")]
    precisions = ["f16", "q8_0", "q4_k_m", "q4_k_s"]
    metrics = sorted({r["metric_name"] for r in rows})

    summary_rows: list[dict[str, Any]] = []
    for p in precisions:
        p_rows = [r for r in rows if r["precision"] == p]
        if not p_rows:
            continue
        for m in metrics:
            pm = [r for r in p_rows if r["metric_name"] == m]
            total = len(pm)
            passed = sum(1 for r in pm if r["passed"] == "True")
            summary_rows.append(
                {
                    "precision": p,
                    "metric_name": m,
                    "passed": passed,
                    "total": total,
                    "pass_rate": passed / total if total else math.nan,
                }
            )

    summary_csv = out_dir / "downstream_10k_pass_rates.csv"
    write_csv(summary_csv, summary_rows, ["precision", "metric_name", "passed", "total", "pass_rate"])

    assay_count = len({r["assay"] for r in rows})
    lines = [
        "### Downstream 10k (M8D)",
        "",
        f"- Source: `{DEFAULT_DOWNSTREAM.relative_to(ROOT)}`",
        f"- Assays: {assay_count}, metrics per assay: {len(metrics)}, precisions: {', '.join(precisions)}",
        "- Pass counts by precision (all metrics aggregated):",
    ]
    for p in precisions:
        p_rows = [r for r in summary_rows if r["precision"] == p]
        passed = sum(int(r["passed"]) for r in p_rows)
        total = sum(int(r["total"]) for r in p_rows)
        lines.append(f"  - `{p}`: {passed}/{total} ({fmt(passed / total if total else math.nan, 3)})")

    return summary_rows, lines, [str(summary_csv.relative_to(ROOT))]


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def write_svg_grouped_bar_chart(
    *,
    path: Path,
    title: str,
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    y_label: str,
) -> None:
    width, height = 1400, 650
    margin_left, margin_right, margin_top, margin_bottom = 90, 20, 70, 170
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    ymax = max((max(vals) for _, vals in series if vals), default=1.0)
    ymax = ymax * 1.1 if ymax > 0 else 1.0
    n_groups = len(x_labels)
    n_series = max(1, len(series))
    group_w = plot_w / max(1, n_groups)
    bar_w = max(4.0, group_w / (n_series + 2))
    colors = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
    ]

    def y_to_px(v: float) -> float:
        return margin_top + (plot_h - (v / ymax) * plot_h)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-size="22" font-family="Arial">{_svg_escape(title)}</text>',
        # axes
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333"/>',
        f'<text x="20" y="{margin_top + plot_h/2:.1f}" transform="rotate(-90 20,{margin_top + plot_h/2:.1f})" font-size="14" font-family="Arial">{_svg_escape(y_label)}</text>',
    ]

    # y ticks
    for i in range(6):
        v = ymax * i / 5
        y = y_to_px(v)
        parts.append(
            f'<line x1="{margin_left-5}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#666"/>'
        )
        parts.append(
            f'<text x="{margin_left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{v:.2f}</text>'
        )
        if i > 0:
            parts.append(
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" stroke="#eee"/>'
            )

    # bars + x labels
    for gi, xl in enumerate(x_labels):
        gx = margin_left + gi * group_w
        for si, (_, vals) in enumerate(series):
            v = vals[gi] if gi < len(vals) else 0.0
            x = gx + bar_w + si * bar_w
            y = y_to_px(v)
            h = margin_top + plot_h - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.85:.1f}" height="{h:.1f}" fill="{colors[si % len(colors)]}"/>'
            )
        lx = gx + group_w / 2
        parts.append(
            f'<text x="{lx:.1f}" y="{margin_top + plot_h + 20}" text-anchor="middle" font-size="12" font-family="Arial">{_svg_escape(xl)}</text>'
        )

    # legend
    leg_x, leg_y = margin_left, height - 95
    for i, (name, _) in enumerate(series):
        y = leg_y + (i // 4) * 20
        x = leg_x + (i % 4) * 320
        c = colors[i % len(colors)]
        parts.append(f'<rect x="{x}" y="{y-10}" width="12" height="12" fill="{c}"/>')
        parts.append(
            f'<text x="{x+18}" y="{y}" font-size="12" font-family="Arial">{_svg_escape(name)}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def generate_svg_fallback(
    throughput_rows: list[dict[str, str]],
    memory_rows: list[dict[str, str]],
    downstream_rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    # throughput grouped bars
    buckets = ["short", "medium", "long"]
    x_labels = buckets
    t_series = []
    for name, selector in [
        ("PT MPS f32", lambda r: r["backend"] == "pytorch_mps"),
        ("PT CPU f32", lambda r: r["backend"] == "pytorch_cpu"),
        ("Metal Q4_K_S", lambda r: r["backend"] == "metal" and r["precision"] == "q4_k_s"),
        ("Metal f16", lambda r: r["backend"] == "metal" and r["precision"] == "f16"),
        ("CPU f16", lambda r: r["backend"] == "cpu" and r["precision"] == "f16"),
    ]:
        vals = [to_float(next((r for r in throughput_rows if r["sequence_bucket"] == b and selector(r)), None), "seq_per_s") for b in buckets]
        t_series.append((name, vals))
    p = out_dir / "throughput_seqps.svg"
    write_svg_grouped_bar_chart(
        path=p,
        title="Throughput by bucket (M8E)",
        x_labels=x_labels,
        series=t_series,
        y_label="seq/s",
    )
    paths.append(str(p.relative_to(ROOT)))

    # memory long peak rss
    long_rows = [r for r in memory_rows if r["sequence_bucket"] == "long"]
    long_rows.sort(key=lambda r: to_float(r, "peak_rss_mib"), reverse=True)
    p = out_dir / "memory_long_rss.svg"
    write_svg_grouped_bar_chart(
        path=p,
        title="Memory long-bucket peak RSS (M8F)",
        x_labels=[f"{r['backend']}/{r['precision']}" for r in long_rows],
        series=[("peak_rss_mib", [to_float(r, "peak_rss_mib") for r in long_rows])],
        y_label="MiB",
    )
    paths.append(str(p.relative_to(ROOT)))

    # downstream pass rate
    precisions = ["f16", "q8_0", "q4_k_m", "q4_k_s"]
    metrics = sorted({r["metric_name"] for r in downstream_rows})
    d_series: list[tuple[str, list[float]]] = []
    for m in metrics:
        vals = []
        for p0 in precisions:
            row = next((r for r in downstream_rows if r["precision"] == p0 and r["metric_name"] == m), None)
            vals.append(float(row["pass_rate"]) if row else math.nan)
        d_series.append((m, vals))
    p = out_dir / "downstream_10k_pass_rate.svg"
    write_svg_grouped_bar_chart(
        path=p,
        title="Downstream 10k pass rate by precision (M8D)",
        x_labels=precisions,
        series=d_series,
        y_label="pass rate",
    )
    paths.append(str(p.relative_to(ROOT)))
    return paths


def generate_plots(
    throughput_rows: list[dict[str, str]],
    memory_rows: list[dict[str, str]],
    downstream_rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return generate_svg_fallback(throughput_rows, memory_rows, downstream_rows, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []

    # Throughput plot
    buckets = ["short", "medium", "long"]
    labels = ["PT MPS f32", "PT CPU f32", "Metal Q4_K_S", "Metal f16", "CPU f16"]
    selectors = [
        lambda r: r["backend"] == "pytorch_mps",
        lambda r: r["backend"] == "pytorch_cpu",
        lambda r: r["backend"] == "metal" and r["precision"] == "q4_k_s",
        lambda r: r["backend"] == "metal" and r["precision"] == "f16",
        lambda r: r["backend"] == "cpu" and r["precision"] == "f16",
    ]

    x = range(len(buckets))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, select) in enumerate(zip(labels, selectors)):
        vals = []
        for b in buckets:
            row = next((r for r in throughput_rows if r["sequence_bucket"] == b and select(r)), None)
            vals.append(to_float(row, "seq_per_s"))
        positions = [xi + (i - 2) * width for xi in x]
        ax.bar(positions, vals, width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(buckets)
    ax.set_ylabel("seq/s")
    ax.set_title("Throughput by bucket (M8E)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    p_png = out_dir / "throughput_seqps.png"
    p_pdf = out_dir / "throughput_seqps.pdf"
    fig.tight_layout()
    fig.savefig(p_png, dpi=180)
    fig.savefig(p_pdf)
    plt.close(fig)
    plot_paths.append(str(p_png.relative_to(ROOT)))
    plot_paths.append(str(p_pdf.relative_to(ROOT)))

    # Memory plot (long bucket)
    long_rows = [r for r in memory_rows if r["sequence_bucket"] == "long"]
    long_rows.sort(key=lambda r: to_float(r, "peak_rss_mib"), reverse=True)
    labels = [f"{r['implementation']}/{r['backend']}/{r['precision']}" for r in long_rows]
    vals = [to_float(r, "peak_rss_mib") for r in long_rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Peak RSS (MiB)")
    ax.set_title("Memory long-bucket peak RSS (M8F)")
    ax.grid(axis="y", alpha=0.3)
    p_png = out_dir / "memory_long_rss.png"
    p_pdf = out_dir / "memory_long_rss.pdf"
    fig.tight_layout()
    fig.savefig(p_png, dpi=180)
    fig.savefig(p_pdf)
    plt.close(fig)
    plot_paths.append(str(p_png.relative_to(ROOT)))
    plot_paths.append(str(p_pdf.relative_to(ROOT)))

    # Downstream pass-rate plot
    precisions = ["f16", "q8_0", "q4_k_m", "q4_k_s"]
    metrics = sorted({r["metric_name"] for r in downstream_rows})
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.14
    for i, m in enumerate(metrics):
        vals = []
        for p0 in precisions:
            row = next(r for r in downstream_rows if r["precision"] == p0 and r["metric_name"] == m)
            vals.append(float(row["pass_rate"]))
        positions = [j + (i - (len(metrics) - 1) / 2) * width for j in range(len(precisions))]
        ax.bar(positions, vals, width=width, label=m)
    ax.set_xticks(range(len(precisions)))
    ax.set_xticklabels(precisions)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Pass rate")
    ax.set_title("Downstream 10k pass rate by precision (M8D)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    p = out_dir / "downstream_10k_pass_rate.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    plot_paths.append(str(p.relative_to(ROOT)))

    return plot_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throughput-csv", type=Path, default=DEFAULT_THROUGHPUT)
    parser.add_argument("--memory-csv", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--downstream-csv", type=Path, default=DEFAULT_DOWNSTREAM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    throughput_csv = args.throughput_csv if args.throughput_csv.is_absolute() else ROOT / args.throughput_csv
    memory_csv = args.memory_csv if args.memory_csv.is_absolute() else ROOT / args.memory_csv
    downstream_csv = args.downstream_csv if args.downstream_csv.is_absolute() else ROOT / args.downstream_csv
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    throughput_rows = read_csv(throughput_csv)
    memory_rows = read_csv(memory_csv)
    downstream_rows = read_csv(downstream_csv)

    _, throughput_lines, throughput_files = throughput_artifacts(throughput_rows, out_dir)
    downstream_summary_rows, downstream_lines, downstream_files = downstream_artifacts(downstream_rows, out_dir)
    _, memory_lines, memory_files = memory_artifacts(memory_rows, out_dir)
    plot_files = generate_plots(throughput_rows, memory_rows, downstream_summary_rows, out_dir)

    md = out_dir / "paper_artifact_summary.md"
    lines = [
        "# Paper Artifact Summary (M8G)",
        "",
        *throughput_lines,
        "",
        *memory_lines,
        "",
        *downstream_lines,
        "",
        "## Generated files",
        "",
    ]
    for p in [*throughput_files, *memory_files, *downstream_files, *plot_files]:
        lines.append(f"- `{p}`")

    md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md}")
    for p in [*throughput_files, *memory_files, *downstream_files, *plot_files]:
        print(f"Wrote {ROOT / p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

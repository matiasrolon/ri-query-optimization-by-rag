# -*- coding: utf-8 -*-
"""
Visualization module for multi-run summary benchmark analysis.

Generates publication-quality figures (LNCS / LaTeX serif style, PDF + PNG):
1. 01_mrr_vs_fb_terms.pdf (.png)          - MRR@10 vs. fb_terms (X-Y plot).
2. 02_mrr_vs_fb_docs.pdf (.png)           - MRR@10 vs. fb_docs (X-Y plot).
3. 03_effectiveness_vs_cost.pdf (.png)    - MRR@10 vs. query time (log-scale X scatter).
4. 04_inference_time_vs_fb_docs.pdf (.png)- LLM inference time vs. fb_docs (linear trend).
"""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from analysis.analyzer_summary import SummaryBenchmarkAnalyzer, SummaryAnalysisMetrics
from analysis.plots_single_run import _apply_plot_style, _save_figure


DOCS_COLORS = {
    3: "#2563EB",   # Blue
    5: "#15803D",   # Deep green
    10: "#7C3AED",  # Purple
}
DOCS_MARKERS = {
    3: "o",
    5: "s",
    10: "^",
}

TERMS_COLORS = {
    1: "#15803D",   # Deep green
    2: "#2563EB",   # Blue
    3: "#D97706",   # Amber
    5: "#B91C1C",   # Deep red
}
TERMS_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
    5: "D",
}


# ── Chart 1: MRR@10 vs. fb_terms ──────────────────────────────────────────────
def plot_mrr_vs_fb_terms(
    analyzer: SummaryBenchmarkAnalyzer,
    metrics: SummaryAnalysisMetrics,
    output_dir: str,
) -> str:
    """
    Generate Chart 1: MRR@10 vs. FEEDBACK_TERMS for RAG (LLM) and PRF across fb_docs,
    plus BM25 baseline reference.
    """
    _apply_plot_style()
    df = analyzer.df
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    # BM25 horizontal baseline
    ax.axhline(
        metrics.mrr_bm25_baseline,
        color="#334155",
        linestyle=":",
        linewidth=1.1,
        label=f"BM25 Base ({metrics.mrr_bm25_baseline:.4f})",
        zorder=2,
    )

    for fb_docs in metrics.fb_docs_values:
        sub = df[df["fb_docs"] == fb_docs].sort_values("fb_terms")
        color = DOCS_COLORS.get(fb_docs, "#1E293B")
        marker = DOCS_MARKERS.get(fb_docs, "o")

        # RAG (LLM) — solid line
        ax.plot(
            sub["fb_terms"],
            sub["mrr_llm_selected"],
            color=color,
            linestyle="-",
            linewidth=1.4,
            marker=marker,
            markersize=4.5,
            label=f"RAG ($d={fb_docs}$)",
            zorder=4,
        )

        # PRF — dashed line
        ax.plot(
            sub["fb_terms"],
            sub["mrr_prf_selected"],
            color=color,
            linestyle="--",
            linewidth=1.0,
            marker=marker,
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
            alpha=0.85,
            label=f"PRF ($d={fb_docs}$)",
            zorder=3,
        )

    ax.set_xticks(metrics.fb_terms_values)
    ax.set_xlabel("Términos de expansión ($fb\\_terms$)")
    ax.set_ylabel("MRR@10")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=1.8,
        columnspacing=1.0,
    )

    return _save_figure(fig, output_dir, "01_mrr_vs_fb_terms")


# ── Chart 2: MRR@10 vs. fb_docs ───────────────────────────────────────────────
def plot_mrr_vs_fb_docs(
    analyzer: SummaryBenchmarkAnalyzer,
    metrics: SummaryAnalysisMetrics,
    output_dir: str,
) -> str:
    """
    Generate Chart 2: MRR@10 vs. FEEDBACK_DOCS for RAG (LLM) and PRF across fb_terms,
    plus BM25 baseline reference.
    """
    _apply_plot_style()
    df = analyzer.df
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    # BM25 horizontal baseline
    ax.axhline(
        metrics.mrr_bm25_baseline,
        color="#334155",
        linestyle=":",
        linewidth=1.1,
        label=f"BM25 Base ({metrics.mrr_bm25_baseline:.4f})",
        zorder=2,
    )

    for fb_terms in metrics.fb_terms_values:
        sub = df[df["fb_terms"] == fb_terms].sort_values("fb_docs")
        color = TERMS_COLORS.get(fb_terms, "#1E293B")
        marker = TERMS_MARKERS.get(fb_terms, "o")

        # RAG (LLM) — solid line
        ax.plot(
            sub["fb_docs"],
            sub["mrr_llm_selected"],
            color=color,
            linestyle="-",
            linewidth=1.4,
            marker=marker,
            markersize=4.5,
            label=f"RAG ($t={fb_terms}$)",
            zorder=4,
        )

        # PRF — dashed line
        ax.plot(
            sub["fb_docs"],
            sub["mrr_prf_selected"],
            color=color,
            linestyle="--",
            linewidth=1.0,
            marker=marker,
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
            alpha=0.85,
            label=f"PRF ($t={fb_terms}$)",
            zorder=3,
        )

    ax.set_xticks(metrics.fb_docs_values)
    ax.set_xlabel("Documentos de contexto ($fb\\_docs$)")
    ax.set_ylabel("MRR@10")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=6.8,
        handlelength=1.8,
        columnspacing=0.9,
    )

    return _save_figure(fig, output_dir, "02_mrr_vs_fb_docs")


# ── Chart 3: Effectiveness vs. Cost (Log-scale X scatter) ─────────────────────
def plot_effectiveness_vs_cost(
    analyzer: SummaryBenchmarkAnalyzer,
    metrics: SummaryAnalysisMetrics,
    output_dir: str,
) -> str:
    """
    Generate Chart 3: Scatter plot of MRR@10 (Y) vs. average time per query in seconds
    on a logarithmic X axis, one point per configuration.
    """
    _apply_plot_style()
    df = analyzer.df
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    # BM25 baseline line + point
    ax.axhline(
        metrics.mrr_bm25_baseline,
        color="#64748B",
        linestyle=":",
        linewidth=0.9,
        zorder=1,
    )
    ax.scatter(
        [metrics.time_bm25_avg_baseline],
        [metrics.mrr_bm25_baseline],
        color="#0F172A",
        marker="*",
        s=55,
        label="BM25 Base",
        zorder=5,
    )

    # PRF points (one per configuration)
    for fb_docs in metrics.fb_docs_values:
        sub = df[df["fb_docs"] == fb_docs]
        color = DOCS_COLORS.get(fb_docs, "#2563EB")
        marker = DOCS_MARKERS.get(fb_docs, "o")
        ax.scatter(
            sub["time_prf_avg"],
            sub["mrr_prf_selected"],
            facecolors="white",
            edgecolors=color,
            marker=marker,
            s=30,
            linewidths=1.2,
            label=f"PRF ($d={fb_docs}$)",
            zorder=4,
        )

    # Custom non-overlapping offsets per (d, t)
    label_offsets = {
        (3, 1): (-5, 3, "right"),
        (3, 2): (-5, -5, "right"),
        (3, 3): (-5, -2, "right"),
        (3, 5): (-5, -2, "right"),
        (5, 1): (0, 5, "center"),
        (5, 2): (5, 2, "left"),
        (5, 3): (5, -4, "left"),
        (5, 5): (5, -2, "left"),
        (10, 1): (5, 1, "left"),
        (10, 2): (5, -2, "left"),
        (10, 3): (5, -2, "left"),
        (10, 5): (5, -2, "left"),
    }

    # RAG (LLM) points (one per configuration) + compact (d,t) annotations
    for fb_docs in metrics.fb_docs_values:
        sub = df[df["fb_docs"] == fb_docs]
        color = DOCS_COLORS.get(fb_docs, "#15803D")
        marker = DOCS_MARKERS.get(fb_docs, "o")
        ax.scatter(
            sub["time_llm_avg"],
            sub["mrr_llm_selected"],
            color=color,
            marker=marker,
            s=32,
            label=f"RAG ($d={fb_docs}$)",
            zorder=5,
        )
        for _, row in sub.iterrows():
            d = int(row["fb_docs"])
            t = int(row["fb_terms"])
            x_val = float(row["time_llm_avg"])
            y_val = float(row["mrr_llm_selected"])
            dx, dy, ha_align = label_offsets.get((d, t), (4, 2, "left"))
            ax.annotate(
                f"({d},{t})",
                xy=(x_val, y_val),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha_align,
                va="center",
                fontsize=6.2,
                color="#1E293B",
            )

    ax.set_xscale("log")
    ax.set_xlim(0.035, 30.0)
    all_mrr = pd.concat([df["mrr_prf_selected"], df["mrr_llm_selected"]]).dropna()
    ax.set_ylim(float(all_mrr.min()) - 0.0018, max(float(all_mrr.max()), metrics.mrr_bm25_baseline) + 0.0022)
    ax.set_xlabel("Tiempo promedio por consulta [s] (escala log)")
    ax.set_ylabel("MRR@10")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, which="both", linestyle="--", alpha=0.3, linewidth=0.5)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=6.8,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    return _save_figure(fig, output_dir, "03_effectiveness_vs_cost")


# ── Chart 4: Inference Time vs. fb_docs (Near-linear relationship) ────────────
def plot_inference_time_vs_fb_docs(
    analyzer: SummaryBenchmarkAnalyzer,
    metrics: SummaryAnalysisMetrics,
    output_dir: str,
) -> str:
    """
    Generate Chart 4: LLM inference / resolution time per query vs. FEEDBACK_DOCS,
    highlighting individual fb_terms series and the near-linear regression fit.
    """
    _apply_plot_style()
    df = analyzer.df
    fig, ax = plt.subplots(figsize=(3.5, 2.3))

    for fb_terms in metrics.fb_terms_values:
        sub = df[df["fb_terms"] == fb_terms].sort_values("fb_docs")
        color = TERMS_COLORS.get(fb_terms, "#2563EB")
        marker = TERMS_MARKERS.get(fb_terms, "o")
        ax.plot(
            sub["fb_docs"],
            sub["time_llm_avg"],
            color=color,
            linestyle="-",
            linewidth=1.1,
            marker=marker,
            markersize=4.2,
            alpha=0.85,
            label=f"$fb\\_terms={fb_terms}$",
            zorder=3,
        )

    # Linear regression trend line across fb_docs
    x_line = np.linspace(min(metrics.fb_docs_values) - 0.3, max(metrics.fb_docs_values) + 0.3, 50)
    y_line = metrics.linear_fit_slope_sec_per_doc * x_line + metrics.linear_fit_intercept_sec
    ax.plot(
        x_line,
        y_line,
        color="#0F172A",
        linestyle="--",
        linewidth=1.4,
        label=f"Ajuste lineal ($R^2={metrics.linear_fit_r2:.3f}$)",
        zorder=4,
    )

    # Annotate mean time per fb_docs above the highest point at each fb_docs
    grouped_means = df.groupby("fb_docs")["time_llm_avg"].mean()
    grouped_max = df.groupby("fb_docs")["time_llm_avg"].max()
    for fb_docs, mean_t in grouped_means.items():
        top_y = float(grouped_max.loc[fb_docs])
        ax.annotate(
            f"Media: {mean_t:.2f}s",
            xy=(fb_docs, top_y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        )

    min_t = float(df["time_llm_avg"].min())
    max_t = float(df["time_llm_avg"].max())
    ax.set_ylim(min_t - 0.6, max_t + 1.3)
    ax.set_xticks(metrics.fb_docs_values)
    ax.set_xlabel("Documentos de contexto ($fb\\_docs$)")
    ax.set_ylabel("Tiempo por consulta RAG [s]")
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=1.6,
        columnspacing=0.9,
    )

    return _save_figure(fig, output_dir, "04_inference_time_vs_fb_docs")


# ── Full Summary Suite Generation ─────────────────────────────────────────────
def generate_summary_plots(
    analyzer: SummaryBenchmarkAnalyzer,
    output_dir: str | None = None,
) -> tuple[SummaryAnalysisMetrics, dict[str, str]]:
    """
    Run general summary analysis and generate the 4 figures (PDF + PNG) + JSON/TXT reports.
    """
    metrics = analyzer.analyze()

    if output_dir is None:
        scope_suffix = "todas" if analyzer.query_scope == "all" else "solo_expandidas"
        output_dir = os.path.join("output", "figures", f"summary_{scope_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    fig_paths: dict[str, str] = {}
    fig_paths["mrr_vs_fb_terms"] = plot_mrr_vs_fb_terms(analyzer, metrics, output_dir)
    fig_paths["mrr_vs_fb_docs"] = plot_mrr_vs_fb_docs(analyzer, metrics, output_dir)
    fig_paths["effectiveness_vs_cost"] = plot_effectiveness_vs_cost(analyzer, metrics, output_dir)
    fig_paths["inference_time_vs_fb_docs"] = plot_inference_time_vs_fb_docs(analyzer, metrics, output_dir)

    # Save JSON summary
    json_path = os.path.join(output_dir, "summary_general_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
    fig_paths["summary_json"] = json_path

    # Save text report
    txt_path = os.path.join(output_dir, "summary_general_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"REPORTE DE ANÁLISIS GENERAL ({metrics.query_scope_label.upper()})\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Archivo origen             : {metrics.source_file}\n")
        f.write(f"Alcance de consultas       : {metrics.query_scope_label} ({metrics.query_scope})\n")
        f.write(f"Configuraciones analizadas : {metrics.total_configurations}\n")
        f.write(f"Valores de fb_docs         : {metrics.fb_docs_values}\n")
        f.write(f"Valores de fb_terms        : {metrics.fb_terms_values}\n")
        f.write(f"Línea base BM25 MRR@10     : {metrics.mrr_bm25_baseline:.4f} ({metrics.time_bm25_avg_baseline:.4f} s/query)\n\n")

        f.write("── MEJORES CONFIGURACIONES ──────────────────────────────────────────\n")
        f.write(
            f"  • Mejor RAG (LLM) : fb_docs={metrics.best_rag_config['fb_docs']}, "
            f"fb_terms={metrics.best_rag_config['fb_terms']} → "
            f"MRR@10={metrics.best_rag_config['mrr']:.4f} "
            f"({metrics.best_rag_config['time_avg']:.4f} s/query)\n"
        )
        f.write(
            f"  • Mejor PRF       : fb_docs={metrics.best_prf_config['fb_docs']}, "
            f"fb_terms={metrics.best_prf_config['fb_terms']} → "
            f"MRR@10={metrics.best_prf_config['mrr']:.4f} "
            f"({metrics.best_prf_config['time_avg']:.4f} s/query)\n\n"
        )

        f.write("── ESCALAMIENTO DE TIEMPO DE INFERENCIA VS. FB_DOCS ─────────────────\n")
        f.write(
            f"  • Regresión lineal: T(fb_docs) = {metrics.linear_fit_slope_sec_per_doc:.4f} * fb_docs "
            f"+ {metrics.linear_fit_intercept_sec:.4f} s  (R² = {metrics.linear_fit_r2:.4f})\n\n"
        )

        f.write("── TABLA DE CONFIGURACIONES ─────────────────────────────────────────\n")
        f.write("  fb_docs | fb_terms | MRR PRF | MRR RAG | Time PRF (s/q) | Time RAG (s/q)\n")
        f.write("  --------+----------+---------+---------+----------------+---------------\n")
        for r in metrics.records:
            f.write(
                f"  {r['fb_docs']:>7d} | {r['fb_terms']:>8d} | "
                f"{r['mrr_prf_selected']:>7.4f} | {r['mrr_llm_selected']:>7.4f} | "
                f"{r['time_prf_avg']:>14.4f} | {r['time_llm_avg']:>14.4f}\n"
            )
        if metrics.fallback_configs:
            f.write(
                f"\nNota: En modo 'Todas las queries', las configuraciones "
                f"{', '.join(metrics.fallback_configs)} tenían vacías las celdas 'TODAS' "
                f"en el CSV y tomaron como fallback los valores de 'Solo expandidas'.\n"
            )
        f.write("=" * 72 + "\n")
    fig_paths["summary_txt"] = txt_path

    return metrics, fig_paths

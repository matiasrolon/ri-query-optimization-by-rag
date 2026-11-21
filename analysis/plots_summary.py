# -*- coding: utf-8 -*-
"""
Visualization module for multi-run summary benchmark analysis.

Supports two discrimination modes via ``analyzer.group_by``:
- ``"config"`` (default): discriminates by hyperparameter configuration (k=fb_docs, m=fb_terms).
- ``"model"``: discriminates by the ``LLM Model`` column.

Generates high-resolution PNG figures (300 DPI):
1. 01_mrr_vs_fb_terms.png          - MRR@10 vs. fb_terms (or by LLM Model if fb_terms is fixed).
2. 02_mrr_vs_fb_docs.png           - MRR@10 vs. fb_docs (or LLM vs PRF gain by Model if fb_docs is fixed).
3. 03_effectiveness_vs_cost.png    - MRR@10 vs. query time (log-scale X).
4. 04_inference_time_vs_fb_docs.png- LLM inference time vs. fb_docs (or by LLM Model).
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

MODEL_PALETTE = [
    "#2563EB",  # Blue
    "#15803D",  # Deep green
    "#7C3AED",  # Purple
    "#D97706",  # Amber
    "#B91C1C",  # Deep red
    "#0284C7",  # Teal
]
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _model_style(idx: int) -> tuple[str, str]:
    return (
        MODEL_PALETTE[idx % len(MODEL_PALETTE)],
        MODEL_MARKERS[idx % len(MODEL_MARKERS)],
    )


# ── Chart 1: MRR@10 vs. fb_terms (or by Model) ───────────────────────────────
def plot_mrr_vs_fb_terms(
    analyzer: SummaryBenchmarkAnalyzer,
    metrics: SummaryAnalysisMetrics,
    output_dir: str,
) -> str:
    """
    Generate Chart 1:
    - In 'config' mode: MRR@10 vs. FEEDBACK_TERMS for LLM and PRF across fb_docs.
    - In 'model' mode: MRR@10 vs. FEEDBACK_TERMS per LLM Model (or MRR@10 vs. LLM Model
      when fb_terms is fixed).
    """
    _apply_plot_style()
    df = analyzer.df
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    ax.axhline(
        metrics.mrr_bm25_baseline,
        color="#334155",
        linestyle=":",
        linewidth=1.1,
        label=f"BM25 Base ({metrics.mrr_bm25_baseline:.4f})",
        zorder=2,
    )

    if analyzer.group_by == "model":
        if len(metrics.fb_terms_values) > 1:
            for idx, model_name in enumerate(metrics.llm_models):
                sub = df[df["llm_model"] == model_name].sort_values("fb_terms")
                color, marker = _model_style(idx)
                ax.plot(
                    sub["fb_terms"],
                    sub["mrr_llm_selected"],
                    color=color,
                    linestyle="-",
                    linewidth=1.4,
                    marker=marker,
                    markersize=4.5,
                    label=f"LLM ({model_name})",
                    zorder=4,
                )
            prf_by_t = df.groupby("fb_terms", as_index=False)["mrr_prf_selected"].mean().sort_values("fb_terms")
            ax.plot(
                prf_by_t["fb_terms"],
                prf_by_t["mrr_prf_selected"],
                color="#64748B",
                linestyle="--",
                linewidth=1.1,
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label="PRF",
                zorder=3,
            )
            ax.set_xticks(metrics.fb_terms_values)
            ax.set_xlabel("Términos de expansión ($fb\\_terms$)")
        else:
            # Single fb_terms value: plot models along X axis
            x = np.arange(len(metrics.llm_models))
            prf_mean = float(df["mrr_prf_selected"].mean())
            ax.axhline(
                prf_mean,
                color="#64748B",
                linestyle="--",
                linewidth=1.1,
                label=f"PRF ({prf_mean:.4f})",
                zorder=2,
            )
            y_vals = []
            for idx, model_name in enumerate(metrics.llm_models):
                sub = df[df["llm_model"] == model_name]
                val = float(sub["mrr_llm_selected"].mean())
                y_vals.append(val)
                color, marker = _model_style(idx)
                ax.plot(
                    [x[idx]],
                    [val],
                    color=color,
                    linestyle="none",
                    marker=marker,
                    markersize=5.5,
                    label=f"LLM ({model_name})",
                    zorder=5,
                )
            ax.plot(x, y_vals, color="#94A3B8", linestyle="-", linewidth=1.0, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics.llm_models, fontsize=7.8)
            ax.set_xlim(-0.4, len(metrics.llm_models) - 0.6)
            k_val = metrics.fb_docs_values[0] if metrics.fb_docs_values else 5
            m_val = metrics.fb_terms_values[0] if metrics.fb_terms_values else 1
            ax.set_xlabel(f"Modelo LLM ($k={k_val}, m={m_val}$)")
    else:
        for fb_docs in metrics.fb_docs_values:
            sub = df[df["fb_docs"] == fb_docs].sort_values("fb_terms")
            color = DOCS_COLORS.get(fb_docs, "#1E293B")
            marker = DOCS_MARKERS.get(fb_docs, "o")

            ax.plot(
                sub["fb_terms"],
                sub["mrr_llm_selected"],
                color=color,
                linestyle="-",
                linewidth=1.4,
                marker=marker,
                markersize=4.5,
                label=f"LLM ($k={fb_docs}$)",
                zorder=4,
            )
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
                label=f"PRF ($k={fb_docs}$)",
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


# ── Chart 2: MRR@10 vs. fb_docs (or Relative Gain by Model) ──────────────────
def plot_mrr_vs_fb_docs(
    analyzer: SummaryBenchmarkAnalyzer,
    metrics: SummaryAnalysisMetrics,
    output_dir: str,
) -> str:
    """
    Generate Chart 2:
    - In 'config' mode: MRR@10 vs. FEEDBACK_DOCS for LLM and PRF across fb_terms.
    - In 'model' mode: MRR@10 vs. FEEDBACK_DOCS per LLM Model (or relative improvement %
      over PRF per LLM Model when fb_docs is fixed).
    """
    _apply_plot_style()
    df = analyzer.df
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    if analyzer.group_by == "model":
        if len(metrics.fb_docs_values) > 1:
            ax.axhline(
                metrics.mrr_bm25_baseline,
                color="#334155",
                linestyle=":",
                linewidth=1.1,
                label=f"BM25 Base ({metrics.mrr_bm25_baseline:.4f})",
                zorder=2,
            )
            for idx, model_name in enumerate(metrics.llm_models):
                sub = df[df["llm_model"] == model_name].sort_values("fb_docs")
                color, marker = _model_style(idx)
                ax.plot(
                    sub["fb_docs"],
                    sub["mrr_llm_selected"],
                    color=color,
                    linestyle="-",
                    linewidth=1.4,
                    marker=marker,
                    markersize=4.5,
                    label=f"LLM ({model_name})",
                    zorder=4,
                )
            prf_by_d = df.groupby("fb_docs", as_index=False)["mrr_prf_selected"].mean().sort_values("fb_docs")
            ax.plot(
                prf_by_d["fb_docs"],
                prf_by_d["mrr_prf_selected"],
                color="#64748B",
                linestyle="--",
                linewidth=1.1,
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label="PRF",
                zorder=3,
            )
            ax.set_xticks(metrics.fb_docs_values)
            ax.set_xlabel("Documentos de contexto ($fb\\_docs$)")
            ax.set_ylabel("MRR@10")
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        else:
            # Fixed fb_docs: plot relative gain (%) over PRF per LLM Model
            x = np.arange(len(metrics.llm_models))
            ax.axhline(0.0, color="#334155", linestyle=":", linewidth=1.0, label="Base PRF (0.0%)", zorder=2)
            gains = []
            for idx, model_name in enumerate(metrics.llm_models):
                sub = df[df["llm_model"] == model_name]
                mrr_l = float(sub["mrr_llm_selected"].mean())
                mrr_p = float(sub["mrr_prf_selected"].mean())
                gain_pct = ((mrr_l - mrr_p) / mrr_p * 100.0) if mrr_p > 0 else 0.0
                gains.append(gain_pct)
                color, marker = _model_style(idx)
                ax.plot(
                    [x[idx]],
                    [gain_pct],
                    color=color,
                    linestyle="none",
                    marker=marker,
                    markersize=5.5,
                    label=f"LLM ({model_name})",
                    zorder=5,
                )
            ax.plot(x, gains, color="#94A3B8", linestyle="-", linewidth=1.0, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics.llm_models, fontsize=7.8)
            ax.set_xlim(-0.4, len(metrics.llm_models) - 0.6)
            k_val = metrics.fb_docs_values[0] if metrics.fb_docs_values else 5
            m_val = metrics.fb_terms_values[0] if metrics.fb_terms_values else 1
            ax.set_xlabel(f"Modelo LLM ($k={k_val}, m={m_val}$)")
            ax.set_ylabel("Mejora relativa vs. PRF [%]")
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+.1f%%"))
    else:
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

            ax.plot(
                sub["fb_docs"],
                sub["mrr_llm_selected"],
                color=color,
                linestyle="-",
                linewidth=1.4,
                marker=marker,
                markersize=4.5,
                label=f"LLM ($m={fb_terms}$)",
                zorder=4,
            )
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
                label=f"PRF ($m={fb_terms}$)",
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


# ── Chart 3: Effectiveness vs. Cost (Log-scale X) ────────────────────────────
def plot_effectiveness_vs_cost(
    analyzer: SummaryBenchmarkAnalyzer,
    metrics: SummaryAnalysisMetrics,
    output_dir: str,
) -> str:
    """
    Generate Chart 3: MRR@10 (Y) vs. average time per query in seconds on a logarithmic X axis,
    discriminated either by configuration (k=fb_docs) or by LLM Model.
    """
    _apply_plot_style()
    df = analyzer.df
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    ax.axhline(
        metrics.mrr_bm25_baseline,
        color="#334155",
        linestyle=":",
        linewidth=1.1,
        label=f"BM25 Base ({metrics.mrr_bm25_baseline:.4f})",
        zorder=2,
    )

    if analyzer.group_by == "model":
        for idx, model_name in enumerate(metrics.llm_models):
            sub = df[df["llm_model"] == model_name].sort_values(["fb_docs", "fb_terms"])
            color, marker = _model_style(idx)
            ax.plot(
                sub["time_llm_avg"],
                sub["mrr_llm_selected"],
                color=color,
                linestyle="-" if len(sub) > 1 else "none",
                linewidth=1.4,
                marker=marker,
                markersize=5.0,
                label=f"LLM ({model_name})",
                zorder=4,
            )

        # Plot PRF reference
        k_val = metrics.fb_docs_values[0] if len(metrics.fb_docs_values) == 1 else None
        prf_label = f"PRF ($k={k_val}$)" if k_val is not None else "PRF"
        ax.plot(
            df["time_prf_avg"],
            df["mrr_prf_selected"],
            color="#64748B",
            linestyle="--" if len(df) > 1 else "none",
            linewidth=1.0,
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=prf_label,
            zorder=3,
        )
    else:
        for fb_docs in metrics.fb_docs_values:
            sub = df[df["fb_docs"] == fb_docs].sort_values("fb_terms")
            color = DOCS_COLORS.get(fb_docs, "#1E293B")
            marker = DOCS_MARKERS.get(fb_docs, "o")

            ax.plot(
                sub["time_llm_avg"],
                sub["mrr_llm_selected"],
                color=color,
                linestyle="-",
                linewidth=1.4,
                marker=marker,
                markersize=4.5,
                label=f"LLM ($k={fb_docs}$)",
                zorder=4,
            )
            ax.plot(
                sub["time_prf_avg"],
                sub["mrr_prf_selected"],
                color=color,
                linestyle="--",
                linewidth=1.0,
                marker=marker,
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=1.0,
                alpha=0.85,
                label=f"PRF ($k={fb_docs}$)",
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_xlim(0.035, 22.0)
    all_mrr = pd.concat([df["mrr_prf_selected"], df["mrr_llm_selected"]]).dropna()
    ax.set_ylim(
        float(all_mrr.min()) - 0.0012,
        max(float(all_mrr.max()), metrics.mrr_bm25_baseline) + 0.0012,
    )
    ax.set_xlabel("Tiempo promedio por consulta [s] (log)")
    ax.set_ylabel("MRR@10")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, which="major", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.grid(False, which="minor")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=1.8,
        columnspacing=1.0,
    )

    return _save_figure(fig, output_dir, "03_effectiveness_vs_cost")


# ── Chart 4: Inference Time vs. fb_docs (or by LLM Model) ────────────────────
def plot_inference_time_vs_fb_docs(
    analyzer: SummaryBenchmarkAnalyzer,
    metrics: SummaryAnalysisMetrics,
    output_dir: str,
) -> str:
    """
    Generate Chart 4:
    - In 'config' mode: LLM inference time per query vs. FEEDBACK_DOCS across fb_terms.
    - In 'model' mode: LLM inference time per query vs. FEEDBACK_DOCS per LLM Model
      (or vs. LLM Model when fb_docs is fixed).
    """
    _apply_plot_style()
    df = analyzer.df
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    if analyzer.group_by == "model":
        if len(metrics.fb_docs_values) > 1:
            for idx, model_name in enumerate(metrics.llm_models):
                sub = df[df["llm_model"] == model_name].sort_values("fb_docs")
                color, marker = _model_style(idx)
                ax.plot(
                    sub["fb_docs"],
                    sub["time_llm_avg"],
                    color=color,
                    linestyle="-",
                    linewidth=1.4,
                    marker=marker,
                    markersize=4.5,
                    label=f"LLM ({model_name})",
                    zorder=3,
                )
            ax.set_xticks(metrics.fb_docs_values)
            ax.set_xlabel("Documentos de contexto ($fb\\_docs$)")
        else:
            x = np.arange(len(metrics.llm_models))
            t_vals = []
            for idx, model_name in enumerate(metrics.llm_models):
                sub = df[df["llm_model"] == model_name]
                t_val = float(sub["time_llm_avg"].mean())
                t_vals.append(t_val)
                color, marker = _model_style(idx)
                ax.plot(
                    [x[idx]],
                    [t_val],
                    color=color,
                    linestyle="none",
                    marker=marker,
                    markersize=5.5,
                    label=f"LLM ({model_name})",
                    zorder=5,
                )
            ax.plot(x, t_vals, color="#94A3B8", linestyle="-", linewidth=1.0, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics.llm_models, fontsize=7.8)
            ax.set_xlim(-0.4, len(metrics.llm_models) - 0.6)
            k_val = metrics.fb_docs_values[0] if metrics.fb_docs_values else 5
            m_val = metrics.fb_terms_values[0] if metrics.fb_terms_values else 1
            ax.set_xlabel(f"Modelo LLM ($k={k_val}, m={m_val}$)")
    else:
        for fb_terms in metrics.fb_terms_values:
            sub = df[df["fb_terms"] == fb_terms].sort_values("fb_docs")
            color = TERMS_COLORS.get(fb_terms, "#2563EB")
            marker = TERMS_MARKERS.get(fb_terms, "o")
            ax.plot(
                sub["fb_docs"],
                sub["time_llm_avg"],
                color=color,
                linestyle="-",
                linewidth=1.4,
                marker=marker,
                markersize=4.5,
                label=f"LLM ($m={fb_terms}$)",
                zorder=3,
            )

        if metrics.linear_fit_slope_sec_per_doc > 0:
            x_line = np.linspace(min(metrics.fb_docs_values) - 0.3, max(metrics.fb_docs_values) + 0.3, 50)
            y_line = metrics.linear_fit_slope_sec_per_doc * x_line + metrics.linear_fit_intercept_sec
            ax.plot(
                x_line,
                y_line,
                color="#334155",
                linestyle=":",
                linewidth=1.2,
                label=f"Ajuste lineal ($R^2={metrics.linear_fit_r2:.3f}$)",
                zorder=4,
            )

        ax.set_xticks(metrics.fb_docs_values)
        ax.set_xlabel("Documentos de contexto ($fb\\_docs$)")

    ax.set_ylabel("Tiempo por consulta LLM [s]")
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

    return _save_figure(fig, output_dir, "04_inference_time_vs_fb_docs")


# ── Full Summary Suite Generation ─────────────────────────────────────────────
def generate_summary_plots(
    analyzer: SummaryBenchmarkAnalyzer,
    output_dir: str | None = None,
) -> tuple[SummaryAnalysisMetrics, dict[str, str]]:
    """
    Run general summary analysis and generate the 4 PNG figures + JSON/TXT reports.
    """
    metrics = analyzer.analyze()

    if output_dir is None:
        scope_suffix = "all" if analyzer.query_scope == "all" else "expanded"
        if analyzer.group_by == "model":
            output_dir = os.path.join("output", "figures", f"summary_models_{scope_suffix}")
        else:
            output_dir = os.path.join("output", "figures", f"summary_{scope_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    fig_paths: dict[str, str] = {}
    fig_paths["mrr_vs_fb_terms"] = plot_mrr_vs_fb_terms(analyzer, metrics, output_dir)
    fig_paths["mrr_vs_fb_docs"] = plot_mrr_vs_fb_docs(analyzer, metrics, output_dir)
    fig_paths["effectiveness_vs_cost"] = plot_effectiveness_vs_cost(analyzer, metrics, output_dir)
    fig_paths["inference_time_vs_fb_docs"] = plot_inference_time_vs_fb_docs(analyzer, metrics, output_dir)

    json_path = os.path.join(output_dir, "summary_general_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
    fig_paths["summary_json"] = json_path

    txt_path = os.path.join(output_dir, "summary_general_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(
            f"REPORTE DE ANÁLISIS GENERAL "
            f"({metrics.group_by_label.upper()} — {metrics.query_scope_label.upper()})\n"
        )
        f.write("=" * 72 + "\n\n")
        f.write(f"Archivo origen             : {metrics.source_file}\n")
        f.write(f"Discriminado por           : {metrics.group_by_label} ({metrics.group_by})\n")
        f.write(f"Alcance de consultas       : {metrics.query_scope_label} ({metrics.query_scope})\n")
        f.write(f"Corridas analizadas        : {metrics.total_configurations}\n")
        f.write(f"Modelos LLM                : {metrics.llm_models}\n")
        f.write(f"Valores de fb_docs (k)     : {metrics.fb_docs_values}\n")
        f.write(f"Valores de fb_terms (m)    : {metrics.fb_terms_values}\n")
        f.write(f"Línea base BM25 MRR@10     : {metrics.mrr_bm25_baseline:.4f} ({metrics.time_bm25_avg_baseline:.4f} s/query)\n\n")

        f.write("── MEJORES CONFIGURACIONES ──────────────────────────────────────────\n")
        f.write(
            f"  • Mejor LLM : model={metrics.best_rag_config['llm_model']}, "
            f"k={metrics.best_rag_config['fb_docs']}, "
            f"m={metrics.best_rag_config['fb_terms']} → "
            f"MRR@10={metrics.best_rag_config['mrr']:.4f} "
            f"({metrics.best_rag_config['time_avg']:.4f} s/query)\n"
        )
        f.write(
            f"  • Mejor PRF : k={metrics.best_prf_config['fb_docs']}, "
            f"m={metrics.best_prf_config['fb_terms']} → "
            f"MRR@10={metrics.best_prf_config['mrr']:.4f} "
            f"({metrics.best_prf_config['time_avg']:.4f} s/query)\n\n"
        )

        if metrics.linear_fit_slope_sec_per_doc > 0:
            f.write("── ESCALAMIENTO DE TIEMPO DE INFERENCIA VS. FB_DOCS ─────────────────\n")
            f.write(
                f"  • Regresión lineal: T(k) = {metrics.linear_fit_slope_sec_per_doc:.4f} * k "
                f"+ {metrics.linear_fit_intercept_sec:.4f} s  (R² = {metrics.linear_fit_r2:.4f})\n\n"
            )

        f.write("── TABLA DE CORRIDAS ────────────────────────────────────────────────\n")
        f.write("  LLM Model     | k  | m  | MRR PRF | MRR LLM | Time PRF (s/q) | Time LLM (s/q)\n")
        f.write("  --------------+----+----+---------+---------+----------------+---------------\n")
        for r in metrics.records:
            f.write(
                f"  {str(r['llm_model']):<13s} | {r['fb_docs']:>2d} | {r['fb_terms']:>2d} | "
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

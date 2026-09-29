# -*- coding: utf-8 -*-
"""
Visualization module for benchmark analysis.

Generates publication-quality charts:
1. Gain/Loss analysis per query (RR difference silhouette & distribution).
2. Stacked bar chart of execution time breakdown (PRF vs RAG).
3. Candidate terms proposed vs kept / lexicon filtering (LLM hallucination quantification).
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

from analysis.analyzer import BenchmarkAnalyzer, AnalysisMetrics


# ── Color Palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "gain": "#10B981",          # Emerald green (improved)
    "loss": "#EF4444",          # Vivid red (degraded)
    "neutral": "#94A3B8",       # Slate gray (unchanged)
    "first_pass": "#3B82F6",    # Blue
    "text_fetch": "#F59E0B",    # Amber
    "llm_inference": "#8B5CF6", # Purple
    "second_pass": "#10B981",   # Teal/Emerald
    "original_terms": "#64748B",# Slate
    "proposed_terms": "#8B5CF6",# Purple
    "kept_terms": "#10B981",    # Green
    "discarded_terms": "#EF4444"# Red
}


def _apply_plot_style():
    """Apply clean, modern scientific plotting styles."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.0,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


def plot_gain_loss_per_query(
    analyzer: BenchmarkAnalyzer,
    metrics: AnalysisMetrics,
    output_dir: str
) -> str:
    """
    Generate Chart 1: Per-query Reciprocal Rank gain/loss analysis.
    
    Draws a continuous silhouette of queries ordered by Delta = RR_RAG - RR_PRF,
    alongside a delta distribution histogram and a summary KPI card.
    """
    _apply_plot_style()
    merged = analyzer.merged_df.sort_values(by="delta_rr", ascending=True).reset_index(drop=True)
    delta = merged["delta_rr"].to_numpy()
    x = np.arange(len(delta))

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1.0], width_ratios=[1.3, 1.0])

    # ── 1. Top Panel: Continuous Query Waterfall Silhouette ────────────────────
    ax_top = fig.add_subplot(gs[0, :])

    # Masks for colors
    eps = 1e-7
    pos_mask = delta > eps
    neg_mask = delta < -eps
    zero_mask = np.abs(delta) <= eps

    # Draw vertical bars
    ax_top.bar(x[neg_mask], delta[neg_mask], color=PALETTE["loss"], width=1.0, align="center", label=f"Degradadas: RAG < PRF ({metrics.gain_loss.degraded_pct}%)")
    ax_top.bar(x[zero_mask], delta[zero_mask], color=PALETTE["neutral"], width=1.0, align="center", label=f"Neutras: RAG == PRF ({metrics.gain_loss.neutral_pct}%)")
    ax_top.bar(x[pos_mask], delta[pos_mask], color=PALETTE["gain"], width=1.0, align="center", label=f"Mejoradas: RAG > PRF ({metrics.gain_loss.improved_pct}%)")

    ax_top.axhline(0, color="#1E293B", linewidth=1.2, zorder=5)
    ax_top.set_xlim(-10, len(delta) + 10)
    ax_top.set_ylim(-1.05, 1.05)
    ax_top.set_xlabel("Consultas ordenadas por ganancia / pérdida (índice 0 a N)")
    ax_top.set_ylabel(r"$\Delta$ Reciprocal Rank ($RR_{RAG} - RR_{PRF}$)")
    ax_top.set_title("Análisis de Ganancias y Pérdidas por Consulta: RAG (LLM) vs. PRF Clásico", pad=12)
    ax_top.grid(axis="y", linestyle="--", alpha=0.5)
    ax_top.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1")

    # Annotate summary badge in top panel
    badge_text = (
        f"Consultas: {metrics.gain_loss.total_queries:,}\n"
        f"Δ Promedio: {metrics.gain_loss.mean_delta:+.4f}\n"
        f"Balance Neto: {metrics.gain_loss.net_wins:+d} consultas\n"
        f"Área Neta: {metrics.gain_loss.net_area:+.2f}"
    )
    ax_top.text(
        0.985, 0.05, badge_text,
        transform=ax_top.transAxes,
        ha="right", va="bottom",
        fontsize=10, fontweight="normal",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.95)
    )

    # ── 2. Bottom-Left Panel: Delta Distribution Histogram ────────────────────
    ax_hist = fig.add_subplot(gs[1, 0])
    
    # Custom bins focusing on key RR deltas
    bins = np.linspace(-1.0, 1.0, 41)
    counts, edges = np.histogram(delta, bins=bins)
    
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        center = 0.5 * (left + right)
        if center > eps:
            c = PALETTE["gain"]
        elif center < -eps:
            c = PALETTE["loss"]
        else:
            c = PALETTE["neutral"]
        ax_hist.bar(center, count, width=(right - left) * 0.9, color=c, alpha=0.85, edgecolor="none")

    ax_hist.axvline(0, color="#1E293B", linestyle="--", linewidth=1.0)
    ax_hist.set_xlabel(r"$\Delta$ Reciprocal Rank")
    ax_hist.set_ylabel("Frecuencia (N° de consultas)")
    ax_hist.set_title("Distribución de Diferencias de Reciprocal Rank", fontsize=11.5)
    ax_hist.grid(axis="both", linestyle="--", alpha=0.5)
    ax_hist.set_yscale("log")  # Log scale makes rare +/-1.0 jumps clear while accommodating neutral peak
    ax_hist.set_ylabel("Frecuencia (Escala Log)")

    # ── 3. Bottom-Right Panel: Categorical Breakdown (Donut) ──────────────────
    ax_donut = fig.add_subplot(gs[1, 1])
    categories = [
        f"Mejoradas\n({metrics.gain_loss.improved_count:,})",
        f"Neutras\n({metrics.gain_loss.neutral_count:,})",
        f"Degradadas\n({metrics.gain_loss.degraded_count:,})"
    ]
    counts_donut = [
        metrics.gain_loss.improved_count,
        metrics.gain_loss.neutral_count,
        metrics.gain_loss.degraded_count
    ]
    donut_colors = [PALETTE["gain"], PALETTE["neutral"], PALETTE["loss"]]

    wedges, texts, autotexts = ax_donut.pie(
        counts_donut,
        labels=categories,
        colors=donut_colors,
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax_donut.set_title("Proporción Ganancias / Empates / Pérdidas", fontsize=11.5)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "01_gain_loss_per_query.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_time_breakdown(
    analyzer: BenchmarkAnalyzer,
    metrics: AnalysisMetrics,
    output_dir: str
) -> str:
    """
    Generate Chart 2: Stacked bar chart of execution time breakdown.
    
    Shows first pass, text fetch, LLM inference, and second pass for PRF vs RAG.
    Includes both absolute time in seconds (linear scale) and 100% relative breakdown.
    """
    _apply_plot_style()
    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    methods = ["PRF", "RAG (LLM)"]
    t_prf = metrics.timings_prf
    t_rag = metrics.timings_rag

    # Component data
    first_pass = [t_prf.first_pass_mean, t_rag.first_pass_mean]
    text_fetch = [t_prf.text_fetch_mean, t_rag.text_fetch_mean]
    llm_infer  = [t_prf.llm_inference_mean, t_rag.llm_inference_mean]
    second_pass= [t_prf.second_pass_mean, t_rag.second_pass_mean]

    # Percentages
    pct_first  = [t_prf.first_pass_pct, t_rag.first_pass_pct]
    pct_fetch  = [t_prf.text_fetch_pct, t_rag.text_fetch_pct]
    pct_llm    = [t_prf.llm_inference_pct, t_rag.llm_inference_pct]
    pct_second = [t_prf.second_pass_pct, t_rag.second_pass_pct]

    colors = [
        PALETTE["first_pass"],
        PALETTE["text_fetch"],
        PALETTE["llm_inference"],
        PALETTE["second_pass"]
    ]
    labels = [
        "1ª Pasada (BM25)",
        "Obtención de Texto (Text Fetch)",
        "Inferencia LLM",
        "2ª Pasada (Búsqueda expandida)"
    ]

    width = 0.55
    x = np.arange(len(methods))

    # ── Left Plot: Absolute Stacked Bars (Seconds) ────────────────────────────
    b1 = ax_abs.bar(x, first_pass, width, color=colors[0], label=labels[0], edgecolor="white", linewidth=1)
    b2 = ax_abs.bar(x, text_fetch, width, bottom=first_pass, color=colors[1], label=labels[1], edgecolor="white", linewidth=1)
    bottom_3 = np.array(first_pass) + np.array(text_fetch)
    b3 = ax_abs.bar(x, llm_infer, width, bottom=bottom_3, color=colors[2], label=labels[2], edgecolor="white", linewidth=1)
    bottom_4 = bottom_3 + np.array(llm_infer)
    b4 = ax_abs.bar(x, second_pass, width, bottom=bottom_4, color=colors[3], label=labels[3], edgecolor="white", linewidth=1)

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(methods, fontsize=11, fontweight="bold")
    ax_abs.set_ylabel("Tiempo promedio por consulta (segundos)")
    ax_abs.set_title("Desglose Absoluto de Tiempos de Ejecución", pad=12)
    ax_abs.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate totals and LLM callout
    total_prf = t_prf.total_time_mean
    total_rag = t_rag.total_time_mean
    ax_abs.text(0, total_prf + 0.15, f"{total_prf:.3f} s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_abs.text(1, total_rag + 0.15, f"{total_rag:.3f} s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Inset callout highlighting LLM dominance and marginal text fetch
    callout_txt = (
        f"INFERENCIA LLM PROMEDIO: {t_rag.llm_inference_mean:.2f} s ({t_rag.llm_inference_pct:.1f}% del total)\n"
        f"• Text Fetch RAG: {t_rag.text_fetch_mean*1000:.1f} ms ({t_rag.text_fetch_pct:.2f}% - Marginal)\n"
        f"• Ratio tiempo RAG / PRF: {(total_rag / total_prf):.1f}x"
    )
    ax_abs.text(
        0.5, 0.70, callout_txt,
        transform=ax_abs.transAxes,
        ha="center", va="center",
        fontsize=9.5, fontweight="normal",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#F5F3FF", edgecolor="#8B5CF6", alpha=0.95)
    )

    # ── Right Plot: Relative 100% Stacked Bars (%) ───────────────────────────
    ax_pct.bar(x, pct_first, width, color=colors[0], label=labels[0], edgecolor="white", linewidth=1)
    ax_pct.bar(x, pct_fetch, width, bottom=pct_first, color=colors[1], label=labels[1], edgecolor="white", linewidth=1)
    bottom_pct_3 = np.array(pct_first) + np.array(pct_fetch)
    ax_pct.bar(x, pct_llm, width, bottom=bottom_pct_3, color=colors[2], label=labels[2], edgecolor="white", linewidth=1)
    bottom_pct_4 = bottom_pct_3 + np.array(pct_llm)
    ax_pct.bar(x, pct_second, width, bottom=bottom_pct_4, color=colors[3], label=labels[3], edgecolor="white", linewidth=1)

    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(methods, fontsize=11, fontweight="bold")
    ax_pct.set_ylabel("Proporción del tiempo total (%)")
    ax_pct.set_ylim(0, 105)
    ax_pct.set_title("Composición Porcentual del Tiempo de Búsqueda", pad=12)
    ax_pct.grid(axis="y", linestyle="--", alpha=0.5)

    # Add text labels inside bars on the percentage plot
    for i, m_pct in enumerate([pct_first, pct_fetch, pct_llm, pct_second]):
        for j in range(2):
            val = m_pct[j]
            if val >= 5.0:  # Only label if visible
                # Compute center y
                if i == 0:
                    y_pos = val / 2.0
                elif i == 1:
                    y_pos = pct_first[j] + val / 2.0
                elif i == 2:
                    y_pos = bottom_pct_3[j] + val / 2.0
                else:
                    y_pos = bottom_pct_4[j] + val / 2.0
                ax_pct.text(j, y_pos, f"{val:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    # Shared legend placed centrally at the top
    handles = [Patch(facecolor=c, edgecolor="none", label=lbl) for c, lbl in zip(colors, labels)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=True, facecolor="white", edgecolor="#CBD5E1")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "02_time_breakdown_stacked.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_lexicon_filtering(
    analyzer: BenchmarkAnalyzer,
    metrics: AnalysisMetrics,
    output_dir: str
) -> str:
    """
    Generate Chart 3: LLM candidate terms proposed vs kept (lexicon filter).
    
    Quantifies LLM hallucination rate and terms surviving the lexicon filter.
    """
    _apply_plot_style()
    lex = metrics.lexicon

    fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.2])

    rag = analyzer.rag_df

    # ── Panel 1: Query Term Lifecycle Comparison ──────────────────────────────
    ax_stages = fig.add_subplot(gs[0, 0])
    
    stages = [
        "Query\nOriginal",
        "Candidatos\nPropuestos (LLM)",
        "Aceptados\n(Lexicón)",
        "Query Final\nExpandida"
    ]
    stage_vals = [
        lex.q_terms_original_mean,
        lex.n_terms_proposed_mean,
        lex.n_terms_kept_mean,
        lex.q_terms_expanded_mean
    ]
    stage_colors = [
        PALETTE["original_terms"],
        PALETTE["proposed_terms"],
        PALETTE["kept_terms"],
        "#3B82F6"
    ]

    bars = ax_stages.bar(stages, stage_vals, color=stage_colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax_stages.set_ylabel("Cantidad promedio de términos")
    ax_stages.set_title("Ciclo de Términos en RAG", pad=12)
    ax_stages.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, stage_vals):
        ax_stages.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    # ── Panel 2: LLM Candidate Fate (Donut: Kept vs Discarded/Hallucinated) ───
    ax_donut = fig.add_subplot(gs[0, 1])

    if lex.has_term_data and lex.total_proposed > 0:
        fate_labels = [
            f"Aceptados\n(En Lexicón)\n{lex.total_kept:,}",
            f"Descartados\n(Alucinación)\n{lex.total_discarded:,}"
        ]
        fate_counts = [lex.total_kept, lex.total_discarded]
        fate_colors = [PALETTE["kept_terms"], PALETTE["discarded_terms"]]

        wedges, texts, autotexts = ax_donut.pie(
            fate_counts,
            labels=fate_labels,
            colors=fate_colors,
            autopct="%1.1f%%",
            pctdistance=0.72,
            startangle=90,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(10.5)
            at.set_fontweight("bold")
            at.set_color("white")
        ax_donut.set_title("Destino de Términos del LLM", pad=12)
    else:
        ax_donut.text(0.5, 0.5, "Datos de candidatos\nno disponibles", ha="center", va="center")
        ax_donut.axis("off")

    # ── Panel 3: Distribution per Query (Proposed vs Kept) ────────────────────
    ax_dist = fig.add_subplot(gs[0, 2])

    if lex.has_term_data and "n_terms_proposed" in rag.columns and "n_terms_kept" in rag.columns:
        max_term = int(max(rag["n_terms_proposed"].max(), 5))
        bins = np.arange(0, max_term + 2) - 0.5

        ax_dist.hist(
            [rag["n_terms_proposed"], rag["n_terms_kept"]],
            bins=bins,
            color=[PALETTE["proposed_terms"], PALETTE["kept_terms"]],
            label=["Términos Propuestos", "Términos Aceptados"],
            rwidth=0.75,
            edgecolor="white"
        )
        ax_dist.set_xlabel("Número de términos por consulta")
        ax_dist.set_ylabel("N° de consultas")
        ax_dist.set_title("Distribución de Propuestos vs Aceptados", pad=12)
        ax_dist.grid(axis="y", linestyle="--", alpha=0.5)
        ax_dist.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    else:
        ax_dist.text(0.5, 0.5, "Distribución no disponible", ha="center", va="center")
        ax_dist.axis("off")

    # Bottom summary badge
    summary_txt = (
        f"Resumen de Filtrado de Lexicón / Alucinación:\n"
        f"• Términos propuestos por LLM: {lex.total_proposed:,}  (Promedio: {lex.n_terms_proposed_mean:.2f})\n"
        f"• Términos aceptados en lexicón: {lex.total_kept:,} ({lex.survival_rate_pct:.1f}% supervivencia)\n"
        f"• Términos descartados / alucinados: {lex.total_discarded:,} ({lex.hallucination_rate_pct:.1f}% tasa de alucinación)\n"
        f"• Expansión neta efectiva: +{lex.n_terms_kept_mean:.2f} términos por consulta"
    )
    fig.text(
        0.5, -0.06, summary_txt,
        ha="center", va="top",
        fontsize=10.5, fontweight="normal",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95)
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "03_lexicon_filtering_hallucination.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_plots(
    analyzer: BenchmarkAnalyzer,
    output_dir: str | None = None
) -> tuple[AnalysisMetrics, dict[str, str]]:
    """
    Run full analysis and generate all figures + metrics summary file.
    
    Parameters
    ----------
    analyzer : BenchmarkAnalyzer
        Initialized analyzer instance.
    output_dir : str | None
        Directory where figures will be stored. Defaults to ./figures/<file_stem>/.
        
    Returns
    -------
    tuple[AnalysisMetrics, dict[str, str]]
        Computed metrics object and mapping of figure name -> saved file path.
    """
    metrics = analyzer.analyze()

    if output_dir is None:
        output_dir = os.path.join("figures", metrics.file_stem)
    os.makedirs(output_dir, exist_ok=True)

    fig_paths = {}
    fig_paths["gain_loss"] = plot_gain_loss_per_query(analyzer, metrics, output_dir)
    fig_paths["time_breakdown"] = plot_time_breakdown(analyzer, metrics, output_dir)
    fig_paths["lexicon_filtering"] = plot_lexicon_filtering(analyzer, metrics, output_dir)

    # Export JSON metrics summary
    json_path = os.path.join(output_dir, "summary_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
    fig_paths["summary_json"] = json_path

    # Export human-readable text report
    txt_path = os.path.join(output_dir, "summary_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"REPORTE DE ANÁLISIS: {metrics.file_stem}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Archivo origen: {metrics.source_file}\n")
        f.write(f"Consultas comunes analizadas (RAG vs PRF): {metrics.common_query_count:,}\n\n")

        f.write("── 1. MÉTRICAS DE EFECTIVIDAD (MRR@10) ────────────────────────────────\n")
        f.write(f"  • MRR@10 RAG (LLM) : {metrics.mrr_rag_mean:.4f}\n")
        f.write(f"  • MRR@10 PRF       : {metrics.mrr_prf_mean:.4f}\n")
        f.write(f"  • Diferencia (Δ)   : {metrics.mrr_diff:+.4f} ({metrics.mrr_relative_gain_pct:+.2f}% relativo)\n\n")

        f.write("── 2. GANANCIAS Y PÉRDIDAS POR CONSULTA ───────────────────────────────\n")
        f.write(f"  • Consultas Mejoradas (RAG > PRF) : {metrics.gain_loss.improved_count:,} ({metrics.gain_loss.improved_pct:.2f}%)\n")
        f.write(f"  • Consultas Degradadas (RAG < PRF): {metrics.gain_loss.degraded_count:,} ({metrics.gain_loss.degraded_pct:.2f}%)\n")
        f.write(f"  • Consultas Neutras (RAG == PRF)  : {metrics.gain_loss.neutral_count:,} ({metrics.gain_loss.neutral_pct:.2f}%)\n")
        f.write(f"  • Balance Neto                    : {metrics.gain_loss.net_wins:+d} consultas\n")
        f.write(f"  • Δ RR Promedio                   : {metrics.gain_loss.mean_delta:+.4f}\n")
        f.write(f"  • Área Positiva vs Negativa       : {metrics.gain_loss.positive_area:.2f} vs {metrics.gain_loss.negative_area:.2f} (Neta: {metrics.gain_loss.net_area:+.2f})\n\n")

        f.write("── 3. TIEMPOS DE EJECUCIÓN E INFERENCIA ───────────────────────────────\n")
        f.write(f"  • ⭐ TIEMPO PROMEDIO DE INFERENCIA LLM : {metrics.mean_inference_time_seconds:.4f} s\n")
        f.write(f"  • RAG Tiempo Total Promedio            : {metrics.timings_rag.total_time_mean:.4f} s\n")
        f.write(f"      - 1ª Pasada (BM25)                 : {metrics.timings_rag.first_pass_mean:.4f} s ({metrics.timings_rag.first_pass_pct:.1f}%)\n")
        f.write(f"      - Text Fetch                       : {metrics.timings_rag.text_fetch_mean:.4f} s ({metrics.timings_rag.text_fetch_pct:.2f}%)\n")
        f.write(f"      - Inferencia LLM                   : {metrics.timings_rag.llm_inference_mean:.4f} s ({metrics.timings_rag.llm_inference_pct:.1f}%)\n")
        f.write(f"      - 2ª Pasada                        : {metrics.timings_rag.second_pass_mean:.4f} s ({metrics.timings_rag.second_pass_pct:.1f}%)\n")
        f.write(f"  • PRF Tiempo Total Promedio            : {metrics.timings_prf.total_time_mean:.4f} s\n")
        f.write(f"      - 1ª Pasada                        : {metrics.timings_prf.first_pass_mean:.4f} s ({metrics.timings_prf.first_pass_pct:.1f}%)\n")
        f.write(f"      - 2ª Pasada                        : {metrics.timings_prf.second_pass_mean:.4f} s ({metrics.timings_prf.second_pass_pct:.1f}%)\n\n")

        f.write("── 4. FILTRADO DE LEXICÓN Y ALUCINACIÓN DEL LLM ───────────────────────\n")
        lex = metrics.lexicon
        f.write(f"  • Longitud Promedio Query Original   : {lex.q_terms_original_mean:.2f} términos\n")
        f.write(f"  • Términos Propuestos por LLM (media): {lex.n_terms_proposed_mean:.2f} (Total: {lex.total_proposed:,})\n")
        f.write(f"  • Términos Aceptados en Lexicón      : {lex.n_terms_kept_mean:.2f} (Total: {lex.total_kept:,})\n")
        f.write(f"  • Términos Descartados (Alucinación) : {lex.n_terms_discarded_mean:.2f} (Total: {lex.total_discarded:,})\n")
        f.write(f"  • Tasa de Supervivencia en Lexicón   : {lex.survival_rate_pct:.2f}%\n")
        f.write(f"  • Tasa de Alucinación / Descarte     : {lex.hallucination_rate_pct:.2f}%\n")
        f.write(f"  • Longitud Promedio Query Expandida  : {lex.q_terms_expanded_mean:.2f} términos\n")
        f.write("=" * 70 + "\n")
    fig_paths["summary_txt"] = txt_path

    return metrics, fig_paths

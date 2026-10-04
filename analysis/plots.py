# -*- coding: utf-8 -*-
"""
Visualization module for benchmark analysis.

Generates publication-quality figures formatted for scientific papers (LNCS / LaTeX):
- Serif typography (Times New Roman / DejaVu Serif, STIX mathtext).
- Single-column standard width (~3.4 inches).
- Clean visual minimalism (no top/right spines, subtle grids, uncluttered labels).
- Vector PDF output (with companion PNG for local previews).

Charts generated:
1.  01_gain_loss_per_query.pdf (.png)       - Per-query Reciprocal Rank gain/loss waterfall.
1b. 01b_gain_loss_donut.pdf (.png)          - Donut: Improved / Neutral / Degraded.
2.  02_time_breakdown_stacked.pdf (.png)    - Stacked 100% time breakdown with latency callouts.
3.  03_lexicon_filtering_donut.pdf (.png)   - Donut: LLM proposed terms kept vs hallucinated.
4.  04_terms_distribution_per_query.pdf (.png) - Per-query terms distribution (X axis [0, 5]).
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


# ── Color Palette (Sobrio / Scientific) ────────────────────────────────────────
PALETTE = {
    "gain": "#15803D",          # Deep academic green
    "loss": "#B91C1C",          # Deep academic red
    "neutral": "#94A3B8",       # Slate gray
    "first_pass": "#2563EB",    # Blue
    "text_fetch": "#D97706",    # Amber
    "llm_inference": "#7C3AED", # Purple
    "second_pass": "#059669",   # Emerald
    "proposed_terms": "#7C3AED",# Purple
    "kept_terms": "#15803D",    # Deep green
    "discarded_terms": "#B91C1C",# Deep red
    "rejected_stopword": "#94A3B8",   # Slate gray
    "rejected_duplicate": "#D97706",  # Amber
    "rejected_lexicon": "#B91C1C",    # Deep red
    "rejected_truncated": "#0284C7",  # Deep sky blue / teal
}


def _apply_plot_style():
    """Apply scientific publication style (LNCS / LaTeX compatible)."""
    plt.rcParams.update({
        # Tipografía serif, para que combine con LNCS
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        # Tamaños pensados para una figura que se reduce al ancho de columna
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # Sobriedad: sin marco superior/derecho, grilla tenue
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,

        # Salida vectorial, sin recorte de etiquetas
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _format_time_compact(sec: float) -> str:
    """Format time compactly for small publication figures."""
    if sec >= 1.0:
        return f"{sec:.2f}s"
    elif sec >= 0.01:
        return f"{sec * 1000:.0f}ms"
    elif sec > 0.00001:
        return f"{sec * 1000:.1f}ms"
    else:
        return "0ms"


def _save_figure(fig: plt.Figure, output_dir: str, filename_stem: str) -> str:
    """
    Save figure as publication-ready vector PDF and companion PNG.
    
    Returns the path to the primary vector PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{filename_stem}.pdf")
    png_path = os.path.join(output_dir, f"{filename_stem}.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path


# ── Chart 1: Per-query Gain/Loss Waterfall Silhouette ──────────────────────────
def plot_gain_loss_per_query(
    analyzer: BenchmarkAnalyzer,
    metrics: AnalysisMetrics,
    output_dir: str
) -> str:
    """
    Generate Chart 1: Per-query Reciprocal Rank gain/loss waterfall silhouette.
    Column width ~3.4 inches.
    """
    _apply_plot_style()
    merged = analyzer.merged_df.sort_values(by="delta_rr", ascending=True).reset_index(drop=True)
    delta = merged["delta_rr"].to_numpy()
    x = np.arange(len(delta))

    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    eps = 1e-7
    pos_mask = delta > eps
    neg_mask = delta < -eps
    zero_mask = np.abs(delta) <= eps

    ax.bar(
        x[neg_mask], delta[neg_mask],
        color=PALETTE["loss"], width=1.0, align="center",
        label=f"Degr. ({metrics.gain_loss.degraded_pct:.1f}%)"
    )
    ax.bar(
        x[zero_mask], delta[zero_mask],
        color=PALETTE["neutral"], width=1.0, align="center",
        label=f"Neutras ({metrics.gain_loss.neutral_pct:.1f}%)"
    )
    ax.bar(
        x[pos_mask], delta[pos_mask],
        color=PALETTE["gain"], width=1.0, align="center",
        label=f"Mej. ({metrics.gain_loss.improved_pct:.1f}%)"
    )

    ax.axhline(0, color="#1E293B", linewidth=0.8, zorder=5)
    ax.set_xlim(-max(len(delta) * 0.02, 1), len(delta) + max(len(delta) * 0.02, 1))
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Consultas (ordenadas)")
    ax.set_ylabel(r"$\Delta$ RR ($RR_{\mathrm{RAG}} - RR_{\mathrm{PRF}}$)")
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=1.0)

    # Inset badge
    badge = f"Neto: {metrics.gain_loss.net_wins:+d}\n" + r"$\Delta_{\mathrm{med}}$: " + f"{metrics.gain_loss.mean_delta:+.3f}"
    ax.text(
        0.98, 0.06, badge,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9, lw=0.6)
    )

    return _save_figure(fig, output_dir, "01_gain_loss_per_query")


# ── Chart 1b: Gain / Loss Proportion Donut (Separated) ────────────────────────
def plot_gain_loss_donut(
    analyzer: BenchmarkAnalyzer,
    metrics: AnalysisMetrics,
    output_dir: str
) -> str:
    """
    Generate Chart 1b: Categorical breakdown donut (Improved vs Neutral vs Degraded).
    Column width ~3.4 inches.
    """
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))

    categories = [
        f"Mej.\n{metrics.gain_loss.improved_pct:.1f}%",
        f"Neutras\n{metrics.gain_loss.neutral_pct:.1f}%",
        f"Degr.\n{metrics.gain_loss.degraded_pct:.1f}%"
    ]
    counts = [
        metrics.gain_loss.improved_count,
        metrics.gain_loss.neutral_count,
        metrics.gain_loss.degraded_count
    ]
    donut_colors = [PALETTE["gain"], PALETTE["neutral"], PALETTE["loss"]]

    wedges, texts = ax.pie(
        counts,
        labels=categories,
        colors=donut_colors,
        startangle=90,
        textprops=dict(fontsize=8),
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2)
    )

    # Center label with net balance
    ax.text(
        0, 0,
        f"Neto\n{metrics.gain_loss.net_wins:+d}",
        ha="center", va="center",
        fontsize=8.5, fontweight="bold", color="#1E293B"
    )

    return _save_figure(fig, output_dir, "01b_gain_loss_donut")


# ── Chart 2: Time Breakdown with Combined Times and Percentages ────────────────
def plot_time_breakdown(
    analyzer: BenchmarkAnalyzer,
    metrics: AnalysisMetrics,
    output_dir: str
) -> str:
    """
    Generate Chart 2: Stacked 100% bar chart combining percentages and absolute times.
    Column width ~3.4 inches.
    """
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    methods = ["PRF", "RAG"]
    t_prf = metrics.timings_prf
    t_rag = metrics.timings_rag

    # Raw times in seconds
    times = [
        [t_prf.first_pass_mean, t_rag.first_pass_mean],
        [t_prf.text_fetch_mean, t_rag.text_fetch_mean],
        [t_prf.llm_inference_mean, t_rag.llm_inference_mean],
        [t_prf.second_pass_mean, t_rag.second_pass_mean]
    ]

    # Percentages
    pcts = [
        [t_prf.first_pass_pct, t_rag.first_pass_pct],
        [t_prf.text_fetch_pct, t_rag.text_fetch_pct],
        [t_prf.llm_inference_pct, t_rag.llm_inference_pct],
        [t_prf.second_pass_pct, t_rag.second_pass_pct]
    ]

    colors = [
        PALETTE["first_pass"],
        PALETTE["text_fetch"],
        PALETTE["llm_inference"],
        PALETTE["second_pass"]
    ]
    labels = ["1ª Pasada", "Text Fetch", "Inferencia LLM", "2ª Pasada"]

    width = 0.32
    x = np.array([0.22, 0.68])

    # Stack bars
    bottom = np.zeros(len(methods))
    for i in range(4):
        ax.bar(
            x, pcts[i], width,
            bottom=bottom, color=colors[i], label=labels[i],
            edgecolor="white", linewidth=0.8
        )
        bottom += np.array(pcts[i])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Tiempo total (%)")
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.1, 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

    # ── Annotate PRF (x[0]): Inside labels ─────────────────────────────────────
    bottom_prf = 0.0
    for i in range(4):
        p = pcts[i][0]
        t = times[i][0]
        if p >= 8.0:
            y_pos = bottom_prf + p / 2.0
            ax.text(
                x[0], y_pos,
                f"{p:.0f}%\n({_format_time_compact(t)})",
                ha="center", va="center",
                color="white", fontweight="bold", fontsize=7.5
            )
        bottom_prf += p

    # ── Annotate RAG (x[1]): LLM inside, marginal slices callouts ─────────────
    p_llm = pcts[2][1]
    t_llm = times[2][1]
    bottom_rag_llm = pcts[0][1] + pcts[1][1]
    y_llm_center = bottom_rag_llm + p_llm / 2.0
    ax.text(
        x[1], y_llm_center,
        f"{p_llm:.1f}%\n({_format_time_compact(t_llm)})",
        ha="center", va="center",
        color="white", fontweight="bold", fontsize=7.5
    )

    # Side callouts for RAG marginal phases
    p_sec = pcts[3][1]
    t_sec = times[3][1]
    y_sec_center = bottom_rag_llm + p_llm + p_sec / 2.0
    ax.annotate(
        f"2ª: {_format_time_compact(t_sec)}",
        xy=(x[1] + width / 2, y_sec_center),
        xytext=(x[1] + width / 2 + 0.12, 98.0),
        arrowprops=dict(arrowstyle="->", color=colors[3], lw=0.8),
        va="center", fontsize=7.5, color=colors[3], fontweight="bold"
    )

    t_fetch = times[1][1]
    t_first = times[0][1]
    ax.annotate(
        f"Fetch: {_format_time_compact(t_fetch)}\n1ª: {_format_time_compact(t_first)}",
        xy=(x[1] + width / 2, 1.0),
        xytext=(x[1] + width / 2 + 0.12, 14.0),
        arrowprops=dict(arrowstyle="->", color=colors[1], lw=0.8),
        va="center", fontsize=7.5, color="#B45309", fontweight="bold"
    )

    # Total time labels above each bar
    total_prf = t_prf.total_time_mean
    total_rag = t_rag.total_time_mean
    ax.text(
        x[0], 102.0,
        f"{_format_time_compact(total_prf)}",
        ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1E293B"
    )
    ax.text(
        x[1], 102.0,
        f"{_format_time_compact(total_rag)}",
        ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1E293B"
    )

    # Clean compact legend at top
    handles = [Patch(facecolor=c, edgecolor="none", label=lbl) for c, lbl in zip(colors, labels)]
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False, fontsize=7.2, handlelength=0.8
    )

    return _save_figure(fig, output_dir, "02_time_breakdown_stacked")


# ── Chart 3: Lexicon Filtering / Hallucination Donut (Separated) ──────────────
def plot_lexicon_filtering_donut(
    analyzer: BenchmarkAnalyzer,
    metrics: AnalysisMetrics,
    output_dir: str
) -> str:
    """
    Generate Chart 3: Donut chart of candidate terms fate (Kept vs Discarded / Hallucinated).
    Column width ~3.4 inches.
    """
    _apply_plot_style()
    lex = metrics.lexicon
    fig, ax = plt.subplots(figsize=(3.4, 2.3))

    if lex.has_term_data and lex.total_proposed > 0:
        if lex.has_breakdown and (
            lex.total_rejected_stopword > 0
            or lex.total_rejected_duplicate > 0
            or lex.total_rejected_lexicon > 0
            or lex.total_rejected_truncated > 0
        ):
            fate_labels = []
            fate_counts = []
            fate_colors = []

            fate_labels.append(f"Aceptados\n{lex.survival_rate_pct:.1f}%")
            fate_counts.append(lex.total_kept)
            fate_colors.append(PALETTE["kept_terms"])

            if lex.total_rejected_lexicon > 0:
                fate_labels.append(f"Alucinados\n{lex.rejected_lexicon_pct:.1f}%")
                fate_counts.append(lex.total_rejected_lexicon)
                fate_colors.append(PALETTE["rejected_lexicon"])

            if lex.total_rejected_truncated > 0:
                fate_labels.append(f"Truncados\n{lex.rejected_truncated_pct:.1f}%")
                fate_counts.append(lex.total_rejected_truncated)
                fate_colors.append(PALETTE["rejected_truncated"])

            if lex.total_rejected_duplicate > 0:
                fate_labels.append(f"Duplicados\n{lex.rejected_duplicate_pct:.1f}%")
                fate_counts.append(lex.total_rejected_duplicate)
                fate_colors.append(PALETTE["rejected_duplicate"])

            if lex.total_rejected_stopword > 0:
                fate_labels.append(f"Stopwords\n{lex.rejected_stopword_pct:.1f}%")
                fate_counts.append(lex.total_rejected_stopword)
                fate_colors.append(PALETTE["rejected_stopword"])
        else:
            fate_labels = [
                f"Aceptados\n{lex.survival_rate_pct:.1f}%",
                f"Alucinados\n{lex.hallucination_rate_pct:.1f}%"
            ]
            fate_counts = [lex.total_kept, lex.total_discarded]
            fate_colors = [PALETTE["kept_terms"], PALETTE["discarded_terms"]]

        wedges, texts = ax.pie(
            fate_counts,
            labels=fate_labels,
            colors=fate_colors,
            startangle=90,
            textprops=dict(fontsize=8),
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2)
        )

        ax.text(
            0, 0,
            f"Términos\n{lex.total_proposed:,}",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#1E293B"
        )
    else:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", fontsize=8)
        ax.axis("off")

    return _save_figure(fig, output_dir, "03_lexicon_filtering_donut")


# ── Chart 4: Terms Distribution per Query (Range 0 to 5) ──────────────────────
def plot_terms_distribution(
    analyzer: BenchmarkAnalyzer,
    metrics: AnalysisMetrics,
    output_dir: str
) -> str:
    """
    Generate Chart 4: Per-query proposed vs kept terms distribution (X axis range [0, 5]).
    Column width ~3.4 inches.
    """
    _apply_plot_style()
    lex = metrics.lexicon
    rag = analyzer.rag_df

    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    if lex.has_term_data and "n_terms_proposed" in rag.columns and "n_terms_kept" in rag.columns:
        bins = np.arange(0, 7) - 0.5

        proposed = np.clip(rag["n_terms_proposed"].fillna(0), 0, 5)
        kept = np.clip(rag["n_terms_kept"].fillna(0), 0, 5)

        counts_prop, _ = np.histogram(proposed, bins=bins)
        counts_kept, _ = np.histogram(kept, bins=bins)

        x = np.arange(6)
        width = 0.35

        b1 = ax.bar(
            x - width / 2, counts_prop, width,
            color=PALETTE["proposed_terms"], label="Propuestos",
            edgecolor="white", linewidth=0.8
        )
        b2 = ax.bar(
            x + width / 2, counts_kept, width,
            color=PALETTE["kept_terms"], label="Aceptados",
            edgecolor="white", linewidth=0.8
        )

        max_count = max(counts_prop.max(), counts_kept.max())
        y_offset = max(max_count * 0.02, 0.15)
        ax.set_ylim(0, max_count * 1.25 + 0.5)

        # Annotate counts on top of bars
        for bar, count in zip(b1, counts_prop):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                    f"{int(count):,}",
                    ha="center", va="bottom", fontsize=7,
                    color=PALETTE["proposed_terms"], fontweight="bold"
                )
        for bar, count in zip(b2, counts_kept):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                    f"{int(count):,}",
                    ha="center", va="bottom", fontsize=7,
                    color=PALETTE["kept_terms"], fontweight="bold"
                )

        ax.set_xlim(-0.6, 5.6)
        ax.set_xticks(range(6))
        ax.set_xticklabels(["0", "1", "2", "3", "4", "5"], fontsize=8)
        ax.set_xlabel("Términos por consulta")
        ax.set_ylabel("N° Consultas")
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=1.0)
    else:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", fontsize=8)
        ax.axis("off")

    return _save_figure(fig, output_dir, "04_terms_distribution_per_query")


# ── Full Suite Generation ──────────────────────────────────────────────────────
def generate_all_plots(
    analyzer: BenchmarkAnalyzer,
    output_dir: str | None = None
) -> tuple[AnalysisMetrics, dict[str, str]]:
    """
    Run full analysis and generate all figures (PDF + PNG) + metrics summary files.
    """
    metrics = analyzer.analyze()

    if output_dir is None:
        output_dir = os.path.join("output", "figures", metrics.file_stem)
    os.makedirs(output_dir, exist_ok=True)

    fig_paths = {}
    fig_paths["gain_loss"] = plot_gain_loss_per_query(analyzer, metrics, output_dir)
    fig_paths["gain_loss_donut"] = plot_gain_loss_donut(analyzer, metrics, output_dir)
    fig_paths["time_breakdown"] = plot_time_breakdown(analyzer, metrics, output_dir)
    fig_paths["lexicon_filtering"] = plot_lexicon_filtering_donut(analyzer, metrics, output_dir)
    fig_paths["terms_distribution"] = plot_terms_distribution(analyzer, metrics, output_dir)

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
        f.write(f"  • TIEMPO PROMEDIO DE INFERENCIA LLM : {metrics.mean_inference_time_seconds:.4f} s\n")
        f.write(f"  • RAG Tiempo Total Promedio         : {metrics.timings_rag.total_time_mean:.4f} s\n")
        f.write(f"      - 1ª Pasada (BM25)              : {metrics.timings_rag.first_pass_mean:.4f} s ({metrics.timings_rag.first_pass_pct:.1f}%)\n")
        f.write(f"      - Text Fetch                    : {metrics.timings_rag.text_fetch_mean:.4f} s ({metrics.timings_rag.text_fetch_pct:.2f}%)\n")
        f.write(f"      - Inferencia LLM                : {metrics.timings_rag.llm_inference_mean:.4f} s ({metrics.timings_rag.llm_inference_pct:.1f}%)\n")
        f.write(f"      - 2ª Pasada                     : {metrics.timings_rag.second_pass_mean:.4f} s ({metrics.timings_rag.second_pass_pct:.1f}%)\n")
        f.write(f"  • PRF Tiempo Total Promedio         : {metrics.timings_prf.total_time_mean:.4f} s\n")
        f.write(f"      - 1ª Pasada                     : {metrics.timings_prf.first_pass_mean:.4f} s ({metrics.timings_prf.first_pass_pct:.1f}%)\n")
        f.write(f"      - 2ª Pasada                     : {metrics.timings_prf.second_pass_mean:.4f} s ({metrics.timings_prf.second_pass_pct:.1f}%)\n\n")

        f.write("── 4. FILTRADO DE LEXICÓN Y RECHAZOS DEL LLM ─────────────────────────\n")
        lex = metrics.lexicon
        f.write(f"  • Longitud Promedio Query Original   : {lex.q_terms_original_mean:.2f} términos\n")
        f.write(f"  • Términos Propuestos por LLM (media): {lex.n_terms_proposed_mean:.2f} (Total: {lex.total_proposed:,})\n")
        f.write(f"  • Términos Aceptados en Consulta     : {lex.n_terms_kept_mean:.2f} (Total: {lex.total_kept:,})\n")
        if lex.has_breakdown:
            f.write(f"  • Desglose de Términos Rechazados / No Utilizados:\n")
            f.write(f"      - Por stopwords                     : {lex.total_rejected_stopword:,} ({lex.rejected_stopword_pct:.2f}%)\n")
            f.write(f"      - Por duplicación                   : {lex.total_rejected_duplicate:,} ({lex.rejected_duplicate_pct:.2f}%)\n")
            f.write(f"      - Por no estar en léxico (alucinac) : {lex.total_rejected_lexicon:,} ({lex.rejected_lexicon_pct:.2f}%)\n")
            if lex.total_rejected_truncated > 0:
                f.write(f"      - No evaluados (por truncamiento)   : {lex.total_rejected_truncated:,} ({lex.rejected_truncated_pct:.2f}%)\n")
        else:
            f.write(f"  • Términos Descartados (Alucinación)    : {lex.n_terms_discarded_mean:.2f} (Total: {lex.total_discarded:,})\n")
        f.write(f"  • Tasa de Supervivencia en Consulta  : {lex.survival_rate_pct:.2f}%\n")
        f.write(f"  • Tasa de Rechazo / Alucinación      : {lex.hallucination_rate_pct:.2f}%\n")
        f.write(f"  • Longitud Promedio Query Expandida  : {lex.q_terms_expanded_mean:.2f} términos\n")
        f.write("=" * 70 + "\n")
    fig_paths["summary_txt"] = txt_path

    return metrics, fig_paths

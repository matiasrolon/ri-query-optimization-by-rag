# -*- coding: utf-8 -*-
"""
Compatibility alias for ``analysis.plots_single_run``.
"""

from analysis.plots_single_run import (
    PALETTE,
    plot_gain_loss_per_query,
    plot_gain_loss_donut,
    plot_time_breakdown,
    plot_lexicon_filtering_donut,
    plot_terms_distribution,
    plot_llm_attempts,
    generate_all_plots,
)

__all__ = [
    "PALETTE",
    "plot_gain_loss_per_query",
    "plot_gain_loss_donut",
    "plot_time_breakdown",
    "plot_lexicon_filtering_donut",
    "plot_terms_distribution",
    "plot_llm_attempts",
    "generate_all_plots",
]

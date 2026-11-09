# -*- coding: utf-8 -*-
"""
Analysis module for query expansion benchmarks (RAG vs PRF).

Provides:
- Single-run analysis (``analyzer_single_run``, ``plots_single_run``, ``cli_single_run``)
- General multi-run summary analysis (``analyzer_summary``, ``plots_summary``)
"""

from analysis.analyzer_single_run import BenchmarkAnalyzer, AnalysisMetrics
from analysis.plots_single_run import generate_all_plots
from analysis.analyzer_summary import SummaryBenchmarkAnalyzer, SummaryAnalysisMetrics
from analysis.plots_summary import generate_summary_plots

__all__ = [
    "BenchmarkAnalyzer",
    "AnalysisMetrics",
    "generate_all_plots",
    "SummaryBenchmarkAnalyzer",
    "SummaryAnalysisMetrics",
    "generate_summary_plots",
]

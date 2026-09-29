# -*- coding: utf-8 -*-
"""
Analysis module for query expansion benchmarks (RAG vs PRF).

Provides functions and CLI tools to analyze benchmark result files,
computing performance gains/losses per query, execution time breakdowns,
and LLM candidate term hallucination / lexicon survival rates.
"""

from analysis.analyzer import BenchmarkAnalyzer, AnalysisMetrics
from analysis.plots import generate_all_plots

__all__ = [
    "BenchmarkAnalyzer",
    "AnalysisMetrics",
    "generate_all_plots",
]

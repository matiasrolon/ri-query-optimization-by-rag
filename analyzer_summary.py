#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point for the general multi-run summary benchmark analysis.

Usage:
    python analyzer_summary.py --file data/benchmark_summary.csv
    python analyzer_summary.py --file data/benchmark_summary.csv --queries all
    python analyzer_summary.py --file data/benchmark_summary.csv --queries expanded
"""

import sys
from analysis.analyzer_summary import (
    SummaryBenchmarkAnalyzer,
    SummaryAnalysisMetrics,
    SummaryRunRecord,
    main,
)
from analysis.plots_summary import generate_summary_plots

__all__ = [
    "SummaryBenchmarkAnalyzer",
    "SummaryAnalysisMetrics",
    "SummaryRunRecord",
    "generate_summary_plots",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())


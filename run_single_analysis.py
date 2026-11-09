#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience script to execute the single-run benchmark analysis module.

Usage:
    python run_single_analysis.py --file /path/to/benchmark_results_*.csv
"""

import sys
from analysis.cli_single_run import main

if __name__ == "__main__":
    sys.exit(main())

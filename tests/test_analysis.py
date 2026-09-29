# -*- coding: utf-8 -*-
"""
Tests for the analysis module.
"""

import os
import sys
import shutil
import tempfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.analyzer import BenchmarkAnalyzer
from analysis.plots import generate_all_plots


def test_analyzer_with_synthetic_data():
    """Verify that analyzer calculations are mathematically consistent."""
    # Create synthetic benchmark DataFrame
    data = {
        "queryid": ["1", "2", "3", "4", "5", "1", "2", "3", "4", "5"],
        "method": ["rag", "rag", "rag", "rag", "rag", "prf", "prf", "prf", "prf", "prf"],
        "q_terms_original": [4, 5, 3, 4, 6, 4, 5, 3, 4, 6],
        "q_terms_expanded": [6, 7, 4, 5, 8, 6, 7, 4, 5, 8],
        "n_terms_proposed": [3, 4, 2, 2, 5, 0, 0, 0, 0, 0],
        "n_terms_kept": [2, 2, 1, 1, 2, 0, 0, 0, 0, 0],
        "time_seconds": [5.1, 5.2, 4.9, 5.0, 5.3, 0.08, 0.07, 0.09, 0.08, 0.08],
        "time_first_pass": [0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03],
        "time_text_fetch": [0.001, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
        "time_llm": [5.0, 5.1, 4.8, 4.9, 5.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        "time_second_pass": [0.069, 0.069, 0.079, 0.069, 0.069, 0.05, 0.04, 0.07, 0.05, 0.05],
        "mrr": [1.0, 0.5, 0.25, 0.0, 0.33, 0.5, 0.5, 0.5, 0.0, 0.1],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_benchmark.csv")
        df.to_csv(csv_path, index=False)

        analyzer = BenchmarkAnalyzer(csv_path)
        metrics = analyzer.analyze()

        # Check basic counts
        assert metrics.common_query_count == 5
        assert metrics.rag_query_count == 5
        assert metrics.prf_query_count == 5

        # Check gain/loss consistency
        gl = metrics.gain_loss
        assert gl.total_queries == 5
        assert gl.improved_count + gl.neutral_count + gl.degraded_count == 5
        assert gl.net_wins == gl.improved_count - gl.degraded_count
        assert gl.improved_count == 2  # q1: 1.0 - 0.5 = +0.5, q5: 0.33 - 0.1 = +0.23
        assert gl.neutral_count == 2   # q2: 0.5 - 0.5 = 0.0, q4: 0.0 - 0.0 = 0.0
        assert gl.degraded_count == 1  # q3: 0.25 - 0.5 = -0.25

        # Check LLM inference time
        expected_infer_mean = (5.0 + 5.1 + 4.8 + 4.9 + 5.2) / 5
        assert abs(metrics.mean_inference_time_seconds - expected_infer_mean) < 1e-4

        # Check lexicon filtering
        lex = metrics.lexicon
        assert lex.total_proposed == 16
        assert lex.total_kept == 8
        assert lex.total_discarded == 8
        assert lex.survival_rate_pct == 50.0
        assert lex.hallucination_rate_pct == 50.0

        # Check plot generation
        fig_dir = os.path.join(tmpdir, "figures")
        m, paths = generate_all_plots(analyzer, output_dir=fig_dir)
        assert os.path.isfile(paths["gain_loss"])
        assert os.path.isfile(paths["time_breakdown"])
        assert os.path.isfile(paths["lexicon_filtering"])
        assert os.path.isfile(paths["summary_json"])
        assert os.path.isfile(paths["summary_txt"])


if __name__ == "__main__":
    test_analyzer_with_synthetic_data()
    print("All tests passed!")

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
        assert os.path.isfile(paths["gain_loss_donut"])
        assert os.path.isfile(paths["time_breakdown"])
        assert os.path.isfile(paths["lexicon_filtering"])
        assert os.path.isfile(paths["terms_distribution"])
        assert os.path.isfile(paths["summary_json"])
        assert os.path.isfile(paths["summary_txt"])


def test_analyzer_with_breakdown_and_unexpanded_queries():
    """Verify analyzer handling of query strings, rejection breakdown, and exclusion of unexpanded queries from MRR."""
    data = {
        "queryid": ["1", "2", "3", "4", "1", "2", "3", "4"],
        "method": ["rag", "rag", "rag", "rag", "prf", "prf", "prf", "prf"],
        "query_original": [
            "treating headaches", "prime rate canada", "solar panels", "weather today",
            "treating headaches", "prime rate canada", "solar panels", "weather today"
        ],
        "query_expanded": [
            "treating headaches migraine ibuprofen",
            "prime rate canada bank interest",
            "solar panels photovoltaic",
            "weather today",  # unexpanded query (stayed as original)
            "treating headaches pain^1.2",
            "prime rate canada rate^1.1",
            "solar panels cell^1.3",
            "weather today forecast^1.5"
        ],
        "q_terms_original": [2, 3, 2, 2, 2, 3, 2, 2],
        "q_terms_expanded": [4, 5, 3, 2, 3, 4, 3, 3],
        "n_terms_proposed": [5, 5, 4, 6, 0, 0, 0, 0],
        "n_terms_kept": [2, 2, 1, 0, 0, 0, 0, 0],  # query 4 has 0 kept terms
        "n_rejected_stopword": [1, 1, 1, 2, 0, 0, 0, 0],
        "n_rejected_duplicate": [1, 1, 1, 2, 0, 0, 0, 0],
        "n_rejected_lexicon": [0, 1, 0, 2, 0, 0, 0, 0],
        "n_rejected_truncated": [1, 0, 1, 0, 0, 0, 0, 0],
        "time_seconds": [5.1, 5.2, 4.9, 12.0, 0.08, 0.07, 0.09, 0.08],
        "time_first_pass": [0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.02, 0.03],
        "time_text_fetch": [0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0],
        "time_llm": [5.0, 5.1, 4.8, 11.9, 0.0, 0.0, 0.0, 0.0],
        "time_second_pass": [0.069, 0.069, 0.079, 0.069, 0.05, 0.04, 0.07, 0.05],
        # Query 4 in RAG is unexpanded, so its MRR is empty/NaN (not evaluated)
        "mrr": [1.0, 0.5, 0.25, None, 0.5, 0.5, 0.5, 0.2],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_benchmark_breakdown.csv")
        df.to_csv(csv_path, index=False)

        analyzer = BenchmarkAnalyzer(csv_path)
        metrics = analyzer.analyze()

        # Unexpanded query 4 must be excluded from common_query_count and MRR
        assert metrics.common_query_count == 3
        # Expected RAG MRR: (1.0 + 0.5 + 0.25) / 3 = 0.5833, NOT diluted by query 4
        expected_rag_mrr = (1.0 + 0.5 + 0.25) / 3
        assert abs(metrics.mrr_rag_mean - expected_rag_mrr) < 1e-4

        # Gain loss: only 3 queries evaluated
        gl = metrics.gain_loss
        assert gl.total_queries == 3
        assert gl.improved_count == 1   # q1: 1.0 - 0.5 = +0.5
        assert gl.neutral_count == 1    # q2: 0.5 - 0.5 = 0.0
        assert gl.degraded_count == 1   # q3: 0.25 - 0.5 = -0.25

        # Check breakdown
        lex = metrics.lexicon
        assert lex.has_breakdown is True
        assert lex.total_rejected_stopword == 5
        assert lex.total_rejected_duplicate == 5
        assert lex.total_rejected_lexicon == 3
        assert lex.total_rejected_truncated == 2
        assert lex.total_kept == 5
        assert lex.total_proposed == 20

        # Check plot generation with breakdown
        fig_dir = os.path.join(tmpdir, "figures")
        m, paths = generate_all_plots(analyzer, output_dir=fig_dir)
        assert os.path.isfile(paths["gain_loss"])
        assert os.path.isfile(paths["lexicon_filtering"])
        assert os.path.isfile(paths["summary_txt"])


if __name__ == "__main__":
    test_analyzer_with_synthetic_data()
    test_analyzer_with_breakdown_and_unexpanded_queries()
    print("All tests passed!")

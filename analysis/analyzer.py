# -*- coding: utf-8 -*-
"""
Analyzer module for benchmark CSV results.

Processes per-query retrieval metrics and timing breakdowns for PRF vs RAG (LLM),
extracting gain/loss profiles, time distributions, and lexicon filtering metrics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any
import pandas as pd
import numpy as np


@dataclass
class GainLossMetrics:
    """Summary of per-query reciprocal rank gains and losses."""
    total_queries: int
    improved_count: int
    improved_pct: float
    degraded_count: int
    degraded_pct: float
    neutral_count: int
    neutral_pct: float
    mean_delta: float
    median_delta: float
    std_delta: float
    max_gain: float
    max_loss: float
    net_wins: int
    positive_area: float
    negative_area: float
    net_area: float


@dataclass
class TimingBreakdown:
    """Mean and breakdown of execution times for a retrieval method."""
    method: str
    first_pass_mean: float
    first_pass_std: float
    text_fetch_mean: float
    text_fetch_std: float
    llm_inference_mean: float
    llm_inference_std: float
    second_pass_mean: float
    second_pass_std: float
    total_time_mean: float
    total_time_std: float
    # Relative percentages
    first_pass_pct: float
    text_fetch_pct: float
    llm_inference_pct: float
    second_pass_pct: float


@dataclass
class LexiconFilterMetrics:
    """Statistics on LLM candidate terms proposed vs kept (lexicon filter)."""
    has_term_data: bool
    q_terms_original_mean: float
    q_terms_expanded_mean: float
    n_terms_proposed_mean: float
    n_terms_kept_mean: float
    n_terms_discarded_mean: float
    total_proposed: int
    total_kept: int
    total_discarded: int
    survival_rate_pct: float
    hallucination_rate_pct: float
    # Breakdown by rejection reason
    has_breakdown: bool = False
    total_rejected_stopword: int = 0
    total_rejected_duplicate: int = 0
    total_rejected_lexicon: int = 0
    total_rejected_truncated: int = 0
    rejected_stopword_pct: float = 0.0
    rejected_duplicate_pct: float = 0.0
    rejected_lexicon_pct: float = 0.0
    rejected_truncated_pct: float = 0.0


@dataclass
class AnalysisMetrics:
    """Aggregated analysis results from a benchmark CSV file."""
    source_file: str
    file_stem: str
    total_rows: int
    rag_query_count: int
    prf_query_count: int
    common_query_count: int
    
    # Core highlights
    mean_inference_time_seconds: float
    mrr_rag_mean: float
    mrr_prf_mean: float
    mrr_diff: float
    mrr_relative_gain_pct: float
    
    # Sub-metrics
    gain_loss: GainLossMetrics
    timings_rag: TimingBreakdown
    timings_prf: TimingBreakdown
    lexicon: LexiconFilterMetrics

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a serializable dictionary."""
        return asdict(self)


class BenchmarkAnalyzer:
    """
    Parses and analyzes a benchmark results CSV file.
    """

    def __init__(self, file_path: str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"El archivo especificado no existe: {file_path}")
        self.file_path = os.path.abspath(file_path)
        self.file_stem = os.path.splitext(os.path.basename(file_path))[0]
        self._df: pd.DataFrame | None = None
        self._rag_df: pd.DataFrame | None = None
        self._prf_df: pd.DataFrame | None = None
        self._merged_df: pd.DataFrame | None = None

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load CSV and normalize column and method names."""
        df = pd.read_csv(self.file_path)

        # Standardize column names (lowercase and strip)
        df.columns = [c.lower().strip() for c in df.columns]

        # Handle alias columns for query id
        if "queryid" not in df.columns and "qid" in df.columns:
            df["queryid"] = df["qid"]

        # Handle alias columns for proposed and kept terms
        if "n_terms_proposed" not in df.columns:
            for alt in ["q_terms_proposed", "terms_proposed", "n_proposed"]:
                if alt in df.columns:
                    df["n_terms_proposed"] = df[alt]
                    break

        if "n_terms_kept" not in df.columns:
            for alt in ["q_terms_kept", "terms_kept", "n_kept"]:
                if alt in df.columns:
                    df["n_terms_kept"] = df[alt]
                    break

        # Standardize method string
        if "method" in df.columns:
            df["method_norm"] = df["method"].astype(str).str.lower().str.strip()
        else:
            raise ValueError("El archivo CSV no contiene la columna requerida 'method'.")

        # Ensure mrr is numeric (empty strings or NaNs become np.nan)
        if "mrr" in df.columns:
            df["mrr"] = pd.to_numeric(df["mrr"], errors="coerce")

        self._df = df

        # Filter RAG and PRF
        self._rag_df = df[df["method_norm"] == "rag"].copy()
        self._prf_df = df[df["method_norm"].isin(["prf", "prfc", "prf_classic"])].copy()

        if self._rag_df.empty:
            raise ValueError("No se encontraron registros para el método 'rag' en el archivo.")
        if self._prf_df.empty:
            raise ValueError("No se encontraron registros para el método 'prf'/'prfc' en el archivo.")

        # Ensure queryid is string
        self._rag_df["queryid"] = self._rag_df["queryid"].astype(str)
        self._prf_df["queryid"] = self._prf_df["queryid"].astype(str)

        # In RAG, queries where n_terms_kept == 0 stayed as the original query
        # and should be excluded from MRR@10 and gain/loss evaluation
        if "n_terms_kept" in self._rag_df.columns:
            self._rag_df.loc[self._rag_df["n_terms_kept"] == 0, "mrr"] = np.nan

        # Merge on queryid to compare query by query
        merged = pd.merge(
            self._rag_df,
            self._prf_df,
            on="queryid",
            suffixes=("_rag", "_prf")
        )
        merged["delta_rr"] = merged["mrr_rag"] - merged["mrr_prf"]
        self._merged_df = merged

        return df

    def compute_gain_loss_metrics(self) -> GainLossMetrics:
        """Compute query-by-query Reciprocal Rank gain/loss statistics."""
        if self._merged_df is None:
            self.load_and_preprocess()

        merged = self._merged_df
        # Drop rows where delta_rr is NaN (queries excluded from MRR)
        valid_merged = merged.dropna(subset=["delta_rr"])
        delta = valid_merged["delta_rr"]
        total = len(delta)
        if total == 0:
            raise ValueError("No hay consultas coincidentes entre RAG y PRF con delta_rr válido.")

        eps = 1e-7
        improved = delta > eps
        degraded = delta < -eps
        neutral = delta.abs() <= eps

        imp_count = int(improved.sum())
        deg_count = int(degraded.sum())
        neu_count = int(neutral.sum())

        pos_area = float(delta[improved].sum())
        neg_area = float(delta[degraded].abs().sum())

        return GainLossMetrics(
            total_queries=total,
            improved_count=imp_count,
            improved_pct=round((imp_count / total) * 100, 2),
            degraded_count=deg_count,
            degraded_pct=round((deg_count / total) * 100, 2),
            neutral_count=neu_count,
            neutral_pct=round((neu_count / total) * 100, 2),
            mean_delta=float(delta.mean()),
            median_delta=float(delta.median()),
            std_delta=float(delta.std()),
            max_gain=float(delta.max()),
            max_loss=float(delta.min()),
            net_wins=imp_count - deg_count,
            positive_area=round(pos_area, 4),
            negative_area=round(neg_area, 4),
            net_area=round(pos_area - neg_area, 4),
        )

    def _compute_timing_breakdown(self, df_sub: pd.DataFrame, method_name: str) -> TimingBreakdown:
        """Compute execution time component statistics."""
        t_first = df_sub.get("time_first_pass", pd.Series(0.0, index=df_sub.index))
        t_fetch = df_sub.get("time_text_fetch", pd.Series(0.0, index=df_sub.index))
        t_llm = df_sub.get("time_llm", pd.Series(0.0, index=df_sub.index))
        t_second = df_sub.get("time_second_pass", pd.Series(0.0, index=df_sub.index))
        t_total = df_sub.get("time_seconds", t_first + t_fetch + t_llm + t_second)

        total_mean = float(t_total.mean()) if not t_total.empty else 0.0
        first_mean = float(t_first.mean()) if not t_first.empty else 0.0
        fetch_mean = float(t_fetch.mean()) if not t_fetch.empty else 0.0
        llm_mean = float(t_llm.mean()) if not t_llm.empty else 0.0
        second_mean = float(t_second.mean()) if not t_second.empty else 0.0

        denom = total_mean if total_mean > 0 else 1.0

        return TimingBreakdown(
            method=method_name,
            first_pass_mean=round(first_mean, 5),
            first_pass_std=round(float(t_first.std()), 5) if len(t_first) > 1 else 0.0,
            text_fetch_mean=round(fetch_mean, 5),
            text_fetch_std=round(float(t_fetch.std()), 5) if len(t_fetch) > 1 else 0.0,
            llm_inference_mean=round(llm_mean, 5),
            llm_inference_std=round(float(t_llm.std()), 5) if len(t_llm) > 1 else 0.0,
            second_pass_mean=round(second_mean, 5),
            second_pass_std=round(float(t_second.std()), 5) if len(t_second) > 1 else 0.0,
            total_time_mean=round(total_mean, 5),
            total_time_std=round(float(t_total.std()), 5) if len(t_total) > 1 else 0.0,
            first_pass_pct=round((first_mean / denom) * 100, 2),
            text_fetch_pct=round((fetch_mean / denom) * 100, 2),
            llm_inference_pct=round((llm_mean / denom) * 100, 2),
            second_pass_pct=round((second_mean / denom) * 100, 2),
        )

    def compute_lexicon_filter_metrics(self) -> LexiconFilterMetrics:
        """Compute term filtering, survival, and hallucination metrics for RAG."""
        if self._rag_df is None:
            self.load_and_preprocess()

        rag = self._rag_df
        has_data = ("n_terms_proposed" in rag.columns and "n_terms_kept" in rag.columns)

        if not has_data:
            return LexiconFilterMetrics(
                has_term_data=False,
                q_terms_original_mean=float(rag.get("q_terms_original", pd.Series(0.0)).mean()),
                q_terms_expanded_mean=float(rag.get("q_terms_expanded", pd.Series(0.0)).mean()),
                n_terms_proposed_mean=0.0,
                n_terms_kept_mean=0.0,
                n_terms_discarded_mean=0.0,
                total_proposed=0,
                total_kept=0,
                total_discarded=0,
                survival_rate_pct=0.0,
                hallucination_rate_pct=0.0,
            )

        orig_mean = float(rag.get("q_terms_original", pd.Series(0.0)).mean())
        exp_mean = float(rag.get("q_terms_expanded", pd.Series(0.0)).mean())

        prop_sum = int(rag["n_terms_proposed"].sum())
        kept_sum = int(rag["n_terms_kept"].sum())
        disc_sum = max(0, prop_sum - kept_sum)

        prop_mean = float(rag["n_terms_proposed"].mean())
        kept_mean = float(rag["n_terms_kept"].mean())
        disc_mean = max(0.0, prop_mean - kept_mean)

        survival_rate = (kept_sum / prop_sum * 100.0) if prop_sum > 0 else 0.0
        hallucination_rate = (disc_sum / prop_sum * 100.0) if prop_sum > 0 else 0.0

        has_breakdown = any(
            c in rag.columns
            for c in ["n_rejected_stopword", "n_rejected_duplicate", "n_rejected_lexicon", "n_rejected_truncated"]
        )
        tot_stop = int(rag["n_rejected_stopword"].sum()) if "n_rejected_stopword" in rag.columns else 0
        tot_dup = int(rag["n_rejected_duplicate"].sum()) if "n_rejected_duplicate" in rag.columns else 0
        tot_lex = int(rag["n_rejected_lexicon"].sum()) if "n_rejected_lexicon" in rag.columns else 0
        tot_trunc = int(rag["n_rejected_truncated"].sum()) if "n_rejected_truncated" in rag.columns else 0

        stop_pct = (tot_stop / prop_sum * 100.0) if prop_sum > 0 else 0.0
        dup_pct = (tot_dup / prop_sum * 100.0) if prop_sum > 0 else 0.0
        lex_pct = (tot_lex / prop_sum * 100.0) if prop_sum > 0 else 0.0
        trunc_pct = (tot_trunc / prop_sum * 100.0) if prop_sum > 0 else 0.0

        return LexiconFilterMetrics(
            has_term_data=True,
            q_terms_original_mean=round(orig_mean, 2),
            q_terms_expanded_mean=round(exp_mean, 2),
            n_terms_proposed_mean=round(prop_mean, 2),
            n_terms_kept_mean=round(kept_mean, 2),
            n_terms_discarded_mean=round(disc_mean, 2),
            total_proposed=prop_sum,
            total_kept=kept_sum,
            total_discarded=disc_sum,
            survival_rate_pct=round(survival_rate, 2),
            hallucination_rate_pct=round(hallucination_rate, 2),
            has_breakdown=has_breakdown,
            total_rejected_stopword=tot_stop,
            total_rejected_duplicate=tot_dup,
            total_rejected_lexicon=tot_lex,
            total_rejected_truncated=tot_trunc,
            rejected_stopword_pct=round(stop_pct, 2),
            rejected_duplicate_pct=round(dup_pct, 2),
            rejected_lexicon_pct=round(lex_pct, 2),
            rejected_truncated_pct=round(trunc_pct, 2),
        )

    def analyze(self) -> AnalysisMetrics:
        """Run complete analysis and return aggregated metrics."""
        self.load_and_preprocess()

        gain_loss = self.compute_gain_loss_metrics()
        timings_rag = self._compute_timing_breakdown(self._rag_df, "RAG (LLM)")
        timings_prf = self._compute_timing_breakdown(self._prf_df, "PRF")
        lexicon = self.compute_lexicon_filter_metrics()

        mrr_rag = float(self._rag_df["mrr"].dropna().mean())
        mrr_prf = float(self._prf_df["mrr"].dropna().mean())
        mrr_diff = mrr_rag - mrr_prf
        mrr_gain_pct = ((mrr_diff) / mrr_prf * 100.0) if mrr_prf > 0 else 0.0

        valid_common = len(self._merged_df.dropna(subset=["delta_rr"])) if self._merged_df is not None else 0

        return AnalysisMetrics(
            source_file=self.file_path,
            file_stem=self.file_stem,
            total_rows=len(self._df),
            rag_query_count=len(self._rag_df),
            prf_query_count=len(self._prf_df),
            common_query_count=valid_common,
            mean_inference_time_seconds=timings_rag.llm_inference_mean,
            mrr_rag_mean=round(mrr_rag, 4),
            mrr_prf_mean=round(mrr_prf, 4),
            mrr_diff=round(mrr_diff, 4),
            mrr_relative_gain_pct=round(mrr_gain_pct, 2),
            gain_loss=gain_loss,
            timings_rag=timings_rag,
            timings_prf=timings_prf,
            lexicon=lexicon,
        )

    @property
    def merged_df(self) -> pd.DataFrame:
        if self._merged_df is None:
            self.load_and_preprocess()
        return self._merged_df

    @property
    def rag_df(self) -> pd.DataFrame:
        if self._rag_df is None:
            self.load_and_preprocess()
        return self._rag_df

    @property
    def prf_df(self) -> pd.DataFrame:
        if self._prf_df is None:
            self.load_and_preprocess()
        return self._prf_df

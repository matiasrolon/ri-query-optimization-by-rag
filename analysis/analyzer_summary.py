# -*- coding: utf-8 -*-
"""
General summary analyzer module across multiple benchmark runs/configurations.

Processes a multi-run summary CSV table (varying FEEDBACK_DOCS and FEEDBACK_TERMS)
and generates:
  1. MRR@10 vs. fb_terms (line chart)
  2. MRR@10 vs. fb_docs (line chart)
  3. Effectiveness vs. Cost: MRR@10 vs. average query time on log scale (scatter plot)
  4. Inference Time vs. fb_docs: shows the near-linear scaling with context length

Supports selecting between:
  - "all" ("Todas las queries", default)
  - "expanded" ("Solo expandidas")

Usage:
    python -m analysis.analyzer_summary --file data/benchmark_summary.csv
    python -m analysis.analyzer_summary --file data/benchmark_summary.csv --queries expanded
    python analyzer_summary.py --file data/benchmark_summary.csv --queries all
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd

QueryScope = Literal["all", "expanded"]

DEFAULT_SUMMARY_CSV = os.path.join("data", "benchmark_summary.csv")
DEFAULT_TOTAL_QUERIES = 6980


def _parse_locale_float(val: Any) -> float:
    """
    Parse a numeric string that may use Spanish locale formatting
    (e.g. '35.045,6' -> 35045.6, '0,1842' -> 0.1842, '6,32%' -> 6.32).
    """
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return np.nan
    s = s.replace("%", "").strip()
    if "." in s and "," in s:
        # '.' is thousands separator, ',' is decimal separator
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_total_queries_from_comment(comment: Any, default: int = DEFAULT_TOTAL_QUERIES) -> int:
    """
    Extract total queries from comments like:
    'sobre 6698 queries expandidas; 282 sin cambios excluidas' -> 6980.
    """
    if not comment or pd.isna(comment):
        return default
    nums = [int(n) for n in re.findall(r"\b(\d+)\b", str(comment))]
    if len(nums) >= 2 and 6000 <= (nums[0] + nums[1]) <= 8000:
        return nums[0] + nums[1]
    if len(nums) == 1 and 6000 <= nums[0] <= 8000:
        return nums[0]
    return default


def _normalize_col_name(col: str) -> str:
    """Collapse whitespace/newlines and lowercase a column header."""
    return re.sub(r"\s+", " ", str(col).strip()).lower()


@dataclass
class SummaryRunRecord:
    """Normalized record for a single (fb_docs, fb_terms) configuration."""
    fecha: str
    llm_model: str
    fb_docs: int
    fb_terms: int
    mrr_bm25: float
    mrr_prf_expanded: float
    mrr_llm_expanded: float
    mrr_prf_all: float
    mrr_llm_all: float
    mrr_prf_selected: float
    mrr_llm_selected: float
    used_fallback_for_all: bool
    time_bm25_total: float
    time_prf_total: float
    time_llm_total: float
    time_bm25_avg: float
    time_prf_avg: float
    time_llm_avg: float
    total_queries: int
    source_file: str
    comentarios: str


@dataclass
class SummaryAnalysisMetrics:
    """Aggregated summary metrics across all parameter configurations."""
    source_file: str
    file_stem: str
    query_scope: str
    query_scope_label: str
    total_configurations: int
    fb_docs_values: list[int]
    fb_terms_values: list[int]
    mrr_bm25_baseline: float
    time_bm25_avg_baseline: float
    best_rag_config: dict[str, Any]
    best_prf_config: dict[str, Any]
    linear_fit_slope_sec_per_doc: float
    linear_fit_intercept_sec: float
    linear_fit_r2: float
    fallback_configs: list[str]
    records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SummaryBenchmarkAnalyzer:
    """
    Loads and analyzes the multi-run summary CSV across (FEEDBACK_DOCS, FEEDBACK_TERMS).
    """

    def __init__(self, file_path: str, query_scope: QueryScope = "all"):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"El archivo especificado no existe: {file_path}")
        self.file_path = os.path.abspath(file_path)
        self.file_stem = os.path.splitext(os.path.basename(file_path))[0]
        self.query_scope: QueryScope = self._normalize_scope(query_scope)
        self._df: pd.DataFrame | None = None
        self._records: list[SummaryRunRecord] = []

    @staticmethod
    def _normalize_scope(scope: str) -> QueryScope:
        s = str(scope).strip().lower()
        if s in ("all", "todas", "todas las queries", "all_queries"):
            return "all"
        if s in ("expanded", "expandidas", "solo expandidas", "expanded_only", "solo_expandidas"):
            return "expanded"
        raise ValueError(
            f"Valor inválido para el alcance de consultas: '{scope}'. "
            "Use 'all' ('todas') o 'expanded' ('expandidas')."
        )

    @property
    def query_scope_label(self) -> str:
        return "Todas las queries" if self.query_scope == "all" else "Solo expandidas"

    def load_and_preprocess(self) -> pd.DataFrame:
        """Read the summary CSV (handling group header row and multi-line column names)."""
        with open(self.file_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))

        if not rows:
            raise ValueError(f"El archivo CSV está vacío: {self.file_path}")

        # Find the row index that contains FEEDBACK_DOCS / FEEDBACK_TERMS
        header_idx = 0
        for idx, row in enumerate(rows[:5]):
            norm_cells = [_normalize_col_name(c) for c in row]
            if "feedback_docs" in norm_cells and "feedback_terms" in norm_cells:
                header_idx = idx
                break

        headers = [_normalize_col_name(c) for c in rows[header_idx]]
        data_rows = rows[header_idx + 1 :]

        records: list[SummaryRunRecord] = []
        for raw_row in data_rows:
            if not any(str(c).strip() for c in raw_row):
                continue
            row_map = {
                headers[i]: (raw_row[i].strip() if i < len(raw_row) else "")
                for i in range(len(headers))
            }

            fb_docs_val = _parse_locale_float(row_map.get("feedback_docs"))
            fb_terms_val = _parse_locale_float(row_map.get("feedback_terms"))
            if np.isnan(fb_docs_val) or np.isnan(fb_terms_val):
                continue

            fb_docs = int(round(fb_docs_val))
            fb_terms = int(round(fb_terms_val))
            comentarios = row_map.get("comentarios", "")
            total_queries = _parse_total_queries_from_comment(comentarios, DEFAULT_TOTAL_QUERIES)

            mrr_bm25 = _parse_locale_float(row_map.get("mrr bm25"))
            mrr_prf_exp = _parse_locale_float(row_map.get("mrr prf"))
            mrr_llm_exp = _parse_locale_float(row_map.get("mrr llm"))
            mrr_prf_all = _parse_locale_float(row_map.get("mrr prf todas"))
            mrr_llm_all = _parse_locale_float(row_map.get("mrr llm todas"))

            time_bm25_total = _parse_locale_float(row_map.get("time bm25"))
            time_prf_total = _parse_locale_float(row_map.get("time prf total"))
            time_llm_total = _parse_locale_float(row_map.get("time llm total"))

            time_prf_avg = _parse_locale_float(row_map.get("time prf avg"))
            if np.isnan(time_prf_avg) and not np.isnan(time_prf_total) and total_queries > 0:
                time_prf_avg = round(time_prf_total / total_queries, 4)

            time_llm_avg = _parse_locale_float(row_map.get("time llm avg"))
            if np.isnan(time_llm_avg) and not np.isnan(time_llm_total) and total_queries > 0:
                time_llm_avg = round(time_llm_total / total_queries, 4)

            time_bm25_avg = (
                round(time_bm25_total / total_queries, 4)
                if not np.isnan(time_bm25_total) and total_queries > 0
                else np.nan
            )

            used_fallback = False
            if self.query_scope == "all":
                if not np.isnan(mrr_prf_all) and not np.isnan(mrr_llm_all):
                    sel_prf = mrr_prf_all
                    sel_llm = mrr_llm_all
                else:
                    # Fallback to expanded columns if TODAS cell is blank in CSV
                    sel_prf = mrr_prf_all if not np.isnan(mrr_prf_all) else mrr_prf_exp
                    sel_llm = mrr_llm_all if not np.isnan(mrr_llm_all) else mrr_llm_exp
                    used_fallback = True
            else:
                sel_prf = mrr_prf_exp if not np.isnan(mrr_prf_exp) else mrr_prf_all
                sel_llm = mrr_llm_exp if not np.isnan(mrr_llm_exp) else mrr_llm_all

            records.append(
                SummaryRunRecord(
                    fecha=row_map.get("fecha", ""),
                    llm_model=row_map.get("llm model", ""),
                    fb_docs=fb_docs,
                    fb_terms=fb_terms,
                    mrr_bm25=mrr_bm25,
                    mrr_prf_expanded=mrr_prf_exp,
                    mrr_llm_expanded=mrr_llm_exp,
                    mrr_prf_all=mrr_prf_all,
                    mrr_llm_all=mrr_llm_all,
                    mrr_prf_selected=sel_prf,
                    mrr_llm_selected=sel_llm,
                    used_fallback_for_all=used_fallback,
                    time_bm25_total=time_bm25_total,
                    time_prf_total=time_prf_total,
                    time_llm_total=time_llm_total,
                    time_bm25_avg=time_bm25_avg,
                    time_prf_avg=time_prf_avg,
                    time_llm_avg=time_llm_avg,
                    total_queries=total_queries,
                    source_file=row_map.get("file", ""),
                    comentarios=comentarios,
                )
            )

        if not records:
            raise ValueError("No se encontraron filas válidas con FEEDBACK_DOCS y FEEDBACK_TERMS.")

        self._records = records
        df = pd.DataFrame([asdict(r) for r in records])
        df = df.sort_values(by=["fb_docs", "fb_terms"]).reset_index(drop=True)
        self._df = df
        return df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load_and_preprocess()
        return self._df

    def analyze(self) -> SummaryAnalysisMetrics:
        """Compute summary statistics, best configurations, and linear latency fit."""
        df = self.df

        # Baseline BM25 MRR (use median to be robust against single-row typos like 0.1932 vs 0.1842)
        valid_bm25 = df["mrr_bm25"].dropna()
        mrr_bm25_baseline = float(valid_bm25.median()) if not valid_bm25.empty else 0.1842

        valid_bm25_t = df["time_bm25_avg"].dropna()
        time_bm25_baseline = float(valid_bm25_t.median()) if not valid_bm25_t.empty else 0.0560

        fb_docs_values = sorted(df["fb_docs"].unique().tolist())
        fb_terms_values = sorted(df["fb_terms"].unique().tolist())

        # Best RAG config
        best_rag_idx = df["mrr_llm_selected"].idxmax()
        best_rag_row = df.loc[best_rag_idx]
        best_rag = {
            "fb_docs": int(best_rag_row["fb_docs"]),
            "fb_terms": int(best_rag_row["fb_terms"]),
            "mrr": round(float(best_rag_row["mrr_llm_selected"]), 4),
            "time_avg": round(float(best_rag_row["time_llm_avg"]), 4),
        }

        # Best PRF config
        best_prf_idx = df["mrr_prf_selected"].idxmax()
        best_prf_row = df.loc[best_prf_idx]
        best_prf = {
            "fb_docs": int(best_prf_row["fb_docs"]),
            "fb_terms": int(best_prf_row["fb_terms"]),
            "mrr": round(float(best_prf_row["mrr_prf_selected"]), 4),
            "time_avg": round(float(best_prf_row["time_prf_avg"]), 4),
        }

        # Linear fit: time_llm_avg vs fb_docs
        valid_time = df.dropna(subset=["fb_docs", "time_llm_avg"])
        if len(valid_time) >= 2:
            x = valid_time["fb_docs"].to_numpy(dtype=float)
            y = valid_time["time_llm_avg"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        else:
            slope, intercept, r2 = 0.0, 0.0, 0.0

        fallback_configs = [
            f"(fb_docs={int(r['fb_docs'])}, fb_terms={int(r['fb_terms'])})"
            for _, r in df[df["used_fallback_for_all"]].iterrows()
        ]

        return SummaryAnalysisMetrics(
            source_file=self.file_path,
            file_stem=self.file_stem,
            query_scope=self.query_scope,
            query_scope_label=self.query_scope_label,
            total_configurations=len(df),
            fb_docs_values=fb_docs_values,
            fb_terms_values=fb_terms_values,
            mrr_bm25_baseline=round(mrr_bm25_baseline, 4),
            time_bm25_avg_baseline=round(time_bm25_baseline, 4),
            best_rag_config=best_rag,
            best_prf_config=best_prf,
            linear_fit_slope_sec_per_doc=round(float(slope), 4),
            linear_fit_intercept_sec=round(float(intercept), 4),
            linear_fit_r2=round(float(r2), 4),
            fallback_configs=fallback_configs,
            records=df.to_dict(orient="records"),
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the multi-run summary analyzer."""
    from analysis.plots_summary import generate_summary_plots

    parser = argparse.ArgumentParser(
        description=(
            "Analizador general de resultados agregados (múltiples corridas: "
            "FEEDBACK_DOCS x FEEDBACK_TERMS)"
        )
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=DEFAULT_SUMMARY_CSV,
        help=f"Ruta al archivo CSV de resumen general (por defecto: {DEFAULT_SUMMARY_CSV}).",
    )
    parser.add_argument(
        "--queries", "-q",
        type=str,
        default="all",
        choices=["all", "expanded", "todas", "expandidas"],
        help=(
            "Conjunto de consultas a utilizar para calcular los gráficos de MRR@10: "
            "'all'/'todas' (Todas las queries, por defecto) o "
            "'expanded'/'expandidas' (Solo expandidas)."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directorio de salida para las figuras (por defecto: output/figures/summary_<scope>/).",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.file):
        print(f"❌ Error: El archivo especificado no existe:\n   {args.file}", file=sys.stderr)
        return 1

    analyzer = SummaryBenchmarkAnalyzer(args.file, query_scope=args.queries)
    scope_suffix = "todas" if analyzer.query_scope == "all" else "solo_expandidas"
    out_dir = args.output_dir or os.path.join("output", "figures", f"summary_{scope_suffix}")

    print("\n" + "=" * 72)
    print(" 📊 MÓDULO DE ANÁLISIS GENERAL (MULTI-CORRIDA): RAG vs PRF")
    print("=" * 72)
    print(f"  • Archivo analizado   : {args.file}")
    print(f"  • Modo de consultas   : {analyzer.query_scope_label} ({analyzer.query_scope})")
    print(f"  • Directorio salida   : {os.path.abspath(out_dir)}")
    print("=" * 72 + "\n")

    try:
        metrics, fig_paths = generate_summary_plots(analyzer, output_dir=out_dir)
    except Exception as e:
        print(f"\n❌ Error durante el análisis general: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print("─" * 72)
    print(f" 📈 RESUMEN GENERAL ({metrics.query_scope_label.upper()})")
    print("─" * 72)
    print(f"  • Configuraciones evaluadas : {metrics.total_configurations} "
          f"(fb_docs={metrics.fb_docs_values} × fb_terms={metrics.fb_terms_values})")
    print(f"  • Línea base MRR@10 BM25    : {metrics.mrr_bm25_baseline:.4f}")
    print(f"  • Mejor config RAG (LLM)    : fb_docs={metrics.best_rag_config['fb_docs']}, "
          f"fb_terms={metrics.best_rag_config['fb_terms']} → "
          f"MRR@10={metrics.best_rag_config['mrr']:.4f} "
          f"({metrics.best_rag_config['time_avg']:.4f} s/query)")
    print(f"  • Mejor config PRF          : fb_docs={metrics.best_prf_config['fb_docs']}, "
          f"fb_terms={metrics.best_prf_config['fb_terms']} → "
          f"MRR@10={metrics.best_prf_config['mrr']:.4f} "
          f"({metrics.best_prf_config['time_avg']:.4f} s/query)")
    print(f"  • Escalamiento lineal LLM   : T(fb_docs) ≈ {metrics.linear_fit_slope_sec_per_doc:.4f}·fb_docs "
          f"+ {metrics.linear_fit_intercept_sec:.4f} s  (R² = {metrics.linear_fit_r2:.4f})")
    if metrics.fallback_configs:
        print(f"  • Nota (celdas 'TODAS' vacías en CSV, usando valor 'Solo expandidas'): "
              f"{', '.join(metrics.fallback_configs)}")
    print("─" * 72)

    print("\n📁 Gráficos y reportes generados:")
    for key, path in fig_paths.items():
        print(f"  • [{key}] → {path}")

    print("\n✅ Análisis general completado con éxito.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# -*- coding: utf-8 -*-
"""
Benchmark module — compares RAG vs PRF query expansion.

Loads development queries from ``queries.dev.small.tsv`` and relevance judgements
from ``qrels.dev.small.tsv``, then runs both expansion strategies measuring:
  - Resolution time per query.
  - MRR (Mean Reciprocal Rank) per query.

Results are exported to a CSV with the following schema:
    queryid, method, q_terms_original, q_terms_expanded, time_seconds, mrr
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime

import pandas as pd
import pyterrier as pt

import config
from indexing import get_indexer
from indexing.base import BaseIndexer
from query_expansion.baseline import BM25Baseline
from query_expansion.prf import PRFExpander
from query_expansion.rag import RAGExpander


# ── Constants ─────────────────────────────────────────────────────────────────
_IRDS_DATASET = "irds:msmarco-passage/dev/small"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_queries(path: str | None = None) -> pd.DataFrame:
    """
    Load dev/small queries.

    Resolution order:
    1. Explicit *path* argument.
    2. Local file at ``config.QUERIES_FILE`` (if it exists on disk).
    3. Automatic download via ``pt.get_dataset("irds:msmarco-passage/dev/small")``.

    Returns a DataFrame with columns ``qid`` and ``query``.
    """
    path = path or config.QUERIES_FILE

    if os.path.isfile(path):
        print(f"   Queries: leyendo de archivo local → {path}")
        df = pd.read_csv(
            path, sep="\t", header=None, names=["qid", "query"], dtype={"qid": str}
        )
        return df

    # Fallback: download via ir_datasets
    print(f"   Queries: archivo local no encontrado ({path})")
    print(f"   Descargando vía ir_datasets ({_IRDS_DATASET})...")
    ds = pt.get_dataset(_IRDS_DATASET)
    df = ds.get_topics()
    df["qid"] = df["qid"].astype(str)
    return df


def load_qrels(path: str | None = None) -> dict[str, set[str]]:
    """
    Load dev/small relevance judgements.

    Resolution order:
    1. Explicit *path* argument.
    2. Local file at ``config.QRELS_FILE`` (if it exists on disk).
    3. Automatic download via ``pt.get_dataset("irds:msmarco-passage/dev/small")``.

    Returns a dict mapping ``qid`` → set of relevant ``docno`` strings.
    """
    path = path or config.QRELS_FILE

    if os.path.isfile(path):
        print(f"   Qrels:   leyendo de archivo local → {path}")
        qrels: dict[str, set[str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                qid, _, docno, rel = parts[0], parts[1], parts[2], parts[3]
                if int(rel) > 0:
                    qrels.setdefault(qid, set()).add(docno)
        return qrels

    # Fallback: download via ir_datasets
    print(f"   Qrels:   archivo local no encontrado ({path})")
    print(f"   Descargando vía ir_datasets ({_IRDS_DATASET})...")
    ds = pt.get_dataset(_IRDS_DATASET)
    qrels_df = ds.get_qrels()
    qrels: dict[str, set[str]] = {}
    for _, row in qrels_df.iterrows():
        if int(row["label"]) > 0:
            qrels.setdefault(str(row["qid"]), set()).add(str(row["docno"]))
    return qrels


# ── MRR computation ──────────────────────────────────────────────────────────

def compute_mrr(
    results: pd.DataFrame, relevant_docs: set[str], cutoff: int = 10
) -> float:
    """
    Compute the Reciprocal Rank at cutoff (MRR@k) for a single query result set.

    Parameters
    ----------
    results : pd.DataFrame
        Retrieval results with at least ``docno`` column, ordered by rank.
    relevant_docs : set[str]
        Set of docnos considered relevant for this query.
    cutoff : int
        Cutoff threshold (default 10 for the official MS MARCO benchmark standard).

    Returns
    -------
    float
        1/rank of the first relevant document within the cutoff, or 0.0 if none found.
    """
    if results.empty or not relevant_docs:
        return 0.0

    for rank, docno in enumerate(results["docno"].values[:cutoff], start=1):
        if str(docno) in relevant_docs:
            return 1.0 / rank

    return 0.0


# ── Term counting ─────────────────────────────────────────────────────────────

def count_terms(query: str) -> int:
    """Count the number of whitespace-delimited tokens in a query string."""
    return len(query.strip().split())


# ── Benchmark runners ─────────────────────────────────────────────────────────

def run_bm25_benchmark(
    queries: pd.DataFrame,
    qrels: dict[str, set[str]],
    indexer: BaseIndexer,
    cutoff: int = 10,
) -> list[dict]:
    """
    Run baseline BM25 retrieval (no query expansion) over all queries and collect metrics.

    Returns a list of dicts, one per query, ready for CSV export.
    """
    searcher = BM25Baseline(indexer=indexer)
    results_list: list[dict] = []
    total = len(queries)

    for i, (_, row) in enumerate(queries.iterrows(), start=1):
        qid = str(row["qid"])
        original_query = row["query"]
        relevant = qrels.get(qid, set())

        print(f"  [{i}/{total}] BM25 qid={qid}: \"{original_query}\"")

        t0 = time.time()
        safe_query, search_results = searcher.expand_and_search(original_query)
        elapsed = time.time() - t0

        mrr = compute_mrr(search_results, relevant, cutoff=cutoff)

        results_list.append({
            "queryid": qid,
            "method": "bm25",
            "q_terms_original": count_terms(original_query),
            "q_terms_expanded": count_terms(original_query),
            "time_seconds": round(elapsed, 4),
            "time_first_pass": round(elapsed, 4),
            "time_get_texts": 0.0,
            "time_llm": 0.0,
            "time_second_pass": 0.0,
            "mrr": round(mrr, 6),
        })

        print(f"         MRR@{cutoff}={mrr:.4f}  time={elapsed:.3f}s  "
              f"terms: {count_terms(original_query)} (sin expansión)")

    return results_list


def run_prf_benchmark(
    queries: pd.DataFrame,
    qrels: dict[str, set[str]],
    indexer: BaseIndexer,
    cutoff: int = 10,
) -> list[dict]:
    """
    Run the PRF expansion pipeline over all queries and collect metrics.

    Returns a list of dicts, one per query, ready for CSV export.
    """
    expander = PRFExpander(indexer=indexer)
    results_list: list[dict] = []
    total = len(queries)

    for i, (_, row) in enumerate(queries.iterrows(), start=1):
        qid = str(row["qid"])
        original_query = row["query"]
        relevant = qrels.get(qid, set())

        print(f"  [{i}/{total}] PRF  qid={qid}: \"{original_query}\"")

        t0 = time.time()
        expanded_query, search_results = expander.expand_and_search(original_query)
        elapsed = time.time() - t0
        timings = getattr(expander, "last_timings", {})

        mrr = compute_mrr(search_results, relevant, cutoff=cutoff)

        results_list.append({
            "queryid": qid,
            "method": "prf",
            "q_terms_original": count_terms(original_query),
            "q_terms_expanded": count_terms(expanded_query),
            "time_seconds": round(elapsed, 4),
            "time_first_pass": round(timings.get("time_first_pass", 0.0), 4),
            "time_get_texts": 0.0,
            "time_llm": 0.0,
            "time_second_pass": round(timings.get("time_second_pass", 0.0), 4),
            "mrr": round(mrr, 6),
        })

        print(f"         MRR@{cutoff}={mrr:.4f}  time={elapsed:.3f}s  "
              f"terms: {count_terms(original_query)}→{count_terms(expanded_query)}")

    return results_list


def run_rag_benchmark(
    queries: pd.DataFrame,
    qrels: dict[str, set[str]],
    indexer: BaseIndexer,
    cutoff: int = 10,
) -> list[dict]:
    """
    Run the RAG expansion pipeline over all queries and collect metrics.

    Returns a list of dicts, one per query, ready for CSV export.
    """
    expander = RAGExpander(indexer=indexer, verbose=False)
    results_list: list[dict] = []
    total = len(queries)

    for i, (_, row) in enumerate(queries.iterrows(), start=1):
        qid = str(row["qid"])
        original_query = row["query"]
        relevant = qrels.get(qid, set())

        print(f"  [{i}/{total}] RAG  qid={qid}: \"{original_query}\"")

        t0 = time.time()
        expanded_query, search_results = expander.expand_and_search(original_query)
        elapsed = time.time() - t0
        timings = getattr(expander, "last_timings", {})

        mrr = compute_mrr(search_results, relevant, cutoff=cutoff)

        results_list.append({
            "queryid": qid,
            "method": "rag",
            "q_terms_original": count_terms(original_query),
            "q_terms_expanded": count_terms(expanded_query),
            "time_seconds": round(elapsed, 4),
            "time_first_pass": round(timings.get("time_first_pass", 0.0), 4),
            "time_get_texts": round(timings.get("time_get_texts", 0.0), 4),
            "time_llm": round(timings.get("time_llm", 0.0), 4),
            "time_second_pass": round(timings.get("time_second_pass", 0.0), 4),
            "mrr": round(mrr, 6),
        })

        print(f"         MRR@{cutoff}={mrr:.4f}  time={elapsed:.3f}s  "
              f"[1st:{timings.get('time_first_pass', 0.0):.3f}s, texts:{timings.get('time_get_texts', 0.0):.3f}s, "
              f"llm:{timings.get('time_llm', 0.0):.3f}s, 2nd:{timings.get('time_second_pass', 0.0):.3f}s]  "
              f"terms: {count_terms(original_query)}→{count_terms(expanded_query)}")

    return results_list


# ── CSV export ────────────────────────────────────────────────────────────────

def export_results(
    results: list[dict],
    output_path: str | None = None,
) -> str:
    """
    Export benchmark results to a CSV file.

    Parameters
    ----------
    results : list[dict]
        Combined list of metric dicts from BM25, PRF and RAG runs.
    output_path : str | None
        Destination CSV path. Defaults to ``config.OUTPUT_DIR/benchmark_results.csv``.

    Returns
    -------
    str
        The absolute path of the written CSV file.
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            config.OUTPUT_DIR, f"benchmark_results_{timestamp}.csv"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "queryid",
        "method",
        "q_terms_original",
        "q_terms_expanded",
        "time_seconds",
        "time_first_pass",
        "time_get_texts",
        "time_llm",
        "time_second_pass",
        "mrr",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return output_path


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_benchmark(
    queries_path: str | None = None,
    qrels_path: str | None = None,
    output_path: str | None = None,
    max_queries: int | None = None,
    offset: int = 0,
    methods: list[str] | None = None,
    cutoff: int = 10,
) -> str:
    """
    Run the benchmark (BM25, PRF, RAG) over dev queries and export to CSV.

    Parameters
    ----------
    queries_path : str | None
        Path to the queries TSV file.
    qrels_path : str | None
        Path to the qrels TSV file.
    output_path : str | None
        Path for the output CSV.
    max_queries : int | None
        If set, limit the evaluation to N queries (useful for
        testing / debugging or multi-pass runs).
    offset : int
        Number of queries to skip from the beginning (default 0). Applied
        after filtering for queries with qrels. Allows multi-pass execution
        over large query sets.
    methods : list[str] | None
        List of methods to evaluate: "bm25", "prf", "rag".
        Defaults to all three: ["bm25", "prf", "rag"].
    cutoff : int
        Cutoff threshold for MRR calculation (default 10 for MRR@10).

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    if not pt.started():
        pt.init()

    valid_methods = {"bm25", "prf", "rag"}
    if methods is None:
        selected_methods = ["bm25", "prf", "rag"]
    else:
        selected_methods = [m.lower().strip() for m in methods if m.lower().strip() in valid_methods]
        if not selected_methods:
            raise ValueError(
                f"No se especificaron métodos válidos. Opciones disponibles: {sorted(list(valid_methods))}"
            )

    # Load data
    print("=" * 65)
    print("📊 Cargando queries y qrels...")
    print("=" * 65)

    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    # Filter to only queries that have qrels (to compute meaningful MRR)
    queries_with_qrels = queries[queries["qid"].isin(qrels.keys())]
    total_with_qrels = len(queries_with_qrels)
    print(f"   Queries totales    : {len(queries):,}")
    print(f"   Queries con qrels  : {total_with_qrels:,}")

    # Apply offset
    if offset > 0:
        queries_with_qrels = queries_with_qrels.iloc[offset:]
        print(f"   Offset             : {offset}")

    # Apply limit
    if max_queries is not None:
        queries_with_qrels = queries_with_qrels.head(max_queries)
        print(f"   Limitado a         : {max_queries}")

    print(f"   Queries a evaluar  : {len(queries_with_qrels):,}"
          f"  (rango {offset}–{offset + len(queries_with_qrels) - 1})")
    print(f"   Métodos a ejecutar : {', '.join(selected_methods).upper()}")
    print(f"   Métrica principal  : MRR@{cutoff}")

    print()

    # Initialise shared indexer
    indexer = get_indexer()
    if indexer.index_exists():
        indexer.load_index()
    else:
        raise RuntimeError(
            "No se encontró un índice construido. "
            "Ejecute primero el proceso de indexación."
        )

    all_results: list[dict] = []
    stats_summary: dict[str, dict[str, float]] = {}

    # 1. Run BM25 baseline benchmark
    if "bm25" in selected_methods:
        print("=" * 65)
        print("⚡ Ejecutando benchmark BM25 (baseline sin expansión)...")
        print("=" * 65)
        t0 = time.time()
        bm25_results = run_bm25_benchmark(queries_with_qrels, qrels, indexer, cutoff=cutoff)
        bm25_time = time.time() - t0
        bm25_mrr_avg = (
            sum(r["mrr"] for r in bm25_results) / len(bm25_results)
            if bm25_results
            else 0.0
        )
        stats_summary["bm25"] = {"mrr": bm25_mrr_avg, "time": bm25_time, "count": len(bm25_results)}
        all_results.extend(bm25_results)
        print(f"\n   BM25 completado: {len(bm25_results)} queries en {bm25_time:.1f}s")
        print(f"   MRR@{cutoff} promedio BM25: {bm25_mrr_avg:.4f}")
        print()

    # 2. Run PRF benchmark
    if "prf" in selected_methods:
        print("=" * 65)
        print("🔄 Ejecutando benchmark PRF...")
        print("=" * 65)
        t0 = time.time()
        prf_results = run_prf_benchmark(queries_with_qrels, qrels, indexer, cutoff=cutoff)
        prf_time = time.time() - t0
        prf_mrr_avg = (
            sum(r["mrr"] for r in prf_results) / len(prf_results)
            if prf_results
            else 0.0
        )
        stats_summary["prf"] = {"mrr": prf_mrr_avg, "time": prf_time, "count": len(prf_results)}
        all_results.extend(prf_results)
        print(f"\n   PRF completado: {len(prf_results)} queries en {prf_time:.1f}s")
        print(f"   MRR@{cutoff} promedio PRF: {prf_mrr_avg:.4f}")
        print()

    # 3. Run RAG benchmark
    if "rag" in selected_methods:
        print("=" * 65)
        print("🤖 Ejecutando benchmark RAG...")
        print("=" * 65)
        t0 = time.time()
        rag_results = run_rag_benchmark(queries_with_qrels, qrels, indexer, cutoff=cutoff)
        rag_time = time.time() - t0
        rag_mrr_avg = (
            sum(r["mrr"] for r in rag_results) / len(rag_results)
            if rag_results
            else 0.0
        )
        stats_summary["rag"] = {"mrr": rag_mrr_avg, "time": rag_time, "count": len(rag_results)}
        all_results.extend(rag_results)
        print(f"\n   RAG completado: {len(rag_results)} queries en {rag_time:.1f}s")
        print(f"   MRR@{cutoff} promedio RAG: {rag_mrr_avg:.4f}")
        print()

    # Export to CSV
    csv_path = export_results(all_results, output_path)

    print("=" * 65)
    print("✅ Benchmark finalizado")
    print("=" * 65)
    print(f"   Queries evaluadas   : {len(queries_with_qrels)}")
    print(f"   Métrica de ranking  : MRR@{cutoff}")
    for method_key in selected_methods:
        s = stats_summary[method_key]
        name = {"bm25": "BM25 (baseline)", "prf": "PRF (DFR Bo1)", "rag": "RAG (LLM)"}[method_key]
        print(f"   MRR@{cutoff} {name:<18}: {s['mrr']:.4f}  (tiempo total: {s['time']:.1f}s)")
    print(f"   CSV exportado a     : {csv_path}")
    print()

    return csv_path

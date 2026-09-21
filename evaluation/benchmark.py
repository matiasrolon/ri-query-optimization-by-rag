# -*- coding: utf-8 -*-
"""
Benchmark module — compares BM25 vs PRF vs RAG query expansion.

Loads development queries from ``queries.dev.small.tsv`` and relevance judgements
from ``qrels.dev.small.tsv``, then runs the three evaluation arms measuring:
  - Resolution time per query (with breakdown: 1st pass, text fetch, LLM, 2nd pass).
  - MRR@10 (Mean Reciprocal Rank @ 10) per query.

Results are exported to a CSV with the following schema:
    queryid, method, q_terms_original, q_terms_expanded, time_seconds,
    time_first_pass, time_text_fetch, time_llm, time_second_pass, mrr
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime

import pandas as pd
import pyterrier as pt

import config
from indexing import get_indexer
from indexing.base import BaseIndexer
from query_expansion.prf import PRFExpander
from query_expansion.rag import RAGExpander
from query_expansion.utils import sanitize_query_str

logger = logging.getLogger(__name__)


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


# ── MRR@10 computation ────────────────────────────────────────────────────────

def compute_mrr(
    results: pd.DataFrame, relevant_docs: set[str], k: int = 10
) -> float:
    """
    Compute MRR@k for a single query result set.

    Parameters
    ----------
    results : pd.DataFrame
        Retrieval results with at least ``docno`` column, ordered by rank.
    relevant_docs : set[str]
        Set of docnos considered relevant for this query.
    k : int, default 10
        Cutoff rank for MRR computation (default 10 for MRR@10).

    Returns
    -------
    float
        1/rank of the first relevant document if rank <= k, or 0.0 if none found.
    """
    if results.empty or not relevant_docs:
        return 0.0

    for rank, docno in enumerate(results["docno"].values[:k], start=1):
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
) -> list[dict]:
    """
    Run the standard BM25 retrieval baseline (no expansion) over all queries.

    Returns a list of dicts, one per query, ready for CSV export.
    """
    retriever = indexer.bm25_retriever()
    results_list: list[dict] = []
    total = len(queries)

    for i, (_, row) in enumerate(queries.iterrows(), start=1):
        qid = str(row["qid"])
        original_query = row["query"]
        relevant = qrels.get(qid, set())

        print(f"  [{i}/{total}] BM25 qid={qid}: \"{original_query}\"")

        safe_query = sanitize_query_str(original_query)

        t0 = time.time()
        search_results = retriever.search(safe_query)
        elapsed_first = time.time() - t0

        mrr = compute_mrr(search_results, relevant, k=10)

        results_list.append({
            "queryid": qid,
            "method": "bm25",
            "q_terms_original": count_terms(original_query),
            "q_terms_expanded": count_terms(safe_query),
            "n_terms_proposed": 0,
            "n_terms_kept": 0,
            "time_seconds": round(elapsed_first, 4),
            "time_first_pass": round(elapsed_first, 4),
            "time_text_fetch": 0.0,
            "time_llm": 0.0,
            "time_second_pass": 0.0,
            "mrr": round(mrr, 6),
        })

        logger.info(
            "[BM25] qid=%s | original='%s' | MRR@10=%.4f | time=%.3fs | terms: %d",
            qid, original_query, mrr, elapsed_first, count_terms(original_query),
        )

        print(f"         MRR@10={mrr:.4f}  time={elapsed_first:.3f}s  "
              f"terms: {count_terms(original_query)}")

    return results_list


def run_prf_benchmark(
    queries: pd.DataFrame,
    qrels: dict[str, set[str]],
    indexer: BaseIndexer,
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
        expanded_query, search_results, timings = expander.expand_and_search(original_query)
        elapsed = time.time() - t0

        mrr = compute_mrr(search_results, relevant, k=10)

        n_proposed = timings.get("n_terms_proposed", 0)
        n_kept = timings.get("n_terms_kept", 0)

        results_list.append({
            "queryid": qid,
            "method": "prf",
            "q_terms_original": count_terms(original_query),
            "q_terms_expanded": count_terms(expanded_query),
            "n_terms_proposed": n_proposed,
            "n_terms_kept": n_kept,
            "time_seconds": round(elapsed, 4),
            "time_first_pass": timings.get("time_first_pass", 0.0),
            "time_text_fetch": timings.get("time_text_fetch", 0.0),
            "time_llm": timings.get("time_llm", 0.0),
            "time_second_pass": timings.get("time_second_pass", 0.0),
            "mrr": round(mrr, 6),
        })

        logger.info(
            "[PRF] qid=%s | original='%s' | expanded='%s' | "
            "MRR@10=%.4f | time=%.3fs | terms: %d→%d (proposed: %d, kept: %d)",
            qid, original_query, expanded_query,
            mrr, elapsed,
            count_terms(original_query), count_terms(expanded_query),
            n_proposed, n_kept,
        )

        print(f"         📝 Query Expandida: \"{expanded_query}\"")
        print(f"         MRR@10={mrr:.4f}  time={elapsed:.3f}s  "
              f"terms: {count_terms(original_query)}→{count_terms(expanded_query)} "
              f"(propuestos: {n_proposed}, aceptados: {n_kept})")

    return results_list


def run_rag_benchmark(
    queries: pd.DataFrame,
    qrels: dict[str, set[str]],
    indexer: BaseIndexer,
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
        expanded_query, search_results, timings = expander.expand_and_search(original_query)
        elapsed = time.time() - t0

        mrr = compute_mrr(search_results, relevant, k=10)

        n_proposed = timings.get("n_terms_proposed", 0)
        n_kept = timings.get("n_terms_kept", 0)

        results_list.append({
            "queryid": qid,
            "method": "rag",
            "q_terms_original": count_terms(original_query),
            "q_terms_expanded": count_terms(expanded_query),
            "n_terms_proposed": n_proposed,
            "n_terms_kept": n_kept,
            "time_seconds": round(elapsed, 4),
            "time_first_pass": timings.get("time_first_pass", 0.0),
            "time_text_fetch": timings.get("time_text_fetch", 0.0),
            "time_llm": timings.get("time_llm", 0.0),
            "time_second_pass": timings.get("time_second_pass", 0.0),
            "mrr": round(mrr, 6),
        })

        logger.info(
            "[RAG] qid=%s | original='%s' | expanded='%s' | "
            "MRR@10=%.4f | time=%.3fs | terms: %d→%d (proposed: %d, kept: %d)",
            qid, original_query, expanded_query,
            mrr, elapsed,
            count_terms(original_query), count_terms(expanded_query),
            n_proposed, n_kept,
        )

        print(f"         📝 Query Expandida: \"{expanded_query}\"")
        print(f"         MRR@10={mrr:.4f}  time={elapsed:.3f}s  "
              f"terms: {count_terms(original_query)}→{count_terms(expanded_query)} "
              f"(propuestos: {n_proposed}, aceptados: {n_kept})")

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
        Combined list of metric dicts from BM25, PRF, and RAG runs.
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
        "n_terms_proposed",
        "n_terms_kept",
        "time_seconds",
        "time_first_pass",
        "time_text_fetch",
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
) -> str:
    """
    Run the full benchmark: BM25 + PRF + RAG over dev queries, export to CSV.

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

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    if not pt.started():
        pt.init()

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

    # Run BM25 baseline benchmark
    print("=" * 65)
    print("🎯 Ejecutando benchmark BM25 Base (sin expansión)...")
    print("=" * 65)
    t0 = time.time()
    bm25_results = run_bm25_benchmark(queries_with_qrels, qrels, indexer)
    bm25_time = time.time() - t0
    bm25_mrr_avg = (
        sum(r["mrr"] for r in bm25_results) / len(bm25_results)
        if bm25_results
        else 0.0
    )
    print(f"\n   BM25 completado: {len(bm25_results)} queries en {bm25_time:.1f}s")
    print(f"   MRR@10 promedio BM25: {bm25_mrr_avg:.4f}")
    print()

    # Run PRF benchmark
    print("=" * 65)
    print("🔄 Ejecutando benchmark PRF...")
    print("=" * 65)
    t0 = time.time()
    prf_results = run_prf_benchmark(queries_with_qrels, qrels, indexer)
    prf_time = time.time() - t0
    prf_mrr_avg = (
        sum(r["mrr"] for r in prf_results) / len(prf_results)
        if prf_results
        else 0.0
    )
    print(f"\n   PRF completado: {len(prf_results)} queries en {prf_time:.1f}s")
    print(f"   MRR@10 promedio PRF: {prf_mrr_avg:.4f}")
    print()

    # Run RAG benchmark
    print("=" * 65)
    print("🤖 Ejecutando benchmark RAG...")
    print("=" * 65)
    t0 = time.time()
    rag_results = run_rag_benchmark(queries_with_qrels, qrels, indexer)
    rag_time = time.time() - t0
    rag_mrr_avg = (
        sum(r["mrr"] for r in rag_results) / len(rag_results)
        if rag_results
        else 0.0
    )
    print(f"\n   RAG completado: {len(rag_results)} queries en {rag_time:.1f}s")
    print(f"   MRR@10 promedio RAG: {rag_mrr_avg:.4f}")
    print()

    # Combine and export
    all_results = bm25_results + prf_results + rag_results
    csv_path = export_results(all_results, output_path)

    print("=" * 65)
    print("✅ Benchmark finalizado")
    print("=" * 65)
    print(f"   Queries evaluadas   : {len(queries_with_qrels)}")
    print(f"   MRR@10 promedio BM25: {bm25_mrr_avg:.4f}")
    print(f"   MRR@10 promedio PRF : {prf_mrr_avg:.4f}")
    print(f"   MRR@10 promedio RAG : {rag_mrr_avg:.4f}")
    print(f"   Tiempo total BM25   : {bm25_time:.1f}s")
    print(f"   Tiempo total PRF    : {prf_time:.1f}s")
    print(f"   Tiempo total RAG    : {rag_time:.1f}s")
    print(f"   CSV exportado a     : {csv_path}")
    print()

    return csv_path

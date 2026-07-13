# -*- coding: utf-8 -*-
"""
Benchmark module — compares RAG vs PRF query expansion.

Loads development queries from ``queries.dev.tsv`` and relevance judgements
from ``qrels.dev.tsv``, then runs both expansion strategies measuring:
  - Resolution time per query.
  - MRR (Mean Reciprocal Rank) per query.

Results are exported to a CSV with the following schema:
    queryid, metodo, cant_terminos_original, cant_terminos_expandida,
    tiempo_resolucion, mrr
"""

from __future__ import annotations

import csv
import os
import time

import pandas as pd
import pyterrier as pt

import config
from indexing import get_indexer
from indexing.base import BaseIndexer
from query_expansion.prf import PRFExpander
from query_expansion.rag import RAGExpander


# ── Data loading ──────────────────────────────────────────────────────────────

def load_queries(path: str | None = None) -> pd.DataFrame:
    """
    Load queries from a TSV file (format: ``qid\\tquery``).

    Returns a DataFrame with columns ``qid`` and ``query``.
    """
    path = path or os.path.join(config.QUERIES_DIR, "queries.dev.tsv")
    df = pd.read_csv(
        path, sep="\t", header=None, names=["qid", "query"], dtype={"qid": str}
    )
    return df


def load_qrels(path: str | None = None) -> dict[str, set[str]]:
    """
    Load relevance judgements from a TSV file.

    MS MARCO qrels format: ``qid  0  docno  relevance``

    Returns a dict mapping ``qid`` → set of relevant ``docno`` strings.
    """
    path = path or config.QRELS_FILE
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


# ── MRR computation ──────────────────────────────────────────────────────────

def compute_mrr(
    results: pd.DataFrame, relevant_docs: set[str]
) -> float:
    """
    Compute the Reciprocal Rank for a single query result set.

    Parameters
    ----------
    results : pd.DataFrame
        Retrieval results with at least ``docno`` column, ordered by rank.
    relevant_docs : set[str]
        Set of docnos considered relevant for this query.

    Returns
    -------
    float
        1/rank of the first relevant document, or 0.0 if none found.
    """
    if results.empty or not relevant_docs:
        return 0.0

    for rank, docno in enumerate(results["docno"].values, start=1):
        if str(docno) in relevant_docs:
            return 1.0 / rank

    return 0.0


# ── Term counting ─────────────────────────────────────────────────────────────

def count_terms(query: str) -> int:
    """Count the number of whitespace-delimited tokens in a query string."""
    return len(query.strip().split())


# ── Benchmark runners ─────────────────────────────────────────────────────────

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
        expanded_query = expander.expand(original_query)
        search_results = expander._second_pass.search(expanded_query)
        elapsed = time.time() - t0

        mrr = compute_mrr(search_results, relevant)

        results_list.append({
            "queryid": qid,
            "metodo": "prf",
            "cant_terminos_original": count_terms(original_query),
            "cant_terminos_expandida": count_terms(expanded_query),
            "tiempo_resolucion": round(elapsed, 4),
            "mrr": round(mrr, 6),
        })

        print(f"         MRR={mrr:.4f}  time={elapsed:.3f}s  "
              f"terms: {count_terms(original_query)}→{count_terms(expanded_query)}")

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
    expander = RAGExpander(indexer=indexer)
    results_list: list[dict] = []
    total = len(queries)

    for i, (_, row) in enumerate(queries.iterrows(), start=1):
        qid = str(row["qid"])
        original_query = row["query"]
        relevant = qrels.get(qid, set())

        print(f"  [{i}/{total}] RAG  qid={qid}: \"{original_query}\"")

        t0 = time.time()
        expanded_query = expander.expand(original_query)
        search_results = expander._second_pass.search(expanded_query)
        elapsed = time.time() - t0

        mrr = compute_mrr(search_results, relevant)

        results_list.append({
            "queryid": qid,
            "metodo": "rag",
            "cant_terminos_original": count_terms(original_query),
            "cant_terminos_expandida": count_terms(expanded_query),
            "tiempo_resolucion": round(elapsed, 4),
            "mrr": round(mrr, 6),
        })

        print(f"         MRR={mrr:.4f}  time={elapsed:.3f}s  "
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
        Combined list of metric dicts from both PRF and RAG runs.
    output_path : str | None
        Destination CSV path. Defaults to ``config.OUTPUT_DIR/benchmark_results.csv``.

    Returns
    -------
    str
        The absolute path of the written CSV file.
    """
    output_path = output_path or os.path.join(config.OUTPUT_DIR, "benchmark_results.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "queryid",
        "metodo",
        "cant_terminos_original",
        "cant_terminos_expandida",
        "tiempo_resolucion",
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
) -> str:
    """
    Run the full benchmark: PRF + RAG over dev queries, export to CSV.

    Parameters
    ----------
    queries_path : str | None
        Path to the queries TSV file.
    qrels_path : str | None
        Path to the qrels TSV file.
    output_path : str | None
        Path for the output CSV.
    max_queries : int | None
        If set, limit the evaluation to the first N queries (useful for
        testing / debugging).

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
    print(f"   Queries totales    : {len(queries):,}")
    print(f"   Queries con qrels  : {len(queries_with_qrels):,}")

    if max_queries is not None:
        queries_with_qrels = queries_with_qrels.head(max_queries)
        print(f"   Limitado a         : {max_queries}")

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
    print(f"   MRR promedio PRF: {prf_mrr_avg:.4f}")
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
    print(f"   MRR promedio RAG: {rag_mrr_avg:.4f}")
    print()

    # Combine and export
    all_results = prf_results + rag_results
    csv_path = export_results(all_results, output_path)

    print("=" * 65)
    print("✅ Benchmark finalizado")
    print("=" * 65)
    print(f"   Queries evaluadas  : {len(queries_with_qrels)}")
    print(f"   MRR promedio PRF   : {prf_mrr_avg:.4f}")
    print(f"   MRR promedio RAG   : {rag_mrr_avg:.4f}")
    print(f"   Tiempo total PRF   : {prf_time:.1f}s")
    print(f"   Tiempo total RAG   : {rag_time:.1f}s")
    print(f"   CSV exportado a    : {csv_path}")
    print()

    return csv_path

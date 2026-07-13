# -*- coding: utf-8 -*-
"""
Main entry point — MS MARCO passage indexing & query expansion benchmark.

1. Reads the indexing method from .env (INDEXING_METHOD) and ensures the
   index is available (loading from disk or building from scratch).
2. Runs both query expansion strategies (PRF and RAG) over the development
   queries, measuring per-query MRR and resolution time.
3. Exports combined results to ``output/benchmark_results.csv``.

Usage:
    python main.py                  # full benchmark (all queries with qrels)
    python main.py --max-queries 50 # limit to first N queries for testing
"""

import argparse
import os

import pyterrier as pt

import config
from indexing import get_indexer


def ensure_index() -> None:
    """Make sure the index exists (build it if necessary)."""
    indexer = get_indexer()

    if indexer.index_exists() and not config.FORCE_REINDEX:
        indexer.load_index()
    else:
        # Only load the dataset when we actually need to build
        from indexing.dataset import load_collection

        sample = load_collection()
        indexer.build_index(sample, config.THREADS)

    print("\n" + "=" * 50)
    print("Indexación completada.")


def run_evaluation(max_queries: int | None = None) -> None:
    """Run PRF + RAG benchmarks and export CSV."""
    from evaluation.benchmark import run_benchmark

    csv_path = run_benchmark(max_queries=max_queries)
    print(f"\n📄 Resultados guardados en: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MS MARCO query expansion benchmark (PRF vs RAG)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limitar la evaluación a las primeras N queries (para testing).",
    )
    args = parser.parse_args()

    # Initialise PyTerrier
    if not pt.started():
        pt.init()

    print(f"CPUs: {os.cpu_count()} | THREADS = {config.THREADS}")
    print(f"Método de indexación: {config.INDEXING_METHOD.upper()}")
    if config.FORCE_REINDEX:
        print("⚠  FORCE_REINDEX activado — se re-construirá el índice.")
    print("-" * 50)

    # Step 1: Ensure index
    ensure_index()

    # Step 2: Run benchmarks
    print()
    run_evaluation(max_queries=args.max_queries)


if __name__ == "__main__":
    main()

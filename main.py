# -*- coding: utf-8 -*-
"""
Main entry point — MS MARCO passage indexing & query expansion benchmark.

1. Reads the indexing method from .env (INDEXING_METHOD) and ensures the
   index is available (loading from disk or building from scratch).
2. Runs baseline BM25 (no expansion) alongside PRF and RAG query expansion
   over the development queries, measuring per-query MRR@10 and per-stage timings.
3. Exports combined results to ``output/benchmark_results.csv``.

Usage:
    python main.py                              # full benchmark (bm25, prf, rag)
    python main.py --methods bm25               # only BM25 baseline
    python main.py --methods bm25 prf           # BM25 + PRF
    python main.py --max-queries 50             # first 50 queries
    python main.py --offset 100 --max-queries 50
"""

import argparse
import logging
import os

import pyterrier as pt

import config
from indexing import get_indexer


# ── Logging setup ─────────────────────────────────────────────────────────────
_log_file = os.path.join(config.OUTPUT_DIR, "benchmark.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(),  # also print to console
    ],
)
# Keep console output clean: only WARNING+ on screen
logging.getLogger().handlers[1].setLevel(logging.WARNING)


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


def run_evaluation(
    max_queries: int | None = None,
    offset: int = 0,
    methods: list[str] | None = None,
    cutoff: int = 10,
) -> None:
    """Run BM25, PRF, and/or RAG benchmarks and export CSV."""
    from evaluation.benchmark import run_benchmark

    csv_path = run_benchmark(
        max_queries=max_queries,
        offset=offset,
        methods=methods,
        cutoff=cutoff,
    )
    print(f"\n📄 Resultados guardados en: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MS MARCO benchmark: BM25 baseline vs PRF vs RAG"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limitar la evaluación a N queries (para testing o ejecución por pasadas).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Saltar las primeras N queries (para ejecución por pasadas).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "prf", "rag"],
        choices=["bm25", "prf", "rag"],
        help="Métodos a evaluar (por defecto: bm25 prf rag).",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=10,
        help="Umbral de corte para el cálculo de MRR (por defecto 10 para MRR@10).",
    )
    args = parser.parse_args()

    # Initialise PyTerrier
    if not pt.started():
        pt.init()

    print(f"CPUs: {os.cpu_count()} | THREADS = {config.THREADS}")
    print(f"Método de indexación: {config.INDEXING_METHOD.upper()}")
    print(f"Métodos seleccionados: {', '.join(args.methods).upper()}")
    print(f"Métrica de evaluación: MRR@{args.cutoff}")
    if config.FORCE_REINDEX:
        print("⚠  FORCE_REINDEX activado — se re-construirá el índice.")
    print("-" * 50)

    # Step 1: Ensure index
    ensure_index()

    # Step 2: Run benchmarks
    print()
    run_evaluation(
        max_queries=args.max_queries,
        offset=args.offset,
        methods=args.methods,
        cutoff=args.cutoff,
    )


if __name__ == "__main__":
    main()

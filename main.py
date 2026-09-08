# -*- coding: utf-8 -*-
"""
Main entry point — MS MARCO passage indexing & query expansion benchmark.

1. Reads the indexing method from .env (INDEXING_METHOD) and ensures the
   index is available (loading from disk or building from scratch).
2. Runs BM25 baseline, PRF, and RAG query expansion benchmarks over the
   development queries, measuring per-query MRR@10 and time breakdown.
3. Exports combined results to ``output/benchmark_results_<timestamp>.csv``.

Usage:
    python main.py                              # full benchmark
    python main.py --max-queries 50              # first 50 queries
    python main.py --offset 100 --max-queries 50 # queries 100–149
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
# Silence noisy jnius reflection logs
logging.getLogger("kivy.jnius.reflect").setLevel(logging.WARNING)


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
) -> None:
    """Run BM25 + PRF + RAG benchmarks and export CSV."""
    from evaluation.benchmark import run_benchmark

    csv_path = run_benchmark(max_queries=max_queries, offset=offset)
    print(f"\n📄 Resultados guardados en: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MS MARCO query expansion benchmark (BM25 vs PRF vs RAG)"
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
    args = parser.parse_args()

    # Initialise PyTerrier
    if not pt.started():
        pt.init()

    print("\n" + "=" * 60)
    print(" ⚙  PARÁMETROS DE CONFIGURACIÓN DEL EXPERIMENTO (.env)")
    print("=" * 60)
    print(f"  • INDEXING_METHOD : {config.INDEXING_METHOD.upper()}")
    print(f"  • EXPANSION_MODEL : {config.EXPANSION_MODEL.upper()}")
    print(f"  • FEEDBACK_DOCS   : {config.FEEDBACK_DOCS}")
    print(f"  • FEEDBACK_TERMS  : {config.FEEDBACK_TERMS}")
    print(f"  • FEEDBACK_LAMBDA : {config.FEEDBACK_LAMBDA}")
    print(f"  • LLM_MODEL       : {config.LLM_MODEL}")
    print(f"  • LLM_BASE_URL    : {config.LLM_BASE_URL}")
    print(f"  • CPUs / THREADS  : {os.cpu_count()} / {config.THREADS}")
    if config.FORCE_REINDEX:
        print("  • ⚠  FORCE_REINDEX: activado — se re-construirá el índice.")
    print("=" * 60 + "\n")

    logging.info("=== CONFIGURACIÓN DE EXPERIMENTO ===")
    logging.info("INDEXING_METHOD=%s", config.INDEXING_METHOD)
    logging.info("EXPANSION_MODEL=%s", config.EXPANSION_MODEL)
    logging.info("FEEDBACK_DOCS=%d", config.FEEDBACK_DOCS)
    logging.info("FEEDBACK_TERMS=%d", config.FEEDBACK_TERMS)
    logging.info("FEEDBACK_LAMBDA=%.2f", config.FEEDBACK_LAMBDA)
    logging.info("LLM_MODEL=%s", config.LLM_MODEL)
    logging.info("LLM_BASE_URL=%s", config.LLM_BASE_URL)


    # Step 1: Ensure index
    ensure_index()

    # Step 2: Run benchmarks
    print()
    run_evaluation(max_queries=args.max_queries, offset=args.offset)


if __name__ == "__main__":
    main()

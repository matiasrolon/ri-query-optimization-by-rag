# -*- coding: utf-8 -*-
"""
Main entry point — MS MARCO passage indexing benchmark.

Reads the indexing method from .env (INDEXING_METHOD) and runs the
corresponding indexer (PyTerrier or PISA).

If a previously built index is found on disk it is loaded directly,
skipping both the dataset load and the indexing step (unless
FORCE_REINDEX=true is set in .env).

Usage:
    python main.py
"""

import os

import pyterrier as pt

import config
from indexing import get_indexer


def main() -> None:
    # Initialise PyTerrier
    if not pt.started():
        pt.init()

    print(f"CPUs: {os.cpu_count()} | THREADS = {config.THREADS}")
    print(f"Método de indexación: {config.INDEXING_METHOD.upper()}")
    if config.FORCE_REINDEX:
        print("⚠  FORCE_REINDEX activado — se re-construirá el índice.")
    print("-" * 50)

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


if __name__ == "__main__":
    main()

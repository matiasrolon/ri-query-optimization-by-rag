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


def _resolve_index_path() -> str:
    """Return the expected index sub-directory for the current method."""
    suffix = "idx_terrier" if config.INDEXING_METHOD == "pyterrier" else "idx_pisa"
    return os.path.join(config.INDEX_DIR, suffix)


def _index_already_exists(index_path: str) -> bool:
    """Check if an index already exists on disk (method-specific heuristic)."""
    if config.INDEXING_METHOD == "pyterrier":
        return os.path.isfile(os.path.join(index_path, "data.properties"))
    else:  # pisa
        return os.path.isfile(os.path.join(index_path, "inv.pisa"))


def main() -> None:
    # Initialise PyTerrier
    if not pt.started():
        pt.init()

    print(f"CPUs: {os.cpu_count()} | THREADS = {config.THREADS}")
    print(f"Método de indexación: {config.INDEXING_METHOD.upper()}")
    if config.FORCE_REINDEX:
        print("⚠  FORCE_REINDEX activado — se re-construirá el índice.")
    print("-" * 50)

    # Determine if we can skip loading the dataset
    index_path = _resolve_index_path()
    need_indexing = config.FORCE_REINDEX or not _index_already_exists(index_path)

    if need_indexing:
        # Load collection from local TSV file
        from indexing.dataset import load_collection
        sample = load_collection()
    else:
        print("Índice existente detectado en disco — omitiendo carga del dataset.")
        sample = []  # Won't be used; the indexer will load from disk

    # Dispatch to the configured indexer
    if config.INDEXING_METHOD == "pyterrier":
        from indexing.pyterrier_indexer import index
    elif config.INDEXING_METHOD == "pisa":
        from indexing.pisa_indexer import index

    index(sample, config.INDEX_DIR, config.THREADS)

    print("\n" + "=" * 50)
    print("Indexación completada.")


if __name__ == "__main__":
    main()

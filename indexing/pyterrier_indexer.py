# -*- coding: utf-8 -*-
"""
PyTerrier (Terrier) indexing module.
"""

import os
import shutil
import time

import pyterrier as pt

import config
from indexing.base import print_stats


def _index_exists(terrier_path: str) -> bool:
    """Check whether a valid Terrier index already exists on disk."""
    # Terrier stores a 'data.properties' file inside the index directory.
    return os.path.isfile(os.path.join(terrier_path, "data.properties"))


def _load_existing(terrier_path: str) -> None:
    """Load an existing Terrier index from disk and display its stats."""
    print(f"Índice Terrier encontrado en: {terrier_path}")
    print("Cargando índice existente (sin re-indexar)...")

    t0 = time.time()
    index_ref = pt.IndexRef.of(os.path.join(terrier_path, "data.properties"))
    idx = pt.IndexFactory.of(index_ref)
    elapsed = time.time() - t0

    stats = idx.getCollectionStatistics()
    print(f"Índice cargado en {elapsed:.2f}s")
    print(f"  Documentos      : {stats.getNumberOfDocuments():,}")
    print(f"  Términos únicos : {stats.getNumberOfUniqueTerms():,}")
    print(f"  Tokens          : {stats.getNumberOfTokens():,}")


def index(sample: list[dict], index_dir: str, threads: int) -> None:
    """
    Build a Terrier inverted index from the given document sample.

    If a previously built index already exists on disk and FORCE_REINDEX
    is False, the existing index is loaded instead.

    Parameters
    ----------
    sample : list[dict]
        Documents to index (each with 'docno' and 'text' keys).
    index_dir : str
        Directory where the Terrier index will be stored.
    threads : int
        Number of threads to use for indexing.
    """
    terrier_path = os.path.join(index_dir, "idx_terrier")

    # ── Reuse existing index if available ──────────────────────────────────
    if _index_exists(terrier_path) and not config.FORCE_REINDEX:
        _load_existing(terrier_path)
        return

    # ── Build new index ────────────────────────────────────────────────────
    if config.FORCE_REINDEX and os.path.exists(terrier_path):
        print("FORCE_REINDEX activado — eliminando índice anterior...")
        shutil.rmtree(terrier_path)

    # meta={'docno': 20}: only store docno, we are benchmarking index construction
    indexer = pt.IterDictIndexer(
        terrier_path, meta={"docno": 20}, threads=threads
    )

    t0 = time.time()
    index_ref = indexer.index(sample)
    elapsed = time.time() - t0

    stats = pt.IndexFactory.of(index_ref).getCollectionStatistics()

    print_stats(
        engine_name="Terrier (PyTerrier)",
        time_secs=elapsed,
        num_docs=stats.getNumberOfDocuments(),
        num_terms=stats.getNumberOfUniqueTerms(),
        index_path=terrier_path,
        sample_size=len(sample),
        num_tokens=stats.getNumberOfTokens(),
    )

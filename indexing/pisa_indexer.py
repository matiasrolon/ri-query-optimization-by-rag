# -*- coding: utf-8 -*-
"""
PISA (pyterrier_pisa) indexing module.
"""

import os
import shutil
import time

import pyterrier as pt
from pyterrier_pisa import PisaIndex
from pyterrier.java import autoclass

import config
from indexing.base import print_stats


def _index_exists(pisa_path: str) -> bool:
    """Check whether a valid PISA index already exists on disk."""
    # PisaIndex stores several files; check for the canonical 'inv.pisa' file.
    return os.path.isfile(os.path.join(pisa_path, "inv.docs"))


def _load_existing(pisa_path: str) -> None:
    """Load an existing PISA index from disk and display its stats."""
    print(f"Índice PISA encontrado en: {pisa_path}")
    print("Cargando índice existente (sin re-indexar)...")

    t0 = time.time()
    pisa_index = PisaIndex(pisa_path)
    elapsed = time.time() - t0

    print(f"Índice cargado en {elapsed:.2f}s")
    print(f"  Documentos      : {pisa_index.num_docs():,}")
    print(f"  Términos únicos : {pisa_index.num_terms():,}")


def _preprocess_for_pisa(sample: list[dict]) -> list[dict]:
    """
    Apply tokenisation and stopword removal to match Terrier's default
    behaviour. PISA does not remove stopwords by default, so we preprocess
    the text to ensure a fair comparison.

    Uses Terrier's Java classes (Tokeniser and Stopwords) via pyterrier.java.

    Parameters
    ----------
    sample : list[dict]
        Original documents with 'docno' and 'text' keys.

    Returns
    -------
    list[dict]
        Documents with stopwords removed from the 'text' field.
    """

    StringReader = autoclass("java.io.StringReader")
    Tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser")
    Stopwords = autoclass("org.terrier.terms.Stopwords")

    tokeniser = Tokeniser.getTokeniser()
    stopwords = Stopwords(None)

    print("Aplicando preprocesamiento (eliminación de stopwords) a la muestra para PISA...")
    t0 = time.time()

    processed = []
    for doc in sample:
        reader = StringReader(doc["text"])
        token_stream = tokeniser.tokenise(reader)
        tokens = []
        while token_stream.hasNext():
            tok = token_stream.next()
            if tok is not None and not stopwords.isStopword(tok):
                tokens.append(tok)
        processed.append({"docno": doc["docno"], "text": " ".join(tokens)})

    print(f"Preprocesamiento completado en {time.time() - t0:.1f}s")
    return processed


def index(sample: list[dict], index_dir: str, threads: int) -> None:
    """
    Build a PISA inverted index from the given document sample.

    If a previously built index already exists on disk and FORCE_REINDEX
    is False, the existing index is loaded instead.

    Parameters
    ----------
    sample : list[dict]
        Documents to index (each with 'docno' and 'text' keys).
    index_dir : str
        Directory where the PISA index will be stored.
    threads : int
        Number of threads to use for indexing.
    """
    pisa_path = os.path.join(index_dir, "idx_pisa")

    # ── Reuse existing index if available ──────────────────────────────────
    if _index_exists(pisa_path) and not config.FORCE_REINDEX:
        _load_existing(pisa_path)
        returnº

    # ── Build new index ────────────────────────────────────────────────────
    if config.FORCE_REINDEX and os.path.exists(pisa_path):
        print("FORCE_REINDEX activado — eliminando índice anterior...")
        shutil.rmtree(pisa_path)

    processed_sample = _preprocess_for_pisa(sample)

    pisa_index = PisaIndex(pisa_path, stemmer="porter2", threads=threads)

    t0 = time.time()
    pisa_index.index(processed_sample)
    elapsed = time.time() - t0

    print_stats(
        engine_name="PISA (pyterrier_pisa)",
        time_secs=elapsed,
        num_docs=pisa_index.num_docs(),
        num_terms=pisa_index.num_terms(),
        index_path=pisa_path,
        sample_size=len(sample),
    )

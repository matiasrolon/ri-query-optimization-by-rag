# -*- coding: utf-8 -*-
"""
Verify that a built index is correctly loaded by looking up individual terms
and running a simple retrieval query.

Usage:
    python verify_index.py                  # uses INDEXING_METHOD from .env
    python verify_index.py "information"    # look up a specific term
    python verify_index.py "machine learning"  # run a multi-term query
"""

import os
import sys

import pyterrier as pt

import config


def verify_terrier(query_text: str) -> None:
    """Verify a Terrier index by looking up a term in the lexicon."""
    terrier_path = os.path.join(config.INDEX_DIR, "idx_terrier")
    props_file = os.path.join(terrier_path, "data.properties")

    if not os.path.isfile(props_file):
        print(f"❌ No se encontró el índice Terrier en: {terrier_path}")
        return

    index_ref = pt.IndexRef.of(props_file)
    idx = pt.IndexFactory.of(index_ref)
    stats = idx.getCollectionStatistics()

    print("=" * 60)
    print("📊 Estadísticas del Índice Terrier")
    print("=" * 60)
    print(f"  Documentos      : {stats.getNumberOfDocuments():,}")
    print(f"  Términos únicos : {stats.getNumberOfUniqueTerms():,}")
    print(f"  Tokens totales  : {stats.getNumberOfTokens():,}")
    print()

    # ── Lexicon lookup ─────────────────────────────────────────────────────
    lexicon = idx.getLexicon()
    term = query_text.split()[0].lower()  # look up the first word
    entry = lexicon.getLexiconEntry(term)

    if entry is not None:
        print(f"🔍 Búsqueda del término: '{term}'")
        print(f"  Term ID         : {entry.getTermId()}")
        print(f"  Frecuencia (Tf) : {entry.getFrequency():,}")
        print(f"  Docs con término: {entry.getDocumentFrequency():,}")
    else:
        print(f"⚠  Término '{term}' no encontrado en el léxico.")
        print("   Probá con otro término (ej: 'the', 'information', 'world').")
    print()

    # ── Run a retrieval query ──────────────────────────────────────────────
    print(f"🔎 Ejecutando query: \"{query_text}\"")
    bm25 = pt.terrier.Retriever(idx, wmodel="BM25", num_results=5)
    results = bm25.search(query_text)

    if results.empty:
        print("   No se encontraron resultados.")
    else:
        print(f"   Top {len(results)} resultados:")
        print(results[["docno", "score", "rank"]].to_string(index=False))
    print()


def verify_pisa(query_text: str) -> None:
    """Verify a PISA index by checking stats and running a query."""
    from pyterrier_pisa import PisaIndex

    pisa_path = os.path.join(config.INDEX_DIR, "idx_pisa")

    if not os.path.isfile(os.path.join(pisa_path, "inv.docs")):
        print(f"❌ No se encontró el índice PISA en: {pisa_path}")
        return

    pisa_index = PisaIndex(pisa_path)

    print("=" * 60)
    print("📊 Estadísticas del Índice PISA")
    print("=" * 60)
    print(f"  Documentos      : {pisa_index.num_docs():,}")
    print(f"  Términos únicos : {pisa_index.num_terms():,}")
    print()

    # ── Run a retrieval query ──────────────────────────────────────────────
    print(f"🔎 Ejecutando query: \"{query_text}\"")
    bm25 = pisa_index.bm25(num_results=5)
    results = bm25.search(query_text)

    if results.empty:
        print("   No se encontraron resultados.")
    else:
        print(f"   Top {len(results)} resultados:")
        print(results[["docno", "score", "rank"]].to_string(index=False))
    print()


def main() -> None:
    if not pt.started():
        pt.init()

    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "information retrieval"

    print(f"\n🧪 Verificación del índice ({config.INDEXING_METHOD.upper()})")
    print(f"   Query de prueba: \"{query_text}\"\n")

    if config.INDEXING_METHOD == "pyterrier":
        verify_terrier(query_text)
    elif config.INDEXING_METHOD == "pisa":
        verify_pisa(query_text)


if __name__ == "__main__":
    main()

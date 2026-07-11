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

# Ensure the project root is in sys.path so that `config` is importable
# when running from the tests/ dir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyterrier as pt

import config
from indexing import get_indexer


def main() -> None:
    if not pt.started():
        pt.init()

    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "information retrieval"

    indexer = get_indexer()

    print(f"\n🧪 Verificación del índice ({indexer.engine_name})")
    print(f"   Query de prueba: \"{query_text}\"\n")

    if not indexer.index_exists():
        print(f"❌ No se encontró el índice en: {indexer.index_path}")
        return

    indexer.load_index()

    # ── Statistics ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"📊 Estadísticas del Índice ({indexer.engine_name})")
    print("=" * 60)
    print(f"  Documentos      : {indexer.get_num_docs():,}")
    print(f"  Términos únicos : {indexer.get_num_terms():,}")
    num_tokens = indexer.get_num_tokens()
    if num_tokens is not None:
        print(f"  Tokens totales  : {num_tokens:,}")
    print()

    # ── Term lookup (if supported) ─────────────────────────────────────────
    term = query_text.split()[0].lower()
    info = indexer.lookup_term(term)

    if info is not None:
        print(f"🔍 Búsqueda del término: '{term}'")
        print(f"  Term ID         : {info['term_id']}")
        print(f"  Frecuencia (Tf) : {info['frequency']:,}")
        print(f"  Docs con término: {info['document_frequency']:,}")
    else:
        cf = indexer.get_collection_frequency(term)
        if cf > 0:
            print(f"🔍 Término '{term}': collection frequency ≈ {cf:,}")
        else:
            print(f"⚠  Término '{term}' no encontrado en el índice.")
    print()

    # ── Run a retrieval query ──────────────────────────────────────────────
    print(f"🔎 Ejecutando query: \"{query_text}\"")
    bm25 = indexer.bm25_retriever(num_results=5)
    results = bm25.search(query_text)

    if results.empty:
        print("   No se encontraron resultados.")
    else:
        print(f"   Top {len(results)} resultados:")
        print(results[["docno", "score", "rank"]].to_string(index=False))
    print()


if __name__ == "__main__":
    main()

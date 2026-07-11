# -*- coding: utf-8 -*-
"""
Test the RAG-based query expansion pipeline end-to-end.

Runs a single call to RAGExpander.search(), which internally performs:
  1. First-pass BM25 retrieval (top-k documents).
  2. LLM reformulation of the query using the retrieved passages.
  3. Second-pass BM25 retrieval with the reformulated query.

Per-stage timings are printed by the class itself.

Usage:
    python search_query_rag.py "what is machine learning"
    python search_query_rag.py "how does photosynthesis work"
"""

import os
import sys
import time

import pandas as pd

# Ensure the project root is in sys.path so that `config` and
# `query_expansion` are importable when running from the tests/ dir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyterrier as pt

import config
from query_expansion.rag import RAGExpander
from indexing import get_indexer

# ── Hardcoded display settings ────────────────────────────────────────────
# Set to True to include the full document text as the last column in results.
SHOW_TEXT = False


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python search_query_rag.py <query>")
        print('Ejemplo: python search_query_rag.py "what is machine learning"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    if not pt.started():
        pt.init()

    # ── Show current configuration ────────────────────────────────────────
    print()
    print("=" * 65)
    print("⚙  Configuración")
    print("=" * 65)
    print(f"  INDEXING_METHOD : {config.INDEXING_METHOD}")
    print(f"  FEEDBACK_DOCS   : {config.FEEDBACK_DOCS}")
    print(f"  FEEDBACK_TERMS  : {config.FEEDBACK_TERMS}")
    print(f"  FEEDBACK_LAMBDA : {config.FEEDBACK_LAMBDA}")
    print(f"  LLM_MODEL       : {config.LLM_MODEL}")
    print(f"  LLM_BASE_URL    : {config.LLM_BASE_URL}")
    print()

    # ── Initialize RAGExpander ────────────────────────────────────────────
    print("🔧 Inicializando RAGExpander...")
    t0 = time.time()
    indexer = get_indexer()
    expander = RAGExpander(indexer=indexer)
    print(f"   Listo en {time.time() - t0:.2f}s")
    print()

    # ── Run search() – single call covers expand + retrieval ──────────────
    print("=" * 65)
    print(f"🔎 Búsqueda RAG para: \"{query}\"")
    print("=" * 65)

    t0 = time.time()
    results = expander.search(query)
    total_time = time.time() - t0

    print(f"  ─────────────────────────────────")
    print(f"  ⏱  Total                 : {total_time:.3f}s")
    print()

    # ── Display results ───────────────────────────────────────────────────
    if results.empty:
        print("   No se encontraron resultados.")
    else:
        top_n = min(10, len(results))
        print(f"📄 Top {top_n} de {len(results)} resultados:")
        print()

        # Build column order: rank first, then docno/score, text last (if enabled)
        display_cols = ["rank"] + [c for c in ["docno", "score"] if c in results.columns]

        display_df = results[display_cols].head(top_n).copy()

        if SHOW_TEXT:
            top_results = results.head(top_n)
            texts = indexer.get_texts(top_results)
            display_df["text"] = texts

        pd.set_option("display.max_colwidth", None)
        print(display_df.to_string(index=False))

    print()


if __name__ == "__main__":
    main()

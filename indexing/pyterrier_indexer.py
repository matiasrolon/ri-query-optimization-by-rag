# -*- coding: utf-8 -*-
"""
PyTerrier (Terrier) indexer implementation.
"""

from __future__ import annotations

import os
import shutil
import time

import pandas as pd
import pyterrier as pt

import config
from indexing.base import BaseIndexer, print_stats


class TerrierIndexer(BaseIndexer):
    """Indexer backed by the Terrier engine (via PyTerrier)."""

    def __init__(self, index_dir: str) -> None:
        super().__init__(index_dir)
        self._index = None  # Populated by load_index / build_index

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def index_path(self) -> str:
        return os.path.join(self._index_dir, "idx_terrier")

    @property
    def engine_name(self) -> str:
        return "Terrier (PyTerrier)"

    # ── Index lifecycle ───────────────────────────────────────────────────

    def index_exists(self) -> bool:
        return os.path.isfile(os.path.join(self.index_path, "data.properties"))

    def load_index(self) -> None:
        print(f"Índice Terrier encontrado en: {self.index_path}")
        print("Cargando índice existente (sin re-indexar)...")

        t0 = time.time()
        index_ref = pt.IndexRef.of(
            os.path.join(self.index_path, "data.properties")
        )
        self._index = pt.IndexFactory.of(index_ref)
        elapsed = time.time() - t0

        stats = self._index.getCollectionStatistics()
        print(f"Índice cargado en {elapsed:.2f}s")
        print(f"  Documentos      : {stats.getNumberOfDocuments():,}")
        print(f"  Términos únicos : {stats.getNumberOfUniqueTerms():,}")
        print(f"  Tokens          : {stats.getNumberOfTokens():,}")

    def build_index(self, sample: list[dict], threads: int) -> None:
        if config.FORCE_REINDEX and os.path.exists(self.index_path):
            print("FORCE_REINDEX activado — eliminando índice anterior...")
            shutil.rmtree(self.index_path)

        indexer = pt.IterDictIndexer(
            self.index_path, meta={"docno": 20}, threads=threads
        )

        t0 = time.time()
        index_ref = indexer.index(sample)
        elapsed = time.time() - t0

        self._index = pt.IndexFactory.of(index_ref)
        stats = self._index.getCollectionStatistics()

        print_stats(
            engine_name=self.engine_name,
            time_secs=elapsed,
            num_docs=stats.getNumberOfDocuments(),
            num_terms=stats.getNumberOfUniqueTerms(),
            index_path=self.index_path,
            sample_size=len(sample),
            num_tokens=stats.getNumberOfTokens(),
        )

    # ── Retrieval ─────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the index if it hasn't been loaded yet."""
        if self._index is None:
            if not self.index_exists():
                raise FileNotFoundError(
                    f"No se encontró el índice Terrier en: {self.index_path}"
                )
            self.load_index()

    def bm25_retriever(self, num_results: int | None = None):
        self._ensure_loaded()
        kwargs: dict = {"wmodel": "BM25"}
        if num_results is not None:
            kwargs["num_results"] = num_results
        return pt.terrier.Retriever(self._index, **kwargs)

    # ── Document text access ──────────────────────────────────────────────

    def get_texts(self, results: pd.DataFrame) -> list[str]:
        if "text" in results.columns:
            return results["text"].tolist()

        # Try MetaIndex (only works if text was stored at index time)
        self._ensure_loaded()
        try:
            meta = self._index.getMetaIndex()
            texts = []
            for docid in results["docid"]:
                try:
                    texts.append(meta.getItem("text", int(docid)))
                except Exception:
                    texts.append("")
            if any(t for t in texts):
                return texts
        except Exception:
            pass

        # Fallback to collection.tsv
        return self._texts_from_collection(results["docno"].tolist())

    # ── Collection statistics ─────────────────────────────────────────────

    def get_num_docs(self) -> int:
        self._ensure_loaded()
        return self._index.getCollectionStatistics().getNumberOfDocuments()

    def get_num_terms(self) -> int:
        self._ensure_loaded()
        return self._index.getCollectionStatistics().getNumberOfUniqueTerms()

    def get_num_tokens(self) -> int | None:
        self._ensure_loaded()
        return self._index.getCollectionStatistics().getNumberOfTokens()

    def get_collection_frequency(self, term: str) -> int:
        self._ensure_loaded()
        entry = self._index.getLexicon().getLexiconEntry(term)
        if entry is None:
            return 0
        return entry.getFrequency()

    # ── Optional capabilities ─────────────────────────────────────────────

    def lookup_term(self, term: str) -> dict | None:
        self._ensure_loaded()
        entry = self._index.getLexicon().getLexiconEntry(term)
        if entry is None:
            return None
        return {
            "term_id": entry.getTermId(),
            "frequency": entry.getFrequency(),
            "document_frequency": entry.getDocumentFrequency(),
        }

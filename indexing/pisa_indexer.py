# -*- coding: utf-8 -*-
"""
PISA (pyterrier_pisa) indexer implementation.
"""

from __future__ import annotations

import os
import shutil
import time

import pyterrier as pt
from pyterrier_pisa import PisaIndex

import config
from indexing.base import BaseIndexer, print_stats


class PisaIndexer(BaseIndexer):
    """Indexer backed by the PISA engine (via pyterrier_pisa)."""

    def __init__(self, index_dir: str) -> None:
        super().__init__(index_dir)
        self._pisa_index: PisaIndex | None = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def index_path(self) -> str:
        return os.path.join(self._index_dir, "idx_pisa")

    @property
    def engine_name(self) -> str:
        return "PISA (pyterrier_pisa)"

    # ── Index lifecycle ───────────────────────────────────────────────────

    def index_exists(self) -> bool:
        return os.path.isfile(os.path.join(self.index_path, "inv.docs"))

    def load_index(self) -> None:
        print(f"Índice PISA encontrado en: {self.index_path}")
        print("Cargando índice existente (sin re-indexar)...")

        t0 = time.time()
        self._pisa_index = PisaIndex(self.index_path)
        elapsed = time.time() - t0

        print(f"Índice cargado en {elapsed:.2f}s")
        print(f"  Documentos      : {self._pisa_index.num_docs():,}")
        print(f"  Términos únicos : {self._pisa_index.num_terms():,}")

    def build_index(self, sample: list[dict], threads: int) -> None:
        if config.FORCE_REINDEX and os.path.exists(self.index_path):
            print("FORCE_REINDEX activado — eliminando índice anterior...")
            shutil.rmtree(self.index_path)

        self._pisa_index = PisaIndex(
            self.index_path, stemmer="porter2", stops="terrier", threads=threads
        )

        t0 = time.time()
        self._pisa_index.index(sample)
        elapsed = time.time() - t0

        print_stats(
            engine_name=self.engine_name,
            time_secs=elapsed,
            num_docs=self._pisa_index.num_docs(),
            num_terms=self._pisa_index.num_terms(),
            index_path=self.index_path,
            sample_size=len(sample),
        )


    # ── Retrieval ─────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the index if it hasn't been loaded yet."""
        if self._pisa_index is None:
            if not self.index_exists():
                raise FileNotFoundError(
                    f"No se encontró el índice PISA en: {self.index_path}"
                )
            self.load_index()

    def bm25_retriever(self, num_results: int | None = None):
        self._ensure_loaded()
        kwargs: dict = {}
        if num_results is not None:
            kwargs["num_results"] = num_results
        return self._pisa_index.bm25(**kwargs)

    # ── Collection statistics ─────────────────────────────────────────────

    def get_num_docs(self) -> int:
        self._ensure_loaded()
        return self._pisa_index.num_docs()

    def get_num_terms(self) -> int:
        self._ensure_loaded()
        return self._pisa_index.num_terms()

    def get_collection_frequency(self, term: str) -> int:
        """
        Approximate collection frequency using a single-term search.

        PISA does not expose direct lexicon access.
        """
        self._ensure_loaded()
        try:
            res = self._pisa_index.bm25().search(term)
            return len(res) if not res.empty else 0
        except Exception:
            return 0

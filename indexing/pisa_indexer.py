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
from pyterrier.java import autoclass

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

        processed_sample = self._preprocess_for_pisa(sample)

        self._pisa_index = PisaIndex(
            self.index_path, stemmer="porter2", threads=threads
        )

        t0 = time.time()
        self._pisa_index.index(processed_sample)
        elapsed = time.time() - t0

        print_stats(
            engine_name=self.engine_name,
            time_secs=elapsed,
            num_docs=self._pisa_index.num_docs(),
            num_terms=self._pisa_index.num_terms(),
            index_path=self.index_path,
            sample_size=len(sample),
        )

    # ── Preprocessing ─────────────────────────────────────────────────────

    @staticmethod
    def _preprocess_for_pisa(sample: list[dict]) -> list[dict]:
        """
        Apply tokenisation and stopword removal to match Terrier's default
        behaviour.  PISA does not remove stopwords by default, so we
        preprocess the text to ensure a fair comparison.

        Uses Terrier's Java classes (Tokeniser and Stopwords) via
        pyterrier.java.

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

        print(
            "Aplicando preprocesamiento (eliminación de stopwords) "
            "a la muestra para PISA..."
        )
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

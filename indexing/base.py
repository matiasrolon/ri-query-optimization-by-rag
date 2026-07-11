# -*- coding: utf-8 -*-
"""
Base indexer interface and shared utilities for indexing modules.
"""

from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod

import pandas as pd

import config


def dir_mb(path: str) -> float:
    """Calculate the total size of a directory in megabytes."""
    return sum(
        f.stat().st_size
        for f in pathlib.Path(path).rglob("*")
        if f.is_file()
    ) / 1e6


def print_stats(
    engine_name: str,
    time_secs: float,
    num_docs: int,
    num_terms: int,
    index_path: str,
    sample_size: int,
    num_tokens: int | None = None,
) -> None:
    """Print standardised indexing statistics."""
    print(f"\n===== {engine_name} =====")
    print(f"Tiempo de indexacion : {time_secs:8.1f} s")
    print(f"Documentos           : {num_docs:,}")
    print(f"Terminos unicos      : {num_terms:,}")
    if num_tokens is not None:
        print(f"Tokens               : {num_tokens:,}")
    print(f"Tamano en disco      : {dir_mb(index_path):.1f} MB")
    print(f"Throughput           : {sample_size / time_secs:,.0f} docs/s")


class BaseIndexer(ABC):
    """
    Abstract base class for index engines.

    Subclasses must implement all abstract methods so that the rest of
    the codebase can work with any engine without conditional branching.
    """

    def __init__(self, index_dir: str) -> None:
        self._index_dir = index_dir

    # ── Properties ────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def index_path(self) -> str:
        """Full path to the engine-specific index subdirectory."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Human-readable engine name for display."""

    # ── Index lifecycle ───────────────────────────────────────────────────

    @abstractmethod
    def index_exists(self) -> bool:
        """Check whether a valid index already exists on disk."""

    @abstractmethod
    def build_index(self, sample: list[dict], threads: int) -> None:
        """Build the index from the given document sample."""

    @abstractmethod
    def load_index(self) -> None:
        """Load an existing index from disk and print its stats."""

    # ── Retrieval ─────────────────────────────────────────────────────────

    @abstractmethod
    def bm25_retriever(self, num_results: int | None = None):
        """
        Return a BM25 retriever (a PyTerrier Transformer).

        Parameters
        ----------
        num_results : int | None
            Maximum number of results.  When *None* the engine default
            is used.
        """

    # ── Document text access ──────────────────────────────────────────────

    def get_texts(self, results: pd.DataFrame) -> list[str]:
        """
        Return the textual content for the retrieved documents.

        The default implementation looks up texts from ``collection.tsv``.
        Subclasses may override to provide faster engine-specific access.
        """
        if "text" in results.columns:
            return results["text"].tolist()
        return self._texts_from_collection(results["docno"].tolist())

    @staticmethod
    def _texts_from_collection(docnos: list[str]) -> list[str]:
        """Look up document texts from the TSV collection file."""
        docno_set = set(docnos)
        mapping: dict[str, str] = {}
        try:
            with open(config.COLLECTION_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) == 2 and parts[0] in docno_set:
                        mapping[parts[0]] = parts[1]
                        if len(mapping) == len(docno_set):
                            break
        except FileNotFoundError:
            pass
        return [mapping.get(d, "") for d in docnos]

    # ── Collection statistics ─────────────────────────────────────────────

    @abstractmethod
    def get_num_docs(self) -> int:
        """Return the total number of documents in the index."""

    @abstractmethod
    def get_num_terms(self) -> int:
        """Return the number of unique terms in the index."""

    def get_num_tokens(self) -> int | None:
        """Return total tokens in the collection, or None if unsupported."""
        return None

    @abstractmethod
    def get_collection_frequency(self, term: str) -> int:
        """
        Return the collection frequency of *term*.

        Returns 0 if the term is not found in the index.
        """

    # ── Optional capabilities ─────────────────────────────────────────────

    def lookup_term(self, term: str) -> dict | None:
        """
        Look up detailed term statistics from the lexicon.

        Returns a dict with keys like ``term_id``, ``frequency``,
        ``document_frequency``, or *None* if not supported by the engine.
        """
        return None

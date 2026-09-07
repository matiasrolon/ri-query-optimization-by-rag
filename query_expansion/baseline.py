# -*- coding: utf-8 -*-
"""
Baseline BM25 retrieval without query expansion.

Provides a standard BM25 retrieval pipeline matching the interface of
PRFExpander and RAGExpander, used as the control baseline to measure
the true effectiveness and latency trade-offs of query expansion.
"""

from __future__ import annotations

import time
import pandas as pd
import pyterrier as pt

from indexing import get_indexer
from indexing.base import BaseIndexer
from query_expansion.utils import sanitize_query_str


class BM25Baseline:
    """
    Standard BM25 baseline retriever without query expansion.

    Parameters
    ----------
    indexer : BaseIndexer | None
        Pre-built indexer instance. When *None*, one is created
        automatically via ``get_indexer()``.
    num_results : int | None
        Maximum number of documents to retrieve.
    """

    def __init__(
        self,
        indexer: BaseIndexer | None = None,
        num_results: int | None = None,
    ) -> None:
        if not pt.started():
            pt.init()

        self._indexer = indexer or get_indexer()
        self._retriever = self._indexer.bm25_retriever(num_results=num_results)
        self.last_timings: dict[str, float] = {
            "time_first_pass": 0.0,
            "time_get_texts": 0.0,
            "time_llm": 0.0,
            "time_second_pass": 0.0,
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def expand(self, query: str) -> str:
        """
        Return the sanitized query without adding any expansion terms.
        """
        return sanitize_query_str(query)

    def search(self, query: str) -> pd.DataFrame:
        """
        Run direct BM25 retrieval for the query.
        """
        safe_query = self.expand(query)
        t0 = time.time()
        results = self._retriever.search(safe_query)
        elapsed = time.time() - t0

        self.last_timings = {
            "time_first_pass": round(elapsed, 4),
            "time_get_texts": 0.0,
            "time_llm": 0.0,
            "time_second_pass": 0.0,
        }
        return results

    def expand_and_search(self, query: str) -> tuple[str, pd.DataFrame]:
        """
        Sanitize query and run BM25 retrieval in one call.

        Returns
        -------
        tuple[str, pd.DataFrame]
            A tuple of (sanitized_query, search_results).
        """
        safe_query = self.expand(query)
        t0 = time.time()
        results = self._retriever.search(safe_query)
        elapsed = time.time() - t0

        self.last_timings = {
            "time_first_pass": round(elapsed, 4),
            "time_get_texts": 0.0,
            "time_llm": 0.0,
            "time_second_pass": 0.0,
        }
        return safe_query, results

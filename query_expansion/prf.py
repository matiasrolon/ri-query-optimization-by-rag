# -*- coding: utf-8 -*-
"""
Pseudo-Relevance Feedback (PRF) query expansion.

Implements a classic two-pass retrieval approach:
  1. Run an initial BM25 retrieval to obtain top-k documents.
  2. Extract discriminative terms from those documents using the
     Divergence from Randomness Bo1 model (Terrier's default PRF model).
  3. Interpolate the expanded terms with the original query.
  4. Re-run BM25 with the expanded query.

Supports both PyTerrier (Terrier) and PISA indices via the unified
``BaseIndexer`` interface.

Reference:
    Amati, G. (2003). Probabilistic Models for Information Retrieval
    based on Divergence from Randomness. PhD Thesis.
"""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd
import pyterrier as pt

import config
from indexing import get_indexer
from indexing.base import BaseIndexer


class PRFExpander:
    """
    Pseudo-Relevance Feedback query expander.

    Parameters
    ----------
    fb_docs : int
        Number of top-ranked documents to treat as pseudo-relevant
        in the initial retrieval pass.
    fb_terms : int
        Number of expansion terms to add to the original query.
    fb_lambda : float
        Interpolation weight for the **original** query (0.0–1.0).
        The expanded terms receive weight ``(1 - fb_lambda)``.
    indexer : BaseIndexer | None
        Pre-built indexer instance.  When *None* one is created
        automatically via ``get_indexer()``.
    """

    def __init__(
        self,
        fb_docs: int | None = None,
        fb_terms: int | None = None,
        fb_lambda: float | None = None,
        indexer: BaseIndexer | None = None,
    ) -> None:
        self.fb_docs = fb_docs if fb_docs is not None else config.FEEDBACK_DOCS
        self.fb_terms = fb_terms if fb_terms is not None else config.FEEDBACK_TERMS
        self.fb_lambda = fb_lambda if fb_lambda is not None else config.FEEDBACK_LAMBDA

        # Ensure PyTerrier is initialised
        if not pt.started():
            pt.init()

        # Indexer (engine-agnostic)
        self._indexer = indexer or get_indexer()
        self._first_pass = self._indexer.bm25_retriever(num_results=self.fb_docs)
        self._second_pass = self._indexer.bm25_retriever()

    # ── Term scoring (Bo1 – Bose-Einstein 1) ──────────────────────────────

    def _score_terms_bo1(
        self, term_freqs: Counter, total_tokens_fb: int
    ) -> list[tuple[str, float]]:
        """
        Score candidate expansion terms using the Bo1 divergence model.

        Bo1 weight for a term *t* with term-frequency *tf* in the
        feedback set:

            w(t) = tf * log2(1 + P_n) + log2(1 + 1/P_n)

        where P_n = F / N (collection frequency / number of docs in
        the collection).

        Returns a list of ``(term, weight)`` sorted by descending weight.
        """
        N = self._indexer.get_num_docs()
        scored: list[tuple[str, float]] = []

        for term, tf in term_freqs.items():
            F = self._indexer.get_collection_frequency(term)
            if F == 0:
                continue

            P_n = F / N if N > 0 else 0
            if P_n <= 0:
                continue

            weight = tf * math.log2(1 + P_n) + math.log2(1 + 1.0 / P_n)
            scored.append((term, weight))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.fb_terms]

    # ── Core expansion logic ──────────────────────────────────────────────

    def _extract_terms_from_results(
        self, results: pd.DataFrame
    ) -> tuple[Counter, int]:
        """
        Tokenise the text of the retrieved documents and count term
        frequencies.

        Returns ``(term_counter, total_tokens)`` over the feedback set.
        """
        counter: Counter = Counter()
        total = 0

        texts = self._indexer.get_texts(results)

        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)
            total += len(tokens)

        return counter, total

    def _build_expanded_query(
        self, original_query: str, scored_terms: list[tuple[str, float]]
    ) -> str:
        """
        Build a weighted query string combining the original query
        terms with the expansion terms using the configured fb_lambda.

        Terrier's query language supports ``term^weight`` syntax.
        """
        if not scored_terms:
            return original_query

        # Normalise expansion weights to [0, 1]
        max_w = scored_terms[0][1] if scored_terms else 1.0
        if max_w == 0:
            max_w = 1.0

        parts: list[str] = []

        # Original query terms with weight fb_lambda
        for token in original_query.lower().split():
            parts.append(f"{token}^{self.fb_lambda:.4f}")

        # Expansion terms with weight (1 - fb_lambda) * normalised_score
        expansion_weight = 1.0 - self.fb_lambda
        for term, score in scored_terms:
            w = expansion_weight * (score / max_w)
            parts.append(f"{term}^{w:.4f}")

        return " ".join(parts)

    # ── Public API ─────────────────────────────────────────────────────────

    def expand(self, query: str) -> str:
        """
        Expand a single query using Pseudo-Relevance Feedback.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        str
            The expanded (weighted) query string.
        """
        # 1. First-pass retrieval
        first_results = self._first_pass.search(query)

        if first_results.empty:
            return query

        # 2. Extract term frequencies from pseudo-relevant docs
        term_freqs, total_tokens = self._extract_terms_from_results(
            first_results
        )

        # 3. Score expansion terms with Bo1
        scored = self._score_terms_bo1(term_freqs, total_tokens)

        # 4. Build expanded query
        expanded = self._build_expanded_query(query, scored)
        return expanded

    def search(self, query: str) -> pd.DataFrame:
        """
        End-to-end PRF retrieval: first-pass → expansion → second-pass.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        pd.DataFrame
            Retrieval results from the second pass with the expanded
            query.  Columns include ``qid``, ``docno``, ``score``,
            ``rank``.
        """
        expanded_query = self.expand(query)
        return self._second_pass.search(expanded_query)

    def search_batch(self, topics: pd.DataFrame) -> pd.DataFrame:
        """
        Run PRF expansion + retrieval for a batch of queries.

        Parameters
        ----------
        topics : pd.DataFrame
            A PyTerrier-style topics frame with at least ``qid`` and
            ``query`` columns.

        Returns
        -------
        pd.DataFrame
            Combined results for all queries.
        """
        all_results = []
        for _, row in topics.iterrows():
            expanded = self.expand(row["query"])
            result = self._second_pass.search(expanded)
            result["qid"] = row["qid"]
            all_results.append(result)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

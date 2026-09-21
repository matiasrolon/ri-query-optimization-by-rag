# -*- coding: utf-8 -*-
"""
Pseudo-Relevance Feedback (PRF) query expansion using PyTerrier native QE.

Delegates the full expansion process to Terrier's built-in DFR models
(Bo1 or KL), which internally handle:
  - Stopword removal
  - PorterStemmer stemming
  - Deduplication against the original query
  - Term scoring via the selected DFR model
  - Lexicon-based filtering

The expansion model is selected via the ``EXPANSION_MODEL`` environment
variable (``bo1`` or ``kl``).

Pipeline: BM25 (first pass) → QE rewriter → BM25 (second pass).

References:
    Amati, G. & van Rijsbergen, C. J. (2002). Probabilistic models of
    information retrieval based on measuring the divergence from
    randomness. ACM TOIS 20(4).

    Amati, G. (2003). Probability models for information retrieval
    based on divergence from randomness. PhD Thesis, U. Glasgow.
"""

from __future__ import annotations

import pandas as pd
import pyterrier as pt

import config
from indexing import get_indexer
from query_expansion.utils import (
    sanitize_query_str,
    sanitize_query,
    stem_term,
    is_stopword,
)


class PRFExpander:
    """
    Pseudo-Relevance Feedback query expander using PyTerrier native QE.

    Parameters
    ----------
    fb_docs : int
        Number of top-ranked documents to treat as pseudo-relevant
        in the initial retrieval pass.
    fb_terms : int
        Number of expansion terms to add to the original query.
    fb_lambda : float
        Interpolation weight for the **original** query (0.0–1.0).
        Currently used only for interface parity; Bo1/KL use their
        own internal weighting.
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

        # Indexer (must be a TerrierIndexer for native QE)
        self._indexer = indexer or get_indexer()

        if not hasattr(self._indexer, "_index"):
            raise RuntimeError(
                "PRF con QE nativo requiere un TerrierIndexer. "
                "El indexer PISA no está soportado para esta funcionalidad."
            )

        self._indexer._ensure_loaded()
        native_index = self._indexer._index

        self._first_pass = self._indexer.bm25_retriever(num_results=self.fb_docs)
        self._second_pass = self._indexer.bm25_retriever()

        # Select DFR query expansion model
        if config.EXPANSION_MODEL == "kl":
            self._qe = pt.terrier.rewrite.KLQueryExpansion(
                native_index,
                fb_terms=self.fb_terms,
                fb_docs=self.fb_docs,
            )
        else:
            self._qe = pt.terrier.rewrite.Bo1QueryExpansion(
                native_index,
                fb_terms=self.fb_terms,
                fb_docs=self.fb_docs,
            )

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
            The expanded query string in TerrierQL format (with weights).
        """
        safe_query = sanitize_query_str(query)
        first_results = self._first_pass.search(safe_query)

        if first_results.empty:
            return safe_query

        # Apply native QE transformer (R → Q)
        expanded_df = self._qe.transform(first_results)

        if expanded_df.empty or "query" not in expanded_df.columns:
            return safe_query

        return expanded_df.iloc[0]["query"]

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
        _, results, _ = self.expand_and_search(query)
        return results

    def expand_and_search(
        self, query: str
    ) -> tuple[str, pd.DataFrame, dict[str, float]]:
        """
        Expand the query and run the second-pass retrieval in one call.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        tuple[str, pd.DataFrame, dict[str, float]]
            A tuple of (expanded_query, search_results, timings_dict).
        """
        import time

        safe_query = sanitize_query_str(query)

        t0 = time.time()
        first_results = self._first_pass.search(safe_query)
        elapsed_first = time.time() - t0

        if first_results.empty:
            timings = {
                "time_first_pass": round(elapsed_first, 4),
                "time_text_fetch": 0.0,
                "time_llm": 0.0,
                "time_second_pass": 0.0,
            }
            return safe_query, first_results, timings

        t1 = time.time()
        expanded_df = self._qe.transform(first_results)
        if expanded_df.empty or "query" not in expanded_df.columns:
            expanded_query = safe_query
            n_prf = 0
        else:
            expanded_query = expanded_df.iloc[0]["query"]
            clean_exp = expanded_query.replace("applypipeline:off ", "")
            exp_parts = clean_exp.split()
            orig_stemmed = set(stem_term(t) for t in sanitize_query(query) if not is_stopword(t))
            prf_new_terms = [
                t.split("^")[0]
                for t in exp_parts
                if stem_term(t.split("^")[0]) not in orig_stemmed
            ]
            n_prf = len(prf_new_terms)

        results = self._second_pass.search(expanded_query)
        elapsed_second = time.time() - t1

        timings = {
            "time_first_pass": round(elapsed_first, 4),
            "time_text_fetch": 0.0,
            "time_llm": 0.0,
            "time_second_pass": round(elapsed_second, 4),
            "n_terms_proposed": n_prf,
            "n_terms_kept": n_prf,
        }

        return expanded_query, results, timings

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
            expanded, result, _ = self.expand_and_search(row["query"])
            result["qid"] = row["qid"]
            all_results.append(result)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

# -*- coding: utf-8 -*-
"""
RAG-based query expansion.

Implements a two-pass retrieval approach where the query reformulation
step is delegated to a Large Language Model (LLM):
  1. Run an initial BM25 retrieval to obtain top-k documents.
  2. Send the original query + retrieved document texts as context
     to the LLM, asking it to produce an improved search query.
  3. Post-process the LLM output (stopwords, stemming, lexicon
     filtering, dedup, truncation to fb_terms).
  4. Re-run BM25 with the cleaned expanded query.

Uses an OpenAI-compatible API (by default Ollama running locally).
No API key is required for local Ollama deployments.

Reference:
    Gao, L. et al. (2023). Precise Zero-Shot Dense Retrieval without
    Relevance Labels (HyDE). ACL 2023.
"""

from __future__ import annotations

import os
import re
import time
import textwrap
import logging

import pandas as pd
import pyterrier as pt
import httpx
from openai import OpenAI, APITimeoutError, APIConnectionError

import config
from indexing import get_indexer
from indexing.base import BaseIndexer
from query_expansion.utils import (
    sanitize_query_str,
    extract_feedback_terms,
    postprocess_expanded_query,
)

logger = logging.getLogger(__name__)

# ── Default LLM settings ──────────────────────────────────────────────────
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_TIMEOUT = 300  # 5 minutes per request
_DEFAULT_MAX_RETRIES = 3

_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are a search-query optimiser.  Given a user's original search
    query and a set of potentially relevant document passages, your
    task is to produce an improved, more precise search query that
    would retrieve the most relevant documents for the user's
    information need.

    Rules:
    - Output ONLY the improved query, nothing else.
    - Do NOT include explanations, numbering, or bullet points.
    - The improved query must have at most {max_words} words \
({n_original} original + up to {fb_terms} new terms).
    - Do NOT rewrite the original terms; only ADD new relevant terms.
    - Use English unless the original query is in another language.
""")


class RAGExpander:
    """
    RAG-based query expander using an LLM.

    Parameters
    ----------
    fb_docs : int
        Number of top-ranked documents to retrieve in the first pass
        and supply as context to the LLM.
    fb_terms : int
        Maximum number of **additional** terms the LLM should add.
        Also used to truncate the post-processed expansion.
    fb_lambda : float
        *Reserved for interface parity with PRFExpander.*
    indexer : BaseIndexer | None
        Pre-built indexer instance.  When *None* one is created
        automatically via ``get_indexer()``.
    model : str | None
        LLM model identifier.  When *None*, read from ``LLM_MODEL``
        in the ``.env`` file (default: ``qwen2.5:7b``).
    base_url : str | None
        API base URL.  When *None*, read from ``LLM_BASE_URL``
        in the ``.env`` file (default: ``http://localhost:11434/v1``).
    api_key : str | None
        API key.  When *None*, read from the ``OPENAI_API_KEY``
        environment variable.  Not required for local Ollama.
    verbose : bool
        If *True* (default), print the full LLM prompt, retrieved
        passages, and per-step timing to stdout.  Set to *False*
        when running batch evaluations to reduce noise.
    max_tokens : int
        Maximum number of tokens in the LLM response.
    temperature : float
        Sampling temperature for the LLM.
    """

    def __init__(
        self,
        fb_docs: int | None = None,
        fb_terms: int | None = None,
        fb_lambda: float | None = None,
        indexer: BaseIndexer | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        verbose: bool = True,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        self.fb_docs = fb_docs if fb_docs is not None else config.FEEDBACK_DOCS
        self.fb_terms = fb_terms if fb_terms is not None else config.FEEDBACK_TERMS
        self.fb_lambda = fb_lambda if fb_lambda is not None else config.FEEDBACK_LAMBDA

        # LLM settings
        self.model = model if model is not None else config.LLM_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

        resolved_base_url = base_url if base_url is not None else config.LLM_BASE_URL
        # Ollama doesn't need an API key; use 'ollama' as a dummy value
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "") or "ollama"
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url,
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT, connect=30.0),
            max_retries=0,  # we handle retries ourselves for better logging
        )

        # Ensure PyTerrier is initialised
        if not pt.started():
            pt.init()

        # Indexer (must be TerrierIndexer for lexicon-based post-processing)
        self._indexer = indexer or get_indexer()

        if not hasattr(self._indexer, "_index"):
            raise RuntimeError(
                "RAG con post-procesamiento requiere un TerrierIndexer. "
                "El indexer PISA no está soportado para esta funcionalidad."
            )

        self._indexer._ensure_loaded()

        self._first_pass = self._indexer.bm25_retriever(num_results=self.fb_docs)
        self._second_pass = self._indexer.bm25_retriever()
        self.last_timings: dict[str, float] = {
            "time_first_pass": 0.0,
            "time_get_texts": 0.0,
            "time_llm": 0.0,
            "time_second_pass": 0.0,
        }

    # ── Dynamic prompt construction ───────────────────────────────────────

    def _build_system_prompt(self, original_query: str) -> str:
        """Build a system prompt with a dynamic word limit."""
        n_original = len(original_query.split())
        max_words = n_original + self.fb_terms
        return _SYSTEM_PROMPT_TEMPLATE.format(
            max_words=max_words,
            n_original=n_original,
            fb_terms=self.fb_terms,
        )

    def _build_user_prompt(
        self, original_query: str, passages: list[str]
    ) -> str:
        """Build the user-facing prompt with context passages."""
        numbered = "\n".join(
            f"[{i + 1}] {p[:500]}" for i, p in enumerate(passages)
        )
        return (
            f"Original query: {original_query}\n\n"
            f"Retrieved passages:\n{numbered}\n\n"
            f"Improved query:"
        )

    # ── LLM interaction ───────────────────────────────────────────────────

    def _call_llm(self, original_query: str, passages: list[str]) -> str:
        """Call the LLM to reformulate the query, with retry on timeout."""
        system_prompt = self._build_system_prompt(original_query)
        user_prompt = self._build_user_prompt(original_query, passages)

        if self.verbose:
            print("  ┌─ Prompt enviado al LLM ────────────────────────────")
            print(f"  │ [system] {system_prompt[:200]}...")
            print(f"  │ [user]   {user_prompt[:200]}...")
            print("  └────────────────────────────────────────────────────")

        last_exc: Exception | None = None
        for attempt in range(1, _DEFAULT_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                reformulated = response.choices[0].message.content.strip()

                # Safety: if the LLM returns nothing useful, fall back to
                # the original query.
                if not reformulated:
                    return original_query

                return reformulated

            except (APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                wait = 2 ** attempt  # 2s, 4s, 8s
                logger.warning(
                    "LLM timeout (intento %d/%d) para query '%s'. "
                    "Reintentando en %ds...",
                    attempt, _DEFAULT_MAX_RETRIES, original_query[:60], wait,
                )
                time.sleep(wait)

        # All retries exhausted — fall back to the original query
        logger.error(
            "LLM no respondió tras %d intentos para query '%s'. "
            "Usando query original. Error: %s",
            _DEFAULT_MAX_RETRIES, original_query[:60], last_exc,
        )
        return original_query

    # ── Public API ─────────────────────────────────────────────────────────

    def expand(self, query: str) -> str:
        """
        Expand a single query using RAG-based LLM reformulation.

        The LLM output is post-processed to apply the same linguistic
        treatment as the index (stopwords, stemming, lexicon filtering)
        and filtered to terms that appear in the feedback documents.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        str
            The post-processed expanded query (plain text, no weights).
        """
        # 1. First-pass retrieval
        t0 = time.time()
        safe_query = sanitize_query_str(query)
        first_results = self._first_pass.search(safe_query)
        time_first_pass = time.time() - t0
        if self.verbose:
            print(f"  ⏱  First-pass retrieval : {time_first_pass:.3f}s")

        if first_results.empty:
            self.last_timings = {
                "time_first_pass": round(time_first_pass, 4),
                "time_get_texts": 0.0,
                "time_llm": 0.0,
                "time_second_pass": 0.0,
            }
            return safe_query

        # 2. Get passage texts
        t0 = time.time()
        passages = self._indexer.get_texts(first_results)
        time_get_texts = time.time() - t0
        if self.verbose:
            print(f"  ⏱  Recuperación de texto: {time_get_texts:.3f}s")
        passages = [p for p in passages if p.strip()]

        if not passages:
            self.last_timings = {
                "time_first_pass": round(time_first_pass, 4),
                "time_get_texts": round(time_get_texts, 4),
                "time_llm": 0.0,
                "time_second_pass": 0.0,
            }
            return safe_query

        # 3. Extract stemmed terms from feedback docs (for filtering)
        feedback_terms = extract_feedback_terms(passages)

        # 4. Ask the LLM to reformulate the query
        t0 = time.time()
        raw_llm_output = self._call_llm(query, passages)
        time_llm = time.time() - t0
        if self.verbose:
            print(f"  ⏱  Llamada al LLM      : {time_llm:.3f}s")
            print(f"  📝 Salida cruda LLM     : \"{raw_llm_output}\"")

        # 5. Post-process: stopwords, stemming, lexicon, dedup, truncate
        expanded = postprocess_expanded_query(
            original_query=query,
            raw_expanded_text=raw_llm_output,
            index=self._indexer._index,
            fb_terms=self.fb_terms,
            feedback_doc_terms=feedback_terms,
        )

        if self.verbose:
            print(f"  📝 Query post-procesada : \"{expanded}\"")

        self.last_timings = {
            "time_first_pass": round(time_first_pass, 4),
            "time_get_texts": round(time_get_texts, 4),
            "time_llm": round(time_llm, 4),
            "time_second_pass": 0.0,
        }
        return expanded

    def search(self, query: str) -> pd.DataFrame:
        """
        End-to-end RAG retrieval: first-pass → LLM expansion → second-pass.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        pd.DataFrame
            Retrieval results from the second pass with the
            reformulated query.
        """
        _, results = self.expand_and_search(query)
        return results

    def expand_and_search(self, query: str) -> tuple[str, pd.DataFrame]:
        """
        Expand the query and run the second-pass retrieval in one call.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        tuple[str, pd.DataFrame]
            A tuple of (expanded_query, search_results).
        """
        expanded = self.expand(query)

        t0 = time.time()
        results = self._second_pass.search(expanded)
        time_second_pass = time.time() - t0
        if self.verbose:
            print(f"  ⏱  Second-pass retrieval: {time_second_pass:.3f}s")

        self.last_timings["time_second_pass"] = round(time_second_pass, 4)
        return expanded, results

    def search_batch(self, topics: pd.DataFrame) -> pd.DataFrame:
        """
        Run RAG expansion + retrieval for a batch of queries.

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

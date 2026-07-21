# -*- coding: utf-8 -*-
"""
RAG-based query expansion.

Implements a two-pass retrieval approach where the query reformulation
step is delegated to a Large Language Model (LLM):
  1. Run an initial BM25 retrieval to obtain top-k documents.
  2. Send the original query + retrieved document texts as context
     to the LLM, asking it to produce an improved search query.
  3. Re-run BM25 with the reformulated query.

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

logger = logging.getLogger(__name__)

# ── Default LLM settings ──────────────────────────────────────────────────
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_TIMEOUT = 300  # 5 minutes per request
_DEFAULT_MAX_RETRIES = 3

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a search-query optimiser.  Given a user's original search
    query and a set of potentially relevant document passages, your
    task is to produce an improved, more precise search query that
    would retrieve the most relevant documents for the user's
    information need.

    Rules:
    - Output ONLY the improved query, nothing else.
    - Do NOT include explanations, numbering, or bullet points.
    - Keep the query concise (no more than ~30 words).
    - Do NOT rewrite the query; only provide terms to ADD.
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
        *Reserved for interface parity with PRFExpander.*
        Not directly used by the LLM reformulation, but kept so that
        the caller can swap strategies without changing the constructor
        signature.
    fb_lambda : float
        *Reserved for interface parity with PRFExpander.*
        Not directly used, since the LLM produces a complete new query
        rather than a weighted interpolation.
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
        self.fb_terms = fb_terms if fb_terms is not None else config.FEEDBACK_TERMS  # kept for interface parity
        self.fb_lambda = fb_lambda if fb_lambda is not None else config.FEEDBACK_LAMBDA  # kept for interface parity

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

        # Indexer (engine-agnostic)
        self._indexer = indexer or get_indexer()
        self._first_pass = self._indexer.bm25_retriever(num_results=self.fb_docs)
        self._second_pass = self._indexer.bm25_retriever()

    # ── Query sanitisation ────────────────────────────────────────────────

    _TERRIER_SPECIAL_RE = re.compile(r'[/+\-!(){}\[\]:^~\\"\']')

    @classmethod
    def _sanitize_query(cls, text: str) -> str:
        """
        Strip TerrierQL-special characters from a query string so that
        the Terrier query parser does not raise a lexical error.
        """
        cleaned = cls._TERRIER_SPECIAL_RE.sub(" ", text)
        return " ".join(cleaned.split())  # collapse whitespace

    # ── LLM interaction ───────────────────────────────────────────────────

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

    def _call_llm(self, original_query: str, passages: list[str]) -> str:
        """Call the LLM to reformulate the query, with retry on timeout."""
        user_prompt = self._build_user_prompt(original_query, passages)

        if self.verbose:
            print("  ┌─ Prompt enviado al LLM ────────────────────────────")
            print(f"  │ [system] {_SYSTEM_PROMPT}...")
            print(f"  │ [user]   {user_prompt}...")
            print("  └────────────────────────────────────────────────────")

        last_exc: Exception | None = None
        for attempt in range(1, _DEFAULT_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
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

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        str
            The LLM-reformulated query string.
        """
        # 1. First-pass retrieval (sanitise to avoid TerrierQL parse errors)
        t0 = time.time()
        safe_query = self._sanitize_query(query)
        first_results = self._first_pass.search(safe_query)
        if self.verbose:
            print(f"  ⏱  First-pass retrieval : {time.time() - t0:.3f}s")

        if first_results.empty:
            return query

        # 2. Get passage texts
        passages = self._indexer.get_texts(first_results)
        passages = [p for p in passages if p.strip()]

        if not passages:
            return query

        # 3. Ask the LLM to reformulate the query
        t0 = time.time()
        reformulated = self._call_llm(query, passages)
        if self.verbose:
            print(f"  ⏱  Llamada al LLM      : {time.time() - t0:.3f}s")
            print(f"  📝 Query expandida      : \"{reformulated}\"")

        return reformulated

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

        Useful when the caller needs both the expanded query text
        (e.g. for metric logging) and the retrieval results without
        invoking the LLM twice.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        tuple[str, pd.DataFrame]
            A tuple of (expanded_query, search_results).
        """
        reformulated = self.expand(query)

        t0 = time.time()
        safe_reformulated = self._sanitize_query(reformulated)
        results = self._second_pass.search(safe_reformulated)
        if self.verbose:
            print(f"  ⏱  Second-pass retrieval: {time.time() - t0:.3f}s")

        return reformulated, results

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
            reformulated = self.expand(row["query"])
            result = self._second_pass.search(self._sanitize_query(reformulated))
            result["qid"] = row["qid"]
            all_results.append(result)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

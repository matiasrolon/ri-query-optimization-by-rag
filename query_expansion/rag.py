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
    postprocess_expanded_query,
)

logger = logging.getLogger(__name__)

# ── Default LLM settings ──────────────────────────────────────────────────
_DEFAULT_MAX_TOKENS = 48
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_TIMEOUT = 300  # 5 minutes per request
_DEFAULT_MAX_RETRIES = 3

_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an expert Search Engine Indexing Specialist.
    Given a user's original search query and a set of relevant document passages, your task is to generate EXACTLY {fb_terms} additional, highly specific, technical, and domain-discriminative expansion keywords.
    Document passages are ordered by relevance; prefer words from or related to the first ones.

    Rules:
    - Output ONLY a space-separated list of EXACTLY {fb_terms} new expansion keywords.
    - Do NOT include or repeat any words already present in the original query.
    - Do NOT include explanations, numbering, punctuation, or bullet points.
    - Select highly specific technical nouns, domain synonyms, or exact entity names.
    - DO NOT output generic search words such as: definition, explanation, overview, summary, guide, meaning, type, list, cause, effect, symptom, cost, price, history.

    Examples:

    Example 1:
    fb_terms=5
    Original query: treating tension headaches
    Retrieved passages:
    [1] Ibuprofen and acetaminophen are common over-the-counter pain relievers for neurological stress and migraines.
    Additional expansion keywords:
    migraine ibuprofen stress analgesics neurological

    Example 2:
    fb_terms=2
    Original query: prime rate in canada
    Retrieved passages:
    [1] The Bank of Canada sets the overnight lending rate affecting commercial mortgage interest and inflation.
    Additional expansion keywords:
    interest banking

    Example 3:
    fb_terms=4
    Original query: how do solar panels generate electricity
    Retrieved passages:
    [1] Photovoltaic cells made of crystalline silicon absorb photons, exciting electrons across the semiconductor junction to produce direct current.
    Additional expansion keywords:
    photovoltaic silicon semiconductor photons
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
        If True, print progress metrics to stdout.
    """

    def __init__(
        self,
        indexer: BaseIndexer | None = None,
        fb_docs: int | None = None,
        fb_terms: int | None = None,
        fb_lambda: float | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        verbose: bool = False,
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

    # ── Dynamic prompt construction ───────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Build a system prompt requesting exactly fb_terms expansion terms."""
        return _SYSTEM_PROMPT_TEMPLATE.format(
            fb_terms=self.fb_terms,
        )

    def _build_user_prompt(
        self, original_query: str, passages: list[str]
    ) -> str:
        """Build the user-facing prompt with full context passages (untruncated)."""
        numbered = "\n".join(
            f"[{i + 1}] {p.strip()}" for i, p in enumerate(passages)
        )
        return (
            f"Original query: {original_query}\n\n"
            f"Retrieved passages:\n{numbered}\n\n"
            f"Additional {self.fb_terms} expansion keywords:"
        )

    # ── LLM interaction ───────────────────────────────────────────────────

    def _call_llm(self, original_query: str, passages: list[str]) -> str:
        """Call the LLM to reformulate the query, with retry on timeout."""
        system_prompt = self._build_system_prompt()
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
                text = response.choices[0].message.content or ""
                return text.strip()
            except (APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                logger.warning(
                    "LLM timeout/connection error (intento %d/%d): %s",
                    attempt, _DEFAULT_MAX_RETRIES, exc,
                )
                if self.verbose:
                    print(
                        f"  ⚠️ Timeout LLM (intento {attempt}/{_DEFAULT_MAX_RETRIES}). Reintentando..."
                    )
                time.sleep(2.0 * attempt)
            except Exception as exc:
                logger.error("Error inesperado en LLM: %s", exc)
                raise

        raise RuntimeError(
            f"LLM request falló tras {_DEFAULT_MAX_RETRIES} intentos"
        ) from last_exc

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
            The post-processed expanded query (plain text, no weights).
        """
        expanded, _, _ = self.expand_and_search(query)
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
        # 1. First-pass retrieval
        t0 = time.time()
        safe_query = sanitize_query_str(query)
        first_results = self._first_pass.search(safe_query)
        elapsed_first = time.time() - t0
        if self.verbose:
            print(f"  ⏱  First-pass retrieval : {elapsed_first:.3f}s")

        logger.info(
            "[1st pass] query='%s' | %d docs retrieved in %.3fs",
            safe_query, len(first_results), elapsed_first,
        )

        if first_results.empty:
            timings = {
                "time_first_pass": round(elapsed_first, 4),
                "time_text_fetch": 0.0,
                "time_llm": 0.0,
                "time_second_pass": 0.0,
                "n_terms_proposed": 0,
                "n_terms_kept": 0,
            }
            return safe_query, first_results, timings

        # 2. Get passage texts
        t0 = time.time()
        passages = self._indexer.get_texts(first_results)
        passages = [p for p in passages if p.strip()]
        elapsed_text = time.time() - t0
        if self.verbose:
            print(f"  ⏱  Text fetch          : {elapsed_text:.3f}s")

        if not passages:
            timings = {
                "time_first_pass": round(elapsed_first, 4),
                "time_text_fetch": round(elapsed_text, 4),
                "time_llm": 0.0,
                "time_second_pass": 0.0,
                "n_terms_proposed": 0,
                "n_terms_kept": 0,
            }
            return safe_query, first_results, timings

        # 3. Ask the LLM to reformulate the query
        t0 = time.time()
        raw_llm_output = self._call_llm(query, passages)
        elapsed_llm = time.time() - t0
        if self.verbose:
            print(f"  ⏱  Llamada al LLM      : {elapsed_llm:.3f}s")
            print(f"  📝 Salida cruda LLM     : \"{raw_llm_output}\"")

        logger.info(
            "[LLM] raw output (%.3fs): '%s'",
            elapsed_llm, raw_llm_output,
        )

        # Post-process LLM output
        expanded, n_proposed, n_kept = postprocess_expanded_query(
            original_query=query,
            raw_expanded_text=raw_llm_output,
            index=self._indexer._index,
            fb_terms=self.fb_terms,
        )

        if self.verbose:
            print(f"  📝 Query post-procesada : \"{expanded}\"")

        logger.info(
            "[Optimized] original='%s' → expanded='%s'",
            query, expanded,
        )

        # 4. Second-pass retrieval
        t0 = time.time()
        results = self._second_pass.search(expanded)
        elapsed_second = time.time() - t0
        if self.verbose:
            print(f"  ⏱  Second-pass retrieval: {elapsed_second:.3f}s")

        logger.info(
            "[2nd pass] expanded_query='%s' | %d docs retrieved in %.3fs",
            expanded, len(results), elapsed_second,
        )
        if not results.empty:
            top_docs = results.head(10)
            docs_summary = ", ".join(
                f"{r['docno']}({r['score']:.4f})"
                for _, r in top_docs.iterrows()
            )
            logger.info("[2nd pass] top results: %s", docs_summary)

        timings = {
            "time_first_pass": round(elapsed_first, 4),
            "time_text_fetch": round(elapsed_text, 4),
            "time_llm": round(elapsed_llm, 4),
            "time_second_pass": round(elapsed_second, 4),
            "n_terms_proposed": n_proposed,
            "n_terms_kept": n_kept,
        }

        return expanded, results, timings

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
            expanded, result, _ = self.expand_and_search(row["query"])
            result["qid"] = row["qid"]
            all_results.append(result)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

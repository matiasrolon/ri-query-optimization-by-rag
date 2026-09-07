# -*- coding: utf-8 -*-
"""
Shared utilities for query expansion post-processing.

Provides stemming, stopword removal and lexicon filtering using
Terrier's native Java term pipeline (PorterStemmer + Stopwords),
ensuring consistency between the indexing and expansion phases.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

import pyterrier as pt


# ── TerrierQL sanitisation ────────────────────────────────────────────────────

_TERRIER_SPECIAL_RE = re.compile(r'[/+\-!(){}[\]:^~\\\"\'.?*<>&|@#$%=;,]')


def sanitize_query(text: str) -> list[str]:
    """Sanitise raw text into a list of lowercase tokens safe for TerrierQL."""
    cleaned = _TERRIER_SPECIAL_RE.sub(" ", text)
    return [t for t in cleaned.lower().split() if t]


def sanitize_query_str(text: str) -> str:
    """Sanitise raw text and return as a single space-joined string."""
    return " ".join(sanitize_query(text))


# ── Terrier NLP components ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_stemmer():
    """Lazy-load Terrier's PorterStemmer (requires ``pt.init()``)."""
    from jnius import autoclass
    return autoclass("org.terrier.terms.PorterStemmer")()


@lru_cache(maxsize=1)
def _get_stopwords():
    """Lazy-load Terrier's Stopwords checker (requires ``pt.init()``)."""
    from jnius import autoclass
    return autoclass("org.terrier.terms.Stopwords")(None)


def stem_term(term: str) -> str:
    """Apply Terrier's PorterStemmer to a single lowercase term."""
    return _get_stemmer().stem(term)


def is_stopword(term: str) -> bool:
    """Return *True* if *term* is in Terrier's stopword list."""
    return _get_stopwords().isStopword(term)


def term_in_lexicon(index, stemmed_term: str) -> bool:
    """Check whether a **stemmed** term exists in the index lexicon."""
    return index.getLexicon().getLexiconEntry(stemmed_term) is not None


# ── Feedback-document term extraction ─────────────────────────────────────────

def extract_feedback_terms(texts: list[str]) -> set[str]:
    """
    Tokenise, stem and de-stop a list of document texts.

    Returns the set of stemmed terms found across all feedback documents.
    """
    terms: set[str] = set()
    for text in texts:
        for tok in sanitize_query(text):
            if not is_stopword(tok):
                terms.add(stem_term(tok))
    return terms


# ── Unified post-processing ──────────────────────────────────────────────────

def postprocess_expanded_query(
    original_query: str,
    raw_expanded_text: str,
    index,
    fb_terms: int,
    feedback_doc_terms: set[str] | None = None,
) -> str:
    """
    Unified post-processing pipeline for expanded queries.

    Applied identically to both PRF and RAG outputs so that the
    expanded query always satisfies the same linguistic constraints.

    Steps
    -----
    1. Tokenise and sanitise both original and expanded text.
    2. Remove stopwords (Terrier's list).
    3. Stem every token (PorterStemmer).
    4. Deduplicate against the original query (stemmed).
    5. Keep only terms present in the index lexicon.
    6. If *feedback_doc_terms* is provided, keep only terms that
       appear in the feedback documents.
    7. Truncate to *fb_terms* additional terms.
    8. Return ``original_safe + new_terms`` as **plain text** (no weights).

    Parameters
    ----------
    original_query : str
        The user's original query (raw text).
    raw_expanded_text : str
        The raw expanded text (from LLM or QE transformer).
    index
        A Terrier ``Index`` Java object (for lexicon look-ups).
    fb_terms : int
        Maximum number of *new* terms to add.
    feedback_doc_terms : set[str] | None
        Optional set of stemmed terms from the feedback documents.
        When provided, expansion terms not in this set are discarded.
    """
    # Stem original query tokens
    original_stemmed: set[str] = set()
    for tok in sanitize_query(original_query):
        if not is_stopword(tok):
            original_stemmed.add(stem_term(tok))

    # Process candidate expansion terms
    new_terms: list[str] = []
    seen: set[str] = set()

    for tok in sanitize_query(raw_expanded_text):
        if is_stopword(tok):
            continue
        stemmed = stem_term(tok)
        if stemmed in original_stemmed or stemmed in seen:
            continue
        if not term_in_lexicon(index, stemmed):
            continue
        if feedback_doc_terms is not None and stemmed not in feedback_doc_terms:
            continue
        seen.add(stemmed)
        new_terms.append(stemmed)
        if len(new_terms) >= fb_terms:
            break

    safe_original = sanitize_query_str(original_query)
    if new_terms:
        return safe_original + " " + " ".join(new_terms)
    return safe_original


# ── Hybrid LLM + DFR Re-ranking & Term Weighting ───────────────────

def rerank_and_weight_candidate_terms(
    original_query: str,
    raw_expanded_text: str,
    index,
    passages: list[str],
    fb_terms: int = 10,
    fb_lambda: float = 0.6,
    alpha: float = 0.5,
) -> str:
    """
    Hybrid Composite LLM Rank + DFR Term Weighting and Re-ranking.

    Combines the LLM's candidate confidence order with DFR Bo1 feedback scores:
      Score(t) = alpha * Score_LLM(t) + (1 - alpha) * Score_DFR(t)

    This ensures that novel LLM synonyms (which do not appear in the 1st-pass
    passages) are NOT discarded, while terms backed by the 1st-pass passages
    receive an additional statistical boost.
    """
    original_tokens = [
        tok for tok in sanitize_query(original_query) if not is_stopword(tok)
    ]
    original_stemmed_set = set(stem_term(tok) for tok in original_tokens)

    # 1. Extract candidates preserving LLM rank order
    candidates: list[str] = []
    seen: set[str] = set()

    for tok in sanitize_query(raw_expanded_text):
        if is_stopword(tok):
            continue
        stemmed = stem_term(tok)
        if stemmed in original_stemmed_set or stemmed in seen:
            continue
        if not term_in_lexicon(index, stemmed):
            continue
        seen.add(stemmed)
        candidates.append(stemmed)

    if not candidates:
        return sanitize_query_str(original_query)

    # 2. Count term frequency in feedback passages
    passages_stemmed: list[str] = []
    for p in passages:
        for tok in sanitize_query(p):
            if not is_stopword(tok):
                passages_stemmed.append(stem_term(tok))
    tf_fb = Counter(passages_stemmed)

    # 3. Compute DFR Bo1 raw weights
    stats = index.getCollectionStatistics()
    N = stats.getNumberOfTokens()
    lex = index.getLexicon()

    dfr_weights: dict[str, float] = {}
    for term in candidates:
        entry = lex.getLexiconEntry(term)
        if entry is None:
            continue
        F = entry.getFrequency()
        Pn = F / N
        freq = tf_fb[term]
        w = freq * math.log2((1.0 + Pn) / Pn) + math.log2(1.0 + Pn)
        dfr_weights[term] = w

    if not dfr_weights:
        return sanitize_query_str(original_query)

    # 4. Compute Composite Score (LLM Rank + DFR)
    num_c = len(candidates)
    max_dfr = max(dfr_weights.values()) if dfr_weights.values() else 1.0
    if max_dfr <= 0:
        max_dfr = 1.0

    composite_scores: dict[str, float] = {}
    for idx, term in enumerate(candidates):
        # LLM confidence rank score: 1.0 for first term, decaying down to ~0.05
        s_llm = 1.0 - (idx / float(num_c))
        # DFR score normalized: 0.0 to 1.0
        s_dfr = dfr_weights.get(term, 0.0) / max_dfr
        composite_scores[term] = alpha * s_llm + (1.0 - alpha) * s_dfr

    # 5. Sort candidates by Composite Score descending and pick top fb_terms
    sorted_candidates = sorted(
        composite_scores.keys(),
        key=lambda t: composite_scores[t],
        reverse=True,
    )[:fb_terms]

    max_comp = max(composite_scores[t] for t in sorted_candidates)
    if max_comp <= 0:
        max_comp = 1.0

    # 6. Format weighted TerrierQL query string
    weighted_parts = [f"{t}^1.0" for t in original_tokens]
    for t in sorted_candidates:
        norm_w = round(fb_lambda * (composite_scores[t] / max_comp), 4)
        if norm_w > 0:
            weighted_parts.append(f"{t}^{norm_w}")
        else:
            weighted_parts.append(f"{t}^{round(fb_lambda * 0.1, 4)}")

    return "applypipeline:off " + " ".join(weighted_parts)


# ── Alternativa 2: LLM Self-Weighting Post-Processing ─────────────────────────

def postprocess_llm_weighted_query(
    original_query: str,
    raw_expanded_text: str,
    index,
    fb_terms: int = 10,
    fb_lambda: float = 0.6,
) -> str:
    """
    LLM Self-Weighting Post-Processing.

    Parses expansion terms and self-assigned weights generated by the LLM
    (format: `word^weight` or `word:weight`), applies stemming, stop-word removal,
    lexicon validation, and formats the output into a weighted TerrierQL query string.
    """
    original_tokens = [
        tok for tok in sanitize_query(original_query) if not is_stopword(tok)
    ]
    original_stemmed_set = set(stem_term(tok) for tok in original_tokens)

    tokens = raw_expanded_text.replace(",", " ").split()
    candidates: list[tuple[str, float]] = []
    seen: set[str] = set()

    for token in tokens:
        if "^" in token:
            parts = token.split("^", 1)
            word, w_str = parts[0], parts[1]
        elif ":" in token:
            parts = token.split(":", 1)
            word, w_str = parts[0], parts[1]
        else:
            word, w_str = token, "0.6"

        cleaned_word = re.sub(r"[^a-zA-Z0-9_-]", "", word).strip()
        if not cleaned_word or is_stopword(cleaned_word):
            continue

        stemmed = stem_term(cleaned_word)
        if stemmed in original_stemmed_set or stemmed in seen:
            continue
        if not term_in_lexicon(index, stemmed):
            continue

        try:
            w = float(re.sub(r"[^\d.]", "", w_str))
            w = max(0.1, min(1.0, w))
        except ValueError:
            w = 0.6

        seen.add(stemmed)
        candidates.append((stemmed, round(w * fb_lambda, 4)))
        if len(candidates) >= fb_terms:
            break

    if not candidates:
        return sanitize_query_str(original_query)

    weighted_parts = [f"{t}^1.0" for t in original_tokens]
    for stemmed_t, w in candidates:
        weighted_parts.append(f"{stemmed_t}^{w}")

    return "applypipeline:off " + " ".join(weighted_parts)


# ── Option A: Fixed-Lambda Weighting Post-Processing ───────────────────────────

def postprocess_fixed_lambda_query(
    original_query: str,
    raw_expanded_text: str,
    index,
    fb_terms: int = 10,
    fb_lambda: float = 0.6,
) -> str:
    """
    Option A: Fixed-Lambda Weighting Post-Processing.

    The LLM outputs plain expansion keywords (no weights).
    Post-processing stems, deduplicates against original query terms,
    validates against the index lexicon, picks up to `fb_terms`, and assigns
    a uniform fixed weight `fb_lambda` to each expansion term in TerrierQL:
      applypipeline:off orig1^1.0 orig2^1.0 term1^fb_lambda term2^fb_lambda ...
    """
    original_tokens = [
        tok for tok in sanitize_query(original_query) if not is_stopword(tok)
    ]
    original_stemmed_set = set(stem_term(tok) for tok in original_tokens)

    new_terms: list[str] = []
    seen: set[str] = set()

    for tok in sanitize_query(raw_expanded_text):
        if is_stopword(tok):
            continue
        stemmed = stem_term(tok)
        if stemmed in original_stemmed_set or stemmed in seen:
            continue
        if not term_in_lexicon(index, stemmed):
            continue
        seen.add(stemmed)
        new_terms.append(stemmed)
        if len(new_terms) >= fb_terms:
            break

    if not new_terms:
        return sanitize_query_str(original_query)

    weighted_parts = [f"{t}^1.0" for t in original_tokens]
    for stemmed_t in new_terms:
        weighted_parts.append(f"{stemmed_t}^{fb_lambda}")

    return "applypipeline:off " + " ".join(weighted_parts)



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
    previously_rejected: set[str] | None = None,
) -> tuple[str, int, int, dict[str, int], set[str]]:
    """
    Unified post-processing pipeline for expanded queries.

    Applied identically to RAG outputs so that the expanded query always
    satisfies the same linguistic constraints.

    Steps
    -----
    1. Tokenise and sanitise both original and expanded text.
    2. Remove stopwords (Terrier's list), tracking rejections.
    3. Stem tokens (PorterStemmer).
    4. Deduplicate against the original query, previously seen candidate terms,
       and previously rejected terms, tracking rejections.
    5. Keep only terms present in the index lexicon, tracking rejections.
    6. If *feedback_doc_terms* is provided, keep only terms that
       appear in the feedback documents.
    7. Truncate to *fb_terms* additional terms.
    8. Return ``(expanded_query_str, n_terms_proposed, n_terms_kept, rejection_counts, rejected_tokens)``.

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
    previously_rejected : set[str] | None
        Set of tokens or stems previously rejected across earlier LLM attempts.

    Returns
    -------
    tuple[str, int, int, dict[str, int], set[str]]
        (expanded_query, n_terms_proposed, n_terms_kept, rejection_counts, rejected_tokens).
    """
    # Stem original query tokens
    original_tokens = sanitize_query(original_query)
    original_stemmed: set[str] = set()
    for tok in original_tokens:
        if not is_stopword(tok):
            original_stemmed.add(stem_term(tok))

    # Process candidate expansion terms
    new_terms: list[str] = []
    seen_in_proposal: set[str] = set()
    rejected_tokens: set[str] = set()

    n_rejected_stopword = 0
    n_rejected_duplicate = 0
    n_rejected_lexicon = 0
    n_rejected_truncated = 0

    candidate_tokens = sanitize_query(raw_expanded_text)

    for i, tok in enumerate(candidate_tokens):
        # If we have already collected fb_terms accepted terms,
        # any remaining tokens in the candidate list are discarded due to truncation
        # (they were not evaluated, NOT hallucinated).
        if len(new_terms) >= fb_terms:
            n_rejected_truncated += len(candidate_tokens) - i
            break

        # Reason 1: Stopword
        if is_stopword(tok):
            n_rejected_stopword += 1
            rejected_tokens.add(tok)
            continue

        stemmed = stem_term(tok)

        # Reason 2: Duplicate (in original query, in this response, or previously rejected)
        if (
            stemmed in original_stemmed
            or stemmed in seen_in_proposal
            or (previously_rejected is not None and (stemmed in previously_rejected or tok in previously_rejected))
        ):
            n_rejected_duplicate += 1
            rejected_tokens.add(tok)
            continue

        seen_in_proposal.add(stemmed)

        # Reason 3: Not in lexicon (true hallucination)
        if not term_in_lexicon(index, stemmed):
            n_rejected_lexicon += 1
            rejected_tokens.add(tok)
            continue

        if feedback_doc_terms is not None and stemmed not in feedback_doc_terms:
            n_rejected_lexicon += 1
            rejected_tokens.add(tok)
            continue

        # Accepted term
        new_terms.append(stemmed)

    safe_original = sanitize_query_str(original_query)
    n_terms_kept = len(new_terms)
    n_terms_proposed = (
        n_terms_kept
        + n_rejected_stopword
        + n_rejected_duplicate
        + n_rejected_lexicon
        + n_rejected_truncated
    )

    rejection_counts = {
        "stopword": n_rejected_stopword,
        "duplicate": n_rejected_duplicate,
        "lexicon": n_rejected_lexicon,
        "truncated": n_rejected_truncated,
    }

    if new_terms:
        expanded_query = safe_original + " " + " ".join(new_terms)
    else:
        expanded_query = safe_original

    return expanded_query, n_terms_proposed, n_terms_kept, rejection_counts, rejected_tokens




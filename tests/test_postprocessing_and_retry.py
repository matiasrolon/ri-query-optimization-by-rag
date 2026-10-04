# -*- coding: utf-8 -*-
"""
Unit tests for query expansion post-processing, rejection breakdown, and RAG retries.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyterrier as pt
if not pt.started():
    pt.init()

from query_expansion.utils import postprocess_expanded_query, sanitize_query_str
from query_expansion.rag import RAGExpander


class MockLexicon:
    def __init__(self, vocab: set[str]):
        self.vocab = vocab

    def getLexiconEntry(self, term: str):
        return "entry" if term in self.vocab else None


class MockIndex:
    def __init__(self, vocab: set[str]):
        self._lexicon = MockLexicon(vocab)

    def getLexicon(self):
        return self._lexicon


class TestPostprocessExpandedQuery(unittest.TestCase):
    def setUp(self):
        # Index vocabulary with stemmed words
        # "migrain", "ibuprofen", "analges", "headach", "tension"
        self.vocab = {"migrain", "ibuprofen", "analges", "headach", "tension"}
        self.mock_index = MockIndex(self.vocab)

    def test_rejection_reasons_breakdown(self):
        original_query = "treating tension headaches"
        # Proposed text has:
        # - "the": stopword -> rejected_stopword (1)
        # - "headaches": duplicate of query -> rejected_duplicate (1)
        # - "tension": duplicate of query -> rejected_duplicate (2)
        # - "migraine": valid in vocab -> kept (1)
        # - "migraine": duplicate in proposal -> rejected_duplicate (3)
        # - "ibuprofen": valid in vocab -> kept (2)
        # - "xyznonexistent": not in vocab -> rejected_lexicon (1)
        raw_text = "the headaches tension migraine migraine ibuprofen xyznonexistent"

        expanded, n_prop, n_kept, counts, rejected_tokens = postprocess_expanded_query(
            original_query=original_query,
            raw_expanded_text=raw_text,
            index=self.mock_index,
            fb_terms=5,
        )

        self.assertEqual(n_kept, 2)
        self.assertIn("migrain", expanded)
        self.assertIn("ibuprofen", expanded)
        self.assertEqual(counts["stopword"], 1)   # "the"
        self.assertEqual(counts["duplicate"], 3)  # "headaches", "tension", 2nd "migraine"
        self.assertEqual(counts["lexicon"], 1)    # "xyznonexistent"
        self.assertEqual(n_prop, n_kept + counts["stopword"] + counts["duplicate"] + counts["lexicon"])
        self.assertEqual(n_prop, 7)

    def test_previously_rejected_treated_as_duplicate(self):
        original_query = "treating tension"
        raw_text = "migraine"
        prev_rejected = {"migrain"}

        expanded, n_prop, n_kept, counts, rejected_tokens = postprocess_expanded_query(
            original_query=original_query,
            raw_expanded_text=raw_text,
            index=self.mock_index,
            fb_terms=5,
            previously_rejected=prev_rejected,
        )

        self.assertEqual(n_kept, 0)
        self.assertEqual(counts["duplicate"], 1)
        self.assertEqual(expanded, sanitize_query_str(original_query))

    def test_truncation_when_fb_terms_reached(self):
        original_query = "treating"
        # Proposed text has 3 valid vocabulary words ("migraine", "ibuprofen", "analgesic")
        # With fb_terms=1, "migraine" is kept, and "ibuprofen" & "analgesic" are truncated (not evaluated).
        raw_text = "migraine ibuprofen analges"

        expanded, n_prop, n_kept, counts, rejected_tokens = postprocess_expanded_query(
            original_query=original_query,
            raw_expanded_text=raw_text,
            index=self.mock_index,
            fb_terms=1,
        )

        self.assertEqual(n_kept, 1)
        self.assertEqual(counts["truncated"], 2)
        self.assertEqual(counts["lexicon"], 0)
        self.assertEqual(counts["stopword"], 0)
        self.assertEqual(counts["duplicate"], 0)
        self.assertEqual(n_prop, 3)
        self.assertEqual(n_prop, n_kept + counts["truncated"])
        self.assertIn("migrain", expanded)
        self.assertNotIn("ibuprofen", expanded)


class TestRAGExpanderRetry(unittest.TestCase):
    @patch("query_expansion.rag.get_indexer")
    def test_rag_expander_retries_and_succeeds(self, mock_get_indexer):
        mock_indexer = MagicMock()
        mock_indexer._index = MockIndex({"migrain", "analges"})
        mock_indexer.get_texts.return_value = ["passage about headaches and migraine"]
        
        first_pass_mock = MagicMock()
        first_pass_mock.search.return_value = MagicMock(empty=False)
        first_pass_mock.search.return_value.__len__.return_value = 1
        
        second_pass_mock = MagicMock()
        second_pass_mock.search.return_value = MagicMock(empty=False)
        second_pass_mock.search.return_value.__len__.return_value = 1
        second_pass_mock.search.return_value.head.return_value = MagicMock(iterrows=lambda: iter([]))
        
        mock_indexer.bm25_retriever.side_effect = [first_pass_mock, second_pass_mock]
        mock_get_indexer.return_value = mock_indexer

        expander = RAGExpander(indexer=mock_indexer, fb_terms=2)

        # Mock _call_llm:
        # Attempt 1: returns words that will be rejected (e.g. stopword "the", query word "tension")
        # Attempt 2: returns valid word "migraine"
        call_outputs = ["the tension", "migraine"]
        expander._call_llm = MagicMock(side_effect=call_outputs)

        expanded, results, timings = expander.expand_and_search("tension")

        self.assertTrue(timings["is_expanded"])
        self.assertEqual(timings["llm_attempts"], 2)
        self.assertEqual(timings["n_terms_kept"], 1)
        self.assertIn("migrain", expanded)

    @patch("query_expansion.rag.get_indexer")
    def test_rag_expander_fallback_after_3_attempts(self, mock_get_indexer):
        mock_indexer = MagicMock()
        mock_indexer._index = MockIndex({"migrain"})
        mock_indexer.get_texts.return_value = ["passage about headaches"]
        
        first_pass_mock = MagicMock()
        first_pass_mock.search.return_value = MagicMock(empty=False)
        first_pass_mock.search.return_value.__len__.return_value = 1
        
        second_pass_mock = MagicMock()
        second_pass_mock.search.return_value = MagicMock(empty=False)
        second_pass_mock.search.return_value.__len__.return_value = 1
        second_pass_mock.search.return_value.head.return_value = MagicMock(iterrows=lambda: iter([]))
        
        mock_indexer.bm25_retriever.side_effect = [first_pass_mock, second_pass_mock]
        mock_get_indexer.return_value = mock_indexer

        expander = RAGExpander(indexer=mock_indexer, fb_terms=2)

        # All 3 attempts return non-existent or stopword terms
        expander._call_llm = MagicMock(return_value="the for with")

        expanded, results, timings = expander.expand_and_search("treating tension")

        self.assertFalse(timings["is_expanded"])
        self.assertEqual(timings["llm_attempts"], 3)
        self.assertEqual(timings["n_terms_kept"], 0)
        self.assertEqual(expanded, "treating tension")


if __name__ == "__main__":
    unittest.main()

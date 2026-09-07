# -*- coding: utf-8 -*-
"""
Query Expansion & Retrieval package.

Provides retrieval and expansion strategies over an existing index:
  - BM25 Baseline: pure BM25 retrieval without query expansion.
  - PRF (Pseudo-Relevance Feedback): classical term-based expansion.
  - RAG: LLM-based query reformulation using retrieved context.
"""

from query_expansion.baseline import BM25Baseline
from query_expansion.prf import PRFExpander
from query_expansion.rag import RAGExpander

__all__ = ["BM25Baseline", "PRFExpander", "RAGExpander"]

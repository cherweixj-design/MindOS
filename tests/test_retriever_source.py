"""Tests for source tracking in Retriever."""

from typing import List

import pytest

from src.rag.base_embedding import BaseEmbedding
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.retriever import Retriever


class FakeEmbedding(BaseEmbedding):
    """Deterministic embedding for testing."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class TestRetrieverSource:
    """Tests for source passthrough in Retriever."""

    def test_retriever_search_returns_source_from_vector_store(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["年假政策"],
            vectors=[[1.0, 0.0, 0.0]],
            sources=["leave.md"],
        )
        retriever = Retriever(
            embedding=FakeEmbedding(),
            vector_store=store,
        )
        results = retriever.search(question="年假", top_k=1)
        assert len(results) == 1
        assert results[0].source == "leave.md"

    def test_retriever_preserves_source_after_min_score_filter(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["年假政策", "病假政策", "无关内容"],
            vectors=[[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
            sources=["leave.md", "sick.md", "other.md"],
        )
        retriever = Retriever(
            embedding=FakeEmbedding(),
            vector_store=store,
            min_score=0.5,
        )
        results = retriever.search(question="年假", top_k=3)
        assert len(results) == 2
        assert results[0].source == "leave.md"
        assert results[1].source == "sick.md"

    def test_retriever_min_score_zero_returns_all_with_source(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["年假政策", "病假政策"],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            sources=["leave.md", "sick.md"],
        )
        retriever = Retriever(
            embedding=FakeEmbedding(),
            vector_store=store,
            min_score=0.0,
        )
        results = retriever.search(question="年假", top_k=5)
        assert len(results) == 2
        assert results[0].source == "leave.md"
        assert results[1].source == "sick.md"

    def test_retriever_top_k_respects_source_integrity(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["政策A", "政策B", "政策C"],
            vectors=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
            sources=["a.md", "b.md", "c.md"],
        )
        retriever = Retriever(
            embedding=FakeEmbedding(),
            vector_store=store,
        )
        results = retriever.search(question="政策", top_k=2)
        assert len(results) == 2
        assert results[0].source == "a.md"
        assert results[1].source == "b.md"

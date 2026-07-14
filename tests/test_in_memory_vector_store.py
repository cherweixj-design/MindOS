import math

import pytest

from src.rag.in_memory_vector_store import InMemoryVectorStore


class TestInMemoryVectorStore:
    """Tests for the InMemoryVectorStore class."""

    # -- add() tests --

    def test_add_saves_texts_and_vectors(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["文本A", "文本B"],
            vectors=[[1.0, 0.0], [0.0, 1.0]],
            sources=["a.md", "b.md"],
        )
        assert store.texts == ["文本A", "文本B"]
        assert store.vectors == [[1.0, 0.0], [0.0, 1.0]]

    def test_add_saves_sources(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["年假政策", "病假政策"],
            vectors=[[1.0, 0.0], [0.0, 1.0]],
            sources=["leave.md", "sick.md"],
        )
        assert store.sources == ["leave.md", "sick.md"]

    def test_add_raises_error_on_mismatched_lengths(self):
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="same length"):
            store.add(
                texts=["文本A"],
                vectors=[[1.0, 0.0], [0.0, 1.0]],
                sources=["a.md"],
            )

    def test_add_raises_error_on_mismatched_sources(self):
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="same length"):
            store.add(
                texts=["文本A", "文本B"],
                vectors=[[1.0, 0.0], [0.0, 1.0]],
                sources=["a.md"],
            )

    def test_add_appends_to_existing_data(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["第一组"],
            vectors=[[1.0, 0.0]],
            sources=["a.md"],
        )
        store.add(
            texts=["第二组"],
            vectors=[[0.0, 1.0]],
            sources=["b.md"],
        )
        assert len(store.texts) == 2
        assert len(store.vectors) == 2
        assert len(store.sources) == 2

    # -- search() tests --

    def test_search_returns_results_sorted_by_score_descending(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["节假日政策", "病假政策", "办公地点"],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]],
            sources=["holiday.md", "sick.md", "office.md"],
        )
        results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].text == "节假日政策"

    def test_top_k_limits_results(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["A", "B", "C", "D", "E"],
            vectors=[[1, 0], [0, 1], [0.8, 0.2], [0.6, 0.4], [0.3, 0.7]],
            sources=["f1.md", "f2.md", "f3.md", "f4.md", "f5.md"],
        )
        results = store.search(query_vector=[1.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_search_empty_store_returns_empty_list(self):
        store = InMemoryVectorStore()
        results = store.search(query_vector=[1.0, 0.0], top_k=3)
        assert results == []

    def test_cosine_similarity_raises_on_dimension_mismatch(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["A"],
            vectors=[[1.0, 0.0, 0.0]],
            sources=["a.md"],
        )
        with pytest.raises(ValueError, match="same length"):
            store.search(query_vector=[1.0, 0.0], top_k=1)

    def test_zero_vector_does_not_cause_zero_division(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["零向量"],
            vectors=[[0.0, 0.0, 0.0]],
            sources=["zero.md"],
        )
        results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        result = results[0]
        assert result.text == "零向量"
        assert result.score == 0.0
        assert not math.isnan(result.score)

    def test_search_result_has_text_score_and_source_attributes(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["测试"],
            vectors=[[0.5, 0.5]],
            sources=["test.md"],
        )
        results = store.search(query_vector=[0.5, 0.5], top_k=1)
        assert len(results) == 1
        result = results[0]
        assert isinstance(result.text, str)
        assert isinstance(result.score, float)
        assert isinstance(result.source, str)
        assert result.source == "test.md"

    def test_search_preserves_source_after_sort(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["办公地点", "节假日政策"],
            vectors=[[0.5, 0.0, 0.5], [1.0, 0.0, 0.0]],
            sources=["office.md", "holiday.md"],
        )
        results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=2)
        assert results[0].text == "节假日政策"
        assert results[0].source == "holiday.md"
        assert results[1].text == "办公地点"
        assert results[1].source == "office.md"

    def test_perfect_match_returns_score_close_to_1(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["完全相同"],
            vectors=[[1.0, 0.0, 0.0]],
            sources=["a.md"],
        )
        results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=1)
        assert results[0].score == pytest.approx(1.0)

    def test_orthogonal_vectors_return_score_close_to_0(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["正交"],
            vectors=[[0.0, 1.0]],
            sources=["orth.md"],
        )
        results = store.search(query_vector=[1.0, 0.0], top_k=1)
        assert results[0].score == pytest.approx(0.0, abs=1e-10)

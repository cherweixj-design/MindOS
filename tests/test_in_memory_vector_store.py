import math

import pytest

from src.rag.in_memory_vector_store import InMemoryVectorStore


class TestInMemoryVectorStore:
    """Tests for the InMemoryVectorStore class."""

    def test_add_saves_texts_and_vectors(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["文本A", "文本B"],
            vectors=[[1.0, 0.0], [0.0, 1.0]],
        )
        assert store.texts == ["文本A", "文本B"]
        assert store.vectors == [[1.0, 0.0], [0.0, 1.0]]

    def test_add_raises_error_on_mismatched_lengths(self):
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="same length"):
            store.add(
                texts=["文本A"],
                vectors=[[1.0, 0.0], [0.0, 1.0]],
            )

    def test_search_returns_results_sorted_by_score_descending(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["节假日政策", "病假政策", "办公地点"],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]],
        )
        results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0][0] == "节假日政策"

    def test_top_k_limits_results(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["A", "B", "C", "D", "E"],
            vectors=[[1, 0], [0, 1], [0.8, 0.2], [0.6, 0.4], [0.3, 0.7]],
        )
        results = store.search(query_vector=[1.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_search_empty_store_returns_empty_list(self):
        store = InMemoryVectorStore()
        results = store.search(query_vector=[1.0, 0.0], top_k=3)
        assert results == []

    def test_cosine_similarity_raises_on_dimension_mismatch(self):
        store = InMemoryVectorStore()
        store.add(texts=["A"], vectors=[[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="same length"):
            store.search(query_vector=[1.0, 0.0], top_k=1)

    def test_zero_vector_does_not_cause_zero_division(self):
        store = InMemoryVectorStore()
        store.add(
            texts=["零向量"],
            vectors=[[0.0, 0.0, 0.0]],
        )
        results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        text, score = results[0]
        assert text == "零向量"
        assert score == 0.0
        assert not math.isnan(score)

    def test_search_returns_tuple_of_text_and_float_score(self):
        store = InMemoryVectorStore()
        store.add(texts=["测试"], vectors=[[0.5, 0.5]])
        results = store.search(query_vector=[0.5, 0.5], top_k=1)
        assert len(results) == 1
        text, score = results[0]
        assert isinstance(text, str)
        assert isinstance(score, float)

    def test_add_appends_to_existing_data(self):
        store = InMemoryVectorStore()
        store.add(texts=["第一组"], vectors=[[1.0, 0.0]])
        store.add(texts=["第二组"], vectors=[[0.0, 1.0]])
        assert len(store.texts) == 2
        assert len(store.vectors) == 2

    def test_perfect_match_returns_score_close_to_1(self):
        store = InMemoryVectorStore()
        store.add(texts=["完全相同"], vectors=[[1.0, 0.0, 0.0]])
        results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=1)
        assert results[0][1] == pytest.approx(1.0)

    def test_orthogonal_vectors_return_score_close_to_0(self):
        store = InMemoryVectorStore()
        store.add(texts=["正交"], vectors=[[0.0, 1.0]])
        results = store.search(query_vector=[1.0, 0.0], top_k=1)
        assert results[0][1] == pytest.approx(0.0, abs=1e-10)

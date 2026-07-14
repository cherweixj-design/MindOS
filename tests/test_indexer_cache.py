"""Tests for Indexer + KnowledgeCache integration."""

import json
from pathlib import Path
from typing import List

import pytest

from src.rag.base_embedding import BaseEmbedding
from src.rag.base_vector_store import BaseVectorStore
from src.rag.indexer import Indexer
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.knowledge_cache import KnowledgeCache
from src.rag.markdown_loader import MarkdownLoader
from src.rag.paragraph_splitter import ParagraphSplitter


class CountingFakeEmbedding(BaseEmbedding):
    """Embedding that counts calls and returns a fixed vector."""

    def __init__(self):
        self.call_count = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        return [[1.0, 0.0, 0.0] for _ in texts]


class RecordingVectorStore(BaseVectorStore):
    """Vector store that records add() calls without exposing internal lists."""

    def __init__(self):
        self._texts: List[str] = []
        self._vectors: List[List[float]] = []
        self._sources: List[str] = []

    def add(
        self,
        texts: List[str],
        vectors: List[List[float]],
        sources: List[str],
    ) -> None:
        self._texts.extend(texts)
        self._vectors.extend(vectors)
        self._sources.extend(sources)

    def search(self, query_vector, top_k=3):
        return []

    def get_all_data(self):
        """Return stored data for test assertions only."""
        return (
            list(self._texts),
            list(self._vectors),
            list(self._sources),
        )


def create_file(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def make_cache(tmp_path, cache_key="test-model"):
    return KnowledgeCache(
        cache_path=str(tmp_path / "cache.json"),
        cache_key=cache_key,
    )


def make_indexer(embedding, vector_store, cache=None):
    kwargs = dict(
        loader=MarkdownLoader(),
        splitter=ParagraphSplitter(),
        embedding=embedding,
        vector_store=vector_store,
    )
    if cache is not None:
        kwargs["cache"] = cache
    return Indexer(**kwargs)


class TestIndexerCache:
    """Tests for Indexer + KnowledgeCache integration."""

    def test_first_index_calls_embedding_and_creates_cache(self, tmp_path):
        create_file(tmp_path, "a.md", "内容A")
        create_file(tmp_path, "b.md", "内容B")
        embedding = CountingFakeEmbedding()
        vector_store = InMemoryVectorStore()
        cache = make_cache(tmp_path)
        indexer = make_indexer(embedding, vector_store, cache)
        count = indexer.index_directory(str(tmp_path))
        assert count == 2
        assert embedding.call_count > 0
        assert (tmp_path / "cache.json").exists()

    def test_second_index_reuses_cache(self, tmp_path):
        create_file(tmp_path, "a.md", "内容A")
        create_file(tmp_path, "b.md", "内容B")
        cache = make_cache(tmp_path)
        # First index to create cache
        indexer1 = make_indexer(CountingFakeEmbedding(), InMemoryVectorStore(), cache)
        first_count = indexer1.index_directory(str(tmp_path))
        # Second index with new objects
        embedding2 = CountingFakeEmbedding()
        indexer2 = make_indexer(embedding2, InMemoryVectorStore(), cache)
        second_count = indexer2.index_directory(str(tmp_path))
        assert second_count == first_count
        assert embedding2.call_count == 0

    def test_cached_data_matches_first_index(self, tmp_path):
        create_file(tmp_path, "a.md", "内容A\n\n内容A2")
        create_file(tmp_path, "b.md", "内容B")
        cache = make_cache(tmp_path)
        # First index
        store1 = RecordingVectorStore()
        indexer1 = make_indexer(CountingFakeEmbedding(), store1, cache)
        first_count = indexer1.index_directory(str(tmp_path))
        t1, v1, s1 = store1.get_all_data()
        # Second index with new store
        store2 = RecordingVectorStore()
        indexer2 = make_indexer(CountingFakeEmbedding(), store2, cache)
        second_count = indexer2.index_directory(str(tmp_path))
        t2, v2, s2 = store2.get_all_data()
        assert second_count == first_count
        assert t1 == t2
        assert v1 == v2
        assert s1 == s2

    def test_cache_preserves_source_filenames(self, tmp_path):
        create_file(tmp_path, "employee.md", "年假政策")
        create_file(tmp_path, "attendance.md", "考勤制度")
        cache = make_cache(tmp_path)
        indexer1 = make_indexer(CountingFakeEmbedding(), InMemoryVectorStore(), cache)
        indexer1.index_directory(str(tmp_path))
        store2 = InMemoryVectorStore()
        indexer2 = make_indexer(CountingFakeEmbedding(), store2, cache)
        indexer2.index_directory(str(tmp_path))
        assert store2.sources == ["attendance.md", "employee.md"]

    def test_modified_content_invalidates_cache(self, tmp_path):
        create_file(tmp_path, "a.md", "原始内容")
        cache = make_cache(tmp_path)
        indexer1 = make_indexer(CountingFakeEmbedding(), InMemoryVectorStore(), cache)
        indexer1.index_directory(str(tmp_path))
        # Modify file
        create_file(tmp_path, "a.md", "修改后的内容")
        embedding2 = CountingFakeEmbedding()
        store2 = InMemoryVectorStore()
        indexer2 = make_indexer(embedding2, store2, cache)
        indexer2.index_directory(str(tmp_path))
        assert embedding2.call_count > 0
        assert "修改后的内容" in " ".join(store2.texts)

    def test_corrupt_cache_rebuilds_index(self, tmp_path):
        create_file(tmp_path, "a.md", "内容")
        cache = make_cache(tmp_path)
        indexer1 = make_indexer(CountingFakeEmbedding(), InMemoryVectorStore(), cache)
        indexer1.index_directory(str(tmp_path))
        # Corrupt cache file
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("{corrupt", encoding="utf-8")
        embedding2 = CountingFakeEmbedding()
        store2 = InMemoryVectorStore()
        indexer2 = make_indexer(embedding2, store2, cache)
        count = indexer2.index_directory(str(tmp_path))
        assert count == 1
        assert embedding2.call_count > 0
        # Cache should be overwritten with valid data
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["indexed_file_count"] == 1

    def test_empty_directory_creates_empty_cache(self, tmp_path):
        cache = make_cache(tmp_path)
        indexer = make_indexer(CountingFakeEmbedding(), InMemoryVectorStore(), cache)
        count = indexer.index_directory(str(tmp_path))
        assert count == 0
        assert (tmp_path / "cache.json").exists()
        cache = KnowledgeCache(
            cache_path=str(tmp_path / "cache.json"),
            cache_key="test-model",
        )
        result = cache.load_if_valid(str(tmp_path))
        assert result is not None
        assert result.indexed_file_count == 0
        assert result.texts == []
        assert result.vectors == []
        assert result.sources == []

    def test_indexer_without_cache_still_works(self, tmp_path):
        create_file(tmp_path, "a.md", "内容")
        embedding = CountingFakeEmbedding()
        vector_store = InMemoryVectorStore()
        indexer = make_indexer(embedding, vector_store)
        count = indexer.index_directory(str(tmp_path))
        assert count == 1
        assert embedding.call_count > 0

    def test_recording_vector_store_works_with_cache(self, tmp_path):
        create_file(tmp_path, "a.md", "内容A")
        create_file(tmp_path, "b.md", "内容B")
        cache = make_cache(tmp_path)
        # First index with RecordingVectorStore
        store1 = RecordingVectorStore()
        indexer1 = make_indexer(CountingFakeEmbedding(), store1, cache)
        first_count = indexer1.index_directory(str(tmp_path))
        t1, v1, s1 = store1.get_all_data()
        # Second index with new RecordingVectorStore
        store2 = RecordingVectorStore()
        embedding2 = CountingFakeEmbedding()
        indexer2 = make_indexer(embedding2, store2, cache)
        second_count = indexer2.index_directory(str(tmp_path))
        assert second_count == first_count
        assert embedding2.call_count == 0
        t2, v2, s2 = store2.get_all_data()
        assert len(t2) > 0
        assert t1 == t2
        assert v1 == v2
        assert s1 == s2

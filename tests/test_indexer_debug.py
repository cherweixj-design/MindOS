"""Tests for Indexer cache debug logs."""

from pathlib import Path
from typing import List

import pytest

from src.rag.base_embedding import BaseEmbedding
from src.rag.base_loader import BaseLoader
from src.rag.base_splitter import BaseSplitter
from src.rag.indexer import Indexer
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.knowledge_cache import CachedIndex


class FakeHitCache:
    """Cache that always hits."""

    def load_if_valid(self, directory_path: str):
        return CachedIndex(
            indexed_file_count=2,
            texts=["文本A", "文本B"],
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            sources=["a.md", "b.md"],
        )

    def compute_fingerprint(self, directory_path: str) -> str:
        return "fake-fingerprint"

    def save(self, **kwargs):
        pass


class FakeMissCache:
    """Cache that always misses."""

    def load_if_valid(self, directory_path: str):
        return None

    def compute_fingerprint(self, directory_path: str) -> str:
        return "fake-fingerprint"

    def save(self, **kwargs):
        pass


class FakeLoader(BaseLoader):
    """Loader returning fixed content."""

    def load(self, file_path: str) -> str:
        return "一些知识内容"


class FakeSplitter(BaseSplitter):
    """Splitter returning a single chunk."""

    def split(self, text: str) -> List[str]:
        return ["一些知识内容"]


class FakeEmbedding(BaseEmbedding):
    """Embedding returning a fixed vector."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def make_indexer(cache=None, debug=None):
    kwargs = dict(
        loader=FakeLoader(),
        splitter=FakeSplitter(),
        embedding=FakeEmbedding(),
        vector_store=InMemoryVectorStore(),
        cache=cache,
    )
    if debug is not None:
        kwargs["debug"] = debug
    return Indexer(**kwargs)


def create_file(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


class TestIndexerCacheDebug:
    """Tests for Indexer cache debug logs."""

    def test_debug_cache_hit(self, tmp_path, capsys):
        indexer = make_indexer(cache=FakeHitCache(), debug=True)
        indexer.index_directory(str(tmp_path))
        captured = capsys.readouterr()
        assert "[DEBUG] 知识索引缓存：命中。" in captured.out

    def test_debug_cache_hit_no_miss(self, tmp_path, capsys):
        indexer = make_indexer(cache=FakeHitCache(), debug=True)
        indexer.index_directory(str(tmp_path))
        captured = capsys.readouterr()
        assert "[DEBUG] 知识索引缓存：未命中" not in captured.out

    def test_debug_cache_miss(self, tmp_path, capsys):
        create_file(tmp_path, "a.md", "内容")
        indexer = make_indexer(cache=FakeMissCache(), debug=True)
        indexer.index_directory(str(tmp_path))
        captured = capsys.readouterr()
        assert "[DEBUG] 知识索引缓存：未命中，正在重新构建。" in captured.out

    def test_debug_cache_miss_no_hit(self, tmp_path, capsys):
        create_file(tmp_path, "a.md", "内容")
        indexer = make_indexer(cache=FakeMissCache(), debug=True)
        indexer.index_directory(str(tmp_path))
        captured = capsys.readouterr()
        assert "[DEBUG] 知识索引缓存：命中。" not in captured.out

    def test_debug_false_hit(self, tmp_path, capsys):
        indexer = make_indexer(cache=FakeHitCache())
        indexer.index_directory(str(tmp_path))
        captured = capsys.readouterr()
        assert "[DEBUG] 知识索引缓存：命中。" not in captured.out

    def test_debug_false_miss(self, tmp_path, capsys):
        create_file(tmp_path, "a.md", "内容")
        indexer = make_indexer(cache=FakeMissCache())
        indexer.index_directory(str(tmp_path))
        captured = capsys.readouterr()
        assert "[DEBUG] 知识索引缓存：未命中" not in captured.out

    def test_no_cache_no_logs(self, tmp_path, capsys):
        indexer = make_indexer(cache=None)
        indexer.index_directory(str(tmp_path))
        captured = capsys.readouterr()
        assert "[DEBUG] 知识索引缓存：命中。" not in captured.out
        assert "[DEBUG] 知识索引缓存：未命中" not in captured.out

    def test_logs_no_sensitive_data(self, tmp_path, capsys):
        indexer = make_indexer(cache=FakeHitCache(), debug=True)
        indexer.index_directory(str(tmp_path))
        captured = capsys.readouterr()
        assert "fake-fingerprint" not in captured.out
        assert "[0.1, 0.2]" not in captured.out
        assert "[0.3, 0.4]" not in captured.out

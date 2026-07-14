"""Tests for source tracking in Indexer."""

from pathlib import Path
from typing import List

import pytest

from src.rag.base_embedding import BaseEmbedding
from src.rag.indexer import Indexer
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.markdown_loader import MarkdownLoader
from src.rag.paragraph_splitter import ParagraphSplitter


class FakeEmbedding(BaseEmbedding):
    """Deterministic embedding for testing."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def make_indexer() -> Indexer:
    """Create an Indexer with fake embedding."""
    return Indexer(
        loader=MarkdownLoader(),
        splitter=ParagraphSplitter(),
        embedding=FakeEmbedding(),
        vector_store=InMemoryVectorStore(),
    )


def create_md_file(directory: Path, name: str, content: str) -> Path:
    """Write a file into the given directory."""
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


class TestIndexerSource:
    """Tests for source tracking in Indexer."""

    def test_index_stores_filename_as_source(self, tmp_path):
        """index() should store the filename as the source of each chunk."""
        file_path = create_md_file(tmp_path, "policy.md", "内容A\n\n内容B")
        indexer = make_indexer()
        indexer.index(str(file_path))
        assert len(indexer.vector_store.sources) == 2
        assert indexer.vector_store.sources[0] == "policy.md"
        assert indexer.vector_store.sources[1] == "policy.md"

    def test_index_directory_stores_correct_source_for_each_file(self, tmp_path):
        """Chunks from different files should have different source values."""
        create_md_file(tmp_path, "leave.md", "年假政策")
        create_md_file(tmp_path, "sick.md", "病假政策")
        indexer = make_indexer()
        indexer.index_directory(str(tmp_path))
        assert len(indexer.vector_store.sources) == 2
        assert indexer.vector_store.sources[0] == "leave.md"
        assert indexer.vector_store.sources[1] == "sick.md"

    def test_index_directory_source_matches_text_order(self, tmp_path):
        """Source values should be in the same order as their corresponding texts."""
        create_md_file(tmp_path, "b.md", "第二份文件")
        create_md_file(tmp_path, "a.md", "第一份文件")
        indexer = make_indexer()
        indexer.index_directory(str(tmp_path))
        assert indexer.vector_store.texts[0] == "第一份文件"
        assert indexer.vector_store.sources[0] == "a.md"
        assert indexer.vector_store.texts[1] == "第二份文件"
        assert indexer.vector_store.sources[1] == "b.md"

    def test_index_directory_multiple_chunks_same_source(self, tmp_path):
        """Multiple chunks from the same file should all have the same source."""
        create_md_file(tmp_path, "annual.md", "年假天数\n\n年假结转")
        indexer = make_indexer()
        indexer.index_directory(str(tmp_path))
        assert len(indexer.vector_store.sources) == 2
        assert indexer.vector_store.sources == ["annual.md", "annual.md"]

    def test_index_single_file_source(self):
        """index() on the real knowledge file should store employee.md as source."""
        indexer = make_indexer()
        indexer.index("knowledge/employee.md")
        assert len(indexer.vector_store.sources) > 0
        for source in indexer.vector_store.sources:
            assert source == "employee.md"

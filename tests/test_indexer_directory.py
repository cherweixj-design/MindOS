"""Tests for Indexer.index_directory()."""

from pathlib import Path
from typing import List

import pytest

from src.rag.base_embedding import BaseEmbedding
from src.rag.indexer import Indexer
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.markdown_loader import MarkdownLoader
from src.rag.paragraph_splitter import ParagraphSplitter


class FakeEmbedding(BaseEmbedding):
    """Deterministic embedding that returns a fixed vector per text."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def make_indexer() -> Indexer:
    """Create an Indexer wired with FakeEmbedding for testing."""
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


class TestIndexDirectory:
    """Tests for index_directory()."""

    def test_returns_count_of_indexed_files(self, tmp_path):
        """index_directory returns the number of successfully indexed files."""
        create_md_file(tmp_path, "a.md", "内容A")
        create_md_file(tmp_path, "b.md", "内容B")
        indexer = make_indexer()
        count = indexer.index_directory(str(tmp_path))
        assert count == 2

    def test_only_processes_md_files(self, tmp_path):
        """Non-.md files are ignored."""
        create_md_file(tmp_path, "policy.md", "年假政策")
        create_md_file(tmp_path, "notes.txt", "一些笔记")
        create_md_file(tmp_path, "data.json", '{"key": "value"}')
        indexer = make_indexer()
        count = indexer.index_directory(str(tmp_path))
        assert count == 1
        assert len(indexer.vector_store.texts) >= 1
        all_text = " ".join(indexer.vector_store.texts)
        assert "年假政策" in all_text
        assert "一些笔记" not in all_text

    def test_processes_md_files_in_sorted_order(self, tmp_path):
        """Files are processed by alphabetical filename order."""
        create_md_file(tmp_path, "c.md", "内容C")
        create_md_file(tmp_path, "a.md", "内容A")
        create_md_file(tmp_path, "b.md", "内容B")
        indexer = make_indexer()
        indexer.index_directory(str(tmp_path))
        assert len(indexer.vector_store.texts) == 3
        assert indexer.vector_store.texts[0] == "内容A"
        assert indexer.vector_store.texts[1] == "内容B"
        assert indexer.vector_store.texts[2] == "内容C"

    def test_skips_empty_md_file_and_continues(self, tmp_path):
        """An empty .md file is skipped without stopping the whole process."""
        create_md_file(tmp_path, "valid.md", "有效内容")
        create_md_file(tmp_path, "empty.md", "")
        indexer = make_indexer()
        count = indexer.index_directory(str(tmp_path))
        assert count == 1
        all_text = " ".join(indexer.vector_store.texts)
        assert "有效内容" in all_text

    def test_skips_whitespace_only_md_file(self, tmp_path):
        """A .md file with only whitespace is skipped."""
        create_md_file(tmp_path, "valid.md", "有效内容")
        create_md_file(tmp_path, "blank.md", "   \n  \t  ")
        indexer = make_indexer()
        count = indexer.index_directory(str(tmp_path))
        assert count == 1

    def test_empty_directory_returns_zero(self, tmp_path):
        """A directory with no files returns 0."""
        indexer = make_indexer()
        count = indexer.index_directory(str(tmp_path))
        assert count == 0

    def test_directory_with_only_non_md_files_returns_zero(self, tmp_path):
        """A directory with only non-.md files returns 0."""
        create_md_file(tmp_path, "readme.txt", "README")
        create_md_file(tmp_path, "data.json", "{}")
        indexer = make_indexer()
        count = indexer.index_directory(str(tmp_path))
        assert count == 0

    def test_nonexistent_directory_raises_file_not_found(self, tmp_path):
        """A path that does not exist raises FileNotFoundError."""
        bad_path = str(tmp_path / "no_such_dir")
        indexer = make_indexer()
        with pytest.raises(FileNotFoundError):
            indexer.index_directory(bad_path)

    def test_file_instead_of_directory_raises_not_a_directory(self, tmp_path):
        """A path that is a regular file raises NotADirectoryError."""
        file_path = create_md_file(tmp_path, "file.md", "内容")
        indexer = make_indexer()
        with pytest.raises(NotADirectoryError):
            indexer.index_directory(str(file_path))

    def test_indexed_content_is_stored_in_vector_store(self, tmp_path):
        """Chunks from indexed files are actually stored and retrievable."""
        create_md_file(tmp_path, "annual.md", "员工每年有15天年假。\n\n未休年假可以结转。")
        create_md_file(tmp_path, "sick.md", "病假需要提交相关证明。")
        indexer = make_indexer()
        count = indexer.index_directory(str(tmp_path))
        assert count == 2
        # annual.md produces 2 chunks, sick.md produces 1
        assert len(indexer.vector_store.texts) == 3
        all_text = " ".join(indexer.vector_store.texts)
        assert "年假" in all_text
        assert "病假" in all_text

    def test_returns_int(self, tmp_path):
        """Return value is an integer."""
        indexer = make_indexer()
        result = indexer.index_directory(str(tmp_path))
        assert isinstance(result, int)

    def test_file_in_subdirectory_is_not_indexed(self, tmp_path):
        """Only files in the immediate directory are indexed, not subdirectories."""
        sub = tmp_path / "sub"
        sub.mkdir()
        create_md_file(sub, "deep.md", "深层文件")
        create_md_file(tmp_path, "top.md", "顶层文件")
        indexer = make_indexer()
        count = indexer.index_directory(str(tmp_path))
        assert count == 1
        assert len(indexer.vector_store.texts) == 1
        assert indexer.vector_store.texts[0] == "顶层文件"

import pytest

from src.rag.paragraph_splitter import ParagraphSplitter


class TestParagraphSplitter:
    """Tests for the ParagraphSplitter class."""

    def test_empty_string_returns_empty_list(self):
        splitter = ParagraphSplitter()
        result = splitter.split("")
        assert result == []

    def test_single_paragraph(self):
        splitter = ParagraphSplitter()
        text = "这是一段话。"
        result = splitter.split(text)
        assert result == ["这是一段话。"]

    def test_multiple_paragraphs(self):
        splitter = ParagraphSplitter()
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        result = splitter.split(text)
        assert result == ["第一段内容。", "第二段内容。", "第三段内容。"]

    def test_consecutive_blank_lines_produce_no_empty_chunks(self):
        splitter = ParagraphSplitter()
        text = "第一段。\n\n\n\n第二段。"
        result = splitter.split(text)
        assert result == ["第一段。", "第二段。"]

    def test_chunks_are_stripped(self):
        splitter = ParagraphSplitter()
        text = "  带空格段落  \n\n  \t另一个段落  "
        result = splitter.split(text)
        assert result == ["带空格段落", "另一个段落"]

    def test_only_whitespace_paragraphs_are_skipped(self):
        splitter = ParagraphSplitter()
        text = "第一段。\n\n   \n\n第三段。"
        result = splitter.split(text)
        assert result == ["第一段。", "第三段。"]

    def test_newline_only_text_returns_empty_list(self):
        splitter = ParagraphSplitter()
        result = splitter.split("\n\n\n")
        assert result == []

    def test_text_with_no_double_newline_returns_single_chunk(self):
        splitter = ParagraphSplitter()
        text = "这是一行。\n这是同一段。\n还是同一段。"
        result = splitter.split(text)
        assert result == ["这是一行。\n这是同一段。\n还是同一段。"]

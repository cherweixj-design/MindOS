"""Tests for MindOS Query Rewriting debug logs."""

from typing import Dict, List

import pytest

from src.llm.base import BaseLLM
from src.memory.memory import Memory
from src.mindos import MindOS
from src.prompt.prompt_builder import PromptBuilder
from src.query.base_query_rewriter import BaseQueryRewriter, QueryRewriteError
from src.rag.base_retriever import BaseRetriever
from src.rag.retrieval_result import RetrievalResult


class RecordingQueryRewriter(BaseQueryRewriter):
    """Rewriter that returns a fixed result."""

    def __init__(self, rewritten: str = "员工的年假没休完怎么办？"):
        self.rewritten = rewritten

    def rewrite(self, question: str, history: list) -> str:
        return self.rewritten


class ErrorQueryRewriter(BaseQueryRewriter):
    """Rewriter that raises QueryRewriteError."""

    def rewrite(self, question: str, history: list) -> str:
        raise QueryRewriteError("Rewrite failed")


class FakeLLM(BaseLLM):
    """Fake LLM returning a fixed answer."""

    def __init__(self, answer: str = "回答。"):
        self.answer = answer

    def chat(self, messages: list) -> str:
        return self.answer


class FakeRetriever(BaseRetriever):
    """Fake retriever returning pre-defined results."""

    def __init__(self, results: List[RetrievalResult] = None):
        self._results = results or []

    def search(self, question: str, top_k: int = 3) -> list:
        return self._results[:top_k]


ORIGINAL = "没休完怎么办？"
REWRITTEN = "员工的年假没休完怎么办？"


class TestMindosRewriteDebug:
    """Tests for Query Rewriting debug logs."""

    def test_debug_shows_original_question(self, capsys):
        mindos = MindOS(
            llm=FakeLLM(),
            memory=Memory(),
            retriever=FakeRetriever(),
            prompt_builder=PromptBuilder(),
            rewriter=RecordingQueryRewriter(REWRITTEN),
            top_k=3,
            debug=True,
        )
        mindos.chat(ORIGINAL)
        captured = capsys.readouterr()
        assert f"[DEBUG] 原始问题：{ORIGINAL}" in captured.out

    def test_debug_shows_rewritten_question(self, capsys):
        mindos = MindOS(
            llm=FakeLLM(),
            memory=Memory(),
            retriever=FakeRetriever(),
            prompt_builder=PromptBuilder(),
            rewriter=RecordingQueryRewriter(REWRITTEN),
            top_k=3,
            debug=True,
        )
        mindos.chat(ORIGINAL)
        captured = capsys.readouterr()
        assert f"[DEBUG] 检索问题：{REWRITTEN}" in captured.out

    def test_debug_shows_both_when_same(self, capsys):
        q = "年假多少天？"
        mindos = MindOS(
            llm=FakeLLM(),
            memory=Memory(),
            retriever=FakeRetriever(),
            prompt_builder=PromptBuilder(),
            rewriter=RecordingQueryRewriter(q),
            top_k=3,
            debug=True,
        )
        mindos.chat(q)
        captured = capsys.readouterr()
        assert f"[DEBUG] 原始问题：{q}" in captured.out
        assert f"[DEBUG] 检索问题：{q}" in captured.out

    def test_rewrite_error_fallback_log(self, capsys):
        mindos = MindOS(
            llm=FakeLLM(),
            memory=Memory(),
            retriever=FakeRetriever(),
            prompt_builder=PromptBuilder(),
            rewriter=ErrorQueryRewriter(),
            top_k=3,
            debug=True,
        )
        mindos.chat(ORIGINAL)
        captured = capsys.readouterr()
        assert "[DEBUG] 问题改写失败，已使用原始问题检索。" in captured.out

    def test_rewrite_error_shows_original(self, capsys):
        mindos = MindOS(
            llm=FakeLLM(),
            memory=Memory(),
            retriever=FakeRetriever(),
            prompt_builder=PromptBuilder(),
            rewriter=ErrorQueryRewriter(),
            top_k=3,
            debug=True,
        )
        mindos.chat(ORIGINAL)
        captured = capsys.readouterr()
        assert f"[DEBUG] 原始问题：{ORIGINAL}" in captured.out
        assert f"[DEBUG] 检索问题：{ORIGINAL}" in captured.out

    def test_debug_false_no_rewrite_logs(self, capsys):
        mindos = MindOS(
            llm=FakeLLM(),
            memory=Memory(),
            retriever=FakeRetriever(),
            prompt_builder=PromptBuilder(),
            rewriter=RecordingQueryRewriter(REWRITTEN),
            top_k=3,
            debug=False,
        )
        mindos.chat(ORIGINAL)
        captured = capsys.readouterr()
        assert "[DEBUG] 原始问题：" not in captured.out
        assert "[DEBUG] 检索问题：" not in captured.out
        assert "[DEBUG] 问题改写失败" not in captured.out

    def test_rewriter_none_no_rewrite_logs(self, capsys):
        mindos = MindOS(
            llm=FakeLLM(),
            memory=Memory(),
            retriever=FakeRetriever(),
            prompt_builder=PromptBuilder(),
            rewriter=None,
            top_k=3,
            debug=True,
        )
        mindos.chat(ORIGINAL)
        captured = capsys.readouterr()
        assert "[DEBUG] 原始问题：" not in captured.out
        assert "[DEBUG] 检索问题：" not in captured.out

    def test_debug_does_not_leak_history(self, capsys):
        memory = Memory()
        memory.add("user", "这是不应出现在调试日志中的历史内容")
        memory.add("assistant", "一些回答。")
        mindos = MindOS(
            llm=FakeLLM(),
            memory=memory,
            retriever=FakeRetriever(),
            prompt_builder=PromptBuilder(),
            rewriter=RecordingQueryRewriter(REWRITTEN),
            top_k=3,
            debug=True,
        )
        mindos.chat(ORIGINAL)
        captured = capsys.readouterr()
        assert "这是不应出现在调试日志中的历史内容" not in captured.out

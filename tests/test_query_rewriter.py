"""Tests for Query Rewriter."""

from typing import Dict, List

import pytest


def _get_base():
    """Dynamically import base_query_rewriter module."""
    from importlib import import_module
    return import_module("src.query.base_query_rewriter")


def _get_llm_rewriter():
    """Dynamically import llm_query_rewriter module."""
    from importlib import import_module
    return import_module("src.query.llm_query_rewriter")


class RecordingFakeLLM:
    """Fake LLM that records calls and returns a fixed answer."""

    def __init__(self, answer: str = "改写后的问题"):
        self.answer = answer
        self.call_count = 0
        self.last_messages = None

    def chat(self, messages: List[Dict[str, str]]) -> str:
        self.call_count += 1
        self.last_messages = messages
        return self.answer


class ErrorFakeLLM:
    """Fake LLM that raises an exception."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise ConnectionError("API failed")


class TestQueryRewriteError:
    """Tests for QueryRewriteError exception."""

    def test_query_rewrite_error_is_runtime_error(self):
        base = _get_base()
        assert issubclass(base.QueryRewriteError, RuntimeError)


class TestBaseQueryRewriter:
    """Tests for BaseQueryRewriter abstract class."""

    def test_base_rewrite_raises_not_implemented(self):
        base = _get_base()
        rewriter = base.BaseQueryRewriter()
        with pytest.raises(NotImplementedError):
            rewriter.rewrite("问题", [])


class TestLLMQueryRewriter:
    """Tests for LLMQueryRewriter."""

    def test_no_history_returns_original_question(self):
        llm_mod = _get_llm_rewriter()
        llm = RecordingFakeLLM()
        rewriter = llm_mod.LLMQueryRewriter(llm=llm)
        result = rewriter.rewrite("员工年假多少天？", [])
        assert result == "员工年假多少天？"

    def test_no_history_does_not_call_llm(self):
        llm_mod = _get_llm_rewriter()
        llm = RecordingFakeLLM()
        rewriter = llm_mod.LLMQueryRewriter(llm=llm)
        rewriter.rewrite("员工年假多少天？", [])
        assert llm.call_count == 0

    def test_with_history_calls_llm_once(self):
        llm_mod = _get_llm_rewriter()
        llm = RecordingFakeLLM()
        rewriter = llm_mod.LLMQueryRewriter(llm=llm)
        history = [
            {"role": "user", "content": "年假多少天？"},
            {"role": "assistant", "content": "15天。"},
        ]
        rewriter.rewrite("没休完怎么办？", history)
        assert llm.call_count == 1

    def test_prompt_contains_history(self):
        llm_mod = _get_llm_rewriter()
        llm = RecordingFakeLLM()
        rewriter = llm_mod.LLMQueryRewriter(llm=llm)
        history = [
            {"role": "user", "content": "年假多少天？"},
            {"role": "assistant", "content": "15天。"},
        ]
        rewriter.rewrite("没休完怎么办？", history)
        messages = llm.last_messages
        assert any("年假多少天？" in str(m) for m in messages)
        assert any("15天。" in str(m) for m in messages)

    def test_prompt_contains_current_question(self):
        llm_mod = _get_llm_rewriter()
        llm = RecordingFakeLLM()
        rewriter = llm_mod.LLMQueryRewriter(llm=llm)
        history = [
            {"role": "user", "content": "年假多少天？"},
            {"role": "assistant", "content": "15天。"},
        ]
        rewriter.rewrite("没休完怎么办？", history)
        messages = llm.last_messages
        assert any("没休完怎么办？" in str(m) for m in messages)

    def test_system_prompt_requires_rewrite_only(self):
        llm_mod = _get_llm_rewriter()
        llm = RecordingFakeLLM()
        rewriter = llm_mod.LLMQueryRewriter(llm=llm)
        history = [
            {"role": "user", "content": "年假多少天？"},
            {"role": "assistant", "content": "15天。"},
        ]
        rewriter.rewrite("没休完怎么办？", history)
        messages = llm.last_messages
        assert messages[0]["role"] == "system"
        content = messages[0]["content"]
        assert "改写" in content
        assert "不要回答" in content
        assert "只输出" in content or "只返回" in content

    def test_return_value_is_stripped(self):
        llm_mod = _get_llm_rewriter()
        llm = RecordingFakeLLM(answer="  员工的年假没休完怎么办？  ")
        rewriter = llm_mod.LLMQueryRewriter(llm=llm)
        history = [
            {"role": "user", "content": "年假多少天？"},
            {"role": "assistant", "content": "15天。"},
        ]
        result = rewriter.rewrite("没休完怎么办？", history)
        assert result == "员工的年假没休完怎么办？"

    def test_blank_llm_output_falls_back_to_original(self):
        llm_mod = _get_llm_rewriter()
        llm = RecordingFakeLLM(answer="  ")
        rewriter = llm_mod.LLMQueryRewriter(llm=llm)
        history = [
            {"role": "user", "content": "年假多少天？"},
            {"role": "assistant", "content": "15天。"},
        ]
        result = rewriter.rewrite("没休完怎么办？", history)
        assert result == "没休完怎么办？"

    def test_llm_exception_raises_query_rewrite_error(self):
        base = _get_base()
        llm_mod = _get_llm_rewriter()
        llm = ErrorFakeLLM()
        rewriter = llm_mod.LLMQueryRewriter(llm=llm)
        history = [
            {"role": "user", "content": "年假多少天？"},
            {"role": "assistant", "content": "15天。"},
        ]
        with pytest.raises(base.QueryRewriteError) as exc_info:
            rewriter.rewrite("没休完怎么办？", history)
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    def test_history_is_not_mutated(self):
        llm_mod = _get_llm_rewriter()
        llm = RecordingFakeLLM()
        rewriter = llm_mod.LLMQueryRewriter(llm=llm)
        history = [
            {"role": "user", "content": "年假多少天？"},
            {"role": "assistant", "content": "15天。"},
        ]
        original = list(history)
        rewriter.rewrite("没休完怎么办？", history)
        assert history == original

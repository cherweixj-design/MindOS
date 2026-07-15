"""Tests for MindOS + Query Rewriter integration."""

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
    """Rewriter that records calls and returns a fixed result."""

    def __init__(self, rewritten: str = "员工的年假没休完怎么办？"):
        self.rewritten = rewritten
        self.call_count = 0
        self.last_question = None
        self.last_history = None

    def rewrite(self, question: str, history: List[Dict[str, str]]) -> str:
        self.call_count += 1
        self.last_question = question
        self.last_history = history
        return self.rewritten


class ErrorQueryRewriter(BaseQueryRewriter):
    """Rewriter that raises QueryRewriteError."""

    def rewrite(self, question: str, history: List[Dict[str, str]]) -> str:
        raise QueryRewriteError("Rewrite failed")


class UnexpectedErrorQueryRewriter(BaseQueryRewriter):
    """Rewriter that raises an unexpected exception."""

    def rewrite(self, question: str, history: List[Dict[str, str]]) -> str:
        raise RuntimeError("Unexpected error")


class FakeLLM(BaseLLM):
    """Fake LLM returning a fixed answer."""

    def __init__(self, answer: str = "这是回答。"):
        self.answer = answer

    def chat(self, messages: list) -> str:
        return self.answer


class RecordingRetriever(BaseRetriever):
    """Retriever that records the last query."""

    def __init__(self, results: List[RetrievalResult] = None):
        self.last_question = None
        self.last_top_k = None
        self._results = results or []

    def search(self, question: str, top_k: int = 3) -> list:
        self.last_question = question
        self.last_top_k = top_k
        return self._results[:top_k]


class RecordingPromptBuilder(PromptBuilder):
    """Prompt builder that records inputs."""

    def __init__(self):
        super().__init__()
        self.last_question = None
        self.last_messages = None

    def build(
        self,
        history: list,
        knowledge: list,
        question: str,
    ) -> List[Dict[str, str]]:
        self.last_question = question
        self.last_messages = super().build(history, knowledge, question)
        return self.last_messages


REWRITTEN = "员工的年假没休完怎么办？"
ORIGINAL = "没休完怎么办？"


def make_mindos_with_rewriter(
    rewriter=None,
    llm_answer="这是回答。",
    retriever_results=None,
):
    retriever = RecordingRetriever(retriever_results)
    prompt = RecordingPromptBuilder()
    return (
        MindOS(
            llm=FakeLLM(llm_answer),
            memory=Memory(),
            retriever=retriever,
            prompt_builder=prompt,
            rewriter=rewriter,
            top_k=3,
        ),
        retriever,
        prompt,
    )


class TestMindosQueryRewrite:
    """Tests for Query Rewriting in MindOS."""

    def test_rewritten_question_goes_to_retriever(self):
        rewriter = RecordingQueryRewriter(REWRITTEN)
        mindos, retriever, _ = make_mindos_with_rewriter(rewriter=rewriter)
        mindos.chat(ORIGINAL)
        assert retriever.last_question == REWRITTEN
        assert retriever.last_question != ORIGINAL

    def test_prompt_builder_receives_original_question(self):
        rewriter = RecordingQueryRewriter(REWRITTEN)
        mindos, retriever, prompt = make_mindos_with_rewriter(rewriter=rewriter)
        mindos.chat(ORIGINAL)
        assert prompt.last_question == ORIGINAL
        assert prompt.last_question != REWRITTEN

    def test_memory_saves_original_question(self):
        memory = Memory()
        rewriter = RecordingQueryRewriter(REWRITTEN)
        retriever = RecordingRetriever()
        prompt = RecordingPromptBuilder()
        mindos = MindOS(
            llm=FakeLLM(),
            memory=memory,
            retriever=retriever,
            prompt_builder=prompt,
            rewriter=rewriter,
            top_k=3,
        )
        mindos.chat(ORIGINAL)
        messages = memory.get()
        assert any(
            msg["role"] == "user" and msg["content"] == ORIGINAL
            for msg in messages
        )
        assert not any(
            msg["role"] == "user" and msg["content"] == REWRITTEN
            for msg in messages
        )

    def test_history_passed_to_rewriter(self):
        memory = Memory()
        rewriter = RecordingQueryRewriter(REWRITTEN)
        retriever = RecordingRetriever()
        prompt = RecordingPromptBuilder()
        mindos = MindOS(
            llm=FakeLLM(answer="15天。"),
            memory=memory,
            retriever=retriever,
            prompt_builder=prompt,
            rewriter=rewriter,
            top_k=3,
        )
        mindos.chat("员工每年有多少天年假？")
        expected_history = [
            {"role": "user", "content": "员工每年有多少天年假？"},
            {"role": "assistant", "content": "15天。"},
        ]
        mindos.chat(ORIGINAL)
        assert rewriter.last_history == expected_history

    def test_no_rewriter_uses_original_question(self):
        retriever = RecordingRetriever()
        mindos = MindOS(
            llm=FakeLLM(),
            memory=Memory(),
            retriever=retriever,
            prompt_builder=PromptBuilder(),
            top_k=3,
        )
        mindos.chat(ORIGINAL)
        assert retriever.last_question == ORIGINAL

    def test_query_rewrite_error_falls_back(self):
        rewriter = ErrorQueryRewriter()
        mindos, retriever, _ = make_mindos_with_rewriter(rewriter=rewriter)
        mindos.chat(ORIGINAL)
        assert retriever.last_question == ORIGINAL

    def test_unexpected_error_not_swallowed(self):
        rewriter = UnexpectedErrorQueryRewriter()
        mindos, retriever, _ = make_mindos_with_rewriter(rewriter=rewriter)
        with pytest.raises(RuntimeError):
            mindos.chat(ORIGINAL)

    def test_rewriter_called_exactly_once(self):
        rewriter = RecordingQueryRewriter(REWRITTEN)
        mindos, retriever, _ = make_mindos_with_rewriter(rewriter=rewriter)
        mindos.chat(ORIGINAL)
        assert rewriter.call_count == 1

    def test_rewritten_question_not_in_prompt_messages(self):
        rewriter = RecordingQueryRewriter(REWRITTEN)
        mindos, retriever, prompt = make_mindos_with_rewriter(rewriter=rewriter)
        mindos.chat(ORIGINAL)
        for msg in prompt.last_messages:
            content = msg.get("content", "")
            assert REWRITTEN not in str(content)
        assert ORIGINAL in str(prompt.last_messages)

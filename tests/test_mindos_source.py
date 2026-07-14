"""Tests for source display in MindOS output."""

from typing import List

import pytest

from src.llm.base import BaseLLM
from src.memory.memory import Memory
from src.mindos import MindOS
from src.prompt.prompt_builder import PromptBuilder
from src.rag.base_retriever import BaseRetriever


class _Result:
    """Fake search result that supports tuple unpacking and has .source."""
    def __init__(self, text: str, score: float, source: str):
        self.text = text
        self.score = score
        self.source = source

    def __iter__(self):
        return iter((self.text, self.score))


class FakeLLM(BaseLLM):
    """Fake LLM that returns a fixed answer."""

    def __init__(self, answer: str = "这是回答。"):
        self.answer = answer

    def chat(self, messages: list) -> str:
        return self.answer


class FakeRetriever(BaseRetriever):
    """Fake retriever that returns pre-defined results."""

    def __init__(self, results: List[_Result]):
        self._results = results

    def search(self, question: str, top_k: int = 3) -> list:
        return self._results[:top_k]


def make_mindos(retriever, llm_answer="这是回答。", debug=False):
    """Create a MindOS instance with fake dependencies."""
    return MindOS(
        llm=FakeLLM(llm_answer),
        memory=Memory(),
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        top_k=5,
        debug=debug,
    )


class TestMindosDebugSource:
    """Tests for source info in debug output."""

    def test_debug_output_contains_source(self, capsys):
        retriever = FakeRetriever([
            _Result("年假政策", 0.9, "employee.md"),
        ])
        mindos = make_mindos(retriever, debug=True)
        mindos.chat("年假多少天？")
        captured = capsys.readouterr()
        assert "Source:" in captured.out
        assert "employee.md" in captured.out

    def test_debug_output_source_format(self, capsys):
        retriever = FakeRetriever([
            _Result("病假政策", 0.85, "sick.md"),
        ])
        mindos = make_mindos(retriever, debug=True)
        mindos.chat("病假？")
        captured = capsys.readouterr()
        assert "Source:" in captured.out
        assert "sick.md" in captured.out

    def test_debug_output_shows_source_for_each_chunk(self, capsys):
        retriever = FakeRetriever([
            _Result("年假政策", 0.9, "employee.md"),
            _Result("病假政策", 0.8, "sick.md"),
        ])
        mindos = make_mindos(retriever, debug=True)
        mindos.chat("假期政策？")
        captured = capsys.readouterr()
        assert captured.out.count("Source:") == 2
        assert "employee.md" in captured.out
        assert "sick.md" in captured.out


class TestMindosAnswerSource:
    """Tests for source info in the final answer."""

    def test_answer_ends_with_source(self):
        retriever = FakeRetriever([
            _Result("年假政策", 0.9, "employee.md"),
        ])
        mindos = make_mindos(retriever, "根据知识库，员工有15天年假。")
        answer = mindos.chat("年假多少天？")
        assert "来源：" in answer
        assert "employee.md" in answer

    def test_multiple_chunks_same_source_deduped(self):
        retriever = FakeRetriever([
            _Result("年假天数", 0.9, "employee.md"),
            _Result("年假结转", 0.8, "employee.md"),
        ])
        mindos = make_mindos(retriever, "年假规定。")
        answer = mindos.chat("年假？")
        assert "来源：" in answer
        assert answer.count("employee.md") == 1

    def test_multiple_sources_ordered_and_deduped(self):
        retriever = FakeRetriever([
            _Result("年假政策", 0.9, "employee.md"),
            _Result("病假政策", 0.8, "sick.md"),
            _Result("年假结转", 0.7, "employee.md"),
        ])
        mindos = make_mindos(retriever, "假期政策。")
        answer = mindos.chat("假期？")
        assert "来源：" in answer
        employee_pos = answer.index("employee.md")
        sick_pos = answer.index("sick.md")
        assert employee_pos < sick_pos

    def test_no_source_when_no_knowledge_retrieved(self):
        retriever = FakeRetriever([])
        mindos = make_mindos(retriever, "没有相关知识。")
        answer = mindos.chat("董事长？")
        assert "来源：" not in answer
        assert answer == "没有相关知识。"
        assert not any(r.source for r in retriever._results)

    def test_answer_source_format_with_multiple_files(self):
        retriever = FakeRetriever([
            _Result("年假政策", 0.9, "employee.md"),
            _Result("考勤制度", 0.85, "attendance.md"),
        ])
        mindos = make_mindos(retriever, "制度说明。")
        answer = mindos.chat("制度？")
        assert "来源：" in answer
        assert "employee.md" in answer
        assert "attendance.md" in answer

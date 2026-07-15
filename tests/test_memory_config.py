"""Tests for Memory sliding window configuration."""

from pathlib import Path

import pytest


class FakeLoader:
    def load(self, path):
        return ""


class FakeSplitter:
    def split(self, text):
        return []


class FakeEmbedding:
    def embed(self, texts):
        return []


class FakeVectorStore:
    def add(self, texts, vectors, sources):
        pass

    def search(self, query_vector, top_k=3):
        return []


class FakeCache:
    def __init__(self, **kwargs):
        pass


class FakeIndexer:
    def __init__(self, **kwargs):
        pass

    def index_directory(self, path):
        return 0


class FakeRetriever:
    def __init__(self, **kwargs):
        pass


class FakeLLM:
    def chat(self, messages):
        return ""


class FakeRewriter:
    def __init__(self, llm=None):
        pass


class FakePromptBuilder:
    def __init__(self, **kwargs):
        pass

    def build(self, history, knowledge, question):
        return []


import main as main_module


class TestMemoryConfig:
    """Tests for Memory sliding window configuration."""

    def test_settings_has_memory_max_turns(self):
        from src.config.settings import Settings
        assert hasattr(Settings, "MEMORY_MAX_TURNS")
        value = Settings.MEMORY_MAX_TURNS
        assert isinstance(value, int)
        assert not isinstance(value, bool)
        assert value > 0

    def test_build_mindos_passes_max_turns_to_memory(self, monkeypatch):
        captured = {}

        class RecordingMemory:
            def __init__(self, max_turns=None):
                captured["max_turns"] = max_turns

        monkeypatch.setattr(main_module, "Memory", RecordingMemory, raising=False)
        monkeypatch.setattr(main_module, "MarkdownLoader", FakeLoader, raising=False)
        monkeypatch.setattr(
            main_module, "ParagraphSplitter", FakeSplitter, raising=False
        )
        monkeypatch.setattr(
            main_module, "SiliconFlowEmbedding", FakeEmbedding, raising=False
        )
        monkeypatch.setattr(
            main_module, "InMemoryVectorStore", FakeVectorStore, raising=False
        )
        monkeypatch.setattr(main_module, "KnowledgeCache", FakeCache, raising=False)
        monkeypatch.setattr(main_module, "Indexer", FakeIndexer, raising=False)
        monkeypatch.setattr(main_module, "Retriever", FakeRetriever, raising=False)
        monkeypatch.setattr(main_module, "DeepSeekLLM", FakeLLM, raising=False)
        monkeypatch.setattr(
            main_module, "LLMQueryRewriter", FakeRewriter, raising=False
        )
        monkeypatch.setattr(
            main_module, "PromptBuilder", FakePromptBuilder, raising=False
        )
        monkeypatch.setattr(
            main_module.Settings,
            "MEMORY_MAX_TURNS",
            7,
            raising=False,
        )

        main_module.build_mindos()

        assert captured["max_turns"] == 7

    def test_env_example_contains_memory_max_turns(self):
        env_path = Path(__file__).resolve().parent.parent / ".env.example"
        assert env_path.exists()
        content = env_path.read_text(encoding="utf-8")
        assert "MEMORY_MAX_TURNS=" in content

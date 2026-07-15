"""Tests for main.py DEBUG wiring."""

import pytest

import main as main_module


class FakeEmbedding:
    def embed(self, texts):
        return []


class FakeLLM:
    def chat(self, messages):
        return ""


class TestMainDebugConfig:
    """Tests that build_mindos() passes Settings.DEBUG to Indexer."""

    @pytest.mark.parametrize("debug_value", [True, False])
    def test_build_mindos_passes_debug_to_indexer(
        self, monkeypatch, debug_value
    ):
        captured = {}

        class RecordingIndexer:
            def __init__(self, **kwargs):
                captured["debug"] = kwargs.get("debug")

            def index_directory(self, directory_path):
                return 0

        monkeypatch.setattr(
            main_module, "Indexer", RecordingIndexer, raising=False
        )
        monkeypatch.setattr(
            main_module, "SiliconFlowEmbedding", FakeEmbedding, raising=False
        )
        monkeypatch.setattr(
            main_module, "DeepSeekLLM", FakeLLM, raising=False
        )
        monkeypatch.setattr(
            main_module.Settings, "DEBUG", debug_value, raising=False
        )

        main_module.build_mindos()

        assert captured["debug"] is debug_value

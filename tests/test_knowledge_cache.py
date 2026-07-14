"""Tests for KnowledgeCache persistence."""

import json
from pathlib import Path

import pytest


def _get_classes():
    """Dynamically import KnowledgeCache and CachedIndex.
    Fails in RED phase because the module does not exist yet.
    """
    from importlib import import_module
    mod = import_module("src.rag.knowledge_cache")
    return mod.KnowledgeCache, mod.CachedIndex


def create_file(directory: Path, name: str, content: str) -> Path:
    """Write a file into the given directory."""
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


class TestFingerprint:
    """Tests for compute_fingerprint()."""

    def test_stable_fingerprint_for_same_directory(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容A")
        create_file(tmp_path, "b.md", "内容B")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp1 = cache.compute_fingerprint(str(tmp_path))
        fp2 = cache.compute_fingerprint(str(tmp_path))
        assert fp1 == fp2
        assert isinstance(fp1, str)
        assert len(fp1) > 0

    def test_new_md_file_changes_fingerprint(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容A")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp_before = cache.compute_fingerprint(str(tmp_path))
        create_file(tmp_path, "b.md", "内容B")
        fp_after = cache.compute_fingerprint(str(tmp_path))
        assert fp_before != fp_after

    def test_deleted_md_file_changes_fingerprint(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容A")
        create_file(tmp_path, "b.md", "内容B")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp_before = cache.compute_fingerprint(str(tmp_path))
        (tmp_path / "b.md").unlink()
        fp_after = cache.compute_fingerprint(str(tmp_path))
        assert fp_before != fp_after

    def test_modified_md_content_changes_fingerprint(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "原始内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp_before = cache.compute_fingerprint(str(tmp_path))
        create_file(tmp_path, "a.md", "修改后的内容")
        fp_after = cache.compute_fingerprint(str(tmp_path))
        assert fp_before != fp_after

    def test_renamed_md_file_changes_fingerprint(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp_before = cache.compute_fingerprint(str(tmp_path))
        (tmp_path / "a.md").rename(tmp_path / "b.md")
        fp_after = cache.compute_fingerprint(str(tmp_path))
        assert fp_before != fp_after

    def test_modifying_non_md_file_does_not_change_fingerprint(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp_before = cache.compute_fingerprint(str(tmp_path))
        create_file(tmp_path, "notes.txt", "一些笔记")
        create_file(tmp_path, "data.json", '{"key": "value"}')
        fp_after = cache.compute_fingerprint(str(tmp_path))
        assert fp_before == fp_after

    def test_modifying_md_in_subdirectory_does_not_change_fingerprint(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp_before = cache.compute_fingerprint(str(tmp_path))
        sub = tmp_path / "sub"
        sub.mkdir()
        create_file(sub, "deep.md", "子目录文件")
        fp_after = cache.compute_fingerprint(str(tmp_path))
        assert fp_before == fp_after


class TestSaveLoad:
    """Tests for save() and load_if_valid()."""

    def test_save_creates_cache_file(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp = cache.compute_fingerprint(str(tmp_path))
        cache.save(
            indexed_file_count=1,
            texts=["内容"],
            vectors=[[0.1, 0.2]],
            sources=["a.md"],
            fingerprint=fp,
        )
        assert (tmp_path / "cache.json").exists()

    def test_save_auto_creates_parent_directory(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache_dir = tmp_path / "sub" / "cache"
        cache = KnowledgeCache(cache_path=str(cache_dir / "cache.json"), cache_key="test-model")
        fp = cache.compute_fingerprint(str(tmp_path))
        cache.save(
            indexed_file_count=1,
            texts=["内容"],
            vectors=[[0.1, 0.2]],
            sources=["a.md"],
            fingerprint=fp,
        )
        assert (cache_dir / "cache.json").exists()

    def test_save_load_roundtrip_returns_same_data(self, tmp_path):
        KnowledgeCache, CachedIndex = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp = cache.compute_fingerprint(str(tmp_path))
        cache.save(
            indexed_file_count=1,
            texts=["内容"],
            vectors=[[0.1, 0.2]],
            sources=["a.md"],
            fingerprint=fp,
        )
        result = cache.load_if_valid(str(tmp_path))
        assert isinstance(result, CachedIndex)
        assert result.indexed_file_count == 1
        assert result.texts == ["内容"]
        assert result.vectors == [[0.1, 0.2]]
        assert result.sources == ["a.md"]

    def test_save_raises_on_length_mismatch(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp = cache.compute_fingerprint(str(tmp_path))
        with pytest.raises(ValueError):
            cache.save(
                indexed_file_count=3,
                texts=["a", "b", "c"],
                vectors=[[1.0, 2.0], [3.0, 4.0]],
                sources=["x.md", "y.md", "z.md"],
                fingerprint=fp,
            )

    @pytest.mark.parametrize("bad_value", [-1, 1.5, "1", None, True])
    def test_save_raises_on_invalid_indexed_file_count(self, tmp_path, bad_value):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp = cache.compute_fingerprint(str(tmp_path))
        with pytest.raises(ValueError):
            cache.save(
                indexed_file_count=bad_value,
                texts=["内容"],
                vectors=[[0.1, 0.2]],
                sources=["a.md"],
                fingerprint=fp,
            )


class TestLoadInvalid:
    """Tests for load_if_valid() returning None."""

    def _write_cache(self, path: Path, data: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def test_cache_file_missing_returns_none(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        result = cache.load_if_valid(str(tmp_path))
        assert result is None

    def test_corrupt_json_returns_none(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("{не_json", encoding="utf-8")
        cache = KnowledgeCache(cache_path=str(cache_file), cache_key="test-model")
        result = cache.load_if_valid(str(tmp_path))
        assert result is None

    def test_missing_required_fields_returns_none(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        self._write_cache(tmp_path / "cache.json", {"schema_version": 1})
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        result = cache.load_if_valid(str(tmp_path))
        assert result is None

    @pytest.mark.parametrize("bad_version", [999, True, 1.0, "1", None])
    def test_wrong_schema_version_returns_none(self, tmp_path, bad_version):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp = cache.compute_fingerprint(str(tmp_path))
        cache.save(
            indexed_file_count=1,
            texts=["内容"],
            vectors=[[0.1, 0.2]],
            sources=["a.md"],
            fingerprint=fp,
        )
        with open(tmp_path / "cache.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        data["schema_version"] = bad_version
        with open(tmp_path / "cache.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        result = cache.load_if_valid(str(tmp_path))
        assert result is None

    def test_cache_key_mismatch_returns_none(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache_v1 = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="old-model")
        fp = cache_v1.compute_fingerprint(str(tmp_path))
        cache_v1.save(
            indexed_file_count=1,
            texts=["内容"],
            vectors=[[0.1, 0.2]],
            sources=["a.md"],
            fingerprint=fp,
        )
        cache_v2 = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="new-model")
        result = cache_v2.load_if_valid(str(tmp_path))
        assert result is None

    def test_fingerprint_mismatch_returns_none(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "原始内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp = cache.compute_fingerprint(str(tmp_path))
        cache.save(
            indexed_file_count=1,
            texts=["原始内容"],
            vectors=[[0.1, 0.2]],
            sources=["a.md"],
            fingerprint=fp,
        )
        create_file(tmp_path, "a.md", "修改后的内容")
        result = cache.load_if_valid(str(tmp_path))
        assert result is None

    def test_length_mismatch_returns_none(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        self._write_cache(tmp_path / "cache.json", {
            "schema_version": 1,
            "cache_key": "test-model",
            "fingerprint": "any",
            "indexed_file_count": 2,
            "texts": ["a", "b"],
            "vectors": [[1.0]],
            "sources": ["a.md", "b.md"],
        })
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        result = cache.load_if_valid(str(tmp_path))
        assert result is None

    def test_invalid_indexed_file_count_returns_none(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        self._write_cache(tmp_path / "cache.json", {
            "schema_version": 1,
            "cache_key": "test-model",
            "fingerprint": "any",
            "indexed_file_count": -1,
            "texts": ["a"],
            "vectors": [[1.0]],
            "sources": ["a.md"],
        })
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        result = cache.load_if_valid(str(tmp_path))
        assert result is None

    def test_texts_vectors_sources_all_empty_returns_valid(self, tmp_path):
        KnowledgeCache, CachedIndex = _get_classes()
        create_file(tmp_path, "empty.md", "")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp = cache.compute_fingerprint(str(tmp_path))
        cache.save(
            indexed_file_count=0,
            texts=[],
            vectors=[],
            sources=[],
            fingerprint=fp,
        )
        result = cache.load_if_valid(str(tmp_path))
        assert isinstance(result, CachedIndex)
        assert result.indexed_file_count == 0

    def test_cache_json_contains_only_expected_keys(self, tmp_path):
        KnowledgeCache, _ = _get_classes()
        create_file(tmp_path, "a.md", "内容")
        cache = KnowledgeCache(cache_path=str(tmp_path / "cache.json"), cache_key="test-model")
        fp = cache.compute_fingerprint(str(tmp_path))
        cache.save(
            indexed_file_count=1,
            texts=["内容"],
            vectors=[[0.1, 0.2]],
            sources=["a.md"],
            fingerprint=fp,
        )
        with open(tmp_path / "cache.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert set(data.keys()) == {
            "schema_version",
            "cache_key",
            "fingerprint",
            "indexed_file_count",
            "texts",
            "vectors",
            "sources",
        }

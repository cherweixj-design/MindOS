"""Knowledge cache for persisting vector indices."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


SCHEMA_VERSION = 1

REQUIRED_KEYS = frozenset({
    "schema_version",
    "cache_key",
    "fingerprint",
    "indexed_file_count",
    "texts",
    "vectors",
    "sources",
})


@dataclass(frozen=True)
class CachedIndex:
    """Immutable container for indexed knowledge data loaded from cache."""
    indexed_file_count: int
    texts: List[str]
    vectors: List[List[float]]
    sources: List[str]


class KnowledgeCache:
    """Manage persistence of indexed knowledge data to a JSON cache file."""

    def __init__(self, cache_path: str, cache_key: str):
        self._cache_path = Path(cache_path)
        self._cache_key = cache_key

    def compute_fingerprint(self, directory_path: str) -> str:
        """Compute a stable SHA-256 fingerprint of the knowledge directory."""
        path = Path(directory_path)
        md_files = sorted(path.glob("*.md"))

        hasher = hashlib.sha256()

        for md_file in md_files:
            name_bytes = md_file.name.encode("utf-8")
            content_bytes = md_file.read_bytes()

            hasher.update(str(len(name_bytes)).encode("utf-8"))
            hasher.update(b":")
            hasher.update(name_bytes)
            hasher.update(str(len(content_bytes)).encode("utf-8"))
            hasher.update(b":")
            hasher.update(content_bytes)

        return hasher.hexdigest()

    def save(
        self,
        *,
        indexed_file_count,
        texts,
        vectors,
        sources,
        fingerprint: str,
    ) -> None:
        """Validate and save indexed data to the cache JSON file."""

        if not (
            isinstance(indexed_file_count, int)
            and not isinstance(indexed_file_count, bool)
            and indexed_file_count >= 0
        ):
            raise ValueError(
                f"indexed_file_count must be a non-negative integer, "
                f"got {type(indexed_file_count).__name__}: {indexed_file_count}"
            )

        if not (len(texts) == len(vectors) == len(sources)):
            raise ValueError(
                "texts, vectors, and sources must have the same length."
            )

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "schema_version": SCHEMA_VERSION,
            "cache_key": self._cache_key,
            "fingerprint": fingerprint,
            "indexed_file_count": indexed_file_count,
            "texts": texts,
            "vectors": vectors,
            "sources": sources,
        }

        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load_if_valid(
        self,
        directory_path: str,
    ) -> Optional[CachedIndex]:
        """Return CachedIndex if cache is valid, otherwise None."""

        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

        if not isinstance(data, dict):
            return None

        if not all(key in data for key in REQUIRED_KEYS):
            return None

        schema_version = data.get("schema_version")
        if not (
            isinstance(schema_version, int)
            and not isinstance(schema_version, bool)
            and schema_version == SCHEMA_VERSION
        ):
            return None

        if data.get("cache_key") != self._cache_key:
            return None

        current_fingerprint = self.compute_fingerprint(directory_path)
        if data.get("fingerprint") != current_fingerprint:
            return None

        texts = data.get("texts")
        vectors = data.get("vectors")
        sources = data.get("sources")

        if not isinstance(texts, list):
            return None
        if not isinstance(vectors, list):
            return None
        if not isinstance(sources, list):
            return None

        if not (len(texts) == len(vectors) == len(sources)):
            return None

        indexed_file_count = data.get("indexed_file_count")
        if not (
            isinstance(indexed_file_count, int)
            and not isinstance(indexed_file_count, bool)
            and indexed_file_count >= 0
        ):
            return None

        return CachedIndex(
            indexed_file_count=indexed_file_count,
            texts=texts,
            vectors=vectors,
            sources=sources,
        )

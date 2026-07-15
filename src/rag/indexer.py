from pathlib import Path
from typing import List, Optional, Tuple

from .base_embedding import BaseEmbedding
from .base_loader import BaseLoader
from .base_splitter import BaseSplitter
from .base_vector_store import BaseVectorStore
from .knowledge_cache import KnowledgeCache


class Indexer:
    """Build a searchable index from a document."""

    def __init__(
        self,
        loader: BaseLoader,
        splitter: BaseSplitter,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        cache: Optional[KnowledgeCache] = None,
        debug: bool = False,
    ):
        self.loader = loader
        self.splitter = splitter
        self.embedding = embedding
        self.vector_store = vector_store
        self.cache = cache
        self.debug = debug

    def index(self, file_path: str) -> None:
        """Load, split, embed, and store a document."""
        if not self._index_file(Path(file_path)):
            raise ValueError(
                f"Knowledge file is empty: {file_path}"
            )

    def index_directory(self, directory_path: str) -> int:
        """Index all non-empty Markdown files in a directory.

        Uses cache when available. Returns the number of files
        that were successfully indexed.
        """
        path = Path(directory_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory_path}"
            )

        if not path.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {directory_path}"
            )

        if self.cache is not None:
            cached = self.cache.load_if_valid(directory_path)
            if cached is not None:
                if self.debug:
                    print("[DEBUG] 知识索引缓存：命中。")
                self.vector_store.add(
                    texts=cached.texts,
                    vectors=cached.vectors,
                    sources=cached.sources,
                )
                return cached.indexed_file_count

            if self.debug:
                print("[DEBUG] 知识索引缓存：未命中，正在重新构建。")

        md_files = sorted(path.glob("*.md"))

        all_texts: List[str] = []
        all_vectors: List[List[float]] = []
        all_sources: List[str] = []
        count = 0

        for md_file in md_files:
            result = self._prepare_file_index(md_file)
            if result is not None:
                texts, vectors, sources = result
                self.vector_store.add(
                    texts=texts,
                    vectors=vectors,
                    sources=sources,
                )
                all_texts.extend(texts)
                all_vectors.extend(vectors)
                all_sources.extend(sources)
                count += 1

        if self.cache is not None:
            self.cache.save(
                indexed_file_count=count,
                texts=all_texts,
                vectors=all_vectors,
                sources=all_sources,
                fingerprint=self.cache.compute_fingerprint(directory_path),
            )

        return count

    def _prepare_file_index(
        self,
        file_path: Path,
    ) -> Optional[Tuple[List[str], List[List[float]], List[str]]]:
        """Load, split, and embed a single file.

        Returns (texts, vectors, sources) or None if file is empty.
        Raises ValueError on data integrity issues.
        """
        text = self.loader.load(str(file_path))

        if not text.strip():
            return None

        chunks = self.splitter.split(text)

        if not chunks:
            raise ValueError(
                f"No text chunks were created: {file_path}"
            )

        vectors = self.embedding.embed(chunks)

        if len(chunks) != len(vectors):
            raise ValueError(
                "The number of chunks and vectors does not match."
            )

        sources = [file_path.name] * len(chunks)

        return chunks, vectors, sources

    def _index_file(self, file_path: Path) -> bool:
        """Index a single file if it has non-empty content.

        Returns True if content was indexed, False if the file was empty.
        """
        result = self._prepare_file_index(file_path)

        if result is None:
            return False

        texts, vectors, sources = result

        self.vector_store.add(
            texts=texts,
            vectors=vectors,
            sources=sources,
        )
        return True

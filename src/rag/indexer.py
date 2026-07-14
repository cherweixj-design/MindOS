from pathlib import Path

from .base_embedding import BaseEmbedding
from .base_loader import BaseLoader
from .base_splitter import BaseSplitter
from .base_vector_store import BaseVectorStore


class Indexer:
    """Build a searchable index from a document."""

    def __init__(
        self,
        loader: BaseLoader,
        splitter: BaseSplitter,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
    ):
        self.loader = loader
        self.splitter = splitter
        self.embedding = embedding
        self.vector_store = vector_store

    def index(self, file_path: str) -> None:
        """Load, split, embed, and store a document."""
        if not self._index_file(Path(file_path)):
            raise ValueError(
                f"Knowledge file is empty: {file_path}"
            )

    def index_directory(self, directory_path: str) -> int:
        """Index all non-empty Markdown files in a directory.

        Returns the number of files that were successfully indexed.
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

        md_files = sorted(path.glob("*.md"))

        count = 0
        for md_file in md_files:
            if self._index_file(md_file):
                count += 1

        return count

    def _index_file(self, file_path: Path) -> bool:
        """Index a single file if it has non-empty content.

        Returns True if content was indexed, False if the file was empty.
        """
        text = self.loader.load(str(file_path))

        if not text.strip():
            return False

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

        self.vector_store.add(
            texts=chunks,
            vectors=vectors,
        )
        return True

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

        text = self.loader.load(file_path)

        if not text.strip():
            raise ValueError(
                f"Knowledge file is empty: {file_path}"
            )

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
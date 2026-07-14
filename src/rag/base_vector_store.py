from typing import List

from .retrieval_result import RetrievalResult


class BaseVectorStore:
    """Base class for all vector stores."""

    def add(
        self,
        texts: List[str],
        vectors: List[List[float]],
        sources: List[str],
    ) -> None:
        """Add texts, vectors and their sources to the store."""
        raise NotImplementedError

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
    ) -> List[RetrievalResult]:
        """Return relevant results with text, score, and source."""
        raise NotImplementedError

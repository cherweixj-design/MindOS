from typing import List, Tuple


class BaseVectorStore:
    """Base class for all vector stores."""

    def add(
        self,
        texts: List[str],
        vectors: List[List[float]],
    ) -> None:
        """Add texts and their vectors to the store."""
        raise NotImplementedError

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return relevant texts and their similarity scores."""
        raise NotImplementedError
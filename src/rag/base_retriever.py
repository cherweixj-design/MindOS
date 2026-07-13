from typing import List, Tuple


class BaseRetriever:
    """Base class for all retrievers."""

    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return relevant chunks and similarity scores."""
        raise NotImplementedError
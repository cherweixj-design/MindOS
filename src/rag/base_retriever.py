from typing import List

from .retrieval_result import RetrievalResult


class BaseRetriever:
    """Base class for all retrievers."""

    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> List[RetrievalResult]:
        """Return relevant results with text, score, and source."""
        raise NotImplementedError

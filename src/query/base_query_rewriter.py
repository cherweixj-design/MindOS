"""Base class and exception for query rewriting."""

from typing import Dict, List


class QueryRewriteError(RuntimeError):
    """Raised when a query cannot be rewritten by the LLM."""


class BaseQueryRewriter:
    """Base interface for query rewriting."""

    def rewrite(
        self,
        question: str,
        history: List[Dict[str, str]],
    ) -> str:
        """Rewrite a question using conversation history.

        Returns a self-contained question for retrieval.
        Returns the original question unchanged when history is empty.
        """
        raise NotImplementedError

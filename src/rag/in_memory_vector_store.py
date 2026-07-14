from math import sqrt
from typing import List, Tuple

from .base_vector_store import BaseVectorStore
from .retrieval_result import RetrievalResult


class InMemoryVectorStore(BaseVectorStore):
    """Store vectors in memory and search by cosine similarity."""

    def __init__(self):
        self.texts: List[str] = []
        self.vectors: List[List[float]] = []
        self.sources: List[str] = []

    def add(
        self,
        texts: List[str],
        vectors: List[List[float]],
        sources: List[str],
    ) -> None:
        """Add texts, vectors, and sources to memory."""

        if len(texts) != len(vectors) or len(texts) != len(sources):
            raise ValueError(
                "Texts, vectors, and sources must have the same length."
            )

        self.texts.extend(texts)
        self.vectors.extend(vectors)
        self.sources.extend(sources)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
    ) -> List[RetrievalResult]:
        """Return results sorted by cosine similarity."""

        scored_results: List[Tuple[str, float, str]] = []

        for text, vector, source in zip(
            self.texts, self.vectors, self.sources
        ):
            score = self._cosine_similarity(
                query_vector,
                vector,
            )
            scored_results.append((text, score, source))

        scored_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        top_results = scored_results[:top_k]

        return [
            RetrievalResult(text=text, score=score, source=source)
            for text, score, source in top_results
        ]

    def _cosine_similarity(
        self,
        vector_a: List[float],
        vector_b: List[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""

        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Vectors must have the same length."
            )

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        length_a = sqrt(
            sum(a * a for a in vector_a)
        )

        length_b = sqrt(
            sum(b * b for b in vector_b)
        )

        if length_a == 0 or length_b == 0:
            return 0.0

        return dot_product / (
            length_a * length_b
        )

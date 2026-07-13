from math import sqrt
from typing import List, Tuple

from .base_vector_store import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """Store vectors in memory and search by cosine similarity."""

    def __init__(self):
        self.texts: List[str] = []
        self.vectors: List[List[float]] = []

    def add(
        self,
        texts: List[str],
        vectors: List[List[float]],
    ) -> None:
        """Add texts and their vectors to memory."""

        if len(texts) != len(vectors):
            raise ValueError(
                "Texts and vectors must have the same length."
            )

        self.texts.extend(texts)
        self.vectors.extend(vectors)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return texts and cosine-similarity scores."""

        scored_results: List[Tuple[str, float]] = []

        for text, vector in zip(
            self.texts,
            self.vectors,
        ):
            score = self._cosine_similarity(
                query_vector,
                vector,
            )

            scored_results.append(
                (text, score)
            )

        scored_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_results[:top_k]

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
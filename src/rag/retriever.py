from typing import List

from .base_embedding import BaseEmbedding
from .base_retriever import BaseRetriever
from .base_vector_store import BaseVectorStore
from .retrieval_result import RetrievalResult


class Retriever(BaseRetriever):
    """Retrieve relevant texts for a natural-language question."""

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        min_score: float = 0.0,
    ):
        self.embedding = embedding
        self.vector_store = vector_store
        self.min_score = min_score

    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> List[RetrievalResult]:
        """Embed a question and retrieve relevant knowledge."""

        query_vector = self.embedding.embed(
            [question]
        )[0]

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        return [
            result
            for result in results
            if result.score >= self.min_score
        ]

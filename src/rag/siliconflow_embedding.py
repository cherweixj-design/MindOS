from typing import List

from openai import OpenAI

from src.config.settings import Settings
from .base_embedding import BaseEmbedding


class SiliconFlowEmbedding(BaseEmbedding):
    """Generate text embeddings with SiliconFlow."""

    def __init__(self):
        if not Settings.EMBEDDING_API_KEY:
            raise ValueError(
                "SILICONFLOW_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=Settings.EMBEDDING_API_KEY,
            base_url=Settings.EMBEDDING_BASE_URL,
        )

    def embed(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Convert multiple texts into vectors."""
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=Settings.EMBEDDING_MODEL,
            input=texts,
        )

        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        return [
            item.embedding
            for item in ordered_data
        ]
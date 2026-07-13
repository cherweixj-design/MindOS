from typing import List


class BaseEmbedding:
    """Base class for all embeddings."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts into vectors."""
        raise NotImplementedError
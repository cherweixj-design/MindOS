from typing import List


class BaseSplitter:
    """Base class for all text splitters."""

    def split(self, text: str) -> List[str]:
        """Split a text into chunks."""
        raise NotImplementedError
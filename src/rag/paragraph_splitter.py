from typing import List

from .base_splitter import BaseSplitter


class ParagraphSplitter(BaseSplitter):
    """Split text into paragraphs."""

    def split(self, text: str) -> List[str]:
        chunks = text.split("\n\n")
        return [chunk.strip() for chunk in chunks if chunk.strip()]
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval result with text, score, and source."""
    text: str
    score: float
    source: str

from typing import Dict, List


class BaseLLM:
    """Base class for all LLM providers."""

    def chat(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Send messages to the language model."""
        raise NotImplementedError
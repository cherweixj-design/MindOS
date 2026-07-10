class BaseLLM:
    """Base class for all LLM providers."""
    def chat(self, prompt: str) -> str:
        raise NotImplementedError
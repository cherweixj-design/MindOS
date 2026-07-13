from typing import Dict, List

from openai import OpenAI

from src.config.settings import Settings
from .base import BaseLLM


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM implementation."""

    def __init__(self):
        if not Settings.API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set."
            )

        self.client = OpenAI(
            api_key=Settings.API_KEY,
            base_url=Settings.BASE_URL,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Send messages to DeepSeek."""
        response = self.client.chat.completions.create(
            model=Settings.MODEL_NAME,
            messages=messages,
        )

        return response.choices[0].message.content or ""
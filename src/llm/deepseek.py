from openai import OpenAI

from src.config.settings import Settings

from .base import BaseLLM


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM implementation."""

    def __init__(self):
        self.client = OpenAI(
            api_key=Settings.API_KEY, 
            base_url=Settings.BASE_URL
            )

    def chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
        model=Settings.MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

        return response.choices[0].message.content
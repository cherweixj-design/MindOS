import os

print("Loading Settings...")

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Application settings."""

    MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-pro")
    BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
    API_KEY = os.getenv("DEEPSEEK_API_KEY")

    if API_KEY is None:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set."
            )
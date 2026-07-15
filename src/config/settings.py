import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Application settings."""

    DEBUG = os.getenv(
        "DEBUG",
        "false",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    MIN_RETRIEVAL_SCORE = float(
        os.getenv(
            "MIN_RETRIEVAL_SCORE",
            "0.5",
        )
    )

    MODEL_NAME = os.getenv(
        "MODEL_NAME",
        "deepseek-v4-pro",
    )

    BASE_URL = os.getenv(
        "BASE_URL",
        "https://api.deepseek.com",
    )

    API_KEY = os.getenv("DEEPSEEK_API_KEY")

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-m3",
    )

    EMBEDDING_BASE_URL = os.getenv(
        "EMBEDDING_BASE_URL",
        "https://api.siliconflow.cn/v1",
    )

    EMBEDDING_API_KEY = os.getenv(
        "SILICONFLOW_API_KEY"
    )

    MEMORY_MAX_TURNS = int(
        os.getenv(
            "MEMORY_MAX_TURNS",
            "10",
        )
    )

from pathlib import Path

from .base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
    """Loader for Markdown files."""

    def load(self, file_path: str) -> str:
        """Load a Markdown document and return its text."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        return text
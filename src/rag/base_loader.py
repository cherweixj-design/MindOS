class BaseLoader:
    """Base class for all documentloaders."""

    def load(self, file_path: str) -> str:
        """Load a document and return its text."""
        raise NotImplementedError
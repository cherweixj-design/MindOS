"""pytest configuration: ignore script-style test files that call real APIs."""

collect_ignore = [
    "test_embedding.py",
    "test_real_retrieval.py",
    "test_retrieval_threshold.py",
    "test_retriever.py",
    "test_vector_store.py",
]

from typing import List

from src.rag.base_embedding import BaseEmbedding
from src.rag.indexer import Indexer
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.markdown_loader import MarkdownLoader
from src.rag.paragraph_splitter import ParagraphSplitter
from src.rag.retriever import Retriever


class FakeEmbedding(BaseEmbedding):
    """Create simple test vectors based on keywords."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []

        for text in texts:
            if "年假" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "病假" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        return vectors


loader = MarkdownLoader()
splitter = ParagraphSplitter()
embedding = FakeEmbedding()
vector_store = InMemoryVectorStore()

indexer = Indexer(
    loader=loader,
    splitter=splitter,
    embedding=embedding,
    vector_store=vector_store,
)

indexer.index("knowledge/employee.md")

retriever = Retriever(
    embedding=embedding,
    vector_store=vector_store,
)

results = retriever.search(
    question="员工一年有多少天年假？",
    top_k=1,
)

print(results)
from typing import List

from src.rag.base_embedding import BaseEmbedding
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.retriever import Retriever


class FakeEmbedding(BaseEmbedding):
    """Create simple test vectors based on keywords."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []

        for text in texts:
            if "年假" in text:
                vectors.append([1.0, 0.0])
            elif "病假" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])

        return vectors


embedding = FakeEmbedding()
vector_store = InMemoryVectorStore()

vector_store.add(
    texts=[
        "员工每年有15天年假。",
        "病假需要提交相关证明。",
    ],
    vectors=[
        [1.0, 0.0],
        [0.0, 1.0],
    ],
)

retriever = Retriever(
    embedding=embedding,
    vector_store=vector_store,
)

results = retriever.search(
    question="员工有多少天年假？",
    top_k=1,
)

print(results)
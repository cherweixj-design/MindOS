"""Manual check: Retriever search."""

from typing import List

from src.rag.base_embedding import BaseEmbedding
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.retriever import Retriever


class FakeEmbedding(BaseEmbedding):
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


def main() -> int:
    embedding = FakeEmbedding()
    store = InMemoryVectorStore()

    store.add(
        texts=["员工每年有15天年假。", "病假需要提交相关证明。"],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        sources=["employee.md", "employee.md"],
    )

    retriever = Retriever(embedding=embedding, vector_store=store)

    results = retriever.search(question="员工有多少天年假？", top_k=1)

    print("Retriever 检索结果：")
    for i, result in enumerate(results, start=1):
        print(
            f"  {i}. text={result.text!r}, "
            f"score={result.score:.4f}, "
            f"source={result.source!r}"
        )

    assert len(results) > 0
    assert results[0].source == "employee.md"

    print("\n✓ Retriever 手动检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

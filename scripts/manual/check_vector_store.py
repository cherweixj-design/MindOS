"""Manual check: InMemoryVectorStore search."""

from src.rag.in_memory_vector_store import InMemoryVectorStore


def main() -> int:
    store = InMemoryVectorStore()

    texts = [
        "员工每年有15天年假。",
        "病假需要提交相关证明。",
        "公司办公室位于上海。",
    ]

    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2],
    ]

    sources = ["employee.md", "employee.md", "office.md"]

    store.add(texts=texts, vectors=vectors, sources=sources)

    results = store.search(query_vector=[1.0, 0.0], top_k=2)

    print("VectorStore 检索结果：")
    for i, result in enumerate(results, start=1):
        print(
            f"  {i}. text={result.text!r}, "
            f"score={result.score:.4f}, "
            f"source={result.source!r}"
        )

    assert len(results) > 0, "应至少返回一个结果"
    assert results[0].text == "员工每年有15天年假。"
    assert results[0].source == "employee.md"

    print("\n✓ VectorStore 手动检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

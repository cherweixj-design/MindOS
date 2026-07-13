from src.rag.in_memory_vector_store import InMemoryVectorStore


vector_store = InMemoryVectorStore()

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

vector_store.add(
    texts=texts,
    vectors=vectors,
)

results = vector_store.search(
    query_vector=[1.0, 0.0],
    top_k=2,
)

print(results)
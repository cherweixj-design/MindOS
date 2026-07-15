"""Manual check: full retrieval pipeline with real Embedding API."""

import sys

from src.config.settings import Settings
from src.rag.indexer import Indexer
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.markdown_loader import MarkdownLoader
from src.rag.paragraph_splitter import ParagraphSplitter
from src.rag.retriever import Retriever
from src.rag.siliconflow_embedding import SiliconFlowEmbedding


def main() -> int:
    if not Settings.EMBEDDING_API_KEY:
        print("未配置 SILICONFLOW_API_KEY，无法执行真实检索检查。")
        return 1

    loader = MarkdownLoader()
    splitter = ParagraphSplitter()
    embedding = SiliconFlowEmbedding()
    vector_store = InMemoryVectorStore()

    indexer = Indexer(
        loader=loader,
        splitter=splitter,
        embedding=embedding,
        vector_store=vector_store,
    )

    indexed_count = indexer.index_directory("knowledge")
    print(f"已索引文件数量：{indexed_count}")
    assert indexed_count > 0, "应至少索引到一个非空 Markdown 文件"

    retriever = Retriever(embedding=embedding, vector_store=vector_store)

    questions = [
        "员工每年能享受几天带薪休假？",
        "员工生病请假需要准备什么？",
        "公司在哪里办公？",
    ]

    for question in questions:
        results = retriever.search(question=question, top_k=1)
        assert len(results) > 0, f"问题 '{question}' 应至少获得一个结果"
        result = results[0]
        print(f"问题：{question}")
        print(f"  匹配：{result.text!r}")
        print(f"  分数：{result.score:.4f}")
        print(f"  来源：{result.source!r}")
        print("-" * 40)

    print("\n✓ 多文件真实检索检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

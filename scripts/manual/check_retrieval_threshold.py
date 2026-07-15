"""Manual check: Retrieval with score threshold."""

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
        print("未配置 SILICONFLOW_API_KEY，无法执行阈值检索检查。")
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

    indexer.index_directory("knowledge")

    min_score = Settings.MIN_RETRIEVAL_SCORE
    retriever = Retriever(
        embedding=embedding,
        vector_store=vector_store,
        min_score=min_score,
    )

    questions = [
        "员工一年可以休几天？",
        "年假没有休完怎么办？",
        "生病请假需要准备什么材料？",
        "上班地点在哪座城市？",
        "公司的董事长是谁？",
        "公司的主营业务是什么？",
    ]

    print(f"当前相似度阈值：{min_score}")

    for question in questions:
        results = retriever.search(question=question, top_k=2)
        print("\n" + "=" * 50)
        print(f"问题：{question}")

        if not results:
            print("没有结果达到阈值（这不一定是失败）")
            continue

        for result in results:
            assert result.score >= min_score, (
                f"结果分数 {result.score:.4f} 不应低于阈值 {min_score}"
            )
            print(f"\n  Chunk | Score: {result.score:.4f} | Source: {result.source}")
            print(f"  {result.text}")

    print("\n✓ Retrieval 阈值手动检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

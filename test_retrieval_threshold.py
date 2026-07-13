from src.config.settings import Settings
from src.rag.indexer import Indexer
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.markdown_loader import MarkdownLoader
from src.rag.paragraph_splitter import ParagraphSplitter
from src.rag.retriever import Retriever
from src.rag.siliconflow_embedding import SiliconFlowEmbedding


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

indexer.index("knowledge/employee.md")

retriever = Retriever(
    embedding=embedding,
    vector_store=vector_store,
    min_score=Settings.MIN_RETRIEVAL_SCORE,
)

questions = [
    "员工一年可以休几天？",
    "年假没有休完怎么办？",
    "生病请假需要准备什么材料？",
    "上班地点在哪座城市？",
    "公司的董事长是谁？",
    "公司的主营业务是什么？",
]

print(
    f"当前相似度阈值："
    f"{Settings.MIN_RETRIEVAL_SCORE}"
)

for question in questions:
    results = retriever.search(
        question=question,
        top_k=2,
    )

    print("\n" + "=" * 50)
    print("问题：", question)

    if not results:
        print("检索结果：没有知识达到阈值")
        continue

    for index, (text, score) in enumerate(
        results,
        start=1,
    ):
        print(
            f"\nChunk {index} "
            f"| Score: {score:.4f}"
        )
        print(text)
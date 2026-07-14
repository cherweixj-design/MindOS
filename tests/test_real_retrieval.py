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

# 建立知识库
indexer.index("knowledge/employee.md")

retriever = Retriever(
    embedding=embedding,
    vector_store=vector_store,
)

questions = [
    "员工每年能享受几天带薪休假？",
    "员工生病请假需要准备什么？",
    "公司在哪里办公？",
]

for question in questions:
    results = retriever.search(
        question=question,
        top_k=1,
    )

    print("问题：", question)
    print("检索结果：", results[0])
    print("-" * 40)
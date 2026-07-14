from src.config.settings import Settings
from src.llm.deepseek import DeepSeekLLM
from src.memory.memory import Memory
from src.mindos import MindOS
from src.prompt.prompt_builder import PromptBuilder
from src.rag.indexer import Indexer
from src.rag.in_memory_vector_store import InMemoryVectorStore
from src.rag.knowledge_cache import KnowledgeCache
from src.rag.markdown_loader import MarkdownLoader
from src.rag.paragraph_splitter import ParagraphSplitter
from src.rag.retriever import Retriever
from src.rag.siliconflow_embedding import SiliconFlowEmbedding


def build_mindos() -> MindOS:
    """Create and connect all MindOS components."""

    loader = MarkdownLoader()
    splitter = ParagraphSplitter()
    embedding = SiliconFlowEmbedding()
    vector_store = InMemoryVectorStore()

    cache = KnowledgeCache(
        cache_path="data/knowledge_index.json",
        cache_key=Settings.EMBEDDING_MODEL,
    )

    indexer = Indexer(
        loader=loader,
        splitter=splitter,
        embedding=embedding,
        vector_store=vector_store,
        cache=cache,
    )

    indexed_count = indexer.index_directory("knowledge")

    if Settings.DEBUG:
        print(f"[DEBUG] 已索引 Markdown 文件数量：{indexed_count}")

    retriever = Retriever(
        embedding=embedding,
        vector_store=vector_store,
        min_score=Settings.MIN_RETRIEVAL_SCORE,
    )

    llm = DeepSeekLLM()
    memory = Memory()
    prompt_builder = PromptBuilder()

    return MindOS(
        llm=llm,
        memory=memory,
        retriever=retriever,
        prompt_builder=prompt_builder,
        top_k=2,
        debug=Settings.DEBUG,
    )


def show_error(
    message: str,
    error: Exception,
) -> None:
    """Show a friendly error message."""

    print(f"\n{message}")

    if Settings.DEBUG:
        print(
            f"[DEBUG] "
            f"{type(error).__name__}: {error}"
        )


def main() -> None:
    """Start the MindOS command-line application."""

    try:
        mindos = build_mindos()

    except FileNotFoundError as error:
        show_error(
            "MindOS 启动失败：知识库文件不存在。",
            error,
        )
        return

    except ValueError as error:
        show_error(
            "MindOS 启动失败：配置或知识库数据不正确。",
            error,
        )
        return

    except Exception as error:
        show_error(
            "MindOS 启动失败，请检查网络、API 配置和知识库文件。",
            error,
        )
        return

    print("MindOS 已启动，输入 exit 退出。")

    while True:
        try:
            question = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nMindOS 已退出。")
            break

        if question.lower() == "exit":
            print("MindOS 已退出。")
            break

        if not question:
            continue

        try:
            answer = mindos.chat(question)

        except Exception as error:
            show_error(
                "本次回答失败，请稍后重试。",
                error,
            )
            continue

        print("\nMindOS:", answer)


if __name__ == "__main__":
    main()

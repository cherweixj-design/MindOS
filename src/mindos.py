from typing import List

from src.llm.base import BaseLLM
from src.memory.memory import Memory
from src.prompt.prompt_builder import PromptBuilder
from src.rag.base_retriever import BaseRetriever
from src.rag.retrieval_result import RetrievalResult


class MindOS:
    """Coordinate retrieval, memory, prompting, and the LLM."""

    def __init__(
        self,
        llm: BaseLLM,
        memory: Memory,
        retriever: BaseRetriever,
        prompt_builder: PromptBuilder,
        top_k: int = 3,
        debug: bool = False,
    ):
        self.llm = llm
        self.memory = memory
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.top_k = top_k
        self.debug = debug

    def chat(self, question: str) -> str:
        """Answer a user question with retrieved knowledge."""

        # 1. 检索知识
        retrieval_results = self.retriever.search(
            question=question,
            top_k=self.top_k,
        )

        # 2. Debug 模式下显示文本、相似度和来源
        if self.debug:
            self._show_retrieval_results(
                retrieval_results
            )

        # 3. 提取知识文本給 PromptBuilder
        knowledge = [
            result.text
            for result in retrieval_results
        ]

        # 4. 组合提示词、知识、历史和当前问题
        messages = self.prompt_builder.build(
            history=self.memory.get(),
            knowledge=knowledge,
            question=question,
        )

        # 5. 调用大模型
        answer = self.llm.chat(messages)

        # 6. 有检索结果时追加真实来源（去重、保持首次出现顺序）
        if retrieval_results:
            seen_sources: List[str] = []
            for result in retrieval_results:
                if result.source not in seen_sources:
                    seen_sources.append(result.source)
            answer += f"\n\n来源：{'、'.join(seen_sources)}"

        # 7. 保存当前会话历史
        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        return answer

    def _show_retrieval_results(
        self,
        results: List[RetrievalResult],
    ) -> None:
        """Print retrieved knowledge, similarity scores and sources."""

        print("\n[DEBUG] Retriever 检索结果：")

        if not results:
            print("没有达到相似度要求的知识。")
            return

        for index, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"\n--- Chunk {index} "
                f"| Score: {result.score:.4f} "
                f"| Source: {result.source} ---"
            )
            print(result.text)

from typing import List, Tuple

from src.llm.base import BaseLLM
from src.memory.memory import Memory
from src.prompt.prompt_builder import PromptBuilder
from src.rag.base_retriever import BaseRetriever


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

        # 1. 检索知识及其相似度分数
        retrieval_results = self.retriever.search(
            question=question,
            top_k=self.top_k,
        )

        # 2. Debug 模式下显示文本和相似度
        if self.debug:
            self._show_retrieval_results(
                retrieval_results
            )

        # 3. PromptBuilder 只需要文本，不需要分数
        knowledge = [
            text
            for text, score in retrieval_results
        ]

        # 4. 组合提示词、知识、历史和当前问题
        messages = self.prompt_builder.build(
            history=self.memory.get(),
            knowledge=knowledge,
            question=question,
        )

        # 5. 调用大模型
        answer = self.llm.chat(messages)

        # 6. 保存当前会话历史
        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        return answer

    def _show_retrieval_results(
        self,
        results: List[Tuple[str, float]],
    ) -> None:
        """Print retrieved knowledge and similarity scores."""

        print("\n[DEBUG] Retriever 检索结果：")

        if not results:
            print("没有达到相似度要求的知识。")
            return

        for index, (text, score) in enumerate(
            results,
            start=1,
        ):
            print(
                f"\n--- Chunk {index} "
                f"| Score: {score:.4f} ---"
            )
            print(text)
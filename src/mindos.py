from typing import List, Optional

from src.llm.base import BaseLLM
from src.memory.memory import Memory
from src.prompt.prompt_builder import PromptBuilder
from src.query.base_query_rewriter import BaseQueryRewriter, QueryRewriteError
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
        rewriter: Optional[BaseQueryRewriter] = None,
    ):
        self.llm = llm
        self.memory = memory
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.top_k = top_k
        self.debug = debug
        self.rewriter = rewriter

    def chat(self, question: str) -> str:
        """Answer a user question with retrieved knowledge."""

        # 0. 读取当前历史（rewriter 需要）
        history = self.memory.get()

        # 1. Query Rewriting
        search_question = question
        if self.rewriter is not None:
            try:
                search_question = self.rewriter.rewrite(
                    question=question,
                    history=list(history),
                )
            except QueryRewriteError:
                search_question = question

        # 2. 用改写问题检索
        retrieval_results = self.retriever.search(
            question=search_question,
            top_k=self.top_k,
        )

        # 3. Debug 模式下显示文本、相似度和来源
        if self.debug:
            self._show_retrieval_results(
                retrieval_results
            )

        # 4. 提取知识文本給 PromptBuilder
        knowledge = [
            result.text
            for result in retrieval_results
        ]

        # 5. 组合提示词、知识、历史和当前问题（原始问题）
        messages = self.prompt_builder.build(
            history=history,
            knowledge=knowledge,
            question=question,
        )

        # 6. 调用大模型
        answer = self.llm.chat(messages)

        # 7. 有检索结果时追加真实来源（去重、保持首次出现顺序）
        if retrieval_results:
            seen_sources: List[str] = []
            for result in retrieval_results:
                if result.source not in seen_sources:
                    seen_sources.append(result.source)
            answer += f"\n\n来源：{'、'.join(seen_sources)}"

        # 8. 保存当前会话历史（原始问题）
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

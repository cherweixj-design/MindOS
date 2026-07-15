"""LLM-based query rewriting implementation."""

from typing import Dict, List

from src.llm.base import BaseLLM
from .base_query_rewriter import BaseQueryRewriter, QueryRewriteError


QUERY_REWRITE_SYSTEM_PROMPT = """
你是一个专门改写问题的AI。

请根据对话历史，将当前用户问题改写为一个可以独立理解的检索问题。

规则：
1. 只输出改写后的问题，不要输出任何其他文字。
2. 不要回答用户的问题。
3. 不要添加"改写后："之类的前缀或标签。
4. 不要编造对话历史中不存在的信息。
5. 使用当前问题所使用的语言。
6. 如果当前问题本身已经是独立的，可以直接原样输出。
"""


class LLMQueryRewriter(BaseQueryRewriter):
    """Rewrite questions using an LLM."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def rewrite(
        self,
        question: str,
        history: List[Dict[str, str]],
    ) -> str:
        """Rewrite a question using conversation history.

        Returns the original question when there is no history.
        Falls back to the original question when the LLM returns blank output.
        """
        if not history:
            return question

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": QUERY_REWRITE_SYSTEM_PROMPT.strip(),
            },
            *history,
            {
                "role": "user",
                "content": question,
            },
        ]

        try:
            rewritten = self.llm.chat(messages)
        except Exception as error:
            raise QueryRewriteError(
                "Failed to rewrite query."
            ) from error

        cleaned = rewritten.strip()

        if not cleaned:
            return question

        return cleaned

from typing import Dict, List

from .system import SYSTEM_PROMPT


class PromptBuilder:
    """Build messages for the language model."""

    def build(
        self,
        history: List[Dict[str, str]],
        knowledge: List[str],
        question: str,
    ) -> List[Dict[str, str]]:
        """Combine system prompt, knowledge, history, and question."""

        if knowledge:
            context = "\n\n".join(knowledge)
        else:
            context = "没有检索到相关知识。"

        system_content = f"""
{SYSTEM_PROMPT.strip()}

# 知识库内容

{context}

# 知识库使用规则

- 优先根据以上知识回答。
- 如果知识库不足以回答问题，请明确说明。
- 不要假装知识库中存在没有提供的信息。
"""

        messages = [
            {
                "role": "system",
                "content": system_content,
            }
        ]

        messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        return messages
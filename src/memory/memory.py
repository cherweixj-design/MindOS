from typing import Dict, List, Optional


class Memory:
    """Conversation memory."""

    def __init__(
        self,
        max_turns: Optional[int] = None,
    ):
        if max_turns is not None:
            if not (
                isinstance(max_turns, int)
                and not isinstance(max_turns, bool)
                and max_turns > 0
            ):
                raise ValueError(
                    "max_turns must be a positive integer or None."
                )
        self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []

    def add(self, role: str, content: str):
        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        if self.max_turns is not None:
            limit = self.max_turns * 2
            while len(self.messages) > limit:
                del self.messages[:2]

    def get(self):
        return [
            message.copy()
            for message in self.messages
        ]

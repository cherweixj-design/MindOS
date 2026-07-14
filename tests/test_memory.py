import pytest

from src.memory.memory import Memory


class TestMemory:
    """Tests for the Memory class."""

    def test_initial_history_is_empty(self):
        memory = Memory()
        assert memory.get() == []

    def test_add_saves_user_message(self):
        memory = Memory()
        memory.add("user", "Hello")
        history = memory.get()
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "Hello"}

    def test_add_saves_assistant_message(self):
        memory = Memory()
        memory.add("assistant", "Hi there")
        history = memory.get()
        assert len(history) == 1
        assert history[0] == {"role": "assistant", "content": "Hi there"}

    def test_get_returns_saved_messages(self):
        memory = Memory()
        memory.add("user", "What is the policy?")
        memory.add("assistant", "The policy is 15 days.")
        history = memory.get()
        assert len(history) == 2

    def test_message_order_is_preserved(self):
        memory = Memory()
        memory.add("user", "Q1")
        memory.add("assistant", "A1")
        memory.add("user", "Q2")
        memory.add("assistant", "A2")
        history = memory.get()
        assert history[0] == {"role": "user", "content": "Q1"}
        assert history[1] == {"role": "assistant", "content": "A1"}
        assert history[2] == {"role": "user", "content": "Q2"}
        assert history[3] == {"role": "assistant", "content": "A2"}

    def test_multiple_adds_accumulate(self):
        memory = Memory()
        for i in range(10):
            memory.add("user", f"message {i}")
        assert len(memory.get()) == 10

    def test_messages_are_dicts_with_role_and_content(self):
        memory = Memory()
        memory.add("user", "test")
        msg = memory.get()[0]
        assert isinstance(msg, dict)
        assert "role" in msg
        assert "content" in msg

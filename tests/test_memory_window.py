"""Tests for Memory sliding window."""

import pytest

from src.memory.memory import Memory


class TestMemoryWindowParams:
    """Tests for max_turns parameter handling."""

    def test_no_params_unlimited(self):
        memory = Memory()
        for _ in range(20):
            memory.add("user", "msg")
            memory.add("assistant", "msg")
        assert len(memory.get()) == 40

    @pytest.mark.parametrize("bad_value", [0, -1, 1.5, "2", True, False])
    def test_invalid_max_turns_raises_value_error(self, bad_value):
        with pytest.raises(ValueError):
            Memory(max_turns=bad_value)

    @pytest.mark.parametrize("good_value", [None, 1, 2])
    def test_valid_max_turns(self, good_value):
        memory = Memory(max_turns=good_value)
        assert memory.max_turns == good_value


class TestMemoryWindowTrim:
    """Tests for window trimming behavior."""

    def test_max_turns_1_keeps_last_two(self):
        memory = Memory(max_turns=1)
        memory.add("user", "U1")
        memory.add("assistant", "A1")
        memory.add("user", "U2")
        memory.add("assistant", "A2")
        history = memory.get()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "U2"}
        assert history[1] == {"role": "assistant", "content": "A2"}

    def test_max_turns_2_keeps_last_four(self):
        memory = Memory(max_turns=2)
        memory.add("user", "U1")
        memory.add("assistant", "A1")
        memory.add("user", "U2")
        memory.add("assistant", "A2")
        memory.add("user", "U3")
        memory.add("assistant", "A3")
        history = memory.get()
        assert len(history) == 4
        assert history[0] == {"role": "user", "content": "U2"}
        assert history[1] == {"role": "assistant", "content": "A2"}
        assert history[2] == {"role": "user", "content": "U3"}
        assert history[3] == {"role": "assistant", "content": "A3"}

    def test_order_preserved_after_trim(self):
        memory = Memory(max_turns=2)
        memory.add("user", "U1")
        memory.add("assistant", "A1")
        memory.add("user", "U2")
        memory.add("assistant", "A2")
        memory.add("user", "U3")
        memory.add("assistant", "A3")
        history = memory.get()
        for i in range(1, len(history)):
            assert history[i - 1] != history[i]

    def test_partial_turn_not_trimmed_too_early(self):
        memory = Memory(max_turns=2)
        memory.add("user", "U1")
        memory.add("assistant", "A1")
        memory.add("user", "U2")
        memory.add("assistant", "A2")
        memory.add("user", "U3")
        history = memory.get()
        assert len(history) == 3
        assert history[0] == {"role": "user", "content": "U2"}
        assert history[1] == {"role": "assistant", "content": "A2"}
        assert history[2] == {"role": "user", "content": "U3"}

    def test_partial_turn_completed(self):
        memory = Memory(max_turns=2)
        memory.add("user", "U1")
        memory.add("assistant", "A1")
        memory.add("user", "U2")
        memory.add("assistant", "A2")
        memory.add("user", "U3")
        memory.add("assistant", "A3")
        history = memory.get()
        assert len(history) == 4
        assert history[0] == {"role": "user", "content": "U2"}
        assert history[1] == {"role": "assistant", "content": "A2"}
        assert history[2] == {"role": "user", "content": "U3"}
        assert history[3] == {"role": "assistant", "content": "A3"}


class TestMemoryWindowGetCopy:
    """Tests for get() defensive copy."""

    def test_get_copy_append_safe(self):
        memory = Memory()
        memory.add("user", "U1")
        history = memory.get()
        history.append({"role": "assistant", "content": "A1"})
        assert len(memory.get()) == 1

    def test_get_copy_dict_safe(self):
        memory = Memory()
        memory.add("user", "U1")
        history = memory.get()
        history[0]["content"] = "modified"
        assert memory.get()[0]["content"] == "U1"

    def test_get_returns_new_object(self):
        memory = Memory()
        memory.add("user", "U1")
        r1 = memory.get()
        r2 = memory.get()
        assert r1 == r2
        assert r1 is not r2

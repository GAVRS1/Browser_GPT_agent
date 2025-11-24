import unittest
from unittest.mock import patch

from agent import agent_loop


class _DummyMessage:
    def __init__(self, content: str):
        self.content = content


class _DummyChoice:
    def __init__(self, content: str):
        self.message = _DummyMessage(content)


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [_DummyChoice(content)]


class _DummyModels:
    class _Item:
        id = "dummy-model"

    def list(self):
        return type("_Container", (), {"data": [self._Item()]})()


class _DummyChatCompletions:
    def __init__(self):
        self.last_messages = []

    def create(self, model, messages, temperature):
        self.last_messages = messages
        plan = (
            "1. Открыть магазин и найти поле поиска.\n"
            "2. Ввести название товара без жёстких селекторов.\n"
            "3. Открыть подходящую карточку, добавить в корзину и проверить, что она не пуста."
        )
        return _DummyResponse(plan)


class _DummyClient:
    def __init__(self):
        self.chat = type("_Chat", (), {"completions": _DummyChatCompletions()})()
        self.models = _DummyModels()


class PlanningTests(unittest.TestCase):
    def test_add_item_goal_has_generic_plan(self):
        dummy_client = _DummyClient()

        with patch.object(agent_loop, "get_client", return_value=dummy_client):
            plan = agent_loop._run_llm_planning("найди и добавь товар")

        messages = dummy_client.chat.completions.last_messages
        self.assertTrue(messages)
        system_prompt = messages[0]["content"]
        self.assertIn("Never rely on hardcoded DOM structure", system_prompt)

        self.assertIn("добавить в корзину", plan)
        self.assertNotIn("#", plan)
        self.assertNotIn("css", plan.lower())


if __name__ == "__main__":
    unittest.main()

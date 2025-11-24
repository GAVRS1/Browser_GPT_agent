import unittest
from io import StringIO

from loguru import logger

from agent.agent_loop import _is_risky_goal
from agent.browser_tools import (
    _collect_candidates_for_logging,
    _describe_locator,
    _log_interaction,
)


class FakeLocator:
    def __init__(self, text: str, selector_repr: str = "#hard-selector") -> None:
        self._text = text
        self._selector_repr = selector_repr

    @property
    def first(self):
        return self

    def is_visible(self, timeout: int = 1500) -> bool:  # noqa: ARG002
        return True

    def evaluate(self, expr: str):  # noqa: ANN001, ARG002
        if "tagName" in expr:
            return "button"
        raise RuntimeError("unexpected evaluation")

    def inner_text(self, timeout: int = 800) -> str:  # noqa: ARG002
        return self._text

    def get_attribute(self, name: str):  # noqa: ANN001
        return None

    def __repr__(self) -> str:
        return f"<Locator selector={self._selector_repr}>"


class SafetyAndLoggingTests(unittest.TestCase):
    def test_risky_goal_requires_confirmation(self) -> None:
        self.assertTrue(_is_risky_goal("удали все письма"))
        self.assertTrue(_is_risky_goal("оформи и оплати заказ"))
        self.assertFalse(_is_risky_goal("прочитай письмо"))

    def test_logs_use_readable_descriptions(self) -> None:
        locator = FakeLocator("Подтвердить оплату")
        candidates, chosen = _collect_candidates_for_logging([locator])

        buffer = StringIO()
        sink_id = logger.add(buffer, level="INFO")
        try:
            _log_interaction(
                action="click",
                query="Оплатить заказ",
                candidates=candidates,
                chosen_summary=_describe_locator(chosen),
                result="Клик выполнен",
            )
        finally:
            logger.remove(sink_id)

        log_output = buffer.getvalue()
        self.assertIn("Оплатить заказ", log_output)
        self.assertIn("Подтвердить оплату", log_output)
        self.assertNotIn("#hard-selector", log_output)


if __name__ == "__main__":
    unittest.main()

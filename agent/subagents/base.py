from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.subagents import SubAgentResult


class BaseSubAgent:
    """
    Базовый интерфейс под-агентов.

    У каждого под-агента есть:
    - name: человекочитаемое имя
    - matches(goal): возвращает True, если этот под-агент подходит для цели.
    - run(goal, plan): запускает под-агента с текстом цели и планом LLM.
    """

    name: str

    def matches(self, goal: str) -> bool:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def run(self, goal: str, plan: str) -> SubAgentResult:  # pragma: no cover - интерфейс
        raise NotImplementedError

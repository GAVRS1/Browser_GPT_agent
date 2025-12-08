from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SubAgentResult:
    """
    Унифицированный результат работы под-агента.

    success: итоговое булево — получилось ли в целом выполнить задачу.
    status:  "completed" | "failed" | "needs_input" | другое служебное состояние.
    details: человекочитаемый отчёт о том, что сделал агент.
    error:   строка с кодом/описанием ошибки (если есть).
    """

    success: bool
    status: str
    details: str
    error: Optional[str] = None


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

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from .yandex_mail import YandexMailSubAgent
from .yandex_lavka import YandexLavkaSubAgent
from .hhru import HhRuSubAgent


@dataclass
class SubAgentResult:
    """
    Унифицированный результат работы под-агента.

    success: итоговое булево — получилось ли в целом выполнить задачу.
    status:  "completed" | "failed" | "needs_input" | другое служебное состояние.
    details: человекочитаемый отчет о том, что сделал агент.
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
    - name: человекочитаемое имя ("Яндекс.Почта", "Интернет-магазин" и т.п.)
    - matches(goal): возвращает True, если этот под-агент подходит для цели.
    - run(goal, plan): запускает под-агента с текстом цели и планом LLM.
    """

    name: str

    def matches(self, goal: str) -> bool:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def run(self, goal: str, plan: str) -> SubAgentResult:  # pragma: no cover - интерфейс
        raise NotImplementedError


# Здесь регистрируем все специализированные под-агенты.
_SUBAGENTS: List[BaseSubAgent] = [
    YandexMailSubAgent(),   # обработка писем, спам и т.д.
    YandexLavkaSubAgent(),  # подготовка поиска и страницы Лавки
    HhRuSubAgent(),
]


def pick_subagent(goal: str) -> Optional[BaseSubAgent]:
    """
    Выбирает подходящего под-агента по тексту цели.

    Если ни один под-агент не подходит — возвращает None,
    и тогда основной агент будет работать "вручную" через общие инструменты.
    """

    lowered = goal.lower()
    for subagent in _SUBAGENTS:
        try:
            if subagent.matches(lowered):
                logger.info(f"[subagents] Picked subagent: {subagent.name}")
                return subagent
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[subagents] Error while matching {subagent}: {exc}")

    logger.info("[subagents] No specialized subagent matched this goal")
    return None


__all__ = ["SubAgentResult", "BaseSubAgent", "pick_subagent"]

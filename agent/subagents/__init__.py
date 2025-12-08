from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

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


from .base import BaseSubAgent
from .rental import RentalPaymentSubAgent, RentalReservationSubAgent, RentalSearchSubAgent


# Здесь регистрируем компетенции под аренду.
_SUBAGENTS: List[BaseSubAgent] = [
    RentalSearchSubAgent(),
    RentalReservationSubAgent(),
    RentalPaymentSubAgent(),
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

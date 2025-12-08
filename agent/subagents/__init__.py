from __future__ import annotations

from typing import List, Optional

from loguru import logger

from .base import BaseSubAgent, SubAgentResult
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

from __future__ import annotations

from typing import List

from loguru import logger

from agent.subagents.base import BaseSubAgent, SubAgentResult


class RentalSearchSubAgent(BaseSubAgent):
    """Подготовка поиска свободных слотов аренды."""

    name = "Поиск слотов аренды"

    _keywords: List[str] = [
        "найди",
        "найти",
        "поиск",
        "подбери",
        "подбирать",
        "слот",
        "слоты",
        "окно",
        "окна",
        "доступное время",
        "расписание",
        "свободн",
        "аренда",
        "бронь",
        "брони",
        "booking",
        "available slot",
        "availability",
    ]

    def matches(self, goal: str) -> bool:
        lowered = goal.lower()
        return any(k in lowered for k in self._keywords)

    def run(self, goal: str, plan: str) -> SubAgentResult:
        logger.info("[rental] Preparing search workflow")
        hint = (
            "Ты готовишь интерфейс для поиска слотов аренды.\n"
            "- Оцени текущую страницу: есть ли фильтры по датам, времени, локации.\n"
            "- Используй текстовые поля, выпадающие списки и фильтры, чтобы отобразить актуальные слоты.\n"
            "- Покажи пользователю найденные варианты, отмечая время, цену и ограничения."
        )
        return SubAgentResult(success=False, status="prepared", details=hint)


class RentalReservationSubAgent(BaseSubAgent):
    """Подсказки для оформления брони слота."""

    name = "Бронирование слота"

    _keywords: List[str] = [
        "заброни",
        "бронь",
        "резерв",
        "запиши",
        "записать",
        "book",
        "reserve",
        "оформи брон",
        "подтверди брон",
        "закрепи слот",
        "подтверждение брони",
    ]

    def matches(self, goal: str) -> bool:
        lowered = goal.lower()
        return any(k in lowered for k in self._keywords)

    def run(self, goal: str, plan: str) -> SubAgentResult:
        logger.info("[rental] Guiding reservation workflow")
        hint = (
            "Ты занимаешься оформлением брони.\n"
            "- Убедись, что выбран нужный слот: дата, время, длительность, локация.\n"
            "- Заполняй формы с данными пользователя, не переходя к оплате без запроса.\n"
            "- Перед подтверждением покажи, какая бронь будет создана, и остановись перед финальной отправкой."
        )
        return SubAgentResult(success=False, status="prepared", details=hint)


class RentalPaymentSubAgent(BaseSubAgent):
    """Безопасные подсказки для этапа оплаты аренды."""

    name = "Оплата аренды"

    _keywords: List[str] = [
        "оплат",
        "плати",
        "оплата",
        "payment",
        "checkout",
        "счет",
        "invoice",
        "к оплате",
        "закрыть аренду",
        "пролонгация",
        "продлить аренду",
    ]

    def matches(self, goal: str) -> bool:
        lowered = goal.lower()
        return any(k in lowered for k in self._keywords)

    def run(self, goal: str, plan: str) -> SubAgentResult:
        logger.info("[rental] Payment safeguards in place")
        hint = (
            "Ты работаешь с оплатой аренды.\n"
            "- Проверь, что корзина/форма оплаты содержит правильный слот и сумму.\n"
            "- Не нажимай окончательную кнопку оплаты без явного подтверждения пользователя.\n"
            "- Подготовь экран оплаты и перечисли шаги, которые остались для завершения транзакции."
        )
        return SubAgentResult(success=False, status="needs_input", details=hint, error=None)

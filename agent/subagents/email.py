from __future__ import annotations

from typing import List

from loguru import logger

from agent.subagents import BaseSubAgent, SubAgentResult


EMAIL_CONTEXT_KEYWORDS: List[str] = [
    "почта",
    "email",
    "e-mail",
    "емейл",
    "имейл",
    "inbox",
    "входящие",
    "письмо",
    "письма",
    "message",
    "messages",
    "mail",
]


def _has_keyword(goal: str, keywords: List[str]) -> bool:
    return any(keyword in goal for keyword in keywords)


def _matches_email_intent(goal: str, action_keywords: List[str]) -> bool:
    lowered = goal.lower()
    return _has_keyword(lowered, action_keywords) and _has_keyword(lowered, EMAIL_CONTEXT_KEYWORDS)


class EmailViewSubAgent(BaseSubAgent):
    """Навигация по списку писем и просмотру содержания."""

    name = "Просмотр почты"

    _action_keywords: List[str] = [
        "посмотри",
        "посмотреть",
        "проверь",
        "проверить",
        "открой",
        "открыть",
        "просмотр",
        "входящ",
        "новые",
        "новых",
        "inbox",
        "view",
        "check",
        "read",
    ]

    def matches(self, goal: str) -> bool:
        return _matches_email_intent(goal, self._action_keywords)

    def run(self, goal: str, plan: str) -> SubAgentResult:
        logger.info("[email] Preparing inbox review workflow")
        hint = (
            "Ты занимаешься просмотром почты.\n"
            "- Оцени список писем: отправитель, тема, дата, метки прочитано/непрочитано.\n"
            "- Открой письмо, если нужно увидеть содержание или вложения.\n"
            "- Сообщи пользователю краткое резюме найденных писем и возможные следующие шаги."
        )
        return SubAgentResult(success=False, status="prepared", details=hint)


class EmailComposeSubAgent(BaseSubAgent):
    """Подготовка нового письма."""

    name = "Создание письма"

    _action_keywords: List[str] = [
        "напиши",
        "написать",
        "создай",
        "создать",
        "новое письмо",
        "compose",
        "draft",
        "черновик",
        "письмо",
        "message",
    ]

    def matches(self, goal: str) -> bool:
        return _matches_email_intent(goal, self._action_keywords)

    def run(self, goal: str, plan: str) -> SubAgentResult:
        logger.info("[email] Preparing compose workflow")
        hint = (
            "Ты готовишь новое письмо.\n"
            "- Заполни поле получателя(ей), тему и тело письма.\n"
            "- Уточни, нужны ли вложения или копии (CC/BCC).\n"
            "- Перед отправкой покажи итоговый черновик и попроси подтверждение."
        )
        return SubAgentResult(success=False, status="prepared", details=hint)


class EmailReplySubAgent(BaseSubAgent):
    """Ответ и пересылка писем."""

    name = "Ответ на письмо"

    _action_keywords: List[str] = [
        "ответь",
        "ответить",
        "ответ",
        "reply",
        "forward",
        "перешли",
        "переслать",
        "цитируй",
        "re:",
    ]

    def matches(self, goal: str) -> bool:
        return _matches_email_intent(goal, self._action_keywords)

    def run(self, goal: str, plan: str) -> SubAgentResult:
        logger.info("[email] Preparing reply workflow")
        hint = (
            "Ты отвечаешь на письмо или пересылаешь его.\n"
            "- Убедись, что открыт правильный тред и выбран адресат.\n"
            "- Сохрани важный контекст из исходного письма, добавь ответ.\n"
            "- Перед отправкой покажи итоговый текст и уточни подтверждение."
        )
        return SubAgentResult(success=False, status="prepared", details=hint)


class EmailDeleteSubAgent(BaseSubAgent):
    """Удаление писем с проверкой пользователя."""

    name = "Удаление писем"

    _action_keywords: List[str] = [
        "удали",
        "удалить",
        "удаление",
        "delete",
        "trash",
        "корзин",
        "очисти",
        "очистить",
    ]

    def matches(self, goal: str) -> bool:
        return _matches_email_intent(goal, self._action_keywords)

    def run(self, goal: str, plan: str) -> SubAgentResult:
        logger.info("[email] Preparing delete workflow")
        hint = (
            "Ты готовишь удаление писем.\n"
            "- Проверь список выбранных писем и покажи пользователю, что будет удалено.\n"
            "- Сообщи, можно ли восстановить письма из корзины.\n"
            "- Остановись перед финальным удалением и запроси явное подтверждение."
        )
        return SubAgentResult(success=False, status="needs_input", details=hint)

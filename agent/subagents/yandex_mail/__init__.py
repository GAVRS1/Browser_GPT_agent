from __future__ import annotations

import re
from typing import Any
import contextlib
import os

from loguru import logger

from browser.context import get_page
from agent.subagents.utils import matches_domain
from agent.subagents.yandex_mail.common import open_mailbox
from agent.subagents.yandex_mail.read_delete import run_read_delete_flow
from agent.subagents.yandex_mail.reply import run_reply_flow
from agent.subagents.yandex_mail.compose import run_compose_flow  # новый режим


DEBUG_THOUGHTS = os.getenv("AGENT_DEBUG_THOUGHTS", "1") != "0"


def log_thought(prefix: str, text: str) -> None:
    """
    Единая точка для вывода "мыслей" под-агента в логи и консоль.
    prefix — короткое имя агента: 'yandex-mail', 'yandex-mail-spam' и т.п.
    """
    if not text:
        return
    logger.info(f"[{prefix}] thought: {text}")
    if DEBUG_THOUGHTS:
        print(f"\n🤖 {prefix} думает:\n{text.strip()}\n")


class YandexMailSubAgent:
    """
    Специализированный под-агент для работы с Яндекс.Почтой.

    ВАЖНО:
    - нет жёстко зашитых селекторов под конкретную разметку писем;
    - нет готовых “скриптов” удаления спама;
    - решения принимаются динамически на основе DOM + цели пользователя + плана LLM.
    """

    name = "Яндекс.Почта"

    _keywords = [
        "яндекс почт",
        "yandex mail",
        "mail.yandex",
        "входящие",
        "почту",
        "почта",
        "спам",
        "письма",
    ]

    _domains = ["mail.yandex.ru", "mail.yandex.com", "mail.yandex"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def matches(self, goal: str) -> bool:
        lowered = goal.lower()
        return matches_domain(lowered, self._domains) or any(
            k in lowered for k in self._keywords
        )

    def run(self, goal: str, plan: str):
        """
        Основная точка входа под-агента.

        1. Открываем/фокусируем Яндекс.Почту.
        2. Если цель — ответить на письмо, запускаем режим ответа.
        3. Если цель — написать новое письмо, запускаем режим написания.
        4. В остальных случаях: читаем N писем, классифицируем спам, решаем об удалении.
        """
        from agent.subagents import SubAgentResult  # локальный импорт, чтобы не ловить циклы

        log_thought(
            "yandex-mail",
            f"Новая задача для подагента Яндекс.Почта:\n"
            f"Цель: {goal}\n"
            f"План верхнего уровня:\n{plan or '—'}",
        )

        page = get_page()
        with contextlib.suppress(Exception):
            page.bring_to_front()

        login_msg = open_mailbox(page)
        if login_msg is not None:
            log_thought(
                "yandex-mail",
                "Обнаружена страница логина Яндекс.Почты — жду ручной авторизации.",
            )
            return SubAgentResult(
                success=False,
                status="needs_input",
                details=login_msg,
            )

        # 1) Режим "ответь на уже существующее письмо ..."
        if _is_reply_goal(goal):
            reply_text = _extract_reply_text(goal)
            log_thought(
                "yandex-mail",
                f"Определён режим ОТВЕТА на письмо.\n"
                f"Текст ответа (из цели): {reply_text!r}",
            )
            return run_reply_flow(page, goal, plan, reply_text)

        # 2) Режим "создать/отправить НОВОЕ письмо ..."
        if _is_compose_goal(goal):
            reply_text = _extract_reply_text(goal)
            to_addr = _extract_email_address(goal)
            log_thought(
                "yandex-mail",
                "Определён режим НОВОГО письма.\n"
                f"Адрес получателя (из цели): {to_addr!r}\n"
                f"Текст письма (из цели или по умолчанию): {reply_text!r}",
            )
            return run_compose_flow(page, goal, plan, to_addr, reply_text)

        # 3) Режим чтения/чистки почты
        limit = _extract_limit_from_goal(goal, default=1)
        allow_delete = _should_delete(goal)

        log_thought(
            "yandex-mail",
            "Определён режим чтения/чистки почты.\n"
            f"Количество писем: {limit}, удаление разрешено: {bool(allow_delete)}",
        )

        return run_read_delete_flow(page, goal, plan, limit, allow_delete)


# ------------------------------------------------------------------
# Разбор цели (goal) и плана
# ------------------------------------------------------------------
def _extract_limit_from_goal(goal: str, default: int = 1) -> int:
    """
    Пытается вытащить из формулировки цели количество писем для обработки.
    """
    import re as _re

    text = goal.lower()

    m = _re.search(r"последн(?:ие|их)\s+(\d+)\s+письм", text)
    if m:
        try:
            v = int(m.group(1))
            if 1 <= v <= 200:
                return v
        except ValueError:
            pass

    m_any = _re.search(r"(\d+)\s+письм", text)
    if m_any:
        try:
            v = int(m_any.group(1))
            if 1 <= v <= 200:
                return v
        except ValueError:
            pass

    # Любое число в тексте — как fallback
    m_num = _re.search(r"(\d+)", text)
    if m_num:
        try:
            v = int(m_num.group(1))
            if 1 <= v <= 200:
                return v
        except ValueError:
            pass

    return default


def _should_delete(goal: str) -> bool:
    """
    Определяет, разрешено ли вообще что-то удалять.
    """
    text = goal.lower()
    if "не удал" in text:
        return False

    keywords = [
        "удали",
        "удалить",
        "удаляй",
        "почисти",
        "очисти",
        "очистить",
        "в корзину",
        "в спам",
        "spam",
        "delete",
    ]
    return any(k in text for k in keywords)


def _is_reply_goal(goal: str) -> bool:
    """
    Проверяем, просил ли пользователь именно ОТВЕТИТЬ на уже существующее письмо.
    """
    text = goal.lower()
    reply_keywords = [
        "ответь",
        "ответить",
        "ответ на письмо",
        "ответь на письмо",
        "reply",
    ]
    return any(k in text for k in reply_keywords)


def _is_compose_goal(goal: str) -> bool:
    """
    Проверяем, просил ли пользователь НАПИСАТЬ / ОТПРАВИТЬ НОВОЕ письмо.
    """
    text = goal.lower()
    compose_keywords = [
        "напиши письмо",
        "напишите письмо",
        "написать письмо",
        "создай письмо",
        "создать письмо",
        "отправь письмо",
        "отправить письмо",
        "отправь e-mail",
        "отправь email",
        "new email",
        "compose email",
    ]
    return any(k in text for k in compose_keywords)


def _extract_email_address(goal: str) -> str | None:
    """
    Достаём первый похожий на e-mail адрес из текста цели.
    """
    m = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9_.-]+)", goal)
    if m:
        return m.group(1)
    return None


def _extract_reply_text(goal: str) -> str:
    """
    Пытаемся вытащить текст письма из кавычек.
    Пример: '... текст письма \"Привет\" ...' → "Привет".
    Если не нашли — по умолчанию "Привет".
    """
    # Двойные кавычки
    m = re.search(r"\"([^\"]{1,200})\"", goal)
    if m:
        return m.group(1)
    # Одинарные кавычки
    m2 = re.search(r"'([^']{1,200})'", goal)
    if m2:
        return m2.group(1)
    return "Привет"


__all__ = ["YandexMailSubAgent"]

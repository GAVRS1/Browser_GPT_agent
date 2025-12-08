from __future__ import annotations

import contextlib
import re
from typing import List

from agent.debug_thoughts import log_thought
from loguru import logger
from playwright.sync_api import Page

from agent.subagents.yandex_mail.common import (
    MessageDraft,
    collect_previews,
    ensure_mail_list,
    extract_body,
    format_summary,
)


def run_reply_flow(page: Page, goal: str, plan: str, reply_text: str):
    """
    Режим "ответь на письмо ...".
    Открываем первое письмо во входящих и пытаемся ответить текстом из цели.
    """
    from agent.subagents import SubAgentResult

    log_thought(
        "yandex-mail-reply",
        f"Запущен режим ответа на письмо.\n"
        f"Цель: {goal}\n"
        f"План верхнего уровня:\n{plan or '—'}\n"
        f"Текст ответа: {reply_text!r}",
    )

    previews = collect_previews(page, limit=1)
    if not previews:
        msg = "Не удалось найти письма для ответа."
        log_thought("yandex-mail-reply", msg)
        return SubAgentResult(
            success=False,
            status="failed",
            error="no_messages",
            details=msg,
        )

    draft = previews[0]
    logger.info(f"[yandex_mail] Replying to message #{draft.index}: {draft.subject!r}")
    log_thought(
        "yandex-mail-reply",
        f"Отвечаю на письмо #{draft.index}: {draft.subject!r}",
    )

    with contextlib.suppress(Exception):
        draft.locator.scroll_into_view_if_needed()
    with contextlib.suppress(Exception):
        draft.locator.click(timeout=3000)
        page.wait_for_timeout(1200)

    draft.body = extract_body(page)

    sent = _reply_to_open_message(page, reply_text)
    draft.reply_sent = sent
    draft.reason = (
        f"Ответ с текстом {reply_text!r} "
        f"{'успешно отправлен' if sent else 'не удалось отправить'}"
    )

    log_thought(
        "yandex-mail-reply",
        f"Результат отправки ответа для письма '{draft.subject}': "
        f"{'успех' if sent else 'не удалось'}",
    )

    with contextlib.suppress(Exception):
        page.go_back()
        page.wait_for_timeout(800)
    ensure_mail_list(page)

    summary = format_summary([draft], plan, goal)
    log_thought("yandex-mail-reply", f"Итог по сценарию ответа:\n{summary}")
    return SubAgentResult(
        success=sent,
        status="completed" if sent else "failed",
        details=summary,
    )


def _reply_to_open_message(page: Page, text: str) -> bool:
    """
    Пытаемся нажать «Ответить», ввести текст и нажать «Отправить».
    """
    logger.info("[yandex_mail] Trying to reply to current message…")

    # Ищем кнопку "Ответить"
    clicked = False
    reply_patterns = re.compile("ответить|reply", re.IGNORECASE)
    reply_candidates = [
        page.get_by_role("button", name=reply_patterns),
        page.get_by_text(reply_patterns),
    ]
    for loc in reply_candidates:
        with contextlib.suppress(Exception):
            btn = loc.first
            if btn.is_visible(timeout=1500):
                btn.click(timeout=2000)
                page.wait_for_timeout(800)
                clicked = True
                break

    if not clicked:
        logger.warning("[yandex_mail] Reply button not found")
        return False

    # Ищем поле для ввода текста — приоритет contenteditable, затем textarea,
    # и только потом generic textbox (с фильтром по placeholder/aria-label).
    field = None

    # 1) contenteditable (обычно тело письма)
    with contextlib.suppress(Exception):
        editable = page.locator("[contenteditable='true']")
        if editable.count() > 0:
            for i in range(editable.count()):
                cand = editable.nth(i)
                if cand.is_visible(timeout=2000):
                    field = cand
                    break

    # 2) textarea
    if field is None:
        with contextlib.suppress(Exception):
            areas = page.locator("textarea")
            if areas.count() > 0:
                for i in range(areas.count()):
                    cand = areas.nth(i)
                    if cand.is_visible(timeout=2000):
                        field = cand
                        break

    # 3) generic textboxes, но пропускаем "Кому"/"Тема"
    if field is None:
        with contextlib.suppress(Exception):
            textboxes = page.get_by_role("textbox")
            count = textboxes.count()
            for i in range(count):
                cand = textboxes.nth(i)
                if not cand.is_visible(timeout=2000):
                    continue
                placeholder = ""
                aria_label = ""
                with contextlib.suppress(Exception):
                    placeholder = (cand.get_attribute("placeholder") or "").lower()
                    aria_label = (cand.get_attribute("aria-label") or "").lower()
                bad_words = ("кому", "тема", "subject", "to:")
                if any(b in placeholder for b in bad_words) or any(
                    b in aria_label for b in bad_words
                ):
                    continue
                field = cand
                break

    if field is None:
        logger.warning("[yandex_mail] Reply text field not found")
        return False

    with contextlib.suppress(Exception):
        field.click()
        field.fill("")
    with contextlib.suppress(Exception):
        field.type(text, delay=50)

    # Ищем кнопку "Отправить"
    send_clicked = False
    send_patterns = re.compile("отправить|send", re.IGNORECASE)
    send_candidates = [
        page.get_by_role("button", name=send_patterns),
        page.get_by_text(send_patterns),
    ]
    for loc in send_candidates:
        with contextlib.suppress(Exception):
            btn = loc.first
            if btn.is_visible(timeout=2000):
                btn.click(timeout=2000)
                page.wait_for_timeout(1200)
                send_clicked = True
                break

    if not send_clicked:
        logger.warning("[yandex_mail] Send button not found")
        return False

    logger.info("[yandex_mail] Reply appears to be sent")
    return True

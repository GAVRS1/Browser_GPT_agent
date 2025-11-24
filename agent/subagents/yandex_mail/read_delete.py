from __future__ import annotations

import contextlib
import json
import os
import re
import textwrap
from typing import List, Tuple

from loguru import logger
from playwright.sync_api import Page

from agent.llm_client import get_client
from agent.subagents.yandex_mail.common import (
    MessageDraft,
    collect_previews,
    ensure_mail_list,
    extract_body,
    format_summary,
)


DEBUG_THOUGHTS = os.getenv("AGENT_DEBUG_THOUGHTS", "1") != "0"


def log_thought(prefix: str, text: str) -> None:
    if not text:
        return
    logger.info(f"[{prefix}] thought: {text}")
    if DEBUG_THOUGHTS:
        print(f"\n🤖 {prefix} думает:\n{text.strip()}\n")


def run_read_delete_flow(
    page: Page,
    goal: str,
    plan: str,
    limit: int,
    allow_delete: bool,
):
    """
    Режим чтения/чистки почты:
    - читает N писем,
    - решает, спам или нет,
    - учитывает цель и план агента, чтобы решить, удалять ли.
    """
    from agent.subagents import SubAgentResult

    log_thought(
        "yandex-mail",
        f"Запускаю режим чтения/чистки почты.\n"
        f"Цель: {goal}\n"
        f"План верхнего уровня:\n{plan or '—'}\n"
        f"Лимит писем: {limit}, удаление разрешено: {bool(allow_delete)}",
    )

    previews = collect_previews(page, limit=max(limit, 10))
    if not previews:
        msg = "Не удалось обнаружить список писем в интерфейсе почты."
        log_thought("yandex-mail", msg)
        return SubAgentResult(
            success=False,
            status="failed",
            error="no_messages",
            details=msg,
        )

    inspected: List[MessageDraft] = []
    for draft in previews[:limit]:
        inspected.append(
            _inspect_message(
                page=page,
                draft=draft,
                goal=goal,
                plan=plan,
                allow_delete=allow_delete,
            )
        )

    summary = format_summary(inspected, plan, goal)
    log_thought("yandex-mail", f"Итог по обработке писем:\n{summary}")
    return SubAgentResult(
        success=True,
        status="completed",
        details=summary,
    )


def _inspect_message(
    page: Page,
    draft: MessageDraft,
    goal: str,
    plan: str,
    allow_delete: bool,
) -> MessageDraft:
    """
    Открывает письмо, извлекает текст, классифицирует спам/не спам
    и принимает решение об удалении.
    """
    logger.info(f"[yandex_mail] Inspecting message #{draft.index}: {draft.subject!r}")
    log_thought(
        "yandex-mail",
        f"Читаю письмо #{draft.index}: {draft.subject!r}",
    )

    # 1. Открываем письмо
    with contextlib.suppress(Exception):
        draft.locator.scroll_into_view_if_needed()
    with contextlib.suppress(Exception):
        draft.locator.click(timeout=3000)
        page.wait_for_timeout(1200)

    # 2. Читаем тело
    body_text = extract_body(page)
    draft.body = body_text

    if body_text:
        log_thought(
            "yandex-mail",
            f"Тело письма (обрезано до 500 символов):\n{body_text[:500]}",
        )
    else:
        log_thought("yandex-mail", "Не удалось извлечь текст тела письма.")

    # 3. Базовая классификация спама (контент письма)
    spam, spam_reason = _classify_spam(draft)
    draft.spam = spam
    draft.reason = spam_reason

    # 4. Решение об удалении: LLM видит и goal, и план, и спам-флаг
    if allow_delete:
        delete_flag, delete_reason = _decide_deletion(draft, goal, plan)
    else:
        delete_flag, delete_reason = False, "Удаление не разрешено формулировкой задачи"

    draft.delete_reason = delete_reason

    # 5. Действуем по решению
    if delete_flag:
        deleted = _delete_current_message(page, draft)
        draft.deleted = deleted
        log_thought(
            "yandex-mail",
            f"Решение: удалить письмо '{draft.subject}'. "
            f"Фактический результат удаления: {deleted}. Причина: {delete_reason}",
        )
    else:
        log_thought(
            "yandex-mail",
            f"Решение: оставить письмо '{draft.subject}'. Причина: {delete_reason}",
        )
        with contextlib.suppress(Exception):
            page.go_back()
            page.wait_for_timeout(800)
        ensure_mail_list(page)

    return draft


# -------------------------
# LLM: классификация спама
# -------------------------
def _classify_spam(draft: MessageDraft) -> Tuple[bool, str]:
    """
    Использует LLM, чтобы решить, является ли письмо спамом.
    """
    client = get_client()
    if client is None:
        return False, "LLM недоступен — классификация спама пропущена"

    preview_text = (draft.preview or "—").strip()
    body_short = (draft.body or "—")[:800]

    think_text = textwrap.dedent(
        f"""
        Оцениваю письмо как спам/не спам.

        Тема: {draft.subject or "—"}
        Превью: {preview_text or "—"}
        Фрагмент тела письма (до 800 символов):
        {body_short}
        """
    ).strip()
    log_thought("yandex-mail-spam", think_text)

    prompt = textwrap.dedent(
        f"""
        Ты помощник, который решает, является ли письмо спамом.

        Тебе дан текст письма:
        - тема,
        - превью,
        - тело письма.

        Под спамом понимаются:
        - навязчивая реклама,
        - подозрительные предложения,
        - фишинговые письма,
        - массовые рассылки без очевидной пользы для получателя.

        Верни JSON вида:
        {{
          "spam": true/false,
          "reason": "краткое объяснение"
        }}

        Тема: {draft.subject or "—"}
        Превью: {preview_text or "—"}
        Текст письма: {draft.body or "—"}
        """
    ).strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content or "{}"
        log_thought("yandex-mail-spam", f"Ответ LLM по спаму:\n{content}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[yandex_mail] Spam classification failed: {exc}")
        return False, "Ошибка классификации спама"

    spam = False
    reason = ""
    with contextlib.suppress(Exception):
        parsed = json.loads(content)
        spam = bool(parsed.get("spam"))
        reason = str(parsed.get("reason", ""))

    return spam, reason or "нет причины"


# -------------------------
# LLM: решение об удалении
# -------------------------
def _decide_deletion(draft: MessageDraft, goal: str, plan: str) -> Tuple[bool, str]:
    """
    Решение: стоит ли удалять КОНКРЕТНО ЭТО письмо в контексте цели и плана.
    """

    client = get_client()
    if client is None:
        return False, "LLM недоступен — решение об удалении не принято"

    goal_clean = goal.strip()
    plan_clean = (plan or "").strip()

    spam_flag = bool(draft.spam)
    spam_reason = draft.reason or ""

    think_text = textwrap.dedent(
        f"""
        Решаю, удалять ли письмо '{draft.subject or "—"}'.

        Цель пользователя:
        {goal_clean or "—"}

        План агента:
        {plan_clean or "—"}

        Флаг спама: {spam_flag}
        Причина спама: {spam_reason or "—"}
        """
    ).strip()
    log_thought("yandex-mail-delete", think_text)

    prompt = textwrap.dedent(
        f"""
        Ты действуешь как контролёр действий автономного агента в почте.
        Агент уже:
        - прочитал письмо,
        - оценил, спам это или нет.

        Твоя задача — на основе ЦЕЛИ ПОЛЬЗОВАТЕЛЯ и ПЛАНА агента
        решить, нужно ли именно ЭТО письмо удалить.

        Важно:
        - Если цель пользователя говорит "удали эти письма, какие бы они ни были",
          или явно просит удалить все просмотренные письма, — нужно удалять,
          даже если письмо не спам.
        - Если цель говорит "удали спам" — удаляем только спам.
        - Если цель НЕ просит ничего удалять — возвращаем delete = false.
        - Если цель просит удалять письма от конкретного сервиса
          (например, от Кинопоиска), то письма от этого сервиса тоже стоит удалить.

        Ответь в JSON:
        {{
          "delete": true/false,
          "reason": "краткое объяснение на русском"
        }}

        Цель пользователя:
        {goal_clean or "—"}

        План агента:
        {plan_clean or "—"}

        Данные по письму:
        - тема: {draft.subject or "—"}
        - превью: {draft.preview or "—"}
        - спам: {str(spam_flag).lower()}
        - причина_спама: {spam_reason or "—"}
        """
    ).strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content or "{}"
        log_thought(
            "yandex-mail-delete",
            f"Ответ LLM по решению об удалении письма '{draft.subject or '—'}':\n{content}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[yandex_mail] Deletion decision failed: {exc}")
        return False, "Ошибка LLM при принятии решения об удалении"

    delete_flag = False
    reason = ""
    with contextlib.suppress(Exception):
        parsed = json.loads(content)
        delete_flag = bool(parsed.get("delete"))
        reason = str(parsed.get("reason", ""))

    return delete_flag, reason or "нет причины"


# -------------------------
# Удаление / проверка
# -------------------------
def _delete_current_message(page: Page, draft: MessageDraft) -> bool:
    """
    Пытается удалить текущее открытое письмо и проверить, исчезло ли оно из списка.
    Обязательно возвращается на страницу со списком писем.
    """
    logger.info("[yandex_mail] Trying to delete current message…")

    clicked = _try_click_delete_controls(page)
    if not clicked:
        logger.warning("[yandex_mail] Could not find delete control")
        with contextlib.suppress(Exception):
            page.go_back()
            page.wait_for_timeout(800)
        ensure_mail_list(page)
        return False

    # Даём UI время применить действие
    with contextlib.suppress(Exception):
        page.wait_for_timeout(800)

    # Возвращаемся к списку писем
    with contextlib.suppress(Exception):
        page.go_back()
        page.wait_for_timeout(800)
    ensure_mail_list(page)

    # Проверяем, осталась ли тема письма в списке
    still_present = _is_message_still_present(page, draft.subject)
    if still_present:
        logger.warning("[yandex_mail] Message still present after delete attempt")
        return False

    logger.info("[yandex_mail] Message successfully deleted (not found in list anymore)")
    return True


def _try_click_delete_controls(page: Page) -> bool:
    """
    Ищет кнопки/контролы удаления письма и пытается по ним кликнуть.
    """
    patterns = re.compile("удалить|в корзину|спам|delete|trash", re.IGNORECASE)

    candidates = [
        page.get_by_role("button", name=patterns),
        page.get_by_label("Удалить", exact=False),
        page.locator("[aria-label*='Удал']", has_text=re.compile("Удал", re.IGNORECASE)),
        page.locator("[title*='Удал']", has_text=re.compile("Удал", re.IGNORECASE)),
    ]

    for loc in candidates:
        with contextlib.suppress(Exception):
            btn = loc.first
            if btn.is_visible(timeout=1500):
                btn.click(timeout=2000)
                logger.info("[yandex_mail] Delete action triggered via button")
                return True
    return False


def _is_message_still_present(page: Page, subject: str) -> bool:
    """
    Быстро пересобираем список превью и проверяем, осталось ли письмо с таким subject.
    """
    previews = collect_previews(page, limit=50)
    subj_norm = subject.lower().strip()
    for msg in previews:
        if msg.subject.lower().strip() == subj_norm:
            return True
    return False

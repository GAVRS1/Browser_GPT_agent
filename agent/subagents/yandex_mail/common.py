from __future__ import annotations

import contextlib
import re
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from loguru import logger
from playwright.sync_api import Locator, Page, TimeoutError as PlaywrightTimeoutError


# -------------------------
# Модель письма
# -------------------------
@dataclass
class MessageDraft:
    """
    Модель письма в текущей сессии просмотра.

    subject: заголовок письма (первая строка превью).
    preview: короткий текст превью (остальные строки).
    locator: элемент списка, по которому можно кликнуть, чтобы открыть письмо.
    body: текст открытого письма.
    spam: результат классификации (True/False).
    reason: объяснение по спаму или по ответу.
    deleted: получилось ли физически удалить письмо.
    delete_reason: почему агент решил удалить (по спаму / по плану / по запросу).
    reply_sent: получилось ли отправить ответ на письмо.
    """

    index: int
    subject: str
    preview: str
    locator: Locator
    body: Optional[str] = None
    spam: Optional[bool] = None
    reason: Optional[str] = None
    deleted: bool = False
    delete_reason: Optional[str] = None
    reply_sent: bool = False


# Очень общий селектор для элементов, которые потенциально могут быть письмами
_MESSAGE_SELECTOR = "[role='listitem'], [role='row'], article"


# -------------------------
# Навигация / логин
# -------------------------
def open_mailbox(page: Page) -> Optional[str]:
    """
    Переходим на mail.yandex.ru и грубо проверяем, требуется ли логин.

    Если требуется — возвращаем текст, который верхнеуровневый агент покажет пользователю.
    Если нет — возвращаем None.
    """
    logger.info("[yandex_mail] Navigating to Yandex Mail…")
    page.goto("https://mail.yandex.ru/", wait_until="domcontentloaded")

    login_required = False

    # Простейшие эвристики по полям логина/пароля
    with contextlib.suppress(PlaywrightTimeoutError, Exception):
        login_field = page.locator("input[type='email'], input[name='login']").first
        if login_field.is_visible(timeout=2000):
            login_required = True

    with contextlib.suppress(PlaywrightTimeoutError, Exception):
        pass_field = page.locator("input[type='password']").first
        if pass_field.is_visible(timeout=2000):
            login_required = True

    if login_required:
        logger.warning("[yandex_mail] Login required; waiting for user action.")
        return (
            "Откройте вкладку Яндекс.Почты и войдите в аккаунт вручную. "
            "После авторизации повторите запрос — агент продолжит читать и чистить почту."
        )

    logger.info("[yandex_mail] Mailbox appears to be open.")
    return None


def ensure_mail_list(page: Page) -> None:
    """
    Пытаемся гарантированно вернуться на страницу со списком писем (обычно 'Входящие'),
    не используя прямой goto, чтобы не ломать состояние.
    """
    # Сначала пробуем по роли ссылки
    with contextlib.suppress(Exception):
        inbox_link = page.get_by_role(
            "link",
            name=re.compile("входящие", re.IGNORECASE),
        ).first
        if inbox_link.is_visible(timeout=1200):
            inbox_link.click(timeout=2000)
            page.wait_for_timeout(800)
            return

    # Фолбэк — по тексту
    with contextlib.suppress(Exception):
        inbox_text = page.get_by_text(re.compile("Входящие", re.IGNORECASE)).first
        if inbox_text.is_visible(timeout=1200):
            inbox_text.click(timeout=2000)
            page.wait_for_timeout(800)


# -------------------------
# Поиск писем / чтение тела
# -------------------------
def _safe_inner_text(locator: Locator) -> str:
    with contextlib.suppress(Exception):
        txt = locator.inner_text(timeout=1500).strip()
        return txt
    return ""


def _split_subject_preview(text: str) -> Tuple[str, str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "(без темы)", ""

    subject = lines[0]
    preview = " ".join(lines[1:]) if len(lines) > 1 else ""
    return subject[:180], preview[:280]


def collect_previews(page: Page, limit: int) -> List[MessageDraft]:
    """
    Находит элементы, похожие на письма, и строит из них превью.
    """
    with contextlib.suppress(Exception):
        page.wait_for_timeout(1500)

    base = page.locator(_MESSAGE_SELECTOR)
    count = 0
    with contextlib.suppress(Exception):
        count = base.count()

    if count == 0:
        logger.warning("[yandex_mail] No message-like elements found by selector")
        return []

    viewport_width = 0
    with contextlib.suppress(Exception):
        size = page.viewport_size
        if size and size.get("width"):
            viewport_width = size["width"]
        else:
            viewport_width = page.evaluate("window.innerWidth || 0")

    previews: List[MessageDraft] = []

    for raw_idx in range(count):
        item = base.nth(raw_idx)

        with contextlib.suppress(Exception):
            box = item.bounding_box()
        if not box:
            continue

        # Отфильтруем явно "левые" элементы (боковые панели и т.п.)
        if viewport_width > 0:
            too_left = box["x"] < viewport_width * 0.12
            too_narrow = box["width"] < viewport_width * 0.25
            if too_left or too_narrow:
                continue

        text_block = _safe_inner_text(item)
        if not text_block:
            continue

        subject, preview = _split_subject_preview(text_block)

        previews.append(
            MessageDraft(
                index=raw_idx,
                subject=subject,
                preview=preview,
                locator=item,
            )
        )
        if len(previews) >= limit:
            break

    logger.info(f"[yandex_mail] Собрано превью писем: {len(previews)}")
    return previews


def extract_body(page: Page) -> str:
    """
    Пытается вытащить основной текст открытого письма.
    """
    candidates: Sequence[str] = ["article", "[role='document']", "[role='main']", "body"]
    for selector in candidates:
        locator = page.locator(selector).first
        with contextlib.suppress(PlaywrightTimeoutError, Exception):
            if locator.is_visible(timeout=1500):
                text = locator.inner_text(timeout=1500).strip()
                if text:
                    logger.debug(f"[yandex_mail] Extracted body via {selector}")
                    return text
    return ""


# -------------------------
# Формирование отчёта
# -------------------------
def format_summary(messages: List[MessageDraft], plan: str, goal: str) -> str:
    """
    Формирует человекочитаемый отчёт о том, что сделал под-агент.

    Без заголовков "Цель", "План LLM", "Итоги проверки N писем",
    "Удалено/Оставлено". Просто список действий по письмам.
    """
    lines: List[str] = ["Результаты обработки писем:"]

    goal_l = (goal or "").lower()
    deletion_requested = any(
        kw in goal_l for kw in ("удал", "удали", "удалить", "спам", "spam", "очист")
    )

    for msg in messages:
        # Приоритет: сначала ответ, потом удаление, потом просто прочтение.
        if msg.reply_sent:
            verdict = "Отправлен ответ на письмо"
        elif msg.deleted and msg.spam:
            verdict = "СПАМ → удалено"
        elif msg.deleted and not msg.spam:
            verdict = "Удалено (по цели пользователя / плану агента)"
        else:
            if deletion_requested and msg.spam:
                verdict = "СПАМ (не удалено — решение агента)"
            elif deletion_requested:
                verdict = "Оставлено (по решению агента)"
            else:
                verdict = "Прочитано (удаление не запрашивалось)"

        reason_parts = []
        if msg.reason:
            reason_parts.append(f"оценка спама: {msg.reason}")
        if msg.delete_reason:
            reason_parts.append(f"решение об удалении: {msg.delete_reason}")

        reason_str = " / ".join(reason_parts) if reason_parts else "—"

        lines.append(f"- {msg.subject} — {verdict}. Причина: {reason_str}")

    return "\n".join(lines)


__all__ = ["MessageDraft", "open_mailbox", "ensure_mail_list", "collect_previews", "extract_body", "format_summary"]

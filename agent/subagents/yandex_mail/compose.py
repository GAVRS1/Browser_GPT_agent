from __future__ import annotations

import contextlib
import re
from typing import Any, Optional

from playwright.sync_api import Page

from agent.debug_thoughts import log_thought
from loguru import logger
from agent.subagents.yandex_mail.common import ensure_mail_list


def _find_first_visible(page: Page, locator) -> Optional[Any]:
    """
    Возвращает первый видимый элемент из локатора или None.
    """
    with contextlib.suppress(Exception):
        count = locator.count()
        for i in range(count):
            cand = locator.nth(i)
            if cand.is_visible(timeout=2000):
                return cand
    return None


def _click_compose_button(page: Page) -> bool:
    """
    Надёжно нажать кнопку 'Написать'.
    """
    # 1) ARIA button с текстом "Написать"
    with contextlib.suppress(Exception):
        btn = page.get_by_role(
            "button",
            name=re.compile(r"Написать", re.IGNORECASE),
        ).first
        btn.wait_for(state="visible", timeout=5000)
        btn.click(timeout=3000)
        page.wait_for_timeout(600)
        return True

    # 2) Любой текст "Написать"
    with contextlib.suppress(Exception):
        btn = page.get_by_text(re.compile(r"\bНаписать\b", re.IGNORECASE)).first
        btn.click(timeout=3000)
        page.wait_for_timeout(600)
        return True

    # 3) Ссылка "Написать"
    with contextlib.suppress(Exception):
        btn = page.locator("a:has-text('Написать')").first
        btn.click(timeout=3000)
        page.wait_for_timeout(600)
        return True

    return False


def _active_element_is_editable(page: Page) -> bool:
    """
    Проверяем, что в фокусе действительно поле ввода.
    Чтобы не делать Ctrl+A по всей странице.
    """
    try:
        return page.evaluate(
            """
            () => {
                const el = document.activeElement;
                if (!el) return false;
                if (el.isContentEditable) return true;
                const tag = el.tagName;
                return tag === 'INPUT' || tag === 'TEXTAREA';
            }
            """
        )
    except Exception:  # noqa: BLE001
        return False


def _focus_recipient_field(page: Page) -> None:
    """
    Пытаемся явно сфокусироваться в поле 'Кому',
    чтобы дальнейшие нажатия клавиш шли туда.
    """
    with contextlib.suppress(Exception):
        komu = page.get_by_text(re.compile(r"\bКому\b", re.IGNORECASE)).first
        komu.click()
        page.wait_for_timeout(200)


def _build_email_content(goal: str, raw_body: str) -> tuple[str, str]:
    """
    Строим ТЕМУ и ТЕКСТ письма из цели пользователя.
    Если цель про 'расскажи что ты ИИ агент' — пишем нормальный текст.
    """
    goal_l = goal.lower()

    if ("ии агент" in goal_l or "ai агент" in goal_l or "ai-agent" in goal_l) and (
        "расскажи" in goal_l or "рассказать" in goal_l or "сообщи" in goal_l
    ):
        subject = "Письмо от ИИ-агента"
        body = (
            "Здравствуйте!\n\n"
            "Пишет автономный ИИ-агент, который управляет браузером и помогает пользователю "
            "выполнять различные задачи в интернете: читать и обрабатывать письма, работать с сайтами, "
            "искать информацию и многое другое.\n\n"
            "Сейчас меня попросили отправить вам это письмо и рассказать, что я являюсь ИИ-агентом. "
            "Я работаю полностью автоматически, но всегда стараюсь действовать аккуратно и безопасно.\n\n"
            "Если у вас есть идеи, какие задачи можно поручить такому агенту, — на стороне пользователя "
            "можно расширять мои возможности и обучение.\n\n"
            "С уважением,\n"
            "Ваш ИИ-агент."
        )
        return subject, body

    # Если явной спец-логики нет — используем то, что пришло из подагента
    text = (raw_body or "Привет!").strip()
    subject = text
    if len(subject) > 60:
        subject = subject[:57] + "..."
    return subject, text


def run_compose_flow(page: Page, goal: str, plan: str, to_addr: Optional[str], body_text: str):
    """
    Режим "напиши / отправь новое письмо ...".
    Основная логика:
      1) открыть форму 'Написать';
      2) сфокусироваться в 'Кому';
      3) заполнить поля через клавиатуру;
      4) нажать 'Отправить'.
    """
    from agent.subagents import SubAgentResult

    log_thought(
        "yandex-mail-compose",
        f"Запущен режим написания нового письма.\n"
        f"Цель: {goal}\n"
        f"План верхнего уровня:\n{plan or '—'}\n"
        f"Адрес получателя: {to_addr!r}\n"
        f"Текст письма (сырой): {body_text!r}",
    )

    if not to_addr:
        msg = (
            "Не удалось понять, на какой адрес отправить письмо — "
            "в цели не найден e-mail. Уточните адрес в формулировке."
        )
        log_thought("yandex-mail-compose", msg)
        return SubAgentResult(
            success=False,
            status="failed",
            error="no_recipient",
            details=msg,
        )

    # Строим нормальные subject + body из цели
    subject_text, final_body = _build_email_content(goal, body_text)

    # На всякий случай убеждаемся, что мы в списке писем
    ensure_mail_list(page)

    # 1) Нажимаем кнопку "Написать"
    if not _click_compose_button(page):
        msg = "Не удалось найти кнопку 'Написать' для создания нового письма."
        log_thought("yandex-mail-compose", msg)
        return SubAgentResult(
            success=False,
            status="failed",
            error="compose_button_not_found",
            details=msg,
        )

    # Немного ждём появления формы поверх всего
    page.wait_for_timeout(700)

    # 2) Пытаемся заполнить всё КЛАВИАТУРОЙ: Кому → Tab → Тема → Tab → Тело
    filled_via_keyboard = False
    try:
        log_thought("yandex-mail-compose", "Пробую заполнить поля письма через клавиатуру.")

        # Убедимся, что фокус в чём-то редактируемом; если нет — кликнем в 'Кому'
        if not _active_element_is_editable(page):
            _focus_recipient_field(page)
            page.wait_for_timeout(150)

        # Теперь чистим и пишем адрес
        if _active_element_is_editable(page):
            page.keyboard.press("Control+A")
            page.keyboard.press("Backspace")
        page.keyboard.type(to_addr, delay=70)

        # Переходим в поле "Тема"
        page.keyboard.press("Tab")
        page.wait_for_timeout(150)
        if _active_element_is_editable(page):
            page.keyboard.press("Control+A")
            page.keyboard.press("Backspace")
        page.keyboard.type(subject_text, delay=60)

        # Переходим в тело письма
        page.keyboard.press("Tab")
        page.wait_for_timeout(150)
        if _active_element_is_editable(page):
            page.keyboard.press("Control+A")
            page.keyboard.press("Backspace")
        page.keyboard.type(final_body, delay=40)

        filled_via_keyboard = True
    except Exception as e:  # noqa: BLE001
        log_thought(
            "yandex-mail-compose",
            f"Клавиатурное заполнение полей не удалось: {e!r}. "
            "Попробую резервные селекторы.",
        )
        filled_via_keyboard = False

    # Если клавиатурный способ не удался — хотя бы тело попробуем вставить по селекторам
    if not filled_via_keyboard:
        body_field = None
        with contextlib.suppress(Exception):
            editable = page.locator("[contenteditable='true']")
            body_field = _find_first_visible(page, editable)
        if body_field is None:
            with contextlib.suppress(Exception):
                textarea = page.locator("textarea")
                body_field = _find_first_visible(page, textarea)
        if body_field is None:
            with contextlib.suppress(Exception):
                textboxes = page.get_by_role("textbox")
                body_field = _find_first_visible(page, textboxes)

        if body_field is None:
            msg = "Не удалось найти поле для ввода текста письма."
            log_thought("yandex-mail-compose", msg)
            return SubAgentResult(
                success=False,
                status="failed",
                error="body_field_not_found",
                details=msg,
            )

        with contextlib.suppress(Exception):
            body_field.click()
            body_field.fill("")
        with contextlib.suppress(Exception):
            body_field.type(final_body, delay=50)

        log_thought(
            "yandex-mail-compose",
            "Предупреждение: поля 'Кому' и 'Тема' могли заполниться некорректно. "
            "Проверьте их вручную.",
        )

    # 3) Ищем кнопку "Отправить"
    send_clicked = False
    send_patterns = re.compile("отправить|send", re.IGNORECASE)
    send_candidates = [
        page.get_by_role("button", name=send_patterns),
        page.get_by_text(send_patterns),
    ]
    for loc in send_candidates:
        btn = _find_first_visible(page, loc)
        if btn:
            with contextlib.suppress(Exception):
                btn.click(timeout=2000)
                page.wait_for_timeout(1200)
            send_clicked = True
            break

    if not send_clicked:
        msg = "Не удалось найти кнопку 'Отправить' при создании письма."
        log_thought("yandex-mail-compose", msg)
        return SubAgentResult(
            success=False,
            status="failed",
            error="send_button_not_found",
            details=msg,
        )

    msg = (
        f"Новое письмо на адрес {to_addr} с темой {subject_text!r} "
        f"и текстом {final_body!r} предположительно отправлено."
    )
    log_thought("yandex-mail-compose", msg)

    return SubAgentResult(
        success=True,
        status="completed",
        details=msg,
    )

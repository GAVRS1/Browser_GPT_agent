from __future__ import annotations

import contextlib
import re
from typing import Optional

from loguru import logger
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from browser.context import get_page
from agent.subagents.utils import matches_domain


class YandexLavkaSubAgent:
    """
    Специализированный под-агент для Яндекс Лавки.

    Что он делает:
    - открывает / фокусирует страницу Яндекс Лавки;
    - пытается найти и сфокусировать именно строку поиска, а не любое меню;
    - готовит страницу для основного автономного агента.

    Важно:
    - Никаких жёстких селекторов под конкретную верстку.
    - Поиск поля осуществляется по роли, placeholder, типу input (search/text) и
      эвристикам по тексту («поиск», «search»).
    - Сам под-агент не описывает пошаговый сценарий заказа — этим занимается
      универсальный агент через свои инструменты.
    """

    name = "Яндекс.Лавка"

    _keywords = [
        "яндекс лавк",
        "yandex lavka",
        "лавка",
        "lavka",
        "доставк",  # «доставка продуктов в лавке» и т.п.
        "продукт из лавки",
        "еда из лавки",
    ]

    _domains = ["lavka.yandex.ru", "yandex-lavka"]

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
        1. Открываем/фокусируем Яндекс Лавку.
        2. Пробуем найти и сфокусировать строку поиска.
        3. Возвращаем "подготовленное" состояние и отдаём задачу
           универсальному агенту (success=False → пойдёт в fallback).
        """
        from agent.subagents import SubAgentResult  # локальный импорт, чтобы не ловить циклы

        page = get_page()
        with contextlib.suppress(Exception):
            page.bring_to_front()

        # 1. Открыть / проверить Лавку
        login_msg = open_lavka_home(page)
        if login_msg is not None:
            # Нужны действия пользователя (логин/гео и т.п.)
            return SubAgentResult(
                success=False,
                status="needs_input",
                details=login_msg,
            )

        # 2. Попробовать найти и сфокусировать поиск
        focus_msg = focus_search_input(page)

        # 3. Добавляем явную подсказку для автономного агента,
        #    как работать с найденными блюдами.
        ai_hint = (
            "Подсказка для ИИ-агента:\n"
            "- Ты находишься на странице Яндекс Лавки.\n"
            "- Для поиска товаров используй именно строку поиска не каталог "
            "(нажми на нее что бы можно было вводить текст).\n"
            "- Когда пользователь просит найти конкретное блюдо или продукт "
            "(например: «найди пиццу пепперони», «найди молоко»):\n"
            "  1) Введи соответствующий текст в поле поиска.\n"
            "  2) Дождись появления списка товаров.\n"
            "  3) Выбери карточку товара, которая лучше всего соответствует(если нету конкретных критериев то первый из) "
            "запросу, и НАЖМИ НА КАРТОЧКУ, чтобы открыть подробную информацию.\n"
            "  4) Если цель — заказать или добавить в корзину, найди на "
            "карточке или на странице товара кнопку, которая по смыслу "
            "означает добавление в корзину (например, с ценой/иконкой корзины), "
            "и нажми её.\n"
            "- Не ограничивайся кликом по верхнему вспомогательному меню или "
            "категориям: важно именно открыть карточку нужного блюда/товара."
        )

        details = f"{focus_msg}\n\n{ai_hint}"

        # success = False намеренно: дальше подключается универсальный
        # автономный режим (_autonomous_browse), но уже на подготовленной
        # странице с Лавкой и понятной подсказкой.
        return SubAgentResult(
            success=False,
            status="prepared",
            details=details,
        )


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def open_lavka_home(page: Page) -> Optional[str]:
    """
    Переходит на lavka.yandex.ru (если мы ещё не там) и грубо проверяет,
    не требуется ли логин/выбор города.

    Если нужно действие пользователя — возвращает текст, который покажет
    верхнеуровневый агент. Если всё ок — возвращает None.
    """
    url = page.url or ""
    if "lavka.yandex" not in url:
        logger.info("[yandex_lavka] Navigating to Yandex Lavka…")
        with contextlib.suppress(Exception):
            page.goto("https://lavka.yandex.ru/", wait_until="domcontentloaded")

    # Дадим интерфейсу чуть времени прогрузиться
    with contextlib.suppress(Exception):
        page.wait_for_timeout(1200)

    # Эвристики: если видим явную форму логина — просим пользователя залогиниться.
    login_required = False

    with contextlib.suppress(PlaywrightTimeoutError, Exception):
        login_button = page.get_by_text(re.compile("Войти", re.IGNORECASE)).first
        if login_button.is_visible(timeout=1500):
            login_required = True

    if login_required:
        logger.warning("[yandex_lavka] Login/authorization seems required.")
        return (
            "Откройте вкладку Яндекс Лавки, выберите город и при необходимости "
            "войдите в аккаунт вручную. После этого повторите запрос — "
            "агент продолжит работу с уже авторизованной Лавкой."
        )

    logger.info("[yandex_lavka] Lavka appears to be open and usable.")
    return None


def focus_search_input(page: Page) -> str:
    """
    Пытается найти и сфокусировать строку поиска на главной странице Лавки.

    Не используем жёсткие селекторы:
    - сперва ищем по роли/placeholder,
    - затем стараемся найти input[type='search'],
    - как fallback — любые текстовые поля, где в placeholder/label есть «поиск».

    Стараемся избежать вспомогательных меню: выбираем именно поля ввода.
    """
    logger.info("[yandex_lavka] Trying to locate search input…")

    patterns = re.compile("поиск|search", re.IGNORECASE)

    # 1. По роли
    candidates = []

    # Явный searchbox
    with contextlib.suppress(Exception):
        rb = page.get_by_role("searchbox")
        if rb.count() > 0:
            candidates.append(rb.first)

    # Textbox с именем, похожим на «поиск»
    with contextlib.suppress(Exception):
        tb = page.get_by_role("textbox", name=patterns)
        if tb.count() > 0:
            candidates.append(tb.first)

    # 2. По placeholder
    with contextlib.suppress(Exception):
        ph = page.get_by_placeholder(patterns)
        if ph.count() > 0:
            candidates.append(ph.first)

    # 3. input[type='search']
    with contextlib.suppress(Exception):
        search_inputs = page.locator("input[type='search']")
        if search_inputs.count() > 0:
            candidates.append(search_inputs.first)

    # 4. Fallback: любой input/textarea с подходящим placeholder/aria-label
    if not candidates:
        with contextlib.suppress(Exception):
            selector = (
                "input[type='text'], input[type='search'], textarea, "
                "[role='searchbox']"
            )
            all_fields = page.locator(selector)
            count = all_fields.count()
            for i in range(min(count, 12)):
                el = all_fields.nth(i)
                # Проверяем через JS placeholder / aria-label / title
                info = page.evaluate(
                    """(el) => {
                        const ph = el.getAttribute('placeholder') || '';
                        const al = el.getAttribute('aria-label') || '';
                        const ti = el.getAttribute('title') || '';
                        return { placeholder: ph, ariaLabel: al, title: ti };
                    }""",
                    el,
                )
                text = " ".join(
                    [info.get("placeholder") or "", info.get("ariaLabel") or "", info.get("title") or ""]
                )
                if patterns.search(text):
                    candidates.append(el)
                    break

    # Попробуем сфокусировать первый подходящий кандидат
    for el in candidates:
        with contextlib.suppress(Exception):
            el.scroll_into_view_if_needed()
        with contextlib.suppress(Exception):
            if el.is_visible(timeout=2000):
                el.click(timeout=2000)
                # Очищаем поле, чтобы основной агент мог вводить свои запросы
                with contextlib.suppress(Exception):
                    el.fill("")
                logger.info("[yandex_lavka] Search input focused.")
                return "Нашёл и сфокусировал поле поиска в Яндекс Лавке."

    logger.warning("[yandex_lavka] Could not reliably locate search input.")
    return (
        "Открыл Яндекс Лавку, но не удалось однозначно найти поле поиска. "
        "Универсальный агент продолжит работу, используя общие инструменты."
    )


__all__ = ["YandexLavkaSubAgent"]
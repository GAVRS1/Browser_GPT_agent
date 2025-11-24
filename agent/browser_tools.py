from __future__ import annotations
import time
import contextlib
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
from playwright.sync_api import Locator, Page

from browser.context import get_page


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ToolResult:
    name: str
    success: bool
    observation: str


class BrowserToolbox:
    """Коллекция универсальных инструментов для LLM-агента.

    Набор покрывает базовые действия: навигация, чтение контекста,
    клики, ввод текста, скролл и возврат назад. Все операции стараются
    быть устойчивыми к ошибкам и возвращают компактные наблюдения,
    которые можно передавать в историю диалога с моделью.

    ДОПОЛНИТЕЛЬНО:
    - есть инструмент take_screenshot, который делает скриншот страницы
      и возвращает путь к файлу. LLM видит это в observation и при
      необходимости может ещё раз запросить скрин/другой участок.
    """

    def __init__(self, page: Optional[Page] = None) -> None:
        self.page: Page = page or get_page()

    # ------------------------------------------------------------
    # OpenAI tool schemas
    # ------------------------------------------------------------
    def openai_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_view",
                    "description": "Собрать краткий список заметных элементов страницы и их текст.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "open_url",
                    "description": "Перейти по указанному URL (используется, если агент хочет вручную открыть страницу).",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "Нажать на элемент по тексту, ARIA name или CSS-селектору (без заранее зашитых селекторов).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Текст/label/placeholder или CSS-селектор для поиска элемента.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "type_text",
                    "description": "Ввести текст в поле по label/placeholder/CSS и при необходимости отправить (Enter).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Подпись/placeholder или CSS-селектор поля ввода.",
                            },
                            "text": {"type": "string"},
                            "press_enter": {
                                "type": "boolean",
                                "description": "Нажать Enter после ввода.",
                                "default": False,
                            },
                        },
                        "required": ["query", "text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "scroll",
                    "description": (
                        "Прокрутить страницу вверх/вниз на указанный отрезок. "
                        "На страницах с сеткой товаров (например, Яндекс Лавка) "
                        "используй скролл осторожно: если уже видна сетка карточек, "
                        "лучше сначала кликать по карточкам, а не скроллить."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["down", "up"],
                                "default": "down",
                            },
                            "amount": {
                                "type": "integer",
                                "description": "Пиксели для скролла (по умолчанию 800).",
                                "default": 800,
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "go_back",
                    "description": "Вернуться на предыдущую страницу в истории браузера.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "click_product_card",
                    "description": (
                        "Клик по одной из крупных карточек товара в основной части страницы. "
                        "Полезно, когда после поиска в интернет-магазине (например, Яндекс Лавка) "
                        "появилась сетка блюд/товаров, но к конкретной карточке сложно обратиться по тексту."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "take_screenshot",
                    "description": (
                        "Сделать скриншот текущей страницы. "
                        "Возвращает путь к файлу скриншота, который можно посмотреть в логах."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "full_page": {
                                "type": "boolean",
                                "description": "Снимать всю страницу целиком (true) или только видимую область (false).",
                                "default": False,
                            }
                        },
                    },
                },
            },
        ]

    # ------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------
    def execute(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        arguments = arguments or {}
        try:
            if tool_name == "read_view":
                return ToolResult(tool_name, True, self.read_view())
            if tool_name == "open_url":
                return ToolResult(tool_name, True, self.open_url(arguments.get("url", "")))
            if tool_name == "click":
                return ToolResult(tool_name, True, self.click(arguments.get("query", "")))
            if tool_name == "type_text":
                return ToolResult(
                    tool_name,
                    True,
                    self.type_text(
                        arguments.get("query", ""),
                        arguments.get("text", ""),
                        bool(arguments.get("press_enter", False)),
                    ),
                )
            if tool_name == "scroll":
                return ToolResult(
                    tool_name,
                    True,
                    self.scroll(
                        direction=arguments.get("direction", "down"),
                        amount=int(arguments.get("amount", 800)),
                    ),
                )
            if tool_name == "go_back":
                return ToolResult(tool_name, True, self.go_back())
            if tool_name == "click_product_card":
                return ToolResult(tool_name, True, self.click_product_card())
            if tool_name == "take_screenshot":
                return ToolResult(
                    tool_name,
                    True,
                    self.take_screenshot(bool(arguments.get("full_page", False))),
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[tools] {tool_name} failed: {exc}")
            return ToolResult(tool_name, False, f"Ошибка: {exc}")

        return ToolResult(tool_name, False, f"Неизвестный инструмент: {tool_name}")

    # ------------------------------------------------------------
    # Tools implementation
    # ------------------------------------------------------------

    def take_screenshot(self, full_page: bool = False) -> str:
        """
        Сделать скриншот текущей страницы.

        full_page = True  – снимаем всю страницу (long screenshot),
        full_page = False – только видимую область.

        Возвращает ПУТЬ до файла со скриншотом (его увидит и LLM, и ты в логах).
        """
        page = self.page

        # Папка для скриншотов внутри проекта
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Имя файла по времени
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = screenshots_dir / f"screenshot_{timestamp}.png"

        try:
            page.screenshot(path=str(path), full_page=full_page)
            logger.info(f"[tools] Screenshot saved to {path}")
            # LLM важен сам путь, текст можно минимальный
            return str(path)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[tools] Screenshot failed: {exc}")
            # Пробрасываем наверх, чтобы execute() пометил инструмент как fail
            raise
    def read_view(self) -> str:
        """Возвращает компактный обзор страницы.

        Учитываем только заметные интерактивные элементы, чтобы не слать
        огромные DOM-фрагменты. Каждый элемент содержит текст, роль и
        эвристический локатор, который можно попробовать в клике/вводе.

        ДОПОЛНИТЕЛЬНО:
        - пытаемся выделить крупные карточки товаров (product_cards), чтобы
          агент понимал, где сетка товаров (например, в Яндекс Лавке).
        - для Яндекс Лавки стараемся извлекать структуру по каждой карточке:
          примерное название, цену, вес (в граммах), описание и состав (если видны).
        """

        page = self.page
        summary: Dict[str, Any] = {"url": page.url, "title": safe_title(page)}

        # --- Базовый список интерактивных элементов (кнопки, ссылки и т.п.) ---
        interactive: List[Dict[str, str]] = []
        roles = [
            ("button", page.get_by_role("button")),
            ("link", page.get_by_role("link")),
            ("textbox", page.get_by_role("textbox")),
            ("searchbox", page.get_by_role("searchbox")),
            ("combobox", page.get_by_role("combobox")),
            ("listitem", page.get_by_role("listitem")),
        ]

        for role_name, locator in roles:
            with contextlib.suppress(Exception):
                count = locator.count()
            if not count:
                continue

            for idx in range(min(count, 8)):
                item = locator.nth(idx)
                text = _safe_text(item)
                attrs = _collect_attributes(item)
                combined_text = text or attrs.get("aria_label") or attrs.get("placeholder")
                combined_text = combined_text or attrs.get("name") or attrs.get("id")
                if not combined_text and not attrs:
                    continue
                interactive.append(
                    {
                        "role": role_name,
                        "text": (combined_text or "")[:160],
                        "locator": f"role={role_name}, idx={idx}",
                        "attrs": attrs,
                    }
                )

        summary["interactive"] = interactive[:40]

        # --- Поиск крупных блоков-карточек товаров ---
        cards: List[Dict[str, Any]] = []
        try:
            root = page.locator("main")
            with contextlib.suppress(Exception):
                if root.count() == 0:
                    root = page.locator("body")

            card_nodes = root.locator("div, article, section")

            with contextlib.suppress(Exception):
                total = card_nodes.count()
            if not total:
                summary["product_cards"] = []
                return json.dumps(summary, ensure_ascii=False)[:1800]

            max_to_check = min(total, 40)

            for i in range(max_to_check):
                el = card_nodes.nth(i)
                try:
                    box = el.bounding_box()
                    if not box:
                        continue

                    w = box.get("width") or 0
                    h = box.get("height") or 0
                    area = w * h

                    # Отсекаем мелкие элементы
                    if w < 120 or h < 140 or area < 15000:
                        continue

                    raw_text = ""
                    with contextlib.suppress(Exception):
                        raw_text = el.inner_text() or ""
                    if not raw_text:
                        continue

                    inner = raw_text.strip().replace("\n", " ")
                    if not inner:
                        continue

                    # Есть ли цена
                    price_match = re.search(
                        r"(\d+[.,]?\d*)\s*(₽|руб)",
                        inner,
                        re.IGNORECASE,
                    )
                    has_price = bool(price_match)

                    # Похоже на нормальное название (2+ слова)
                    has_title = len(inner.split()) >= 2

                    # Пытаемся вытащить вес и привести его к граммам
                    weight_match = re.search(
                        r"(\d+[.,]?\d*)\s*(г|гр|грамм|кг|килограмм)",
                        inner,
                        re.IGNORECASE,
                    )
                    weight_text = weight_match.group(0) if weight_match else ""
                    weight_g = None
                    if weight_match:
                        try:
                            value_str = weight_match.group(1).replace(",", ".")
                            unit = weight_match.group(2).lower()
                            value = float(value_str)
                            if unit.startswith("к"):  # кг
                                weight_g = int(value * 1000)
                            else:  # граммы
                                weight_g = int(value)
                        except Exception:
                            weight_g = None

                    price_text = price_match.group(0) if price_match else ""

                    # Грубая эвристика для имени товара: всё до цены или просто первые слова
                    name_guess = ""
                    if price_match:
                        name_guess = inner[: price_match.start()].strip()
                    if not name_guess:
                        parts = inner.split()
                        name_guess = " ".join(parts[:6])

                    # Очень грубая эвристика для описания и состава
                    description = ""
                    composition = ""
                    try:
                        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
                        desc_lines: List[str] = []
                        comp_lines: List[str] = []
                        comp_started = False
                        for ln in lines:
                            low = ln.lower()
                            if "состав" in low:
                                comp_started = True
                            if comp_started:
                                comp_lines.append(ln)
                            else:
                                if not re.search(r"(₽|руб)", low) and not re.search(
                                    r"\d+\s*(г|гр|грамм|кг|килограмм)", low
                                ):
                                    desc_lines.append(ln)
                        if desc_lines:
                            description = " ".join(desc_lines)[:200]
                        if comp_lines:
                            composition = " ".join(comp_lines)[:300]
                    except Exception:
                        pass

                    cards.append(
                        {
                            "idx": i,
                            "size": f"{int(w)}x{int(h)}",
                            "price": has_price,
                            "price_text": price_text,
                            "title_like": has_title,
                            "name": name_guess[:120],
                            "weight_text": weight_text,
                            "weight_g": weight_g,
                            "description": description,
                            "composition": composition,
                            "text": inner[:150],
                        }
                    )
                except Exception:
                    continue

            summary["product_cards"] = cards[:10]
        except Exception:
            summary["product_cards"] = []

        return json.dumps(summary, ensure_ascii=False)[:1800]

    def open_url(self, url: str) -> str:
        if not url:
            return "URL не задан"
        self.page.goto(url, wait_until="domcontentloaded")
        return f"Открыл {self.page.url}"

    def click(self, query: str) -> str:
        if not query:
            return "Пустой запрос — клик не выполнен"

        page = self.page
        strategies: List[Locator] = []

        if _looks_like_search_intent(query):
            strategies.extend(_search_locators(page, query, include_text_inputs=True))

        strategies.extend(
            [
                page.get_by_role("button", name=re.compile(query, re.IGNORECASE)),
                page.get_by_role("link", name=re.compile(query, re.IGNORECASE)),
                page.get_by_text(query, exact=False),
            ]
        )

        strategies.extend(_attribute_locators(page, query, tags=["button", "a", "input", "summary", "div", "span"]))

        if _looks_like_css(query):
            strategies.append(page.locator(query))

        candidate = _first_visible(strategies)
        if candidate:
            candidate.click(timeout=2000)
            return f"Клик по {query} выполнен"

        return f"Не нашёл, куда кликнуть по запросу: {query}"

    def type_text(self, query: str, text: str, press_enter: bool = False) -> str:
        if not query:
            return "Нет запроса для ввода"
        page = self.page
        strategies: List[Locator] = []

        if _looks_like_search_intent(query):
            strategies.extend(_search_locators(page, query, include_text_inputs=True))

        strategies.extend(
            [
                page.get_by_label(query, exact=False),
                page.get_by_placeholder(query, exact=False),
                page.get_by_text(query, exact=False),
            ]
        )

        strategies.extend(_attribute_locators(page, query, tags=["input", "textarea", "select"]))

        if _looks_like_css(query):
            strategies.append(page.locator(query))

        candidate = _first_visible(strategies)
        if candidate:
            candidate.fill(text, timeout=2000)
            if press_enter:
                candidate.press("Enter")
            return f"Ввёл текст в поле {query}"

        return f"Не удалось найти поле для ввода по запросу: {query}"

    def scroll(self, direction: str = "down", amount: int = 800) -> str:
        """
        Прокрутка страницы.

        ДОП. ЛОГИКА:
        - Если мы на странице Яндекс Лавки и уже видна сетка крупных карточек товаров,
          скролл отключается и агенту возвращается подсказка: нужно кликать по карточкам,
          а не уезжать всё ниже.
        """

        page = self.page

        # Пытаемся понять, что мы на Лавке и уже видим сетку карточек
        try:
            url = page.url or ""
        except Exception:
            url = ""

        if "lavka.yandex" in url:
            try:
                root = page.locator("main")
                if root.count() == 0:
                    root = page.locator("body")

                containers = root.locator("div, article, section")
                with contextlib.suppress(Exception):
                    total = containers.count()
                if total:
                    max_to_check = min(total, 40)
                    big = 0

                    for i in range(max_to_check):
                        el = containers.nth(i)
                        with contextlib.suppress(Exception):
                            box = el.bounding_box()
                            if not box:
                                continue
                            w = box.get("width") or 0
                            h = box.get("height") or 0
                            area = w * h
                            if w >= 120 and h >= 140 and area >= 15000:
                                big += 1
                        if big >= 4:
                            break

                    # Если нашли достаточно крупных блоков — считаем, что это сетка товаров
                    if big >= 4:
                        return (
                            "На странице Яндекс Лавки уже видна сетка карточек товаров. "
                            "Скролл временно отключён: выбери подходящий товар и кликни по его карточке "
                            "(через click по названию или через инструмент click_product_card)."
                        )
            except Exception:
                # Если эвристика сломалась — просто ведём себя как обычный скролл
                pass

        delta = amount if direction != "up" else -amount
        with contextlib.suppress(Exception):
            page.mouse.wheel(0, delta)
        return f"Прокрутил {direction} на {amount}"

    def go_back(self) -> str:
        with contextlib.suppress(Exception):
            self.page.go_back()
        return "Вернулся на предыдущую страницу"

    def click_product_card(self) -> str:
        """
        Улучшенный алгоритм клика по карточке товара.

        Для Яндекс Лавки добавлены специальные правила:
        - если мы на странице рецепта (url содержит 'recipes'), не кликаем по крупному
          блоку рецепта, а ищем кнопку «К продуктам» / «К продуктам +»;
        - при выборе карточек по размеру стараемся НЕ выбирать рецепты с текстом
          вида «15 мин», «30 мин» — они понижаются в приоритете.
        """

        page = self.page

        # 1) Обработка страницы рецепта Яндекс Лавки
        try:
            url = page.url or ""
        except Exception:
            url = ""

        if "lavka.yandex" in url and "recipes" in url:
            # На странице рецепта хотим перейти к продуктам, а не ещё одному рецепту
            try:
                btn = page.get_by_role(
                    "button",
                    name=re.compile("К продуктам", re.IGNORECASE),
                )
                if btn and btn.count() > 0:
                    candidate = btn.first
                    if candidate.is_visible(timeout=2000):
                        candidate.click(timeout=4000)
                        return "Нажал кнопку «К продуктам» на странице рецепта."
            except Exception as exc:
                logger.warning(f"[tools] click_product_card: no 'К продуктам' button: {exc}")

            # Если не получилось, пробуем простой go_back
            with contextlib.suppress(Exception):
                page.go_back()
            return (
                "На странице рецепта не нашёл понятной кнопки перехода к продуктам. "
                "Вернулся назад. Продолжай работу с сеткой товаров или поиском."
            )

        # 2) Обычный выбор большой карточки товара
        root = page.locator("main")
        try:
            if root.count() == 0:
                root = page.locator("body")
        except Exception:
            root = page.locator("body")

        containers = root.locator("div, article, section")

        try:
            total = containers.count()
        except Exception as exc:
            logger.error(f"[tools] click_product_card: cannot count: {exc}")
            return "Не удалось получить список элементов."

        if total == 0:
            return "Не нашёл ни одного контейнера-карточки."

        candidates = []
        max_to_check = min(total, 50)

        for i in range(max_to_check):
            el = containers.nth(i)
            try:
                box = el.bounding_box()
                if not box:
                    continue

                w = box.get("width") or 0
                h = box.get("height") or 0
                area = w * h

                if w < 120 or h < 140 or area < 15000:
                    continue

                with contextlib.suppress(Exception):
                    inner = el.inner_text() or ""
                    inner = inner.strip().replace("\n", " ")
                if not inner:
                    continue

                has_price = bool(
                    re.search(r"\d+[.,]?\d*\s*(₽|руб)", inner, re.IGNORECASE)
                )
                has_title = len(inner.split()) >= 2

                # Обнаружение "рецептности": наличие "мин" рядом с цифрой
                is_recipe = bool(
                    re.search(r"\d+\s*мин", inner, re.IGNORECASE)
                )

                score = area
                if has_price:
                    score += 50000
                if has_title:
                    score += 15000
                if is_recipe:
                    score -= 40000  # рецепты понижаем в приоритете

                candidates.append((score, el))
            except Exception:
                continue

        if not candidates:
            return "Не удалось найти карточку товара."

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]

        try:
            best.scroll_into_view_if_needed()
        except Exception:
            pass

        try:
            best.click(timeout=4000, force=True)
            return "Кликнул по карточке товара."
        except Exception as exc:
            logger.error(f"[tools] click_product_card: click failed: {exc}")
            return f"Не удалось кликнуть по карточке: {exc}"


def safe_title(page: Page) -> str:
    with contextlib.suppress(Exception):
        return page.title()
    return ""


def _safe_text(locator: Locator) -> str:
    with contextlib.suppress(Exception):
        text = locator.inner_text(timeout=1000).strip()
        if text:
            return re.sub(r"\s+", " ", text)
    return ""


def _collect_attributes(locator: Locator) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for attr, key in [
        ("aria-label", "aria_label"),
        ("aria-labelledby", "aria_labelledby"),
        ("placeholder", "placeholder"),
        ("name", "name"),
        ("id", "id"),
        ("type", "type"),
        ("role", "role"),
    ]:
        with contextlib.suppress(Exception):
            value = locator.get_attribute(attr) or ""
            if value:
                attrs[key] = value[:200]
    return attrs


def _looks_like_css(query: str) -> bool:
    return query.startswith(".") or query.startswith("#") or query.startswith("[")


def _looks_like_search_intent(query: str) -> bool:
    cleaned = query.lower().strip()
    return bool(re.search(r"\b(поиск|search|найти)\b", cleaned))


def _first_visible(locators: Iterable[Locator]) -> Optional[Locator]:
    seen: set[str] = set()
    for locator in locators:
        key = repr(locator)
        if key in seen:
            continue
        seen.add(key)
        with contextlib.suppress(Exception):
            candidate = locator.first
            if candidate.is_visible(timeout=1500):
                return candidate
    return None


def _css_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _attribute_locators(page: Page, query: str, tags: List[str]) -> List[Locator]:
    safe = _css_escape(query)
    attributes = [
        "aria-label",
        "aria-labelledby",
        "placeholder",
        "name",
        "id",
        "type",
        "title",
    ]
    locators: List[Locator] = []
    for attr in attributes:
        selectors = [f"{tag}[{attr}*=\"{safe}\" i]" for tag in tags]
        locators.append(page.locator(",".join(selectors)))
    return locators


def _search_locators(page: Page, query: str, include_text_inputs: bool = False) -> List[Locator]:
    safe = _css_escape(query)
    locators: List[Locator] = [page.get_by_role("searchbox")]

    search_keywords = ["поиск", "search", "найти"]
    keyword_selectors = []
    for kw in search_keywords:
        kw_safe = _css_escape(kw)
        keyword_selectors.append(f"input[placeholder*=\"{kw_safe}\" i]")
        keyword_selectors.append(f"input[aria-label*=\"{kw_safe}\" i]")
        keyword_selectors.append(f"input[name*=\"{kw_safe}\" i]")

    locators.append(page.locator(",".join(keyword_selectors)))

    type_selectors = ["input[type='search']"]
    if include_text_inputs:
        type_selectors.append("input[type='text']")
    locators.append(page.locator(",".join(type_selectors)))

    locators.append(page.locator(",".join(
        [
            f"input[placeholder*=\"{safe}\" i]",
            f"input[aria-label*=\"{safe}\" i]",
            f"input[name*=\"{safe}\" i]",
        ]
    )))

    return locators


def format_tool_observation(result: ToolResult) -> str:
    return textwrap.shorten(
        f"{result.name}: {'ok' if result.success else 'fail'} — {result.observation}",
        width=900,
        placeholder="…",
    )


__all__ = ["BrowserToolbox", "format_tool_observation"]

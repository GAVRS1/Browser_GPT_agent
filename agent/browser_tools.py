from __future__ import annotations
import time
import contextlib
import json
import base64
import re
import textwrap
import os
from collections import deque
from urllib.parse import parse_qs, unquote, urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
from playwright.sync_api import Locator, Page, TimeoutError as PlaywrightTimeoutError

from browser.context import get_page
from agent.llm_client import get_client
from agent.tools_init import dom_snapshot


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ToolResult:
    name: str
    success: bool
    observation: str


@dataclass
class ToolSchema:
    """Описание инструмента в формате MCP с конвертацией в OpenAI."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def as_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def as_mcp(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class VisionHint:
    selector: Optional[str] = None
    click: Optional[Dict[str, float]] = None
    tab_steps: int = 0
    reason: str = ""


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

    def __init__(
        self,
        page: Optional[Page] = None,
        screenshots_dir: Optional[Path] = None,
    ) -> None:
        self.page: Page = page or get_page()
        self._apply_default_timeouts(self.page)
        self.screenshots_dir = screenshots_dir or SCREENSHOTS_DIR
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    def _apply_default_timeouts(self, page: Page) -> None:
        """Настраивает дефолтные таймауты Playwright с учётом навигации.

        Таймауты для действий оставляем короткими (по умолчанию ~3 секунды),
        чтобы клики/поиски элементов быстро отваливались без заморозки консоли.
        При этом таймаут навигации увеличен до 12 секунд по умолчанию, чтобы
        типовая загрузка страницы успевала завершиться. Оба значения можно
        переопределить через переменные окружения BROWSER_DEFAULT_TIMEOUT_MS и
        BROWSER_NAVIGATION_TIMEOUT_MS.
        """
        def _timeout_from_env(env_var: str, default: int) -> int:
            try:
                return int(os.getenv(env_var, default))
            except ValueError:
                return default

        default_timeout_ms = _timeout_from_env("BROWSER_DEFAULT_TIMEOUT_MS", 3000)
        navigation_timeout_ms = _timeout_from_env(
            "BROWSER_NAVIGATION_TIMEOUT_MS", 12000
        )

        # Навигацию даём дольше, чтобы страница успела загрузиться; короткие
        # таймауты для действий помогают не зависать на недоступных элементах.
        with contextlib.suppress(Exception):
            page.set_default_timeout(default_timeout_ms)
            page.set_default_navigation_timeout(navigation_timeout_ms)

    def _ensure_page_alive(self) -> None:
        """Переинициализирует вкладку, если текущая была закрыта.

        При длительных сессиях Playwright-страница иногда оказывается закрытой
        (например, после ошибок навигации). В таком случае любой клик зависает
        до таймаута и срабатывает ошибка «Target page/context has been closed».
        Чтобы избежать подвисаний, перед действиями убеждаемся, что страница
        жива, и при необходимости создаём новую через get_page().
        """

        try:
            if self.page and not self.page.is_closed():
                return
        except Exception:  # noqa: BLE001
            pass

        # Если текущая страница недоступна — берём новую из контекста
        try:
            new_page = get_page()
            self.page = new_page
            self._apply_default_timeouts(new_page)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[tools] Failed to refresh page after close: {exc}")

    # ------------------------------------------------------------
    # OpenAI tool schemas
    # ------------------------------------------------------------
    def tool_schemas(self) -> List[ToolSchema]:
        return [
            ToolSchema(
                name="read_view",
                description="Собрать краткий список заметных элементов страницы и их текст.",
                parameters={"type": "object", "properties": {}},
            ),
            ToolSchema(
                name="open_url",
                description=(
                    "Перейти по указанному URL (используется, если агент хочет вручную открыть страницу)."
                ),
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            ),
            ToolSchema(
                name="click",
                description=(
                    "Нажать на элемент по тексту, ARIA name или CSS-селектору (без заранее зашитых селекторов)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Текст/label/placeholder или CSS-селектор для поиска элемента.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ToolSchema(
                name="type_text",
                description="Ввести текст в поле по label/placeholder/CSS и при необходимости отправить (Enter).",
                parameters={
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
            ),
            ToolSchema(
                name="scroll",
                description=(
                    "Прокрутить страницу вверх/вниз на указанный отрезок. "
                    "На страницах с сеткой карточек используй скролл осторожно: "
                    "если уже видна сетка предложений, лучше сначала кликать по "
                    "карточкам, а не скроллить."
                ),
                parameters={
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
            ),
            ToolSchema(
                name="go_back",
                description="Вернуться на предыдущую страницу в истории браузера.",
                parameters={"type": "object", "properties": {}},
            ),
            ToolSchema(
                name="click_product_card",
                description=(
                    "Клик по одной из крупных карточек предложения в основной части страницы. "
                    "Полезно, когда появилась сетка товаров/слотов, но к конкретной карточке "
                    "сложно обратиться по тексту."
                ),
                parameters={"type": "object", "properties": {}},
            ),
            ToolSchema(
                name="take_screenshot",
                description=(
                    "Сделать скриншот текущей страницы. "
                    "Возвращает путь к файлу скриншота, который можно посмотреть в логах."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "full_page": {
                            "type": "boolean",
                            "description": "Снимать всю страницу целиком (true) или только видимую область (false).",
                            "default": False,
                        }
                    },
                },
            ),
            ToolSchema(
                name="snapshot_accessibility",
                description=(
                    "Получить dom snapshot и компактное дерево доступности для анализа a11y рядом с DOM."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "max_nodes": {
                            "type": "integer",
                            "description": "Лимит узлов в accessibility tree для ответа (по умолчанию 120).",
                            "default": 120,
                        }
                    },
                },
            ),
        ]

    def openai_tools(self) -> List[Dict[str, Any]]:
        return [schema.as_openai() for schema in self.tool_schemas()]

    def mcp_tools(self) -> List[Dict[str, Any]]:
        return [schema.as_mcp() for schema in self.tool_schemas()]

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
            if tool_name == "snapshot_accessibility":
                return ToolResult(
                    tool_name,
                    True,
                    self.snapshot_accessibility(int(arguments.get("max_nodes", 120))),
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
        self._ensure_page_alive()
        page = self.page

        # Имя файла по времени
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.screenshots_dir / f"screenshot_{timestamp}.png"

        try:
            page.screenshot(path=str(path), full_page=full_page)
            logger.info(f"[tools] Screenshot saved to {path}")
            title = safe_title(page)
            url = ""
            with contextlib.suppress(Exception):
                url = page.url or ""
            description = (
                f"Screenshot saved to {path}. "
                f"Context: url={url or 'unknown'}, title={title or '—'}. "
                "Повтори ссылку на этот файл в следующих шагах, если нужен тот же снимок."
            )
            return description
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[tools] Screenshot failed: {exc}")
            # Пробрасываем наверх, чтобы execute() пометил инструмент как fail
            raise

    def snapshot_accessibility(self, max_nodes: int = 120) -> str:
        """Возвращает dom snapshot и уменьшенный accessibility tree."""

        self._ensure_page_alive()
        page = self.page

        dom = dom_snapshot(max_text=5000, max_items=60)
        a11y_tree = _safe_accessibility_tree(page, max_nodes=max_nodes)
        payload = {"dom_snapshot": dom, "accessibility_tree": a11y_tree}
        return json.dumps(payload, ensure_ascii=False)[:2400]
    def read_view(self) -> str:
        """Возвращает компактный обзор страницы.

        Учитываем только заметные интерактивные элементы, чтобы не слать
        огромные DOM-фрагменты. Каждый элемент содержит текст, роль и
        эвристический локатор, который можно попробовать в клике/вводе.

        ДОПОЛНИТЕЛЬНО:
        - пытаемся выделить крупные карточки товаров/слотов (product_cards), чтобы
          агент понимал, где сетка предложений.
        """

        self._ensure_page_alive()
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
        self._ensure_page_alive()
        self.page.goto(url, wait_until="domcontentloaded")
        return f"Открыл {self.page.url}"

    def click(self, query: str) -> str:
        if not query:
            return "Пустой запрос — клик не выполнен"

        # Обновляем страницу, если предыдущая вкладка умерла
        self._ensure_page_alive()

        page = self.page
        strategies: List[Locator] = []

        # если это похоже на поиск – тоже пробуем поиск
        if _looks_like_search_intent(query):
            strategies.extend(_search_locators(page, query, include_text_inputs=True))

        strategies.extend(
            [
                page.get_by_role("button", name=re.compile(query, re.IGNORECASE)),
                page.get_by_role("link", name=re.compile(query, re.IGNORECASE)),
                page.get_by_text(query, exact=False),
            ]
        )

        strategies.extend(
            _attribute_locators(
                page,
                query,
                tags=["button", "a", "input", "summary", "div", "span"],
            )
        )

        if _looks_like_css(query):
            strategies.append(page.locator(query))

        candidates, candidate = _collect_candidates_for_logging(strategies)

        result_message = ""
        chosen_summary = ""

        if candidate:
            try:
                # Критическое место: если Playwright залипнет, мы не повиснем навсегда
                candidate.click(timeout=2000)
                result_message = f"Клик по {query} выполнен"
                chosen_summary = _describe_locator(candidate)
            except PlaywrightTimeoutError as exc:
                result_message = f"Клик по {query} не завершился по таймауту: {exc}"
            except Exception as exc:
                result_message = f"Ошибка при клике по {query}: {exc}"

                # Если страница или контекст закрылись, попробуем восстановиться
                if "has been closed" in str(exc).lower():
                    self._ensure_page_alive()
                    result_message += " (вкладка была переподключена, попробуй ещё раз)"
        else:
            result_message = f"Не нашёл, куда кликнуть по запросу: {query}"

        _log_interaction(
            action="click",
            query=query,
            candidates=candidates,
            chosen_summary=chosen_summary,
            result=result_message,
        )

        return result_message


    def type_text(self, query: str, text: str, press_enter: bool = False) -> str:
        if not query:
            return "Нет запроса для ввода"

        self._ensure_page_alive()
        page = self.page

        if _looks_like_search_intent(query):
            direct_locator = _first_visible(
                _search_locators(page, query, include_text_inputs=True)
            )
            if direct_locator:
                direct_locator.fill(text, timeout=2000)
                if press_enter:
                    direct_locator.press("Enter")
                result_message = (
                    "Нашёл поле поиска по общим атрибутам (placeholder/aria-label/name) "
                    f"и ввёл запрос '{query}'."
                )
                _log_interaction(
                    action="type_text",
                    query=query,
                    candidates=[{
                        "summary": _describe_locator(direct_locator),
                        "visible": True,
                    }],
                    chosen_summary=_describe_locator(direct_locator),
                    result=result_message,
                )
                return result_message

            vision_hint = self._vision_locate_input(query)
            if vision_hint:
                applied = self._apply_vision_hint(vision_hint, text, press_enter)
                if applied:
                    result_message = (
                        f"Ввёл текст через vision-fallback для запроса '{query}'. "
                        f"Основание: {vision_hint.reason or 'подсказка модели'}"
                    )
                    candidates = _vision_candidates_for_logging(vision_hint)
                    _log_interaction(
                        action="type_text-vision",
                        query=query,
                        candidates=candidates,
                        chosen_summary=_describe_locator_from_hint(vision_hint),
                        result=result_message,
                    )
                    return result_message

        candidate = self._find_text_locator(query)
        if candidate:
            candidate.fill(text, timeout=2000)
            if press_enter:
                candidate.press("Enter")
            result_message = f"Ввёл текст в поле {query}"
            _log_interaction(
                action="type_text",
                query=query,
                candidates=[{
                    "summary": _describe_locator(candidate),
                    "visible": True,
                }],
                chosen_summary=_describe_locator(candidate),
                result=result_message,
            )
            return result_message

        vision_hint = self._vision_locate_input(query)
        if vision_hint:
            applied = self._apply_vision_hint(vision_hint, text, press_enter)
            if applied:
                result_message = (
                    f"Ввёл текст через vision-fallback для запроса '{query}'. "
                    f"Основание: {vision_hint.reason or 'подсказка модели'}"
                )
                candidates = _vision_candidates_for_logging(vision_hint)
                _log_interaction(
                    action="type_text-vision",
                    query=query,
                    candidates=candidates,
                    chosen_summary=_describe_locator_from_hint(vision_hint),
                    result=result_message,
                )
                return result_message

        result_message = f"Не удалось найти поле для ввода по запросу: {query}"
        _log_interaction(
            action="type_text",
            query=query,
            candidates=[],
            chosen_summary="",
            result=result_message,
        )
        return result_message

    def _find_text_locator(self, query: str) -> Optional[Locator]:
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

        return _first_visible(strategies)

    def _capture_screenshot_bytes(self) -> tuple[str, bytes]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.screenshots_dir / f"vision_fallback_{timestamp}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image_bytes = self.page.screenshot(path=str(path), full_page=False)
        return str(path), image_bytes

    def _vision_locate_input(self, query: str) -> Optional[VisionHint]:
        client = get_client()
        if client is None:
            logger.warning("[vision] Vision fallback skipped: OpenAI client недоступен")
            _log_interaction(
                action="vision_locate_input",
                query=query,
                candidates=[],
                chosen_summary="",
                result="vision client unavailable",
            )
            return None

        screenshot_path, image_bytes = self._capture_screenshot_bytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = textwrap.dedent(
            f"""
            Ты помогатель для браузерного агента. На скриншоте страница. Надо найти поле
            ввода/поиска соответствующее запросу: "{query}".

            Ответь строго JSON с полями:
            - selector: короткий CSS/ARIA, если можно надёжно сослаться на элемент без хардкодов.
            - click: объект {{"x": number, "y": number}} — координаты клика по видимому полю.
            - tab_steps: количество нажатий Tab, чтобы сфокусировать поле, если координаты не подходят.
            - reason: краткое объяснение, почему выбран именно этот вариант.

            Если уверен в селекторе — заполни selector и не указывай click.
            Если селектора нет, укажи только click.
            Минимизируй галлюцинации: используй только то, что видно на скриншоте.
            """
        ).strip()

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # vision-поддержка
                messages=[
                    {"role": "system", "content": "Ты помогаешь роботу найти поле ввода."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    },
                ],
                max_tokens=300,
                temperature=0.1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[vision] Vision fallback failed: {exc}")
            return None

        content = response.choices[0].message.content if response.choices else ""
        hint = _parse_vision_hint(content)
        if hint:
            vision_candidates = _vision_candidates_for_logging(hint)
            summary = _describe_locator_from_hint(hint)
            result_message = (
                f"[vision] Использован vision-fallback для '{query}' (screenshot: {screenshot_path}). "
                f"reason: {hint.reason or 'не указана'}"
            )
            _log_interaction(
                action="vision_locate_input",
                query=query,
                candidates=vision_candidates,
                chosen_summary=summary,
                result=result_message,
            )
            logger.info(result_message)
            return hint

        logger.warning(
            f"[vision] Не удалось распарсить ответ vision для запроса '{query}': {content}"
        )
        _log_interaction(
            action="vision_locate_input",
            query=query,
            candidates=[],
            chosen_summary="",
            result=f"vision hint parse failed for '{query}'",
        )
        return None

    def _apply_vision_hint(self, hint: VisionHint, text: str, press_enter: bool) -> bool:
        page = self.page

        if hint.selector:
            locator = _first_visible([page.locator(hint.selector)])
            if locator:
                locator.click(timeout=2000)
                locator.fill(text, timeout=2000)
                if press_enter:
                    locator.press("Enter")
                return True

        if hint.click and {"x", "y"}.issubset(hint.click):
            try:
                x = float(hint.click.get("x", 0))
                y = float(hint.click.get("y", 0))
                page.mouse.click(x, y)
                page.keyboard.type(text)
                if press_enter:
                    page.keyboard.press("Enter")
                return True
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[vision] Ошибка клика по координатам {hint.click}: {exc}")

        if hint.tab_steps:
            steps = max(0, int(hint.tab_steps))
            for _ in range(steps):
                page.keyboard.press("Tab")
            page.keyboard.type(text)
            if press_enter:
                page.keyboard.press("Enter")
            return steps > 0

        return False

    def scroll(self, direction: str = "down", amount: int = 800) -> str:
        """
        Прокрутка страницы.

        ДОП. ЛОГИКА:
        - Если уже видна сетка крупных карточек, лучше переходить к выбору карточек,
          а не уезжать всё ниже.
        """

        self._ensure_page_alive()
        page = self.page

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

                if big >= 4:
                    return (
                        "Видна сетка карточек предложений. "
                        "Скролл временно отключён: выбери подходящий элемент и кликни по карточке "
                        "(через click по названию или через инструмент click_product_card)."
                    )
        except Exception:
            pass

        delta = amount if direction != "up" else -amount
        with contextlib.suppress(Exception):
            page.mouse.wheel(0, delta)
        return f"Прокрутил {direction} на {amount}"

    def go_back(self) -> str:
        self._ensure_page_alive()
        with contextlib.suppress(Exception):
            self.page.go_back()
        return "Вернулся на предыдущую страницу"

    def click_product_card(self) -> str:
        """
        Улучшенный алгоритм клика по карточке товара или слота.

        При выборе карточек по размеру стараемся НЕ выбирать информационные блоки
        с текстом вида «15 мин» или похожими метками времени — они понижаются в приоритете.
        """

        self._ensure_page_alive()
        page = self.page

        search_tokens = _collect_search_tokens(page)

        # Обычный выбор большой карточки товара
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

                add_button = _find_add_button(el)
                add_button_present = add_button is not None

                token_matches = _count_token_matches(inner, search_tokens)
                score = _compute_card_score(
                    area=area,
                    has_price=has_price,
                    has_title=has_title,
                    is_recipe=is_recipe,
                    has_add_button=add_button_present,
                    token_matches=token_matches,
                )

                candidates.append(
                    {
                        "score": score,
                        "locator": el,
                        "add_button": add_button,
                        "text": inner,
                        "matches": token_matches,
                    }
                )
            except Exception:
                continue

        if not candidates:
            return "Не удалось найти карточку товара."

        candidates.sort(key=lambda x: x["score"], reverse=True)

        failure_reasons: list[str] = []
        for cand in candidates:
            before_state = _snapshot_page_state(page)
            before_cart_filled = _cart_has_items(page)

            try:
                cand["locator"].scroll_into_view_if_needed()
            except Exception:
                pass

            clicked = False
            for target in [cand.get("add_button"), cand.get("locator")]:
                if not target:
                    continue
                try:
                    target.click(timeout=4000, force=True)
                    clicked = True
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"[tools] click_product_card: click failed for candidate: {exc}"
                    )

            if not clicked:
                failure_reasons.append("клик не сработал")
                continue

            time.sleep(1)
            after_state = _snapshot_page_state(page)
            cart_has_items = _cart_has_items(page)
            if _state_changed(before_state, after_state) and (
                cart_has_items or before_cart_filled != cart_has_items
            ):
                if cart_has_items:
                    return "Кликнул по карточке товара и корзина не пустая."
                failure_reasons.append("после клика корзина выглядит пустой")
                continue

            failure_reasons.append("клик не изменил состояние")
            # Пытаемся следующую карточку

        details = "; ".join(failure_reasons) or "кандидаты не подошли"
        return f"Не удалось выбрать карточку товара: {details}."


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


def _truncate(value: str, limit: int = 160) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _describe_locator(locator: Optional[Locator]) -> str:
    if locator is None:
        return ""

    parts: List[str] = []

    with contextlib.suppress(Exception):
        tag = locator.evaluate("el => el.tagName.toLowerCase()") or ""
        if tag:
            parts.append(f"<{tag}>")

    with contextlib.suppress(Exception):
        text = locator.inner_text(timeout=800) or ""
        text = _truncate(re.sub(r"\s+", " ", text.strip()))
        if text:
            parts.append(f"text='{text}'")

    attrs = _collect_attributes(locator)
    if attrs:
        rendered_attrs = ", ".join(
            f"{k}={_truncate(v, 40)}" for k, v in list(attrs.items())[:4]
        )
        parts.append(f"attrs({rendered_attrs})")

    return " | ".join(parts) or "locator"


def _collect_candidates_for_logging(
    locators: Iterable[Locator],
    limit: int = 5,
) -> tuple[list[Dict[str, Any]], Optional[Locator]]:
    seen: set[str] = set()
    candidates: list[Dict[str, Any]] = []
    chosen: Optional[Locator] = None

    for locator in locators:
        candidate: Optional[Locator] = None
        key = repr(locator)
        if key in seen:
            continue
        seen.add(key)

        with contextlib.suppress(Exception):
            candidate = locator.first
        if candidate is None:
            continue

        visible = False
        with contextlib.suppress(Exception):
            visible = candidate.is_visible(timeout=1500)

        summary = _describe_locator(candidate)
        candidates.append({"summary": summary, "visible": visible})

        if chosen is None and visible:
            chosen = candidate

        if len(candidates) >= limit:
            break

    return candidates, chosen


def _log_interaction(
    *,
    action: str,
    query: str,
    candidates: list[Dict[str, Any]],
    chosen_summary: str,
    result: str,
) -> None:
    payload = {
        "action": action,
        "query": _truncate(query, 160),
        "candidates": [
            {"summary": _truncate(item.get("summary", ""), 200), "visible": bool(item.get("visible", False))}
            for item in candidates[:8]
        ],
        "chosen": _truncate(chosen_summary or "", 200),
        "result": _truncate(result, 200),
    }
    logger.info(f"[trace] {json.dumps(payload, ensure_ascii=False)}")


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


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[\w']+", text.lower())
    seen: set[str] = set()
    tokens: list[str] = []
    for w in words:
        if len(w) < 2:
            continue
        if w in seen:
            continue
        seen.add(w)
        tokens.append(w)
    return tokens


def _extract_search_tokens_from_url(url: str) -> list[str]:
    tokens: list[str] = []
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    for key in ["text", "query", "q", "search", "keyword", "term", "s"]:
        for value in query_params.get(key, []):
            tokens.extend(_tokenize(unquote(value)))

    # Попытка достать поисковый текст из последней части пути
    path_chunks = [chunk for chunk in parsed.path.split("/") if chunk]
    if path_chunks:
        tail = unquote(path_chunks[-1])
        tokens.extend(_tokenize(tail))

    seen: set[str] = set()
    unique_tokens: list[str] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        unique_tokens.append(tok)
    return unique_tokens


def _filter_a11y_node(raw: Dict[str, Any]) -> Dict[str, Any]:
    allowed_keys = {
        "role",
        "name",
        "value",
        "description",
        "checked",
        "expanded",
        "disabled",
        "focused",
        "pressed",
        "selected",
        "level",
        "multiline",
        "placeholder",
        "readonly",
        "required",
        "roledescription",
        "autocomplete",
    }
    filtered = {k: v for k, v in raw.items() if k in allowed_keys and v not in (None, "")}
    return filtered


def _safe_accessibility_tree(page: Page, max_nodes: int = 120) -> Dict[str, Any]:
    try:
        snapshot = page.accessibility.snapshot(interesting_only=False)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[tools] Accessibility snapshot failed: {exc}")
        return {"error": str(exc)}

    if not snapshot:
        return {}

    queue: deque[tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = deque()
    queue.append((snapshot, None))
    root_copy: Optional[Dict[str, Any]] = None
    produced = 0
    last_copied: Optional[Dict[str, Any]] = None

    while queue and produced < max_nodes:
        node, parent_copy = queue.popleft()
        filtered = _filter_a11y_node(node)
        produced += 1
        last_copied = filtered

        if parent_copy is None:
            root_copy = filtered
        else:
            parent_copy.setdefault("children", []).append(filtered)

        children = node.get("children") or []
        for child in children:
            if produced + len(queue) >= max_nodes:
                break
            queue.append((child, filtered))

    if queue and last_copied is not None:
        last_copied["truncated"] = True

    return root_copy or {}


def _collect_search_tokens(page: Page) -> list[str]:
    tokens: list[str] = []
    with contextlib.suppress(Exception):
        tokens.extend(_extract_search_tokens_from_url(page.url or ""))

    try:
        search_inputs = page.locator("input[type='search'], input[name*='search'], input[name*='text']")
        total = search_inputs.count()
    except Exception:
        total = 0

    max_inputs = min(total, 3)
    for i in range(max_inputs):
        locator = search_inputs.nth(i)
        with contextlib.suppress(Exception):
            value = locator.input_value(timeout=800) or ""
            tokens.extend(_tokenize(value))

    seen: set[str] = set()
    unique_tokens: list[str] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        unique_tokens.append(tok)
    return unique_tokens


def _count_token_matches(text: str, tokens: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for tok in tokens if tok and tok in lowered)


def _compute_card_score(
    *,
    area: float,
    has_price: bool,
    has_title: bool,
    is_recipe: bool,
    has_add_button: bool,
    token_matches: int,
) -> float:
    score = area
    if has_price:
        score += 50000
    if has_title:
        score += 15000
    if has_add_button:
        score += 60000
    if token_matches:
        score += 70000 * token_matches
    if is_recipe:
        score -= 40000
    return score


def _find_add_button(container: Locator) -> Optional[Locator]:
    try:
        button = container.locator("button, [role='button']").filter(
            has_text=re.compile(
                r"(в корзину|добавить|купить|в\s*корзине|add to cart|add|куплено|\+)",
                re.IGNORECASE,
            )
        )
        if button.count() > 0 and button.first.is_enabled(timeout=800):
            return button.first
    except Exception:
        return None
    return None


def _snapshot_page_state(page: Page) -> dict:
    state = {
        "url": "",
        "cart": "",
        "dom_len": 0,
    }
    with contextlib.suppress(Exception):
        state["url"] = page.url or ""
    with contextlib.suppress(Exception):
        cart_locator = page.get_by_text(re.compile("корзин|basket|cart", re.IGNORECASE))
        if cart_locator and cart_locator.count() > 0:
            state["cart"] = _safe_text(cart_locator.first)
    with contextlib.suppress(Exception):
        content = page.content()
        if content:
            state["dom_len"] = len(content)
    return state


def _state_changed(before: dict, after: dict) -> bool:
    if before.get("url") != after.get("url"):
        return True
    if before.get("cart") != after.get("cart") and (before.get("cart") or after.get("cart")):
        return True
    if abs(after.get("dom_len", 0) - before.get("dom_len", 0)) > 50:
        return True
    return False


def _collect_cart_texts(page: Page) -> list[str]:
    texts: list[str] = []

    selectors = [
        page.get_by_role("link", name=re.compile("корз|basket|cart", re.IGNORECASE)),
        page.get_by_role("button", name=re.compile("корз|basket|cart", re.IGNORECASE)),
        page.locator(
            ",".join(
                [
                    "[aria-label*='корз' i]",
                    "[aria-label*='cart' i]",
                    "[href*='cart']",
                    "[href*='basket']",
                    "[data-testid*='cart' i]",
                    "[data-testid*='basket' i]",
                ]
            )
        ),
    ]

    for locator in selectors:
        with contextlib.suppress(Exception):
            count = locator.count()
            for idx in range(min(count, 3)):
                text = _safe_text(locator.nth(idx))
                if text:
                    texts.append(text)
    return texts


def _cart_has_items(page: Page) -> bool:
    texts = _collect_cart_texts(page)
    combined = " ".join(texts)
    if not combined:
        return False

    if re.search(r"пуст", combined, re.IGNORECASE):
        return False

    numbers = [int(n) for n in re.findall(r"(\d+)", combined) if n.isdigit()]
    return any(n > 0 for n in numbers)


def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _parse_vision_hint(content: str) -> Optional[VisionHint]:
    raw_json = _extract_json_block(content)
    if not raw_json:
        return None
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    selector = data.get("selector") or None
    click = data.get("click") if isinstance(data.get("click"), dict) else None
    tab_steps = data.get("tab_steps") or 0
    reason = data.get("reason") or ""
    return VisionHint(selector=selector, click=click, tab_steps=tab_steps, reason=reason)


def _describe_locator_from_hint(hint: Optional[VisionHint]) -> str:
    if hint is None:
        return ""

    parts: list[str] = []
    if hint.selector:
        parts.append(f"selector='{_truncate(hint.selector, 80)}'")
    if hint.click:
        coords = hint.click
        x = coords.get("x")
        y = coords.get("y")
        parts.append(f"coords=({x}, {y})")
    if hint.tab_steps:
        parts.append(f"tab_steps={hint.tab_steps}")
    if hint.reason:
        parts.append(f"reason='{_truncate(hint.reason, 80)}'")
    return " | ".join(parts)


def _vision_candidates_for_logging(hint: Optional[VisionHint]) -> list[Dict[str, Any]]:
    if hint is None:
        return []

    candidates: list[Dict[str, Any]] = []
    if hint.selector:
        candidates.append({"summary": f"selector: {hint.selector}", "visible": False})
    if hint.click:
        coords = hint.click
        candidates.append(
            {
                "summary": f"coords: x={coords.get('x')}, y={coords.get('y')}",
                "visible": True,
            }
        )
    if hint.tab_steps:
        candidates.append({"summary": f"tab navigation: {hint.tab_steps}", "visible": True})
    return candidates


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

    # Текстовые поля — исключаем checkbox/radio/button, которые нельзя fill()
    text_input_selector = (
        "input:not([type='checkbox']):not([type='radio']):"
        "not([type='button']):not([type='submit']):not([type='reset']):"
        "not([type='file']):not([type='range']):not([type='color'])"
    )

    search_keywords = ["поиск", "search", "найти"]
    keyword_selectors = []
    for kw in search_keywords:
        kw_safe = _css_escape(kw)
        keyword_selectors.append(f"{text_input_selector}[placeholder*=\"{kw_safe}\" i]")
        keyword_selectors.append(f"{text_input_selector}[aria-label*=\"{kw_safe}\" i]")
        keyword_selectors.append(f"{text_input_selector}[name*=\"{kw_safe}\" i]")

    locators.append(page.locator(",".join(keyword_selectors)))

    type_selectors = ["input[type='search']"]
    if include_text_inputs:
        type_selectors.extend(
            [
                "input[type='text']",
                "input[type='email']",
                "input[type='url']",
                "input[type='tel']",
                "input[type='number']",
            ]
        )
    locators.append(page.locator(",".join(type_selectors)))

    locators.append(page.locator(",".join(
        [
            f"{text_input_selector}[placeholder*=\"{safe}\" i]",
            f"{text_input_selector}[aria-label*=\"{safe}\" i]",
            f"{text_input_selector}[name*=\"{safe}\" i]",
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

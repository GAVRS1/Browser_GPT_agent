from __future__ import annotations

from typing import Any, Dict

from loguru import logger
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from browser.context import get_page
from config.proxy import apply_requests_proxy
from config import timeouts


_initialized = False


# ============================================================================
# Low-level browser helpers
# ============================================================================

def _current_page():
    """
    Возвращает активную вкладку браузера.

    Вынесено в отдельную функцию, чтобы можно было централизованно
    логировать любые проблемы с контекстом.
    """
    page = get_page()
    return page


# ============================================================================
# Generic tools (утилиты для под-агентов)
# ============================================================================


def dom_snapshot(max_text: int = 6000, max_items: int = 40) -> Dict[str, Any]:
    """
    Снимает "срез" текущей страницы для анализа моделью.

    ВАЖНО: мы НЕ возвращаем сырое HTML целиком, чтобы не переполнить контекст.
    Вместо этого:
    - даём заголовок страницы
    - текущий URL
    - усечённый видимый текст
    - первые несколько кнопок / ссылок / инпутов с кратким описанием
    """
    page = _current_page()

    try:
        snapshot = page.evaluate(
            """({ maxText, maxItems }) => {
                const byInnerText = (elements) => {
                    return Array.from(elements).slice(0, maxItems).map((el, idx) => {
                        const rect = el.getBoundingClientRect();
                        const text = (el.innerText || el.value || '').trim();
                        const role = el.getAttribute('role') || null;
                        return {
                            index: idx,
                            tag: el.tagName,
                            text: text.slice(0, 200),
                            type: el.getAttribute('type'),
                            role,
                            href: el.getAttribute('href'),
                            id: el.id || null,
                            classes: el.className || null,
                            rect: {
                                x: rect.x,
                                y: rect.y,
                                width: rect.width,
                                height: rect.height
                            }
                        };
                    });
                };

                const bodyText = (document.body.innerText || '').replace(/\\s+/g, ' ').trim();

                return {
                    title: document.title || '',
                    url: window.location.href,
                    text: bodyText.slice(0, maxText),
                    buttons: byInnerText(document.querySelectorAll('button, input[type="button"], input[type="submit"]')),
                    links: byInnerText(document.querySelectorAll('a[href]')),
                    inputs: byInnerText(document.querySelectorAll('input[type="text"], input[type="email"], input[type="search"], textarea'))
                };
            }""",
            {"maxText": max_text, "maxItems": max_items},
        )
        logger.info("[tools] DOM snapshot captured for analysis")
        return snapshot
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[tools] Failed to capture DOM snapshot: {exc}")
        return {
            "title": page.title() if page else "",
            "url": page.url if page else "",
            "text": "",
            "buttons": [],
            "links": [],
            "inputs": [],
            "error": str(exc),
        }


def click(selector: str, timeout_ms: int | None = None) -> Dict[str, Any]:
    """
    Кликает по элементу, выбранному под-агентом через CSS-селектор.

    ВАЖНО: здесь нет хардкода конкретных селекторов — только то, что передаст логика агента.
    """
    page = _current_page()
    logger.info(f"[tools] click(selector={selector!r})")

    if timeout_ms is None:
        timeout_ms = timeouts.tool_action_timeout_ms()

    try:
        locator = page.locator(selector).first
        locator.wait_for(state="visible", timeout=timeout_ms)
        locator.click()
        return {"ok": True, "selector": selector}
    except PlaywrightTimeoutError:
        logger.warning(f"[tools] click timeout for selector={selector!r}")
        return {"ok": False, "selector": selector, "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[tools] click error for selector={selector!r}: {exc}")
        return {"ok": False, "selector": selector, "error": str(exc)}


def click_by_text(
    text_query: str, timeout_ms: int | None = None, fuzzy: bool = True
) -> Dict[str, Any]:
    """
    Находит кликаемый элемент по видимому тексту и нажимает его.

    Это даёт агенту более "человеческий" инструмент: он может искать по словам
    'Удалить', 'Спам', 'Входящие' и т.п., а мы уже сами определяем селектор.
    """
    page = _current_page()
    logger.info(f"[tools] click_by_text(text={text_query!r})")

    normalized_query = text_query.strip().lower()

    try:
        selector = page.evaluate(
            """(query, fuzzy) => {
                const norm = (s) => (s || '').toLowerCase().trim();
                const textMatch = (el) => {
                    const text = norm(el.innerText || el.value || '');
                    if (!text) return false;
                    if (fuzzy) {
                        return text.includes(query);
                    }
                    return text === query;
                };

                const candidates = Array.from(document.querySelectorAll(
                    'button, [role="button"], a, input[type="button"], input[type="submit"]'
                ));

                for (const el of candidates) {
                    if (textMatch(el)) {
                        if (el.id) return '#' + el.id;
                        if (el.tagName === 'A' && el.getAttribute('href')) {
                            return 'a[href="' + el.getAttribute('href') + '"]';
                        }
                        const classes = (el.className || '').toString().trim().split(/\\s+/).filter(Boolean);
                        if (classes.length) {
                            return el.tagName.toLowerCase() + '.' + classes.join('.');
                        }
                        return el.tagName.toLowerCase();
                    }
                }
                return null;
            }""",
            normalized_query,
            fuzzy,
        )

        if not selector:
            logger.info(f"[tools] click_by_text: no element found for text={text_query!r}")
            return {"ok": False, "error": "not_found", "selector": None}

        return click(selector, timeout_ms=timeout_ms)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[tools] click_by_text error for text={text_query!r}: {exc}")
        return {"ok": False, "error": str(exc), "selector": None}


def type_text(
    selector: str, text: str, clear: bool = True, timeout_ms: int | None = None
) -> Dict[str, Any]:
    """
    Вводит текст в поле (input / textarea), выбранное логикой агента через CSS-селектор.
    """
    page = _current_page()
    logger.info(f"[tools] type_text(selector={selector!r}, text_len={len(text)})")

    if timeout_ms is None:
        timeout_ms = timeouts.tool_action_timeout_ms()

    try:
        locator = page.locator(selector).first
        locator.wait_for(state="visible", timeout=timeout_ms)
        if clear:
            locator.fill(text)
        else:
            locator.type(text)
        return {"ok": True, "selector": selector}
    except PlaywrightTimeoutError:
        logger.warning(f"[tools] type_text timeout for selector={selector!r}")
        return {"ok": False, "selector": selector, "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[tools] type_text error for selector={selector!r}: {exc}")
        return {"ok": False, "selector": selector, "error": str(exc)}


def wait_for_dom_stable(timeout_ms: int | None = None) -> Dict[str, Any]:
    """
    Грубый способ дождаться стабилизации страницы:
    ждём пока не изменится размер DOM/документа.
    """
    page = _current_page()
    logger.info("[tools] wait_for_dom_stable")

    if timeout_ms is None:
        timeout_ms = timeouts.tool_dom_stable_timeout_ms()

    try:
        result = page.evaluate(
            """(timeoutMs) => {
                return new Promise((resolve) => {
                    const start = performance.now();
                    let lastSize = document.body.innerHTML.length;
                    let lastChange = performance.now();

                    const check = () => {
                        const now = performance.now();
                        const size = document.body.innerHTML.length;
                        if (size !== lastSize) {
                            lastSize = size;
                            lastChange = now;
                        }
                        if (now - start > timeoutMs) {
                            resolve({ stable: false, reason: 'timeout' });
                            return;
                        }
                        if (now - lastChange > 500) {
                            resolve({ stable: true, reason: 'no_changes_500ms' });
                            return;
                        }
                        requestAnimationFrame(check);
                    };
                    check();
                });
            }""",
            timeout_ms,
        )
        return {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[tools] wait_for_dom_stable error: {exc}")
        return {"ok": False, "error": str(exc)}


# ============================================================================
# Initialization
# ============================================================================

def register_all_tools() -> None:
    """
    Подготовка окружения для работы агента.

    Сейчас:
    - настраиваем HTTP(S)-прокси для LLM/HTTP-запросов.
    - предоставляем набор утилит для работы со страницей (dom_snapshot, click, и т.д.)
      как обычные Python-функции, которые могут вызывать под-агенты.

    Никакого отдельного "реестра" инструментов здесь больше нет —
    поэтому не будет предупреждений в логах.
    """
    global _initialized
    if _initialized:
        return

    apply_requests_proxy()
    logger.info("[tools] Environment prepared (proxy configured).")
    _initialized = True


__all__ = [
    "register_all_tools",
    "dom_snapshot",
    "click",
    "click_by_text",
    "type_text",
    "wait_for_dom_stable",
]

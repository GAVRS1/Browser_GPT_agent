import time
from pathlib import Path
from typing import Optional

import playwright
from loguru import logger
from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright

from config.proxy import (
    apply_requests_proxy,
    get_proxy_url,
    should_use_browser_proxy,
)
from config.sites import BROWSER_START_URL, GOOGLE_SEARCH_URL_TEMPLATE

# Глобальные объекты (единый браузер и контекст)
_playwright: Optional[Playwright] = None
_context: Optional[BrowserContext] = None

# Папка с профилем браузера
PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_DATA_DIR = PROJECT_ROOT / "user_data"


# ============================================================
# INTERNAL: запуск браузера через Playwright
# ============================================================


def _launch_context(playwright: Playwright, use_proxy: bool) -> BrowserContext:
    """
    Внутренняя функция запуска браузера.
    Возвращает persistent BrowserContext.
    """

    proxy_settings = None
    if use_proxy:
        proxy_url = get_proxy_url()
        if proxy_url:
            proxy_settings = {"server": proxy_url}
            logger.info(f"[browser] Starting Chromium WITH proxy: {proxy_url}")
        else:
            logger.info("[browser] BROWSER_PROXY enabled, но URL не получен – запускаем без прокси")
    else:
        logger.info("[browser] Starting Chromium WITHOUT proxy")

    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Запуск persistent контекста
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=str(USER_DATA_DIR),
        headless=False,
        proxy=proxy_settings,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--no-first-run",
            "--no-default-browser-check",
        ],
    )

    logger.info(f"[browser] Persistent profile dir: {USER_DATA_DIR}")
    return context


# ============================================================
# PUBLIC: получить persistent context
# ============================================================


def get_context() -> BrowserContext:
    """
    Возвращает persistent BrowserContext.
    Запускается только один раз за весь runtime.

    Здесь больше нет undetected-chromedriver и CDP-подключения.
    Вся работа идёт через обычный Playwright Chromium
    с сохранением профиля в user_data.
    """

    global _playwright, _context

    if _playwright is not None and _context is not None:
        return _context

    logger.info("Starting Playwright and launching persistent Chromium context...")
    logger.info(f"[browser] Playwright version: {playwright.__version__}")

    # Проксирование HTTP-запросов для LLM / API
    apply_requests_proxy()

    # Запускаем Playwright
    _playwright = sync_playwright().start()

    use_proxy = should_use_browser_proxy()

    # Пытаемся запустить браузер с учётом настроек BROWSER_PROXY.
    # Если включено, но что-то пошло не так — делаем graceful fallback без прокси.
    try:
        _context = _launch_context(_playwright, use_proxy=use_proxy)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            f"[browser] Failed to launch Chromium "
            f"{'WITH' if use_proxy else 'WITHOUT'} proxy: {exc}"
        )
        if use_proxy:
            logger.info("[browser] Retrying launch WITHOUT proxy...")
            _context = _launch_context(_playwright, use_proxy=False)
        else:
            raise

    return _context


# ============================================================
# PUBLIC: получить или создать вкладку
# ============================================================


def get_page() -> Page:
    """
    Возвращает вкладку для работы агента.
    Если вкладок нет — создаёт новую.
    """

    context = get_context()
    start_url = BROWSER_START_URL or GOOGLE_SEARCH_URL_TEMPLATE

    if context.pages:
        page = context.pages[0]
        logger.debug("Reusing existing page.")
    else:
        logger.debug("No pages found, creating a new one...")
        page = context.new_page()

    if page.url == "about:blank":
        if start_url:
            if "{query}" in start_url:
                start_url = start_url.replace("{query}", "")
            logger.debug("Empty tab detected, navigating to start URL.")
            page.goto(start_url)
        else:
            logger.debug("Empty tab detected, but no start URL configured.")

    return page


# ============================================================
# PUBLIC: закрытие браузера
# ============================================================


def shutdown_browser() -> None:
    """
    Аккуратно закрывает браузер и Playwright.
    """

    global _playwright, _context

    logger.info("Shutting down browser and Playwright...")

    # Закрываем контекст
    try:
        if _context is not None:
            _context.close()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[browser] Error closing context: {exc}")
    finally:
        _context = None

    # Останавливаем Playwright
    try:
        if _playwright is not None:
            _playwright.stop()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[browser] Error stopping Playwright: {exc}")
    finally:
        _playwright = None

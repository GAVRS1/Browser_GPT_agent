import os
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from loguru import logger

# Загружаем переменные окружения из .env
load_dotenv()


def _normalize_proxy_env_value(raw: str) -> str:
    return raw.strip().lower()


def _is_disabled_value(raw: str) -> bool:
    raw = _normalize_proxy_env_value(raw)
    return raw in {"", "false", "0", "none", "no", "off"}


def _is_enabled_value(raw: str) -> bool:
    """Возвращает True для truthy-значений."""

    raw = _normalize_proxy_env_value(raw)
    return raw in {"true", "1", "yes", "on", "enable", "enabled"}


def _build_url_from_short_notation(raw: str) -> str | None:
    """
    Поддержка сокращённых форм:
    - ip:port
    - ip:port:user:pass
    Возвращает полноценный URL вида:
    - http://ip:port
    - http://user:pass@ip:port
    """

    parts = raw.split(":")
    if len(parts) == 2:
        host, port = parts
        host = host.strip()
        port = port.strip()
        if not host or not port.isdigit():
            logger.error(f"[proxy] Invalid short PROXY format: {raw!r}")
            return None
        return f"http://{host}:{port}"

    if len(parts) == 4:
        host, port, user, password = parts
        host = host.strip()
        port = port.strip()
        user = user.strip()
        password = password.strip()
        if not host or not port.isdigit() or not user or not password:
            logger.error(f"[proxy] Invalid auth PROXY format: {raw!r}")
            return None
        return f"http://{user}:{password}@{host}:{port}"

    logger.error(f"[proxy] Unsupported PROXY format: {raw!r}")
    return None


def get_proxy_url() -> str | None:
    """
    Возвращает строку прокси из PROXY, если она включена.
    Допустимые форматы:
      - false/0/none/off/""  -> None (прокси выключен)
      - http://user:pass@ip:port
      - socks5://user:pass@ip:port
      - ip:port
      - ip:port:user:pass
    """
    raw = os.getenv("PROXY", "").strip()
    if _is_disabled_value(raw):
        logger.info("[proxy] PROXY disabled.")
        return None

    # Если указали протокол явно — считаем, что это уже URL
    if "://" in raw:
        logger.info(f"[proxy] Using PROXY as URL: {raw}")
        return raw

    # Пытаемся распарсить сокращённую форму
    url = _build_url_from_short_notation(raw)
    if url:
        logger.info(f"[proxy] Normalized short PROXY to URL: {url}")
        return url

    logger.error(f"[proxy] Failed to interpret PROXY: {raw!r}")
    return None


def apply_requests_proxy() -> None:
    """
    Выставляет HTTP(S)_PROXY переменные окружения
    для всех HTTP-клиентов (OpenAI, requests, aiohttp и т.п.).
    Если прокси отключён — очищает переменные.
    """
    proxy = get_proxy_url()
    if not proxy:
        clear_requests_proxy()
        return

    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
    logger.info(f"[proxy] HTTP(S) proxy env vars set to {proxy!r}.")


def clear_requests_proxy() -> None:
    """Очищает HTTP(S) proxy переменные окружения."""

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)
    logger.info("[proxy] HTTP(S) proxy env vars cleared.")


def get_playwright_proxy() -> dict | None:
    """
    Возвращает конфиг для Playwright:
      {"server": "...", "username": "...", "password": "..."}
    или None, если прокси не используется.
    """
    proxy_url = get_proxy_url()
    if not proxy_url:
        return None

    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.hostname:
        logger.error(f"[proxy] Invalid PROXY URL for Playwright: {proxy_url!r}")
        return None

    server = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        server += f":{parsed.port}"

    cfg: dict = {"server": server}

    if parsed.username:
        cfg["username"] = unquote(parsed.username)
    if parsed.password:
        cfg["password"] = unquote(parsed.password)

    logger.info(f"[proxy] Playwright proxy config: {cfg}")
    return cfg


def should_use_browser_proxy() -> bool:
    """Управляет применением прокси в Playwright.

    По умолчанию прокси для браузера выключен, чтобы сетевые настройки
    LLM (через PROXY) не распространялись на взаимодействие с веб-сайтами.
    Включить можно, если явно установить BROWSER_PROXY в truthy-значение.
    """

    raw = os.getenv("BROWSER_PROXY", "false")

    if _is_disabled_value(raw):
        logger.info("[proxy] Browser proxy disabled (BROWSER_PROXY set to 'false').")
        return False

    enabled = _is_enabled_value(raw)
    logger.info(
        f"[proxy] Browser proxy {'enabled' if enabled else 'disabled'} via BROWSER_PROXY={raw!r}."
    )
    return enabled

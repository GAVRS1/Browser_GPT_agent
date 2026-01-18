import importlib.util
import os
from urllib.parse import ParseResult, quote, unquote, urlparse, urlunparse

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


def _ensure_supported_proxy_scheme(parsed: ParseResult) -> bool:
    scheme = (parsed.scheme or "").lower()
    if scheme.startswith("socks"):
        socks_support = importlib.util.find_spec("socksio")
        if socks_support is None:
            logger.error(
                "[proxy] SOCKS proxy requested but httpx[socks] is not installed; "
                "install extras or use an HTTP proxy."
            )
            return False
    return True


def _build_proxy_netloc(
    hostname: str | None,
    port: int | None,
    username: str | None,
    password: str | None,
) -> str:
    if not hostname:
        return ""

    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    if port:
        host = f"{host}:{port}"

    if username:
        user = quote(username, safe="")
        if password:
            pwd = quote(password, safe="")
            return f"{user}:{pwd}@{host}"
        return f"{user}@{host}"

    return host


def _sanitize_proxy_url(url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        logger.error(f"[proxy] Invalid PROXY URL: {url!r}")
        return None

    if not _ensure_supported_proxy_scheme(parsed):
        return None

    netloc = _build_proxy_netloc(
        parsed.hostname,
        parsed.port,
        parsed.username,
        parsed.password,
    )
    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _proxy_log_sanitized(parsed: ParseResult) -> str:
    netloc = _build_proxy_netloc(
        parsed.hostname,
        parsed.port,
        parsed.username,
        None,
    )
    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


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
        sanitized = _sanitize_proxy_url(raw)
        if not sanitized:
            return None
        log_proxy = _proxy_log_sanitized(urlparse(sanitized))
        logger.info(f"[proxy] Using PROXY as URL: {log_proxy}")
        return sanitized

    # Пытаемся распарсить сокращённую форму
    url = _build_url_from_short_notation(raw)
    if url:
        sanitized = _sanitize_proxy_url(url)
        if not sanitized:
            return None
        log_proxy = _proxy_log_sanitized(urlparse(sanitized))
        logger.info(f"[proxy] Normalized short PROXY to URL: {log_proxy}")
        return sanitized

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
    log_proxy = _proxy_log_sanitized(urlparse(proxy))
    logger.info(f"[proxy] HTTP(S) proxy env vars set to {log_proxy!r}.")


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

    cfg_log = cfg.copy()
    cfg_log.pop("password", None)
    logger.info(f"[proxy] Playwright proxy config: {cfg_log}")
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

"""Site-specific URLs and timeout defaults with env overrides."""
from __future__ import annotations

import os
from typing import Callable, TypeVar

from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T")


def _read_env(parser: Callable[[str], T], env_name: str, default: T) -> T:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return parser(raw)
    except (TypeError, ValueError):
        return default


# Common / generic
GOOGLE_SEARCH_URL_TEMPLATE: str = os.getenv(
    "GOOGLE_SEARCH_URL_TEMPLATE", "https://www.google.com/search?q={query}"
)
AGENT_CONFIRMATION_TIMEOUT: float = _read_env(float, "AGENT_CONFIRMATION_TIMEOUT", 60.0)

# hh.ru
HHRU_HOME_URL: str = os.getenv("HHRU_BASE_URL", "https://hh.ru/")

# Yandex Mail
YANDEX_MAIL_HOME_URL: str = os.getenv(
    "YANDEX_MAIL_BASE_URL", "https://mail.yandex.ru/"
)
YANDEX_MAIL_LOGIN_TIMEOUT_MS: int = _read_env(int, "YANDEX_MAIL_LOGIN_TIMEOUT_MS", 2000)
YANDEX_MAIL_LIST_VISIBILITY_TIMEOUT_MS: int = _read_env(
    int, "YANDEX_MAIL_LIST_VISIBILITY_TIMEOUT_MS", 1200
)
YANDEX_MAIL_CLICK_TIMEOUT_MS: int = _read_env(int, "YANDEX_MAIL_CLICK_TIMEOUT_MS", 2000)
YANDEX_MAIL_INNER_TEXT_TIMEOUT_MS: int = _read_env(
    int, "YANDEX_MAIL_INNER_TEXT_TIMEOUT_MS", 1500
)
YANDEX_MAIL_AFTER_CLICK_WAIT_MS: int = _read_env(
    int, "YANDEX_MAIL_AFTER_CLICK_WAIT_MS", 800
)
YANDEX_MAIL_PREVIEW_COLLECTION_WAIT_MS: int = _read_env(
    int, "YANDEX_MAIL_PREVIEW_COLLECTION_WAIT_MS", 1500
)

# Yandex Lavka
YANDEX_LAVKA_HOME_URL: str = os.getenv(
    "YANDEX_LAVKA_BASE_URL", "https://lavka.yandex.ru/"
)
YANDEX_LAVKA_POST_NAV_WAIT_MS: int = _read_env(
    int, "YANDEX_LAVKA_POST_NAV_WAIT_MS", 1200
)
YANDEX_LAVKA_LOGIN_CHECK_TIMEOUT_MS: int = _read_env(
    int, "YANDEX_LAVKA_LOGIN_CHECK_TIMEOUT_MS", 1500
)
YANDEX_LAVKA_SEARCH_CLICK_TIMEOUT_MS: int = _read_env(
    int, "YANDEX_LAVKA_SEARCH_CLICK_TIMEOUT_MS", 2000
)

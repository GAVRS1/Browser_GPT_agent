"""Configurable defaults for navigation and confirmations."""
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

# Время ожидания подтверждения рискованных действий.
# Не привязываемся к сайтам, чтобы архитектура оставалась универсальной.
AGENT_CONFIRMATION_TIMEOUT: float = _read_env(float, "AGENT_CONFIRMATION_TIMEOUT", 60.0)

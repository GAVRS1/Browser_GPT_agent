"""Configurable defaults for navigation and confirmations."""
from __future__ import annotations

import os
from typing import Callable, Optional, TypeVar

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
BROWSER_START_URL: Optional[str] = os.getenv("BROWSER_START_URL")
SEARCH_URL_TEMPLATE: Optional[str] = os.getenv("SEARCH_URL_TEMPLATE")
SEARCH_URL_MODE: str = os.getenv("SEARCH_URL_MODE", "auto").strip().lower()

# Время ожидания подтверждения рискованных действий.
# Не привязываемся к сайтам, чтобы архитектура оставалась универсальной.
AGENT_CONFIRMATION_TIMEOUT: float = _read_env(float, "AGENT_CONFIRMATION_TIMEOUT", 60.0)

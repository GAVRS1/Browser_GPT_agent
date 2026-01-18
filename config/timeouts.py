from __future__ import annotations

import os
from typing import Optional


def _timeout_from_env(env_var: str, default: int, *, floor: Optional[int] = None) -> int:
    """Возвращает числовой таймаут из env с fallback."""

    try:
        value = int(os.getenv(env_var, default))
    except ValueError:
        return default

    if floor is not None:
        return max(value, floor)
    return value


# ---------------------------------------------------------------------------
# Playwright defaults
# ---------------------------------------------------------------------------


def default_action_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_DEFAULT_TIMEOUT_MS", 3000, floor=500)


def navigation_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_NAVIGATION_TIMEOUT_MS", 12000, floor=5000)


def navigation_retry_min_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_NAVIGATION_RETRY_MIN_TIMEOUT_MS", 15000, floor=5000)


# ---------------------------------------------------------------------------
# DOM stability checks
# ---------------------------------------------------------------------------


def dom_stable_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_DOM_STABLE_TIMEOUT_MS", 8000, floor=2000)


def dom_stable_interval_ms() -> int:
    return _timeout_from_env("BROWSER_DOM_STABLE_INTERVAL_MS", 500, floor=200)


def dom_stable_checks() -> int:
    return _timeout_from_env("BROWSER_DOM_STABLE_CHECKS", 3, floor=2)


# ---------------------------------------------------------------------------
# Browser tool actions
# ---------------------------------------------------------------------------


def click_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_CLICK_TIMEOUT_MS", 2000, floor=200)


def card_click_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_CARD_CLICK_TIMEOUT_MS", 4000, floor=500)


def locator_attach_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_LOCATOR_ATTACH_TIMEOUT_MS", 2000, floor=200)


def locator_scroll_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_LOCATOR_SCROLL_TIMEOUT_MS", 2000, floor=200)


def locator_click_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_LOCATOR_CLICK_TIMEOUT_MS", 2000, floor=200)


def locator_fill_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_LOCATOR_FILL_TIMEOUT_MS", 3000, floor=200)


def text_read_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_TEXT_READ_TIMEOUT_MS", 1000, floor=200)


def text_snippet_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_TEXT_SNIPPET_TIMEOUT_MS", 800, floor=200)


def visibility_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_VISIBILITY_TIMEOUT_MS", 1500, floor=200)


def input_value_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_INPUT_VALUE_TIMEOUT_MS", 800, floor=200)


def button_enabled_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_BUTTON_ENABLED_TIMEOUT_MS", 800, floor=200)


# ---------------------------------------------------------------------------
# Generic tool defaults (agent/tools_init.py)
# ---------------------------------------------------------------------------


def tool_action_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_TOOL_ACTION_TIMEOUT_MS", 5000, floor=500)


def tool_dom_stable_timeout_ms() -> int:
    return _timeout_from_env("BROWSER_TOOL_DOM_STABLE_TIMEOUT_MS", 5000, floor=500)

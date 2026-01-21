"""Shared configuration constants with optional environment overrides."""

from __future__ import annotations

import os
from typing import Optional


def _int_from_env(
    name: str,
    default: int,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default

    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


# Сколько записей истории брать в контекст модели.
HISTORY_CONTEXT_LIMIT = _int_from_env("AGENT_HISTORY_CONTEXT_LIMIT", 12, minimum=1)
# Шаг округления координат клика для анти-зацикливания.
CLICK_COORD_ROUNDING = _int_from_env("BROWSER_CLICK_COORD_ROUNDING", 5, minimum=1)
# Окно и порог для детектирования повторяющихся действий агента.
REPEAT_PATTERN_WINDOW = _int_from_env("AGENT_REPEAT_PATTERN_WINDOW", 8, minimum=2)
REPEAT_PATTERN_THRESHOLD = _int_from_env("AGENT_REPEAT_PATTERN_THRESHOLD", 4, minimum=1)

# Параметры для определения достижения цели по ключевым словам.
GOAL_KEYWORD_SHORT_MAX = _int_from_env("AGENT_GOAL_KEYWORD_SHORT_MAX", 2, minimum=1)
GOAL_KEYWORD_HITS_SHORT = _int_from_env("AGENT_GOAL_KEYWORD_HITS_SHORT", 1, minimum=1)
GOAL_KEYWORD_HITS_LONG = _int_from_env("AGENT_GOAL_KEYWORD_HITS_LONG", 2, minimum=1)

# Минимальные размеры карточки, чтобы считать её крупной.
CARD_MIN_WIDTH = _int_from_env("BROWSER_CARD_MIN_WIDTH", 120, minimum=1)
CARD_MIN_HEIGHT = _int_from_env("BROWSER_CARD_MIN_HEIGHT", 140, minimum=1)
CARD_MIN_AREA = _int_from_env("BROWSER_CARD_MIN_AREA", 15000, minimum=1)

# Лимиты обхода карточек при чтении, скролле и клике.
CARD_MAX_CHECK_READ = _int_from_env("BROWSER_CARD_MAX_CHECK_READ", 40, minimum=1)
CARD_MAX_CHECK_SCROLL = _int_from_env("BROWSER_CARD_MAX_CHECK_SCROLL", 40, minimum=1)
CARD_MAX_CHECK_CLICK = _int_from_env("BROWSER_CARD_MAX_CHECK_CLICK", 50, minimum=1)
# Сколько карточек отдавать в read_view.
CARD_OUTPUT_LIMIT = _int_from_env("BROWSER_CARD_OUTPUT_LIMIT", 10, minimum=1)
# Сколько крупных карточек считать «сеткой», чтобы запретить скролл.
CARD_BIG_COUNT_THRESHOLD = _int_from_env("BROWSER_CARD_BIG_COUNT_THRESHOLD", 4, minimum=1)

# Базовый шаг скролла.
DEFAULT_SCROLL_AMOUNT = _int_from_env("BROWSER_SCROLL_AMOUNT", 800, minimum=100)

# Веса для скоринга карточек.
CARD_SCORE_PRICE_BONUS = _int_from_env("BROWSER_CARD_SCORE_PRICE_BONUS", 50000, minimum=0)
CARD_SCORE_TITLE_BONUS = _int_from_env("BROWSER_CARD_SCORE_TITLE_BONUS", 15000, minimum=0)
CARD_SCORE_ADD_BUTTON_BONUS = _int_from_env("BROWSER_CARD_SCORE_ADD_BUTTON_BONUS", 60000, minimum=0)
CARD_SCORE_TOKEN_BONUS = _int_from_env("BROWSER_CARD_SCORE_TOKEN_BONUS", 70000, minimum=0)
CARD_SCORE_RECIPE_PENALTY = _int_from_env("BROWSER_CARD_SCORE_RECIPE_PENALTY", 40000, minimum=0)


def goal_keyword_hit_threshold(keyword_count: int) -> int:
    if keyword_count <= GOAL_KEYWORD_SHORT_MAX:
        return GOAL_KEYWORD_HITS_SHORT
    return GOAL_KEYWORD_HITS_LONG


def is_large_card(width: float, height: float, area: float) -> bool:
    return width >= CARD_MIN_WIDTH and height >= CARD_MIN_HEIGHT and area >= CARD_MIN_AREA

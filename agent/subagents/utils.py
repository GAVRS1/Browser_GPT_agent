from __future__ import annotations

import re
from typing import Iterable


def matches_domain(goal: str, domains: Iterable[str]) -> bool:
    """Heuristic domain matcher for goals with URLs or host mentions.

    Tries to determine whether the goal text references one of the specified
    domains without relying on exact paths. Works both for plain mentions like
    "mail.yandex.ru" and for full URLs with long paths.
    """

    lowered = goal.lower()
    tokens = re.split(r"[^a-z0-9.-]+", lowered)

    for domain in domains:
        normalized = domain.lower().lstrip(".")
        if not normalized:
            continue

        if normalized in lowered:
            return True

        if any(token.endswith(normalized) for token in tokens if token):
            return True

    return False


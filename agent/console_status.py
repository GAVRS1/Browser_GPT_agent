from __future__ import annotations

import re
from typing import Optional

_STEP_PATTERN = re.compile(r"^\s*\d+[\).]")


def _count_numbered_steps(text: str) -> int:
    if not text:
        return 0
    return sum(1 for line in text.splitlines() if _STEP_PATTERN.match(line))


def format_status(action: str, result: str, detail: Optional[str] = None) -> str:
    base = f"{action} → {result}"
    if detail:
        base += f" ({detail})"
    return base


def plan_status(plan_text: str) -> str:
    if not plan_text:
        return format_status("План", "не получен")
    steps = _count_numbered_steps(plan_text)
    detail = f"{steps} шагов" if steps else None
    return format_status("План", "готов", detail)


def step_status(step_index: int, result: str = "ответ получен", detail: Optional[str] = None) -> str:
    return format_status(f"Шаг {step_index + 1}", result, detail)


def tool_status(tool_name: str, success: bool) -> str:
    return format_status(f"Инструмент {tool_name}", "ok" if success else "fail")


def action_status(action: str, result: str, detail: Optional[str] = None) -> str:
    return format_status(action, result, detail)

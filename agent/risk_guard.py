from __future__ import annotations

import re
from typing import List


RISKY_ACTION_KEYWORDS = [
    # Payments / checkout
    "оплатить",
    "оплати",
    "оплата",
    "payment",
    "pay",
    "checkout",
    "buy",
    "purchase",
    "order",
    "place order",
    "submit order",
    "confirm",
    "подтвердить",
    "оформить заказ",
    "оформи заказ",
    "заказать",
    "закажи",
    "добавь в корзину",
    "положи в корзину",
    "положить в корзину",
    # Deletion / destructive actions
    "удали",
    "удалить",
    "удалите",
    "удаляй",
    "удаление",
    "delete",
    "remove",
    "wipe",
    "purge",
    "очисти",
    "очистить",
    "reset",
    # Submissions / sending
    "submit",
    "send",
    "отправить",
    "отправь",
    "send application",
    "submit application",
    "apply",
    "откликнись",
    "откликнуться",
    "откликнись на вакансию",
    "откликнуться на вакансию",
    "отправь отклик",
    "отправить отклик",
    "отправь резюме",
    "отправить резюме",
    "отправь отклик на вакансию",
    "отправь заявку",
    "отправить заявку",
]


def _normalize(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"\s+", " ", lowered).strip()


def risky_keyword_matches(text: str) -> List[str]:
    normalized = _normalize(text)
    if not normalized:
        return []
    matches: List[str] = []
    for keyword in RISKY_ACTION_KEYWORDS:
        if keyword in normalized:
            matches.append(keyword)
    return matches


def is_risky_text(text: str) -> bool:
    return bool(risky_keyword_matches(text))


__all__ = ["RISKY_ACTION_KEYWORDS", "is_risky_text", "risky_keyword_matches"]

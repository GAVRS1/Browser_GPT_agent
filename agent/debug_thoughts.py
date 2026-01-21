# agent/debug_thoughts.py
import os
from loguru import logger

DEBUG_THOUGHTS = os.getenv("AGENT_DEBUG_THOUGHTS", "0") == "1"


def log_thought(prefix: str, text: str) -> None:
    """
    Единая функция для вывода мыслей ИИ в лог и в консоль.
    prefix — короткое имя: 'agent' или любое название под-агента.
    """
    if not text:
        return

    if DEBUG_THOUGHTS:
        logger.info(f"[{prefix}] thought: {text}")
        print(f"\n🤖 {prefix} думает:\n{text.strip()}\n")
    else:
        logger.info(text)

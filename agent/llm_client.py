import os
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from config.proxy import apply_requests_proxy

# Загружаем .env
load_dotenv()

_client: Optional[OpenAI] = None
_disabled_reason: Optional[str] = None


def disable_client(reason: str):
    """Globally disable LLM usage after a fatal error (e.g., auth failure)."""

    global _client, _disabled_reason
    _client = None
    _disabled_reason = reason
    logger.error(f"[llm_client] LLM client disabled: {reason}")


def get_client() -> Optional[OpenAI]:
    """
    Возвращает OpenAI-клиента, создаёт его один раз.
    Учитывает прокси через apply_requests_proxy().
    Если клиент недоступен — возвращает None (агент перейдёт в fallback).
    """

    global _client, _disabled_reason

    if _disabled_reason:
        logger.error(f"[llm_client] LLM client unavailable: {_disabled_reason}")
        return None

    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        logger.error("[llm_client] OPENAI_API_KEY is not set!")
        return None

    # Настраиваем прокси для HTTP
    apply_requests_proxy()

    try:
        _client = OpenAI(api_key=api_key)
        logger.info("[llm_client] OpenAI client initialized.")
        return _client

    except Exception as exc:
        logger.error(f"[llm_client] Failed to initialize OpenAI client: {exc}")
        _client = None
        return None

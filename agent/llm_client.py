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
_provider: Optional[str] = None


def disable_client(reason: str):
    """Globally disable LLM usage after a fatal error (e.g., auth failure)."""

    global _client, _disabled_reason, _provider
    _client = None
    _provider = None
    _disabled_reason = reason
    logger.error(f"[llm_client] LLM client disabled: {reason}")


def get_llm_provider() -> str:
    raw = os.getenv("LLM_PROVIDER", "gpt").strip().lower()
    return raw or "gpt"


def _resolve_model_id(client: OpenAI, provider: str) -> str:
    if provider in ("gpt", "openai"):
        override = os.getenv("OPENAI_MODEL", "").strip()
        if override:
            return override

        model_list = getattr(getattr(client, "models", None), "list", lambda: None)()
        if model_list and getattr(model_list, "data", None):
            return model_list.data[0].id

        return "gpt-4o-mini"

    if provider == "glm":
        return os.getenv("GLM_MODEL", "glm-4.6").strip() or "glm-4.6"

    return "gpt-4o-mini"


def get_model_id(client: OpenAI) -> str:
    provider = _provider or get_llm_provider()
    return _resolve_model_id(client, provider)


def get_client() -> Optional[OpenAI]:
    """
    Возвращает OpenAI-клиента, создаёт его один раз.
    Учитывает прокси через apply_requests_proxy().
    Если клиент недоступен — возвращает None (агент перейдёт в fallback).
    """

    global _client, _disabled_reason, _provider

    if _disabled_reason:
        logger.error(f"[llm_client] LLM client unavailable: {_disabled_reason}")
        return None

    if _client is not None:
        return _client

    provider = get_llm_provider()
    api_key = ""
    base_url = None

    if provider in ("gpt", "openai"):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        if not api_key:
            logger.error("[llm_client] OPENAI_API_KEY is not set!")
            return None
    elif provider == "glm":
        api_key = os.getenv("GLM_API_KEY", "").strip()
        base_url = os.getenv("GLM_BASE_URL", "https://api.z.ai/v1").strip()
        if not api_key:
            logger.error("[llm_client] GLM_API_KEY is not set!")
            return None
    else:
        logger.error(f"[llm_client] Unknown LLM_PROVIDER: {provider!r}")
        return None

    # Настраиваем прокси для HTTP
    apply_requests_proxy()

    try:
        _client = OpenAI(api_key=api_key, base_url=base_url)
        _provider = provider
        logger.info("[llm_client] OpenAI client initialized.")
        return _client

    except Exception as exc:
        logger.error(f"[llm_client] Failed to initialize OpenAI client: {exc}")
        _client = None
        return None

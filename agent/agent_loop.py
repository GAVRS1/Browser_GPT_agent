import os
import sys
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from agent.browser_tools import ToolResult, format_tool_observation
from agent.llm_client import get_client, get_model_id
from agent.tool_client import ToolClient, get_shared_tool_client
from agent.risk_guard import is_risky_text
from agent.subagents import pick_subagent
from browser.context import shutdown_browser
from agent.debug_thoughts import DEBUG_THOUGHTS, log_thought
from agent.console_status import (
    action_status,
    plan_status,
    step_status,
    tool_status,
)
from config.prompt_templates import (
    BROWSER_ACTION_RULES,
    BROWSER_CONTEXT,
    FINAL_REPORT,
    RENTAL_FLOWS,
    SAFETY_LIMITS,
    SCREENSHOT_GUIDE,
    SESSION_RULES,
    compose_prompt,
)
from config.proxy import get_proxy_url
from config.sites import AGENT_CONFIRMATION_TIMEOUT, SEARCH_URL_TEMPLATE


@dataclass
class AttemptRecord:
    goal: str
    timestamp: float
    status: str
    details: str = ""
    error: Optional[str] = None


@dataclass
class ScreenshotCache:
    """Хранит ссылку на последний сделанный скриншот и подсказывает LLM."""

    last_link: Optional[str] = None
    _needs_reminder: bool = False

    def remember(self, observation: str) -> None:
        self.last_link = observation
        self._needs_reminder = True

    def reminder_message(self) -> Optional[Dict[str, str]]:
        if not (self.last_link and self._needs_reminder):
            return None

        self._needs_reminder = False
        return {
            "role": "system",
            "content": (
                "Последний скриншот уже есть: "
                f"{self.last_link}. Если нужно смотреть изображение, "
                "используй эту же ссылку вместо нового вызова take_screenshot."
            ),
        }


_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "busy": False,
    "last_goal": None,
    "history": [],
    "awaiting_confirmation": False,
    "confirmation_response": None,
    "last_error": None,
    "last_report": None,  # краткий отчёт о последней задаче
}

_console_confirmation_enabled = False
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AGENT_STATE_DIR = os.path.join(_ROOT_DIR, "agent_state")
_HISTORY_LOG_PATH = os.path.join(_AGENT_STATE_DIR, "history.log")
_HISTORY_CONTEXT_LIMIT = 5

# ============================================================================
# State helpers
# ============================================================================

def agent_is_busy() -> bool:
    with _state_lock:
        return bool(_state.get("busy", False))


def _set_busy(value: bool) -> None:
    with _state_lock:
        _state["busy"] = value


def _push_history(record: AttemptRecord) -> None:
    with _state_lock:
        history: List[AttemptRecord] = _state.setdefault("history", [])
        history.append(record)


def _set_status(**kwargs: Any) -> None:
    with _state_lock:
        _state.update(kwargs)


def _format_timestamp(epoch_seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_seconds))


def _summarize_record(record: AttemptRecord, limit: int = 220) -> str:
    summary = record.details.strip() if record.details else ""
    if not summary and record.error:
        summary = record.error.strip()
    if len(summary) > limit:
        summary = summary[: limit - 3].rstrip() + "..."
    return summary


def _append_history_log(record: AttemptRecord) -> None:
    if record.status == "started":
        return

    os.makedirs(_AGENT_STATE_DIR, exist_ok=True)
    entry = {
        "goal": record.goal,
        "timestamp": _format_timestamp(record.timestamp),
        "status": record.status,
        "summary": _summarize_record(record),
    }
    with open(_HISTORY_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_recent_history(limit: int = _HISTORY_CONTEXT_LIMIT) -> List[Dict[str, str]]:
    if not os.path.exists(_HISTORY_LOG_PATH):
        return []

    try:
        with open(_HISTORY_LOG_PATH, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError:
        return []

    recent: List[Dict[str, str]] = []
    for line in lines[-limit:]:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            recent.append(parsed)
    return recent


# ============================================================================
# Confirmation helpers (консольный режим)
# ============================================================================

def enable_console_confirmation() -> None:
    """
    Включает подтверждения действий через консоль.
    """
    global _console_confirmation_enabled
    _console_confirmation_enabled = True


def _await_console_confirmation(timeout: Optional[float]) -> bool:
    """
    Блокирующе ждём ответа пользователя в консоли.
    Если timeout=None — ждём бесконечно.
    """

    if not sys.stdin.isatty():
        logger.warning("[agent] stdin is not a TTY; rejecting risky action.")
        _set_status(awaiting_confirmation=False, confirmation_response=False)
        return False

    print("Требуется подтверждение для выполнения потенциально рискованного действия.")
    print("Введите 'y' или 'д' чтобы подтвердить, 'n' или 'н' чтобы отменить.")

    start = time.time()
    while True:
        if timeout is not None and (time.time() - start) > timeout:
            logger.warning("[agent] Console confirmation timed out.")
            _set_status(awaiting_confirmation=False, confirmation_response=False)
            return False

        try:
            user_input = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            logger.warning("[agent] Console confirmation interrupted")
            _set_status(awaiting_confirmation=False, confirmation_response=False)
            return False

        if user_input in {"y", "yes", "д", "да"}:
            _set_status(awaiting_confirmation=False, confirmation_response=True)
            return True
        if user_input in {"n", "no", "н", "нет"}:
            _set_status(awaiting_confirmation=False, confirmation_response=False)
            return False

        print("Некорректный ввод. Пожалуйста, введите 'y'/'д' или 'n'/'н'.")


def _await_confirmation(timeout: float = AGENT_CONFIRMATION_TIMEOUT) -> bool:
    """
    Обёртка над _await_console_confirmation.
    """
    _set_status(awaiting_confirmation=True, confirmation_response=None)

    logger.info("[agent] Awaiting console confirmation for potentially risky action...")

    if not _console_confirmation_enabled:
        logger.warning("[agent] Console confirmation disabled; rejecting risky action.")
        _set_status(awaiting_confirmation=False, confirmation_response=False)
        return False

    approved = _await_console_confirmation(None)
    return approved


# ============================================================================
# Status reporting
# ============================================================================

def get_agent_status() -> Dict[str, Any]:
    with _state_lock:
        status = dict(_state)
        status["history"] = list(status.get("history", []))
        return status


# ============================================================================
# Risky goals detection
# ============================================================================

def _is_risky_goal(goal: str) -> bool:
    """
    Определяет, требует ли цель пользователя обязательного подтверждения действий.

    Сюда попадают:
    - удаление / очистка чего-либо;
    - оформление/оплата заказов или аренды;
    - отклики на вакансии / отправка резюме, заявок.
    """
    return is_risky_text(goal)


# ============================================================================
# Core logic
# ============================================================================

def _run_llm_planning(goal: str) -> str:
    """
    Генерирует короткий план действий с помощью настроенной LLM.
    """

    client = get_client()
    if client is None:
        return ""

    model_id = get_model_id(client)

    system_text = (
        "You are an autonomous browser automation agent.\n"
        "\n"
        "Your job:\n"
        "- Receive a high-level goal from the user.\n"
        "- Think step-by-step.\n"
        "- Use browser tools to explore and transform web pages.\n"
        "- Adapt when actions fail (elements moved, labels changed, etc.).\n"
        "- Never rely on hardcoded DOM structure or fixed selectors.\n"
        "\n"
        "Available tools (conceptually):\n"
        "- dom_snapshot(): read the current page (title, url, visible text, buttons, links, inputs).\n"
        "- click(selector): click an element chosen by you via CSS selector.\n"
        "- click_by_text(text): find a clickable element by its visible text and click it.\n"
        "- type_text(selector, text): type into an input or textarea.\n"
        "- wait_for_dom_stable(): wait until the page finishes loading/updating.\n"
        "- take_screenshot(full_page: bool): capture a screenshot of the current page when\n"
        "  DOM summary is not enough to understand layout or visual state.\n"
        "\n"
        "Guidelines:\n"
        "- First, open or focus the relevant website in the browser.\n"
        "- Use dom_snapshot to understand what is on the screen.\n"
        "- Decide what to do next based on the snapshot, then call a browser tool.\n"
        "- After each important action, refresh your understanding (dom_snapshot again if needed).\n"
        "- If an action fails, try an alternative approach (different text, different selector, scroll, etc.).\n"
        "- Do NOT assume element ids/classes/paths — infer selectors dynamically from the page contents.\n"
        "- Avoid sending large raw HTML into the context; work with summarized snapshots instead.\n"
        "- Call take_screenshot only when DOM/text is confusing or you suspect a visual problem\n"
        "  (e.g. cards visible but not clickable, unexpected layout).\n"
        "\n"
        "You must produce a short, numbered action plan:\n"
        "- Each step describes what to inspect, click, or type on the page.\n"
        "- The plan is not a rigid script: you are allowed to adapt if the page differs.\n"
    )

    messages = [
        {
            "role": "system",
            "content": system_text,
        },
        {"role": "user", "content": goal},
    ]

    def _request_plan(active_client) -> str:
        response = active_client.chat.completions.create(
            model=get_model_id(active_client),
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content if response.choices else ""

    try:
        content = _request_plan(client)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[agent] Failed to query LLM: {exc}")
        if not get_proxy_url():
            return ""
        logger.warning("[agent] Retrying LLM request without proxy.")
        fallback_client = get_client(force_no_proxy=True)
        if fallback_client is None:
            return ""
        try:
            content = _request_plan(fallback_client)
        except Exception as retry_exc:  # noqa: BLE001
            logger.error(f"[agent] LLM retry without proxy failed: {retry_exc}")
            return ""

    logger.info(f"[agent] LLM plan: {content}")

    if content:
        log_thought("agent-plan", content)
        if DEBUG_THOUGHTS:
            print(content.strip())
            print("-------------------\n")
        else:
            print(action_status("План", "готов"))

    return content or ""


def _parse_needs_login(observation: str) -> Dict[str, Any]:
    try:
        payload = json.loads(observation)
    except json.JSONDecodeError:
        return {"needs_login": False}
    if not isinstance(payload, dict):
        return {"needs_login": False}
    return {
        "needs_login": bool(payload.get("needs_login")),
        "login_indicators": payload.get("login_indicators", []),
    }


def _parse_needs_input(observation: str) -> Dict[str, Any]:
    try:
        payload = json.loads(observation)
    except json.JSONDecodeError:
        return {"needs_input": False}
    if not isinstance(payload, dict):
        return {"needs_input": False}
    return {
        "needs_input": bool(payload.get("needs_input")),
        "manual_input_indicators": payload.get("manual_input_indicators", []),
    }


def _safe_navigation(
    tool_client: ToolClient,
    url: Optional[str],
) -> tuple[Optional[ToolResult], Optional[str]]:
    if not url:
        fallback_hint = ""
        if not SEARCH_URL_TEMPLATE:
            fallback_hint = " Задайте SEARCH_URL_TEMPLATE в .env."
        return (
            None,
            "Стартовый URL не задан. Пожалуйста, укажите стартовую страницу в запросе."
            + fallback_hint,
        )
    call_result = tool_client.call_tool("open_url", {"url": url})
    return ToolResult("open_url", call_result.success, call_result.observation), None


def _autonomous_browse(
    goal: str,
    plan_text: str,
    prev_context: Optional[str] = None,
) -> tuple[str, str]:
    """Запускает универсальный цикл работы с инструментами браузера."""

    client = get_client()
    if client is None:
        return "failed", "LLM недоступен — не могу управлять браузером"

    tool_client = get_shared_tool_client()
    tools_for_client = tool_client.openai_tools()
    observation_result = tool_client.call_tool("read_view", {})
    observation = observation_result.observation
    login_state = _parse_needs_login(observation)
    if login_state.get("needs_login"):
        indicators = ", ".join(login_state.get("login_indicators", [])) or "unknown"
        return (
            "needs_input",
            "Обнаружена страница входа (признаки: "
            f"{indicators}). Пожалуйста, войдите вручную в браузере, "
            "затем повторно запустите задачу — агент продолжит с текущей сессией.",
        )
    manual_state = _parse_needs_input(observation)
    if manual_state.get("needs_input"):
        indicators = ", ".join(manual_state.get("manual_input_indicators", [])) or "unknown"
        return (
            "needs_input",
            "Обнаружен запрос на CAPTCHA/2FA/оплату (признаки: "
            f"{indicators}). Пожалуйста, выполните требуемое действие вручную "
            "в браузере, затем повторно запустите задачу — агент продолжит "
            "с текущей сессией.",
        )
    screenshot_cache = ScreenshotCache()
    actions: List[str] = []

    # АНТИ-ЗАЦИКЛИВАНИЕ
    recent_signatures: List[str] = []  # история последних действий (имя + аргументы)
    no_progress_steps = 0              # шаги подряд без изменения наблюдения
    last_observation = observation     # последнее observation, чтобы сравнивать
    waited_for_dom = False
    no_tool_calls = 0

    if DEBUG_THOUGHTS:
        print("\n=== Старт автономного режима ===")
        print(f"Цель пользователя: {goal}")
        if plan_text:
            print("\nКраткий план (из планировщика):")
            print(plan_text.strip())
        print("\nТекущее краткое наблюдение за страницей:")
        print(observation)
        print("=================================\n")
    else:
        print(action_status("Автономный режим", "старт"))

    system_prompt = compose_prompt(
        BROWSER_CONTEXT,
        SESSION_RULES,
        "Общие правила:\n" + BROWSER_ACTION_RULES,
        "Информация о карточках и слотах:\n"
        "- В наблюдении может быть поле 'product_cards' — список крупных карточек предложений.\n"
        "- Для каждой карточки там есть как минимум 'text'; используй его, чтобы выбрать подходящий слот/услугу.\n"
        "- Если пользователь просит конкретное время, стоимость или параметры, опирайся на содержимое этих карточек.",
        RENTAL_FLOWS,
        SAFETY_LIMITS,
        SCREENSHOT_GUIDE,
        FINAL_REPORT,
    )

    user_parts: List[str] = []
    if prev_context:
        user_parts.append("Контекст предыдущих действий агента в браузере:\n" + prev_context)
    user_parts.append(f"Текущая цель пользователя: {goal}")
    user_parts.append(f"Твой внутренний план: {plan_text or '—'}")
    user_parts.append(f"Текущая страница в браузере (краткое наблюдение): {observation}")
    user_content = "\n\n".join(user_parts)

    base_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    pending_messages: List[Dict[str, Any]] = []
    response_id: Optional[str] = None
    model_id = get_model_id(client)

    def _request_response(
        *,
        extra_input: Optional[List[Dict[str, Any]]] = None,
    ):
        nonlocal response_id
        payload: Dict[str, Any] = {
            "model": model_id,
            "temperature": 0.1,
            "tools": tools_for_client,
        }

        if response_id:
            payload["previous_response_id"] = response_id

        # В Responses API input — это список “items”.
        # Мы всегда докидываем новые items через extra_input.
        if response_id is None:
            payload["input"] = base_messages + (extra_input or [])
        else:
            payload["input"] = extra_input or []

        response = client.responses.create(**payload)
        response_id = response.id
        return response

    def _extract_response_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        output = getattr(response, "output", None) or []
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, dict):
                item_type = item.get("type")
            if item_type == "message":
                content = getattr(item, "content", None) or item.get("content", [])
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            return str(block.get("text", "")).strip()
                    else:
                        if getattr(block, "type", None) == "text":
                            return str(getattr(block, "text", "")).strip()
        return ""

    def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
        output = getattr(response, "output", None) or []
        tool_calls: List[Dict[str, Any]] = []

        for item in output:
            # item может быть объектом или dict
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, dict):
                item_type = item.get("type")

            # Responses API чаще отдаёт function_call
            if item_type not in ("tool_call", "function_call"):
                continue

            # Достаём id
            # В Responses API id вызова функции часто называется call_id
            call_id = (
                getattr(item, "call_id", None)
                or (item.get("call_id") if isinstance(item, dict) else None)
                or getattr(item, "id", None)
                or (item.get("id") if isinstance(item, dict) else None)
                or getattr(item, "tool_call_id", None)
                or (item.get("tool_call_id") if isinstance(item, dict) else None)
            )

            # Достаём name/arguments (они могут лежать по-разному)
            name = getattr(item, "name", None) if not isinstance(item, dict) else item.get("name")
            arguments = getattr(item, "arguments", None) if not isinstance(item, dict) else item.get("arguments")

            # Иногда это вложено в item.function или item.tool_call
            fn = getattr(item, "function", None) if not isinstance(item, dict) else item.get("function")
            if fn:
                if not name:
                    name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
                if arguments is None:
                    arguments = getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
                if not call_id:
                    call_id = (
                        getattr(fn, "call_id", None)
                        or (fn.get("call_id") if isinstance(fn, dict) else None)
                        or getattr(fn, "id", None)
                        or (fn.get("id") if isinstance(fn, dict) else None)
                    )

            tool_call = getattr(item, "tool_call", None) if not isinstance(item, dict) else item.get("tool_call")
            if tool_call:
                if not name:
                    name = (
                        getattr(tool_call, "name", None)
                        if not isinstance(tool_call, dict)
                        else tool_call.get("name")
                    )
                if arguments is None:
                    arguments = (
                        getattr(tool_call, "arguments", None)
                        if not isinstance(tool_call, dict)
                        else tool_call.get("arguments")
                    )
                if not call_id:
                    call_id = (
                        getattr(tool_call, "call_id", None)
                        or (tool_call.get("call_id") if isinstance(tool_call, dict) else None)
                        or getattr(tool_call, "id", None)
                        or (tool_call.get("id") if isinstance(tool_call, dict) else None)
                    )

            tool_calls.append({"id": call_id, "name": name, "arguments": arguments})

        return tool_calls

    def _wait_for_dom(reason: str) -> None:
        nonlocal waited_for_dom
        wait_call = tool_client.call_tool("wait_for_dom_stable", {})
        wait_result = ToolResult("wait_for_dom_stable", wait_call.success, wait_call.observation)
        actions.append(f"{wait_result.name}: {'ok' if wait_result.success else 'fail'}")
        actions.append(format_tool_observation(wait_result))
        pending_messages.append(
            {
                "role": "system",
                "content": f"wait_for_dom_stable ({reason}): {wait_result.observation}",
            }
        )
        waited_for_dom = True

    # Лимит шагов, чтобы не крутиться бесконечно
    for step_idx in range(30):
        reminder = screenshot_cache.reminder_message()
        if reminder:
            pending_messages.append(reminder)

        response = _request_response(extra_input=pending_messages)
        pending_messages = []

        message_text = _extract_response_text(response)
        tool_calls = _extract_tool_calls(response)

        # Вариант A: печатаем мысли агента на каждом шаге
        if message_text:
            thought = message_text.strip()
            if thought:
                logger.info(f"[agent] LLM thought (step {step_idx}): {thought}")
                if DEBUG_THOUGHTS:
                    print("\n🤖 Мысли агента (шаг):")
                    print(thought)
                    print()
                else:
                    print(step_status(step_idx))

        if tool_calls:
            no_tool_calls = 0
            step_made_progress = False
            waited_for_dom = False
            tool_outputs: List[Dict[str, str]] = []
            skip_remaining_calls = False

            for call in tool_calls:
                if skip_remaining_calls:
                    tool_outputs.append(
                        {
                            "tool_call_id": call["id"],
                            "output": (
                                "skipped_tool_call: skipped due to loop guard; "
                                "choose a different action."
                            ),
                        }
                    )
                    continue

                # Подпись действия для детектора циклов
                sig = f"{call['name']}:{call['arguments']}"
                recent_signatures.append(sig)
                recent_signatures = recent_signatures[-6:]

                # Логирование использования инструмента
                logger.info(f"[agent] Using tool: {call['name']} args={call['arguments']}")

                # ВЫПОЛНЯЕМ инструмент до использования result
                args = json.loads(call["arguments"] or "{}")
                if call["name"] == "read_view" and not waited_for_dom:
                    _wait_for_dom("before read_view")
                if call["name"] == "open_url":
                    result, missing_url_message = _safe_navigation(
                        tool_client,
                        args.get("url"),
                    )
                    if missing_url_message:
                        return "needs_input", missing_url_message
                else:
                    call_result = tool_client.call_tool(call["name"], args)
                    result = ToolResult(
                        call["name"],
                        call_result.success,
                        call_result.observation,
                    )
                if isinstance(result.observation, str) and result.observation.startswith(
                    "needs_confirmation:"
                ):
                    approved = _await_confirmation()
                    actions.append(
                        f"confirmation: {'approved' if approved else 'denied'}"
                    )
                    if approved:
                        retry_args = dict(args)
                        retry_args["_confirmed"] = True
                        retry_result = tool_client.call_tool(call["name"], retry_args)
                        result = ToolResult(call["name"], retry_result.success, retry_result.observation)

                # Краткая строка результата
                short_line = f"{result.name}: {'ok' if result.success else 'fail'}"
                actions.append(short_line)

                formatted = format_tool_observation(result)
                actions.append(formatted)

                if result.name == "take_screenshot" and result.success:
                    screenshot_cache.remember(result.observation)
                    actions.append(f"last_screenshot_cached: {screenshot_cache.last_link}")
                if result.name == "open_url":
                    _wait_for_dom("after open_url")
                if result.name == "read_view":
                    login_state = _parse_needs_login(result.observation)
                    if login_state.get("needs_login"):
                        indicators = ", ".join(
                            login_state.get("login_indicators", [])
                        ) or "unknown"
                        return (
                            "needs_input",
                            "Обнаружена страница входа (признаки: "
                            f"{indicators}). Пожалуйста, войдите вручную в браузере, "
                            "затем повторно запустите задачу — агент продолжит "
                            "с текущей сессией.",
                        )
                    manual_state = _parse_needs_input(result.observation)
                    if manual_state.get("needs_input"):
                        indicators = ", ".join(
                            manual_state.get("manual_input_indicators", [])
                        ) or "unknown"
                        return (
                            "needs_input",
                            "Обнаружен запрос на CAPTCHA/2FA/оплату (признаки: "
                            f"{indicators}). Пожалуйста, выполните требуемое действие "
                            "вручную в браузере, затем повторно запустите задачу — агент "
                            "продолжит с текущей сессией.",
                        )

                if DEBUG_THOUGHTS:
                    print(f"🛠 {short_line}")
                    print(f"   Аргументы: {call['arguments']}")
                else:
                    print(tool_status(result.name, result.success))

                # Проверяем, изменилось ли наблюдение (DOM / состояние)
                if result.observation and result.observation != last_observation:
                    step_made_progress = True
                    last_observation = result.observation

                # ВСЕГДА отправляем ответ инструмента для этого tool_call_id
                tool_outputs.append(
                    {
                        "tool_call_id": call["id"],
                        "output": result.observation,
                    }
                )

                # --- детектор зацикливания по одинаковому инструменту ---
                if len(recent_signatures) >= 3 and len(set(recent_signatures[-3:])) == 1:
                    msg = (
                        "Агент три раза подряд выполнил одно и то же действие. "
                        "Скорее всего, он застрял и нужно сменить стратегию."
                    )
                    if DEBUG_THOUGHTS:
                        print("⚠ " + msg)

                    # Добавляем системную подсказку в историю, чтобы модель перестала
                    # повторять одно и то же действие и попробовала другой инструмент.
                    pending_messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Ты три шага подряд вызвал один и тот же инструмент "
                                "с одинаковыми аргументами. Перестань повторять его; "
                                "попробуй другой инструмент или другие аргументы. "
                                "Если ты уже ввёл текст в строку поиска, не вводи его снова, "
                                "а проанализируй текущую страницу, кликни по подходящему элементу, "
                                "выбери карточку товара, прокрути страницу и т.п."
                            ),
                        }
                    )

                    # Сбрасываем счётчики зацикливания и помечаем оставшиеся вызовы как пропущенные.
                    no_progress_steps = 0
                    recent_signatures.clear()
                    step_made_progress = False
                    skip_remaining_calls = True

            if step_made_progress:
                no_progress_steps = 0
            else:
                no_progress_steps += 1

            if no_progress_steps >= 3:
                msg = (
                    "Несколько действий подряд не привели к заметным изменениям на странице. "
                    "Ранее агент останавливался, чтобы не зациклиться, но теперь он "
                    "пытается сменить стратегию и продолжить работу."
                )
                if DEBUG_THOUGHTS:
                    print("⚠ " + msg)

                # Вместо немедленного завершения добавляем системную подсказку модели:
                pending_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Ты сделал несколько шагов подряд, которые не привели к изменениям "
                            "на странице. Не повторяй те же действия. Проанализируй текущее "
                            "наблюдение и попробуй другой инструмент или последовательность: "
                            "например, клик по другому элементу, прокрутку страницы, переход "
                            "к карточке товара и т.п."
                        ),
                    }
                )
                _wait_for_dom("no progress")
                refreshed_call = tool_client.call_tool("read_view", {})
                refreshed_view = refreshed_call.observation
                actions.append("read_view: ok")
                actions.append(f"read_view: {refreshed_view}")
                last_observation = refreshed_view
                pending_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Обновлённое наблюдение после ожидания динамического контента: "
                            f"{refreshed_view}"
                        ),
                    }
                )
                no_progress_steps = 0

            # --- ВАЖНО: вернуть tool outputs модели в формате Responses API ---
            function_output_items: List[Dict[str, Any]] = []
            for out in tool_outputs:
                function_output_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": out["tool_call_id"],  # ВАЖНО: call_id, не tool_call_id
                        "output": out["output"],
                    }
                )

            extra = function_output_items + (pending_messages or [])
            response = _request_response(extra_input=extra)
            pending_messages = []
            continue

        # Нет tool_calls — если инструментов ещё не было, пробуем перезапросить
        if not actions:
            no_tool_calls += 1
            if no_tool_calls < 2:
                pending_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Ты не вызвал ни одного инструмента, поэтому браузер не был "
                            "управляем. Обязательно используй инструменты (read_view, open_url, "
                            "click, type_text и т.д.) и продолжай выполнение."
                        ),
                    }
                )
                continue
            msg = (
                "LLM не запросил ни одного инструмента, поэтому браузер не был "
                "управляем. Проверь доступность инструментов и повтори задачу."
            )
            if DEBUG_THOUGHTS:
                print("⚠ " + msg)
            return "failed", msg

        # Нет tool_calls — считаем, что это финальный ответ
        final_text = message_text or ""
        summary = "\n".join(actions[-8:])
        report_parts = [
            "Автономный отчёт:",
            summary or "(действия не требовались)",
            "",
            final_text,
        ]
        full_report = "\n".join([part for part in report_parts if part])

        if DEBUG_THOUGHTS:
            print("\n✅ Финальный ответ агента:")
            print(final_text)
            print()
        else:
            print(action_status("Ответ", "получен"))

        return "completed", full_report

    msg = "Автономный режим завершился без финального ответа после 30 шагов"
    if DEBUG_THOUGHTS:
        print("⚠ " + msg)
    return "failed", msg

def run_agent(goal: str) -> None:
    """
    Основная точка входа: готовит браузер, строит план через LLM и
    далее либо делегирует задачу под-агенту, либо выполняет навигацию.
    """

    if agent_is_busy():
        raise RuntimeError("Agent is already busy")

    # берём контекст предыдущей задачи ДО обновления last_goal
    with _state_lock:
        previous_goal = _state.get("last_goal")
        previous_report = _state.get("last_report")

    recent_history = _load_recent_history()

    _set_busy(True)
    _set_status(last_goal=goal, last_error=None)

    record = AttemptRecord(goal=goal, timestamp=time.time(), status="started")
    _push_history(record)

    try:
        # Единый детектор рискованных целей (удаление, заказы, отклики)
        if _is_risky_goal(goal):
            logger.info("[agent] Goal considered risky, requesting confirmation.")
            approved = _await_confirmation()
            if not approved:
                record.status = "cancelled"
                record.error = "confirmation_denied"
                record.details = (
                    "Запрошено подтверждение для рискованной цели (удаление/оплата/заявки), "
                    "но пользователь не подтвердил выполнение."
                )
                _set_status(last_error="confirmation_denied", last_report=record.details)
                return

        plan_text = _run_llm_planning(goal)
        if not plan_text:
            record.status = "failed"
            record.error = "llm_unavailable"
            _set_status(last_error="llm_unavailable")
            logger.warning("[agent] LLM unavailable; aborting without executing tools")
            return

        record.details = plan_text

        prev_context_parts: List[str] = []
        if previous_goal and previous_report:
            trimmed_report = previous_report[-1500:]
            prev_context_parts.append(
                "Предыдущая задача пользователя: "
                f"{previous_goal}\n"
                "Краткий отчёт о том, что уже сделано в браузере:\n"
                f"{trimmed_report}"
            )
        if recent_history:
            history_lines = [
                "- [{timestamp}] {status}: {goal} — {summary}".format(
                    timestamp=item.get("timestamp", "unknown"),
                    status=item.get("status", "unknown"),
                    goal=item.get("goal", "unknown"),
                    summary=item.get("summary", "—"),
                )
                for item in recent_history
            ]
            prev_context_parts.append(
                "Журнал предыдущих действий (последние записи):\n"
                + "\n".join(history_lines)
            )
        prev_context = "\n\n".join(prev_context_parts) if prev_context_parts else None

        subagent = pick_subagent(goal)
        if subagent:
            logger.info(f"[agent] Delegating goal to subagent: {subagent.name}")
            sub_result = subagent.run(goal, plan_text)
            record.status = sub_result.status
            record.details = sub_result.details
            record.error = sub_result.error

            if sub_result.error:
                _set_status(last_error=sub_result.error)

            if sub_result.details:
                _set_status(last_report=sub_result.details)

            if sub_result.success or sub_result.status == "needs_input":
                return

        tool_status, tool_details = _autonomous_browse(goal, plan_text, prev_context=prev_context)
        record.status = tool_status
        record.details = tool_details

        _set_status(last_report=tool_details)

        # Больше не выполняем автоматический переход на страницу поиска.
        # Агент завершает задачу с тем статусом, который вернул автономный цикл.
        return

    except Exception as exc:  # noqa: BLE001
        logger.error(f"[agent] Fatal error while running goal '{goal}': {exc}")
        record.status = "failed"
        record.error = str(exc)
        _set_status(last_error=str(exc))
        try:
            shutdown_browser()
        except Exception:
            logger.exception("[agent] Failed to shutdown browser after error")
    finally:
        _set_busy(False)
        _print_report(record)
        _append_history_log(record)


def _print_report(record: AttemptRecord) -> None:
    """Печатает финальный отчёт в консоль."""

    header = f"\n=== Итог по задаче: {record.goal} ==="
    status_line = f"Статус: {record.status}"
    if record.error:
        status_line += f" (ошибка: {record.error})"

    if DEBUG_THOUGHTS:
        print("\n".join([header, status_line, "", record.details]))
        return

    summary = _summarize_record(record, limit=300)
    lines = [header, status_line]
    if summary:
        lines.extend(["", summary])
    print("\n".join(lines))


__all__ = [
    "run_agent",
    "agent_is_busy",
    "get_agent_status",
    "enable_console_confirmation",
]

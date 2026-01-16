import os
import sys
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from loguru import logger

from agent.browser_tools import BrowserToolbox, format_tool_observation
from agent.llm_client import get_client
from agent.subagents import pick_subagent
from browser.context import get_page, shutdown_browser
from agent.debug_thoughts import DEBUG_THOUGHTS, log_thought
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
from config.sites import AGENT_CONFIRMATION_TIMEOUT, GOOGLE_SEARCH_URL_TEMPLATE


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
    text = goal.lower()

    delete_keywords = [
        "удали",
        "удалить",
        "удаляй",
        "почисти",
        "очисти",
        "очистить",
        "wipe",
        "delete",
        "remove",
    ]

    order_keywords = [
        "закажи",
        "заказать",
        "закажи еду",
        "закажи продукт",
        "оформи заказ",
        "оформить заказ",
        "оформи",
        "оформить",
        "добавь в корзину",
        "положи в корзину",
        "положить в корзину",
        "оплати",
        "оплатить",
        "оплата",
        "checkout",
        "pay",
        "buy",
        "purchase",
    ]

    job_keywords = [
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
        "apply",
        "send application",
        "submit application",
    ]

    keywords = delete_keywords + order_keywords + job_keywords
    return any(k in text for k in keywords)


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

    model_list = getattr(getattr(client, "models", None), "list", lambda: None)()
    if model_list and getattr(model_list, "data", None):
        model_id = model_list.data[0].id
    else:
        model_id = "gpt-4o-mini"

    override = os.getenv("OPENAI_MODEL")
    if override:
        model_id = override

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

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.2,
        )
        content = response.choices[0].message.content if response.choices else ""

        logger.info(f"[agent] LLM plan: {content}")

        if content:
            log_thought("agent-plan", content)
            print(content.strip())
            print("-------------------\n")

        return content or ""
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[agent] Failed to query LLM: {exc}")
        return ""


def _safe_navigation(goal: str) -> str:
    """Открывает страницу поиска в браузере как безопасное действие по умолчанию."""

    page = get_page()
    try:
        page.bring_to_front()
    except Exception:
        pass

    search_url = GOOGLE_SEARCH_URL_TEMPLATE.format(query=quote_plus(goal))
    logger.info(f"[agent] Navigating to {search_url}")
    page.goto(search_url)
    return search_url


def _extract_needs_input(observation: str) -> Optional[str]:
    try:
        payload = json.loads(observation)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict) and payload.get("needs_input"):
        return payload.get("needs_input_reason") or "страница требует ручного действия"
    return None


def _autonomous_browse(
    goal: str,
    plan_text: str,
    prev_context: Optional[str] = None,
) -> tuple[str, str]:
    """Запускает универсальный цикл работы с инструментами браузера."""

    client = get_client()
    if client is None:
        return "failed", "LLM недоступен — не могу управлять браузером"

    toolbox = BrowserToolbox()
    mcp_tools = toolbox.mcp_tools()
    tools_for_client = toolbox.openai_tools()
    observation = toolbox.read_view()
    needs_input_reason = _extract_needs_input(observation)
    if needs_input_reason:
        return (
            "needs_input",
            f"Требуется ручное действие: {needs_input_reason}. "
            "Автономный цикл остановлен.",
        )
    screenshot_cache = ScreenshotCache()
    actions: List[str] = []

    # АНТИ-ЗАЦИКЛИВАНИЕ
    recent_signatures: List[str] = []  # история последних действий (имя + аргументы)
    no_progress_steps = 0              # шаги подряд без изменения наблюдения
    last_observation = observation     # последнее observation, чтобы сравнивать
    waited_for_dom = False

    if DEBUG_THOUGHTS:
        print("\n=== Старт автономного режима ===")
        print(f"Цель пользователя: {goal}")
        if plan_text:
            print("\nКраткий план (из планировщика):")
            print(plan_text.strip())
        print("\nТекущее краткое наблюдение за страницей:")
        print(observation)
        print("=================================\n")

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

    mcp_note = {
        "role": "system",
        "content": (
            "Инструменты описаны в формате MCP (name, description, input_schema). "
            f"Список: {json.dumps(mcp_tools, ensure_ascii=False)}"
        ),
    }

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        mcp_note,
        {"role": "user", "content": user_content},
    ]

    def _wait_for_dom(reason: str) -> None:
        nonlocal waited_for_dom
        wait_result = toolbox.execute("wait_for_dom_stable")
        actions.append(f"{wait_result.name}: {'ok' if wait_result.success else 'fail'}")
        actions.append(format_tool_observation(wait_result))
        messages.append(
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
            messages.append(reminder)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_for_client,
            temperature=0.1,
        )

        message = response.choices[0].message

        # Вариант A: печатаем мысли агента на каждом шаге
        if message.content:
            thought = (message.content or "").strip()
            if thought:
                logger.info(f"[agent] LLM thought (step {step_idx}): {thought}")
                if DEBUG_THOUGHTS:
                    print("\n🤖 Мысли агента (шаг):")
                    print(thought)
                    print()

        assistant_message: Dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_message["tool_calls"] = message.tool_calls
        messages.append(assistant_message)

        if message.tool_calls:
            step_made_progress = False
            waited_for_dom = False

            for call in message.tool_calls:
                # Подпись действия для детектора циклов
                sig = f"{call.function.name}:{call.function.arguments}"
                recent_signatures.append(sig)
                recent_signatures = recent_signatures[-6:]

                # Логирование использования инструмента
                logger.info(
                    f"[agent] Using tool: {call.function.name} args={call.function.arguments}"
                )

                # ВЫПОЛНЯЕМ инструмент до использования result
                args = json.loads(call.function.arguments or "{}")
                if call.function.name == "read_view" and not waited_for_dom:
                    _wait_for_dom("before read_view")
                result = toolbox.execute(call.function.name, args)

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
                if result.name == "read_view" and result.success:
                    needs_input_reason = _extract_needs_input(result.observation)
                    if needs_input_reason:
                        summary = "\n".join(actions[-8:])
                        report_parts = [
                            "Автономный отчёт:",
                            summary or "(действия не требовались)",
                            "",
                            (
                                "Требуется ручное действие: "
                                f"{needs_input_reason}. Автономный цикл остановлен."
                            ),
                        ]
                        full_report = "\n".join([part for part in report_parts if part])
                        return "needs_input", full_report

                if DEBUG_THOUGHTS:
                    print(f"🛠 {short_line}")
                    print(f"   Аргументы: {call.function.arguments}")

                # Проверяем, изменилось ли наблюдение (DOM / состояние)
                if result.observation and result.observation != last_observation:
                    step_made_progress = True
                    last_observation = result.observation

                # ВСЕГДА отправляем ответ инструмента для этого tool_call_id
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.observation,
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
                    messages.append(
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

                    # Сбрасываем счётчики зацикливания и выходим из цикла по tool_calls.
                    no_progress_steps = 0
                    recent_signatures.clear()
                    step_made_progress = False
                    break

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
                messages.append(
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
                refreshed_view = toolbox.read_view()
                actions.append("read_view: ok")
                actions.append(f"read_view: {refreshed_view}")
                last_observation = refreshed_view
                needs_input_reason = _extract_needs_input(refreshed_view)
                if needs_input_reason:
                    report_parts = [
                        "Автономный отчёт:",
                        "\n".join(actions[-8:]) or "(действия не требовались)",
                        "",
                        (
                            "Требуется ручное действие: "
                            f"{needs_input_reason}. Автономный цикл остановлен."
                        ),
                    ]
                    full_report = "\n".join([part for part in report_parts if part])
                    return "needs_input", full_report
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Обновлённое наблюдение после ожидания динамического контента: "
                            f"{refreshed_view}"
                        ),
                    }
                )
                no_progress_steps = 0

            continue

        # Нет tool_calls — считаем, что это финальный ответ
        final_text = message.content or ""
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

        prev_context = None
        if previous_goal and previous_report:
            trimmed_report = previous_report[-1500:]
            prev_context = (
                f"Предыдущая задача пользователя: {previous_goal}\n"
                f"Краткий отчёт о том, что уже сделано в браузере:\n{trimmed_report}"
            )

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


def _print_report(record: AttemptRecord) -> None:
    """Печатает финальный отчёт в консоль."""

    header = f"\n=== Итог по задаче: {record.goal} ==="
    status_line = f"Статус: {record.status}"
    if record.error:
        status_line += f" (ошибка: {record.error})"

    print("\n".join([header, status_line, "", record.details]))


__all__ = [
    "run_agent",
    "agent_is_busy",
    "get_agent_status",
    "enable_console_confirmation",
]

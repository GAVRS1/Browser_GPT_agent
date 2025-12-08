from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from playwright.sync_api import Page

from browser.context import get_page
from agent.debug_thoughts import log_thought
from agent.subagents import SubAgentResult
from agent.subagents.utils import matches_domain
from agent.browser_tools import BrowserToolbox, format_tool_observation
from agent.llm_client import get_client
from agent.tools_init import dom_snapshot
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
from config.sites import HHRU_HOME_URL


# ----------------------------------------------------------------------------
# Подагент hh.ru
# ----------------------------------------------------------------------------

class HhRuSubAgent:
    """Подагент, который работает с арендуемыми ресурсами через браузер."""

    name: str = "HeadHunter"

    _domains = ["hh.ru", "headhunter"]
    _keywords = ["hh.ru", "headhunter", "ваканси", "резюме", "hh"]

    def matches(self, goal: str) -> bool:
        lowered = goal.lower()
        return matches_domain(lowered, self._domains) or any(
            k in lowered for k in self._keywords
        )

    def run(self, goal: str, plan: str) -> SubAgentResult:
        log_thought(
            "hhru",
            f"Новая задача для hh.ru:\n{goal}\n\nПлан верхнего уровня:\n{plan}",
        )

        page = get_page()
        try:
            page.bring_to_front()
        except Exception:
            pass

        # Если страница не открыта — навигируемся на стартовый адрес
        if "hh.ru" not in (page.url or "") and "HeadHunter" not in (page.title() or ""):
            self._open_hh_home(page)
        else:
            logger.info("[hhru] Reusing already opened hh.ru tab")

        # Первичное краткое описание страницы
        try:
            initial_dom_summary = dom_snapshot()
        except Exception:
            initial_dom_summary = ""

        status, details, error = self._run_llm_session(
            page=page,
            goal=goal,
            plan=plan,
            initial_context=initial_dom_summary,
        )

        return SubAgentResult(
            success=(status == "completed"),
            status=status,
            details=details,
            error=error,
        )

    # ---------------------------------------------------------------------
    # Низкоуровневые действия
    # ---------------------------------------------------------------------
    def _open_hh_home(self, page: Page) -> None:
        logger.info("[hhru] Navigating to hh.ru…")
        try:
            page.goto(HHRU_HOME_URL, wait_until="domcontentloaded")
            logger.info("[hhru] hh.ru appears to be open.")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[hhru] Failed to open hh.ru: {exc}")
            raise

    # ---------------------------------------------------------------------
    # Цикл LLM + инструменты
    # ---------------------------------------------------------------------
    def _run_llm_session(
        self,
        page: Page,
        goal: str,
        plan: str,
        initial_context: str = "",
    ) -> tuple[str, str, Optional[str]]:
        """Диалоговый цикл с LLM, который управляет арендуемыми ресурсами через BrowserToolbox."""

        client = get_client()
        if client is None:
            msg = "LLM клиент недоступен, не могу управлять арендуемыми ресурсами"
            logger.error(f"[hhru] {msg}")
            return "failed", msg, "llm_unavailable"

        toolbox = BrowserToolbox()

        system_prompt = compose_prompt(
            "Ты подагент, который помогает работать с арендуемыми ресурсами через браузер.",
            BROWSER_CONTEXT,
            BROWSER_ACTION_RULES,
            SESSION_RULES,
            RENTAL_FLOWS,
            SAFETY_LIMITS,
            SCREENSHOT_GUIDE,
            FINAL_REPORT,
        )

        user_parts: List[str] = []
        user_parts.append(f"Цель пользователя: {goal}")
        if plan:
            user_parts.append(f"План верхнего уровня от основного агента:\n{plan}")
        if initial_context:
            user_parts.append(f"Краткий снимок DOM при входе (сжатый):\n{initial_context}")
        user_parts.append(
            "Сначала сделай read_view и опиши, что видишь на странице, "
            "затем шаг за шагом двигайся к цели."
        )
        user_content = "\n\n".join(user_parts)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Начальное наблюдение read_view — КАК ТЕКСТ ДЛЯ МОДЕЛИ, а не как tool-сообщение.
        try:
            observation = toolbox.read_view()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[hhru] Failed to read initial view: {exc}")
            observation = ""

        if observation:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Результат первого вызова инструмента read_view "
                        "(краткое описание текущей страницы):\n"
                        f"{observation}"
                    ),
                }
            )

        actions_log: List[str] = []
        no_progress_steps = 0
        last_observation = observation

        for step_idx in range(30):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=toolbox.openai_tools(),
                temperature=0.1,
            )

            msg = response.choices[0].message

            # Мысли LLM в этом шаге
            if msg.content:
                log_thought("hhru", f"Шаг {step_idx}:\n{msg.content}".strip())

            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content,
            }
            if msg.tool_calls:
                assistant_msg["tool_calls"] = msg.tool_calls
            messages.append(assistant_msg)

            if msg.tool_calls:
                step_made_progress = False

                for call in msg.tool_calls:
                    logger.info(
                        f"[hhru] Using tool: {call.function.name} args={call.function.arguments}"
                    )
                    log_thought(
                        "hhru",
                        f"Вызываю инструмент {call.function.name} "
                        f"с аргументами {call.function.arguments}",
                    )

                    args = json.loads(call.function.arguments or "{}")  # type: ignore[arg-type]
                    result = toolbox.execute(call.function.name, args)
                    formatted = format_tool_observation(result)
                    actions_log.append(formatted)

                    log_thought(
                        "hhru",
                        f"Результат инструмента {call.function.name}: {formatted}",
                    )

                    # проверка изменения observation
                    if result.observation and result.observation != last_observation:
                        step_made_progress = True
                        last_observation = result.observation

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.observation,
                        }
                    )

                if step_made_progress:
                    no_progress_steps = 0
                else:
                    no_progress_steps += 1

                if no_progress_steps >= 3:
                    msg_text = (
                        "Несколько шагов подряд не привели к заметным изменениям на странице. "
                        "Подагент остановился, чтобы не зациклиться. Попробуйте сузить задачу."
                    )
                    log_thought("hhru", msg_text)
                    return "failed", msg_text, "no_progress"

                continue

            # Нет tool_calls — считаем это финальным ответом подагента
            final_text = msg.content or ""
            summary = "\n".join(actions_log[-10:])
            report_parts = [
                "Отчёт подагента по арендуемым ресурсам:",
                summary or "(действий с браузером не потребовалось)",
                "",
                final_text,
            ]
            full_report = "\n".join([p for p in report_parts if p])

            log_thought(
                "hhru", f"Финальный ответ подагента hh.ru:\n{final_text}".strip()
            )

            return "completed", full_report, None

        # Лимит шагов исчерпан
        msg_text = (
            "Цикл подагента завершился без финального ответа после максимального числа шагов."
        )
        logger.warning(f"[hhru] {msg_text}")
        log_thought("hhru", msg_text)
        return "failed", msg_text, "no_final_answer"


__all__ = ["HhRuSubAgent"]

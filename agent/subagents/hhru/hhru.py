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
from config.sites import HHRU_HOME_URL


# ----------------------------------------------------------------------------
# Подагент hh.ru
# ----------------------------------------------------------------------------

class HhRuSubAgent:
    """Подагент, специализирующийся на работе с сайтом hh.ru.

    Задачи:
    - открыть hh.ru (если ещё не открыт);
    - найти и отфильтровать подходящие вакансии;
    - при необходимости сформировать отклик / сопроводительное письмо;
    - действовать осторожно: НЕ отправлять отклики автоматически без явного запроса
      и подтверждения на верхнем уровне (это уже обрабатывается в run_agent).
    """

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

        # Если hh уже открыт — не перезаходим
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
        """Диалоговый цикл с LLM, который управляет hh.ru через BrowserToolbox."""

        client = get_client()
        if client is None:
            msg = "LLM клиент недоступен, не могу управлять hh.ru"
            logger.error(f"[hhru] {msg}")
            return "failed", msg, "llm_unavailable"

        toolbox = BrowserToolbox()

        system_prompt = (
            "Ты подагент для работы с сайтом hh.ru (HeadHunter).\n"
            "\n"
            "Общие правила:\n"
            "- Ты управляешь браузером только через предоставленные инструменты.\n"
            "- Сначала анализируешь текущую страницу, затем кликаешь / вводишь текст / скроллишь.\n"
            "- Не полагайся на фиксированные id/class — ищи элементы по видимому тексту и структуре.\n"
            "- Умей продолжать с уже открытой страницы (история действий сохраняется).\n"
            "- Важно: не отправляй отклики/резюме в один клик без явного указания.\n"
            "  Если задача — \"найди вакансии\" без слова \"откликнись\", ограничься поиском и анализом.\n"
            "\n"
            "Сценарии на hh.ru:\n"
            "- Поиск вакансий по названию (например, AI-инженер, Python-разработчик).\n"
            "- Фильтрация по городу, зарплате, типу занятости.\n"
            "- Открытие карточек вакансий, чтение описаний и требований.\n"
            "- Подготовка черновика сопроводительного письма (без отправки).\n"
            "\n"
            "Ограничения безопасности:\n"
            "- Никогда не нажимай кнопки, которые окончательно отправляют что-то важное:\n"
            "  «Откликнуться», «Отправить отклик», «Отправить резюме», \"Submit\", \"Apply\" и т.п.,\n"
            "  если явно не указано и не получено подтверждение на верхнем уровне.\n"
            "- Если ты дошёл до шага отправки отклика, лучше остановись и дай подробный отчёт:\n"
            "  какие вакансии нашёл, какие из них подходят и что собирался бы сделать дальше.\n"
            "\n"
            "Информация о контексте:\n"
            "- Браузер и вкладка сохраняются между запросами пользователя.\n"
            "- Новый запрос может быть продолжением предыдущего (например,\n"
            "  сначала \"найди вакансии\", потом \"добавь фильтр по зарплате\").\n"
            "\n"
            "В конце обязательно выдай отчёт по выполненным действиям и найденным результатам.\n"
        )

        user_parts: List[str] = []
        user_parts.append(f"Цель пользователя (для hh.ru): {goal}")
        if plan:
            user_parts.append(f"План верхнего уровня от основного агента:\n{plan}")
        if initial_context:
            user_parts.append(
                f"Краткий снимок DOM при входе на hh.ru (сжатый):\n{initial_context}"
            )
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
                        "(краткое описание текущей страницы на hh.ru):\n"
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
                        "Несколько шагов подряд не привели к заметным изменениям на hh.ru. "
                        "Подагент остановился, чтобы не зациклиться. Попробуйте сузить задачу."
                    )
                    log_thought("hhru", msg_text)
                    return "failed", msg_text, "no_progress"

                continue

            # Нет tool_calls — считаем это финальным ответом подагента
            final_text = msg.content or ""
            summary = "\n".join(actions_log[-10:])
            report_parts = [
                "Отчёт подагента hh.ru:",
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
            "Цикл работы с hh.ru завершился без финального ответа после максимального числа шагов."
        )
        logger.warning(f"[hhru] {msg_text}")
        log_thought("hhru", msg_text)
        return "failed", msg_text, "no_final_answer"


__all__ = ["HhRuSubAgent"]

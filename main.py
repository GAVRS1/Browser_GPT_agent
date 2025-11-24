from loguru import logger
from agent.agent_loop import enable_console_confirmation, agent_is_busy, run_agent
from agent.tools_init import register_all_tools

# Инициализация инструментов
register_all_tools()
enable_console_confirmation()


def main():
    logger.info("=== Browser AI Agent (Console Mode) ===")
    logger.info("Браузер запускается...")

    print("\nВведите задачу для агента.")

    while True:
        try:
            goal = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not goal:
            print("Введите корректный запрос.")
            continue

        if goal.lower() in {"exit", "quit", "выход"}:
            break

        if agent_is_busy():
            print("Ассистент занят. Подождите...")
            continue

        try:
            run_agent(goal)
        except Exception as exc:
            logger.error(f"[main] Ошибка агента: {exc}")
            print("Ошибка выполнения задачи.")

    print("\nЗавершение работы агента.")


if __name__ == "__main__":
    main()

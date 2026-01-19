import sys

from loguru import logger
from agent.agent_loop import enable_console_confirmation, agent_is_busy, run_agent
from agent.mcp_client import MCPToolClient
from agent.tools_init import register_all_tools
from browser.context import get_page

_NOISY_MODULES = {"config.proxy", "agent.llm_client", "agent.agent_loop"}


def _configure_logging() -> None:
    logger.remove()

    def _console_filter(record) -> bool:
        if record["level"].no < logger.level("WARNING").no:
            return record["name"] not in _NOISY_MODULES
        return True

    logger.add(sys.stderr, level="INFO", filter=_console_filter)


_configure_logging()

# Инициализация инструментов
register_all_tools()
enable_console_confirmation()


def _initialize_environment() -> bool:
    logger.info("Проверка браузера и MCP инструментов...")

    try:
        page = get_page()
        if page is None:
            raise RuntimeError("Page was not created")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[main] Не удалось открыть браузер: {exc}")
        print("Не удалось открыть браузер / инструменты недоступны.")
        return False

    mcp_client = MCPToolClient()
    try:
        mcp_client.list_tools()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[main] MCP инструменты недоступны: {exc}")
        print("Не удалось открыть браузер / инструменты недоступны.")
        return False
    finally:
        try:
            mcp_client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[main] Не удалось закрыть MCP клиент: {exc}")

    return True


def main():
    logger.info("=== Browser AI Agent (Console Mode) ===")
    logger.info("Браузер запускается...")

    if not _initialize_environment():
        return

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

import unittest
import unittest.mock
from pathlib import Path
from typing import List, Tuple

from loguru import logger

from agent.browser_tools import BrowserToolbox


class FakeMouse:
    def __init__(self) -> None:
        self.clicks: List[Tuple[float, float]] = []

    def click(self, x: float, y: float) -> None:  # pragma: no cover - simple recorder
        self.clicks.append((x, y))


class FakeKeyboard:
    def __init__(self) -> None:
        self.events: List[Tuple[str, str]] = []

    def type(self, text: str) -> None:  # pragma: no cover - simple recorder
        self.events.append(("type", text))

    def press(self, key: str) -> None:  # pragma: no cover - simple recorder
        self.events.append(("press", key))


class FakeLocator:
    def __init__(self) -> None:
        self.filled: List[str] = []
        self.pressed: List[str] = []

    def fill(self, text: str, timeout: int = 0) -> None:  # pragma: no cover - simple recorder
        self.filled.append(text)

    def press(self, key: str) -> None:  # pragma: no cover - simple recorder
        self.pressed.append(key)

    def click(self, timeout: int = 0) -> None:  # pragma: no cover - no-op
        return None


class FakePage:
    def __init__(self) -> None:
        self.mouse = FakeMouse()
        self.keyboard = FakeKeyboard()
        self.url = "http://local.test/blind"
        self.saved_screenshots: List[str] = []

    def screenshot(self, path: str, full_page: bool = False) -> bytes:  # pragma: no cover - I/O helper
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"fake image")
        self.saved_screenshots.append(path)
        return b"fake image"

    # These locators are intentionally minimal; in the test we bypass DOM search
    def locator(self, selector: str) -> FakeLocator:  # pragma: no cover - simple factory
        return FakeLocator()


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Message", (), {"content": content})


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **_: object):  # pragma: no cover - controlled output
        return type("Response", (), {"choices": [FakeChoice(self._content)]})


class FakeClient:
    def __init__(self, content: str) -> None:
        completions = FakeCompletions(content)
        self.chat = type("Chat", (), {"completions": completions})


class VisionFallbackTest(unittest.TestCase):
    def test_vision_fallback_clicks_by_coordinates_when_dom_blind(self) -> None:
        page = FakePage()
        toolbox = BrowserToolbox(page=page)

        # DOM-поиск заведомо возвращает None, чтобы активировался vision-fallback
        toolbox._find_text_locator = lambda query: None  # type: ignore[attr-defined]

        fake_client = FakeClient('{"click": {"x": 42, "y": 88}, "reason": "blind dom"}')

        logs: List[str] = []
        sink_id = logger.add(lambda m: logs.append(m), format="{message}")
        try:
            with unittest.mock.patch("agent.browser_tools.get_client", return_value=fake_client):
                result = toolbox.type_text("слепой поиск", "hello", press_enter=True)
        finally:
            logger.remove(sink_id)

        self.assertIn("vision-fallback", result)
        self.assertEqual(page.mouse.clicks, [(42.0, 88.0)])
        self.assertIn(("type", "hello"), page.keyboard.events)
        self.assertIn(("press", "Enter"), page.keyboard.events)
        self.assertTrue(any("vision-fallback" in msg for msg in logs))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

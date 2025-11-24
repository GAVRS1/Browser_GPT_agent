import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.agent_loop import ScreenshotCache
from agent.browser_tools import BrowserToolbox


class _DummyPage:
    def __init__(self) -> None:
        self.url = "https://example.test/page"

    def screenshot(self, path: str, full_page: bool = False) -> None:  # noqa: ARG002
        Path(path).write_bytes(b"test-image")

    def title(self) -> str:
        return "Example"


class ScreenshotToolTests(unittest.TestCase):
    def test_take_screenshot_returns_descriptive_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            toolbox = BrowserToolbox(page=_DummyPage(), screenshots_dir=Path(tmpdir))

            observation = toolbox.take_screenshot()

            self.assertIn(str(Path(tmpdir)), observation)
            self.assertIn("url=https://example.test/page", observation)

            files = list(Path(tmpdir).glob("screenshot_*.png"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].exists())


class ScreenshotCacheTests(unittest.TestCase):
    def test_cache_adds_single_reminder_and_reuses_link(self) -> None:
        cache = ScreenshotCache()

        cache.remember("screenshots/screenshot_123.png — url=example")
        reminder1 = cache.reminder_message()

        self.assertIsNotNone(reminder1)
        assert reminder1 is not None  # for type checkers
        self.assertIn("screenshots/screenshot_123.png", reminder1["content"])
        self.assertIsNone(cache.reminder_message())

        cache.remember("screenshots/screenshot_456.png — url=example2")
        reminder2 = cache.reminder_message()

        self.assertIsNotNone(reminder2)
        assert reminder2 is not None
        self.assertIn("screenshot_456", reminder2["content"])
        self.assertEqual(cache.last_link, "screenshots/screenshot_456.png — url=example2")


if __name__ == "__main__":
    unittest.main()

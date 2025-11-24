import json
from pathlib import Path
import sys

import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.browser_tools import BrowserToolbox


@pytest.fixture()
def page():
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Chromium is not available: {exc}")
        page = browser.new_page()
        yield page
        browser.close()


def test_read_view_includes_common_attributes(page):
    page.set_content(
        """
        <main>
          <input id="search" type="search" placeholder="Поиск товаров" aria-label="Поиск" />
          <button aria-label="Отправить">Отправить</button>
        </main>
        """
    )

    toolbox = BrowserToolbox(page)
    summary = json.loads(toolbox.read_view())
    interactive = summary.get("interactive", [])

    search_entry = next(item for item in interactive if item.get("attrs", {}).get("id") == "search")
    assert search_entry["attrs"].get("placeholder") == "Поиск товаров"
    assert search_entry["attrs"].get("aria_label") == "Поиск"


def test_type_text_prefers_search_by_aria_label(page):
    page.set_content(
        """
        <main>
          <input id="query" type="search" aria-label="Поиск по сайту" />
          <input id="name" type="text" aria-label="Имя" />
        </main>
        """
    )

    toolbox = BrowserToolbox(page)
    result = toolbox.type_text("поиск", "яблоки")

    assert "ввёл текст" in result.lower()
    assert page.eval_on_selector("#query", "el => el.value") == "яблоки"
    assert page.eval_on_selector("#name", "el => el.value") == ""


def test_type_text_uses_placeholder_for_search(page):
    page.set_content(
        """
        <main>
          <input id="search" type="text" placeholder="Поиск по товарам" />
          <input id="email" type="email" placeholder="Email" />
        </main>
        """
    )

    toolbox = BrowserToolbox(page)
    toolbox.type_text("поиск", "чай")

    assert page.eval_on_selector("#search", "el => el.value") == "чай"
    assert page.eval_on_selector("#email", "el => el.value") == ""

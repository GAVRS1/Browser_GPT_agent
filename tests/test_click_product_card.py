import unittest


from agent.browser_tools import (
    _compute_card_score,
    _count_token_matches,
    _extract_search_tokens_from_url,
)


class ClickProductCardTests(unittest.TestCase):
    def test_extract_search_tokens_from_url_prefers_query_and_path(self) -> None:
        url = "https://shop.test/search?text=зеленые+яблоки&foo=bar"
        tokens = _extract_search_tokens_from_url(url)

        self.assertIn("зеленые", tokens)
        self.assertIn("яблоки", tokens)

        path_url = "https://shop.test/search/черный-хлеб"
        path_tokens = _extract_search_tokens_from_url(path_url)
        self.assertIn("черный", path_tokens)
        self.assertIn("хлеб", path_tokens)

    def test_score_prefers_tokens_and_add_button(self) -> None:
        tokens = ["арбуз", "спелый"]
        matched = _count_token_matches("Свежий спелый арбуз", tokens)
        matched_score = _compute_card_score(
            area=20000,
            has_price=True,
            has_title=True,
            is_recipe=False,
            has_add_button=True,
            token_matches=matched,
        )

        unmatched_score = _compute_card_score(
            area=26000,
            has_price=False,
            has_title=True,
            is_recipe=False,
            has_add_button=False,
            token_matches=0,
        )

        self.assertGreater(
            matched_score,
            unmatched_score,
            "Карточка с совпадением по поиску и кнопкой должна ранжироваться выше",
        )


if __name__ == "__main__":
    unittest.main()

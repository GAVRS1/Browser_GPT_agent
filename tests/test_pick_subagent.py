from agent.subagents import pick_subagent
from agent.subagents.hhru import HhRuSubAgent
from agent.subagents.yandex_lavka import YandexLavkaSubAgent
from agent.subagents.yandex_mail import YandexMailSubAgent


def test_pick_subagent_by_mail_domain():
    goal = "Открой https://mail.yandex.ru/u/0/inbox и прочитай последние письма"
    subagent = pick_subagent(goal)

    assert isinstance(subagent, YandexMailSubAgent)


def test_pick_subagent_by_lavka_domain():
    goal = "Найди молоко на https://lavka.yandex.ru/ и добавь в корзину"
    subagent = pick_subagent(goal)

    assert isinstance(subagent, YandexLavkaSubAgent)


def test_pick_subagent_by_hh_domain():
    goal = "Перейди на hh.ru и найди вакансии разработчика"
    subagent = pick_subagent(goal)

    assert isinstance(subagent, HhRuSubAgent)


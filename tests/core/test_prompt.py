"""Сборка промпта ответа (спека clone-response: промпт, прерванный ответ)."""

from terebot.config import Persona
from terebot.core.prompt import SaidByClone, SaidByParticipant, build_messages

PERSONA = Persona(
    passport="Паспорт: ты Теребот.",
    fewshot=(("Вася: как дела?", "Норм."),),
    lexicon=(),
)


def words(messages) -> int:
    """Подменённый счётчик токенов: слова во всех сообщениях."""
    return sum(len(m["content"].split()) for m in messages)


def test_answer_with_conversation_context():
    history = [SaidByParticipant("Вася", "он вчера опять проиграл")]
    question = [SaidByParticipant("Петя", "Теребот, что скажешь?")]
    messages, dropped = build_messages(PERSONA, history, question, 1000, words)
    assert dropped == 0
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("Паспорт: ты Теребот.")
    assert messages[1:3] == [
        {"role": "user", "content": "Вася: как дела?"},
        {"role": "assistant", "content": "Норм."},
    ]
    assert messages[-1] == {
        "role": "user",
        "content": "Вася: он вчера опять проиграл\nПетя: Теребот, что скажешь?",
    }


def test_long_session_drops_oldest_keeps_passport():
    history = [SaidByParticipant("Вася", f"реплика номер {i} " + "слово " * 10) for i in range(50)]
    question = [SaidByParticipant("Петя", "Теребот, ну?")]
    full, _ = build_messages(PERSONA, history, question, 10**6, words)
    budget = words(full) // 3
    messages, dropped = build_messages(PERSONA, history, question, budget, words)
    assert words(messages) <= budget
    assert dropped > 0
    assert messages[0]["content"].startswith(PERSONA.passport)
    text = messages[-1]["content"]
    assert "реплика номер 49" in text and "реплика номер 0 " not in text
    assert text.endswith("Петя: Теребот, ну?")


def test_passport_kept_even_if_over_budget():
    messages, _ = build_messages(PERSONA, [], [SaidByParticipant("Петя", "привет")], 1, words)
    assert messages[0]["content"].startswith(PERSONA.passport)


def test_interrupted_answer_only_spoken_part():
    history = [
        SaidByParticipant("Петя", "Теребот, расскажи"),
        SaidByClone("Короче, было дело. Мы пошли", interrupted=True),
    ]
    question = [SaidByParticipant("Петя", "стоп, другое")]
    messages, _ = build_messages(PERSONA, history, question, 1000, words)
    assert messages[-2] == {
        "role": "assistant",
        "content": "Короче, было дело. Мы пошли (перебили)",
    }


def test_clone_silent_answer_skipped():
    history = [SaidByParticipant("Петя", "раз"), SaidByClone("")]
    messages, _ = build_messages(PERSONA, history, [SaidByParticipant("Петя", "два")], 100, words)
    assert messages[-1]["content"] == "Петя: раз\nПетя: два"

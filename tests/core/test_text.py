"""Фразы и текст для Голоса (спека clone-response: «Текст для Голоса», «Устный ответ»)."""

import pytest

from terebot.core.text import PhraseSplitter, for_voice

LEXICON = (("cs", "кс"), ("Discord", "дискорд"))


def stream(text: str, step: int = 3, min_chars: int = 20, max_chars: int = 200) -> list[str]:
    s = PhraseSplitter(min_chars, max_chars)
    phrases = []
    for i in range(0, len(text), step):
        phrases += s.feed(text[i : i + step])
    return phrases + s.flush()


def test_emoji_removed():
    assert for_voice("ахах, ну ты даёшь 😂", ()) == "ахах, ну ты даёшь"


def test_digits():
    assert for_voice("их было 3", ()) == "их было три"


def test_nick_from_lexicon():
    assert for_voice("го в cs", LEXICON) == "го в кс"
    assert for_voice("Го в CS!", LEXICON) == "Го в кс!"


def test_lexicon_by_word_boundary():
    assert for_voice("csgo и cs", LEXICON) == "csgo и кс"


def test_markup_and_lists_removed():
    text = "**Короче**, вот:\n- раз\n- два [тут](http://x.ru) `код` https://y.ru"
    assert for_voice(text, ()) == "Короче, вот: раз два тут код"


def test_decimal():
    assert for_voice("где-то 2,5 часа", ()) == "где-то две целых пять десятых часа"


def test_three_sentences_first_before_end():
    s = PhraseSplitter(20, 200)
    first = s.feed("Короче, я вчера опять проиграл. Потом ")
    assert first == ["Короче, я вчера опять проиграл."]
    assert s.feed("ещё раз зашёл в игру! И снова") == ["Потом ещё раз зашёл в игру!"]
    assert s.flush() == ["И снова"]


def test_short_sentence_glued_to_next():
    assert stream("Ну. Короче, было дело вчера вечером. Да.") == [
        "Ну. Короче, было дело вчера вечером.",
        "Да.",
    ]


def test_phrases_not_shorter_than_min():
    text = "Ага. Я понял тебя, брат. Нет. Так не пойдёт, честно говоря. Ну ладно, давай."
    phrases = stream(text)
    assert " ".join(phrases) == text
    # Короче минимума может быть только последняя фраза ответа.
    assert all(len(p) >= 20 for p in phrases[:-1])
    assert len(phrases) == 3


def test_long_phrase_cut_at_comma():
    text = "раз два три, " * 30
    phrases = stream(text, max_chars=60)
    assert all(len(p) <= 60 for p in phrases)
    assert all(p.endswith(",") for p in phrases[:-1])


def test_ellipsis_and_multiple_marks_kept_together():
    assert stream("Ты чего, серьёзно?! Ну ладно, бывает...", min_chars=15) == [
        "Ты чего, серьёзно?!",
        "Ну ладно, бывает...",
    ]


@pytest.mark.parametrize("step", [1, 2, 5, 100])
def test_think_block_removed_in_stream(step):
    text = "<think>надо подумать. долго.</think>Короче, всё нормально у меня."
    assert stream(text, step=step) == ["Короче, всё нормально у меня."]


def test_unclosed_lt_is_text():
    assert stream("Ответ такой: 3 <thi", step=2) == ["Ответ такой: 3 <thi"]

import pytest

from terebot.voice_eval.cer import cer, normalize


def test_normalize_case_punctuation_yo():
    assert normalize("Ёлки-палки, ВСЁ!  Готово?..") == "елки палки все готово"


def test_cer_zero_when_only_punctuation_and_yo_differ():
    assert cer("Да ну, серьёзно?", "да ну серьезно") == 0


def test_cer_counts_character_edits():
    # «кот» → «кит»: одна замена на три символа.
    assert cer("кот", "кит") == pytest.approx(1 / 3)


def test_cer_empty_hypothesis_is_one():
    assert cer("ага", "") == 1


def test_cer_rejects_empty_reference():
    with pytest.raises(ValueError):
        cer("?!", "ага")

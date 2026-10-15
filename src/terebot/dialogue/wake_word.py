"""Нечёткий поиск Слова-вызова в тексте Реплики."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz

_NOT_WORD = re.compile(r"[^\w\s]")


def normalize(text: str) -> list[str]:
    """Нижний регистр, ё → е, без пунктуации, токены по пробелам."""
    text = text.lower().replace("ё", "е")
    return _NOT_WORD.sub(" ", text).split()


def _candidates(tokens: list[str]) -> Iterable[str]:
    # Склейка соседних токенов ловит «тере бот», разбитый распознаванием.
    yield from tokens
    yield from (a + b for a, b in zip(tokens, tokens[1:], strict=False))


@dataclass(frozen=True)
class WakeMatch:
    candidate: str
    wake_word: str
    score: float


class WakeWordMatcher:
    def __init__(self, wake_words: Iterable[str], threshold: float, blocklist: Iterable[str] = ()):
        self._wake_words = [" ".join(normalize(word)) for word in wake_words]
        self._blocklist = [" ".join(normalize(word)) for word in blocklist]
        self._threshold = threshold

    def find(self, text: str) -> WakeMatch | None:
        """Лучшее совпадение в тексте или None."""
        best: WakeMatch | None = None
        for candidate in _candidates(normalize(text)):
            match = self._match(candidate)
            if match and (best is None or match.score > best.score):
                best = match
        return best

    def _match(self, candidate: str) -> WakeMatch | None:
        score, wake_word = max((fuzz.ratio(candidate, w), w) for w in self._wake_words)
        if score < self._threshold:
            return None
        # Кандидат, который похож на имя Прототипа не меньше, чем на Слово-вызов, не будит
        # Клона. Сравнение с wake score, а не только с порогом, нужно, чтобы Слово-вызов,
        # близкое к блок-листу («тереша» ≈ «терема»), всё ещё срабатывало.
        blocked = max((fuzz.ratio(candidate, b) for b in self._blocklist), default=0)
        if blocked >= self._threshold and blocked >= score:
            return None
        return WakeMatch(candidate=candidate, wake_word=wake_word, score=score)

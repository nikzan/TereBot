"""Текст ответа LLM → фразы → текст для Голоса. Чистая логика.

`PhraseSplitter` режет поток токенов на фразы по знакам конца предложения, не короче
минимума (чтобы не синтезировать «Ну.» отдельно; исключение — последняя фраза ответа)
и не длиннее максимума. `for_voice` убирает
разметку и эмодзи, применяет лексикон произношения и переводит оставшиеся числа в слова.
"""

import re

from num2words import num2words

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_END = ".!?…\n"


class PhraseSplitter:
    def __init__(self, min_chars: int, max_chars: int):
        self._min = min_chars
        self._max = max_chars
        self._buf = ""
        # Блок рассуждений вырезается целиком, даже если пришёл по кускам.
        self._raw = ""
        self._in_think = False

    def feed(self, chunk: str) -> list[str]:
        self._buf += self._strip_think(chunk)
        return self._cut(final=False)

    def flush(self) -> list[str]:
        if not self._in_think:
            # Незакрытый тег — не рассуждения, а текст.
            self._buf += self._raw
        self._raw = ""
        return self._cut(final=True)

    def _strip_think(self, chunk: str) -> str:
        self._raw += chunk
        out = ""
        while self._raw:
            if self._in_think:
                end = self._raw.find(THINK_CLOSE)
                if end < 0:
                    # Хвост может оказаться началом закрывающего тега.
                    self._raw = self._raw[-len(THINK_CLOSE) :]
                    return out
                self._raw = self._raw[end + len(THINK_CLOSE) :]
                self._in_think = False
                continue
            start = self._raw.find(THINK_OPEN)
            if start >= 0:
                out += self._raw[:start]
                self._raw = self._raw[start + len(THINK_OPEN) :]
                self._in_think = True
                continue
            # Не отдаём хвост, похожий на начало "<think>".
            prefixes = range(len(THINK_OPEN) - 1, 0, -1)
            keep = next((k for k in prefixes if self._raw.endswith(THINK_OPEN[:k])), 0)
            out += self._raw[: len(self._raw) - keep]
            self._raw = self._raw[len(self._raw) - keep :]
            return out
        return out

    def _cut(self, final: bool) -> list[str]:
        phrases = []
        while True:
            cut = self._find_cut(final)
            if cut is None:
                break
            phrase, self._buf = self._buf[:cut].strip(), self._buf[cut:]
            if phrase:
                phrases.append(phrase)
        if final:
            # Конец ответа: хвост — последняя фраза, даже если короче минимума. Склеить его
            # с предыдущей нельзя — та уже ушла в синтез.
            rest, self._buf = self._buf.strip(), ""
            if rest:
                phrases.append(rest)
        return phrases

    def _find_cut(self, final: bool) -> int | None:
        buf = self._buf
        for i, c in enumerate(buf):
            if c not in _END or len(buf[: i + 1].strip()) < self._min:
                continue
            # Знаки конца подряд («?!», «...») — одна граница; ждём символ после них.
            j = i
            while j + 1 < len(buf) and buf[j + 1] in _END:
                j += 1
            if j + 1 < len(buf):
                if buf[j + 1].isspace() or c == "\n":
                    return j + 1
            elif final:
                return j + 1
        if len(buf) > self._max:
            head = buf[: self._max]
            comma = head.rfind(",")
            if comma >= self._min:
                return comma + 1
            space = head.rfind(" ")
            return space if space >= self._min else self._max
        return None


_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff"
    "\U00002190-\U000021ff\U00002b00-\U00002bff️‍⃣]+"
)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_MARKUP = re.compile(r"[*_`#>~|]+")
_NUMBER = re.compile(r"(?<![\w.,])(\d+)(?:[.,](\d+))?(?![\w])")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?…:;])")


def _number(m: re.Match) -> str:
    whole, frac = m.group(1), m.group(2)
    value = float(f"{whole}.{frac}") if frac else int(whole)
    return num2words(value, lang="ru")


def for_voice(text: str, lexicon: tuple[tuple[str, str], ...]) -> str:
    """Фраза от LLM → текст, который Голос произносит правильно."""
    text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _LIST_MARK.sub("", text)
    text = _MARKUP.sub("", text)
    text = _EMOJI.sub("", text)
    for word, spoken in lexicon:
        text = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", spoken, text, flags=re.IGNORECASE)
    text = _NUMBER.sub(_number, text)
    text = " ".join(text.split())
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text).strip()

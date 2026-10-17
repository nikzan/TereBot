"""Промпт ответа Клона: Паспорт, правила устной речи, few-shot и история Сессии.

Чистая логика: число токенов считает переданная функция (в ядре — `/tokenize` llama-server).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from terebot.config import Persona

ORAL_RULES = """\
Ты говоришь голосом в голосовом чате Discord, твой ответ сразу зачитывается вслух.
Отвечай коротко, одной–тремя фразами, как в живом разговоре.
Без разметки, списков, эмодзи и ссылок. Числа пиши словами.
Говори от своего лица, как описано выше. Не говори, что ты ИИ или ассистент.
Реплики собеседников приходят в виде «Имя: текст»."""

INTERRUPTED = "(перебили)"


@dataclass(frozen=True)
class SaidByParticipant:
    name: str
    text: str


@dataclass(frozen=True)
class SaidByClone:
    """Прозвучавшая часть ответа Клона."""

    text: str
    interrupted: bool = False


type HistoryItem = SaidByParticipant | SaidByClone
type Message = dict[str, str]
type TokenCounter = Callable[[list[Message]], int]


def system_prompt(passport: str) -> str:
    return f"{passport}\n\n{ORAL_RULES}"


def _messages(
    persona: Persona, history: Sequence[HistoryItem], question: Sequence[SaidByParticipant]
) -> list[Message]:
    messages: list[Message] = [{"role": "system", "content": system_prompt(persona.passport)}]
    for user, answer in persona.fewshot:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    for item in (*history, *question):
        if isinstance(item, SaidByParticipant):
            role, content = "user", f"{item.name}: {item.text}"
        else:
            if not item.text and not item.interrupted:
                continue
            role = "assistant"
            content = f"{item.text} {INTERRUPTED}".strip() if item.interrupted else item.text
        # Подряд идущие Реплики — одно сообщение: шаблоны чата ждут чередования ролей.
        if messages[-1]["role"] == role and len(messages) > 1 + 2 * len(persona.fewshot):
            messages[-1] = {"role": role, "content": f"{messages[-1]['content']}\n{content}"}
        else:
            messages.append({"role": role, "content": content})
    return messages


def build_messages(
    persona: Persona,
    history: Sequence[HistoryItem],
    question: Sequence[SaidByParticipant],
    budget: int,
    count_tokens: TokenCounter,
) -> tuple[list[Message], int]:
    """Сообщения для /v1/chat/completions и сколько старых записей истории отброшено.

    Паспорт и Реплики, на которые нужен ответ, остаются всегда; если не помещается,
    отбрасываются самые старые записи истории.
    """
    history = list(history)
    dropped = 0
    messages = _messages(persona, history, question)
    while history and count_tokens(messages) > budget:
        history.pop(0)
        dropped += 1
        messages = _messages(persona, history, question)
    return messages, dropped

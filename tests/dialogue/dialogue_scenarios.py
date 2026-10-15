"""Загрузка и прогон YAML-сценариев Политики Диалога. Формат — README.md рядом."""

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from terebot.config import DialogueSettings
from terebot.dialogue import (
    AddToContext,
    CloneFinished,
    ClonePhraseEnded,
    CloneStarted,
    Decision,
    DialoguePolicy,
    ParticipantLeft,
    Respond,
    ResponseFailed,
    SpeechActivity,
    StartDialogue,
    SwitchInterlocutor,
    Utterance,
)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

CLONE_EVENTS = {
    "started": CloneStarted,
    "phrase_ended": ClonePhraseEnded,
    "finished": CloneFinished,
    "failed": ResponseFailed,
}
DECISION_NAMES = {
    "StartDialogue", "Respond", "AddToContext", "InterruptClone", "StopAfterCurrentPhrase",
    "CancelPendingResponse", "SwitchInterlocutor", "EndDialogue",
}  # fmt: skip
STATE_FIELDS = {
    "dialogue": "in_dialogue",
    "interlocutor": "interlocutor",
    "clone": "clone",
    "stopping": "stopping",
    "window_left": "window_left_s",
}
_SHORT = re.compile(r"^(\w+)(?:\((\w+)\))?$")


class ScenarioError(Exception):
    """Сценарий записан не по формату README."""


@dataclass
class Expected:
    name: str
    user: str | None = None
    text: str | None = None
    texts: list[str] | None = None
    reason: str | None = None

    def __str__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in dataclasses.asdict(self).items() if k != "name" and v]
        return f"{self.name}({', '.join(parts)})" if parts else self.name

    def matches(self, d: Decision) -> bool:
        if type(d).__name__ != self.name:
            return False
        if self.user is not None and _user(d) != self.user:
            return False
        if self.text is not None and self.text not in _text(d):
            return False
        if self.texts is not None and (
            not isinstance(d, Respond) or [u.text for u in d.utterances] != self.texts
        ):
            return False
        return self.reason is None or self.reason in d.reason


@dataclass
class Scenario:
    name: str
    path: Path
    whitelist: list[str]
    settings: DialogueSettings
    steps: list[dict[str, Any]] = field(default_factory=list)


def _user(d: Decision) -> str | None:
    if isinstance(d, StartDialogue | SwitchInterlocutor):
        return d.user_id
    if isinstance(d, AddToContext):
        return d.utterance.user_id
    return None


def _text(d: Decision) -> str:
    if isinstance(d, StartDialogue | AddToContext):
        return d.utterance.text
    if isinstance(d, Respond):
        return " | ".join(u.text for u in d.utterances)
    return ""


def describe(d: Decision) -> str:
    name = type(d).__name__
    args = [a for a in (_user(d), _text(d) and repr(_text(d))) if a]
    return f"{name}({', '.join(args)}) — {d.reason}"


def _parse_expected(item: Any, where: str) -> Expected:
    if isinstance(item, str):
        m = _SHORT.match(item.strip())
        if not m or m.group(1) not in DECISION_NAMES:
            raise ScenarioError(f"{where}: непонятное решение {item!r}")
        return Expected(m.group(1), user=m.group(2))
    if isinstance(item, dict) and len(item) == 1:
        [(name, fields)] = item.items()
        if name not in DECISION_NAMES or not isinstance(fields, dict):
            raise ScenarioError(f"{where}: непонятное решение {item!r}")
        unknown = set(fields) - {"user", "text", "texts", "reason"}
        if unknown:
            raise ScenarioError(f"{where}: неизвестные поля {sorted(unknown)} у {name}")
        return Expected(name, **{k: v if k == "texts" else str(v) for k, v in fields.items()})
    raise ScenarioError(f"{where}: решение должно быть строкой или {{Имя: {{поля}}}}: {item!r}")


def load(path: Path) -> Scenario:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "steps" not in raw or "whitelist" not in raw:
        raise ScenarioError(f"{path.name}: нужны поля name, whitelist и steps")
    overrides = raw.get("settings") or {}
    for key in ("wake_words", "wake_blocklist"):
        if key in overrides:
            overrides[key] = tuple(overrides[key])
    try:
        settings = dataclasses.replace(DialogueSettings(), **overrides)
    except TypeError as e:
        raise ScenarioError(f"{path.name}: неизвестная настройка в settings: {e}") from None
    return Scenario(
        name=raw.get("name", path.stem),
        path=path,
        whitelist=[str(u) for u in raw["whitelist"]],
        settings=settings,
        steps=raw["steps"],
    )


def all_scenarios() -> list[Path]:
    return sorted(SCENARIOS_DIR.glob("*.yaml"))


def _event(step: dict[str, Any], t: float, where: str):
    kinds = {"utterance", "speech", "clone", "left", "tick"} & step.keys()
    if len(kinds) != 1 or not set(step) <= kinds | {"t"}:
        raise ScenarioError(f"{where}: в шаге с t должно быть ровно одно событие: {step}")
    [kind] = kinds
    value = step[kind]
    if kind == "utterance":
        start = float(value.get("start", t))
        return Utterance(str(value["user"]), str(value["text"]), start, t)
    if kind == "speech":
        return SpeechActivity(str(value["user"]), int(value["ms"]), t)
    if kind == "clone":
        if value not in CLONE_EVENTS:
            raise ScenarioError(f"{where}: clone должно быть одним из {sorted(CLONE_EVENTS)}")
        return CLONE_EVENTS[value](t)
    if kind == "left":
        return ParticipantLeft(str(value), t)
    return None  # tick


def run(scenario: Scenario) -> None:
    """Прогоняет сценарий; при расхождении валит тест с диффом решений."""
    policy = DialoguePolicy(scenario.settings, scenario.whitelist)
    got: list[Decision] = []
    last_t = float("-inf")
    for i, step in enumerate(scenario.steps, start=1):
        where = f"{scenario.path.name}, шаг {i}"
        if not isinstance(step, dict):
            raise ScenarioError(f"{where}: шаг должен быть словарём: {step!r}")
        if "expect" in step:
            expected = [_parse_expected(item, where) for item in step["expect"] or []]
            ok = len(expected) == len(got) and all(
                e.matches(d) for e, d in zip(expected, got, strict=True)
            )
            if not ok:
                pytest.fail(_diff(where, last_t, expected, got), pytrace=False)
            got = []
        elif "state" in step:
            _check_state(policy, step["state"], where)
        elif "t" in step:
            t = float(step["t"])
            if t < last_t:
                raise ScenarioError(f"{where}: время идёт назад ({t} после {last_t})")
            last_t = t
            event = _event(step, t, where)
            got += policy.tick(t) if event is None else policy.handle(event)
        else:
            raise ScenarioError(f"{where}: шаг должен быть событием с t, expect или state")
    if got:
        pytest.fail(_diff(f"{scenario.path.name}, конец сценария", last_t, [], got), pytrace=False)


def _check_state(policy: DialoguePolicy, expected: dict[str, Any], where: str) -> None:
    state = policy.state
    for key, value in expected.items():
        if key not in STATE_FIELDS:
            raise ScenarioError(f"{where}: неизвестное поле state {key!r}")
        actual = getattr(state, STATE_FIELDS[key])
        if key == "window_left" and value is not None and actual is not None:
            same = abs(actual - float(value)) < 1e-6
        else:
            same = actual == value
        if not same:
            pytest.fail(
                f"{where}: state.{key} ожидалось {value!r}, получено {actual!r}\n{state}",
                pytrace=False,
            )


def _diff(where: str, t: float, expected: list[Expected], got: list[Decision]) -> str:
    lines = [f"{where} (t={t}): решения не совпали", "  ожидалось:"]
    lines += [f"    - {e}" for e in expected] or ["    (ничего)"]
    lines.append("  получено:")
    lines += [f"    - {describe(d)}" for d in got] or ["    (ничего)"]
    return "\n".join(lines)

"""Бенчмарк Голоса: скорость и память синтеза для конфигурации «модель × бэкенд × точность».

Запуск — в группе uv своего бэкенда (группы взаимоисключающие):
    uv run --group tts-mlx python scripts/bench_voice.py --config 2 [--with-llm]

Адаптер MLX — общий с Клоном (terebot.voice).

Пишет data/bench/<ts>-<id>/: WAV по фразам и results.json. Метрики качества дописывает
`uv run --group eval python -m terebot.voice_eval <каталог>`.
"""

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from terebot.voice import MlxVoice

ROOT = Path(__file__).resolve().parents[1]
PHRASES = ROOT / "config" / "eval" / "phrases_ru.txt"
REFERENCE_DIR = ROOT / "data" / "bench" / "reference"
OUT_DIR = ROOT / "data" / "bench"
WARMUP = "warmup"
GB = 1024**3
# LLM стартового стека для режима --with-llm (docs/PLAN.md), из кэша Hugging Face.
LLM_GGUF = (
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/*/Qwen3.5-4B-Q4_K_M.gguf"
)


@dataclass(frozen=True)
class Config:
    adapter: str
    model: str
    precision: str


# Матрица из design.md change evaluate-voice-on-mps.
CONFIGS = {
    "1": Config("qwen_torch", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "bf16"),
    "2": Config("qwen_mlx", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", "bf16"),
    "3": Config("qwen_mlx", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit", "8bit"),
    "4": Config("qwen_mlx", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16", "bf16"),
    "5": Config("qwen_torch", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "bf16"),
    "6": Config("chatterbox", "ResembleAI/chatterbox", "default"),
}


@dataclass(frozen=True)
class Phrase:
    idx: int
    category: str
    text: str


def load_phrases(path: Path = PHRASES) -> tuple[list[Phrase], list[Phrase]]:
    """Возвращает (прогрев, набор). Нумерация набора — по порядку в файле, с 1."""
    warmup, phrases = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        category, text = line.split("\t", 1)
        if category == WARMUP:
            warmup.append(Phrase(0, category, text))
        else:
            phrases.append(Phrase(len(phrases) + 1, category, text))
    return warmup, phrases


class Adapter(Protocol):
    """Бэкенд синтеза. Импорты библиотек — внутри load(), чтобы скрипт открывался в любой группе."""

    # Отдаёт ли synthesize() звук по частям до конца синтеза.
    streaming: bool
    sample_rate: int
    # Пакеты, версии которых пишутся в results.json.
    packages: tuple[str, ...]

    def load(self, model: str, precision: str) -> None: ...

    def prepare_voice(self, ref_wav: Path, ref_text: str) -> None: ...

    def synthesize(self, text: str) -> Iterator[np.ndarray]: ...

    def seed(self, seed: int) -> None: ...

    def backend_memory(self) -> tuple[int, str] | None:
        """Текущая память бэкенда в байтах и имя счётчика, если бэкенд её сообщает."""
        ...

    def backend_peak_memory(self) -> int | None:
        """Пик памяти по счётчику самого бэкенда, если он его ведёт."""
        ...


# --- Статистики ---------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Перцентиль с линейной интерполяцией (как numpy.percentile по умолчанию)."""
    if not values:
        raise ValueError("пустой список")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def stats(values: list[float]) -> dict:
    return {"median": statistics.median(values), "p90": percentile(values, 90)}


def summarize(records: list[dict]) -> dict:
    return {
        "rtf": stats([r["rtf"] for r in records]),
        "first_audio_s": stats([r["first_audio_s"] for r in records]),
        "synth_s_total": sum(r["synth_s"] for r in records),
        "audio_s_total": sum(r["audio_s"] for r in records),
    }


# --- Замер --------------------------------------------------------------------------------


@dataclass
class Synthesis:
    audio: np.ndarray
    synth_s: float
    first_audio_s: float
    chunks: int


def timed_synthesis(
    adapter: Adapter, text: str, clock: Callable[[], float] = time.perf_counter
) -> Synthesis:
    chunks = []
    start = clock()
    first = None
    for chunk in adapter.synthesize(text):
        if first is None:
            first = clock() - start
        chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    total = clock() - start
    if not chunks:
        raise RuntimeError(f"бэкенд не вернул звук для фразы: {text}")
    return Synthesis(np.concatenate(chunks), total, first, len(chunks))


def measure_phrases(
    adapter: Adapter,
    phrases: list[Phrase],
    warmup: list[Phrase],
    seed: int,
    on_audio: Callable[[Phrase, np.ndarray], str],
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict]:
    """Прогрев (не входит в результат), затем замер каждой фразы.

    Сид ставится перед каждой фразой, чтобы результат фразы не зависел от порядка прогона.
    on_audio сохраняет звук и возвращает имя файла.
    """
    for p in warmup:
        adapter.seed(seed)
        timed_synthesis(adapter, p.text, clock)

    records = []
    for p in phrases:
        adapter.seed(seed)
        s = timed_synthesis(adapter, p.text, clock)
        audio_s = len(s.audio) / adapter.sample_rate
        records.append(
            {
                "idx": p.idx,
                "category": p.category,
                "text": p.text,
                "wav": on_audio(p, s.audio),
                "synth_s": s.synth_s,
                "audio_s": audio_s,
                "rtf": s.synth_s / audio_s,
                # Без стриминга первый звук появляется только после синтеза всей фразы.
                "first_audio_s": s.first_audio_s if adapter.streaming else s.synth_s,
                "first_audio_is_full_synthesis": not adapter.streaming,
                "chunks": s.chunks,
            }
        )
    return records


# --- Память -------------------------------------------------------------------------------


def system_memory() -> dict:
    """Занятая память системы по vm_stat (active + wired + сжатая) и swap."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))

    def pages(name: str) -> int:
        return int(re.search(rf"{name}:\s+(\d+)", out).group(1))

    used = pages("Pages active") + pages("Pages wired down") + pages("Pages occupied by compressor")
    import psutil

    return {"used_gb": used * page / GB, "swap_used_gb": psutil.swap_memory().used / GB}


class MemorySampler:
    """Фоновый опрос пиков: RSS процесса, счётчик бэкенда, при необходимости — память системы."""

    def __init__(self, adapter: Adapter, with_system: bool, interval_s: float = 0.05):
        import psutil

        self._proc = psutil.Process()
        self._adapter = adapter
        self._with_system = with_system
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.rss_peak = 0
        self.backend_peak = 0
        self.backend_source: str | None = None
        self.system_peak: dict | None = None
        self.system_start = system_memory() if with_system else None

    def _sample(self, n: int) -> None:
        self.rss_peak = max(self.rss_peak, self._proc.memory_info().rss)
        backend = self._adapter.backend_memory()
        if backend:
            self.backend_peak = max(self.backend_peak, backend[0])
            self.backend_source = backend[1]
        # vm_stat — внешний процесс, поэтому система опрашивается раз в секунду.
        if self._with_system and n % max(1, round(1 / self._interval)) == 0:
            cur = system_memory()
            if self.system_peak is None:
                self.system_peak = cur
            else:
                self.system_peak = {k: max(v, cur[k]) for k, v in self.system_peak.items()}

    def _run(self) -> None:
        n = 0
        while not self._stop.is_set():
            self._sample(n)
            n += 1
            self._stop.wait(self._interval)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample(0)

    def report(self) -> dict:
        backend_peak = max(self.backend_peak, self._adapter.backend_peak_memory() or 0)
        rss_gb = self.rss_peak / GB
        backend_gb = backend_peak / GB if self.backend_source else None
        if backend_gb is not None and backend_gb > rss_gb:
            peak, source = backend_gb, self.backend_source
        else:
            peak, source = rss_gb, "rss"
        report = {
            "peak_gb": peak,
            "source": source,
            "rss_peak_gb": rss_gb,
            "backend_peak_gb": backend_gb,
            "backend_source": self.backend_source,
        }
        if self.system_start is not None:
            report["system"] = {"start": self.system_start, "peak": self.system_peak}
        return report


# --- LLM рядом ----------------------------------------------------------------------------


class LlmNotRunning(Exception):
    pass


def check_llm(url: str) -> dict:
    """Проверяет, что llama-server жив, и возвращает его PID и модель для results.json."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
            health = json.load(r)
    except (urllib.error.URLError, OSError) as e:
        raise LlmNotRunning(
            f"llama-server не отвечает на {url}/health ({e}).\n"
            f"Запустите его и прогрейте одним запросом, чтобы веса легли в память:\n"
            f"  llama-server -m {LLM_GGUF} -c 4096 --port {urllib.parse.urlsplit(url).port}\n"
            f'  curl {url}/completion -d \'{{"prompt": "Привет", "n_predict": 8}}\''
        ) from None
    if health.get("status") != "ok":
        raise LlmNotRunning(f"llama-server на {url} ещё не готов: {health}")
    model = None
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=3) as r:
            model = json.load(r)["data"][0]["id"]
    except (urllib.error.URLError, OSError, KeyError, IndexError):
        pass
    pids = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True)
    return {"url": url, "model": model, "pids": [int(p) for p in pids.stdout.split()]}


# --- Окружение ----------------------------------------------------------------------------


def display_path(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def environment(packages: tuple[str, ...]) -> dict:
    versions = {}
    for name in ("numpy", "psutil", *packages):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    return {
        "mac_model": sysctl("hw.model"),
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": int(sysctl("hw.memsize")) / GB,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "packages": versions,
        "git_commit": commit,
        "env": {k: v for k, v in os.environ.items() if k.startswith("PYTORCH_")},
    }


# --- Адаптеры -----------------------------------------------------------------------------


def _seed_common(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class _TorchMps:
    """Общее для бэкендов на PyTorch MPS: импорт torch с фолбэком на CPU, память и сид."""

    _torch = None

    def _import_torch(self):
        # Должно быть выставлено до импорта torch: операторы без MPS-ядра уйдут на CPU.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS недоступен")
        self._torch = torch
        return torch

    def seed(self, seed: int) -> None:
        _seed_common(seed)
        self._torch.manual_seed(seed)

    def backend_memory(self) -> tuple[int, str] | None:
        if self._torch is None:
            return None
        return self._torch.mps.driver_allocated_memory(), "torch.mps.driver_allocated_memory"

    def backend_peak_memory(self) -> int | None:
        return None


class QwenTorchAdapter(_TorchMps):
    """Официальный qwen-tts на PyTorch MPS. Стриминга вывода в его API нет."""

    streaming = False
    packages = ("qwen-tts", "transformers", "accelerate", "torch", "torchaudio")
    language = "Russian"

    def __init__(self):
        self.sample_rate = 0

    def load(self, model: str, precision: str) -> None:
        torch = self._import_torch()
        from qwen_tts import Qwen3TTSModel

        dtype = {"bf16": torch.bfloat16}[precision]
        self._model = Qwen3TTSModel.from_pretrained(
            model, device_map="mps", dtype=dtype, attn_implementation="sdpa"
        )

    def prepare_voice(self, ref_wav: Path, ref_text: str) -> None:
        self._prompt = self._model.create_voice_clone_prompt(
            ref_audio=str(ref_wav), ref_text=ref_text
        )

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        wavs, sr = self._model.generate_voice_clone(
            text=text, language=self.language, voice_clone_prompt=self._prompt
        )
        self.sample_rate = sr
        yield wavs[0]


# Qwen3-TTS на MLX — тот же адаптер, что у Клона (terebot.voice), чтобы синтез совпадал.
QwenMlxAdapter = MlxVoice


class ChatterboxAdapter(_TorchMps):
    """Chatterbox Multilingual на PyTorch MPS. Стриминга нет."""

    streaming = False
    packages = ("chatterbox-tts", "transformers", "torch", "torchaudio")
    language = "ru"

    def load(self, model: str, precision: str) -> None:
        self._import_torch()
        from chatterbox.mtl_tts import REPO_ID, ChatterboxMultilingualTTS

        if model == REPO_ID:
            self._model = ChatterboxMultilingualTTS.from_pretrained(device="mps")
        else:
            self._model = ChatterboxMultilingualTTS.from_local(model, device="mps")
        self.sample_rate = self._model.sr

    def prepare_voice(self, ref_wav: Path, ref_text: str) -> None:
        # Chatterbox клонирует только по звуку, транскрипт не нужен.
        self._model.prepare_conditionals(str(ref_wav))

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        yield self._model.generate(text, language_id=self.language).squeeze(0).numpy()


def make_adapter(name: str, args: argparse.Namespace) -> Adapter:
    if name == "qwen_torch":
        return QwenTorchAdapter()
    if name == "qwen_mlx":
        return QwenMlxAdapter(args.streaming_interval, args.cache_limit_gb)
    if name == "chatterbox":
        return ChatterboxAdapter()
    raise ValueError(f"неизвестный адаптер {name}")


# --- Запуск -------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Path:
    config = CONFIGS[args.config]
    if args.model:
        config = Config(config.adapter, args.model, config.precision)
    llm = check_llm(args.llm_url) if args.with_llm else None

    warmup, phrases = load_phrases(args.phrases)
    warmup = warmup[: args.warmup]
    if args.limit:
        phrases = phrases[: args.limit]
    ref_wav = args.reference / "ref.wav"
    ref_text = (args.reference / "ref.txt").read_text(encoding="utf-8").strip()

    suffix = "-llm" if args.with_llm else ""
    label = f"-{args.label}" if args.label else ""
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = args.out / f"{ts}-{args.config}{label}{suffix}"
    out.mkdir(parents=True)

    import soundfile as sf

    adapter = make_adapter(config.adapter, args)

    def save(p: Phrase, audio: np.ndarray) -> str:
        name = f"{p.idx:04d}.wav"
        sf.write(out / name, audio, adapter.sample_rate)
        return name

    with MemorySampler(adapter, with_system=args.with_llm) as mem:
        t = time.perf_counter()
        adapter.load(config.model, config.precision)
        load_s = time.perf_counter() - t
        t = time.perf_counter()
        adapter.prepare_voice(ref_wav, ref_text)
        prepare_s = time.perf_counter() - t
        # Пик к этому моменту — от загрузки и подготовки голоса, без синтеза фраз.
        memory_after_prepare = mem.report()
        records = measure_phrases(adapter, phrases, warmup, args.seed, save)

    results = {
        "config": {"id": args.config, **asdict(config), "streaming": adapter.streaming},
        "params": {
            "seed": args.seed,
            "warmup_phrases": len(warmup),
            "streaming_interval_s": args.streaming_interval if adapter.streaming else None,
            "cache_limit_gb": args.cache_limit_gb,
            "reference": {
                "wav": display_path(ref_wav),
                "text": ref_text,
                "duration_s": sf.info(ref_wav).duration,
            },
        },
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": environment(adapter.packages),
        "with_llm": llm,
        "load_s": load_s,
        "prepare_voice_s": prepare_s,
        "sample_rate": adapter.sample_rate,
        "summary": summarize(records),
        "memory": {**mem.report(), "after_prepare": memory_after_prepare},
        "phrases": records,
    }
    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--model", help="другой путь или репозиторий модели (своя конвертация)")
    parser.add_argument("--label", help="добавка к имени каталога прогона")
    parser.add_argument("--with-llm", action="store_true", help="рядом запущен llama-server")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080")
    parser.add_argument("--phrases", type=Path, default=PHRASES)
    parser.add_argument("--reference", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, help="только первые N фраз набора")
    parser.add_argument("--warmup", type=int, default=3, help="сколько фраз прогрева")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--streaming-interval",
        type=float,
        default=2.0,
        help="mlx-audio: секунд звука в чанке стриминга (по умолчанию как в библиотеке)",
    )
    parser.add_argument(
        "--cache-limit-gb",
        type=float,
        help="mlx: лимит кэша буферов (mx.set_cache_limit), по умолчанию как в библиотеке",
    )
    args = parser.parse_args()
    try:
        out = run(args)
    except LlmNotRunning as e:
        sys.exit(f"Ошибка: {e}")
    s = json.loads((out / "results.json").read_text(encoding="utf-8"))
    print(
        f"{display_path(out)}: RTF медиана {s['summary']['rtf']['median']:.2f}, "
        f"p90 {s['summary']['rtf']['p90']:.2f}; первый звук медиана "
        f"{s['summary']['first_audio_s']['median']:.2f} с; загрузка {s['load_s']:.1f} с; "
        f"память {s['memory']['peak_gb']:.2f} ГБ ({s['memory']['source']})"
    )


if __name__ == "__main__":
    main()

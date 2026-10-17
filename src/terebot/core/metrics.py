"""Память во время Сессии — тем же способом, что в бенчмарке этапа 3 (scripts/bench_voice.py),
чтобы цифры были сравнимы: RSS ядра, MLX active + cache, RSS llama-server, vm_stat и swap."""

import contextlib
import re
import subprocess
from collections.abc import Callable

import psutil

GB = 1024**3


def system_memory() -> tuple[float, float]:
    """Занятая память системы по vm_stat (active + wired + сжатая) и swap, ГБ."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))

    def pages(name: str) -> int:
        return int(re.search(rf"{name}:\s+(\d+)", out).group(1))

    used = pages("Pages active") + pages("Pages wired down") + pages("Pages occupied by compressor")
    return used * page / GB, psutil.swap_memory().used / GB


class MemorySampler:
    def __init__(self, mlx_memory: Callable[[], int | None], llm_pid: Callable[[], int | None]):
        self._proc = psutil.Process()
        self._mlx = mlx_memory
        self._llm_pid = llm_pid

    def sample(self) -> dict:
        used, swap = system_memory()
        llm_rss = None
        pid = self._llm_pid()
        if pid:
            with contextlib.suppress(psutil.Error):
                llm_rss = psutil.Process(pid).memory_info().rss / GB
        mlx = self._mlx()
        return {
            "core_rss_gb": self._proc.memory_info().rss / GB,
            "mlx_gb": None if mlx is None else mlx / GB,
            "llm_rss_gb": llm_rss,
            "system_used_gb": used,
            "swap_gb": swap,
        }

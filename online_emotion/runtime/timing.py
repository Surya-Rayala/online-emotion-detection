"""Per-stage timing + throughput stats, printed to the terminal during runs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


class _EMA:
    """Exponential moving average (alpha weights the newest sample)."""

    __slots__ = ("alpha", "value", "n")

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.value = 0.0
        self.n = 0

    def update(self, x: float) -> float:
        self.value = x if self.n == 0 else self.alpha * x + (1.0 - self.alpha) * self.value
        self.n += 1
        return self.value


class Stopwatch:
    """``with Stopwatch() as sw: ...`` then read ``sw.elapsed`` (seconds)."""

    __slots__ = ("elapsed", "_t")

    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t


@dataclass
class RunStats:
    """Rolling preprocess / infer / postprocess timings + FPS."""

    frames: int = 0
    dropped: int = 0
    _pre: _EMA = field(default_factory=_EMA)
    _infer: _EMA = field(default_factory=_EMA)
    _post: _EMA = field(default_factory=_EMA)
    _total: _EMA = field(default_factory=_EMA)

    def update(self, pre_s: float, infer_s: float, post_s: float) -> None:
        self.frames += 1
        self._pre.update(pre_s * 1e3)
        self._infer.update(infer_s * 1e3)
        self._post.update(post_s * 1e3)
        self._total.update((pre_s + infer_s + post_s) * 1e3)

    @property
    def preprocess_ms(self) -> float:
        return self._pre.value

    @property
    def infer_ms(self) -> float:
        return self._infer.value

    @property
    def postprocess_ms(self) -> float:
        return self._post.value

    @property
    def latency_ms(self) -> float:
        return self._total.value

    @property
    def fps(self) -> float:
        return 1000.0 / self.latency_ms if self.latency_ms > 0 else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "frames": self.frames,
            "dropped": self.dropped,
            "fps": round(self.fps, 2),
            "latency_ms": round(self.latency_ms, 2),
            "preprocess_ms": round(self.preprocess_ms, 2),
            "infer_ms": round(self.infer_ms, 2),
            "postprocess_ms": round(self.postprocess_ms, 2),
        }

    def format_line(self) -> str:
        return (
            f"frames={self.frames} fps={self.fps:5.1f} "
            f"latency={self.latency_ms:6.1f}ms "
            f"[pre={self.preprocess_ms:5.1f} infer={self.infer_ms:6.1f} post={self.postprocess_ms:5.1f}]"
            + (f" dropped={self.dropped}" if self.dropped else "")
        )

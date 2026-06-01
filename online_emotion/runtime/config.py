"""Resolved, immutable run configuration.

An :class:`InferenceConfig` snapshot is attached to every result and written
into ``export_meta.json`` so any output can be traced back to the exact
family / weights / runtime / device / precision that produced it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

SCHEMA_VERSION = "online-inference-v1"


@dataclass(frozen=True)
class InferenceConfig:
    package: str                       # "online_face" | "online_emotion"
    model: str                         # family name, e.g. "retinaface"
    weights: str                       # resolved weights key or path
    weights_fingerprint: str           # sha256 (first MB) or "" if unknown
    runtime: str                       # resolved runtime
    device: str                        # resolved device string
    precision: str                     # resolved precision
    input_size: Tuple[int, int]        # (H, W)
    schema_version: str = SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.package}:{self.model} weights={self.weights} "
            f"runtime={self.runtime} device={self.device} "
            f"precision={self.precision} input={self.input_size[0]}x{self.input_size[1]}"
        )

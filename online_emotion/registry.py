"""Weights registry for the emotion package.

``weights`` may be a known HSEmotion key (the ``hsemotion`` package fetches the
checkpoint), a path to a ready artifact (``.onnx``/``.engine``/``.torchscript``),
or ``None`` (the family default). Override/extend via
``~/.online/emotion_registry.json`` or ``$ONLINE_EMOTION_REGISTRY``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .families.base import ResolvedWeights
from .models.hsemotion import MODEL_META
from .models.hsemotion.labels import DEFAULT_WEIGHT
from .runtime.errors import UnknownWeightsError
from .runtime.logging import get_logger

_log = get_logger("registry")
ARTIFACT_SUFFIXES = {".onnx", ".engine", ".plan", ".trt", ".torchscript", ".ts"}


def _overrides() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in (Path.home() / ".online" / "emotion_registry.json", Path(os.getenv("ONLINE_EMOTION_REGISTRY", ""))):
        try:
            if p and p.is_file():
                out.update(json.loads(p.read_text()))
        except Exception as e:  # pragma: no cover
            _log.warning("ignoring bad registry override %s: %s", p, e)
    return out


def _models() -> Dict[str, dict]:
    merged = dict(MODEL_META)
    merged.update(_overrides())
    return merged


def available_weights(model: Optional[str] = None) -> List[str]:
    return list(_models().keys())


def _fingerprint_key(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:16]


def resolve_weights(weights: Optional[str], cache) -> ResolvedWeights:
    models = _models()
    if weights is None:
        weights = DEFAULT_WEIGHT
    key = str(weights)

    if key in models:
        m = models[key]
        return ResolvedWeights(
            key=key, path=Path(f"hsemotion:{key}"), fingerprint=_fingerprint_key(key),
            exportable=True, is_artifact=False, arch=m.get("arch", "efficientnet_b0"),
            meta={**m, "model_name": key, "source": "hsemotion"},
        )

    p = Path(key).expanduser()
    if p.exists() and p.suffix.lower() in ARTIFACT_SUFFIXES:
        return ResolvedWeights(
            key=str(p), path=p, fingerprint=_fingerprint_key(p.name), exportable=False,
            is_artifact=True, arch="", meta={"model_name": p.stem, "classes": 8, "img_size": 224, "va": False},
        )

    raise UnknownWeightsError(
        f"unknown weights {key!r}; available: {available_weights()} (or pass a .onnx/.engine/.torchscript path)"
    )

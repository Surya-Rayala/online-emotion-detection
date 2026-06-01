"""Model-family registry for the emotion package.

Third-party families register via the ``online_emotion.model_families``
entry-point group and are discovered at runtime.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import metadata
from typing import Dict, List, Type

from ..runtime.errors import UnknownModelError
from .base import ExportSpec, ModelFamily, ResolvedWeights
from .hsemotion import HSEmotionFamily

__all__ = ["ModelFamily", "ResolvedWeights", "ExportSpec", "get_family", "available_models"]

_BUILTIN: Dict[str, Type[ModelFamily]] = {"hsemotion": HSEmotionFamily}


@lru_cache(maxsize=1)
def _registry() -> Dict[str, Type[ModelFamily]]:
    families: Dict[str, Type[ModelFamily]] = dict(_BUILTIN)
    try:
        eps = metadata.entry_points(group="online_emotion.model_families")
    except TypeError:  # pragma: no cover
        eps = metadata.entry_points().get("online_emotion.model_families", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            families[ep.name] = ep.load()
        except Exception:
            pass
    return families


def available_models() -> List[str]:
    return sorted(_registry().keys())


def get_family(model: str) -> ModelFamily:
    families = _registry()
    if model not in families:
        raise UnknownModelError(f"unknown model family {model!r}; available: {available_models()}")
    return families[model]()

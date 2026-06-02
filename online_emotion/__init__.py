"""online-emotion-detection — streaming, frame-by-frame emotion recognition.

Lazy public API (importing ``online_emotion`` does not import torch):

    from online_emotion import EmotionRecognizer
    emo = EmotionRecognizer("hsemotion", device="auto")
    out = emo.predict_on_boxes(frame, face_boxes)        # -> EmotionFrameResult
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.5"

__all__ = ["EmotionRecognizer", "EmotionResult", "EmotionFrameResult",
           "available_models", "available_weights", "__version__"]


def __getattr__(name: str):
    if name in ("EmotionRecognizer", "EmotionResult", "EmotionFrameResult"):
        from .recognizer import EmotionFrameResult, EmotionRecognizer, EmotionResult

        return {"EmotionRecognizer": EmotionRecognizer, "EmotionResult": EmotionResult,
                "EmotionFrameResult": EmotionFrameResult}[name]
    if name == "available_models":
        from .families import available_models

        return available_models
    if name == "available_weights":
        from .registry import available_weights

        return available_weights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .recognizer import EmotionFrameResult, EmotionRecognizer, EmotionResult

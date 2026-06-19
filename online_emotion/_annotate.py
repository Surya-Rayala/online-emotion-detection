"""Draw emotion labels onto a frame (install with the ``[client]`` extra).

Torch-free public helper (numpy + opencv via :mod:`.runtime.viz`). Emotion results
carry no boxes (crops are positional — crop ``i`` -> emotion ``i``), so the caller
passes the same ``boxes`` it cropped from (typically the face-detector output).
Works on either an in-process ``EmotionFrameResult`` (whose items expose ``probs``)
or the streaming/remote ``EmotionFrameResult`` mirror (whose items expose
``score``). Returns a new annotated image; the input is not modified.

    from online_emotion import annotate
    canvas = annotate(frame, face_boxes, emotion_result, hud="face -> emotion")
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _score(e) -> float:
    if getattr(e, "score", None) is not None:
        return float(e.score)
    probs = getattr(e, "probs", None)                  # in-process result item
    return float(np.asarray(probs).max()) if probs is not None else 0.0


def annotate(frame: np.ndarray, boxes, result, *, thickness: int = 2,
             hud: Optional[str] = None) -> np.ndarray:
    """Return a copy of ``frame`` with ``boxes[i]`` labelled by ``result``'s emotion i.

    ``boxes`` is (N,4) xyxy in ``frame`` coords; ``result`` is an ``EmotionFrameResult``
    (or any object exposing ``emotions`` with ``.label`` + ``.score``/``.probs``).
    Boxes without a matching emotion are drawn unlabelled. ``hud`` draws a top-left line.
    """
    from .runtime.viz import draw_box_label, draw_hud, hash_color

    img = np.ascontiguousarray(np.asarray(frame).copy())
    emotions = getattr(result, "emotions", result)
    boxes = np.asarray(boxes, dtype="float32").reshape(-1, 4)
    for i, box in enumerate(boxes):
        if i < len(emotions):
            e = emotions[i]
            draw_box_label(img, box, hash_color(e.label), f"{e.label} {_score(e):.2f}",
                           thickness=thickness)
        else:
            draw_box_label(img, box, hash_color(i), None, thickness=thickness)
    if hud:
        draw_hud(img, hud)
    return img

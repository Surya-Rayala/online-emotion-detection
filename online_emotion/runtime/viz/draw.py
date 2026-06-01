"""OpenCV overlay helpers for the optional ``display`` window.

Imported lazily (only when ``display=True``) so headless production installs
never pull in highgui. Colours are stable per-key via HSV hashing.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence, Tuple

import numpy as np

BGR = Tuple[int, int, int]


def hash_color(key: object, s: float = 0.9, v: float = 0.95) -> BGR:
    h = int(hashlib.sha1(str(key).encode("utf-8")).hexdigest(), 16)
    hue = (h % 360) / 360.0
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return int(b * 255), int(g * 255), int(r * 255)


def legible_text_color(bgr: BGR) -> BGR:
    b, g, r = bgr
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0) if y > 160 else (255, 255, 255)


def draw_box_label(img: np.ndarray, xyxy: Sequence[float], color: BGR,
                   text: Optional[str] = None, thickness: int = 2) -> None:
    import cv2

    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    if text:
        ts, tf = 0.5, 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ts, tf)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, ts,
                    legible_text_color(color), tf, cv2.LINE_AA)


def draw_points(img: np.ndarray, points: Sequence[Sequence[float]], color: BGR = (0, 255, 255),
                radius: int = 2) -> None:
    import cv2

    for x, y in points:
        cv2.circle(img, (int(round(x)), int(round(y))), radius, color, -1, lineType=cv2.LINE_AA)


def draw_hud(img: np.ndarray, text: str) -> None:
    """Top-left heads-up line (FPS / stats)."""
    import cv2

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (5, th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

"""Lightweight HTTP client proxy (install with the ``[client]`` extra).

Torch-free (``requests`` + ``numpy`` + ``opencv`` only): talks to an
``online-emotion-serve`` endpoint with the same call shape as the local
``EmotionRecognizer.predict_on_boxes`` so a remote pipeline reads like an
in-process one. Returns its own light result mirror.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ._wire import CT_NPY, encode_image, encode_ndarray


@dataclass(frozen=True)
class EmotionResult:
    label: str
    score: float
    label_index: int
    valence: Optional[float]
    arousal: Optional[float]


@dataclass(frozen=True)
class EmotionFrameResult:
    emotions: List[EmotionResult]
    classes: tuple

    def __len__(self) -> int:
        return len(self.emotions)


class EmotionClient:
    """Remote proxy mirroring ``EmotionRecognizer.predict_on_boxes``."""

    def __init__(self, url: str = "http://127.0.0.1:8002", *, encode: str = "png",
                 timeout: float = 30.0, session: Optional[Any] = None) -> None:
        import requests

        self.url = url.rstrip("/")
        self.encode = encode
        self.timeout = timeout
        self._session = session or requests.Session()

    def healthz(self) -> Dict[str, Any]:
        return self._session.get(f"{self.url}/healthz", timeout=self.timeout).json()

    def meta(self) -> Dict[str, Any]:
        return self._session.get(f"{self.url}/meta", timeout=self.timeout).json()

    def predict_on_boxes(self, frame: np.ndarray, boxes, *,
                         frame_index: Optional[int] = None) -> EmotionFrameResult:
        img, ct = encode_image(np.asarray(frame), self.encode)
        boxes_arr = np.asarray(boxes, dtype="float32").reshape(-1, 4)
        files = {
            "frame": (f"frame.{self.encode}", img, ct),
            "boxes": ("boxes.npy", encode_ndarray(boxes_arr), CT_NPY),
        }
        r = self._session.post(f"{self.url}/predict", files=files, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()["outputs"]
        emotions = [EmotionResult(e["label"], float(e["score"]), int(e["label_index"]),
                                  e.get("valence"), e.get("arousal")) for e in out["emotions"]]
        return EmotionFrameResult(emotions, tuple(out.get("classes", ())))

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "EmotionClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

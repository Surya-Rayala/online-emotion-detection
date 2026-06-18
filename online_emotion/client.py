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

from ._wire import CT_NPY, downscale_to_maxside, encode_image, encode_ndarray


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

    def __init__(self, url: str = "http://127.0.0.1:8002", *, encode: str = "jpeg",
                 quality: int = 90, max_side: Optional[int] = None,
                 timeout: float = 30.0, session: Optional[Any] = None) -> None:
        self.url = url.rstrip("/")
        self.encode = encode
        self.quality = quality
        self.max_side = max_side
        self.timeout = timeout
        if session is None:
            import requests
            session = requests.Session()
        self._session = session

    def healthz(self) -> Dict[str, Any]:
        return self._session.get(f"{self.url}/healthz", timeout=self.timeout).json()

    def meta(self) -> Dict[str, Any]:
        return self._session.get(f"{self.url}/meta", timeout=self.timeout).json()

    def predict_on_boxes(self, frame: np.ndarray, boxes, *,
                         frame_index: Optional[int] = None,
                         max_side: Optional[int] = None) -> EmotionFrameResult:
        frame = np.asarray(frame)
        ms = self.max_side if max_side is None else max_side
        sent, scale = downscale_to_maxside(frame, ms)
        img, ct = encode_image(sent, self.encode, self.quality)
        boxes_arr = np.asarray(boxes, dtype="float32").reshape(-1, 4)
        if scale != 1.0:
            # caller's boxes are in original coords; scale DOWN to index the sent frame.
            boxes_arr = boxes_arr * scale
        files = {
            "frame": (f"frame.{self.encode}", img, ct),
            "boxes": ("boxes.npy", encode_ndarray(boxes_arr), CT_NPY),
        }
        r = self._session.post(f"{self.url}/predict", files=files, timeout=self.timeout)
        r.raise_for_status()
        return self._parse(r.json()["outputs"])

    @staticmethod
    def crop_boxes(frame: np.ndarray, boxes) -> List[np.ndarray]:
        """Crop face regions (HWC BGR) from ``frame`` for the crops-only path.

        Clamp logic mirrors the server-side ``crop_resize`` (runtime/tensor.py) so
        these crops are pixel-equivalent to ``predict_on_boxes`` aside from JPEG.
        """
        frame = np.asarray(frame)
        h, w = frame.shape[:2]
        out: List[np.ndarray] = []
        for b in np.asarray(boxes, dtype="float32").reshape(-1, 4):
            x1 = int(max(0, min(w - 1, b[0])))
            y1 = int(max(0, min(h - 1, b[1])))
            x2 = int(max(x1 + 1, min(w, b[2])))
            y2 = int(max(y1 + 1, min(h, b[3])))
            out.append(np.ascontiguousarray(frame[y1:y2, x1:x2]))
        return out

    def predict_on_crops(self, crops, *, frame_index: Optional[int] = None) -> EmotionFrameResult:
        """Send pre-cropped faces (small) instead of the full frame + boxes.

        Each crop is its own small JPEG part; crop i -> emotion i (positional).
        This is the efficient emotion hop for LAN/remote (payload scales with the
        number/size of faces, not the frame resolution)."""
        files = []
        for i, c in enumerate(crops):
            data, ct = encode_image(np.asarray(c), self.encode, self.quality)
            files.append(("crops", (f"crop{i}.{self.encode}", data, ct)))
        if not files:
            return EmotionFrameResult([], ())
        r = self._session.post(f"{self.url}/predict_crops", files=files, timeout=self.timeout)
        r.raise_for_status()
        return self._parse(r.json()["outputs"])

    def predict_on_boxes_via_crops(self, frame: np.ndarray, boxes, *,
                                   frame_index: Optional[int] = None) -> EmotionFrameResult:
        """Convenience: crop client-side then call the crops-only endpoint."""
        return self.predict_on_crops(self.crop_boxes(frame, boxes), frame_index=frame_index)

    @staticmethod
    def _parse(out: Dict[str, Any]) -> EmotionFrameResult:
        emotions = [EmotionResult(e["label"], float(e["score"]), int(e["label_index"]),
                                  e.get("valence"), e.get("arousal")) for e in out["emotions"]]
        return EmotionFrameResult(emotions, tuple(out.get("classes", ())))

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "EmotionClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

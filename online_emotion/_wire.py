"""Wire helpers for the optional HTTP serving/client layer.

Pure ``numpy`` + ``opencv`` (NO torch): encode/decode the typed payloads of the
project's uniform, modality-generalizable inference contract. Shared by
:mod:`online_emotion.serve` and :mod:`online_emotion.client` so the two stay in
lockstep. Kept torch-free so ``pip install online-emotion-detection[client]`` can
talk to a remote model without the torch stack.

Payload types (declared per input/output via ``/meta``):
  * ``image``   -> encoded image bytes (png/jpeg); content-type ``image/*``
  * ``ndarray`` -> portable ``.npy`` bytes;        content-type ``application/x-npy``
  * ``json``/``scalar`` -> UTF-8 JSON;             content-type ``application/json``
  * ``audio``   -> raw bytes (wav/pcm);            content-type ``audio/*``  (future modality)
"""
from __future__ import annotations

import io
import json
from typing import Any, Optional, Tuple

import numpy as np

CT_NPY = "application/x-npy"
CT_JSON = "application/json"


def encode_image(img: np.ndarray, fmt: str = "png") -> Tuple[bytes, str]:
    import cv2

    ext = ".png" if fmt == "png" else ".jpg"
    ok, buf = cv2.imencode(ext, np.ascontiguousarray(img))
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.tobytes(), ("image/png" if fmt == "png" else "image/jpeg")


def decode_image(data: bytes) -> np.ndarray:
    import cv2

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cv2.imdecode failed")
    return arr  # HWC BGR uint8


def encode_ndarray(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, np.ascontiguousarray(arr), allow_pickle=False)
    return bio.getvalue()


def decode_ndarray(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def decode_part(content_type: Optional[str], data: bytes) -> Any:
    """Decode one multipart part by its content-type (the contract's type tag)."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct.startswith("image/"):
        return decode_image(data)
    if ct == CT_NPY:
        return decode_ndarray(data)
    if ct == CT_JSON:
        return json.loads(data.decode("utf-8"))
    if ct.startswith("audio/"):
        return data
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return data

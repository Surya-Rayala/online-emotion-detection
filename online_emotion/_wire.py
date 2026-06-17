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
CT_NPZ = "application/x-npz"
CT_JSON = "application/json"


def encode_image(img: np.ndarray, fmt: str = "jpeg", quality: int = 90) -> Tuple[bytes, str]:
    import cv2

    arr = np.ascontiguousarray(img)
    if fmt == "png":
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise ValueError("cv2.imencode failed")
        return buf.tobytes(), "image/png"
    q = int(max(1, min(100, quality)))
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.tobytes(), "image/jpeg"


def decode_image(data: bytes) -> np.ndarray:
    import cv2

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cv2.imdecode failed")
    return arr  # HWC BGR uint8


def downscale_to_maxside(img: np.ndarray, max_side: Optional[int]) -> Tuple[np.ndarray, float]:
    """Isotropically shrink ``img`` (HWC) so ``max(H, W) <= max_side``.

    Returns ``(out, scale)`` where ``scale`` maps original->sent coords
    (``sent = orig * scale``; ``orig = sent / scale``). ``scale == 1.0`` means
    the image was returned untouched (no upscaling is ever done). Torch-free.
    """
    if not max_side:
        return img, 1.0
    import cv2

    h, w = img.shape[:2]
    m = max(h, w)
    if m <= int(max_side):
        return img, 1.0
    scale = int(max_side) / float(m)
    out = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                     interpolation=cv2.INTER_AREA)
    return out, scale


def encode_ndarray(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, np.ascontiguousarray(arr), allow_pickle=False)
    return bio.getvalue()


def decode_ndarray(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def encode_npz(**arrays: Any) -> bytes:
    """Pack several named arrays into one ``.npz`` blob (for a binary response that
    avoids JSON ``.tolist()`` boxing of many boxes/landmarks)."""
    bio = io.BytesIO()
    np.savez(bio, **{k: np.asarray(v) for k, v in arrays.items()})
    return bio.getvalue()


def decode_npz(data: bytes) -> dict:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def decode_part(content_type: Optional[str], data: bytes) -> Any:
    """Decode one multipart part by its content-type (the contract's type tag)."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct.startswith("image/"):
        return decode_image(data)
    if ct == CT_NPY:
        return decode_ndarray(data)
    if ct == CT_NPZ:
        return decode_npz(data)
    if ct == CT_JSON:
        return json.loads(data.decode("utf-8"))
    if ct.startswith("audio/"):
        return data
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return data

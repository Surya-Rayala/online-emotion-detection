"""Wire-layer tests: encoding roundtrips, the JPEG quality knob, and (later phases)
client-side downscale coordinate correctness. Pure numpy/cv2 — no torch needed."""
from __future__ import annotations

import numpy as np

from online_emotion._wire import decode_image, decode_npz, encode_image, encode_npz


def _smooth_image(h=240, w=320):
    """A smooth gradient (compresses well with JPEG, so PNG stays larger)."""
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    r = (xx / w * 255).astype("uint8")
    g = (yy / h * 255).astype("uint8")
    b = ((xx + yy) / (h + w) * 255).astype("uint8")
    return np.stack([b, g, r], axis=2)  # BGR


def test_png_roundtrip_is_bit_exact():
    img = _smooth_image()
    data, ct = encode_image(img, "png")
    assert ct == "image/png"
    out = decode_image(data)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert np.array_equal(out, img)


def test_jpeg_roundtrip_shape_dtype_and_low_error():
    img = _smooth_image()
    data, ct = encode_image(img, "jpeg", 90)
    assert ct == "image/jpeg"
    out = decode_image(data)
    assert out.shape == img.shape and out.dtype == np.uint8
    mse = np.mean((out.astype("float32") - img.astype("float32")) ** 2)
    assert mse < 25.0  # q90 on a smooth image is near-lossless


def test_default_is_jpeg():
    img = _smooth_image()
    _, ct = encode_image(img)  # no fmt arg -> new default
    assert ct == "image/jpeg"


def test_quality_knob_orders_byte_sizes():
    img = _smooth_image()
    n_q30 = len(encode_image(img, "jpeg", 30)[0])
    n_q90 = len(encode_image(img, "jpeg", 90)[0])
    n_png = len(encode_image(img, "png")[0])
    assert n_q30 < n_q90 < n_png


def test_quality_is_clamped():
    img = _smooth_image()
    # out-of-range quality must not raise
    assert encode_image(img, "jpeg", 0)[1] == "image/jpeg"
    assert encode_image(img, "jpeg", 1000)[1] == "image/jpeg"


def test_npz_roundtrip():
    boxes = np.arange(12, dtype="float32").reshape(3, 4)
    scores = np.array([0.1, 0.2, 0.3], "float32")
    d = decode_npz(encode_npz(boxes=boxes, scores=scores, shape=(480, 640)))
    assert np.array_equal(d["boxes"], boxes)
    assert np.array_equal(d["scores"], scores)
    assert tuple(d["shape"]) == (480, 640)

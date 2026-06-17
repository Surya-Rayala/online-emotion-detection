"""EmotionClient tests with a fake HTTP session — no network, no torch.

Focus: when the client downscales the frame, the caller's boxes (original coords)
must be scaled DOWN to index the sent frame (inverse of the face direction).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from online_emotion._wire import decode_image, decode_ndarray
from online_emotion.client import EmotionClient


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _CaptureSession:
    """Fake server that captures the posted frame + boxes."""

    def __init__(self):
        self.sent_frame_shape = None
        self.sent_boxes = None

    def post(self, url, files=None, timeout=None, **kw):
        self.sent_frame_shape = decode_image(files["frame"][1]).shape[:2]
        self.sent_boxes = decode_ndarray(files["boxes"][1])
        return _Resp({"outputs": {"emotions": [], "classes": []}, "stats": {}})


def test_no_downscale_sends_boxes_unchanged():
    sess = _CaptureSession()
    client = EmotionClient(session=sess)  # max_side=None
    frame = np.zeros((480, 640, 3), "uint8")
    boxes = np.array([[10, 20, 100, 120]], "float32")
    client.predict_on_boxes(frame, boxes)
    assert sess.sent_frame_shape == (480, 640)
    assert np.allclose(sess.sent_boxes, boxes, atol=1e-4)


def test_downscale_scales_boxes_down():
    sess = _CaptureSession()
    client = EmotionClient(session=sess, max_side=600)
    frame = np.zeros((1200, 1800, 3), "uint8")  # scale = 1/3
    boxes = np.array([[30, 60, 300, 360]], "float32")
    client.predict_on_boxes(frame, boxes)
    assert max(sess.sent_frame_shape) == 600
    assert np.allclose(sess.sent_boxes, boxes / 3.0, atol=1e-3)


# --- Phase C: crops-only path -------------------------------------------------

class _CropCaptureSession:
    def __init__(self):
        self.url = None
        self.files = None

    def post(self, url, files=None, timeout=None, **kw):
        self.url = url
        self.files = files
        return _Resp({"outputs": {
            "emotions": [{"label": "Happy", "score": 0.9, "label_index": 1}],
            "classes": ["Anger", "Happy"]}, "stats": {}})


def test_crop_boxes_clamps_and_shapes():
    frame = np.zeros((100, 120, 3), "uint8")
    crops = EmotionClient.crop_boxes(frame, [[10, 20, 50, 60], [-5, -5, 999, 999]])
    assert crops[0].shape == (40, 40, 3)     # (60-20, 50-10)
    assert crops[1].shape == (100, 120, 3)   # clamped to frame bounds


def test_predict_on_crops_posts_repeated_crops_parts():
    sess = _CropCaptureSession()
    client = EmotionClient(session=sess)
    crops = [np.zeros((40, 40, 3), "uint8"), np.zeros((30, 30, 3), "uint8")]
    res = client.predict_on_crops(crops)
    assert sess.url.endswith("/predict_crops")
    assert len(sess.files) == 2 and all(f[0] == "crops" for f in sess.files)
    assert len(res.emotions) == 1 and res.emotions[0].label == "Happy"


def test_empty_crops_returns_empty_without_posting():
    sess = _CropCaptureSession()
    client = EmotionClient(session=sess)
    res = client.predict_on_crops([])
    assert sess.url is None and len(res.emotions) == 0


class _CountSession:
    """Returns one emotion per crop part; later (bigger) items finish faster."""

    def post(self, url, files=None, timeout=None, **kw):
        n = len(files)
        time.sleep(0.02 * ((8 - n) % 8))
        return _Resp({"outputs": {
            "emotions": [{"label": "x", "score": 0.5, "label_index": 0}] * n,
            "classes": []}})


def test_predict_on_crops_stream_preserves_input_order():
    sess = _CountSession()
    client = EmotionClient(session=sess)
    crop_lists = [[np.zeros((10, 10, 3), "uint8")] * (i + 1) for i in range(6)]
    got = list(client.predict_on_crops_stream(crop_lists, max_workers=3))
    assert [len(r.emotions) for r in got] == [i + 1 for i in range(6)]


@pytest.mark.timeout(300)
def test_crops_equivalence_to_predict_on_boxes():
    """In-process: crops-only labels must equal predict_on_boxes labels."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import online_emotion

    try:
        emo = online_emotion.EmotionRecognizer("hsemotion", device="auto", warmup=False)
    except Exception as e:  # offline / download failure
        pytest.skip(f"weights unavailable: {e}")
    frame = (np.random.rand(480, 640, 3) * 255).astype("uint8")
    boxes = np.array([[50, 40, 200, 260], [300, 100, 520, 360]], "float32")
    via_boxes = [e.label for e in emo.predict_on_boxes(frame, boxes).emotions]
    via_crops = [e.label for e in emo(EmotionClient.crop_boxes(frame, boxes)).emotions]
    assert via_boxes == via_crops
    emo.close()

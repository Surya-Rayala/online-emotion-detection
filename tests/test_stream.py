"""WebSocket /stream loopback: must match /predict_crops for the same crops.

Gated — needs torch + weights + the FastAPI TestClient stack (httpx)."""
from __future__ import annotations

import numpy as np
import pytest


def _test_client_or_skip():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from online_emotion.serve import create_app
    try:
        app = create_app("hsemotion", device="auto")
    except Exception as e:  # offline / download failure
        pytest.skip(f"weights unavailable: {e}")
    return TestClient(app)


@pytest.mark.timeout(300)
def test_ws_stream_matches_predict_crops():
    from online_emotion._wire import encode_image

    client = _test_client_or_skip()
    crops = [(np.random.rand(96, 96, 3) * 255).astype("uint8") for _ in range(2)]
    enc = [encode_image(c, "jpeg", 90)[0] for c in crops]

    files = [("crops", (f"c{i}.jpg", b, "image/jpeg")) for i, b in enumerate(enc)]
    ref = client.post("/predict_crops", files=files).json()["outputs"]

    with client.websocket_connect("/stream") as ws:
        ws.send_json({"n": len(enc)})
        for b in enc:
            ws.send_bytes(b)
        out = ws.receive_json()["outputs"]

    assert [e["label"] for e in out["emotions"]] == [e["label"] for e in ref["emotions"]]
    assert len(out["emotions"]) == 2

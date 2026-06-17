"""Optional HTTP serving adapter (install with the ``[serve]`` extra).

Exposes ``EmotionRecognizer`` behind the project's uniform, modality-generalizable
contract:

  * ``GET  /meta``    self-describing: name, modality, named typed inputs/outputs, classes
  * ``GET  /healthz`` readiness + device/runtime
  * ``POST /predict`` multipart: ``frame`` (image) + ``boxes`` (ndarray) -> JSON emotions

``fastapi``/``uvicorn`` are imported lazily; the recognizer is built eagerly in
:func:`create_app` so first-run export finishes before the port opens.

NOTE: this module deliberately does NOT use ``from __future__ import annotations``.
FastAPI resolves endpoint type hints via ``get_type_hints``; with stringized
annotations it cannot resolve ``Request`` (imported inside ``create_app`` to keep
fastapi optional) and would mistake the param for a query field.
"""
import argparse
from typing import Any, Dict, Optional, Sequence

from . import __version__

_INPUTS = [
    {"name": "frame", "type": "image", "required": True},
    {"name": "boxes", "type": "ndarray", "required": True},
]
_OUTPUTS = [{"name": "emotions", "type": "json"}]


def create_app(model: str = "hsemotion", *, weights=None, runtime: str = "auto",
               device: str = "auto", precision: str = "auto", batch_max: int = 32,
               input_size=None):
    """Build a FastAPI app wrapping one eagerly-constructed ``EmotionRecognizer``."""
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    import numpy as np

    from . import _wire
    from .recognizer import EmotionRecognizer

    emo = EmotionRecognizer(model, weights=weights, runtime=runtime, device=device,
                            precision=precision, batch_max=batch_max, input_size=input_size,
                            warmup=True)

    app = FastAPI(title="online-emotion-detection", version=__version__)

    def _meta() -> Dict[str, Any]:
        cfg = emo.config
        return {"name": "online_emotion", "modality": "vision", "model": model,
                "runtime": cfg.runtime, "device": cfg.device,
                "inputs": _INPUTS, "outputs": _OUTPUTS, "classes": list(emo.classes),
                # /predict_crops takes pre-cropped face images directly (one repeated
                # "crops" image part each) — avoids sending the full frame a second time.
                "paths": ["/predict", "/predict_crops"]}

    def _emotions_payload(res) -> list:
        return [{"label": e.label, "score": float(e.probs.max()),
                 "label_index": int(e.label_index),
                 "valence": (None if e.valence is None else float(e.valence)),
                 "arousal": (None if e.arousal is None else float(e.arousal))}
                for e in res.emotions]

    @app.get("/meta")
    def meta() -> Dict[str, Any]:
        return _meta()

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        m = _meta()
        m["ready"] = True
        m["mps"] = emo.device.startswith("mps")
        return m

    @app.post("/predict")
    async def predict(request: Request):
        form = await request.form()
        inputs: Dict[str, Any] = {}
        for key, val in form.items():
            if hasattr(val, "read"):
                inputs[key] = _wire.decode_part(getattr(val, "content_type", None), await val.read())
            else:
                inputs[key] = _wire.decode_part(_wire.CT_JSON, str(val).encode("utf-8"))
        if "frame" not in inputs:
            return JSONResponse({"error": "missing required input 'frame'"}, status_code=422)
        boxes = np.asarray(inputs.get("boxes", []), dtype="float32").reshape(-1, 4)
        res = emo.predict_on_boxes(inputs["frame"], boxes)
        return {"outputs": {"emotions": _emotions_payload(res), "classes": list(res.classes)},
                "stats": emo.stats.as_dict()}

    @app.post("/predict_crops")
    async def predict_crops(request: Request):
        """Pre-cropped faces in (one repeated ``crops`` image part each) -> emotions.

        Lets an orchestrator crop client-side and send only the small face regions
        instead of the whole frame a second time. Crop i -> emotion i (positional).
        """
        form = await request.form()
        crops = [_wire.decode_part(getattr(p, "content_type", None), await p.read())
                 for p in form.getlist("crops") if hasattr(p, "read")]
        if not crops:
            return {"outputs": {"emotions": [], "classes": list(emo.classes)},
                    "stats": emo.stats.as_dict()}
        res = emo(crops)
        return {"outputs": {"emotions": _emotions_payload(res), "classes": list(res.classes)},
                "stats": emo.stats.as_dict()}

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        """Persistent crops path: per frame, a JSON control message ``{"n": K}``
        then K binary crop messages; the reply is one JSON ``{"outputs": {...}}``.
        FIFO replies; inference runs in a threadpool so it never blocks the loop."""
        await ws.accept()
        try:
            while True:
                ctrl = await ws.receive_json()
                k = int(ctrl.get("n", 0))
                crops = [_wire.decode_image(await ws.receive_bytes()) for _ in range(k)]
                if crops:
                    res = await run_in_threadpool(emo, crops)
                    payload = {"emotions": _emotions_payload(res), "classes": list(res.classes)}
                else:
                    payload = {"emotions": [], "classes": list(emo.classes)}
                await ws.send_json({"outputs": payload})
        except WebSocketDisconnect:
            return

    return app


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser("online-emotion-serve",
                                description="Serve HSEmotion recognition over HTTP (uniform contract).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--model", default="hsemotion")
    p.add_argument("--weights", default=None)
    p.add_argument("--runtime", default="auto", choices=["auto", "torch", "torchscript", "onnx", "trt"])
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "int8"])
    p.add_argument("--batch-max", type=int, default=32)
    p.add_argument("--input-size", type=int, default=None)
    args = p.parse_args(argv)

    import uvicorn

    app = create_app(args.model, weights=args.weights, runtime=args.runtime, device=args.device,
                     precision=args.precision, batch_max=args.batch_max, input_size=args.input_size)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

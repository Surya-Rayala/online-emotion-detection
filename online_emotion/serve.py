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
import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union

from . import __version__

_INPUTS = [
    {"name": "frame", "type": "image", "required": True},
    {"name": "boxes", "type": "ndarray", "required": True},
]
_OUTPUTS = [{"name": "emotions", "type": "json"}]


def _parse_instances(spec) -> Union[int, Dict[int, int]]:
    """Parse an ``--instances`` value into a plain int N or a ``{gpu_index: count}`` map.

    Accepts an int (``4``), a bare number string (``"4"``), a dict (``{0: 2, 1: 1}``),
    or a per-GPU spec string (``"0=2,1=1"`` / ``"cuda:0=2,cuda:1=1"``). Raises
    ``ValueError`` on anything malformed so the caller can warn and fall back."""
    if isinstance(spec, bool):                          # guard: bool is an int subclass
        raise ValueError("bool")
    if isinstance(spec, int):
        return spec
    if isinstance(spec, dict):
        return {int(k): int(v) for k, v in spec.items()}
    s = str(spec).strip()
    if not s:
        raise ValueError("empty")
    if "=" not in s:
        return int(s)                                   # bare number (may raise ValueError)
    out: Dict[int, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition("=")
        key = key.strip().lower().replace("cuda:", "")
        out[int(key)] = int(val)
    if not out:
        raise ValueError("no entries")
    return out


def _resolve_instance_devices(spec, device) -> List[str]:
    """Resolve ``--instances`` into a per-instance torch device list.

    Warns and degrades gracefully on bad input: an unparseable spec or a non-CUDA
    box asked for per-GPU placement falls back sensibly, and missing GPU indices
    are skipped. Always returns at least one device."""
    from .runtime.device import resolve_device
    from .runtime.logging import get_logger

    log = get_logger("serve")
    resolved = resolve_device(device)
    try:
        parsed = _parse_instances(spec)
    except (ValueError, TypeError):
        log.warning("could not parse --instances %r; using 1", spec)
        return [resolved]
    try:
        import torch

        cuda_count = int(torch.cuda.device_count())
    except Exception:
        cuda_count = 0

    if isinstance(parsed, int):                         # plain N, auto-placed
        if parsed <= 0:
            log.warning("--instances=%r is not positive; using 1", spec)
            return [resolved]
        if resolved.startswith("cuda") and cuda_count > 1:
            return [f"cuda:{i % cuda_count}" for i in range(parsed)]   # round-robin GPUs
        if parsed > 1:
            log.warning("%d instances share %s (they time-share compute); multi-instance "
                        "mainly helps across multiple GPUs", parsed, resolved)
        return [resolved] * parsed

    if cuda_count == 0:                                 # per-GPU map but no CUDA
        total = sum(c for c in parsed.values() if c > 0) or 1
        log.warning("per-GPU --instances %r needs CUDA (device=%s); placing all %d on %s",
                    spec, resolved, total, resolved)
        return [resolved] * total

    devices: List[str] = []
    for idx in sorted(parsed):
        count = parsed[idx]
        if count <= 0:
            continue
        if idx >= cuda_count:
            log.warning("cuda:%d not found (only %d GPU(s) present); skipping its %d instance(s)",
                        idx, cuda_count, count)
            continue
        devices.extend([f"cuda:{idx}"] * count)
    if not devices:
        log.warning("--instances=%r selected no valid GPUs; using 1 on %s", spec, resolved)
        return [resolved]
    return devices


class _Pool:
    """Fixed-size checkout pool of model instances.

    ``run`` borrows a free instance, runs ``fn(instance)`` in a worker thread (so the
    event loop keeps serving and concurrent requests overlap decode/transfer with
    compute), and returns it. One instance == today's behavior plus a threadpool hop;
    N>1 lets N requests run at once (true parallelism only across distinct GPUs). The
    count is fixed by the operator via ``--instances``; there is no auto-regulation."""

    def __init__(self, instances) -> None:
        self.instances = list(instances)
        self._free: asyncio.Queue = asyncio.Queue()
        for inst in self.instances:
            self._free.put_nowait(inst)

    def __len__(self) -> int:
        return len(self.instances)

    async def run(self, fn):
        from starlette.concurrency import run_in_threadpool

        inst = await self._free.get()
        try:
            return await run_in_threadpool(fn, inst)
        finally:
            self._free.put_nowait(inst)


def create_app(model: str = "hsemotion", *, weights=None, runtime: str = "auto",
               device: str = "auto", precision: str = "auto", batch_max: int = 32,
               input_size=None, stream_queue: int = 32, instances: Union[int, str, dict] = 1):
    """Build a FastAPI app wrapping a pool of ``instances`` ``EmotionRecognizer``s.

    ``instances`` defaults to 1 (one model, today's behavior). It accepts an int,
    a ``"0=2,1=1"`` per-GPU spec string, or a ``{gpu_index: count}`` dict; the pool
    pins one model per GPU on a multi-GPU box (else N copies share one device) and
    dispatches requests across them through a checkout pool."""
    import time

    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    import numpy as np

    from . import _wire
    from .recognizer import EmotionRecognizer

    inst_devices = _resolve_instance_devices(instances, device)
    emos = [EmotionRecognizer(model, weights=weights, runtime=runtime, device=d,
                              precision=precision, batch_max=batch_max, input_size=input_size,
                              warmup=True)
            for d in inst_devices]
    emo = emos[0]                                  # config/classes identical across instances
    pool = _Pool(emos)

    app = FastAPI(title="online-emotion-detection", version=__version__)

    def _meta() -> Dict[str, Any]:
        cfg = emo.config
        return {"name": "online_emotion", "modality": "vision", "model": model,
                "runtime": cfg.runtime, "device": cfg.device,
                "instances": len(pool), "instance_devices": inst_devices,
                "inputs": _INPUTS, "outputs": _OUTPUTS, "classes": list(emo.classes),
                # /predict_crops takes pre-cropped face images directly (one repeated
                # "crops" image part each) — avoids sending the full frame a second time.
                "paths": ["/predict", "/predict_crops"], "stream_protocol": 2}

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
        frame = inputs["frame"]
        res, stats = await pool.run(lambda e: (e.predict_on_boxes(frame, boxes), e.stats.as_dict()))
        return {"outputs": {"emotions": _emotions_payload(res), "classes": list(res.classes)},
                "stats": stats}

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
        res, stats = await pool.run(lambda e: (e(crops), e.stats.as_dict()))
        return {"outputs": {"emotions": _emotions_payload(res), "classes": list(res.classes)},
                "stats": stats}

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        """Stream protocol v2 — pipelined, id-tagged, with server telemetry.

        Per frame the client sends a text control message ``{"id", "n":K, "fmt"}``
        then K binary crop messages. The reply is one JSON ``{"id", "outputs",
        "server": {infer_ms, queue_depth, t_recv, t_send}}``. A receive loop drains
        into a bounded queue (backpressure) while a worker runs inference in a
        threadpool. Replies carry the id, so the client matches without ordering."""
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(stream_queue)))
        n_instances = len(pool)
        send_lock = asyncio.Lock()                          # serialize replies across N workers

        async def receiver():
            while True:
                ctrl = await ws.receive_json()              # {"id", "n", "fmt"}
                k = int(ctrl.get("n", 0))
                crops = [_wire.decode_image(await ws.receive_bytes()) for _ in range(k)]
                await q.put((ctrl, crops, time.perf_counter()))

        async def worker(emo_i):                            # one task per pooled instance
            while True:
                ctrl, crops, t_recv = await q.get()
                try:
                    t0 = time.perf_counter()
                    if crops:
                        res = await run_in_threadpool(emo_i, crops)
                        payload = {"emotions": _emotions_payload(res), "classes": list(res.classes)}
                    else:
                        payload = {"emotions": [], "classes": list(emo_i.classes)}
                    infer_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    async with send_lock:
                        await ws.send_json({"id": ctrl.get("id"), "outputs": payload,
                                            "server": {"infer_ms": infer_ms, "queue_depth": q.qsize(),
                                                       "instances": n_instances,
                                                       "t_recv": t_recv, "t_send": time.perf_counter()}})
                finally:
                    q.task_done()

        tasks = [asyncio.ensure_future(receiver())]
        tasks += [asyncio.ensure_future(worker(e)) for e in pool.instances]
        try:
            await asyncio.gather(*tasks)
        except WebSocketDisconnect:
            pass
        finally:
            for t in tasks:
                t.cancel()

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
    p.add_argument("--instances", default="1",
                   help="server-side inference instances: a number ('4') or per-GPU map "
                        "('cuda:0=2,cuda:1=1'). On one device N copies just time-share it (N x memory); "
                        "running more than one instance mainly helps across multiple GPUs")
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "int8"])
    p.add_argument("--batch-max", type=int, default=32)
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--stream-queue", type=int, default=32,
                   help="max frames buffered per /stream connection (backpressure bound)")
    args = p.parse_args(argv)

    import uvicorn

    app = create_app(args.model, weights=args.weights, runtime=args.runtime, device=args.device,
                     precision=args.precision, batch_max=args.batch_max, input_size=args.input_size,
                     stream_queue=args.stream_queue, instances=args.instances)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

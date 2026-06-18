"""Long-lived async streaming session (install with the ``[client]`` extra).

One ``EmotionStream`` holds WebSocket connections across a **pool of endpoints**,
lets you **push** face crops (each with arbitrary metadata) without buffering, and
yields results **as they complete** (out of order) tagged with your metadata. An
adaptive controller (see :mod:`._autoscale`) ramps in-flight concurrency and the
number of connections toward a target fps/latency.

Sends only small face crops (never the full frame), so payload scales with the
number/size of faces. Torch-free: ``asyncio`` + ``websockets`` + numpy/opencv.

    async with EmotionStream(["http://gpu0:8002"], target_fps=30, max_side=None) as s:
        async def pump():
            for frame, boxes, info in source():
                await s.push_boxes(frame, boxes, meta=info)   # crops client-side
            await s.aclose()
        asyncio.create_task(pump())
        async for result, info in s.results():                # out of order
            handle(info, result)
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Optional, Sequence, Tuple

import numpy as np

from ._autoscale import AutoScaler
from ._wire import encode_image
from .client import EmotionClient, EmotionFrameResult

_SENTINEL = object()


def _ws_url(url: str) -> str:
    u = url.rstrip("/")
    u = u.replace("https://", "wss://").replace("http://", "ws://")
    return u + "/stream"


class EmotionStream:
    def __init__(self, urls, *, target_fps: Optional[float] = None,
                 target_latency_ms: Optional[float] = None,
                 encode: str = "jpeg", quality: int = 90,
                 min_inflight: int = 1, max_inflight: int = 64, max_queue: int = 256,
                 max_connections: Optional[int] = None, open_timeout: float = 30.0,
                 tick_s: float = 0.5, connect=None) -> None:
        self._urls = [urls] if isinstance(urls, str) else list(urls)
        if not self._urls:
            raise ValueError("EmotionStream needs at least one URL")
        self.encode = encode
        self.quality = quality
        self.open_timeout = open_timeout
        self.tick_s = tick_s
        self._connect = connect
        self._scaler = AutoScaler(
            target_fps=target_fps, target_latency_ms=target_latency_ms,
            min_inflight=min_inflight, max_inflight=max_inflight,
            max_connections=max_connections or max(1, len(self._urls)))
        self._max_queue = max_queue
        self._in_q: Optional[asyncio.Queue] = None
        self._out_q: Optional[asyncio.Queue] = None
        self._cap: Optional[asyncio.Condition] = None
        self._conns: list = []
        self._ctrl_task = None
        self._pending: dict = {}
        self._inflight = 0
        self._next_id = 0
        self._started = False
        self._closing = False

    # -- lifecycle ---------------------------------------------------------
    async def _start(self) -> None:
        if self._started:
            return
        self._in_q = asyncio.Queue(maxsize=self._max_queue)
        self._out_q = asyncio.Queue()
        self._cap = asyncio.Condition()
        await self._open_conn(self._urls[0])
        self._ctrl_task = asyncio.ensure_future(self._controller())
        self._started = True

    async def __aenter__(self) -> "EmotionStream":
        await self._start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self, drain_timeout: float = 10.0) -> None:
        if not self._started or self._closing:
            return
        deadline = time.perf_counter() + drain_timeout
        # 1) let the senders flush everything already queued (don't stop them yet)
        while self._in_q is not None and not self._in_q.empty() and time.perf_counter() < deadline:
            await asyncio.sleep(0.01)
        # 2) drain in-flight replies; bail if it stalls (~1s) so a straggler can't hang close
        last, stall = len(self._pending), 0
        while self._pending and time.perf_counter() < deadline:
            await asyncio.sleep(0.05)
            if len(self._pending) == last:
                stall += 1
            else:
                last, stall = len(self._pending), 0
            if stall >= 20:
                break
        # 3) now stop senders/receivers and signal end-of-results
        self._closing = True
        async with self._cap:
            self._cap.notify_all()
        for _ in self._conns:
            await self._in_q.put(None)            # unblock senders parked on get()
        if self._ctrl_task:
            self._ctrl_task.cancel()
        for c in list(self._conns):
            await self._close_conn(c)
        await self._out_q.put(_SENTINEL)

    # -- public API --------------------------------------------------------
    async def push_crops(self, crops: Sequence[np.ndarray], meta: Any = None) -> None:
        """Enqueue a list of pre-cut face crops (+ opaque metadata)."""
        if self._closing:
            raise RuntimeError("EmotionStream is closing")
        await self._start()
        enc = [encode_image(np.asarray(c), self.encode, self.quality)[0] for c in crops]
        fid = self._next_id
        self._next_id += 1
        self._pending[fid] = {"meta": meta, "t_send": None}
        control = json.dumps({"id": fid, "n": len(enc), "fmt": self.encode})
        await self._in_q.put((fid, control, enc))

    async def push_boxes(self, frame: np.ndarray, boxes, meta: Any = None) -> None:
        """Convenience: crop ``boxes`` from ``frame`` client-side, then push crops."""
        await self.push_crops(EmotionClient.crop_boxes(frame, boxes), meta=meta)

    async def results(self) -> AsyncIterator[Tuple[EmotionFrameResult, Any]]:
        """Yield ``(EmotionFrameResult, meta)`` as each frame completes — out of order."""
        await self._start()
        while True:
            item = await self._out_q.get()
            if item is _SENTINEL:
                return
            yield item

    def stats(self) -> dict:
        sc = self._scaler
        return {"conns": len(self._conns), "target_inflight": sc.target_inflight,
                "inflight": self._inflight, "queue": (self._in_q.qsize() if self._in_q else 0),
                "bound": sc.bound, "rtt_ms": round(sc._rtt.get(), 2),
                "srv_ms": round(sc._srv.get(), 2), "infer_ms": round(sc._infer.get(), 2),
                "queue_depth": round(sc._qd.get(), 2)}

    # -- capacity gate -----------------------------------------------------
    async def _acquire(self) -> bool:
        async with self._cap:
            await self._cap.wait_for(
                lambda: self._closing or self._inflight < self._scaler.target_inflight)
            if self._closing:
                return False
            self._inflight += 1
            return True

    async def _release(self) -> None:
        async with self._cap:
            self._inflight = max(0, self._inflight - 1)
            self._cap.notify(1)

    # -- connections -------------------------------------------------------
    async def _open_conn(self, url: str) -> None:
        ws = await self._dial(_ws_url(url))
        conn = {"ws": ws, "url": url, "tasks": []}
        conn["tasks"] = [asyncio.ensure_future(self._sender(conn)),
                         asyncio.ensure_future(self._receiver(conn))]
        self._conns.append(conn)

    async def _dial(self, ws_url: str):
        if self._connect is not None:
            return await self._connect(ws_url)
        import websockets
        return await websockets.connect(ws_url, open_timeout=self.open_timeout, max_size=None)

    async def _close_conn(self, conn: dict) -> None:
        if conn in self._conns:
            self._conns.remove(conn)
        for t in conn["tasks"]:
            t.cancel()
        try:
            await conn["ws"].close()
        except Exception:
            pass

    async def _sender(self, conn: dict) -> None:
        ws = conn["ws"]
        try:
            while not self._closing:
                item = await self._in_q.get()
                if item is None:
                    break
                if not await self._acquire():
                    break
                fid, control, crops = item
                if fid in self._pending:
                    self._pending[fid]["t_send"] = time.perf_counter()
                await ws.send(control)
                for c in crops:
                    await ws.send(c)
        except asyncio.CancelledError:
            pass
        except Exception:
            return

    async def _receiver(self, conn: dict) -> None:
        ws = conn["ws"]
        try:
            while not self._closing:
                msg = await ws.recv()
                now = time.perf_counter()
                rep = json.loads(msg)
                info = self._pending.pop(rep.get("id"), None)
                await self._release()
                if info is None:
                    continue
                result = EmotionClient._parse(rep["outputs"])
                srv = rep.get("server", {})
                t_send = info.get("t_send") or now
                self._scaler.observe(
                    rtt_ms=(now - t_send) * 1000.0,
                    srv_ms=(srv.get("t_send", 0.0) - srv.get("t_recv", 0.0)) * 1000.0,
                    infer_ms=srv.get("infer_ms", 0.0), queue_depth=srv.get("queue_depth", 0.0))
                await self._out_q.put((result, info["meta"]))
        except asyncio.CancelledError:
            pass
        except Exception:
            return

    async def _controller(self) -> None:
        try:
            while not self._closing:
                await asyncio.sleep(self.tick_s)
                st = self._scaler.tick(self.tick_s)
                async with self._cap:
                    self._cap.notify_all()
                while len(self._conns) < st.n_conn:
                    await self._open_conn(self._urls[len(self._conns) % len(self._urls)])
                while len(self._conns) > st.n_conn and len(self._conns) > 1:
                    await self._close_conn(self._conns[-1])
        except asyncio.CancelledError:
            pass

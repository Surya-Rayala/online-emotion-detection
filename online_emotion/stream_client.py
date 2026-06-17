"""Persistent WebSocket streaming client (install with the ``[client]`` extra).

Keeps a warm connection to an ``online-emotion-serve`` ``/stream`` endpoint and
pipelines up to ``max_inflight`` frames so RTT overlaps encode + inference — the
right shape for LAN/remote. Sends only small face *crops* (not the full frame),
so payload scales with the number/size of faces. Torch-free.

Wire per frame: a JSON control message ``{"n": K}`` then K binary crop messages;
one JSON reply per frame. Replies are FIFO (one sequential model), matched by
send order.
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

from .client import EmotionClient, EmotionFrameResult


class EmotionStreamClient:
    def __init__(self, url: str = "http://127.0.0.1:8002", *, encode: str = "jpeg",
                 quality: int = 90, max_inflight: int = 4, open_timeout: float = 30.0) -> None:
        self.ws_url = (url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
                       + "/stream")
        self.encode = encode
        self.quality = quality
        self.max_inflight = max(1, int(max_inflight))
        self.open_timeout = open_timeout

    def predict_stream(self, crop_lists: Iterable[Sequence[np.ndarray]]) -> Iterator[EmotionFrameResult]:
        """Yield one EmotionFrameResult per input crop-list, in input order."""
        from ._wire import encode_image
        from websockets.sync.client import connect

        _DONE = object()
        with connect(self.ws_url, open_timeout=self.open_timeout, max_size=None) as ws:
            sem = threading.Semaphore(self.max_inflight)
            pending: "queue.Queue[Any]" = queue.Queue()
            err: list = []

            def sender():
                try:
                    for crops in crop_lists:
                        crops = list(crops)
                        sem.acquire()
                        ws.send(json.dumps({"n": len(crops)}))      # text control
                        for c in crops:
                            data, _ = encode_image(np.asarray(c), self.encode, self.quality)
                            ws.send(data)                            # binary crop
                        pending.put(True)
                except Exception as e:
                    err.append(e)
                finally:
                    pending.put(_DONE)

            t = threading.Thread(target=sender, daemon=True)
            t.start()
            while True:
                item = pending.get()
                if item is _DONE:
                    break
                reply = ws.recv()                                    # FIFO
                sem.release()
                yield EmotionClient._parse(json.loads(reply)["outputs"])
            t.join()
            if err:
                raise err[0]

    def predict_on_boxes_stream(
        self, items: Iterable[Tuple[np.ndarray, Any]]
    ) -> Iterator[EmotionFrameResult]:
        """Convenience: stream ``(frame, boxes)`` pairs; crops are cut client-side."""
        return self.predict_stream(EmotionClient.crop_boxes(f, b) for f, b in items)

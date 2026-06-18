"""EmotionStream async session via a fake WebSocket transport (no server, no torch).

Uses ``asyncio.run`` so no pytest-asyncio plugin is needed."""
from __future__ import annotations

import asyncio
import json

import numpy as np

from online_emotion.aio import EmotionStream


def _reply(fid, n=1):
    return json.dumps({
        "id": fid,
        "outputs": {"emotions": [{"label": "Happy", "score": 0.9, "label_index": 1}] * n,
                    "classes": ["Anger", "Happy"]},
        "server": {"infer_ms": 1.0, "queue_depth": 0, "t_recv": 0.0, "t_send": 0.001}})


class _FakeConn:
    """Reads a control msg ``{"id","n"}`` then n binary crops, replies once."""

    def __init__(self, reorder=False):
        self._ctrl = None
        self._left = 0
        self._replies = asyncio.Queue()
        self._reorder = reorder

    async def send(self, data):
        if isinstance(data, (bytes, bytearray)):
            self._left -= 1
            if self._left <= 0:                      # last crop -> reply
                fid, n = self._ctrl["id"], self._ctrl["n"]
                if self._reorder and fid % 2 == 0:
                    asyncio.ensure_future(self._delayed(fid, n))
                else:
                    await self._replies.put(_reply(fid, n))
        else:
            self._ctrl = json.loads(data)
            self._left = int(self._ctrl["n"])
            if self._left == 0:                      # zero-crop frame replies immediately
                await self._replies.put(_reply(self._ctrl["id"], 0))

    async def _delayed(self, fid, n):
        await asyncio.sleep(0.03)
        await self._replies.put(_reply(fid, n))

    async def recv(self):
        return await self._replies.get()

    async def close(self):
        pass


def test_results_roundtrip_preserves_metadata():
    async def main():
        async def connect(url):
            return _FakeConn(reorder=True)
        got = []
        async with EmotionStream("http://x", connect=connect, tick_s=0.05) as s:
            for i in range(12):
                crops = [np.zeros((10, 10, 3), "uint8")] * ((i % 3) + 1)
                await s.push_crops(crops, meta=i)
            async for res, meta in s.results():
                got.append(meta)
                assert len(res.emotions) == (meta % 3) + 1
                if len(got) == 12:
                    break
        return sorted(got)

    assert asyncio.run(main()) == list(range(12))

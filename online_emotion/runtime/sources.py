"""Unified frame sources: static file, live stream (webcam / RTSP / HTTP), and
a *simulated* live stream that paces a file at real time and drops to the
latest frame — so a plain video can be exercised through the streaming path
without any external server.

``open_source`` auto-detects: an int / digit string is a webcam; an
``rtsp|http|https|udp`` URL is a live stream; an existing path is a file
(promoted to a simulated stream when ``is_stream=True``).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np

from .errors import SourceError

SourceSpec = Union[str, int, "Path"]


@dataclass
class FrameRef:
    """A single frame plus its bookkeeping; carries the pixels for downstream crops."""

    image: np.ndarray            # BGR HWC (OpenCV native)
    index: int                   # running frame index from the source
    timestamp: float             # seconds since source start


def _looks_like_stream_url(s: str) -> bool:
    return s.lower().split("://", 1)[0] in ("rtsp", "http", "https", "udp", "tcp", "rtmp") and "://" in s


class Source:
    """Base class. Iterating yields :class:`FrameRef`."""

    is_stream: bool = False

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> Tuple[int, int]:
        return self._size  # (W, H)

    @property
    def frame_count(self) -> Optional[int]:
        return self._frame_count

    @property
    def dropped(self) -> int:
        return 0

    def __iter__(self) -> Iterator[FrameRef]:  # pragma: no cover - overridden
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover
        pass

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class FileSource(Source):
    """Decode every frame of a video file in order (no dropping)."""

    is_stream = False

    def __init__(self, path: SourceSpec, max_frames: Optional[int] = None) -> None:
        import cv2

        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise SourceError(f"could not open video file: {self.path}")
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self._size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._frame_count = min(n, max_frames) if (max_frames and n) else (max_frames or (n or None))
        self._max = max_frames

    def __iter__(self) -> Iterator[FrameRef]:
        i = 0
        t0 = time.perf_counter()
        while True:
            if self._max is not None and i >= self._max:
                break
            ok, frame = self.cap.read()
            if not ok:
                break
            yield FrameRef(image=frame, index=i, timestamp=time.perf_counter() - t0)
            i += 1

    def release(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None


class _LatestGrabber:
    """Background reader keeping only the most recent frame (drop-to-latest)."""

    def __init__(self, cap, fps: float, realtime: bool, loop: bool) -> None:
        import cv2  # noqa: F401

        self.cap = cap
        self.fps = fps
        self.realtime = realtime
        self.loop = loop
        self._lock = threading.Lock()
        self._frame = None
        self._fresh = False
        self._produced = 0
        self._dropped = 0
        self._ended = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import cv2

        period = (1.0 / self.fps) if (self.realtime and self.fps > 0) else 0.0
        next_t = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            with self._lock:
                if self._fresh:  # previous frame never consumed -> it is dropped
                    self._dropped += 1
                self._frame = frame
                self._fresh = True
                self._produced += 1
            if period:
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
        with self._lock:
            self._ended = True

    def get(self, poll: float = 0.001):
        while True:
            with self._lock:
                if self._fresh:
                    self._fresh = False
                    return self._frame
                if self._ended:
                    return None
            time.sleep(poll)

    @property
    def dropped(self) -> int:
        return self._dropped

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class StreamSource(Source):
    """Live stream (webcam index / RTSP / HTTP). Keeps only the latest frame."""

    is_stream = True

    def __init__(self, src: SourceSpec, max_frames: Optional[int] = None,
                 realtime: bool = False, loop: bool = False) -> None:
        import cv2

        self.src = int(src) if (isinstance(src, int) or str(src).isdigit()) else str(src)
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise SourceError(f"could not open stream source: {self.src}")
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self._size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._frame_count = None
        self._max = max_frames
        self._realtime = realtime
        self._loop = loop
        self._grabber: Optional[_LatestGrabber] = None

    def __iter__(self) -> Iterator[FrameRef]:
        self._grabber = _LatestGrabber(self.cap, self._fps, self._realtime, self._loop)
        i = 0
        t0 = time.perf_counter()
        while True:
            if self._max is not None and i >= self._max:
                break
            frame = self._grabber.get()
            if frame is None:
                break
            yield FrameRef(image=frame, index=i, timestamp=time.perf_counter() - t0)
            i += 1

    @property
    def dropped(self) -> int:
        return self._grabber.dropped if self._grabber else 0

    def release(self) -> None:
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber = None
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None


class SimulatedStreamSource(StreamSource):
    """A file replayed as a live stream: real-time pacing + drop-to-latest.

    This is how a static test video is "converted to streaming" for testing —
    if the model is slower than the video's FPS, frames are dropped exactly as a
    real camera feed would force.
    """

    def __init__(self, path: SourceSpec, max_frames: Optional[int] = None, loop: bool = False) -> None:
        super().__init__(path, max_frames=max_frames, realtime=True, loop=loop)


def open_source(source: SourceSpec, *, is_stream: Optional[bool] = None,
                max_frames: Optional[int] = None, loop: bool = False) -> Source:
    """Factory with auto-detection (override with ``is_stream``)."""
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return StreamSource(source, max_frames=max_frames, loop=loop)  # webcam
    s = str(source)
    if _looks_like_stream_url(s):
        return StreamSource(s, max_frames=max_frames, loop=loop)
    # a path on disk
    if is_stream:
        return SimulatedStreamSource(s, max_frames=max_frames, loop=loop)
    if not Path(s).exists():
        raise SourceError(f"source not found: {s}")
    return FileSource(s, max_frames=max_frames)

"""Self-contained inference runtime for online_face.

Device resolution, the torch/torchscript/onnx/trt backends, the export +
artifact cache, streaming sources, overlays, timing, and the StreamingModel
base. This package owns its runtime (no cross-package dependency); the layout
mirrors the sibling emotion package by convention.
"""
from __future__ import annotations

from .cache import ArtifactCache, ArtifactRef, default_cache_dir
from .config import InferenceConfig
from .device import DeviceInfo, device_info, resolve_device, resolve_precision
from .errors import (
    DeviceUnavailableError, ExportError, OnlineInferenceError, RuntimeUnavailableError,
    SourceError, UnknownModelError, UnknownWeightsError, WeightsNotAvailableError,
)
from .sources import FileSource, FrameRef, SimulatedStreamSource, Source, StreamSource, open_source
from .streaming import StreamingModel
from .tensor import LetterboxMeta, crop_resize, letterbox, load_frame, rescale_boxes, rescale_points
from .timing import RunStats, Stopwatch

__all__ = [
    "ArtifactCache", "ArtifactRef", "default_cache_dir",
    "InferenceConfig",
    "DeviceInfo", "device_info", "resolve_device", "resolve_precision",
    "OnlineInferenceError", "UnknownModelError", "UnknownWeightsError",
    "WeightsNotAvailableError", "RuntimeUnavailableError", "ExportError",
    "DeviceUnavailableError", "SourceError",
    "FrameRef", "Source", "FileSource", "StreamSource", "SimulatedStreamSource", "open_source",
    "StreamingModel",
    "LetterboxMeta", "load_frame", "letterbox", "rescale_boxes", "rescale_points", "crop_resize",
    "RunStats", "Stopwatch",
]

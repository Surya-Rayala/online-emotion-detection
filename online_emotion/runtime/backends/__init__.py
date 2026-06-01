"""Backend registry, runtime auto-selection, and artifact loading.

``resolve_runtime`` answers "given this device, what should ``auto`` mean?".
``load_artifact_backend`` turns a serialized artifact (or, for torch, an eager
module supplied elsewhere) into a ready :class:`Backend`. The export-then-load
orchestration lives in :mod:`..streaming` to keep this module dependency-light.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Backend
from .torch_backend import TorchBackend
from .torchscript_backend import TorchScriptBackend
from .onnx_backend import OnnxBackend
from .trt_backend import TensorRTBackend
from ..device import device_info

__all__ = [
    "Backend", "TorchBackend", "TorchScriptBackend", "OnnxBackend", "TensorRTBackend",
    "resolve_runtime", "runtime_from_suffix", "load_artifact_backend",
]

_SUFFIX_TO_RUNTIME = {
    ".torchscript": "torchscript",
    ".ts": "torchscript",
    ".onnx": "onnx",
    ".engine": "trt",
    ".plan": "trt",
    ".trt": "trt",
}


def runtime_from_suffix(path) -> Optional[str]:
    return _SUFFIX_TO_RUNTIME.get(Path(path).suffix.lower())


def _trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


def _ort_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def _ort_has_cuda() -> bool:
    try:
        import onnxruntime as ort

        provs = set(ort.get_available_providers())
        return bool(provs & {"CUDAExecutionProvider", "TensorrtExecutionProvider"})
    except Exception:
        return False


def resolve_runtime(device: str, requested: str = "auto") -> str:
    """Resolve ``auto`` to the best runtime for the device."""
    if requested and requested != "auto":
        return requested
    di = device_info(device)
    if di.is_cuda or di.is_jetson:
        if _trt_available():
            return "trt"
        if _ort_has_cuda():
            return "onnx"
        return "torch"
    if di.is_mps:
        return "torch"  # eager MPS is the smooth path; onnx-coreml is opt-in
    return "onnx" if _ort_available() else "torch"


def load_artifact_backend(path, runtime: str, device: str, precision: str,
                          cache_dir: Optional[str] = None) -> Backend:
    """Load a serialized artifact into the matching backend."""
    if runtime == "torchscript":
        return TorchScriptBackend(path, device, precision)
    if runtime == "onnx":
        return OnnxBackend(path, device, precision, cache_dir=cache_dir)
    if runtime == "trt":
        return TensorRTBackend(path, device, precision)
    raise ValueError(f"no artifact backend for runtime {runtime!r}")

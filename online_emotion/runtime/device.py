"""Device resolution and capability probing.

``resolve_device("auto")`` picks the best available device (cuda > mps > cpu).
Probes never hard-fail import: if torch is missing we simply report ``cpu``.
Jetson detection reads the device-tree model string so the export/cache layer
can build TensorRT engines keyed to the exact board.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union

DeviceSpec = Union[str, int, None]


@dataclass(frozen=True)
class DeviceInfo:
    type: str               # "cuda" | "mps" | "cpu"
    index: Optional[int]
    name: str               # human-readable
    is_cuda: bool
    is_mps: bool
    is_cpu: bool
    is_jetson: bool

    @property
    def torch_device(self) -> str:
        return f"cuda:{self.index or 0}" if self.type == "cuda" else self.type


@lru_cache(maxsize=1)
def is_jetson() -> bool:
    """True when running on an NVIDIA Jetson (Tegra) board."""
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(p, "rb") as f:
                model = f.read().decode("utf-8", "ignore").lower()
            if "jetson" in model or ("nvidia" in model and "tegra" in model):
                return True
        except OSError:
            continue
    return os.path.exists("/etc/nv_tegra_release")


def resolve_device(device: DeviceSpec = "auto") -> str:
    """Resolve a device spec to a canonical torch device string.

    "auto" -> cuda:0 > mps > cpu. Also accepts "cpu", "mps", "cuda",
    "cuda:1", "gpu", or an integer GPU index.
    """
    if device is None:
        device = "auto"
    if isinstance(device, int):
        return f"cuda:{device}"
    s = str(device).strip().lower()
    if s in ("", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    if s.isdigit():
        return f"cuda:{int(s)}"
    if s in ("gpu", "cuda"):
        return "cuda:0"
    return s


def device_info(device: DeviceSpec = "auto") -> DeviceInfo:
    dev = resolve_device(device)
    dtype = dev.split(":")[0]
    index = int(dev.split(":")[1]) if ":" in dev else (0 if dtype == "cuda" else None)
    name = dtype
    try:
        import torch

        if dtype == "cuda" and torch.cuda.is_available():
            name = torch.cuda.get_device_name(index or 0)
        elif dtype == "mps":
            name = "Apple Silicon (MPS)"
    except Exception:
        pass
    return DeviceInfo(
        type=dtype,
        index=index,
        name=name,
        is_cuda=(dtype == "cuda"),
        is_mps=(dtype == "mps"),
        is_cpu=(dtype == "cpu"),
        is_jetson=is_jetson(),
    )


def resolve_precision(precision: str, device: DeviceSpec) -> str:
    """``auto`` -> fp16 on CUDA, fp32 on MPS/CPU (MPS fp16 op coverage is patchy)."""
    p = (precision or "auto").strip().lower()
    if p != "auto":
        return p
    return "fp16" if device_info(device).is_cuda else "fp32"


def device_identity(runtime: str, device: DeviceSpec) -> str:
    """Identity string for the artifact cache key.

    TorchScript/ONNX artifacts are portable, so they share one key. TensorRT
    engines are bound to the exact GPU arch + TRT/JetPack version, so the
    identity is specific and prevents an engine built elsewhere from loading.
    """
    if runtime != "trt":
        return "portable"
    di = device_info(device)
    parts = [di.name.replace(" ", "_")]
    try:
        import torch

        if di.is_cuda and torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(di.index or 0)
            parts.append(f"sm{cc[0]}{cc[1]}")
    except Exception:
        pass
    try:
        import tensorrt as trt  # type: ignore

        parts.append(f"trt{trt.__version__}")
    except Exception:
        pass
    if di.is_jetson:
        for p in ("/etc/nv_tegra_release",):
            try:
                with open(p) as f:
                    parts.append("jp_" + f.readline().strip().split(",")[0].replace(" ", ""))
            except OSError:
                pass
    return "|".join(parts)

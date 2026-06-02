"""TorchScript backend — a portable, serialized graph loaded with torch.jit."""
from __future__ import annotations

from pathlib import Path

from .base import Backend


class TorchScriptBackend(Backend):
    runtime = "torchscript"
    supports_dynamic_batch = True

    def __init__(self, path, device: str, precision: str = "fp32") -> None:
        super().__init__(device, precision)
        import torch

        self.module = torch.jit.load(str(Path(path)), map_location=device).eval()
        self._half = precision == "fp16"
        if self._half:
            try:
                self.module = self.module.half()   # match fp16 input to fp16 weights (CUDA)
            except Exception:
                self._half = False

    def infer(self, x):
        import torch

        with torch.no_grad():
            x = (x.half() if self._half else x.float()).to(self.device)
            out = self.module(x)
        return tuple(out) if isinstance(out, (list, tuple)) else (out,)

    def close(self) -> None:
        self.module = None

"""Eager PyTorch backend — the default dev path (and MPS path on macOS)."""
from __future__ import annotations

from .base import Backend


class TorchBackend(Backend):
    runtime = "torch"
    supports_dynamic_batch = True

    def __init__(self, module, device: str, precision: str = "fp32") -> None:
        super().__init__(device, precision)
        self.module = module.to(device).eval()
        self._half = precision == "fp16"
        if self._half:
            try:
                self.module = self.module.half()
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

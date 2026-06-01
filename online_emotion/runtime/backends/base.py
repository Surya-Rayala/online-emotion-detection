"""Backend interface.

A backend's only job is to map a preprocessed input batch to **raw** model
outputs. Decode / NMS / softmax live in the model family's postprocess, so all
runtimes share one postprocess and stay byte-for-byte comparable. Backends
return torch tensors on ``self.device`` regardless of runtime.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple


class Backend(ABC):
    runtime: str = "base"
    #: Whether the loaded graph accepts a variable batch dimension.
    supports_dynamic_batch: bool = True

    def __init__(self, device: str, precision: str = "fp32") -> None:
        self.device = device
        self.precision = precision

    @abstractmethod
    def infer(self, x) -> Tuple["object", ...]:
        """Run the graph on device tensor ``x`` (N,C,H,W) -> tuple of raw tensors."""

    def warmup(self, input_shape: Sequence[int]) -> None:
        try:
            import torch

            dtype = torch.float16 if self.precision == "fp16" else torch.float32
            self.infer(torch.zeros(tuple(input_shape), device=self.device, dtype=dtype))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
        except Exception:
            pass

    def close(self) -> None:  # pragma: no cover - most backends need nothing
        pass

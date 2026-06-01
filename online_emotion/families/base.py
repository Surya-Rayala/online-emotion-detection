"""The model-family extension point.

A family encapsulates everything model-specific: which weights exist, how to
build the eager module, how to preprocess a frame into the network's input, and
how to turn raw outputs into a structured payload. Adding a new detector
(e.g. SCRFD) means writing one ``ModelFamily`` and registering it — no changes
to the runtime, backends, cache, or public API.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ResolvedWeights:
    key: str                 # logical weight key, or the literal path
    path: Path               # local file to load (checkpoint or serialized artifact)
    fingerprint: str         # sha256 of the file head (cache identity)
    exportable: bool         # can be exported to onnx/trt?
    is_artifact: bool        # path is already .onnx/.engine/.torchscript
    arch: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportSpec:
    input_names: List[str]
    output_names: List[str]
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    opset: int = 17
    trt_min_batch: int = 1
    trt_opt_batch: int = 1
    trt_max_batch: int = 1


class ModelFamily(ABC):
    """Base class for all model families."""

    name: str = "base"
    package: str = "online_inference"
    default_input_size: Tuple[int, int] = (640, 640)

    # -- discovery ---------------------------------------------------------
    @abstractmethod
    def available_weights(self) -> List[str]:
        ...

    @abstractmethod
    def resolve_weights(self, weights: Optional[str], cache) -> ResolvedWeights:
        """Resolve a key-or-path to a local file (downloading if possible)."""

    # -- graph -------------------------------------------------------------
    @abstractmethod
    def build_module(self, resolved: ResolvedWeights, device: str, precision: str):
        """Return an eager ``nn.Module`` (used for the torch path and for export)."""

    # -- pre / post --------------------------------------------------------
    @abstractmethod
    def preprocess(self, frame_chw_bgr, input_size: Tuple[int, int]):
        """``frame_chw_bgr`` is a CHW BGR float tensor on device.

        Returns ``(input_batch_nchw, ctx)`` where ``ctx`` carries whatever the
        postprocess needs (e.g. the letterbox transform).
        """

    @abstractmethod
    def postprocess(self, raw_outputs, ctx, params: Dict[str, Any]) -> Dict[str, Any]:
        """Turn raw model outputs into a plain payload dict on CPU/numpy."""

    # -- export / sizing ---------------------------------------------------
    def export_spec(self, input_size: Tuple[int, int]) -> ExportSpec:
        return ExportSpec(input_names=["input"], output_names=["output"])

    def resolve_input_size(self, input_size, resolved=None) -> Tuple[int, int]:
        if input_size is None:
            return self.default_input_size
        if isinstance(input_size, int):
            return (input_size, input_size)
        return (int(input_size[0]), int(input_size[1]))

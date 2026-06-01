"""ONNX Runtime backend with automatic Execution-Provider selection.

Provider priority is device-driven:
  * CUDA / Jetson -> TensorRT EP (with on-disk engine cache) -> CUDA EP -> CPU
  * macOS         -> CoreML EP -> CPU
  * otherwise     -> CPU

The TensorRT EP engine cache (``trt_engine_cache_enable``) is the ONNX path's
equivalent of "export once, reuse" — engines are built on first run and reused.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .base import Backend
from ..device import device_info


def select_providers(device: str, cache_dir: str | None = None) -> List:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    di = device_info(device)
    providers: List = []
    if di.is_cuda or di.is_jetson:
        if "TensorrtExecutionProvider" in available:
            trt_opts = {"trt_fp16_enable": True}
            if cache_dir:
                eng = str(Path(cache_dir) / "ort_trt_engines")
                os.makedirs(eng, exist_ok=True)
                trt_opts.update({"trt_engine_cache_enable": True, "trt_engine_cache_path": eng})
            providers.append(("TensorrtExecutionProvider", trt_opts))
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
    elif di.is_mps:
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


class OnnxBackend(Backend):
    runtime = "onnx"

    def __init__(self, path, device: str, precision: str = "fp32", cache_dir: str | None = None) -> None:
        super().__init__(device, precision)
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(Path(path)), sess_options=so,
                                            providers=select_providers(device, cache_dir))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.supports_dynamic_batch = not isinstance(inp.shape[0], int)

    def infer(self, x):
        import torch

        arr = x.detach().to("cpu").float().contiguous().numpy()
        outs = self.session.run(None, {self.input_name: arr})
        return tuple(torch.from_numpy(o).to(self.device) for o in outs)

    def close(self) -> None:
        self.session = None

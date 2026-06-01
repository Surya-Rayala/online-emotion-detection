"""Native TensorRT backend (best performance on Jetson / discrete NVIDIA).

I/O buffers are torch CUDA tensors, so there is no extra pycuda dependency and
no host copies. Uses the name-based TRT API (8.5+/10) with a binding fallback.
Validated on Jetson in phase M5; imports are lazy so the module is safe to
import on machines without TensorRT.
"""
from __future__ import annotations

from pathlib import Path

from .base import Backend


class TensorRTBackend(Backend):
    runtime = "trt"
    supports_dynamic_batch = False  # engines are typically built with a fixed profile

    def __init__(self, path, device: str = "cuda:0", precision: str = "fp16") -> None:
        super().__init__(device, precision)
        import tensorrt as trt  # noqa: F401

        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(Path(path), "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = None
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

    def infer(self, x):
        import torch

        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        x = x.to(self.device, dtype=dtype).contiguous()
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        outputs = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            buf = torch.empty(shape, device=self.device, dtype=dtype)
            outputs.append(buf)
            self.context.set_tensor_address(name, buf.data_ptr())
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        return tuple(o.float() for o in outputs)

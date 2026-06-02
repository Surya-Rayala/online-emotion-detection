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
        # Honor a dynamic batch profile: engines exported with a min/opt/max batch report a
        # -1 batch dim, so all faces in a frame run in ONE batched call instead of per-item.
        # (If a batched infer ever errors on-device, streaming._infer falls back to per-item.)
        try:
            ishape = self.engine.get_tensor_shape(self.input_name)
            self.supports_dynamic_batch = bool(len(ishape) and int(ishape[0]) == -1)
        except Exception:
            self.supports_dynamic_batch = False

    def _torch_dtype(self, name):
        import torch

        dt = self.engine.get_tensor_dtype(name)
        return {self._trt.DataType.HALF: torch.float16,
                self._trt.DataType.FLOAT: torch.float32,
                self._trt.DataType.INT32: torch.int32}.get(dt, torch.float32)

    def infer(self, x):
        import torch

        # Feed/read the engine's ACTUAL binding dtypes (a standard fp16-flag engine still has fp32 I/O;
        # a strongly-typed engine may have fp16) — don't assume from precision.
        x = x.to(self.device, dtype=self._torch_dtype(self.input_name)).contiguous()
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        outputs = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            buf = torch.empty(shape, device=self.device, dtype=self._torch_dtype(name))
            outputs.append(buf)
            self.context.set_tensor_address(name, buf.data_ptr())
        # Run on a dedicated (non-default) stream: TRT warns that the default stream forces extra
        # cudaStreamSynchronize calls. wait_stream ensures the inputs/outputs (prepared on the current
        # stream) are ready before TRT reads them; synchronize makes outputs ready for postprocess.
        if getattr(self, "_stream", None) is None:
            self._stream = torch.cuda.Stream(device=self.device)
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        self.context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()
        return tuple(o.float() for o in outputs)

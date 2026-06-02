"""Export primitives: torch -> torchscript / onnx, and onnx -> TensorRT engine.

torch -> trt always goes through ONNX (the robust, standard path). These are
thin, dependency-guarded wrappers; the cache layer decides *when* to call them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from .errors import ExportError
from .logging import get_logger

_log = get_logger("export")


def to_torchscript(module, example_input, out_path: Path) -> Path:
    import torch

    module = module.eval()
    try:
        with torch.no_grad():
            ts = torch.jit.trace(module, example_input, strict=False)
        ts.save(str(out_path))
    except Exception as e:  # pragma: no cover
        raise ExportError(f"torchscript trace failed: {e}") from e
    return out_path


def to_onnx(module, example_input, out_path: Path, *, input_names: Sequence[str],
            output_names: Sequence[str], dynamic_axes: Optional[Dict] = None,
            opset: int = 17, simplify: bool = True) -> Path:
    import torch

    module = module.eval()
    try:
        # dynamo=False forces the stable TorchScript-based exporter (no onnxscript
        # dependency) — also the more TensorRT-friendly graph.
        torch.onnx.export(
            module, example_input, str(out_path),
            input_names=list(input_names), output_names=list(output_names),
            dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(
            module, example_input, str(out_path),
            input_names=list(input_names), output_names=list(output_names),
            dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True,
        )
    except Exception as e:
        raise ExportError(f"onnx export failed: {e}") from e
    if simplify:
        try:
            import onnx
            import onnxsim

            model = onnx.load(str(out_path))
            simplified, ok = onnxsim.simplify(model)
            if ok:
                onnx.save(simplified, str(out_path))
                _log.info("onnx simplified")
        except Exception:  # onnxsim is optional
            pass
    return out_path


def onnx_to_trt(onnx_path: Path, engine_path: Path, *, precision: str = "fp16",
                workspace_gb: float = 4.0, input_name: str = "input",
                min_shape: Optional[Sequence[int]] = None,
                opt_shape: Optional[Sequence[int]] = None,
                max_shape: Optional[Sequence[int]] = None) -> Path:
    try:
        import tensorrt as trt
    except Exception as e:  # pragma: no cover
        raise ExportError("tensorrt is not installed; install the [trt] extra on the target device") from e

    logger = trt.Logger(trt.Logger.WARNING)
    try:
        builder = trt.Builder(logger)
    except Exception as e:  # pybind returns nullptr when CUDA can't init (e.g. driver too old)
        raise ExportError(
            "TensorRT could not initialize CUDA. This usually means the NVIDIA driver is older than "
            "the CUDA version the installed 'tensorrt' wheel needs. Update your NVIDIA driver (or install "
            "a tensorrt build matching your driver's CUDA). See the TensorRT note in the README."
        ) from e
    # TensorRT 10 removed the EXPLICIT_BATCH flag (explicit batch is the default now); TRT 8 needs it.
    _eb = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network = builder.create_network(1 << int(_eb) if _eb is not None else 0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ExportError(f"onnx->trt parse failed: {errs}")

    config = builder.create_builder_config()
    # Workspace limit: set_memory_pool_limit is TRT 8.4+/10; older builds use max_workspace_size.
    _workspace = int(workspace_gb * (1 << 30))
    _pool = getattr(trt, "MemoryPoolType", None)
    if hasattr(config, "set_memory_pool_limit") and _pool is not None:
        config.set_memory_pool_limit(_pool.WORKSPACE, _workspace)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = _workspace
    # TRT 10 removed builder.platform_has_fast_fp16; TRT 10.12+ removed BuilderFlag.FP16 too
    # (precision is via strong typing there). Set the fp16 compute flag only when both still exist;
    # either way the engine I/O stays the onnx dtype and the runtime reads the engine's real dtypes.
    _fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    if precision == "fp16" and _fp16_flag is not None and getattr(builder, "platform_has_fast_fp16", True):
        config.set_flag(_fp16_flag)
    if min_shape is not None:
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, tuple(min_shape), tuple(opt_shape or min_shape), tuple(max_shape or min_shape))
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise ExportError("tensorrt engine build returned None (see the TensorRT log above; "
                          "usually a driver/CUDA mismatch or an unsupported op)")
    Path(engine_path).write_bytes(bytes(serialized))   # IHostMemory -> bytes (portable across TRT versions)
    return engine_path

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
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ExportError(f"onnx->trt parse failed: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if min_shape is not None:
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, tuple(min_shape), tuple(opt_shape or min_shape), tuple(max_shape or min_shape))
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise ExportError("tensorrt engine build returned None")
    Path(engine_path).write_bytes(serialized)
    return engine_path

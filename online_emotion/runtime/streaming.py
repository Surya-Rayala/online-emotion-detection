"""Shared orchestration for streaming model wrappers.

``StreamingModel`` owns the boring-but-critical plumbing every model package
needs: resolve device/runtime/precision/weights, build the backend (loading a
ready artifact, or exporting-then-loading via the cache on first use), warm up,
keep stats, and clean up. Subclasses (FaceDetector, EmotionRecognizer) add only
their call shape and result types.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from .backends import (
    TorchBackend, load_artifact_backend, resolve_runtime, runtime_from_suffix,
)
from .cache import ArtifactCache
from .config import InferenceConfig
from .device import resolve_device, resolve_precision
from .errors import RuntimeUnavailableError
from .export import onnx_to_trt, to_onnx, to_torchscript
from .logging import get_logger, warn_once
from .timing import RunStats

_log = get_logger("engine")


class StreamingModel:
    package: str = "online_inference"

    def __init__(self, *, family, weights, runtime, device, precision,
                 input_size, cache_dir, warmup, display) -> None:
        self._family = family
        self._device = resolve_device(device)
        self._precision = resolve_precision(precision, self._device)
        self._cache = ArtifactCache(cache_dir)
        self._resolved = family.resolve_weights(weights, self._cache)
        self._input_size: Tuple[int, int] = family.resolve_input_size(input_size, self._resolved)
        self._backend = None
        self._native_model = None

        if self._resolved.meta.get("native"):
            if runtime not in (None, "auto", "torch"):
                raise RuntimeUnavailableError(
                    f"weights '{self._resolved.key}' are torch-only; runtime={runtime!r} is unsupported")
            self._runtime = "torch"
        elif self._resolved.is_artifact:
            self._runtime = runtime_from_suffix(self._resolved.path) or resolve_runtime(self._device, runtime)
        else:
            self._runtime = resolve_runtime(self._device, runtime)

        self._config = InferenceConfig(
            package=self.package, model=family.name, weights=self._resolved.key,
            weights_fingerprint=self._resolved.fingerprint, runtime=self._runtime,
            device=self._device, precision=self._precision, input_size=self._input_size,
            extra={"arch": self._resolved.arch},
        )

        if self._resolved.meta.get("native"):
            build_native = getattr(family, "build_native", None)
            if build_native is None:
                raise RuntimeUnavailableError(f"family {family.name} has no native runner")
            self._native_model = build_native(self._resolved, self._device)
        else:
            self._backend = self._build_backend()

        self.stats = RunStats()
        self.display = display
        _log.info("ready: %s", self._config.summary())
        if warmup and self._backend is not None:
            self._backend.warmup((1, 3, *self._input_size))

    # -- public introspection ---------------------------------------------
    @property
    def config(self) -> InferenceConfig:
        return self._config

    @property
    def device(self) -> str:
        return self._device

    @property
    def runtime(self) -> str:
        return self._runtime

    @property
    def is_native(self) -> bool:
        """True when running a torch-only native path (e.g. ternaus) instead of a backend."""
        return self._native_model is not None

    # -- backend construction ---------------------------------------------
    def _build_backend(self):
        family, resolved = self._family, self._resolved
        rt, device, precision = self._runtime, self._device, self._precision
        H, W = self._input_size

        if resolved.is_artifact:  # user handed us a ready .onnx/.engine/.torchscript
            return load_artifact_backend(resolved.path, rt, device, precision, cache_dir=str(self._cache.root))

        if rt == "torch":
            return TorchBackend(family.build_module(resolved, device, precision), device, precision)

        if not resolved.exportable:
            raise RuntimeUnavailableError(
                f"{family.name}/{resolved.key} supports runtime='torch' only; "
                f"choose an exportable weight to use runtime={rt!r}."
            )

        spec = family.export_spec(self._input_size)
        dynamic = spec.dynamic_axes is not None

        def export_fn(tmp: Path) -> str:
            import torch

            module = family.build_module(resolved, "cpu", "fp32").eval()
            example = torch.zeros((1, 3, H, W), dtype=torch.float32)
            if rt == "torchscript":
                out = tmp / "model.torchscript"
                to_torchscript(module, example, out)
                return out.name
            onnx_path = tmp / "model.onnx"
            to_onnx(module, example, onnx_path, input_names=spec.input_names,
                    output_names=spec.output_names, dynamic_axes=spec.dynamic_axes, opset=spec.opset)
            if rt == "onnx":
                return onnx_path.name
            engine = tmp / "model.engine"
            onnx_to_trt(onnx_path, engine, precision=precision, input_name=spec.input_names[0],
                        min_shape=(spec.trt_min_batch, 3, H, W),
                        opt_shape=(spec.trt_opt_batch, 3, H, W),
                        max_shape=(spec.trt_max_batch, 3, H, W))
            return engine.name

        ref = self._cache.ensure(config=self._config, runtime=rt, dynamic=dynamic,
                                 export_fn=export_fn, source_weights=str(resolved.path))
        return load_artifact_backend(ref.path, rt, device, precision, cache_dir=str(self._cache.root))

    # -- inference with safety fallback -----------------------------------
    def _infer(self, x):
        """Run the backend; if a fixed-batch artifact rejects N>1, fall back to per-item."""
        import torch

        n = x.shape[0]
        if n > 1 and not self._backend.supports_dynamic_batch:
            return self._per_item(x)
        try:
            return self._backend.infer(x)
        except Exception:
            if n > 1:
                warn_once(_log, "batch_fallback",
                          "backend rejected a batched input; falling back to per-item inference (slower)")
                return self._per_item(x)
            raise

    def _per_item(self, x):
        import torch

        outs = [self._backend.infer(x[i:i + 1]) for i in range(x.shape[0])]
        k = len(outs[0])
        return tuple(torch.cat([o[j] for o in outs], dim=0) for j in range(k))

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_backend", None) is not None:
            self._backend.close()
            self._backend = None
        self._native_model = None

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

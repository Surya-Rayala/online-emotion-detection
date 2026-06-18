"""HSEmotion family: EfficientNet/MobileNet emotion classifiers.

Operates on face crops. Preprocess resizes to the model size, converts BGR->RGB,
scales to [0,1] and applies ImageNet normalisation; multiple crops are stacked
into one batch (the efficiency win). The raw logits graph exports cleanly to
onnx/trt with a dynamic batch axis. ``*_va_mtl`` weights also emit valence/arousal.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .. import registry as _registry
from ..models.hsemotion import load_hsemotion_module
from ..models.hsemotion.labels import AFFECTNET_8
from .base import ExportSpec, ModelFamily, ResolvedWeights

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class HSEmotionFamily(ModelFamily):
    name = "hsemotion"
    package = "online_emotion"
    default_input_size = (224, 224)

    def __init__(self) -> None:
        self._mean = None
        self._std = None

    # -- discovery ---------------------------------------------------------
    def available_weights(self) -> List[str]:
        return _registry.available_weights()

    def resolve_weights(self, weights: Optional[str], cache) -> ResolvedWeights:
        return _registry.resolve_weights(weights, cache)

    def resolve_input_size(self, input_size, resolved=None) -> Tuple[int, int]:
        if input_size is None and resolved is not None:
            s = int(resolved.meta.get("img_size", 224))
            return (s, s)
        return super().resolve_input_size(input_size, resolved)

    # -- graph -------------------------------------------------------------
    def build_module(self, resolved: ResolvedWeights, device: str, precision: str):
        return load_hsemotion_module(resolved.meta["model_name"])

    # -- preprocessing -----------------------------------------------------
    def _norm_tensors(self, device, dtype):
        import torch

        if self._mean is None or self._mean.device != torch.device(device):
            self._mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
            self._std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        return self._mean.to(dtype), self._std.to(dtype)

    def normalize_batch(self, nchw_bgr):
        """BGR [0,255] -> RGB [0,1] -> ImageNet-normalised, on the same device."""
        x = nchw_bgr.flip(1) / 255.0
        mean, std = self._norm_tensors(x.device, x.dtype)
        return (x - mean) / std

    def prepare_crops(self, crops_chw_bgr: List, size: Tuple[int, int]):
        import torch
        import torch.nn.functional as F

        # antialias=True matches HSEmotion's torchvision Resize(antialias=True) — important
        # fidelity to the model's training preprocessing when downscaling large crops.
        resized = [F.interpolate(c.unsqueeze(0), size=size, mode="bilinear",
                                 align_corners=False, antialias=True)
                   for c in crops_chw_bgr]
        return self.normalize_batch(torch.cat(resized, dim=0))

    def preprocess(self, frame_chw_bgr, input_size: Tuple[int, int]):
        return self.prepare_crops([frame_chw_bgr], input_size), {}

    # -- postprocessing ----------------------------------------------------
    def postprocess(self, raw, ctx, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        import torch

        logits = raw[0] if isinstance(raw, (list, tuple)) else raw
        classes = ctx.get("classes", AFFECTNET_8)
        va = ctx.get("va", False)
        n = len(classes)
        probs = torch.softmax(logits[:, :n].float(), dim=1).cpu().numpy()
        out: List[Dict[str, Any]] = []
        for i in range(probs.shape[0]):
            valence = arousal = None
            if va and logits.shape[1] >= n + 2:
                valence = float(logits[i, n])
                arousal = float(logits[i, n + 1])
            idx = int(probs[i].argmax())
            out.append({"label": classes[idx], "label_index": idx, "probs": probs[i],
                        "valence": valence, "arousal": arousal})
        return out

    # -- export ------------------------------------------------------------
    def export_spec(self, input_size: Tuple[int, int]) -> ExportSpec:
        return ExportSpec(
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset=13,
            trt_min_batch=1, trt_opt_batch=8, trt_max_batch=32,
        )

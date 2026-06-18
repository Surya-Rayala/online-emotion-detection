"""HSEmotion model metadata, labels, and the underlying nn.Module loader."""
from __future__ import annotations

from .labels import AFFECTNET_7, AFFECTNET_8, DEFAULT_WEIGHT, MODEL_META, class_names


def load_hsemotion_module(model_name: str):
    """Return the underlying EfficientNet/MobileNet ``nn.Module`` for a model name.

    Delegates weight download/loading to the ``hsemotion`` package, then keeps
    only the raw classifier so we control preprocessing, batching, and export.
    """
    from ...runtime.errors import RuntimeUnavailableError

    try:
        import torch
        from hsemotion.facial_emotions import HSEmotionRecognizer
    except Exception as e:  # pragma: no cover
        raise RuntimeUnavailableError(
            "the 'hsemotion' package is required: pip install 'online-emotion-detection[torch]'"
        ) from e

    # hsemotion ships pickled full models and calls torch.load without weights_only or
    # map_location. torch>=2.6 defaults weights_only=True (rejects them), and some
    # checkpoints (e.g. enet_b2_8) were saved with CUDA tensors, so loading them on a
    # CPU/MPS box raises "deserialize on a CUDA device". Force both while loading this
    # explicitly-installed, trusted checkpoint; the module is moved to the real device after.
    _orig_load = torch.load

    def _compat_load(*a, **k):
        k.setdefault("weights_only", False)
        k.setdefault("map_location", "cpu")
        return _orig_load(*a, **k)

    torch.load = _compat_load
    try:
        rec = HSEmotionRecognizer(model_name=model_name, device="cpu")
    finally:
        torch.load = _orig_load
    feat = getattr(rec, "model", None)
    if feat is None:  # pragma: no cover - guard against upstream API drift
        raise RuntimeUnavailableError("could not extract the torch module from HSEmotionRecognizer")
    _patch_timm_drift(feat)
    feat.eval()

    # CRITICAL: HSEmotionRecognizer sets ``model.classifier = Identity()`` and applies the
    # real classifier separately (numpy ``get_probab``), so ``rec.model`` only emits the
    # backbone FEATURE vector (e.g. 1280-d), NOT emotion logits. Re-attach the saved
    # classifier head so the module outputs logits end-to-end (and exports correctly).
    W = getattr(rec, "classifier_weights", None)
    b = getattr(rec, "classifier_bias", None)
    if W is None or b is None:  # upstream kept the head on the model (older API) -> use as-is
        return feat
    import numpy as np
    import torch.nn as nn

    W = np.asarray(W)
    b = np.asarray(b)
    head = nn.Linear(int(W.shape[1]), int(W.shape[0]))
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(W).float())
        head.bias.copy_(torch.from_numpy(b).float())
    return nn.Sequential(feat, head).eval()


def _patch_timm_drift(model) -> None:
    """Reconcile newer-timm forward() with older pickled EfficientNet instances.

    timm added a space-to-depth path (``conv_s2d``/``bn_s2d``) to its blocks; the
    pickled HSEmotion models predate it, so the live forward() hits an attribute
    that the instance lacks. The old models simply don't use that path -> None.
    """
    import torch.nn as nn

    _blocks = {"DepthwiseSeparableConv", "InvertedResidual", "EdgeResidual", "CondConvResidual"}
    for m in model.modules():
        if type(m).__name__ in _blocks:
            for attr in ("conv_s2d", "bn_s2d"):  # space-to-depth path (skipped when None)
                if not hasattr(m, attr):
                    setattr(m, attr, None)
            if not hasattr(m, "aa"):  # anti-alias is *called*, so it must be a no-op
                m.aa = nn.Identity()


__all__ = ["AFFECTNET_7", "AFFECTNET_8", "MODEL_META", "DEFAULT_WEIGHT", "class_names", "load_hsemotion_module"]

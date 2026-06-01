"""AffectNet class labels and per-model metadata for HSEmotion."""
from __future__ import annotations

from typing import Tuple

# Savchenko's class ordering (alphabetical).
AFFECTNET_8: Tuple[str, ...] = (
    "Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
)
AFFECTNET_7: Tuple[str, ...] = (
    "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
)


def class_names(n: int) -> Tuple[str, ...]:
    return AFFECTNET_8 if n == 8 else AFFECTNET_7


# model_name -> input size / class count / valence-arousal head / backbone
MODEL_META = {
    "enet_b0_8_best_vgaf": {"img_size": 224, "classes": 8, "va": False, "arch": "efficientnet_b0"},
    "enet_b0_8_best_afew": {"img_size": 224, "classes": 8, "va": False, "arch": "efficientnet_b0"},
    "enet_b2_8": {"img_size": 260, "classes": 8, "va": False, "arch": "efficientnet_b2"},
    "enet_b0_8_va_mtl": {"img_size": 224, "classes": 8, "va": True, "arch": "efficientnet_b0"},
}
DEFAULT_WEIGHT = "enet_b0_8_best_vgaf"

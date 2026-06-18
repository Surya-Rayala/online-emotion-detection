"""Regression guard for the HSEmotion module.

HSEmotionRecognizer replaces ``model.classifier`` with Identity and applies the
real head separately, so ``rec.model`` emits a backbone FEATURE vector, not class
logits. ``load_hsemotion_module`` must re-attach the head; otherwise the recognizer
silently returns near-uniform garbage. This test fails if the head is ever dropped
again. Gated on torch + the hsemotion checkpoint; self-skips offline."""
from __future__ import annotations

import pytest


@pytest.mark.timeout(300)
def test_module_outputs_class_logits_not_features():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import torch

    from online_emotion.models.hsemotion import load_hsemotion_module
    from online_emotion.models.hsemotion.labels import MODEL_META

    name = "enet_b0_8_best_vgaf"
    try:
        model = load_hsemotion_module(name)
    except Exception as e:  # offline / checkpoint unavailable
        pytest.skip(f"hsemotion checkpoint unavailable: {e}")

    n = MODEL_META[name]["classes"]            # 8
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    # must be (1, n) logits — NOT the 1280-d backbone feature vector
    assert out.shape[1] == n, f"module emits {out.shape[1]} dims; expected {n} class logits (head dropped?)"


@pytest.mark.timeout(300)
def test_predictions_are_not_uniform_on_a_real_face():
    """A working classifier is confident on a clear face; a head-less model is ~uniform."""
    pytest.importorskip("torch")
    pytest.importorskip("skimage")
    import numpy as np
    from skimage import data

    import online_emotion

    try:
        emo = online_emotion.EmotionRecognizer("hsemotion", device="cpu", runtime="torch")
    except Exception as e:
        pytest.skip(f"hsemotion unavailable: {e}")
    face = data.astronaut()[30:480, 130:430][:, :, ::-1]   # RGB->BGR crop around the face
    probs = np.asarray(emo([np.ascontiguousarray(face)]).emotions[0].probs)
    emo.close()
    # uniform over 8 classes would be 0.125; a working model peaks well above that
    assert probs.max() > 0.25, f"near-uniform probs {probs.round(3)} — classifier head likely missing"

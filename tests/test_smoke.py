"""Smoke tests. The network-dependent end-to-end test self-skips offline."""
from __future__ import annotations

import numpy as np
import pytest


def test_imports_and_discovery():
    import online_emotion

    assert online_emotion.__version__
    assert "hsemotion" in online_emotion.available_models()
    weights = online_emotion.available_weights("hsemotion")
    assert "enet_b0_8_best_vgaf" in weights


def test_device_and_runtime_resolution():
    from online_emotion.runtime.backends import resolve_runtime
    from online_emotion.runtime.device import resolve_device

    dev = resolve_device("auto")
    assert dev.split(":")[0] in ("cuda", "mps", "cpu")
    assert resolve_runtime(dev, "torch") == "torch"


def test_normalize_batch():
    torch = pytest.importorskip("torch")
    from online_emotion.families.hsemotion import HSEmotionFamily

    fam = HSEmotionFamily()
    crops = torch.full((2, 3, 224, 224), 255.0)  # BGR white
    out = fam.normalize_batch(crops)
    assert out.shape == (2, 3, 224, 224)
    # white -> ~ (1 - mean)/std, finite
    assert torch.isfinite(out).all()


def test_labels():
    from online_emotion.models.hsemotion.labels import class_names

    assert len(class_names(8)) == 8
    assert "Happiness" in class_names(8)


@pytest.mark.timeout(300)
def test_end_to_end_emotion_on_synthetic_crops():
    """Builds the default recognizer (downloads HSEmotion weights) and runs a batch."""
    pytest.importorskip("torch")
    pytest.importorskip("hsemotion")
    import online_emotion

    try:
        emo = online_emotion.EmotionRecognizer("hsemotion", device="auto", warmup=False)
    except Exception as e:
        pytest.skip(f"hsemotion weights unavailable: {e}")
    crops = [(np.random.rand(160, 160, 3) * 255).astype("uint8") for _ in range(3)]
    res = emo(crops)
    assert len(res) == 3
    assert res.emotions[0].probs.shape[0] == len(res.classes)
    assert res.emotions[0].label in res.classes
    emo.close()

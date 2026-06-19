"""Server-side ``--instances`` resolver: parsing + per-GPU placement + warnings.

Pure logic — no FastAPI, no real model, no GPU. ``resolve_device`` and
``torch.cuda.device_count`` are monkeypatched so the placement/fallback behavior is
deterministic on any machine.
"""
from __future__ import annotations

import logging

import pytest

from online_emotion import serve


# --- spec parsing ---------------------------------------------------------
def test_parse_plain_number():
    assert serve._parse_instances(4) == 4
    assert serve._parse_instances("4") == 4


def test_parse_per_gpu_map():
    assert serve._parse_instances("0=2,1=1") == {0: 2, 1: 1}
    assert serve._parse_instances("cuda:0=2,cuda:1=1") == {0: 2, 1: 1}
    assert serve._parse_instances({0: 2, 1: 1}) == {0: 2, 1: 1}


def test_parse_rejects_garbage():
    for bad in ["", "abc", "x=y", "=", True]:
        with pytest.raises((ValueError, TypeError)):
            serve._parse_instances(bad)


# --- device resolution (monkeypatched hardware) ---------------------------
class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _resolve(monkeypatch, spec, *, resolved, cuda):
    torch = pytest.importorskip("torch")
    from online_emotion.runtime import device as devmod

    monkeypatch.setattr(devmod, "resolve_device", lambda d: resolved)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: cuda)
    cap = _Capture()
    lg = logging.getLogger("online_inference")
    lg.addHandler(cap)
    try:
        return serve._resolve_instance_devices(spec, "ignored"), cap.msgs
    finally:
        lg.removeHandler(cap)


def test_single_instance_is_one_device_no_warning(monkeypatch):
    devs, msgs = _resolve(monkeypatch, 1, resolved="cpu", cuda=0)
    assert devs == ["cpu"]
    assert msgs == []


def test_n_copies_share_one_device_warns(monkeypatch):
    devs, msgs = _resolve(monkeypatch, 3, resolved="mps", cuda=0)
    assert devs == ["mps", "mps", "mps"]
    assert any("time-share" in m for m in msgs)


def test_plain_n_round_robins_across_gpus(monkeypatch):
    devs, _ = _resolve(monkeypatch, 3, resolved="cuda:0", cuda=2)
    assert devs == ["cuda:0", "cuda:1", "cuda:0"]


def test_per_gpu_map_places_explicitly(monkeypatch):
    devs, _ = _resolve(monkeypatch, "0=2,1=1", resolved="cuda:0", cuda=2)
    assert devs == ["cuda:0", "cuda:0", "cuda:1"]


def test_per_gpu_map_without_cuda_falls_back_with_warning(monkeypatch):
    devs, msgs = _resolve(monkeypatch, {0: 2, 1: 1}, resolved="cpu", cuda=0)
    assert devs == ["cpu", "cpu", "cpu"]                  # total count, all on the resolved device
    assert any("needs CUDA" in m for m in msgs)


def test_missing_gpu_index_is_skipped(monkeypatch):
    devs, msgs = _resolve(monkeypatch, "0=1,3=2", resolved="cuda:0", cuda=1)
    assert devs == ["cuda:0"]                             # cuda:3 skipped (only 1 GPU)
    assert any("cuda:3 not found" in m for m in msgs)


def test_all_invalid_indices_fall_back_to_one(monkeypatch):
    devs, msgs = _resolve(monkeypatch, "5=2", resolved="cuda:0", cuda=2)
    assert devs == ["cuda:0"]
    assert any("no valid GPUs" in m or "not found" in m for m in msgs)


def test_garbage_spec_falls_back_to_one(monkeypatch):
    devs, msgs = _resolve(monkeypatch, "abc", resolved="cpu", cuda=0)
    assert devs == ["cpu"]
    assert any("could not parse" in m for m in msgs)


def test_non_positive_falls_back_to_one(monkeypatch):
    devs, msgs = _resolve(monkeypatch, "0", resolved="cpu", cuda=0)
    assert devs == ["cpu"]
    assert any("not positive" in m for m in msgs)

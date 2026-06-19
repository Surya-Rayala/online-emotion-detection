"""Public annotate() helper (torch-free, numpy+opencv)."""
from __future__ import annotations

import numpy as np

import online_emotion
from online_emotion.client import EmotionFrameResult, EmotionResult


def test_annotate_labels_boxes_by_position():
    frame = np.zeros((90, 120, 3), "uint8")
    boxes = np.array([[10, 10, 70, 70]], "float32")
    res = EmotionFrameResult([EmotionResult("happy", 0.8, 3, None, None)], ("happy",))
    out = online_emotion.annotate(frame, boxes, res, hud="face -> emotion")
    assert out.shape == frame.shape
    assert out is not frame and not np.shares_memory(out, frame)
    assert out.any()
    assert not frame.any()


def test_annotate_box_without_emotion_is_unlabelled():
    frame = np.zeros((40, 40, 3), "uint8")
    boxes = np.array([[1, 1, 20, 20], [21, 21, 39, 39]], "float32")
    res = EmotionFrameResult([EmotionResult("sad", 0.5, 1, None, None)], ("sad",))
    out = online_emotion.annotate(frame, boxes, res)   # 2 boxes, 1 emotion -> no crash
    assert out.shape == frame.shape

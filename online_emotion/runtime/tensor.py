"""The single choke point for turning a frame into a device-resident tensor.

Accepting both NumPy (BGR, HWC) and ``torch.Tensor`` inputs here — and doing
all resize / letterbox / crop work with torch ops on the target device — is
what lets the rest of the pipeline avoid host<->device round trips. Families
apply their own colour/normalisation on top of the canonical BGR float tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np

Frame = Union[np.ndarray, "object"]  # np.ndarray (BGR HWC) or torch.Tensor


@dataclass(frozen=True)
class LetterboxMeta:
    orig_h: int
    orig_w: int
    scale: float
    pad_x: int
    pad_y: int
    out_h: int
    out_w: int


def is_torch_tensor(x: object) -> bool:
    return "torch" in type(x).__module__ and hasattr(x, "device") and hasattr(x, "dim")


def load_frame(frame: Frame, device: str, *, color: str = "bgr"):
    """Return a float32 **CHW BGR** image tensor on ``device`` (no resize).

    NumPy is assumed HWC BGR (OpenCV native). torch tensors may be HWC or CHW
    on any device; if ``color="rgb"`` the channels are flipped to BGR so the
    canonical form is uniform.
    """
    import torch

    if is_torch_tensor(frame):
        t = frame
        if t.dim() == 4 and t.shape[0] == 1:
            t = t[0]
        if t.dim() != 3:
            raise ValueError(f"expected a 3D image tensor, got shape {tuple(t.shape)}")
        # CHW if first dim is a channel count, else HWC.
        if t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
            chw = t
        elif t.shape[-1] in (1, 3):
            chw = t.permute(2, 0, 1)
        else:  # ambiguous (e.g. 3xN): assume CHW
            chw = t
        chw = chw.to(device=device, dtype=torch.float32)
        if chw.shape[0] == 1:
            chw = chw.repeat(3, 1, 1)
    else:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, axis=2)
        if arr.ndim != 3 or arr.shape[2] not in (1, 3):
            raise ValueError(f"expected HWC image, got shape {arr.shape}")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        chw = torch.from_numpy(np.ascontiguousarray(arr)).to(device=device).permute(2, 0, 1).float()

    if color.lower() == "rgb":  # caller says channels are RGB -> flip to BGR canonical
        chw = chw.flip(0)
    return chw


def letterbox(img_chw, size: Tuple[int, int], pad_value: Union[float, Sequence[float]] = 0.0):
    """Aspect-preserving resize + centre pad to ``size`` (H, W).

    Returns ``(out_chw, LetterboxMeta)``. All work stays on the input device.
    """
    import torch
    import torch.nn.functional as F

    c, h, w = img_chw.shape
    out_h, out_w = int(size[0]), int(size[1])
    scale = min(out_h / h, out_w / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = F.interpolate(img_chw.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False)[0]

    out = img_chw.new_empty((c, out_h, out_w))
    if isinstance(pad_value, (int, float)):
        out.fill_(float(pad_value))
    else:
        for ci in range(c):
            out[ci].fill_(float(pad_value[ci]))
    pad_x = (out_w - nw) // 2
    pad_y = (out_h - nh) // 2
    out[:, pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    meta = LetterboxMeta(orig_h=h, orig_w=w, scale=scale, pad_x=pad_x, pad_y=pad_y, out_h=out_h, out_w=out_w)
    return out, meta


def rescale_boxes(boxes, meta: LetterboxMeta):
    """Map xyxy boxes from letterbox-input coords back to original-frame coords."""
    if boxes.numel() == 0:
        return boxes
    b = boxes.clone()
    b[:, 0] = (b[:, 0] - meta.pad_x) / meta.scale
    b[:, 1] = (b[:, 1] - meta.pad_y) / meta.scale
    b[:, 2] = (b[:, 2] - meta.pad_x) / meta.scale
    b[:, 3] = (b[:, 3] - meta.pad_y) / meta.scale
    b[:, 0::2] = b[:, 0::2].clamp(0, meta.orig_w - 1)
    b[:, 1::2] = b[:, 1::2].clamp(0, meta.orig_h - 1)
    return b


def rescale_points(points, meta: LetterboxMeta):
    """Map (N, K, 2) landmark points from letterbox-input coords to original."""
    if points.numel() == 0:
        return points
    p = points.clone()
    p[..., 0] = ((p[..., 0] - meta.pad_x) / meta.scale).clamp(0, meta.orig_w - 1)
    p[..., 1] = ((p[..., 1] - meta.pad_y) / meta.scale).clamp(0, meta.orig_h - 1)
    return p


def crop_resize(img_chw, boxes_xyxy, size: Tuple[int, int]):
    """Crop ``boxes`` from a CHW image and resize each to ``size`` (H, W).

    Returns ``(M, 3, H, W)`` on the same device. Used to feed face crops to the
    emotion model without leaving the device.
    """
    import torch
    import torch.nn.functional as F

    _, H, W = img_chw.shape
    th, tw = int(size[0]), int(size[1])
    out = img_chw.new_zeros((len(boxes_xyxy), 3, th, tw))
    for i, box in enumerate(boxes_xyxy):
        x1 = int(max(0, min(W - 1, float(box[0]))))
        y1 = int(max(0, min(H - 1, float(box[1]))))
        x2 = int(max(x1 + 1, min(W, float(box[2]))))
        y2 = int(max(y1 + 1, min(H, float(box[3]))))
        crop = img_chw[:, y1:y2, x1:x2]
        if crop.shape[0] == 1:
            crop = crop.repeat(3, 1, 1)
        out[i] = F.interpolate(crop.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False)[0]
    return out

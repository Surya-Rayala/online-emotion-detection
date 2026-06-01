"""Public emotion API: ``EmotionRecognizer`` + result types.

The natural unit is a **face crop** (or a batch of crops — batching is the
efficiency win). ``predict_on_boxes(frame, boxes)`` crops on-device, so chaining
a face detector -> emotion never round-trips through the host.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .families import available_models as _available_models
from .families import get_family
from .models.hsemotion import class_names
from .runtime.config import InferenceConfig
from .runtime.sources import FrameRef, open_source
from .runtime.streaming import StreamingModel
from .runtime.tensor import crop_resize, load_frame
from .runtime.timing import Stopwatch
from .runtime.viz import draw_box_label, draw_hud, hash_color

Frame = Union[np.ndarray, "object"]


@dataclass(frozen=True)
class EmotionResult:
    label: str
    label_index: int
    probs: np.ndarray          # (C,) softmax
    valence: Optional[float]   # set only for *_va_mtl weights
    arousal: Optional[float]


@dataclass(frozen=True)
class EmotionFrameResult:
    emotions: List[EmotionResult]
    classes: Tuple[str, ...]
    frame_index: int

    def __len__(self) -> int:
        return len(self.emotions)


class EmotionRecognizer(StreamingModel):
    package = "online_emotion"

    def __init__(self, model: str = "hsemotion", *, weights=None, runtime: str = "auto",
                 device="auto", precision: str = "auto", input_size=None, batch_max: int = 32,
                 cache_dir=None, warmup: bool = True, display: bool = False) -> None:
        self.batch_max = int(batch_max)
        super().__init__(family=get_family(model), weights=weights, runtime=runtime, device=device,
                         precision=precision, input_size=input_size, cache_dir=cache_dir,
                         warmup=warmup, display=display)
        meta = self._resolved.meta
        self._classes = class_names(int(meta.get("classes", 8)))
        self._va = bool(meta.get("va", False))

    @property
    def classes(self) -> Tuple[str, ...]:
        return self._classes

    def _ctx(self):
        return {"classes": self._classes, "va": self._va}

    def _infer_logits(self, batch):
        import torch

        n = batch.shape[0]
        if n <= self.batch_max:
            return self._infer(batch)[0]
        chunks = [self._infer(batch[i:i + self.batch_max])[0] for i in range(0, n, self.batch_max)]
        return torch.cat(chunks, dim=0)

    # -- crops in -> emotions out -----------------------------------------
    def predict(self, crops: Union[Frame, Sequence[Frame]], *,
                frame_index: Optional[int] = None) -> EmotionFrameResult:
        crops_list = list(crops) if isinstance(crops, (list, tuple)) else [crops]
        idx = frame_index if frame_index is not None else self.stats.frames
        if not crops_list:
            self.stats.update(0.0, 0.0, 0.0)
            return EmotionFrameResult([], self._classes, idx)
        with Stopwatch() as t_pre:
            chw = [load_frame(c, self._device) for c in crops_list]
            batch = self._family.prepare_crops(chw, self._input_size)
        with Stopwatch() as t_inf:
            logits = self._infer_logits(batch)
        with Stopwatch() as t_post:
            payload = self._family.postprocess(logits, self._ctx(), {})
        self.stats.update(t_pre.elapsed, t_inf.elapsed, t_post.elapsed)
        return EmotionFrameResult([EmotionResult(**d) for d in payload], self._classes, idx)

    __call__ = predict

    def predict_on_boxes(self, frame: Frame, boxes, *, frame_index: Optional[int] = None) -> EmotionFrameResult:
        idx = frame_index if frame_index is not None else self.stats.frames
        boxes = np.asarray(boxes, dtype="float32").reshape(-1, 4)
        if boxes.shape[0] == 0:
            self.stats.update(0.0, 0.0, 0.0)
            return EmotionFrameResult([], self._classes, idx)
        with Stopwatch() as t_pre:
            frame_chw = load_frame(frame, self._device)
            crops = crop_resize(frame_chw, boxes, self._input_size)  # (M,3,H,W) BGR, on device
            batch = self._family.normalize_batch(crops)
        with Stopwatch() as t_inf:
            logits = self._infer_logits(batch)
        with Stopwatch() as t_post:
            payload = self._family.postprocess(logits, self._ctx(), {})
        self.stats.update(t_pre.elapsed, t_inf.elapsed, t_post.elapsed)
        return EmotionFrameResult([EmotionResult(**d) for d in payload], self._classes, idx)

    # -- stream / file -----------------------------------------------------
    def run_source(self, source, *, boxes_provider=None, is_stream: Optional[bool] = None,
                   display: Optional[bool] = None, max_frames: Optional[int] = None,
                   save_video: Optional[str] = None, print_stats: bool = True
                   ) -> Iterator[Tuple[FrameRef, EmotionFrameResult]]:
        show = self.display if display is None else display
        src = open_source(source, is_stream=is_stream, max_frames=max_frames)
        every = max(1, int(src.fps or 30))
        writer = None
        win = "online_emotion"
        try:
            for fref in src:
                if boxes_provider is not None:
                    boxes = np.asarray(boxes_provider(fref.image), dtype="float32").reshape(-1, 4)
                    res = self.predict_on_boxes(fref.image, boxes, frame_index=fref.index)
                else:
                    boxes = None
                    res = self.predict(fref.image, frame_index=fref.index)
                if show or save_video:
                    canvas = fref.image.copy()
                    self._draw(canvas, res, boxes)
                    draw_hud(canvas, self.stats.format_line())
                    if save_video:
                        if writer is None:
                            import cv2

                            h, w = canvas.shape[:2]
                            writer = cv2.VideoWriter(str(save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     src.fps or 30.0, (w, h))
                        writer.write(canvas)
                    if show:
                        import cv2

                        cv2.imshow(win, canvas)
                        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                            break
                self.stats.dropped = src.dropped
                if print_stats and fref.index % every == 0:
                    top = res.emotions[0].label if res.emotions else "-"
                    print(f"\r[online_emotion] {self.stats.format_line()} top={top:<10}", end="", flush=True)
                yield fref, res
        finally:
            if print_stats:
                print(f"\r[online_emotion] {self.stats.format_line()} (done){' ' * 12}")
            if writer is not None:
                writer.release()
            if show:
                try:
                    import cv2

                    cv2.destroyWindow(win)
                except Exception:
                    pass
            src.release()

    def _draw(self, img: np.ndarray, res: EmotionFrameResult, boxes) -> None:
        if boxes is not None and len(boxes) == len(res.emotions):
            for box, emo in zip(boxes, res.emotions):
                draw_box_label(img, box, hash_color(emo.label), f"{emo.label} {float(emo.probs.max()):.2f}")
        elif res.emotions:
            import cv2

            emo = res.emotions[0]
            cv2.putText(img, f"{emo.label} {float(emo.probs.max()):.2f}", (10, img.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # -- export / discovery ------------------------------------------------
    @classmethod
    def export(cls, model: str = "hsemotion", *, weights=None, runtime: str, device="auto",
               precision: str = "auto", input_size=None, cache_dir=None) -> InferenceConfig:
        rec = cls(model, weights=weights, runtime=runtime, device=device, precision=precision,
                  input_size=input_size, cache_dir=cache_dir, warmup=False)
        cfg = rec.config
        rec.close()
        return cfg

    @staticmethod
    def available_models() -> list:
        return _available_models()

    @staticmethod
    def available_weights(model: str = "hsemotion") -> list:
        return get_family(model).available_weights()

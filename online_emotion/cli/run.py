"""``python -m online_emotion.cli.run`` — run emotion recognition on a video/stream.

Without face boxes, each frame is treated as a single crop (whole-frame emotion);
in a real pipeline, feed face crops via ``predict_on_boxes`` / ``boxes_provider``.
Prints rolling FPS/latency to the terminal.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("online_emotion.cli.run",
                                description="Streaming emotion recognition on a video file or live stream.")
    p.add_argument("--source", help="video path | webcam index (e.g. 0) | rtsp/http url")
    p.add_argument("--model", default="hsemotion")
    p.add_argument("--weights", default=None, help="weight key or a local artifact path")
    p.add_argument("--runtime", default="auto", choices=["auto", "torch", "torchscript", "onnx", "trt"])
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "int8"])
    p.add_argument("--batch-max", type=int, default=32)
    p.add_argument("--stream", action="store_true", help="treat --source as a live stream")
    p.add_argument("--display", action="store_true", help="show an overlay window (press q/ESC to quit)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-video", default=None)
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--list-weights", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .. import EmotionRecognizer

    args = build_parser().parse_args(argv)
    if args.list_models:
        print("\n".join(EmotionRecognizer.available_models()))
        return 0
    if args.list_weights:
        print("\n".join(EmotionRecognizer.available_weights(args.model)))
        return 0
    if not args.source:
        print("error: --source is required (or use --list-models / --list-weights)", file=sys.stderr)
        return 2

    emo = EmotionRecognizer(args.model, weights=args.weights, runtime=args.runtime, device=args.device,
                            precision=args.precision, batch_max=args.batch_max, display=args.display)
    frames = 0
    for _fref, _res in emo.run_source(args.source, is_stream=(True if args.stream else None),
                                      display=args.display, max_frames=args.max_frames,
                                      save_video=args.save_video):
        frames += 1
    print(json.dumps({"frames": frames, "stats": emo.stats.as_dict(), "config": emo.config.to_dict()},
                     indent=2, default=str))
    emo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

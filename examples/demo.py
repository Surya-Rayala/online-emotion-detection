"""Minimal demo.

Standalone (whole-frame emotion, for a quick check):
    python examples/demo.py --source /path/to/video.mp4 --display

Real usage chains a face detector (install online-face-detection too):
    # for frame_ref, fr in face.run_source(src):
    #     er = emo.predict_on_boxes(frame_ref.image, fr.boxes)
"""
from __future__ import annotations

import argparse

from online_emotion import EmotionRecognizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--runtime", default="auto")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    emo = EmotionRecognizer("hsemotion", weights=args.weights, runtime=args.runtime, device="auto")
    print("config:", emo.config.summary(), "| classes:", emo.classes)

    for frame_ref, result in emo.run_source(args.source, is_stream=(True if args.stream else None),
                                            display=args.display, max_frames=args.max_frames):
        pass  # result.emotions aligned to crops; here whole-frame -> one emotion

    print("final stats:", emo.stats.as_dict())
    emo.close()


if __name__ == "__main__":
    main()

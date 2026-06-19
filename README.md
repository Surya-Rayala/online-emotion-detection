# online-emotion-detection

Streaming, **frame-by-frame** facial emotion recognition for real-time pipelines:
face crops in, structured emotions out. Runs under **torch / torchscript / onnx / tensorrt**
with export-once caching, on CPU, CUDA, Apple Silicon (MPS), and Jetson.

```python
from online_emotion import EmotionRecognizer
emo = EmotionRecognizer("hsemotion", device="auto")
res = emo.predict_on_boxes(frame, boxes)    # res.emotions[i].label / .probs
```

> **Models today:** HSEmotion (Savchenko EfficientNet). More emotion families plug in via the
> registry — coming later.

---

## Install

Pick a runtime extra **and** an OpenCV build (`gui` for desktop, `headless` for servers/Docker/Jetson):

```bash
pip install "online-emotion-detection[torch,gui]"        # desktop (enables cv2 windows / --display)
pip install "online-emotion-detection[torch,headless]"   # servers, Docker, Jetson (no GTK/X11)
```

`[torch]` is the default runtime (pulls the HSEmotion weights) and works on CPU, CUDA, and Mac (MPS).
**OpenCV is not bundled by default** — choose `[gui]` (`opencv-python`) or `[headless]`
(`opencv-python-headless`). On **Jetson** the GUI build pulls X11/GTK and frequently fails to build
against the CUDA toolchain (`nvcc` errors), so use `[headless]` — or install with **neither** and rely
on JetPack's system OpenCV (also use JetPack's torch/onnxruntime there). Other backends (`onnx`,
`tensorrt`, serving, client) are optional extras — see [Install options](#install-options).
(Prefer `uv`? See [Misc](#misc).)

---

## Use it (Python)

The natural unit is a **face crop** (or a batch of crops — batching is the speed win).

```python
from online_emotion import EmotionRecognizer

emo = EmotionRecognizer(
    "hsemotion",        # model family (the only one today)
    device="auto",      # "auto" (CUDA > MPS > CPU) | "cpu" | "cuda" | "mps"
    runtime="auto",     # "auto" | "torch" | "torchscript" | "onnx" | "trt"
    batch_max=16,       # max crops per batched forward pass
)

# (a) a full frame + face boxes (crops on-device — no host round-trip):
res = emo.predict_on_boxes(frame, boxes)     # boxes: (N,4) xyxy from a face detector

# (b) face crops directly (one crop, or a list — batched automatically):
res = emo(face_crop)
res = emo([crop_a, crop_b, crop_c])

for e in res.emotions:
    print(e.label, e.probs.max(), e.valence, e.arousal)   # valence/arousal only for *_va_mtl weights
```

**Inputs — what to pass:**
- `frame` / `face_crop` / each crop: a NumPy array in **BGR**, shape **`(H, W, 3)`**, dtype
  **`uint8`** (OpenCV's native format), **or** a `torch.Tensor` of shape **`(3, H, W)`**.
- `boxes` (for `predict_on_boxes`): an array of shape **`(N, 4)`** in **xyxy** pixel coordinates
  (e.g. `FaceFrameResult.boxes` from `online-face-detection`).

**Output —** `EmotionFrameResult(emotions=[EmotionResult(label, label_index, probs (C,), valence?,
arousal?)], classes, frame_index)`, aligned to the input crop/box order. `emo.stats.as_dict()`
gives rolling fps / latency.

Chain it after a face detector (each model independent):

```python
from online_face import FaceDetector
from online_emotion import EmotionRecognizer

with FaceDetector("retinaface") as face, EmotionRecognizer("hsemotion", batch_max=16) as emo:
    for frame_ref, fr in face.run_source("video.mp4"):
        er = emo.predict_on_boxes(frame_ref.image, fr.boxes)
```

### Or from the terminal

```bash
# whole-frame emotion on a video file, with a window + FPS (press q/ESC to quit)
online-emotion --source video.mp4 --device auto --runtime auto --batch-max 16 --display

# webcam (index 0) as a live stream
online-emotion --source 0 --device auto --runtime auto --stream --display

# discover weights, or see every flag
online-emotion --list-weights
online-emotion --help
```

`online-emotion` == `python -m online_emotion.cli.run`. All flags:
`--source` (file path | webcam index | rtsp/http url) · `--device {auto,cpu,cuda,mps}` ·
`--runtime {auto,torch,torchscript,onnx,trt}` · `--batch-max N` · `--stream` · `--display` ·
`--save-video PATH` · `--max-frames N` · `--list-weights`. The CLI runs **whole-frame** emotion
(no face boxes); in a real pipeline, feed boxes via `predict_on_boxes` (above).

---

## Models & weights

`model` is the **family** (`hsemotion` — the only one today); `weights` is the actual weight (a
key, auto-downloaded by the `hsemotion` package, or an artifact path). `weights=None` → default.

| weights key | classes | valence/arousal | notes |
|-------------|---------|-----------------|-------|
| `enet_b0_8_best_vgaf` *(default)* | 8 | – | EfficientNet-B0, fast, edge-friendly |
| `enet_b0_8_best_afew` | 8 | – | EfficientNet-B0 |
| `enet_b2_8` | 8 | – | EfficientNet-B2, higher accuracy |
| `enet_b0_8_va_mtl` | 8 | ✅ | also regresses valence & arousal |

The 8 AffectNet classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.

---

## Runtimes & the export cache

`runtime="auto"` picks the best backend per device: **Jetson/CUDA → tensorrt** (else onnx-CUDA),
**macOS → torch (MPS)**, **CPU → onnx/torch**. The first non-torch run builds the artifact and
caches it under `~/.cache/online_inference/`; later runs load it. Batching many crops into one
call is the main speed win — `batch_max` caps the batch.

> On **macOS the ONNX backend uses CoreML**, which recompiles when the batch size (number of
> faces) changes — slow for a variable face count. On Mac, prefer `torch` (the default) or
> `torchscript` for emotion; ONNX/TensorRT are the edge (CUDA/Jetson) path.

---

## Install options

`[torch]` is all most people need. Add extras for other backends. **Extras are additive** — if
you already installed `[torch]`, running `pip install "online-emotion-detection[serve]"` later
just adds those packages (it won't reinstall torch). Install several at once:
`pip install "online-emotion-detection[torch,onnx,serve]"`.

| Extra | Adds | Install when you want to… |
|-------|------|---------------------------|
| `[gui]` | opencv-python | desktop OpenCV (enables `cv2.imshow` / `--display`) |
| `[headless]` | opencv-python-headless | OpenCV for servers / Docker / Jetson (no GTK/X11) |
| `[torch]` | torch, torchvision, hsemotion, timm | **default** runtime + model weights |
| `[onnx]`  | onnxruntime, onnx, onnxsim | run or export the ONNX backend |
| `[trt]`   | our ONNX export path + fp16 converter (**not** tensorrt) | run the TensorRT backend — install TensorRT yourself first, **see the note below** |
| `[serve]` | fastapi, uvicorn, python-multipart | host the model as an HTTP service (below) |
| `[client]` | requests, websockets | call a remote service (torch-free, below) |

**Which do I actually need?**
- **Always pick an OpenCV build:** `[gui]` (desktop) or `[headless]` (servers/Docker/Jetson). OpenCV is **not** in core, so combine it with a runtime/client extra, e.g. `[torch,gui]` or `[client,headless]`. On Jetson, prefer `[headless]` or rely on JetPack's system OpenCV (install neither).
- `pip install online-emotion-detection` (no `[...]`) → **core only** (numpy/tqdm); no OpenCV, no runtime — can't run. Use only when OpenCV/torch are provided another way (e.g. JetPack).
- `[torch]` → the **foundation**; required to run the model locally (CPU/CUDA/MPS) and it pulls the model weights. Start here (e.g. `[torch,gui]`).
- `[onnx]` / `[trt]` → **add** a backend *on top of* torch (they don't replace it). `[trt]` pulls `[onnx]` + an fp16 converter but **not tensorrt** — install TensorRT for your CUDA yourself (see the note below).
- `[serve]` → runs the model in-process, so it needs torch + an OpenCV build: `pip install "online-emotion-detection[torch,serve,gui]"`.
- `[client]` → the **only torch-free** path; add an OpenCV build: `pip install "online-emotion-detection[client,headless]"`.

> [!CAUTION]
> **TensorRT setup.** `[trt]` adds our ONNX export path + fp16 converter but **does not install TensorRT** — its PyPI wheel always grabs the newest CUDA build (e.g. cu13), which won't match your system. Install TensorRT yourself (plus a matching CUDA build of torch). On an NVIDIA machine:
> 1. **Check your CUDA toolkit:** run `nvcc --version` and note the **release** it reports (e.g. `release 12.1`) — that's your installed CUDA toolkit, the version everything below must match. If `nvcc` isn't found, the toolkit isn't installed (install it, or use `[onnx]`/`[torch]` instead).
> 2. **Check torch matches:** `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`. You need a **CUDA build of torch ≥ 2.1** whose CUDA matches step 1 and prints `True`. If not, reinstall it for your CUDA:
>    ```bash
>    pip uninstall -y torch torchvision
>    # pick the matching command from https://pytorch.org/get-started/previous-versions/  (example: CUDA 12.1)
>    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
>    ```
> 3. **Install TensorRT for your CUDA** following NVIDIA's official guide — pick the build matching step 1, don't rely on a default: <https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html>
> 4. **Then install this package:**
>    ```bash
>    pip install "online-emotion-detection[trt]"   # adds our ONNX export path; uses the TensorRT from step 3
>    ```
> On **Jetson**, skip all this — TensorRT ships with JetPack (see [Jetson](#jetson-jetpack)).

---

## (Optional) Serve it as an HTTP service

Besides the in-process use above, the model can run as its own HTTP service (local or cloud) and
be called by URL. Needs the `[serve]` extra (adds only fastapi/uvicorn on top of `[torch]`).

```bash
pip install "online-emotion-detection[serve]"
online-emotion-serve --model hsemotion --device auto --runtime auto --host 127.0.0.1 --port 8002
```

**Server flags** (all optional; defaults shown): `--model hsemotion` ·
`--weights KEY|PATH` (default: family default) · `--device {auto,cpu,cuda,mps}` ·
`--instances 1` · `--runtime {auto,torch,torchscript,onnx,trt}` · `--precision {auto,fp32,fp16,int8}` ·
`--batch-max 32` · `--input-size N` · `--host 127.0.0.1` · `--port 8002`.

`--instances` runs a pool of N recognizers so concurrent requests overlap (each runs in a worker
thread). A plain number (`--instances 4`) is **round-robined across all visible GPUs**
(`cuda:0, cuda:1, …`) when more than one is present — regardless of which single device the server
itself resolved to (`--device auto` just picks `cuda:0` as the main device; the pool still spreads
across `0..k-1`). If there is only **one** device (single GPU / MPS / CPU) all N copies share it
(N× memory, time-sharing the same compute, so little gain). A per-GPU map
(`--instances cuda:0=2,cuda:1=1`) instead pins exact counts to specific GPUs. So multi-instance
mainly helps across **multiple GPUs**; the threadpool overlap helps regardless. `/meta` reports the
resolved `instances` and `instance_devices`.

| Route | What it does |
|-------|--------------|
| `GET /meta` | self-describing: inputs `frame: image` + `boxes: ndarray`; output `emotions`; plus the class list, `instances`/`instance_devices` |
| `GET /healthz` | readiness + resolved runtime/device |
| `POST /predict` | multipart: a `frame` image part + a `boxes` `.npy` part → JSON `{outputs, stats}` |
| `POST /predict_crops` | multipart: one repeated `crops` image part per face → JSON `{outputs, stats}` (avoids re-sending the whole frame) |
| `WS /stream` | persistent socket: per frame a `{"n":K}` control message + K binary crops → one JSON reply |

```bash
curl http://127.0.0.1:8002/meta
```

### Client — `[client]` proxy (torch-free: numpy + requests + websockets, plus an OpenCV build)

```bash
pip install "online-emotion-detection[client,headless]"   # servers/Jetson; use [client,gui] on desktop
```

```python
from online_emotion.client import EmotionClient

emo = EmotionClient(
    "http://127.0.0.1:8002",   # service URL (local or cloud)
    encode="jpeg",             # wire format (default): "jpeg" (small, no measurable accuracy loss) | "png" (lossless)
    quality=90,                # JPEG quality (ignored for png)
    max_side=None,             # downscale frame before sending on the predict_on_boxes path; boxes scaled to match
    timeout=30,
)
```

There are exactly **two** client tools — pick by workload:

| Tool | What it does | Use when |
|------|--------------|----------|
| **`EmotionClient`** (unary) — `predict_on_boxes(frame, boxes)` or `predict_on_crops(EmotionClient.crop_boxes(frame, boxes))` | one request per call; crops-only avoids re-sending the whole frame (payload scales with #faces). Also `healthz()`/`meta()` | one-offs, or a fresh connection per call |
| **`EmotionStream`** (async, long-lived) — `await s.push_boxes(frame, boxes, meta)` (or `push_crops`) + `async for result, meta in s.results()` | holds WebSockets across a **pool of endpoints**, results **as completed (out of order)** tagged with metadata; auto-scales to a target fps/latency | continuous / many streams over a network |

`crop_boxes` is a static helper (pure numpy slicing); each result maps **crop i → emotion i**. `encode="jpeg"` (default) is far smaller than PNG with no measurable accuracy loss (crops are resized to 224²).

**What you get back** — an `EmotionFrameResult`:

| field | type | meaning |
|---|---|---|
| `emotions` | `list[EmotionResult]` | one per input crop/box, **in input order** |
| `classes` | `tuple[str, …]` | the class label set (7- or 8-class AffectNet) |

Each **client** `EmotionResult` has: `label` (str), `score` (float, top-1 probability), `label_index` (int), `valence`/`arousal` (floats, only for `*_va_mtl` weights, else `None`). `len(result)` == number of faces. `EmotionClient` returns one `EmotionFrameResult`; `EmotionStream.results()` yields `(EmotionFrameResult, meta)`.

> Note: the in-process `EmotionRecognizer` returns the **full** softmax vector as `EmotionResult.probs` `(C,)` (plus `classes`, `frame_index`); the wire **client** sends only the top-1 `score` to keep payloads small.

### `EmotionStream` — continuous, multi-stream, auto-scaling

```python
import asyncio, time
from online_emotion import EmotionStream

async def run(items):
    # `items` yields (frame, boxes, meta). `meta` is ANY object you choose — it is NOT sent to
    # the server; it's kept client-side and returned with that frame's result so YOU can tell
    # which output is which (e.g. meta = {"stream": cam_id, "i": frame_index, "t": time.time()}).
    async with EmotionStream(
        ["http://gpu0:8002"],     # a single URL, or a POOL of replicas/GPUs
        target_fps=30,            # controller aims for this throughput…
        target_latency_ms=150,    # …while keeping end-to-end latency under this
        max_inflight=64,          # ceiling on frames in flight across the pool
    ) as stream:

        async def pump():                                  # producer: push (frame, boxes) in
            for frame, boxes, meta in items:               # boxes: (N,4) xyxy from a face detector
                await stream.push_boxes(frame, boxes, meta=meta)  # crops client-side; sends only crops
            await stream.aclose()                          # end-of-stream (drains in-flight first)
        asyncio.create_task(pump())

        # consumer: results arrive AS COMPLETED — i.e. OUT OF ORDER. Each item is a 2-tuple:
        #     (result: EmotionFrameResult, meta)   # `meta` is exactly the object you pushed
        # Identify / reassemble using `meta` — never assume arrival order == push order.
        async for result, meta in stream.results():
            cam, idx = meta["stream"], meta["i"]           # <- how you know which frame this is
            # result.emotions -> list[EmotionResult], one per box, IN BOX ORDER (crop i -> emotion i)
            # result.classes  -> tuple[str, …]  the label set
            for box_i, e in enumerate(result.emotions):    # e.label / e.score / e.label_index / e.valence / e.arousal
                handle(cam, idx, box_i, e.label, e.score)
            # stream.stats() -> live dict: {conns, target_inflight, rtt_ms, infer_ms, queue_depth, bound, ...}
```

Same adaptive controller as the face package: ramps in-flight while latency stays under target, backs off when the server's queue grows (model-bound) or latency breaches; scales across a URL **pool** when given one. Same device? Use the in-process `EmotionRecognizer` (it crops on-device).

---

## Misc

### Composing face → emotion (no combined package)

The two packages stay independent — there is intentionally no combined package; the glue is two lines of your code.

```python
# In-process (same device) — fastest; crops never leave the device:
from online_face import FaceDetector
from online_emotion import EmotionRecognizer
det, emo = FaceDetector("retinaface"), EmotionRecognizer("hsemotion")
r = det(frame); emotions = emo.predict_on_boxes(frame, r.boxes)

# Over the wire (two services) — send only face crops to emotion, not the whole frame twice:
from online_face.client import FaceClient
from online_emotion.client import EmotionClient
face, emo = FaceClient(FACE_URL), EmotionClient(EMO_URL)
r = face(frame)
emotions = emo.predict_on_crops(EmotionClient.crop_boxes(frame, r.boxes))
```

### Install with uv

Same as pip, with `uv` (include an OpenCV build — `gui` or `headless`):
```bash
uv add "online-emotion-detection[torch,gui]"            # into a uv project
uv pip install "online-emotion-detection[torch,headless]"    # into the active venv
```

### Jetson (JetPack)

On Jetson the whole GPU stack (CUDA / cuDNN / TensorRT) is part of **JetPack**, and torch/onnxruntime
must be NVIDIA's Jetson wheels — the PyPI `[torch]`/`[onnx]` wheels are x86_64 and won't use the GPU.

**1. Pick a JetPack version.**

| Board | JetPack | Stack |
|-------|---------|-------|
| Orin (AGX/NX/Nano) | **6.x** | CUDA 12.6 · TensorRT 10.3 · PyTorch 2.6 wheel |
| Xavier / older | **5.1.x** | torch ~2.1 |

Both are above this package's `torch>=2.1` floor.

**2. Install these into the JetPack env first** — from NVIDIA's
[PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) guide, or the
[jetson-ai-lab](https://pypi.jetson-ai-lab.io) wheel index matched to your JetPack (e.g.
`--index-url https://pypi.jetson-ai-lab.io/jp6/cu126` for JetPack 6.x):

- `torch`, `torchvision` — the **Jetson GPU wheels** (not from PyPI)
- `onnxruntime-gpu` — only if you'll use the ONNX backend
- `hsemotion`, `timm` — the HSEmotion model source (plain `pip`, not Jetson-specific)
- `opencv-python`, `numpy` — usually already present in JetPack; install if missing
- TensorRT — **already installed by JetPack** (nothing to do)

**3. Then install this package with NO runtime extra**, so it uses the system ones:

```bash
pip install online-emotion-detection      # no [torch] / [onnx]
```

It adapts to whatever JetPack provides and keys each cached TensorRT engine to the exact board.

> **Conflicting model requirements?** One Jetson has a single system torch/TRT. If two models need
> incompatible torch/CUDA, run each as its own [HTTP service](#optional-serve-it-as-an-http-service)
> (e.g. an `nvcr.io/nvidia/l4t-pytorch` container) and compose them by URL with the `[client]` proxy.

### Pre-build & cache an artifact

Optional — otherwise built on first use. Choose the runtime you'll deploy with for the target device:
```bash
online-emotion-export --model hsemotion --weights enet_b0_8_best_vgaf --runtime trt --device auto
```
Flags: `--model` · `--weights KEY|PATH` · `--runtime {torchscript,onnx,trt}` ·
`--device {auto,cpu,cuda,mps}` · `--precision {auto,fp32,fp16,int8}` · `--input-size N`.

## License

MIT © Surya Chand Rayala

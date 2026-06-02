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

```bash
pip install "online-emotion-detection[torch]"
```

That's all you need for most setups — `[torch]` is the default runtime and pulls the HSEmotion
model weights; it works on CPU, CUDA, and Mac (MPS). Other backends (`onnx`, `tensorrt`, serving)
are **optional extras** you can add anytime — see [Install options](#install-options). (Prefer
`uv`? See [Misc](#misc).)

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
| `[torch]` | torch, torchvision, hsemotion, timm | **default** runtime + model weights |
| `[onnx]`  | onnxruntime, onnx, onnxsim | run or export the ONNX backend |
| `[trt]`   | tensorrt (also pulls `[onnx]`) | build/run TensorRT engines on NVIDIA — **see the TensorRT note below** |
| `[serve]` | fastapi, uvicorn, python-multipart | host the model as an HTTP service (below) |
| `[client]` | requests | call a remote service (torch-free, below) |

**Which do I actually need?**
- `pip install online-emotion-detection` (no `[...]`) → **core only** (numpy/opencv); **no runtime, can't run inference**. Use this only when torch is provided another way (e.g. Jetson/JetPack wheels).
- `[torch]` → the **foundation**; required to run the model locally (CPU/CUDA/MPS) and it pulls the model weights. Start here.
- `[onnx]` / `[trt]` → **add** a backend *on top of* torch (they don't replace it). `[trt]` also pulls `[onnx]`; for TensorRT you also need a matching **CUDA build of torch** — **see the note below**.
- `[serve]` → runs the model in-process, so it needs torch too: `pip install "online-emotion-detection[torch,serve]"`.
- `[client]` → the **only torch-free** one — it just calls a remote service, so `pip install "online-emotion-detection[client]"` **alone is enough**.

> [!CAUTION]
> **TensorRT setup — `[trt]` alone is not enough.** It installs the bindings; you still need a matching CUDA + a CUDA build of torch. On an NVIDIA machine:
> 1. **Check your CUDA:** run `nvidia-smi` and note the **CUDA Version** it reports (your GPU/driver's CUDA). No NVIDIA GPU → use `[onnx]`/`[torch]` instead.
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
`--runtime {auto,torch,torchscript,onnx,trt}` · `--precision {auto,fp32,fp16,int8}` ·
`--batch-max 32` · `--input-size N` · `--host 127.0.0.1` · `--port 8002`.

| Route | What it does |
|-------|--------------|
| `GET /meta` | self-describing: inputs `frame: image` + `boxes: ndarray`; output `emotions`; plus the class list |
| `GET /healthz` | readiness + resolved runtime/device |
| `POST /predict` | multipart: a `frame` image part + a `boxes` `.npy` part → JSON `{outputs, stats}` |

```bash
curl http://127.0.0.1:8002/meta
```

**Call it from another process** with the torch-free `[client]` proxy (mirrors `predict_on_boxes`):

```bash
pip install "online-emotion-detection[client]"
```
```python
from online_emotion.client import EmotionClient

emo = EmotionClient(
    "http://127.0.0.1:8002",   # the service URL (local or cloud)
    encode="png",              # how frames go over the wire: "png" (lossless) | "jpeg" (smaller)
    timeout=30,                # request timeout, seconds
)
res = emo.predict_on_boxes(frame, boxes)   # res.emotions[i].label / .score
emo.meta()                                 # the service's /meta;  emo.healthz() -> readiness
```

Compose with a hosted face service into a pipeline by URL — see
**[../testing-pipeline](../testing-pipeline)** for a ready-to-run example.

---

## Misc

### Install with uv

Same as pip, with `uv`:
```bash
uv add "online-emotion-detection[torch]"            # into a uv project
uv pip install "online-emotion-detection[torch]"    # into the active venv
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

"""On-disk artifact cache: export once, then reuse forever.

Layout under the cache root (``$ONLINE_INFERENCE_CACHE`` or
``~/.cache/online_inference``)::

    weights/                       downloaded source checkpoints
    artifacts/<key>/               one exported artifact + export_meta.json

The key folds in a device identity so a TensorRT engine built on one GPU never
silently loads on another. Builds happen under a file lock and are published by
atomic rename, so concurrent pipeline workers never collide or read a partial
artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from .config import InferenceConfig
from .device import device_identity
from .logging import get_logger

_log = get_logger("cache")


def default_cache_dir() -> Path:
    env = os.getenv("ONLINE_INFERENCE_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "online_inference"


def _sha1_short(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    runtime: str
    meta: Dict[str, Any]


class ArtifactCache:
    def __init__(self, cache_dir: str | os.PathLike | None = None) -> None:
        self.root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
        self.weights_dir = self.root / "weights"
        self.artifacts_dir = self.root / "artifacts"

    # -- key ---------------------------------------------------------------
    def key(self, *, package: str, model: str, weights_fingerprint: str, runtime: str,
            precision: str, input_shape: Sequence[int], dynamic: bool, device: str) -> str:
        payload = {
            "schema": "online-artifact-v1",
            "package": package,
            "model": model,
            "weights_fingerprint": weights_fingerprint,
            "runtime": runtime,
            "precision": precision,
            "input_shape": list(input_shape),
            "dynamic": bool(dynamic),
            "device_identity": device_identity(runtime, device),
        }
        return _sha1_short(json.dumps(payload, sort_keys=True))

    # -- lock --------------------------------------------------------------
    @contextmanager
    def _lock(self, lockfile: Path, timeout: float = 1800.0):
        lockfile.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        fd = None
        while True:
            try:
                fd = os.open(str(lockfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.perf_counter() - start > timeout:
                    raise TimeoutError(f"timed out waiting for export lock {lockfile}")
                time.sleep(0.25)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            try:
                lockfile.unlink()
            except OSError:
                pass

    # -- ensure ------------------------------------------------------------
    def ensure(self, *, config: InferenceConfig, runtime: str, dynamic: bool,
               export_fn: Callable[[Path], str], source_weights: str = "") -> ArtifactRef:
        """Return a cached artifact, building it (under lock) on a miss.

        ``export_fn(tmp_dir)`` must write the artifact into ``tmp_dir`` and
        return its filename. ``export_meta.json`` is written alongside it.
        """
        key = self.key(package=config.package, model=config.model,
                       weights_fingerprint=config.weights_fingerprint, runtime=runtime,
                       precision=config.precision, input_shape=(1, 3, *config.input_size),
                       dynamic=dynamic, device=config.device)
        art_dir = self.artifacts_dir / key

        hit = self._try_load(art_dir, runtime, config.device)
        if hit is not None:
            return hit

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        with self._lock(self.artifacts_dir / f"{key}.lock"):
            hit = self._try_load(art_dir, runtime, config.device)  # re-check after lock
            if hit is not None:
                return hit
            tmp = Path(tempfile.mkdtemp(prefix=f"{key}_", dir=str(self.artifacts_dir)))
            _log.info("exporting %s artifact -> %s", runtime, art_dir)
            t0 = time.perf_counter()
            artifact_name = export_fn(tmp)
            meta = {
                **config.to_dict(),
                "artifact": artifact_name,
                "runtime": runtime,
                "dynamic": bool(dynamic),
                "device_identity": device_identity(runtime, config.device),
                "source_weights": str(source_weights),
                "elapsed_sec": round(time.perf_counter() - t0, 2),
                "created_unix": time.time(),
            }
            (tmp / "export_meta.json").write_text(json.dumps(meta, indent=2))
            if art_dir.exists():
                shutil.rmtree(art_dir, ignore_errors=True)
            os.replace(tmp, art_dir)  # atomic publish on same filesystem
            _log.info("export complete in %.1fs: %s", meta["elapsed_sec"], art_dir / artifact_name)
            return ArtifactRef(art_dir / artifact_name, runtime, meta)

    def _try_load(self, art_dir: Path, runtime: str, device: str):
        meta_path = art_dir / "export_meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            art = art_dir / meta["artifact"]
            if not art.exists():
                return None
            if runtime == "trt" and meta.get("device_identity") != device_identity(runtime, device):
                _log.info("cached engine device mismatch; will rebuild")
                return None
            _log.info("cache hit: %s", art)
            return ArtifactRef(art, runtime, meta)
        except Exception:
            return None

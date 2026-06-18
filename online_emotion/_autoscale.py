"""Adaptive concurrency / connection controller for the async streaming session.

Pure-python, no deps, no asyncio — easy to unit-test. The session feeds it one
sample per reply (round-trip ms, server service ms, server infer ms, server queue
depth); once per tick it decides:

  * ``target_inflight`` — total frames to keep in flight (AIMD toward a target fps
    while latency stays under target; back off when latency breaches or throughput
    plateaus). Capped by Little's law so we never chase more inflight than the
    latency budget can absorb.
  * ``n_conn`` — how many WebSocket connections to hold across the endpoint pool.
    Scale OUT when we're network-bound and still under target (more sockets add
    real parallel transport / spread model load across replicas); HOLD when every
    endpoint's model is saturated (more inflight just grows queues).

Mirrored verbatim in online_face/_autoscale.py (packages stay independent).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class _EMA:
    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> None:
        self.value = x if self.value is None else (self.alpha * x + (1 - self.alpha) * self.value)

    def get(self, default: float = 0.0) -> float:
        return default if self.value is None else self.value


@dataclass
class ScaleState:
    target_inflight: int
    n_conn: int
    bound: str            # "network" | "model" | "saturated"
    tp: float             # throughput (replies/sec) over the last tick
    rtt_ms: float
    srv_ms: float
    infer_ms: float
    queue_depth: float
    at_target: bool       # meeting target fps within the latency budget


class AutoScaler:
    def __init__(self, *, target_fps=None, target_latency_ms=None,
                 min_inflight: int = 1, max_inflight: int = 64,
                 max_connections: int = 1, ema_alpha: float = 0.3) -> None:
        self.target_fps = target_fps
        self.target_latency_ms = target_latency_ms
        self.min_inflight = max(1, int(min_inflight))
        self.max_inflight = max(self.min_inflight, int(max_inflight))
        self.max_connections = max(1, int(max_connections))
        self.target_inflight = self.min_inflight
        self.n_conn = 1
        self.bound = "network"
        self._rtt = _EMA(ema_alpha)
        self._srv = _EMA(ema_alpha)
        self._infer = _EMA(ema_alpha)
        self._qd = _EMA(ema_alpha)
        self._completions = 0
        self._last_tp = 0.0
        self._last_rtt = float("inf")
        self._up_streak = 0
        self._down_streak = 0

    def observe(self, *, rtt_ms: float, srv_ms: float, infer_ms: float, queue_depth: float) -> None:
        self._completions += 1
        self._rtt.update(rtt_ms)
        self._srv.update(max(0.0, srv_ms))
        self._infer.update(max(0.0, infer_ms))
        self._qd.update(max(0.0, queue_depth))

    def tick(self, dt_s: float) -> ScaleState:
        dt_s = max(1e-3, dt_s)
        tp = self._completions / dt_s
        rtt = self._rtt.get()
        srv = self._srv.get()
        infer = self._infer.get()
        qd = self._qd.get()
        net = max(0.0, rtt - srv)                      # transport + serialization

        latency_ok = (self.target_latency_ms is None) or (rtt == 0.0) or (rtt <= self.target_latency_ms)
        below_fps = (self.target_fps is None) or (tp < self.target_fps)
        rising = tp > self._last_tp * 1.05

        # --- in-flight AIMD ---
        if latency_ok and (rising or below_fps):
            self.target_inflight = min(self.max_inflight,
                                       self.target_inflight + max(1, self.target_inflight // 8))
        elif (not latency_ok) or (tp <= self._last_tp and rtt > self._last_rtt):
            self.target_inflight = max(self.min_inflight, int(self.target_inflight * 0.7))
        if self.target_fps and rtt > 0:                # Little's-law cap: C ≈ fps * latency
            cap = max(self.min_inflight, math.ceil(self.target_fps * rtt / 1000.0) + 1)
            self.target_inflight = min(self.target_inflight, cap)

        # --- bound classification ---
        model_bound = qd > 1.0 and infer >= net
        if (not latency_ok) and self.target_inflight <= self.min_inflight:
            self.bound = "saturated"
        elif model_bound:
            self.bound = "model"
        else:
            self.bound = "network"

        # --- connection (pool) scaling, with 2-tick hysteresis ---
        near_cap = self.target_inflight >= self.max_inflight or self.target_inflight >= self.n_conn * 8
        want_up = below_fps and latency_ok and near_cap and self.n_conn < self.max_connections
        want_down = self.n_conn > 1 and self.target_inflight <= max(self.min_inflight, (self.n_conn - 1) * 2)
        self._up_streak = self._up_streak + 1 if want_up else 0
        self._down_streak = self._down_streak + 1 if want_down else 0
        if self._up_streak >= 2:
            self.n_conn += 1
            self._up_streak = 0
        elif self._down_streak >= 2:
            self.n_conn -= 1
            self._down_streak = 0

        at_target = (not below_fps) and latency_ok
        self._last_tp = tp
        self._last_rtt = rtt
        self._completions = 0
        return ScaleState(self.target_inflight, self.n_conn, self.bound, tp, rtt, srv, infer, qd, at_target)

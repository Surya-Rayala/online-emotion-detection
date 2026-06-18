"""Deterministic unit tests for the adaptive controller (pure logic, no I/O)."""
from __future__ import annotations

from online_emotion._autoscale import AutoScaler


def _feed(s, n, **sample):
    for _ in range(n):
        s.observe(**sample)
    return s.tick(0.5)


def test_inflight_ramps_up_under_low_latency():
    s = AutoScaler(target_fps=1e6, target_latency_ms=100, min_inflight=1, max_inflight=64)
    start = s.target_inflight
    for _ in range(5):
        _feed(s, 10, rtt_ms=10, srv_ms=5, infer_ms=3, queue_depth=0)
    assert s.target_inflight > start


def test_backs_off_when_latency_exceeds_target():
    s = AutoScaler(target_fps=1e6, target_latency_ms=50, min_inflight=1, max_inflight=64)
    for _ in range(4):
        _feed(s, 10, rtt_ms=10, srv_ms=5, infer_ms=3, queue_depth=0)
    high = s.target_inflight
    for _ in range(3):
        _feed(s, 10, rtt_ms=200, srv_ms=150, infer_ms=140, queue_depth=5)
    assert s.target_inflight < high


def test_bound_model_vs_network():
    model = _feed(AutoScaler(target_latency_ms=1e6), 10, rtt_ms=50, srv_ms=45, infer_ms=44, queue_depth=4)
    assert model.bound == "model"
    net = _feed(AutoScaler(target_latency_ms=1e6), 10, rtt_ms=50, srv_ms=5, infer_ms=4, queue_depth=0)
    assert net.bound == "network"


def test_pool_scales_out_when_capped_and_below_target():
    s = AutoScaler(target_fps=1e6, target_latency_ms=1e6,
                   min_inflight=8, max_inflight=8, max_connections=3)
    for _ in range(6):
        _feed(s, 5, rtt_ms=5, srv_ms=2, infer_ms=1, queue_depth=0)
    assert 2 <= s.n_conn <= 3


def test_inflight_respects_bounds():
    s = AutoScaler(target_fps=1e6, target_latency_ms=1e6, min_inflight=2, max_inflight=5)
    for _ in range(20):
        _feed(s, 10, rtt_ms=1, srv_ms=0.5, infer_ms=0.2, queue_depth=0)
    assert 2 <= s.target_inflight <= 5

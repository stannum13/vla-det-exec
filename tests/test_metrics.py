import sys
import numpy as np
import pytest

sys.path.insert(0, ".")
from exp0.metrics import compute_metrics


def _make_outputs(n=100, chunk=10, act_dim=7, identical=True):
    base = np.random.rand(chunk, act_dim).astype(np.float32)
    if identical:
        return [base.copy() for _ in range(n)]
    return [np.random.rand(chunk, act_dim).astype(np.float32) for _ in range(n)]


def test_hash_equal_when_identical():
    r = compute_metrics(_make_outputs(identical=True), [5.0] * 100)
    assert r["hash_equal"] is True


def test_hash_equal_when_different():
    r = compute_metrics(_make_outputs(identical=False), [5.0] * 100)
    assert r["hash_equal"] is False


def test_max_diff_zero_when_identical():
    r = compute_metrics(_make_outputs(identical=True), [5.0] * 100)
    assert r["max_diff"] == pytest.approx(0.0)


def test_latency_stats():
    latencies = list(range(1, 101))
    r = compute_metrics(_make_outputs(identical=True), latencies)
    assert r["latency_p50_ms"] == pytest.approx(50.5, abs=1.0)
    assert r["latency_p99_ms"] == pytest.approx(99.0, abs=1.5)
    assert r["latency_max_ms"] == 100


def test_deadline_miss_rate():
    latencies = [50.0] * 90 + [150.0] * 10
    r = compute_metrics(_make_outputs(identical=True), latencies)
    assert r["deadline_miss_rate"] == pytest.approx(0.1)


def test_token_entropy_none_by_default():
    r = compute_metrics(_make_outputs(identical=True), [5.0] * 100)
    assert r["token_entropy"] is None


def test_token_entropy_computed_when_provided():
    tokens = [0] * 50 + [1] * 50
    r = compute_metrics(_make_outputs(identical=True), [5.0] * 100, tokens=tokens)
    assert r["token_entropy"] == pytest.approx(1.0, abs=0.01)

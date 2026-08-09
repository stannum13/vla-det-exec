"""Metrics for Experiment 0 determinism and latency measurement."""
from __future__ import annotations
import hashlib
import math
from collections import Counter

import numpy as np

DEADLINE_MS = 100.0  # 10 Hz policy rate


def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


def compute_metrics(
    outputs: list,
    latencies: list,
    tokens: list | None = None,
) -> dict:
    """Compute determinism and latency metrics over N inference repetitions.

    Args:
        outputs:   List of float32 ndarrays, each shape (chunk_size, action_dim).
        latencies: Per-inference latencies in milliseconds, same length as outputs.
        tokens:    Optional list of integer code tokens (VQ-BeT only).

    Returns:
        Dict with: hash_equal, max_diff, rms_diff, token_entropy,
        latency_p50_ms, latency_p95_ms, latency_p99_ms, latency_max_ms, deadline_miss_rate.
    """
    assert len(outputs) > 0 and len(latencies) == len(outputs)

    hashes = [_md5(o) for o in outputs]
    hash_equal = len(set(hashes)) == 1

    ref = outputs[0]
    diffs = [np.abs(o - ref) for o in outputs[1:]]
    max_diff = float(max(d.max() for d in diffs)) if diffs else 0.0
    rms_diff = float(np.sqrt(np.mean(np.stack(diffs) ** 2))) if diffs else 0.0

    token_entropy = None
    if tokens is not None:
        counts = Counter(tokens)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        token_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    lat = np.array(latencies, dtype=np.float64)
    return {
        "hash_equal": hash_equal,
        "max_diff": max_diff,
        "rms_diff": rms_diff,
        "token_entropy": token_entropy,
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_p99_ms": float(np.percentile(lat, 99)),
        "latency_max_ms": float(lat.max()),
        "deadline_miss_rate": float((lat > DEADLINE_MS).mean()),
    }

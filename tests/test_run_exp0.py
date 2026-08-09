"""Integration test: harness runs with mock adapters, 2 snaps x 5 reps."""
import sys
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, ".")
from exp0.adapters.base import BaseAdapter


class _MockAdapter(BaseAdapter):
    def __init__(self, mode: str) -> None:
        super().__init__(mode)
        self._fixed = np.zeros((10, 7), dtype=np.float32)
        self.token_log: list[int] = []

    def infer(self, snapshot: dict) -> np.ndarray:
        self.token_log.append(0)
        return self._fixed.copy() if self.mode == "deterministic" \
            else np.random.rand(10, 7).astype(np.float32)


def test_run_produces_summary_csv(tmp_path, monkeypatch):
    import exp0.run_exp0 as harness

    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(harness, "N_REPS", 5)
    monkeypatch.setattr(harness, "N_SNAPS", 2)

    snap_dir = Path("exp0/snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        np.savez(snap_dir / f"snap_{i:02d}.npz",
                 agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8),
                 wrist_rgb=np.zeros((128, 128, 3), dtype=np.uint8),
                 state=np.zeros(9, dtype=np.float32),
                 instruction="test", demo_id=0, timestep=0)

    monkeypatch.setattr(harness, "load_adapter",
                        lambda model, cond, args=None: _MockAdapter(mode=cond))
    harness.run()

    import pandas as pd
    df = pd.read_csv(tmp_path / "summary.csv")
    assert set(df["model"].unique()) == {"act", "vqbet", "molmoact2"}
    assert set(df["condition"].unique()) == {"stochastic", "deterministic"}
    assert "hash_equal" in df.columns and "latency_p99_ms" in df.columns
    assert len(df) == 3 * 2 * 2

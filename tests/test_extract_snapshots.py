# tests/test_extract_snapshots.py
import numpy as np
import pytest
from pathlib import Path

SNAP_DIR = Path("exp0/snapshots")


def test_snapshots_exist():
    snaps = sorted(SNAP_DIR.glob("snap_*.npz"))
    assert len(snaps) == 50, f"Expected 50 snapshots, found {len(snaps)}"


def test_snapshot_keys():
    snap = np.load(SNAP_DIR / "snap_00.npz", allow_pickle=True)
    required = {"agentview_rgb", "wrist_rgb", "state", "instruction", "demo_id", "timestep"}
    assert required.issubset(set(snap.files))


def test_snapshot_shapes():
    snap = np.load(SNAP_DIR / "snap_00.npz", allow_pickle=True)
    assert snap["agentview_rgb"].shape == (128, 128, 3)
    assert snap["wrist_rgb"].shape == (128, 128, 3)
    assert snap["state"].shape == (9,)
    assert snap["state"].dtype == np.float32
    assert snap["agentview_rgb"].dtype == np.uint8

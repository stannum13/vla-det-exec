#!/usr/bin/env python3
"""Extract 50 observation snapshots from LIBERO-Spatial demo HDF5s.

Strategy: 5 tasks x 10 snapshots each from demo_0, at even timestep stride.
"""
import sys
import h5py
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/libero_spatial")
OUT_DIR = Path("exp0/snapshots")

SNAPS_PER_TASK = 10
NUM_TASKS = 5


def extract_from_hdf5(hdf5_path: Path, snap_indices: list) -> list:
    snapshots = []
    with h5py.File(hdf5_path, "r") as f:
        raw = f.attrs.get("problem_info", b"pick and place task")
        instruction = raw.decode() if isinstance(raw, bytes) else str(raw)
        demo = f["data/demo_0"]
        agentview = demo["obs/agentview_rgb"][:]           # (T, 128, 128, 3)
        wrist = demo["obs/robot0_eye_in_hand_image"][:]    # (T, 128, 128, 3)
        joint_pos = demo["obs/robot0_joint_pos"][:]        # (T, 7)
        gripper = demo["obs/robot0_gripper_qpos"][:]       # (T, 2)
        T = agentview.shape[0]
        for t in snap_indices:
            t = min(t, T - 1)
            state = np.concatenate([joint_pos[t], gripper[t]]).astype(np.float32)
            snapshots.append({
                "agentview_rgb": agentview[t].astype(np.uint8),
                "wrist_rgb": wrist[t].astype(np.uint8),
                "state": state,
                "instruction": instruction,
                "demo_id": 0,
                "timestep": int(t),
            })
    return snapshots


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hdf5_files = sorted(DATA_DIR.glob("*.hdf5"))[:NUM_TASKS]
    if not hdf5_files:
        print(f"ERROR: No HDF5 files found in {DATA_DIR}. Run setup/setup.sh first.")
        sys.exit(1)

    snap_idx = 0
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            T = f["data/demo_0/obs/agentview_rgb"].shape[0]
        stride = max(1, T // SNAPS_PER_TASK)
        timesteps = [stride * i for i in range(SNAPS_PER_TASK)]

        for snap in extract_from_hdf5(hdf5_path, timesteps):
            np.savez(OUT_DIR / f"snap_{snap_idx:02d}.npz", **snap)
            snap_idx += 1

    print(f"Saved {snap_idx} snapshots to {OUT_DIR}/")


if __name__ == "__main__":
    main()

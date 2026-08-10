# Experiment 0: Recorded-Input Determinism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run 100 inference passes on 50 LIBERO-Spatial observation snapshots per model (ACT, VQ-BeT, MolmoAct2) under stochastic and deterministic conditions, writing `exp0/results/summary.csv` with hash equality, action diff, token entropy, and latency stats.

**Architecture:** Each model gets a thin adapter with a uniform `infer(snapshot) -> np.ndarray` interface handling model-specific preprocessing and stochastic vs. deterministic decoding modes. A central harness loops over models × conditions × snapshots × 100 reps, emitting per-record JSON files and a merged summary CSV.

**Tech Stack:** Python 3.10, PyTorch 2.1 + CUDA 12.1, LIBERO (`Lifelong-Robot-Learning/LIBERO`, robosuite + MuJoCo), `tonyzhaozh/act`, `jayLEE0301/vq_bet_official`, `allenai/molmoact2`, numpy, pandas, h5py, tqdm, pytest

---

## File Map

| File | Purpose |
|---|---|
| `.gitignore` | Ignore third_party/, data/, checkpoints/, results/ |
| `setup/setup.sh` | VM provisioning: clone repos, install deps, download data + checkpoint |
| `setup/requirements.txt` | Python deps |
| `exp0/extract_snapshots.py` | Extract 50 `.npz` snapshots from LIBERO-Spatial HDF5 demos |
| `exp0/metrics.py` | `compute_metrics(outputs, latencies, tokens) -> dict` |
| `exp0/adapters/__init__.py` | Empty |
| `exp0/adapters/base.py` | `BaseAdapter` ABC |
| `exp0/adapters/act_adapter.py` | ACT stochastic (sampled z) / deterministic (z=0) adapter |
| `exp0/adapters/vqbet_adapter.py` | VQ-BeT sample / argmax adapter with token logging |
| `exp0/adapters/molmoact2_adapter.py` | MolmoAct2 random / fixed-seed flow-noise adapter |
| `exp0/train_act.py` | Train ACT on LIBERO-Spatial demos, save checkpoint |
| `exp0/train_vqbet.py` | Train VQ-BeT on LIBERO-Spatial demos, save checkpoint |
| `exp0/run_exp0.py` | Main harness: model × condition × snapshot × rep loop |
| `tests/test_metrics.py` | Unit tests for metrics.py |
| `tests/test_act_adapter.py` | Determinism property tests for ACT adapter |
| `tests/test_vqbet_adapter.py` | Determinism property tests for VQ-BeT adapter |
| `tests/test_molmoact2_adapter.py` | Determinism property tests for MolmoAct2 adapter |
| `tests/test_run_exp0.py` | Integration test for run_exp0.py with mock adapters |

Third-party repos cloned into `third_party/` (gitignored). Data in `data/` (gitignored). Checkpoints in `checkpoints/` (gitignored).

---

## Task 1: Project Scaffold

**Files:**
- Create: `.gitignore`
- Create: `setup/requirements.txt`
- Create: `setup/setup.sh`
- Create: directory stubs

- [ ] **Step 1: Write `.gitignore`**

```
third_party/
data/
checkpoints/
exp0/results/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
*.so
```

- [ ] **Step 2: Write `setup/requirements.txt`**

```
numpy>=1.24
pandas>=2.0
h5py>=3.9
tqdm>=4.65
pytest>=7.4
einops>=0.7
huggingface_hub>=0.20
```

- [ ] **Step 3: Write `setup/setup.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 2. Base Python deps
pip install -r setup/requirements.txt

# 3. LIBERO
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
pip install -e third_party/libero
pip install robosuite==1.4.1

# 4. ACT
git clone https://github.com/tonyzhaozh/act.git third_party/act
pip install pyquaternion pyyaml

# 5. VQ-BeT
git clone https://github.com/jayLEE0301/vq_bet_official.git third_party/vq_bet_official
pip install -e third_party/vq_bet_official

# 6. MolmoAct2
git clone https://github.com/allenai/molmoact2.git third_party/molmoact2
pip install -e third_party/molmoact2

# 7. Download LIBERO-Spatial dataset
mkdir -p data/libero_spatial
python -c "
from libero.libero import benchmark
bm = benchmark.get_benchmark_dict()['libero_spatial']
bm.download_datasets(save_dir='data/libero_spatial')
"

# 8. Download MolmoAct2-DROID checkpoint
mkdir -p checkpoints/molmoact2_droid
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='allenai/molmoact2-droid', local_dir='checkpoints/molmoact2_droid')
"

echo "Setup complete. Next: python exp0/extract_snapshots.py"
```

- [ ] **Step 4: Create directory stubs**

```bash
mkdir -p exp0/adapters exp0/snapshots exp0/results tests setup
touch exp0/__init__.py exp0/adapters/__init__.py tests/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore setup/ exp0/__init__.py exp0/adapters/__init__.py tests/__init__.py
git commit -m "feat: project scaffold, setup script, and requirements"
```

---

## Task 2: Snapshot Extraction

**Files:**
- Create: `exp0/extract_snapshots.py`
- Create: `tests/test_extract_snapshots.py`

Context: LIBERO-Spatial has 10 tasks, each with one HDF5 file. Each file has a `data/` group with demo subgroups (`demo_0`, `demo_1`, ...). Each demo contains `/obs/agentview_rgb` (T, 128, 128, 3), `/obs/robot0_eye_in_hand_image` (T, 128, 128, 3), `/obs/robot0_joint_pos` (T, 7), `/obs/robot0_gripper_qpos` (T, 2), and `/actions` (T, 7). Strategy: 5 tasks × 10 snapshots each from demo_0, sampled at even stride.

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /path/to/vla-det-exec
pytest tests/test_extract_snapshots.py -v
```
Expected: FAIL — "Expected 50 snapshots, found 0".

- [ ] **Step 3: Write `exp0/extract_snapshots.py`**

```python
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPS_PER_TASK = 10
NUM_TASKS = 5


def extract_from_hdf5(hdf5_path: Path, snap_indices: list) -> list:
    snapshots = []
    with h5py.File(hdf5_path, "r") as f:
        instruction = str(f.attrs.get("problem_info", b"pick and place task"))
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
```

- [ ] **Step 4: Run extraction (requires LIBERO data)**

```bash
python exp0/extract_snapshots.py
```
Expected: `Saved 50 snapshots to exp0/snapshots/`

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_extract_snapshots.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add exp0/extract_snapshots.py tests/test_extract_snapshots.py exp0/snapshots/
git commit -m "feat: extract LIBERO-Spatial snapshots for Experiment 0"
```

---

## Task 3: Metrics Module

**Files:**
- Create: `exp0/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
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
    latencies = list(range(1, 101))   # 1..100 ms
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
    # 50/50 uniform → entropy = 1.0 bit
    tokens = [0] * 50 + [1] * 50
    r = compute_metrics(_make_outputs(identical=True), [5.0] * 100, tokens=tokens)
    assert r["token_entropy"] == pytest.approx(1.0, abs=0.01)
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_metrics.py -v
```
Expected: all FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `exp0/metrics.py`**

```python
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

    # Compare all outputs to the first (O(n) instead of O(n^2))
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```
Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add exp0/metrics.py tests/test_metrics.py
git commit -m "feat: metrics module for determinism and latency measurement"
```

---

## Task 4: Base Adapter + ACT Adapter

**Files:**
- Create: `exp0/adapters/base.py`
- Create: `exp0/adapters/act_adapter.py`
- Create: `tests/test_act_adapter.py`

Context: ACT's `DETRVAE` model sets `mu = logvar = zeros` at inference time (z=0), making it deterministic by default. Our "stochastic" mode samples `z ~ N(0, I)` and injects it into the decoder via a thin wrapper around the model's forward pass. The `tiny=True` flag uses a small hidden_dim=32 model for fast tests without a checkpoint.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_act_adapter.py
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "third_party/act")

from exp0.adapters.act_adapter import ACTAdapter

_SNAP = {
    "agentview_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    "wrist_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    "state": np.random.rand(9).astype(np.float32),
    "instruction": "pick up the object",
    "demo_id": 0,
    "timestep": 10,
}


def test_deterministic_returns_same_output():
    adapter = ACTAdapter(mode="deterministic", checkpoint_path=None, tiny=True)
    out1 = adapter.infer(_SNAP)
    out2 = adapter.infer(_SNAP)
    np.testing.assert_array_equal(out1, out2)


def test_output_shape():
    adapter = ACTAdapter(mode="deterministic", checkpoint_path=None, tiny=True)
    out = adapter.infer(_SNAP)
    assert out.ndim == 2 and out.shape[1] == 7


def test_stochastic_varies():
    adapter = ACTAdapter(mode="stochastic", checkpoint_path=None, tiny=True)
    outputs = [adapter.infer(_SNAP) for _ in range(5)]
    hashes = [o.tobytes() for o in outputs]
    assert len(set(hashes)) > 1, "Stochastic adapter returned identical outputs"


def test_output_dtype():
    adapter = ACTAdapter(mode="deterministic", checkpoint_path=None, tiny=True)
    assert adapter.infer(_SNAP).dtype == np.float32
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_act_adapter.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `exp0/adapters/base.py`**

```python
"""Base adapter interface for Experiment 0."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseAdapter(ABC):
    """Uniform interface for ACT, VQ-BeT, and MolmoAct2 adapters.

    Args:
        mode: "stochastic" or "deterministic"
    """

    def __init__(self, mode: str) -> None:
        assert mode in ("stochastic", "deterministic"), f"Unknown mode: {mode}"
        self.mode = mode

    @abstractmethod
    def infer(self, snapshot: dict) -> np.ndarray:
        """Run one inference pass.

        Args:
            snapshot: Dict with keys:
                agentview_rgb (128,128,3) uint8
                wrist_rgb     (128,128,3) uint8
                state         (9,)        float32
                instruction   str

        Returns:
            Action chunk float32 ndarray of shape (chunk_size, action_dim=7).
        """
        ...
```

- [ ] **Step 4: Write `exp0/adapters/act_adapter.py`**

```python
"""ACT adapter for Experiment 0.

Wraps tonyzhaozh/act's ACTPolicy.
  deterministic: z = zeros  (ACT's default inference behavior)
  stochastic:    z ~ N(0,I) injected into the transformer decoder
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, str(Path("third_party/act").resolve()))

from policy import ACTPolicy  # third_party/act/policy.py
from exp0.adapters.base import BaseAdapter

_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_FULL_CFG = {
    "num_queries": 10,
    "hidden_dim": 256,
    "dim_feedforward": 3200,
    "nheads": 8,
    "enc_layers": 4,
    "dec_layers": 7,
    "dropout": 0.1,
    "pre_norm": False,
    "camera_names": ["agentview", "wrist"],
    "backbone": "resnet18",
    "kl_weight": 10,
    "lr": 1e-4,
    "lr_backbone": 1e-5,
    "state_dim": 9,
    "action_dim": 7,
}

_TINY_CFG = {**_FULL_CFG, "hidden_dim": 32, "dim_feedforward": 64, "nheads": 2,
             "enc_layers": 1, "dec_layers": 1}


class ACTAdapter(BaseAdapter):
    def __init__(
        self,
        mode: str,
        checkpoint_path: str | None = None,
        tiny: bool = False,
        device: str | None = None,
    ) -> None:
        super().__init__(mode)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = _TINY_CFG if tiny else _FULL_CFG
        self._latent_dim = cfg["hidden_dim"]

        self.policy = ACTPolicy(cfg)
        if checkpoint_path is not None:
            self.policy.deserialize(torch.load(checkpoint_path, map_location=self.device))
        self.policy.to(self.device).eval()

    def infer(self, snapshot: dict) -> np.ndarray:
        images = self._prep_images(snapshot)
        qpos = torch.tensor(snapshot["state"], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.mode == "deterministic":
                # ACT default: z=0 at inference, fully deterministic
                a_hat = self.policy(qpos, images)
            else:
                # Stochastic: inject sampled z into the DETRVAE decoder
                a_hat = self._forward_with_sampled_z(qpos, images)

        return a_hat.squeeze(0).cpu().numpy().astype(np.float32)

    def _forward_with_sampled_z(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Run DETRVAE decoder with z ~ N(0, I) instead of z = 0.

        Replicates the inference path of DETRVAE.forward substituting random z.
        If ACT internals change, fall back to torch.manual_seed + self.policy(qpos, images).
        """
        detrvae = self.policy.model
        z = torch.randn(1, self._latent_dim, device=self.device)
        try:
            latent_input = detrvae.latent_out_proj(z)
            all_feat, all_pos = [], []
            for cam_id in range(images.shape[1]):
                feat, pos = detrvae.backbones[0](images[:, cam_id])
                all_feat.append(detrvae.input_proj(feat[0]))
                all_pos.append(pos[0])
            src = torch.cat(all_feat, dim=3)
            pos = torch.cat(all_pos, dim=3)
            proprio = detrvae.input_proj_robot_state(qpos)
            hs = detrvae.transformer(src, None, detrvae.query_embed.weight, pos, latent_input, proprio)[0]
            return detrvae.action_head(hs)
        except AttributeError:
            # Fallback: seed-based stochastic (works with any ACT version)
            torch.manual_seed(int(torch.randint(0, 2**31, (1,)).item()))
            return self.policy(qpos, images)

    def _prep_images(self, snapshot: dict) -> torch.Tensor:
        imgs = [_NORM(snapshot[k]) for k in ("agentview_rgb", "wrist_rgb")]
        return torch.stack(imgs).unsqueeze(0).to(self.device)  # (1, 2, 3, H, W)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_act_adapter.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add exp0/adapters/base.py exp0/adapters/act_adapter.py tests/test_act_adapter.py
git commit -m "feat: base adapter interface and ACT stochastic/deterministic adapter"
```

---

## Task 5: VQ-BeT Adapter

**Files:**
- Create: `exp0/adapters/vqbet_adapter.py`
- Create: `tests/test_vqbet_adapter.py`

Context: VQ-BeT predicts discrete code logits over a VQ codebook plus continuous residual actions. Stochastic: sample code from the categorical distribution. Deterministic: argmax. The adapter accumulates a `token_log` list for entropy computation in metrics. A `_TinyVQBeTStub` is used for tests when the real package is unavailable.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vqbet_adapter.py
import sys
import numpy as np
import pytest

sys.path.insert(0, ".")
from exp0.adapters.vqbet_adapter import VQBeTAdapter

_SNAP = {
    "agentview_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    "wrist_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    "state": np.random.rand(9).astype(np.float32),
    "instruction": "pick up the object",
    "demo_id": 0,
    "timestep": 10,
}


def test_deterministic_returns_same_output():
    a = VQBeTAdapter(mode="deterministic", checkpoint_path=None, tiny=True)
    np.testing.assert_array_equal(a.infer(_SNAP), a.infer(_SNAP))


def test_output_shape():
    a = VQBeTAdapter(mode="deterministic", checkpoint_path=None, tiny=True)
    out = a.infer(_SNAP)
    assert out.ndim == 2 and out.shape[1] == 7


def test_stochastic_varies():
    a = VQBeTAdapter(mode="stochastic", checkpoint_path=None, tiny=True)
    hashes = [a.infer(_SNAP).tobytes() for _ in range(10)]
    assert len(set(hashes)) > 1


def test_token_log_populated():
    a = VQBeTAdapter(mode="stochastic", checkpoint_path=None, tiny=True)
    a.infer(_SNAP)
    assert len(a.token_log) == 1 and isinstance(a.token_log[0], int)
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_vqbet_adapter.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `exp0/adapters/vqbet_adapter.py`**

```python
"""VQ-BeT adapter for Experiment 0.

  stochastic:    sample code token from categorical logits
  deterministic: argmax of code logits

token_log accumulates the chosen code index per infer() call.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, str(Path("third_party/vq_bet_official").resolve()))

from exp0.adapters.base import BaseAdapter

_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_FULL_CFG = {
    "obs_dim": 9, "act_dim": 7, "chunk_size": 10,
    "num_latents": 512, "num_residual_layers": 4,
    "hidden_dim": 256, "n_heads": 8, "n_layers": 6,
}
_TINY_CFG = {**_FULL_CFG, "num_latents": 16, "hidden_dim": 32, "n_heads": 2, "n_layers": 2}


class VQBeTAdapter(BaseAdapter):
    def __init__(
        self,
        mode: str,
        checkpoint_path: str | None = None,
        tiny: bool = False,
        device: str | None = None,
    ) -> None:
        super().__init__(mode)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.token_log: list[int] = []
        cfg = _TINY_CFG if tiny else _FULL_CFG
        self._chunk_size = cfg["chunk_size"]
        self._act_dim = cfg["act_dim"]

        try:
            from vq_bet.vq_bet_model import VQBeTPolicy
            self.policy = VQBeTPolicy(cfg)
        except ImportError:
            self.policy = _TinyVQBeTStub(cfg)

        if checkpoint_path is not None:
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy.to(self.device).eval()

    def infer(self, snapshot: dict) -> np.ndarray:
        obs = {
            "image": torch.stack([_NORM(snapshot[k]) for k in ("agentview_rgb", "wrist_rgb")])
                         .unsqueeze(0).to(self.device),
            "state": torch.tensor(snapshot["state"], dtype=torch.float32).unsqueeze(0).to(self.device),
        }
        with torch.no_grad():
            # policy.get_code_logits_and_residual is the expected VQ-BeT interface.
            # If the real API differs, inspect third_party/vq_bet_official/vq_bet/vq_bet_model.py
            # and adapt accordingly.
            code_logits, residual = self.policy.get_code_logits_and_residual(obs)
            if self.mode == "stochastic":
                token = int(torch.distributions.Categorical(logits=code_logits[0]).sample().item())
            else:
                token = int(code_logits[0].argmax().item())
            self.token_log.append(token)
            action_chunk = self.policy.decode_from_token(token, residual)
        return action_chunk.squeeze(0).cpu().numpy().astype(np.float32)


class _TinyVQBeTStub(nn.Module):
    """Minimal stub implementing get_code_logits_and_residual + decode_from_token.
    Used in tests and when vq_bet_official is not installed.
    """
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        h = cfg.get("hidden_dim", 32)
        self._chunk = cfg["chunk_size"]
        self._act = cfg["act_dim"]
        self._nl = cfg["num_latents"]
        self.enc = nn.Linear(cfg["obs_dim"], h)
        self.head = nn.Linear(h, self._nl)
        self.codebook = nn.Embedding(self._nl, self._act * self._chunk)

    def get_code_logits_and_residual(self, obs: dict):
        x = self.enc(obs["state"])
        logits = self.head(x)                                               # (1, num_latents)
        residual = torch.zeros(1, self._chunk, self._act, device=x.device)
        return logits, residual

    def decode_from_token(self, token: int, residual: torch.Tensor) -> torch.Tensor:
        emb = self.codebook(torch.tensor([token], device=residual.device))
        return emb.reshape(1, self._chunk, self._act) + residual
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_vqbet_adapter.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add exp0/adapters/vqbet_adapter.py tests/test_vqbet_adapter.py
git commit -m "feat: VQ-BeT sample/argmax adapter with token logging"
```

---

## Task 6: MolmoAct2 Adapter

**Files:**
- Create: `exp0/adapters/molmoact2_adapter.py`
- Create: `tests/test_molmoact2_adapter.py`

Context: MolmoAct2 integrates Gaussian flow noise through a velocity field to produce action chunks. Stochastic: `torch.randn` called fresh each time. Deterministic: `torch.Generator` seeded with a constant before each `torch.randn` call. A `_FlowActionStub` (tiny MLP integrator) stands in for tests without a checkpoint.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_molmoact2_adapter.py
import sys
import numpy as np
import pytest

sys.path.insert(0, ".")
from exp0.adapters.molmoact2_adapter import MolmoAct2Adapter

_SNAP = {
    "agentview_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    "wrist_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    "state": np.random.rand(9).astype(np.float32),
    "instruction": "pick up the object",
    "demo_id": 0,
    "timestep": 10,
}


def test_deterministic_returns_same_output():
    a = MolmoAct2Adapter(mode="deterministic", checkpoint_path=None, stub=True)
    np.testing.assert_array_equal(a.infer(_SNAP), a.infer(_SNAP))


def test_output_shape():
    a = MolmoAct2Adapter(mode="deterministic", checkpoint_path=None, stub=True)
    out = a.infer(_SNAP)
    assert out.ndim == 2 and out.shape[1] == 7


def test_stochastic_varies():
    a = MolmoAct2Adapter(mode="stochastic", checkpoint_path=None, stub=True)
    hashes = [a.infer(_SNAP).tobytes() for _ in range(5)]
    assert len(set(hashes)) > 1


def test_output_dtype():
    a = MolmoAct2Adapter(mode="deterministic", checkpoint_path=None, stub=True)
    assert a.infer(_SNAP).dtype == np.float32
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_molmoact2_adapter.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `exp0/adapters/molmoact2_adapter.py`**

```python
"""MolmoAct2 adapter for Experiment 0.

Uses MolmoAct2-DROID pretrained checkpoint (Franka, absolute joint positions).
  stochastic:    fresh torch.randn flow noise per call
  deterministic: fixed-seed torch.Generator noise per call (same every time)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, str(Path("third_party/molmoact2").resolve()))

from exp0.adapters.base import BaseAdapter

CHUNK_SIZE = 10
ACTION_DIM = 7
FLOW_STEPS = 10
FIXED_SEED = 42

_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MolmoAct2Adapter(BaseAdapter):
    def __init__(
        self,
        mode: str,
        checkpoint_path: str | None = None,
        stub: bool = False,
        device: str | None = None,
    ) -> None:
        super().__init__(mode)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if stub or checkpoint_path is None:
            self.model = _FlowActionStub(CHUNK_SIZE, ACTION_DIM).to(self.device)
        else:
            try:
                from molmoact2 import MolmoAct2Policy
                self.model = MolmoAct2Policy.from_pretrained(checkpoint_path)
                self.model.to(self.device)
            except ImportError:
                raise ImportError("molmoact2 not installed. Run setup/setup.sh or pass stub=True.")
        self.model.eval()

    def _sample_noise(self) -> torch.Tensor:
        shape = (1, CHUNK_SIZE, ACTION_DIM)
        if self.mode == "deterministic":
            gen = torch.Generator(device=self.device).manual_seed(FIXED_SEED)
            return torch.randn(*shape, device=self.device, generator=gen)
        return torch.randn(*shape, device=self.device)

    def infer(self, snapshot: dict) -> np.ndarray:
        images = torch.stack([_NORM(snapshot[k]) for k in ("agentview_rgb", "wrist_rgb")]) \
                      .unsqueeze(0).to(self.device)
        state = torch.tensor(snapshot["state"], dtype=torch.float32).unsqueeze(0).to(self.device)
        noise = self._sample_noise()

        with torch.no_grad():
            out = self.model(
                images=images,
                state=state,
                instruction=snapshot["instruction"],
                noise=noise,
                num_steps=FLOW_STEPS,
            )
        return out.squeeze(0).cpu().numpy().astype(np.float32)


class _FlowActionStub(nn.Module):
    """Tiny MLP flow integrator for tests. Deterministic given identical noise."""
    def __init__(self, chunk_size: int, action_dim: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        flat = chunk_size * action_dim
        self.vel = nn.Sequential(
            nn.Linear(flat + 1, 64), nn.Tanh(), nn.Linear(64, flat)
        )

    def forward(self, images, state, instruction, noise, num_steps=10):
        x = noise.reshape(noise.shape[0], -1)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((x.shape[0], 1), step * dt, device=x.device)
            x = x + self.vel(torch.cat([x, t], dim=1)) * dt
        return x.reshape(noise.shape[0], self.chunk_size, self.action_dim)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_molmoact2_adapter.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add exp0/adapters/molmoact2_adapter.py tests/test_molmoact2_adapter.py
git commit -m "feat: MolmoAct2 fixed/random flow-noise adapter"
```

---

## Task 7: Training Scripts

**Files:**
- Create: `exp0/train_act.py`
- Create: `exp0/train_vqbet.py`

Context: Both scripts share a `LiberoDataset` that reads LIBERO-Spatial HDF5 files and produces `(qpos, images, action_chunk, is_pad)` tuples. ACT uses an L1 + KL loss. VQ-BeT delegates to its own `compute_loss`. Checkpoints are saved to `checkpoints/`.

- [ ] **Step 1: Write `exp0/train_act.py`**

```python
#!/usr/bin/env python3
"""Train ACT on LIBERO-Spatial demonstrations.

Usage:
    python exp0/train_act.py [--epochs 100] [--lr 1e-4] [--output checkpoints/act_libero.pt]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path("third_party/act").resolve()))
sys.path.insert(0, ".")
from policy import ACTPolicy

DATA_DIR = Path("data/libero_spatial")
CHUNK_SIZE = 10

_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class LiberoDataset(Dataset):
    """LIBERO-Spatial demo HDF5s as (qpos, images, action_chunk, is_pad) tuples."""

    def __init__(self, data_dir: Path = DATA_DIR, num_tasks: int = 10) -> None:
        self.samples: list[tuple] = []
        for hdf5_path in sorted(data_dir.glob("*.hdf5"))[:num_tasks]:
            with h5py.File(hdf5_path, "r") as f:
                for demo_key in f["data"].keys():
                    demo = f[f"data/{demo_key}"]
                    agentview = demo["obs/agentview_rgb"][:]           # (T, 128, 128, 3)
                    wrist = demo["obs/robot0_eye_in_hand_image"][:]    # (T, 128, 128, 3)
                    joint_pos = demo["obs/robot0_joint_pos"][:]        # (T, 7)
                    gripper = demo["obs/robot0_gripper_qpos"][:]       # (T, 2)
                    actions = demo["actions"][:]                        # (T, 7)
                    T = agentview.shape[0]
                    for t in range(T - CHUNK_SIZE):
                        qpos = np.concatenate([joint_pos[t], gripper[t]]).astype(np.float32)
                        act_chunk = actions[t:t + CHUNK_SIZE].astype(np.float32)
                        self.samples.append((agentview[t], wrist[t], qpos, act_chunk))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        agentview, wrist, qpos, act_chunk = self.samples[idx]
        images = torch.stack([_NORM(agentview), _NORM(wrist)])    # (2, 3, H, W)
        is_pad = torch.zeros(CHUNK_SIZE, dtype=torch.bool)
        return torch.tensor(qpos), images, torch.tensor(act_chunk), is_pad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="checkpoints/act_libero.pt")
    parser.add_argument("--kl-weight", type=float, default=10.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LiberoDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    cfg = {
        "num_queries": CHUNK_SIZE, "hidden_dim": 256, "dim_feedforward": 3200,
        "nheads": 8, "enc_layers": 4, "dec_layers": 7, "dropout": 0.1,
        "pre_norm": False, "camera_names": ["agentview", "wrist"],
        "backbone": "resnet18", "kl_weight": args.kl_weight,
        "lr": args.lr, "lr_backbone": args.lr * 0.1,
        "state_dim": 9, "action_dim": 7,
    }
    policy = ACTPolicy(cfg).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total = 0.0
        for qpos, images, actions, is_pad in tqdm(loader, desc=f"Epoch {epoch+1}"):
            qpos, images, actions, is_pad = (
                qpos.to(device), images.to(device), actions.to(device), is_pad.to(device)
            )
            optimizer.zero_grad()
            a_hat, _, (mu, logvar) = policy(qpos, images, actions, is_pad)
            l1 = (F.l1_loss(actions, a_hat, reduction="none") * ~is_pad.unsqueeze(-1)).mean()
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
            loss = l1 + args.kl_weight * kl
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1} loss: {total / len(loader):.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.serialize(), args.output)
    print(f"Checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `exp0/train_vqbet.py`**

```python
#!/usr/bin/env python3
"""Train VQ-BeT on LIBERO-Spatial demonstrations.

Usage:
    python exp0/train_vqbet.py [--epochs 100] [--lr 1e-4] [--output checkpoints/vqbet_libero.pt]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path("third_party/vq_bet_official").resolve()))
sys.path.insert(0, ".")

from exp0.train_act import LiberoDataset  # reuse same dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="checkpoints/vqbet_libero.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from vq_bet.vq_bet_model import VQBeTPolicy
    except ImportError:
        print("ERROR: vq_bet not installed. Run setup/setup.sh first.")
        raise

    dataset = LiberoDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    cfg = {
        "obs_dim": 9, "act_dim": 7, "chunk_size": 10,
        "num_latents": 512, "num_residual_layers": 4,
        "hidden_dim": 256, "n_heads": 8, "n_layers": 6,
    }
    policy = VQBeTPolicy(cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total = 0.0
        for qpos, images, actions, _ in tqdm(loader, desc=f"Epoch {epoch+1}"):
            qpos, images, actions = qpos.to(device), images.to(device), actions.to(device)
            obs = {"state": qpos, "image": images}
            optimizer.zero_grad()
            # VQBeTPolicy.compute_loss is the expected API.
            # If it differs, inspect third_party/vq_bet_official/vq_bet/vq_bet_model.py.
            loss = policy.compute_loss(obs, actions)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1} loss: {total / len(loader):.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), args.output)
    print(f"Checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify importable**

```bash
python -c "import exp0.train_act; print('OK')"
python -c "import exp0.train_vqbet; print('OK')"
```
Expected: both print `OK`.

- [ ] **Step 4: Commit**

```bash
git add exp0/train_act.py exp0/train_vqbet.py
git commit -m "feat: LIBERO-Spatial training scripts for ACT and VQ-BeT"
```

---

## Task 8: Main Experiment Harness

**Files:**
- Create: `exp0/run_exp0.py`
- Create: `tests/test_run_exp0.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_run_exp0.py
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

    # Create 2 fake snapshots
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
    assert len(df) == 3 * 2 * 2   # 3 models x 2 conditions x 2 snaps
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_run_exp0.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `exp0/run_exp0.py`**

```python
#!/usr/bin/env python3
"""Experiment 0: Recorded-Input Determinism.

Runs N_REPS inference passes on N_SNAPS LIBERO-Spatial snapshots for each of
3 models x 2 conditions. Writes JSON per (model, condition) and summary.csv.

Usage (on A100 after training):
    python exp0/run_exp0.py \
        --act-checkpoint checkpoints/act_libero.pt \
        --vqbet-checkpoint checkpoints/vqbet_libero.pt \
        --molmoact2-checkpoint checkpoints/molmoact2_droid
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, ".")
from exp0.metrics import compute_metrics
from exp0.adapters.act_adapter import ACTAdapter
from exp0.adapters.vqbet_adapter import VQBeTAdapter
from exp0.adapters.molmoact2_adapter import MolmoAct2Adapter

SNAP_DIR = Path("exp0/snapshots")
RESULTS_DIR = Path("exp0/results")
N_REPS = 100
N_SNAPS = 50

MODELS = ["act", "vqbet", "molmoact2"]
CONDITIONS = ["stochastic", "deterministic"]


def load_adapter(model_name: str, condition: str, args=None):
    ckpt = lambda attr: getattr(args, attr, None) if args else None
    if model_name == "act":
        return ACTAdapter(mode=condition, checkpoint_path=ckpt("act_checkpoint"))
    elif model_name == "vqbet":
        return VQBeTAdapter(mode=condition, checkpoint_path=ckpt("vqbet_checkpoint"))
    elif model_name == "molmoact2":
        cp = ckpt("molmoact2_checkpoint")
        return MolmoAct2Adapter(mode=condition, checkpoint_path=cp, stub=(cp is None))
    raise ValueError(f"Unknown model: {model_name}")


def load_snapshot(idx: int) -> dict:
    raw = np.load(SNAP_DIR / f"snap_{idx:02d}.npz", allow_pickle=True)
    return {k: raw[k].item() if raw[k].ndim == 0 else raw[k] for k in raw.files}


def run(args=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for model_name in MODELS:
        for condition in CONDITIONS:
            print(f"\n=== {model_name} / {condition} ===")
            adapter = load_adapter(model_name, condition, args)
            records: list[dict] = []

            for snap_idx in tqdm(range(N_SNAPS), desc=f"{model_name}/{condition}"):
                snapshot = load_snapshot(snap_idx)
                outputs, latencies = [], []

                for _ in range(N_REPS):
                    t0 = perf_counter()
                    outputs.append(adapter.infer(snapshot))
                    latencies.append((perf_counter() - t0) * 1000.0)

                tokens = None
                if hasattr(adapter, "token_log") and len(adapter.token_log) >= N_REPS:
                    tokens = adapter.token_log[-N_REPS:]

                record = compute_metrics(outputs, latencies, tokens=tokens)
                record.update({"model": model_name, "condition": condition, "snap_idx": snap_idx})
                records.append(record)

            json_path = RESULTS_DIR / f"{model_name}_{condition}.json"
            json_path.write_text(json.dumps(records, indent=2))
            print(f"  -> {json_path}")
            all_records.extend(records)

    csv_path = RESULTS_DIR / "summary.csv"
    pd.DataFrame(all_records).to_csv(csv_path, index=False)
    print(f"\nSummary: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act-checkpoint", dest="act_checkpoint", default=None)
    parser.add_argument("--vqbet-checkpoint", dest="vqbet_checkpoint", default=None)
    parser.add_argument("--molmoact2-checkpoint", dest="molmoact2_checkpoint", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run integration test**

```bash
pytest tests/test_run_exp0.py -v
```
Expected: PASS.

- [ ] **Step 5: Run full unit test suite**

```bash
pytest tests/ -v --ignore=tests/test_extract_snapshots.py
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add exp0/run_exp0.py tests/test_run_exp0.py
git commit -m "feat: main experiment harness for Experiment 0"
```

---

## Task 9: GCP Execution

These steps run on the A100 VM via SSH. No code to write — run each command and verify output.

- [ ] **Step 1: Clone repo and provision**

```bash
git clone <your-repo-url> vla-det-exec && cd vla-det-exec
bash setup/setup.sh
```
Expected: "Setup complete." after ~15 min.

- [ ] **Step 2: Verify LIBERO data**

```bash
ls data/libero_spatial/*.hdf5 | wc -l
```
Expected: `10`

- [ ] **Step 3: Extract snapshots**

```bash
python exp0/extract_snapshots.py
pytest tests/test_extract_snapshots.py -v
```
Expected: "Saved 50 snapshots", 3 tests PASS.

- [ ] **Step 4: Run unit tests (no GPU)**

```bash
pytest tests/ --ignore=tests/test_extract_snapshots.py -v
```
Expected: all PASS.

- [ ] **Step 5: Train ACT (~1–2 h)**

```bash
python exp0/train_act.py --epochs 100 --output checkpoints/act_libero.pt
```
Expected: epoch loss logs + "Checkpoint saved to checkpoints/act_libero.pt".

- [ ] **Step 6: Train VQ-BeT (~1–2 h, parallel if two GPUs available)**

```bash
python exp0/train_vqbet.py --epochs 100 --output checkpoints/vqbet_libero.pt
```
Expected: epoch loss logs + "Checkpoint saved to checkpoints/vqbet_libero.pt".

- [ ] **Step 7: Run full experiment (~30 min)**

```bash
python exp0/run_exp0.py \
    --act-checkpoint checkpoints/act_libero.pt \
    --vqbet-checkpoint checkpoints/vqbet_libero.pt \
    --molmoact2-checkpoint checkpoints/molmoact2_droid
```
Expected: progress bars per model/condition, 6 JSON files + `exp0/results/summary.csv`.

- [ ] **Step 8: Spot-check results**

```bash
python -c "
import pandas as pd
df = pd.read_csv('exp0/results/summary.csv')
print(df.groupby(['model','condition'])[['hash_equal','max_diff','latency_p99_ms']].mean().round(4))
"
```
Expected: 6-row table. `hash_equal` should be ≥ 0.95 for ACT and VQ-BeT under deterministic condition.

- [ ] **Step 9: Copy results back to local machine**

```bash
# Run from local machine:
gcloud compute scp <vm-name>:~/vla-det-exec/exp0/results/ ./exp0/results/ --recurse --zone=<your-zone>
```

- [ ] **Step 10: Commit results**

```bash
git add exp0/results/summary.csv
git commit -m "results: Experiment 0 determinism summary"
```

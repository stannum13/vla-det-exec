# Experiment 0: Recorded-Input Determinism Design

**Date:** 2026-08-09
**Scope:** Software-only determinism test for ACT, VQ-BeT, and MolmoAct2 on real LIBERO simulation observations. No physical robot required.

---

## Goal

Quantify how much inference-time nondeterminism exists in each model family, and how much is eliminated by fixing random seeds/latents. This is the cheapest publishable first figure in the "can VLA-deployed tasks be made deterministic?" study.

---

## Project structure

```
vla-det-exec/
├── exp0/
│   ├── extract_snapshots.py      # extracts 50 .npz snapshots from LIBERO demo HDF5s
│   ├── run_exp0.py               # main harness: model × condition × snapshot × rep
│   ├── metrics.py                # measurement utilities
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── act_adapter.py
│   │   ├── vqbet_adapter.py
│   │   └── molmoact2_adapter.py
│   ├── snapshots/                # extracted from LIBERO demos, committed for reproducibility
│   └── results/                  # JSON per (model, condition) + summary CSV
├── setup/
│   ├── setup.sh                  # clone third-party repos, install deps
│   └── requirements.txt
└── docs/
    └── superpowers/specs/
        └── 2026-08-09-exp0-determinism-design.md
```

Third-party repos are cloned into `third_party/` (gitignored).

---

## Dataset and simulation environment

Use **LIBERO** (`Lifelong-Robot-Learning/LIBERO`) as the source of observations and evaluation environment. LIBERO provides:
- MuJoCo/robosuite-based Franka Panda simulation
- 4 task suites (Spatial, Object, Goal, Long), 10 tasks each, 50 demos per task
- RGB observations (agentview 128×128 + wrist 128×128) + 9-dim robot proprioception
- Language-annotated task instructions per demo
- Pretrained baseline policies (BC-RNN, DiffusionPolicy) for comparison

**Selected suite for Experiment 0:** LIBERO-Spatial (tests spatial generalization, most relevant to the pick-and-place hypothesis).

### Snapshot extraction

Snapshots are extracted from LIBERO demo HDF5 files — not synthetically generated. Each snapshot is one timestep sampled from an existing demonstration trajectory.

Each snapshot `.npz` contains:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `agentview_rgb` | `(128, 128, 3)` | `uint8` | Third-person camera from LIBERO demo |
| `wrist_rgb` | `(128, 128, 3)` | `uint8` | Wrist camera from LIBERO demo |
| `state` | `(9,)` | `float32` | LIBERO proprioception (joint pos + gripper) |
| `instruction` | scalar | `str` | LIBERO task language instruction |
| `demo_id` | scalar | `int` | Source demo index for traceability |
| `timestep` | scalar | `int` | Timestep within demo |

50 snapshots are extracted once (stratified across tasks and timesteps) and saved. Every model sees identical inputs.

Each adapter maps this shared schema into its model's native input format.

---

## Model adapters

Each adapter exposes:

```python
class BaseAdapter:
    def __init__(self, mode: Literal["stochastic", "deterministic"]) -> None: ...
    def infer(self, snapshot: dict) -> np.ndarray: ...
    # returns action chunk: float32 (chunk_size, action_dim)
```

### ACT adapter

- Loads `ACTPolicy` from a checkpoint trained on LIBERO-Spatial demos (~1–2 h on A100 using the official ACT training script).
- **Stochastic**: samples CVAE latent `z` from the prior at each call.
- **Deterministic**: uses fixed `z = torch.zeros(1, latent_dim)` at every call.
- Output: action chunk `(chunk_size, 7)` (LIBERO 7-DOF action space).

### VQ-BeT adapter

- Loads `VQBeTPolicy` from a checkpoint trained on the same LIBERO-Spatial demos.
- **Stochastic**: samples discrete code token from the predicted categorical distribution.
- **Deterministic**: takes argmax of code logits; deterministic residual decoder.
- Output: action chunk `(chunk_size, 7)`.

### MolmoAct2 adapter

- Loads MolmoAct2-DROID pretrained checkpoint from AllenAI (Franka-native, no fine-tuning needed for the determinism test).
- VLM backbone runs in fp16 on A100; action expert runs the flow ODE.
- **Stochastic**: samples fresh `torch.randn(...)` flow noise at each call.
- **Deterministic**: `torch.manual_seed(0)` before each `torch.randn(...)` call.
- Output: action chunk `(chunk_size, 7)`.

All adapters call `model.eval()` and run under `torch.no_grad()`. CUDA deterministic mode (`torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True`) is enabled for the deterministic condition only, to measure the delta.

---

## Measurement harness (`run_exp0.py`)

```
for model in [act, vqbet, molmoact2]:
    for condition in [stochastic, deterministic]:
        adapter = load_adapter(model, condition)
        for snap_idx in range(50):
            snapshot = load_snapshot(snap_idx)
            outputs, latencies = [], []
            for rep in range(100):
                t0 = perf_counter()
                action = adapter.infer(snapshot)
                latencies.append((perf_counter() - t0) * 1000)
                outputs.append(action)
            record = compute_metrics(outputs, latencies, snap_idx)
            append_to_results(model, condition, record)
    save_summary_csv()
```

The 100 repetitions per snapshot are run in a tight loop without reloading weights or recreating the model, isolating inference-time randomness from initialization randomness.

---

## Metrics (`metrics.py`)

| Metric | Description |
|---|---|
| `hash_equal` | `True` if all 100 output arrays are bitwise identical |
| `max_diff` | `max |action[i] - action[j]|` over all 4950 pairs |
| `rms_diff` | RMS of all pairwise element-wise differences |
| `token_entropy` | VQ-BeT only: Shannon entropy of the 100 sampled code tokens |
| `latency_p50_ms` | Median inference latency in ms |
| `latency_p95_ms` | 95th-percentile latency |
| `latency_p99_ms` | 99th-percentile latency |
| `latency_max_ms` | Maximum latency |
| `deadline_miss_rate` | Fraction of inferences exceeding 100 ms (10 Hz policy rate) |

Results are written as:
- `results/<model>_<condition>.json` — per-snapshot records (list of 50 dicts)
- `results/summary.csv` — one row per `(model, condition, snap_idx)`

---

## GCP setup (`setup/setup.sh`)

Steps run on the A100 VM after SSH:

1. Install system deps (`git`, `python3-pip`, CUDA already present on A100 image).
2. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
3. Clone `Lifelong-Robot-Learning/LIBERO` → `third_party/libero`; install LIBERO + robosuite.
4. Clone `tonyzhaozh/act` → `third_party/act`
5. Clone `jayLEE0301/vq_bet_official` → `third_party/vq_bet_official`
6. Clone `allenai/molmoact2` → `third_party/molmoact2`
7. `pip install -r setup/requirements.txt` (aggregates each third-party repo's own `requirements.txt` plus `numpy`, `pandas`, `tqdm`, `h5py`)
8. Download LIBERO-Spatial dataset HDF5s via the LIBERO download script → `data/libero_spatial/`
9. Download MolmoAct2-DROID checkpoint from HuggingFace → `checkpoints/molmoact2_droid/`
10. `python exp0/train_act.py --suite libero_spatial` — trains ACT on LIBERO-Spatial demos (~1–2 h)
11. `python exp0/train_vqbet.py --suite libero_spatial` — trains VQ-BeT on same demos (~1–2 h)
12. `python exp0/extract_snapshots.py` — extracts `exp0/snapshots/snap_{00..49}.npz` from LIBERO HDF5s
13. `python exp0/run_exp0.py` — runs full experiment, writes to `exp0/results/`

Estimated wall time: ~4–5 h end-to-end on an A100 (dominated by training).

---

## Expected outputs

After `run_exp0.py` completes:

```
exp0/results/
    act_stochastic.json
    act_deterministic.json
    vqbet_stochastic.json
    vqbet_deterministic.json
    molmoact2_stochastic.json
    molmoact2_deterministic.json
    summary.csv
```

The summary CSV has columns: `model, condition, snap_idx, hash_equal, max_diff, rms_diff, token_entropy, latency_p50_ms, latency_p95_ms, latency_p99_ms, latency_max_ms, deadline_miss_rate`.

---

## What this does not cover

- Physical robot (all observations come from LIBERO simulation)
- Task success evaluation (only inference repeatability is measured, not rollout performance)
- Physical trajectory repeatability (Experiments 1–3, future work)
- Fine-tuning MolmoAct2 on LIBERO (DROID checkpoint used as-is)

---

## Success criteria

Experiment 0 is complete when:

1. `summary.csv` exists with all 6 `(model, condition)` combinations populated.
2. For the deterministic condition, `hash_equal` is `True` for ≥ 95% of snapshots for ACT and VQ-BeT.
3. Latency stats are recorded for all models and conditions.
4. A negative result (deterministic condition still shows variance) is also acceptable — it identifies where nondeterminism lives.

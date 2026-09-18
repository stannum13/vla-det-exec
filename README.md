# Deterministic VLA Execution

An experimental harness for studying a subtle robotics problem: **if a vision-language-action policy is run twice on the same observation, how much of its output changes—and which parts of that variation can be removed safely?**

The first experiment compares deterministic and stochastic inference paths for three action-policy families: ACT, VQ-BeT, and MolmoAct2. It measures exact output agreement, action drift, discrete-token entropy, latency, and deadline misses on fixed LIBERO-Spatial observations.

## Why this matters

Model weights alone do not define a robot's behavior. Sampling choices, flow noise, latent codes, action normalization, chunk length, replanning cadence, and controller conventions all affect what actually reaches the robot.

This project isolates the model-side part of that problem before any physical rollout. The broader research direction is to pair a learned policy with an explicit execution contract: reproducible preprocessing, versioned metadata, bounded actions, phase-aware chunk execution, runtime guards, and clear abort behavior.

## Experiment 0

For each stored observation, the harness runs 100 repeated inference calls in two conditions:

| Model family | Stochastic path | Deterministic path |
| --- | --- | --- |
| ACT | sampled latent | fixed zero latent |
| VQ-BeT | sampled action token | argmax action token |
| MolmoAct2 | fresh flow noise | fixed-seed flow noise |

It records:

- exact action-array hash agreement;
- maximum and RMS action difference;
- token entropy for VQ-BeT;
- p50, p95, p99, and maximum inference latency; and
- misses against a 100 ms policy deadline.

## What is implemented

- LIBERO-Spatial snapshot extraction from HDF5 demonstrations.
- A shared adapter interface for ACT, VQ-BeT, and MolmoAct2.
- Deterministic and stochastic inference modes for each model family.
- Lightweight stand-ins that make adapter behavior testable without downloading large checkpoints.
- ACT and VQ-BeT training entry points for the shared LIBERO observation/action format.
- A benchmark runner that writes per-condition JSON and a combined CSV summary.
- Unit and integration tests for metrics, adapters, token logging, snapshot format, and output generation.

## Project status

The harness and its lightweight tests are implemented. The full benchmark still requires the LIBERO-Spatial dataset, 50 extracted snapshots, third-party model repositories, and the relevant checkpoints. No physical-robot result is claimed here, and stand-in model tests are not presented as evidence about the real checkpoints.

## Repository map

```text
exp0/
├── adapters/             # ACT, VQ-BeT, and MolmoAct2 inference adapters
├── extract_snapshots.py  # fixed-input dataset preparation
├── metrics.py            # determinism and latency metrics
├── run_exp0.py           # repeated-inference benchmark
├── train_act.py          # ACT training entry point
└── train_vqbet.py        # VQ-BeT training entry point

setup/                    # environment and third-party setup
tests/                    # adapter, metric, snapshot, and harness tests
det-vla-exec-rev.md       # literature review and staged research proposal
```

## Setup

The full environment targets Linux with an NVIDIA GPU and CUDA 12.1. The setup script installs PyTorch, LIBERO, ACT, VQ-BeT, MolmoAct2, the LIBERO-Spatial dataset, and the MolmoAct2-DROID checkpoint.

```bash
bash setup/setup.sh
python exp0/extract_snapshots.py
```

Training entry points:

```bash
python exp0/train_act.py --epochs 100 --output checkpoints/act_libero.pt
python exp0/train_vqbet.py --epochs 100 --output checkpoints/vqbet_libero.pt
```

Run the comparison:

```bash
python exp0/run_exp0.py \
  --act-checkpoint checkpoints/act_libero.pt \
  --vqbet-checkpoint checkpoints/vqbet_libero.pt \
  --molmoact2-checkpoint checkpoints/molmoact2_droid
```

Results are written under `exp0/results/` as six per-condition JSON files and one combined `summary.csv`.

## Tests

After installing the Python requirements:

```bash
python -m pytest -q
```

The snapshot-format tests expect all 50 generated `.npz` files. To run the model-independent suite before downloading the dataset:

```bash
python -m pytest -q tests --ignore=tests/test_extract_snapshots.py
```

## Research direction

The staged proposal in [det-vla-exec-rev.md](det-vla-exec-rev.md) continues from recorded-input repeatability to phase-gated action chunks, runtime safety checks, spatial perturbations, and teacher-to-student transfer. The project deliberately starts with the cheapest falsifiable experiment before adding a simulator or physical robot.

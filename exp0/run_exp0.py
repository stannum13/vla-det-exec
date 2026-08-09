#!/usr/bin/env python3
"""Experiment 0: Recorded-Input Determinism.

Usage:
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

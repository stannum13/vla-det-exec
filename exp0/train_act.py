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
                    agentview = demo["obs/agentview_rgb"][:]
                    wrist = demo["obs/robot0_eye_in_hand_image"][:]
                    joint_pos = demo["obs/robot0_joint_pos"][:]
                    gripper = demo["obs/robot0_gripper_qpos"][:]
                    actions = demo["actions"][:]
                    T = agentview.shape[0]
                    for t in range(T - CHUNK_SIZE):
                        qpos = np.concatenate([joint_pos[t], gripper[t]]).astype(np.float32)
                        act_chunk = actions[t:t + CHUNK_SIZE].astype(np.float32)
                        self.samples.append((agentview[t], wrist[t], qpos, act_chunk))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        agentview, wrist, qpos, act_chunk = self.samples[idx]
        images = torch.stack([_NORM(agentview), _NORM(wrist)])
        is_pad = torch.zeros(CHUNK_SIZE, dtype=torch.bool)
        return torch.tensor(qpos), images, torch.tensor(act_chunk), is_pad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--output", default="checkpoints/act_libero.pt")
    args = parser.parse_args()

    act_path = Path("third_party/act")
    if not act_path.exists():
        print(f"ERROR: {act_path} not found. Run setup/setup.sh first.")
        raise SystemExit(1)
    sys.path.insert(0, str(act_path.resolve()))
    from policy import ACTPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LiberoDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    cfg = {
        "num_queries": CHUNK_SIZE, "hidden_dim": 256, "dim_feedforward": 3200,
        "nheads": 8, "enc_layers": 4, "dec_layers": 7, "dropout": 0.1,
        "pre_norm": False, "camera_names": ["agentview", "wrist"],
        "backbone": "resnet18", "kl_weight": args.kl_weight,
        "lr": args.lr, "lr_backbone": args.lr * 0.1, "state_dim": 9, "action_dim": 7,
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

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

sys.path.insert(0, ".")
from exp0.train_act import LiberoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="checkpoints/vqbet_libero.pt")
    args = parser.parse_args()

    vqbet_path = Path("third_party/vq_bet_official")
    if not vqbet_path.exists():
        print(f"ERROR: {vqbet_path} not found. Run setup/setup.sh first.")
        raise SystemExit(1)
    sys.path.insert(0, str(vqbet_path.resolve()))
    from vq_bet.vq_bet_model import VQBeTPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

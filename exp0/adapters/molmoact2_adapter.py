"""MolmoAct2 adapter for Experiment 0.

stochastic:    fresh torch.randn flow noise per call
deterministic: fixed-seed torch.Generator noise per call
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from exp0.adapters.base import BaseAdapter

CHUNK_SIZE = 10
ACTION_DIM = 7
FLOW_STEPS = 10
FIXED_SEED = 42

_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


class MolmoAct2Adapter(BaseAdapter):
    def __init__(self, mode: str, checkpoint_path: str | None = None,
                 stub: bool = False, device: str | None = None) -> None:
        super().__init__(mode)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if stub or checkpoint_path is None:
            self.model = _FlowActionStub(CHUNK_SIZE, ACTION_DIM).to(self.device)
        else:
            molmo_path = Path("third_party/molmoact2")
            if molmo_path.exists():
                sys.path.insert(0, str(molmo_path.resolve()))
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
            out = self.model(images=images, state=state,
                             instruction=snapshot["instruction"],
                             noise=noise, num_steps=FLOW_STEPS)
        return out.squeeze(0).cpu().numpy().astype(np.float32)

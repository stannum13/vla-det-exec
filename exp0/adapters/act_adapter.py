"""ACT adapter for Experiment 0.

deterministic: z = zeros (ACT's default inference behavior)
stochastic:    z ~ N(0,I) injected into decoder
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from exp0.adapters.base import BaseAdapter

_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_FULL_CFG = {
    "num_queries": 10, "hidden_dim": 256, "dim_feedforward": 3200,
    "nheads": 8, "enc_layers": 4, "dec_layers": 7, "dropout": 0.1,
    "pre_norm": False, "camera_names": ["agentview", "wrist"],
    "backbone": "resnet18", "kl_weight": 10,
    "lr": 1e-4, "lr_backbone": 1e-5, "state_dim": 9, "action_dim": 7,
}
_TINY_CFG = {**_FULL_CFG, "hidden_dim": 32, "dim_feedforward": 64, "nheads": 2,
             "enc_layers": 1, "dec_layers": 1}


class _TinyACTStub(nn.Module):
    """Minimal stub: deterministic z=0 branch uses fixed linear, stochastic adds noise."""
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self._latent_dim = cfg["hidden_dim"]
        self._chunk = cfg["num_queries"]
        self._act_dim = cfg["action_dim"]
        self.head = nn.Linear(cfg["hidden_dim"], cfg["num_queries"] * cfg["action_dim"])

    def forward(self, qpos: torch.Tensor, images: torch.Tensor,
                z: torch.Tensor | None = None) -> torch.Tensor:
        if z is None:
            z = torch.zeros(qpos.shape[0], self._latent_dim, device=qpos.device)
        out = self.head(z)
        return out.reshape(qpos.shape[0], self._chunk, self._act_dim)


class ACTAdapter(BaseAdapter):
    def __init__(self, mode: str, checkpoint_path: str | None = None,
                 tiny: bool = False, device: str | None = None) -> None:
        super().__init__(mode)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = _TINY_CFG if tiny else _FULL_CFG
        self._latent_dim = cfg["hidden_dim"]

        # Try loading real ACTPolicy; fall back to stub
        self._use_stub = True
        if not tiny:
            act_path = Path("third_party/act")
            if act_path.exists():
                sys.path.insert(0, str(act_path.resolve()))
                try:
                    from policy import ACTPolicy  # noqa: F401
                    self.policy = ACTPolicy(cfg)
                    self._use_stub = False
                except ImportError:
                    pass

        if self._use_stub:
            self.policy = _TinyACTStub(cfg)

        if checkpoint_path is not None and not self._use_stub:
            self.policy.deserialize(torch.load(checkpoint_path, map_location=self.device))
        self.policy.to(self.device).eval()

    def infer(self, snapshot: dict) -> np.ndarray:
        images = self._prep_images(snapshot)
        qpos = torch.tensor(snapshot["state"], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self._use_stub:
                if self.mode == "deterministic":
                    z = torch.zeros(1, self._latent_dim, device=self.device)
                else:
                    z = torch.randn(1, self._latent_dim, device=self.device)
                a_hat = self.policy(qpos, images, z=z)
            else:
                if self.mode == "deterministic":
                    a_hat = self.policy(qpos, images)
                else:
                    seed = int(torch.randint(0, 2**31, (1,)).item())
                    torch.manual_seed(seed)
                    a_hat = self.policy(qpos, images)

        return a_hat.squeeze(0).cpu().numpy().astype(np.float32)

    def _prep_images(self, snapshot: dict) -> torch.Tensor:
        imgs = [_NORM(snapshot[k]) for k in ("agentview_rgb", "wrist_rgb")]
        return torch.stack(imgs).unsqueeze(0).to(self.device)

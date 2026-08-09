"""VQ-BeT adapter for Experiment 0.

stochastic:    sample code token from categorical logits
deterministic: argmax of code logits

token_log accumulates chosen code indices per infer() call.
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
    "obs_dim": 9, "act_dim": 7, "chunk_size": 10,
    "num_latents": 512, "num_residual_layers": 4,
    "hidden_dim": 256, "n_heads": 8, "n_layers": 6,
}
_TINY_CFG = {**_FULL_CFG, "num_latents": 16, "hidden_dim": 32, "n_heads": 2, "n_layers": 2}


class _TinyVQBeTStub(nn.Module):
    """Stub implementing get_code_logits_and_residual + decode_from_token."""
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
        logits = self.head(x)
        residual = torch.zeros(1, self._chunk, self._act, device=x.device)
        return logits, residual

    def decode_from_token(self, token: int, residual: torch.Tensor) -> torch.Tensor:
        emb = self.codebook(torch.tensor([token], device=residual.device))
        return emb.reshape(1, self._chunk, self._act) + residual


class VQBeTAdapter(BaseAdapter):
    def __init__(self, mode: str, checkpoint_path: str | None = None,
                 tiny: bool = False, device: str | None = None) -> None:
        super().__init__(mode)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.token_log: list[int] = []
        cfg = _TINY_CFG if tiny else _FULL_CFG

        # Try real VQBeTPolicy; fall back to stub
        self._policy = None
        if not tiny:
            vqbet_path = Path("third_party/vq_bet_official")
            if vqbet_path.exists():
                sys.path.insert(0, str(vqbet_path.resolve()))
                try:
                    from vq_bet.vq_bet_model import VQBeTPolicy
                    self._policy = VQBeTPolicy(cfg)
                except ImportError:
                    pass

        if self._policy is None:
            self._policy = _TinyVQBeTStub(cfg)

        if checkpoint_path is not None:
            self._policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self._policy.to(self.device).eval()

    def infer(self, snapshot: dict) -> np.ndarray:
        obs = {
            "image": torch.stack([_NORM(snapshot[k]) for k in ("agentview_rgb", "wrist_rgb")])
                         .unsqueeze(0).to(self.device),
            "state": torch.tensor(snapshot["state"], dtype=torch.float32).unsqueeze(0).to(self.device),
        }
        with torch.no_grad():
            code_logits, residual = self._policy.get_code_logits_and_residual(obs)
            if self.mode == "stochastic":
                token = int(torch.distributions.Categorical(logits=code_logits[0]).sample().item())
            else:
                token = int(code_logits[0].argmax().item())
            self.token_log.append(token)
            action_chunk = self._policy.decode_from_token(token, residual)
        return action_chunk.squeeze(0).cpu().numpy().astype(np.float32)

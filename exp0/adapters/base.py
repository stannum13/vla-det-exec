"""Base adapter interface for Experiment 0."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseAdapter(ABC):
    def __init__(self, mode: str) -> None:
        assert mode in ("stochastic", "deterministic"), f"Unknown mode: {mode}"
        self.mode = mode

    @abstractmethod
    def infer(self, snapshot: dict) -> np.ndarray:
        """Run one inference pass. Returns float32 (chunk_size, action_dim=7)."""
        ...

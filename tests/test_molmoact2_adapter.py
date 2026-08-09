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

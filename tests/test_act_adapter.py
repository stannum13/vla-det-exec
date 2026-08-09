import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from exp0.adapters.act_adapter import ACTAdapter

_SNAP = {
    "agentview_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    "wrist_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    "state": np.random.rand(9).astype(np.float32),
    "instruction": "pick up the object",
    "demo_id": 0,
    "timestep": 10,
}


def test_deterministic_returns_same_output():
    adapter = ACTAdapter(mode="deterministic", checkpoint_path=None, tiny=True)
    out1 = adapter.infer(_SNAP)
    out2 = adapter.infer(_SNAP)
    np.testing.assert_array_equal(out1, out2)


def test_output_shape():
    adapter = ACTAdapter(mode="deterministic", checkpoint_path=None, tiny=True)
    out = adapter.infer(_SNAP)
    assert out.ndim == 2 and out.shape[1] == 7


def test_stochastic_varies():
    adapter = ACTAdapter(mode="stochastic", checkpoint_path=None, tiny=True)
    outputs = [adapter.infer(_SNAP) for _ in range(5)]
    hashes = [o.tobytes() for o in outputs]
    assert len(set(hashes)) > 1, "Stochastic adapter returned identical outputs"


def test_output_dtype():
    adapter = ACTAdapter(mode="deterministic", checkpoint_path=None, tiny=True)
    assert adapter.infer(_SNAP).dtype == np.float32

"""
Test suite for NTMMultiTask.

Wraps a NeuralTuringMachine with a task-id input. call() takes a list
[sequence (B, T, F), task (B, num_tasks)] and returns (B, T, output_dim).
Covers construction, a forward pass, ValueError on a malformed input_shape, and
the M2 full .keras round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.ntm.model_multitask import NTMMultiTask
from dl_techniques.layers.memory.ntm_interface import NTMConfig

OUT_DIM = 4
NUM_TASKS = 3
FEAT = 6
SEQ = 5


def _config():
    return NTMConfig(memory_size=16, memory_dim=8, controller_dim=16,
                     num_read_heads=1, num_write_heads=1)


def _model():
    return NTMMultiTask(ntm_config=_config(), output_dim=OUT_DIM, num_tasks=NUM_TASKS)


def _inputs(batch=2):
    rng = np.random.default_rng(0)
    return [
        rng.random((batch, SEQ, FEAT)).astype("float32"),
        rng.random((batch, NUM_TASKS)).astype("float32"),
    ]


class TestNTMMultiTask:

    def test_forward_shape(self):
        out = _model()(_inputs(), training=False)
        y = out[0] if isinstance(out, (list, tuple)) else out
        assert tuple(y.shape) == (2, SEQ, OUT_DIM)

    def test_malformed_input_shape_raises(self):
        model = _model()
        with pytest.raises(ValueError):
            model.build((None, SEQ, FEAT))  # single shape, not a list of 2

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _inputs()
        out_b = model(x, training=False)
        before = keras.ops.convert_to_numpy(
            out_b[0] if isinstance(out_b, (list, tuple)) else out_b)

        path = os.path.join(str(tmp_path), "ntm_multitask.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        out_a = loaded(x, training=False)
        after = keras.ops.convert_to_numpy(
            out_a[0] if isinstance(out_a, (list, tuple)) else out_a)

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="NTMMultiTask differs after round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

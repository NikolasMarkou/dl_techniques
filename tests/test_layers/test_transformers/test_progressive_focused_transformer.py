"""Tests for the PFTBlock (Progressive Focused Transformer) block."""

import os
import tempfile

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.layers.transformers.progressive_focused_transformer import (
    PFTBlock,
)

B, H, W, DIM = 2, 8, 8, 16


def _x() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((B, H, W, DIM)).astype("float32")


class TestPFTBlock:

    def test_construction(self) -> None:
        block = PFTBlock(dim=DIM, num_heads=4, window_size=8)
        assert block._dim == DIM and block._num_heads == 4

    @pytest.mark.parametrize("kwargs", [
        {"dim": 16, "num_heads": 4, "window_size": 8, "shift_size": 8},  # shift>=window
        {"dim": 16, "num_heads": 4, "shift_size": -1},
        {"dim": 16, "num_heads": 5},                                     # not divisible
        {"dim": 16, "num_heads": 4, "mlp_ratio": 0.0},
        {"dim": 16, "num_heads": 4, "attention_dropout": 1.5},
    ])
    def test_invalid_construction(self, kwargs) -> None:
        with pytest.raises(ValueError):
            PFTBlock(**kwargs)

    def test_forward_pass(self) -> None:
        block = PFTBlock(dim=DIM, num_heads=4, window_size=8)
        out, attn_map = block(_x())
        assert tuple(out.shape) == (B, H, W, DIM)

    def test_compute_output_shape(self) -> None:
        block = PFTBlock(dim=DIM, num_heads=4, window_size=8)
        x = _x()
        out_shape, _ = block.compute_output_shape(x.shape)
        out, _ = block(x)
        assert tuple(out_shape) == tuple(out.shape)

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(H, W, DIM))
        out, _ = PFTBlock(dim=DIM, num_heads=4, window_size=8)(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pft.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-5
        )

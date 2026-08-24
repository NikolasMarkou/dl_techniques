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
        {"dim": 16, "num_heads": 4, "attention_dropout_rate": 1.5},
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


class TestStochasticDepthKwarg:
    """F-02: `PFTBlock` constructed `StochasticDepth(drop_rate=...)`.

    `StochasticDepth.__init__` only accepts `drop_path_rate`, so `drop_rate`
    fell through to `keras.layers.Layer.__init__`, which rejects unknown
    kwargs. Every `drop_path_rate > 0.0` therefore died at CONSTRUCTION -- the
    whole point of the parameter. `models/pft_sr/model.py` schedules
    `linspace(0, drop_path_rate, total_blocks)` across its blocks, so the model
    was dead on arrival for any non-zero rate too.

    RED at HEAD (`9ee28342`):
    ``ValueError: Unrecognized keyword arguments passed to StochasticDepth:
    {'drop_rate': 0.1}`` -- a `ValueError` from Keras' kwarg check, not the
    `TypeError` the plan predicted. Same class of failure (the ctor rejects the
    name), different exception type.

    A one-token rename that merely CONSTRUCTS would be a hollow fix, so the
    drop path is also measured to be live: stochastic under `training=True`
    and exactly identity under `training=False`.
    """

    RATE = 0.5  # large enough that a batch of 8 almost surely contains a drop

    def test_construction_at_nonzero_rate(self) -> None:
        block = PFTBlock(dim=DIM, num_heads=4, window_size=8, drop_path_rate=0.1)
        assert block._drop_path is not None
        assert block._drop_path.drop_path_rate == pytest.approx(0.1)

    def test_forward_pass_at_nonzero_rate(self) -> None:
        block = PFTBlock(dim=DIM, num_heads=4, window_size=8, drop_path_rate=0.1)
        out, _ = block(_x(), training=True)
        assert tuple(out.shape) == (B, H, W, DIM)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))

    def test_zero_rate_still_builds_no_drop_path(self) -> None:
        """CONTROL: the `> 0.0` guard must still short-circuit."""
        block = PFTBlock(dim=DIM, num_heads=4, window_size=8, drop_path_rate=0.0)
        assert block._drop_path is None

    def test_drop_path_is_identity_at_inference(self) -> None:
        """`training=False` must be exactly deterministic and drop nothing."""
        block = PFTBlock(dim=DIM, num_heads=4, window_size=8,
                         drop_path_rate=self.RATE)
        x = _x()
        a = ops.convert_to_numpy(block(x, training=False)[0])
        b = ops.convert_to_numpy(block(x, training=False)[0])
        np.testing.assert_allclose(a, b, rtol=0, atol=0)

    def test_drop_path_actually_drops_under_training(self) -> None:
        """The rename must WIRE something, not merely construct.

        `StochasticDepth` draws a per-sample Bernoulli mask, so with
        `drop_path_rate=0.5` and 8 samples a training forward pass must both
        (a) vary run to run and (b) produce at least one sample whose
        attention/FFN branch was dropped entirely. Sample (b) is what
        distinguishes a live drop path from an inert layer that merely rescales.
        """
        block = PFTBlock(dim=DIM, num_heads=4, window_size=8,
                         drop_path_rate=self.RATE)
        big_x = np.random.default_rng(3).standard_normal(
            (8, H, W, DIM)).astype("float32")

        runs = [ops.convert_to_numpy(block(big_x, training=True)[0])
                for _ in range(6)]
        assert any(
            not np.array_equal(runs[0], r) for r in runs[1:]
        ), "training-mode output is deterministic; the drop path is inert"

        # A dropped residual branch makes the block an exact identity for that
        # sample: attn_output and ffn_output are both zeroed, so output == input.
        eval_out = ops.convert_to_numpy(block(big_x, training=False)[0])
        assert not np.allclose(eval_out, big_x), (
            "the block is an identity even at inference, so 'output == input' "
            "cannot distinguish a dropped branch"
        )
        dropped = [
            np.array_equal(r[i], big_x[i])
            for r in runs for i in range(big_x.shape[0])
        ]
        assert any(dropped), (
            "over 48 sample-forwards at drop_path_rate=0.5, not one residual "
            "branch was ever fully dropped -- StochasticDepth is not wired in"
        )

    def test_serialization_round_trip_at_nonzero_rate(self) -> None:
        """I3: round-trip on a config that exercises the new path, by VALUE."""
        inp = keras.Input(shape=(H, W, DIM))
        out, _ = PFTBlock(dim=DIM, num_heads=4, window_size=8,
                          drop_path_rate=0.1)(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = ops.convert_to_numpy(model(x, training=False))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pft_dp.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = ops.convert_to_numpy(loaded(x, training=False))
        assert float(np.max(np.abs(y0))) > 1e-3, "round-trip compared all-zero values"
        np.testing.assert_allclose(y0, y1, rtol=0, atol=0)
        assert loaded.layers[-1]._drop_path.drop_path_rate == pytest.approx(0.1)


class TestPftSrConsumerSmoke:
    """`models/pft_sr/model.py` is the live `PFTBlock` consumer (blast radius
    derived by grep, not assumed). It schedules a `linspace` of rates across
    every block, so it was dead on arrival for any `drop_path_rate > 0`."""

    def test_pft_sr_builds_and_runs_at_nonzero_drop_path_rate(self) -> None:
        from dl_techniques.models.pft_sr.model import PFTSR
        model = PFTSR(
            scale=2, in_channels=3, embed_dim=16, num_blocks=[1, 1],
            num_heads=4, window_size=8, mlp_ratio=2.0, drop_path_rate=0.1,
        )
        x = np.random.default_rng(5).standard_normal((1, 16, 16, 3)).astype("float32")
        out = ops.convert_to_numpy(model(x, training=True))
        assert out.shape == (1, 32, 32, 3)
        assert np.all(np.isfinite(out))

    def test_pft_sr_schedules_distinct_rates_across_blocks(self) -> None:
        """The linspace schedule must actually reach the blocks."""
        from dl_techniques.models.pft_sr.model import PFTSR
        model = PFTSR(
            scale=2, in_channels=3, embed_dim=16, num_blocks=[2, 2],
            num_heads=4, window_size=8, mlp_ratio=2.0, drop_path_rate=0.4,
        )
        rates = sorted(set(model.dpr))
        assert len(rates) > 1, f"every block got the same rate: {model.dpr}"
        assert max(rates) == pytest.approx(0.4)

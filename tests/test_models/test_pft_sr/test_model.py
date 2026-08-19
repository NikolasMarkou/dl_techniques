"""
Test suite for PFTSR (permuted self-attention super-resolution).

create_pft_sr(scale, variant) builds the model; window_size=8 so input H/W are
kept divisible by 8. NHWC float32 input (B, H, W, 3); at scale=2 the output is
the upsampled image (B, 2H, 2W, 3). Covers a forward pass and the M2 full
.keras save -> load -> identical-output round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.pft_sr.model import create_pft_sr, PFTSR

SIZE = 32
SCALE = 2


def _model():
    return create_pft_sr(scale=SCALE, variant="light")


def _images(batch=2):
    return np.random.default_rng(0).random((batch, SIZE, SIZE, 3)).astype("float32")


class TestPFTSR:

    def test_factory_construction(self):
        assert isinstance(_model(), PFTSR)

    def test_forward_upsamples(self):
        out = _model()(_images(), training=False)
        assert tuple(out.shape) == (2, SIZE * SCALE, SIZE * SCALE, 3)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            create_pft_sr(scale=2, variant="nonexistent")

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _images()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "pft_sr.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="PFTSR differs after .keras round-trip")

    def test_keras_round_trip_at_nonzero_drop_path_rate(self, tmp_path):
        """SC-8: round-trip the STOCHASTIC-DEPTH path, by value.

        ``create_pft_sr`` hard-codes ``drop_path_rate=0.0``, so
        ``test_keras_round_trip`` above exercises the DEFAULT configuration and
        cannot observe the ``drop_path_rate > 0`` branch at all -- the branch
        that was dead on arrival until iter-1/step-8 (a ``StochasticDepth``
        kwarg name plus an ``EagerTensor.item()`` call in the schedule). This is
        the same class of gap step 6 found for ``use_free_transformer``: a
        round-trip test that exists but never reaches the new path.

        ``training=False`` makes ``StochasticDepth`` the identity, so the
        comparison is deterministic; the schedule assertion below is what proves
        the reloaded model really carries the non-zero rates.
        """
        model = PFTSR(
            scale=SCALE, in_channels=3, embed_dim=16, num_blocks=[1, 1],
            num_heads=4, window_size=8, mlp_ratio=2.0, drop_path_rate=0.1,
        )
        x = np.random.default_rng(11).random((1, 16, 16, 3)).astype("float32")
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "pft_sr_dp.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        assert float(np.max(np.abs(before))) > 1e-5, \
            "round-trip compared all-zero values"
        np.testing.assert_allclose(
            before, after, atol=1e-4,
            err_msg="PFTSR differs after .keras round-trip at drop_path_rate>0")
        assert loaded.drop_path_rate == pytest.approx(0.1)
        assert max(loaded.dpr) == pytest.approx(0.1), \
            f"the reloaded model lost its stochastic-depth schedule: {loaded.dpr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestNearestUpsamplerScaleValidation:
    """`_build_nearest_upsampler` stacks `int(log2(scale))` doubling stages.

    At `scale=3` that is ONE stage — a 2x output for a 3x request. The module docstring
    named the defect and the model shipped it anyway; `_build_pixelshuffle_upsampler`
    has always raised for unsupported scales, so only this branch was unguarded.
    `create_pft_sr` hardcodes `upsampler='pixelshuffle'`, which is why the hole was
    reachable only through direct `PFTSR(...)` construction.
    """

    @staticmethod
    def _build(scale):
        """The upsampler is constructed in `build()`, not `__init__` — same as the
        pixelshuffle branch, whose `raise` has always fired at the same moment. So the
        guard must be reached through a build, not a bare constructor call."""
        model = PFTSR(
            scale=scale,
            upsampler="nearest+conv",
            embed_dim=16,
            num_blocks=[1],
            num_heads=2,
            window_size=8,
        )
        model.build((None, 16, 16, 3))
        return model

    @pytest.mark.parametrize("scale", [3, 5, 6, 7, 12])
    def test_a_non_power_of_two_scale_raises(self, scale):
        with pytest.raises(ValueError, match=r"nearest\+conv"):
            self._build(scale)

    @pytest.mark.parametrize("scale", [2, 4, 8])
    def test_a_power_of_two_scale_still_builds(self, scale):
        """Anti-vacuity: the guard must not simply reject everything."""
        assert self._build(scale).scale == scale

    def test_the_realized_scale_is_the_requested_one(self):
        """The property the raise protects: what built must actually upsample by
        `scale`. A guard that admits a scale it cannot realize is no guard."""
        model = PFTSR(
            scale=4,
            upsampler="nearest+conv",
            embed_dim=16,
            num_blocks=[1],
            num_heads=2,
            window_size=8,
        )
        x = np.random.default_rng(0).random((1, 16, 16, 3)).astype("float32")
        out = model(x, training=False)
        assert tuple(out.shape[1:3]) == (64, 64)

    def test_the_pixelshuffle_upsampler_still_supports_three(self):
        """The guard is specific to `nearest+conv`; scale=3 remains legal elsewhere."""
        model = PFTSR(
            scale=3,
            upsampler="pixelshuffle",
            embed_dim=16,
            num_blocks=[1],
            num_heads=2,
            window_size=8,
        )
        assert model.scale == 3


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 10)
# ---------------------------------------------------------------------

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


class TestPFTSRGradientFlow:
    """Every trainable weight must be on the backward graph.

    232 trainable weights across permuted-attention blocks and a pixel-shuffle
    upsampler. This is the size at which an AGGREGATE gradient-norm assertion
    stops meaning anything -- one dead attention projection among 232 live
    tensors moves a global norm by nothing measurable. The oracle asserts per
    weight, keyed by ``Variable.path``.
    """

    def test_gradients_reach_every_trainable_weight(self):
        model = _model()
        x = _images()
        model(x, training=False)  # a subclassed model is unbuilt until first call

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == len(model.trainable_weights)
        assert len(report) > 0
        assert max(v for v in report.values() if v is not None) > 0.0

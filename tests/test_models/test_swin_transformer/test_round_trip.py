"""M2 .keras round-trip + validation tests for SwinTransformer.

Covers: construction via create_swin_transformer / from_variant, a ValueError
input-validation path (H4), forward pass, and a full save -> load ->
identical-output round-trip (atol 1e-5; GPU fp32 reduction noise).
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.swin_transformer.model import (
    SwinTransformer,
    create_swin_transformer,
)


# [CORRECTED 2026-08-01, G-03] This used to say: "The 'tiny' variant uses
# window_size=7, so the feature grid at the deepest stage must be divisible by 7
# -> the smallest legal square input is 224x224." The second half is FALSE and is
# deleted, not softened. `window_size` constrains nothing: SwinTransformerBlock
# pads a short grid up to the window and crops back. Re-measured by execution --
# `create_swin_transformer("tiny", 10, input_shape=(s,s,3))` (window_size=7,
# patch_size=4) builds AND forwards finite logits at s = 224, 96, 64, 56 and 32.
# 224 is kept below because it is the variant's design resolution, not a minimum.
def _images(b=2, s=224, c=3):
    return np.random.rand(b, s, s, c).astype("float32")


class TestSwinValidation:

    def test_invalid_num_classes(self):
        with pytest.raises(ValueError, match="num_classes must be positive"):
            SwinTransformer(num_classes=0, input_shape=(224, 224, 3))

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size must be positive"):
            SwinTransformer(num_classes=10, window_size=0,
                            input_shape=(224, 224, 3))


class TestSwinRoundTrip:

    def test_forward_shape(self):
        model = create_swin_transformer("tiny", 10, input_shape=(224, 224, 3))
        out = model(_images(), training=False)
        assert out.shape == (2, 10)

    def test_keras_round_trip(self):
        model = create_swin_transformer("tiny", 10, input_shape=(224, 224, 3))
        x = _images()
        y0 = model(x, training=False)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "swin.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y1 = reloaded(x, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 10)
# ---------------------------------------------------------------------

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


class TestSwinGradientFlow:
    """Every trainable weight must be on the backward graph.

    Run at 32x32 rather than this file's 224 default: the geometry note above
    establishes that 32 is legal, and the smaller grid keeps the tape step cheap
    (176 trainable weights either way). The interesting tensors here are the
    per-block ``relative_position_bias_table`` entries and the shifted-window
    attention projections -- a shift/roll implementation that dropped one of them
    would keep every shape and every round-trip test green.
    """

    # DECISION plan-2026-08-22T035419-a11304c8/D-030
    # ``drop_path_rate=0.0`` and the seed are BOTH load-bearing; do not restore
    # the variant default here, and do not "fix" the flake by picking a lucky
    # seed at the default rate.
    #
    # This test was flaky at 1 failed / 11 passed over 12 solo runs, with
    # ``170/176 trainable weights receive a live gradient`` and six dead weights
    # confined to ONE block's ``norm2`` + ``mlp`` (the block index moved between
    # observations: stage_3_block_0 in one record, stage_3_block_1 in another).
    # Root cause, measured: ``StochasticDepth.call`` draws an UNSEEDED
    # ``keras.random.uniform`` of shape ``(batch, 1, ..., 1)``
    # (``layers/stochastic_depth.py:172``), one independent Bernoulli per sample.
    # A SwinTransformerBlock has two such draws, one per residual branch, so when
    # both of a batch-of-2's samples are dropped on a branch that branch's
    # weights receive an identically-zero -- not disconnected -- gradient.
    # Probe (8 fresh models per arm, tape step at ``training=True``):
    #   drop_path_rate=0.0 -> 0/8 runs with any dead weight
    #   drop_path_rate=0.1 -> 0/8 (the shipped "tiny" default; ~1/12 in the wild)
    #   drop_path_rate=0.9 -> 8/8, 26-60 dead weights, a DIFFERENT random subset
    #                         of blocks each run, always whole {norm1,attn} or
    #                         {norm2,mlp} branch pairs.
    # The dead set follows the draw exactly, so this is regularization firing,
    # not a broken backward graph.
    #
    # Seeding ALONE is not the fix: over ``set_random_seed(0..15)`` at the
    # default rate, 4 of 16 seeds still produce dead weights (1, 9, 11, 12), so
    # "seed it" reduces to pinning one of the 12 lucky seeds -- a guard that
    # passes by luck and silently re-flakes the moment any upstream RNG
    # consumption shifts the seed-to-draw map. The claim this test makes is
    # backward-graph CONNECTIVITY, which is a structural property; at
    # ``drop_path_rate=0.0`` StochasticDepth early-returns its input
    # (``stochastic_depth.py:157``) and no draw can decide the verdict. The seed
    # additionally pins the weight init so the magnitudes are reproducible.
    def test_gradients_reach_every_trainable_weight(self):
        keras.utils.set_random_seed(0)
        model = create_swin_transformer(
            "tiny", 10, input_shape=(32, 32, 3), drop_path_rate=0.0
        )
        x = _images(s=32)
        model(x, training=False)  # a subclassed model is unbuilt until first call

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == len(model.trainable_weights)
        assert len(report) > 0
        assert max(v for v in report.values() if v is not None) > 0.0

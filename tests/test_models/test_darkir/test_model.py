"""
Test suite for the DarkIR low-light image restoration model.

DarkIR is a pure functional U-Net (create_darkir_model) using FreMLP (FFT path).
Covers a forward pass, the use_side_loss variant, and the M2 full .keras
save -> load -> identical-output round-trip. NHWC float32 input -> restored
image (B, H, W, 3).
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.darkir.model import create_darkir_model

SPATIAL = 32


def _images(batch=2):
    return np.random.default_rng(0).random(
        (batch, SPATIAL, SPATIAL, 3)).astype("float32")


def _primary(out):
    return out[0] if isinstance(out, (list, tuple)) else out


class TestForward:

    def test_forward_shape(self):
        model = create_darkir_model(img_channels=3, width=16)
        out = model(_images(), training=False)
        y = _primary(out)
        assert tuple(y.shape) == (2, SPATIAL, SPATIAL, 3)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(y)))

    def test_side_loss_variant_builds(self):
        enc_blk_nums = [1, 2, 3]
        model = create_darkir_model(
            img_channels=3, width=16, use_side_loss=True,
            enc_blk_nums=enc_blk_nums, dec_blk_nums=[3, 1, 1],
        )
        out = model(_images(), training=False)
        # Two outputs, and the second one is at BOTTLENECK resolution. The
        # `_primary(out)`-only assertion this replaced passed identically for a
        # single-output model, so it could not see C-23 at all.
        assert isinstance(out, (list, tuple)) and len(out) == 2, type(out)
        assert _primary(out).shape[1:] == (SPATIAL, SPATIAL, 3)

        factor = 2 ** len(enc_blk_nums)
        assert tuple(out[1].shape[1:]) == (
            SPATIAL // factor, SPATIAL // factor, 3,
        ), out[1].shape


class TestKerasRoundTrip:

    def test_save_load_identical(self, tmp_path):
        model = create_darkir_model(img_channels=3, width=16)
        x = _images()
        before = keras.ops.convert_to_numpy(_primary(model(x, training=False)))

        path = os.path.join(str(tmp_path), "darkir.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(_primary(loaded(x, training=False)))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="Outputs differ after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 11)
# ---------------------------------------------------------------------
#
# MEASURED 2026-08-19 on ``create_darkir_model(img_channels=3, width=16)`` at
# the suite's own 32x32 input, one tape step, ``default_loss``:
#
#   at init                : 373 weights, 330 with an identically-ZERO gradient
#   after 1 real fit step  : 373 weights,   0 dead
#   after 2 and 3 steps    : 373 weights,   0 dead
#
# The step-10 report carried the hypothesis "darkir ~300 dead weights -- far too
# many to be by-design, treat as a probable finding". That hypothesis is
# **REFUTED**. The 330 are the documented zero-init per-branch scale, stated in
# the model's own module docstring (``models/darkir/model.py``, "Both blocks
# scale each of their two residual branches by a learnable per-channel scale ...
# initialized to **zeros**. Every block therefore begins training as an exact
# [identity]") and enforced at ``model.py`` in ``DarkIREncoderBlock.build`` /
# ``DarkIRDecoderBlock.build`` by ``initializer="zeros"`` on ``beta`` and
# ``gamma``. With beta = gamma = 0 every residual branch contributes exactly
# nothing to the output, so d(loss)/d(branch weight) is exactly 0 -- while
# d(loss)/d(beta) is NOT, which is why the gates themselves are among the 43
# live weights and why the whole thing unblocks after a single optimizer step.
#
# The zero is therefore an INIT TRANSIENT, not a dead component: the state a
# trained darkir occupies for all but its first step has zero dead weights.
# So the adoption asserts the oracle AFTER one real optimizer step, with NO
# waivers at all, and a second test pins the init state so the mechanism cannot
# change silently. See GF-06 in
# plans/plan-2026-08-19T070627-a616f581/findings/gradient-flow-adoption-findings.md

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
)

GF_N_WEIGHTS = 373
GF_N_DEAD_AT_INIT = 330


def _gf_model():
    return create_darkir_model(img_channels=3, width=16)


def _gf_batch():
    return _images(batch=2)


def _n_dead(report) -> int:
    return sum(1 for v in report.values() if v is None or v == 0.0)


class TestDarkIRGradientFlow:
    """Every trainable weight is on the backward graph once training starts."""

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        # DECISION plan-2026-08-19-a616f581/D-012
        # The warm-up is NOT a way to dodge a finding. It is the opposite: with a
        # 330-entry `expect_zero` this suite would assert essentially nothing
        # about darkir forever, and every future dead weight would land inside
        # the waiver unnoticed. Evaluating one optimizer step past the documented
        # zero-init gate lets the oracle run with NO waivers over all 373
        # weights. Do NOT replace this with `expect_zero=(...)`, and do NOT
        # delete the companion init test below -- together they say "the zero is
        # a transient" rather than merely "the zero is tolerated".
        # See D-012 in plans/plan-2026-08-19T070627-a616f581/decisions.md
        model = _gf_model()
        x = _gf_batch()
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        model.fit(x, x, epochs=1, batch_size=2, verbose=0)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == len(model.trainable_weights)
        assert len(report) == GF_N_WEIGHTS, (
            "darkir's weight set changed shape; re-measure the init-time count "
            "in the companion test below"
        )

    def test_at_init_the_zero_gate_makes_most_branches_dead(self):
        """The mechanism, pinned. Without this the warm-up above is unexplained.

        Two claims: (a) the per-branch scales really are initialized to exactly
        0.0, and (b) that init really does zero the branch gradients. A change
        to either -- someone "fixing" the init to a small nonzero value, say --
        makes this test fail and the warm-up above unnecessary, which is a
        result worth being told about.
        """
        model = _gf_model()
        x = _gf_batch()
        model(x, training=False)

        gates = [
            w for w in model.trainable_weights
            if w.path.endswith("/beta") or w.path.endswith("/gamma")
        ]
        gates = [w for w in gates if "layer_normalization" not in w.path]
        assert gates, "no per-branch scale weights found -- has the block changed?"
        for w in gates:
            arr = keras.ops.convert_to_numpy(w)
            assert float(np.max(np.abs(arr))) == 0.0, (
                f"{w.path} is not zero-initialized any more (max|w| = "
                f"{float(np.max(np.abs(arr))):.3e}); the warm-up in the test "
                f"above may now be unnecessary"
            )

        report = gradient_report(model, x)
        assert _n_dead(report) == GF_N_DEAD_AT_INIT, (
            f"expected {GF_N_DEAD_AT_INIT} identically-zero gradients at init, "
            f"measured {_n_dead(report)}"
        )

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
        # DECISION plan-2026-08-19T070627-a616f581/D-012
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


# ---------------------------------------------------------------------
# The block tower, in VALUE space (plan-2026-08-19-a616f581 step 12)
# ---------------------------------------------------------------------

import contextlib

from dl_techniques.models.darkir.model import (
    DarkIRDecoderBlock,
    DarkIREncoderBlock,
)

from ..test_sam.dead_component_oracle import layer_returns_its_input

#: Blocks created by `create_darkir_model(width=16)` at its default
#: `enc_blk_nums`/`dec_blk_nums`. Pinned so a shape change to the tower is
#: reported here rather than silently shrinking the injection.
N_TOWER_BLOCKS = 15


def _tower_blocks(model):
    return [
        layer for layer in model.layers
        if isinstance(layer, (DarkIREncoderBlock, DarkIRDecoderBlock))
    ]


def _tower_kill_delta(model, x):
    """max|output move| when EVERY encoder/decoder block is replaced by identity.

    That injection is exactly the mutation `F-62` names -- deleting both
    `for j in range(num_blocks)` loops from `create_darkir_model` -- expressed
    without editing the product. `layer_returns_its_input` asserts on exit that
    each identity was actually invoked, so a zero delta can never be an artefact
    of an injection that missed the executed path.
    """
    blocks = _tower_blocks(model)
    assert len(blocks) == N_TOWER_BLOCKS, (
        f"expected {N_TOWER_BLOCKS} tower blocks, found {len(blocks)}"
    )
    base = keras.ops.convert_to_numpy(_primary(model(x, training=False)))
    with contextlib.ExitStack() as stack:
        for block in blocks:
            stack.enter_context(layer_returns_its_input(block, name=block.name))
        killed = keras.ops.convert_to_numpy(_primary(model(x, training=False)))
    return float(np.max(np.abs(killed - base)))


class TestDarkIRBlockTowerIsLoadBearing:
    """Value-space cover for the tower. The gradient tests above cover the graph.

    MEASURED, and it is the whole reason this class exists: with the class-level
    `call` of both block types replaced by `lambda inputs: inputs` -- the entire
    encoder/decoder tower deleted -- **12 of the 14 tests in this package pass**.
    Only the two `TestDarkIRGradientFlow` tests notice, and they notice on the
    BACKWARD graph. Every forward, side-loss, round-trip and dilation test in the
    suite is evaluated on an UNTRAINED model, and an untrained darkir is an exact
    identity through every block (see the ruling below), so no value-space
    assertion in this suite could ever have seen the tower disappear.
    """

    def test_at_init_the_whole_block_tower_is_an_exact_no_op(self):
        """The F-62 ruling, as a positive claim rather than a defect report.

        # DECISION plan-2026-08-19T070627-a616f581/D-016
        `F-62` recorded "darkir is an exact identity at initialization" as a
        REAL LIVE DEFECT. It is not a defect; it is the design, and this test
        asserts it. `models/darkir/model.py`'s own module docstring says so in
        as many words -- "Every block therefore begins training as an exact
        identity and the whole tower starts as the global residual connection
        alone. This is the LayerScale/ReZero idea" -- and D-012 already measured
        the consequence and its expiry: 330 of 373 weights carry an identically
        zero gradient at init and **zero** do after one optimizer step.
        MEASURED here: killing all 15 blocks at init moves the output by
        EXACTLY 0.0, and after one Adam(1e-2) step the same kill moves it by
        6.6e-02.

        WHAT NOT TO DO: do not "fix" the zero init to a small nonzero value to
        make this suite easier to write -- that would remove the warmup-free
        trainability the docstring is claiming. Do not delete this test: it is
        the negative control for the one below, and without it a `> 1e-3` delta
        at a warmed state proves only that SOMETHING moved, not that the gate is
        what moved it. See decisions.md D-016.
        """
        model = _gf_model()
        x = _gf_batch()
        model(x, training=False)

        delta = _tower_kill_delta(model, x)

        assert delta == 0.0, (
            "at init the block tower is supposed to be an exact identity "
            f"(LayerScale/ReZero gate at 0), but deleting it moved the output "
            f"by {delta:.6e}"
        )

    def test_once_the_gates_are_nonzero_killing_the_tower_moves_the_output(self):
        """The closing assertion: after training, the tower must carry signal.

        This is the test that goes RED on a darkir whose tower contributes
        nothing -- including the case F-62 actually feared, a gate that never
        leaves zero. Its own RED proof is the injection above applied to the
        PRODUCT (both block `call`s replaced class-wide): both arms then measure
        the identity and the delta is exactly 0.0.
        """
        keras.utils.set_random_seed(0)
        model = _gf_model()
        x = _gf_batch()
        model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
        model.fit(x, x, epochs=1, batch_size=2, verbose=0)

        gates = [
            w for w in model.trainable_weights
            if (w.path.endswith("/beta") or w.path.endswith("/gamma"))
            and "layer_normalization" not in w.path
        ]
        gate_max = max(
            float(np.max(np.abs(keras.ops.convert_to_numpy(w)))) for w in gates
        )
        assert gate_max > 0.0, (
            "precondition: one optimizer step must lift the ReZero gates off "
            f"zero, measured max|gate| = {gate_max:.3e}"
        )

        delta = _tower_kill_delta(model, x)

        # MEASURED on CPU: 6.640846e-02 against an exact 0.0 at init, i.e. the
        # two populations are separated by everything there is. The bound keeps
        # a 66x margin and is deliberately far above any fp32 rounding scale.
        assert delta > 1e-3, (
            "deleting darkir's entire block tower did not move the output of a "
            f"TRAINED model (max|delta| = {delta:.6e}): the tower is carrying "
            "no signal"
        )

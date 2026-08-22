"""LeWM's autoregressive predictor must not read the future.

``ARPredictor`` is named "autoregressive" and its stack is
``AdaLNZeroConditionalBlock``, whose ``use_causal_mask`` defaults to ``True``.
Nothing in the repo checked that. MEASURED 2026-08-21:
``grep -rn -i causal tests/test_models/test_lewm.py`` returned **nothing** — the
package's only test module had ZERO causality coverage, so the mask that makes
``LeWM.rollout`` mean anything was unguarded. (LeWM is tested by that loose module
rather than a directory, so a directory-shaped search finds nothing either.)

This is a MISSING-GUARD-OVER-A-CORRECT-PATH finding: the mask is applied at HEAD.

**Two traps had to be measured out of the way, and both silently defeat the obvious
version of this test:**

1. **A causality probe on an AT-INIT ``ARPredictor`` can never see anything.** Every
   block's ``adaLN_linear`` kernel AND bias are zero-initialized (AdaLN-*zero*), so
   ``gate_msa == gate_mlp == 0`` and BOTH residual branches — attention included —
   contribute exactly nothing. The forward pass reduces to
   ``final_norm(x + pos_embedding)``. A future token could not reach an earlier
   position because NO token reaches ANY other position. The decisive measurement is
   not that the at-init leak is 0.0 -- it is that the at-init leak stays **exactly
   0.0 with the causal mask REMOVED**, so an at-init "no leak" reading is the network
   being disconnected and says nothing about masking. ``_activate_gates`` assigns
   a seeded nonzero ``adaLN_linear`` kernel, standing in for a trained model, and
   ``test_the_probe_is_dead_at_init`` pins that this step is load-bearing rather than
   decorative.
2. **The perturbation must not be a constant channel-wide shift.** Adding the same
   scalar to every channel of one token is a pure mean shift, which the LayerNorms
   remove exactly. The probe REPLACES the token with a seeded random draw.

Three arms, because "masked != unmasked" is satisfied by ANY mask, an inverted one
included:

1. perturb position ``t``; every output position ``< t`` must be **exactly**
   unchanged (``max|delta| == 0.0``);
2. positions ``>= t`` must move (else arm 1 is trap 1 again);
3. **same-weights negative control** — flip ``block.use_causal_mask`` to ``False`` on
   the SAME instance (not one weight differs; no transfer, no re-init, no RNG draw),
   and require the leak to become nonzero. Then restore the flag and require the leak
   to return to exactly 0.0, so the control cannot have left a side effect doing the
   work.

MEASURED (CPU and GPU ``CUDA_VISIBLE_DEVICES=1``, identical): causal leak
``0.000000e+00`` / signal ``3.739430e+00``; mask off leak ``2.575221e-01``; restored
leak ``0.000000e+00``. Arm 1's bound is EXACT rather than a tolerance because a causal
mask makes the earlier positions' arithmetic literally independent of the perturbed
token — this is not a cancelling sum, so it is not reduction-order-dependent (contrast
the key-bias gradient zero of D-106, which is).

``dropout_rate`` and ``emb_dropout_rate`` are pinned to 0.0 and ``training=False`` is explicit:
one stochastic draw between the two forward passes would make arm 1 nonzero for a
reason unrelated to masking.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.lewm.predictor import ARPredictor

SEED, GATE_SEED, DATA_SEED = 20260821, 7, 0
BATCH, NUM_FRAMES, INPUT_DIM = 2, 6, 32
PERTURB_AT = 3


def _inputs():
    rng = np.random.default_rng(DATA_SEED)
    x = rng.standard_normal((BATCH, NUM_FRAMES, INPUT_DIM)).astype("float32")
    c = rng.standard_normal((BATCH, NUM_FRAMES, INPUT_DIM)).astype("float32")
    return x, c


def _built_predictor():
    keras.utils.set_random_seed(SEED)
    pred = ARPredictor(
        num_frames=NUM_FRAMES, depth=2, num_heads=4, dim_head=8, mlp_dim=64,
        input_dim=INPUT_DIM, hidden_dim=INPUT_DIM, dropout_rate=0.0, emb_dropout_rate=0.0,
    )
    x, c = _inputs()
    pred([x, c], training=False)          # build
    return pred


def _activate_gates(pred):
    """Un-zero the AdaLN-zero gates. Without this the attention path is OFF."""
    rng = np.random.default_rng(GATE_SEED)
    for block in pred.blocks:
        kernel = block.adaLN_linear.kernel
        kernel.assign(rng.normal(scale=0.3, size=kernel.shape).astype("float32"))


def _perturbed_x():
    x, _ = _inputs()
    x2 = x.copy()
    # A REPLACEMENT draw, not a constant offset -- LayerNorm eats a mean shift.
    x2[:, PERTURB_AT, :] = (
        np.random.default_rng(GATE_SEED).normal(size=(BATCH, INPUT_DIM)) * 3.0
    ).astype("float32")
    return x2


def _leak_and_signal(pred):
    """(max|delta| strictly BEFORE the perturbed position, max|delta| at/after it)."""
    x, c = _inputs()
    run = lambda xx: keras.ops.convert_to_numpy(
        pred([keras.ops.convert_to_tensor(xx), keras.ops.convert_to_tensor(c)],
             training=False)
    )
    delta = np.abs(run(_perturbed_x()) - run(x))
    return (float(np.max(delta[:, :PERTURB_AT, :])),
            float(np.max(delta[:, PERTURB_AT:, :])))


class TestTheLewmPredictorIsCausal:

    def test_the_probe_is_dead_at_init(self):
        """Trap 1, pinned: AdaLN-zero gates OFF -> no token reaches any other."""
        pred = _built_predictor()
        for block in pred.blocks:
            kernel = keras.ops.convert_to_numpy(block.adaLN_linear.kernel)
            bias = keras.ops.convert_to_numpy(block.adaLN_linear.bias)
            assert np.all(kernel == 0.0) and np.all(bias == 0.0)

        # At init the leak is zero -- and STAYS zero with the causal mask REMOVED,
        # which is the proof that the at-init zero says nothing about masking.
        leak, signal = _leak_and_signal(pred)
        assert leak == 0.0
        assert signal > 1e-3, "the perturbation must at least move its OWN position"

        for block in pred.blocks:
            block.use_causal_mask = False
        open_leak, _ = _leak_and_signal(pred)
        assert open_leak == 0.0, (
            f"the at-init predictor DOES pass information between tokens ({open_leak:.6e}); "
            "`_activate_gates` may no longer be necessary -- re-derive before trusting "
            "the arms below"
        )

    def test_a_future_token_cannot_reach_an_earlier_position(self):
        pred = _built_predictor()
        _activate_gates(pred)
        leak, signal = _leak_and_signal(pred)

        assert signal > 1e-3, (
            f"the perturbation moved nothing ({signal:.6e}); the gates are still off "
            "and arm 1's zero would mean nothing"
        )
        assert leak == 0.0, (
            f"perturbing position {PERTURB_AT} moved an EARLIER output by {leak:.6e}; "
            "the AR predictor reads the future"
        )

    def test_the_same_weights_without_the_mask_do_leak(self):
        """Negative control. One flag flips; not one weight changes."""
        pred = _built_predictor()
        _activate_gates(pred)
        causal_leak, _ = _leak_and_signal(pred)
        assert causal_leak == 0.0

        for block in pred.blocks:
            assert block.use_causal_mask is True
            block.use_causal_mask = False
        open_leak, open_signal = _leak_and_signal(pred)

        assert open_signal > 1e-3
        assert open_leak > 1e-4, (
            "removing the causal mask from the SAME layer changed nothing "
            f"({open_leak:.6e}) -- the probe cannot see a leak, so the causal "
            "reading proves nothing"
        )

        for block in pred.blocks:
            block.use_causal_mask = True
        restored_leak, _ = _leak_and_signal(pred)
        assert restored_leak == 0.0, (
            f"the leak did not return to zero after restoring the mask: {restored_leak:.6e}"
        )

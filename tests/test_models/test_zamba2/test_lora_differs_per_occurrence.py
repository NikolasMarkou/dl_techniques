"""Sentence-named guard: the highest-risk claim in the whole plan.

:class:`Zamba2SharedMLPBlock` is ONE physical instance, built once per
mem-block slot, called at several different depth positions via
``call(hidden_state, occurrence_idx=...)``. The plan's defining mechanic
(invariant 2, plan.md) requires BOTH of these to hold simultaneously:

1. The block's BASE weights (``norm``, ``gate_proj``, ``up_proj``,
   ``down_proj``) are bit-identical ``Variable`` objects at every call site,
   regardless of ``occurrence_idx`` -- plain Keras weight sharing.
2. The owned ``LoRAAdapter``'s contribution DIFFERS across two different
   ``occurrence_idx`` values on identical input, and a gradient update to
   one occurrence's LoRA pair does NOT move another occurrence's pair.

A regression that collapsed occurrence selection into "the stack reads only
the last occurrence" (v2 guide Section 12.7, this plan's Pre-Mortem item 1)
would satisfy claim 1 trivially (the weights really are shared -- that part
of the mechanism is fine) while silently failing claim 2. Conversely, a
regression that built a FRESH block per occurrence (losing the whole point
of the mem-block) would satisfy claim 2 trivially while failing claim 1.
Both must be checked in the same test to close that trap.

At construction every LoRA delta is identically zero (``B`` is zero-init,
the standard LoRA convention -- see ``LoRAAdapter``'s own docstring), so
this guard first takes one training step per occurrence, with a different
target each time, to move the occurrences apart before asserting they
differ.
"""

import numpy as np
import tensorflow as tf
import keras

from dl_techniques.models.language.zamba2.layers import Zamba2SharedMLPBlock


def test_lora_differs_per_occurrence() -> None:
    """Base weights stay ``is``-identical across occurrences; the LoRA
    delta genuinely diverges across occurrences after training; calling
    the SAME occurrence twice reproduces the identical output; and a
    gradient step on one occurrence's LoRA pair leaves another occurrence's
    pair untouched."""
    block = Zamba2SharedMLPBlock(
        d_model=32, num_occurrences=3, hidden_dim=64, lora_rank=4, lora_alpha=8.0
    )
    hidden_state = keras.random.normal(shape=(2, 10, 32))

    # --- Claim 1: base weights are is-identical across two occurrences ---
    _ = block(hidden_state, occurrence_idx=0)
    weights_at_occurrence_0 = list(block.weights)
    _ = block(hidden_state, occurrence_idx=1)
    weights_at_occurrence_1 = list(block.weights)

    assert len(weights_at_occurrence_0) == len(weights_at_occurrence_1)
    for w0, w1 in zip(weights_at_occurrence_0, weights_at_occurrence_1):
        assert w0 is w1, (
            f"Weight {w0.name!r} differs between occurrence_idx=0 and "
            "occurrence_idx=1 call sites -- the mem-block is not actually "
            "shared across occurrences."
        )

    # --- Same occurrence, twice: outputs must be IDENTICAL (no hidden
    # per-call-order state leak). ---
    output_0_first = keras.ops.convert_to_numpy(block(hidden_state, occurrence_idx=0))
    output_0_second = keras.ops.convert_to_numpy(block(hidden_state, occurrence_idx=0))
    np.testing.assert_allclose(
        output_0_first, output_0_second, rtol=0, atol=0,
        err_msg="Calling the same occurrence_idx twice on identical input "
                "must reproduce the identical output -- a hidden per-call "
                "state leak would break this.",
    )

    # --- Claim 2: the LoRA delta genuinely diverges once trained. Before
    # training, B is zero, so every occurrence's output is identical -- this
    # is the correct, non-vacuous starting point, not a defect. ---
    pre_training_output_0 = keras.ops.convert_to_numpy(block(hidden_state, occurrence_idx=0))
    pre_training_output_1 = keras.ops.convert_to_numpy(block(hidden_state, occurrence_idx=1))
    np.testing.assert_allclose(
        pre_training_output_0, pre_training_output_1, rtol=0, atol=1e-6,
        err_msg="At construction (B zero-init) every occurrence's output "
                "should coincide -- if this fails, occurrence_idx is doing "
                "something before training has a chance to.",
    )

    optimizer = keras.optimizers.Adam(learning_rate=1e-1)
    targets = {
        0: keras.ops.ones((2, 10, 32)),
        1: keras.ops.ones((2, 10, 32)) * -1.0,
    }
    for occurrence_idx, target in targets.items():
        with tf.GradientTape() as tape:
            output = block(hidden_state, occurrence_idx=occurrence_idx)
            loss = keras.ops.mean(keras.ops.square(output - target))
        grads = tape.gradient(loss, block.trainable_variables)
        optimizer.apply_gradients(zip(grads, block.trainable_variables))

    output_after_training_0 = keras.ops.convert_to_numpy(block(hidden_state, occurrence_idx=0))
    output_after_training_1 = keras.ops.convert_to_numpy(block(hidden_state, occurrence_idx=1))
    assert not np.allclose(output_after_training_0, output_after_training_1), (
        "Two different occurrence_idx values produced identical output "
        "after diverging training steps -- the LoRA pairs are not "
        "independent (the §12.7 'reads only the last occurrence' failure)."
    )

    # Same occurrence, twice, AFTER training: still identical.
    output_0_after_a = keras.ops.convert_to_numpy(block(hidden_state, occurrence_idx=0))
    output_0_after_b = keras.ops.convert_to_numpy(block(hidden_state, occurrence_idx=0))
    np.testing.assert_allclose(
        output_0_after_a, output_0_after_b, rtol=0, atol=0,
        err_msg="Same occurrence_idx must still reproduce identical output "
                "after training.",
    )

    # --- A gradient computed through an occurrence_idx=0 call must carry a
    # ZERO gradient for occurrence 1's and occurrence 2's LoRA slices. This
    # is checked on the GRADIENT itself, not on a stateful optimizer's
    # post-update weights: Adam's own momentum/velocity accumulators (built
    # up by the upstream training loop) keep moving an already-zero-gradient
    # slice on every subsequent `apply_gradients` call, which is correct
    # Adam behaviour, not a leak in this layer's independent-slice
    # parameterization -- asserting on optimizer state here would fail for
    # a reason that has nothing to do with the claim under test.
    with tf.GradientTape() as tape:
        output = block(hidden_state, occurrence_idx=0)
        loss = keras.ops.mean(keras.ops.square(output - targets[0]))
    lora_a_grad, lora_b_grad = tape.gradient(loss, [block.lora.a, block.lora.b])

    lora_a_grad_np = keras.ops.convert_to_numpy(lora_a_grad)
    lora_b_grad_np = keras.ops.convert_to_numpy(lora_b_grad)

    for unexercised_idx in (1, 2):
        np.testing.assert_allclose(
            lora_a_grad_np[unexercised_idx], 0.0, rtol=0, atol=0,
            err_msg=f"occurrence_idx=0's loss produced a nonzero gradient on "
                    f"occurrence {unexercised_idx}'s LoRA A -- the "
                    "per-occurrence slices are not independently selected.",
        )
        np.testing.assert_allclose(
            lora_b_grad_np[unexercised_idx], 0.0, rtol=0, atol=0,
            err_msg=f"occurrence_idx=0's loss produced a nonzero gradient on "
                    f"occurrence {unexercised_idx}'s LoRA B.",
        )
    assert not np.allclose(lora_a_grad_np[0], 0.0), (
        "occurrence_idx=0's loss produced a zero gradient on occurrence 0's "
        "own LoRA A -- the gradient check above would be vacuous if the "
        "exercised slice itself never received a gradient."
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

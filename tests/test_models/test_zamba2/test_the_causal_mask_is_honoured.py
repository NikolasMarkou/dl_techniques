"""The Zamba2SharedAttentionBlock must not read the future.

Three-armed future-leak probe (plan.md step 2, v2 guide Section 12.1), on
:class:`Zamba2SharedAttentionBlock` directly. A shape test is blind to mask
direction entirely -- a fully-unmasked block and a correctly-causal one
return identically shaped output -- so this file perturbs one token and
checks WHERE the change propagates.

* **arm 1** -- perturb the token at position ``t``; every output position
  ``< t`` is bit-identical (``atol=0``, ``rtol=0``). Exact rather than
  tolerant, because a causal block's arithmetic at an earlier position never
  touches the perturbed token at all -- there is no cancelling sum whose
  reduction order could move the result by a rounding epsilon.
* **arm 2** -- at least one output position ``>= t`` moves. Mandatory twin:
  arm 1 alone passes on a block that ignores its input (e.g. a dead
  attention path), not only on a correctly-causal one.
* **arm 3** -- arm 1 repeated with the perturbation at the LAST position,
  where the "before" window is the whole sequence bar one -- a leak ANYWHERE
  shows up here -- together with the requirement that the last position
  itself moved (so the block is not simply disconnected).
"""

from typing import Tuple

import numpy as np
import keras
import pytest

from dl_techniques.models.language.zamba2.layers import Zamba2SharedAttentionBlock

BATCH_SIZE = 2
SEQ_LEN = 12
D_MODEL = 32
NUM_HEADS = 4
PERTURB_AT = 5
SIGNAL_FLOOR = 1e-6
SEED = 1234


def _block() -> Zamba2SharedAttentionBlock:
    """A freshly constructed block, deterministic at ``training=False``."""
    return Zamba2SharedAttentionBlock(
        d_model=D_MODEL, num_heads=NUM_HEADS, max_seq_len=SEQ_LEN * 2
    )


def _inputs() -> Tuple[np.ndarray, np.ndarray]:
    """A fixed ``(hidden_state, original_embedding)`` pair."""
    rng = np.random.default_rng(SEED)
    hidden_state = rng.normal(size=(BATCH_SIZE, SEQ_LEN, D_MODEL)).astype("float32")
    original_embedding = rng.normal(size=(BATCH_SIZE, SEQ_LEN, D_MODEL)).astype("float32")
    return hidden_state, original_embedding


def _perturbed(hidden_state: np.ndarray, position: int) -> np.ndarray:
    """``hidden_state`` with position ``position`` replaced by a different value."""
    perturbed = hidden_state.copy()
    perturbed[:, position, :] = hidden_state[:, position, :] + 10.0
    return perturbed


def _output(block: Zamba2SharedAttentionBlock, hidden_state: np.ndarray, original_embedding: np.ndarray) -> np.ndarray:
    """Run the block once, deterministically, and return a numpy array."""
    out = block(
        keras.ops.convert_to_tensor(hidden_state),
        keras.ops.convert_to_tensor(original_embedding),
        training=False,
    )
    return keras.ops.convert_to_numpy(out)


def _delta(block: Zamba2SharedAttentionBlock, hidden_state: np.ndarray, original_embedding: np.ndarray, position: int) -> np.ndarray:
    """``|output(perturbed) - output(original)|`` at every position."""
    baseline = _output(block, hidden_state, original_embedding)
    perturbed_output = _output(block, _perturbed(hidden_state, position), original_embedding)
    return np.abs(perturbed_output - baseline)


def test_the_perturbation_changes_every_row() -> None:
    """A perturbation that coincided with the original would make arm 1
    vacuous on that row and the vacuity would never be reported."""
    hidden_state, _ = _inputs()
    for position in (0, PERTURB_AT, SEQ_LEN - 1):
        assert np.all(_perturbed(hidden_state, position)[:, position] != hidden_state[:, position])


def test_the_block_repeats_itself_bit_exactly() -> None:
    """Arm 1's ``atol=0`` is only meaningful if the forward pass is
    deterministic at ``training=False``."""
    block = _block()
    hidden_state, original_embedding = _inputs()

    first = _output(block, hidden_state, original_embedding)
    second = _output(block, hidden_state, original_embedding)

    spread = float(np.max(np.abs(first - second)))
    assert spread == 0.0, f"the forward is not deterministic: {spread:.6e}"


def test_arm_1_a_perturbed_token_cannot_reach_an_earlier_position() -> None:
    """Every output row before ``PERTURB_AT`` is bit-identical to baseline."""
    block = _block()
    hidden_state, original_embedding = _inputs()

    delta = _delta(block, hidden_state, original_embedding, PERTURB_AT)
    before = delta[:, :PERTURB_AT]

    assert before.size > 0, "the 'before' window is empty; arm 1 would be vacuous"
    np.testing.assert_allclose(
        before, np.zeros_like(before), atol=0.0, rtol=0,
        err_msg=(
            f"a token at position {PERTURB_AT} reached at least one output row "
            f"at an EARLIER position (max|delta| = {float(np.max(before)):.6e}); "
            "the shared attention mem-block attends to its own future"
        ),
    )


def test_arm_2_the_perturbed_token_does_reach_its_own_position_and_after() -> None:
    """The mandatory twin. Arm 1 alone passes on a block that ignores its input."""
    block = _block()
    hidden_state, original_embedding = _inputs()

    delta = _delta(block, hidden_state, original_embedding, PERTURB_AT)
    after = delta[:, PERTURB_AT:]

    signal = float(np.max(after))
    assert signal > SIGNAL_FLOOR, (
        f"perturbing the token at position {PERTURB_AT} moved NOTHING at or after "
        f"it (max|delta| = {signal:.6e}); arm 1's zero above is a dead block, not "
        "a causal one"
    )


def test_arm_3_the_last_token_cannot_reach_any_earlier_position() -> None:
    """Arm 1 with the perturbation at the LAST position -- a leak anywhere
    in the sequence shows up here."""
    block = _block()
    hidden_state, original_embedding = _inputs()
    last = SEQ_LEN - 1

    delta = _delta(block, hidden_state, original_embedding, last)
    before = delta[:, :last]

    assert before.shape[1] == SEQ_LEN - 1
    np.testing.assert_allclose(
        before, np.zeros_like(before), atol=0.0, rtol=0,
        err_msg=(
            f"the last token reached an earlier output row "
            f"(max|delta| = {float(np.max(before)):.6e})"
        ),
    )

    last_position_signal = float(np.max(delta[:, last:]))
    assert last_position_signal > SIGNAL_FLOOR, (
        "the last token did not even reach its own output position "
        f"(max|delta| = {last_position_signal:.6e}); a disconnected block would "
        "also satisfy the zero-leak assertion above for the wrong reason"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

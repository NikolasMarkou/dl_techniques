"""
`NAMCell`'s reduction scorer must survive `mixed_float16`.

`models/nam/cell.py` masked its reduction scores with

    scores = scores + (1.0 - token_mask_float) * (-1e9)

where `token_mask_float` carries the layer's `compute_dtype`. Under
`mixed_float16` that dtype is `float16`, whose finite range stops at ~-6.55e4, so
the `-1e9` constant becomes `-inf` on conversion. Two things then go wrong at
once, and the SECOND is the one a "check the masked positions" test misses:

* a masked position gets `-inf`, and a row that is masked END TO END softmaxes
  `[-inf, ...]` to NaN;
* an UNMASKED position multiplies `0.0 * -inf`, which is NaN in IEEE-754 -- so
  the corruption lands on the positions the mask was meant to KEEP, in a row with
  a perfectly ordinary mask.

This is prior finding C-5, the last fp16-unsafe additive mask left in `models/`.
It survived the 10-site `layers/attention/` migration because it is a REDUCTION
SCORER, not attention -- `apply_attention_mask` has zero call sites anywhere
inside `models/`.

RED, measured on CPU before the fix, `NAMCell` at `hidden_size=32`,
`max_expression_len=16`, global policy `mixed_float16`:

    reduction_weights finite: False  (16 of 32 entries NaN, i.e. every position
                                      of both rows, masked and unmasked alike)
    result finite:            False

Both arms are GREEN under `float32` before and after the fix, which is what makes
this a dtype defect rather than a masking defect.

A SECOND, independent fp16 blocker sat further down the same `call`, and this
probe is what found it: the deterministic digit assembly pins itself to
``"float32"`` on purpose while Keras autocasts the incoming `carry` to
`compute_dtype`, so the accumulator add raised

    InvalidArgumentError: cannot compute AddV2 as input #1(zero-based) was
    expected to be a half tensor but is a float tensor

Both repairs were isolated: with the accumulator cast in place and ONLY the mask
sentinel reverted, three of the four fp16 arms below fail again with
``[nan nan nan nan nan nan nan]`` at the KEPT positions. Neither fix hides the
other.

The global dtype policy is PROCESS-GLOBAL. `tests/test_layers/conftest.py` owns
the house fixture for it, but that conftest does not reach `tests/test_models/`,
so the same shape -- capture, set, restore in `finally`, and ASSERT the
restoration -- is reproduced here rather than left implicit.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.nam import NAMCell, NAMConfig


@pytest.fixture
def cell_config() -> NAMConfig:
    return NAMConfig(
        hidden_size=32,
        num_heads=4,
        num_tree_layers=1,
        intermediate_size=64,
        memory_size=8,
        num_read_heads=2,
        max_expression_len=16,
        halt_max_steps=4,
        hidden_dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )


@pytest.fixture
def mixed_float16():
    """Set the GLOBAL policy for one test, then always restore and verify it."""
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield "mixed_float16"
    finally:
        keras.mixed_precision.set_global_policy(previous)
        assert keras.mixed_precision.global_policy().name == previous, (
            "the global dtype policy leaked out of this test; every later "
            "numeric assertion in the session now runs in the wrong regime"
        )


def _run(config: NAMConfig, dtype: str):
    """One cell forward with a PARTIALLY masked row and a FULLY masked row."""
    cell = NAMCell(config=config)
    batch, length = 2, config.max_expression_len

    carry = cell.initialize_carry(batch)
    hidden = ops.cast(keras.random.normal((batch, length, config.hidden_size)), dtype)

    mask = np.zeros((batch, 1, length), dtype="int32")
    mask[0, 0, :7] = 1   # ordinary right-padded row: 7 real tokens, 9 padded
    # row 1 stays all-zero: the fully-masked row softmax([-inf, ...]) = NaN arm

    token_ids = np.zeros((batch, length), dtype=np.int32)
    token_ids[:, 0] = 1
    token_ids[:, 1] = 5
    token_ids[:, 3] = 14
    token_ids[:, 5] = 6
    token_ids[:, 6] = 2

    _, outputs = cell(
        (carry, hidden, ops.convert_to_tensor(mask), ops.convert_to_tensor(token_ids)),
        training=False,
    )
    return {k: np.asarray(ops.convert_to_numpy(v)) for k, v in outputs.items()
            if k in ("reduction_weights", "result")}


class TestReductionMaskUnderMixedFloat16:

    def test_the_kept_positions_are_finite(self, cell_config, mixed_float16) -> None:
        out = _run(cell_config, "float16")
        kept = out["reduction_weights"][0, :7]
        assert np.all(np.isfinite(kept)), (
            "the UNMASKED positions of an ordinarily right-padded row are not "
            f"finite under mixed_float16: {kept}. `0.0 * float16(-1e9)` is "
            "`0.0 * -inf` = NaN, so the mask corrupts the very positions it "
            "keeps (finding C-5)."
        )

    def test_a_fully_masked_row_is_finite(self, cell_config, mixed_float16) -> None:
        out = _run(cell_config, "float16")
        row = out["reduction_weights"][1]
        assert np.all(np.isfinite(row)), (
            f"softmax over an end-to-end masked row is not finite: {row}"
        )

    def test_the_result_is_finite(self, cell_config, mixed_float16) -> None:
        out = _run(cell_config, "float16")
        assert np.all(np.isfinite(out["result"])), (
            f"the cell's arithmetic result is not finite: {out['result']}"
        )

    def test_the_masked_positions_still_get_no_weight(self, cell_config, mixed_float16) -> None:
        """The repair must not weaken the mask it repairs.

        -1e4 is finite in fp16 but it is not -inf, so this arm is what
        distinguishes 'the sentinel was made survivable' from 'the sentinel was
        made ineffective'.
        """
        out = _run(cell_config, "float16")
        padded = out["reduction_weights"][0, 7:]
        assert float(np.max(padded)) < 1e-3, (
            f"padded positions still carry reduction weight: max={np.max(padded)}"
        )


class TestFloat32IsTheControl:
    """The float32 arm is green both before and after -- this is a dtype defect."""

    def test_everything_is_finite_in_float32(self, cell_config) -> None:
        out = _run(cell_config, "float32")
        assert np.all(np.isfinite(out["reduction_weights"]))
        assert np.all(np.isfinite(out["result"]))

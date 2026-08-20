"""`T5Encoder`'s padding mask must work under `mixed_float16`.

Rationale
---------
The additive mask was ``position_bias + (1 - mask) * np.finfo(np.float32).min``.
MEASURED at HEAD, CPU:

    float32        no mask    finite, |max| 3.125694e+00
    float32        with mask  finite, |max| 3.051183e+00
    mixed_float16  no mask    finite, |max| 2.552734e+00
    mixed_float16  with mask  InvalidArgumentError

so the encoder raised under `mixed_float16` for every masked call and was green
for every unmasked one -- i.e. the defect is invisible to any test that does not
supply a mask.

Why `ops.where` and not a bigger cast
-------------------------------------
Casting the same sentinel to the bias dtype makes it ``-inf`` in float16, and
then ``(1 - keep) * -inf`` is ``0 * -inf = NaN`` on the KEPT positions. MEASURED
on ``bias = [-1.0, 0.5, -3.0, 2.0]``, ``keep = [1, 1, 0, 0]``:

    additive, finfo(f32).min cast -> [nan, nan, -inf, -inf], softmax 4 NaN of 4
    additive, finfo(f16).min      -> [-1.0, 0.5, -65504, -65504], 0 NaN
    ops.where, finfo(f16).min     -> [-1.0, 0.5, -65504, -65504], 0 NaN

and on an ALL-MASKED row the additive `-inf` form gives 4 NaN of 4 while
`ops.where` gives 0. A dtype-aware additive form is therefore also correct;
`ops.where` is chosen because it cannot reach either failure.

See decisions.md D-033 (plan-2026-08-19T163559-499b6f0e).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.sd3_mmdit.text_encoders import T5Encoder

# ---------------------------------------------------------------------

BATCH = 2
SEQ = 8
VOCAB = 50
KEPT = 5


@pytest.fixture
def inputs():
    ids = np.random.RandomState(0).randint(
        1, VOCAB, size=(BATCH, SEQ)).astype("int32")
    mask = np.ones((BATCH, SEQ), dtype="int32")
    mask[:, KEPT:] = 0
    return ids, mask


def _encoder() -> T5Encoder:
    return T5Encoder(vocab_size=VOCAB, embed_dim=16, num_layers=1,
                     num_heads=2, ff_dim=32)


@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
@pytest.mark.parametrize("with_mask", [False, True])
def test_the_forward_pass_is_finite(inputs, policy, with_mask):
    """The 2x2 grid whose one broken cell was `mixed_float16` WITH a mask."""
    ids, mask = inputs
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy(policy)
    try:
        encoder = _encoder()
        output = np.array(encoder(
            ids, attention_mask=mask if with_mask else None, training=False))
    finally:
        keras.mixed_precision.set_global_policy(previous)

    assert np.all(np.isfinite(output)), (
        f"policy={policy}, with_mask={with_mask}: "
        f"{int(np.sum(~np.isfinite(output)))} of {output.size} elements are not "
        f"finite"
    )


def test_an_all_masked_row_does_not_produce_nan(inputs):
    """The row shape that an `-inf` sentinel turns into 100% NaN.

    Physically degenerate, but it is the exact input that separates a `where`
    from an additive `-inf`, and a real pipeline reaches it whenever a sequence
    is entirely padding.
    """
    ids, _ = inputs
    all_masked = np.zeros((BATCH, SEQ), dtype="int32")
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        output = np.array(_encoder()(
            ids, attention_mask=all_masked, training=False))
    finally:
        keras.mixed_precision.set_global_policy(previous)
    assert int(np.sum(np.isnan(output))) == 0, (
        f"an all-masked row produced {int(np.sum(np.isnan(output)))} NaN of "
        f"{output.size}"
    )


class TestTheMaskStillMasks:
    """Anti-vacuity: a mask that does nothing would pass every test above."""

    @pytest.fixture(scope="class")
    def encoder(self):
        keras.utils.set_random_seed(1234)
        return _encoder()

    def test_masking_changes_the_output(self, encoder, inputs):
        ids, mask = inputs
        masked = np.array(encoder(ids, attention_mask=mask, training=False))
        unmasked = np.array(encoder(ids, attention_mask=None, training=False))
        delta = float(np.max(np.abs(masked - unmasked)))
        assert delta > 1e-3, (
            f"supplying a mask changed the output by only {delta:.6e}; the mask "
            f"is not reaching the attention bias"
        )

    def test_padded_tokens_cannot_influence_the_kept_positions(
            self, encoder, inputs):
        """The semantic the mask exists for, pinned at EXACTLY zero."""
        ids, mask = inputs
        perturbed = ids.copy()
        perturbed[:, KEPT:] = VOCAB - 1
        a = np.array(encoder(ids, attention_mask=mask, training=False))
        b = np.array(encoder(perturbed, attention_mask=mask, training=False))
        delta = float(np.max(np.abs(a[:, :KEPT] - b[:, :KEPT])))
        assert delta == 0.0, (
            f"changing PADDED tokens moved the KEPT positions by {delta:.6e}; "
            f"the mask leaks"
        )

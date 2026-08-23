"""`BERT.pad_token_id` is ADVISORY ONLY -- nothing reads it, and that is pinned.

Rationale
---------
``pad_token_id`` is declared (``bert.py:321``), stored (``:417``) and serialized
(``:968``) and is read by nothing -- line numbers re-derived after this step's
own docstring edits, which moved all three. It was flagged as an R-120 dead-knob
violation on 2026-08-19 and never closed, while the ``distilbert`` sibling got
the full anchor + docstring + guard-test treatment in
plan-2026-08-10T183739-b007f435 (D-003). This file is that treatment, applied to
``BERT``; its shape is copied from
``tests/test_models/test_distilbert/test_model.py::test_attention_mask_is_never_derived_from_pad_token_id``.

The decision is DOCUMENT-AND-PIN, not wire-up. Deriving
``attention_mask = input_ids != pad_token_id`` inside ``call`` would silently
change the output of every mask-less forward pass that exists today, upstream HF
BERT does not do it either, and the derivation would have to live in the shared
``BertEmbeddings`` -- which ``fnet``, ``distilbert`` and ``modern_bert`` also
use. See ``decisions.md`` D-007
(plan-2026-08-23T203721-009b7ccf) and the ``# DECISION`` anchor at
``bert.py:400``.

MEASURED at HEAD, ``vocab_size=64, hidden_size=32, num_layers=2, num_heads=2``,
a ``(2, 6)`` int32 batch whose padding region is token id **5**:

    two models identical but for pad_token_id (0 vs 5), no mask   max|delta| = 0.0
    mask-less vs explicit all-ones mask                           max|delta| = 0.0
    all-ones mask vs the real (ids != 5) mask                     max|delta| = 0.02156496

The first two zeros are the inertness claim. The third number is the
anti-vacuity arm: without it, both zeros are satisfied by a model that does
nothing at all, which is this repo's documented dominant failure mode.

Assertion order is deliberate and must not be changed: the two equalities are
checked BEFORE the inequality, because a mutation that auto-derives a mask
satisfies neither and the inequality would shadow the equalities.

RED proofs, one injection per assertion, ACTUAL text
----------------------------------------------------
**Injection A -- derive the mask from ``pad_token_id``** in ``BERT.call`` when
none is supplied (``attention_mask = ops.cast(ops.not_equal(input_ids,
self.pad_token_id), "int32")``), i.e. exactly the "fix" D-007 forbids. Result:
**2 failed, 1 passed** of 3.

    AssertionError: pad_token_id is being READ: two models differing only in
    pad_token_id (0 vs 5) gave different outputs, max|delta| = 0.021564960479736328
    -- D-007 requires it to stay advisory

    AssertionError: a mask-less forward pass differs from an explicit all-ones
    one (max|delta| = 0.021564960479736328); a mask is being derived from
    pad_token_id, which D-007 forbids

**Injection B -- drop the mask before the encoder** (pass
``attention_mask=None`` at the ``encoder_layer(...)`` call site in ``call``).
The two inertness equalities still hold -- they are equalities between two
equally-dead forwards -- and ONLY the anti-vacuity arm fires. Result:
**1 failed, 2 passed**:

    AssertionError: an explicit attention_mask does not change the output
    (max|delta| = 0.0), so the two inertness assertions above are vacuous:
    they would pass against a model that ignores masking entirely
"""

import keras
import numpy as np

from dl_techniques.models.bert.bert import BERT

# ---------------------------------------------------------------------

SEED = 1234
PAD_REGION_TOKEN = 5

#: A batch whose tail is ``PAD_REGION_TOKEN`` -- the id one of the two models
#: below calls its padding token and the other does not.
TOKENS = np.array(
    [
        [3, 7, PAD_REGION_TOKEN, PAD_REGION_TOKEN, PAD_REGION_TOKEN, PAD_REGION_TOKEN],
        [9, PAD_REGION_TOKEN, 2, PAD_REGION_TOKEN, PAD_REGION_TOKEN, PAD_REGION_TOKEN],
    ],
    dtype="int32",
)


def _model(pad_token_id: int) -> BERT:
    """Two calls with the same seed give bit-identical weights.

    That is what makes ``pad_token_id`` the ONLY difference between the two
    models compared below -- without the seed the arm would be comparing two
    random draws and could never assert 0.0.
    """
    keras.utils.set_random_seed(SEED)
    return BERT(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        pad_token_id=pad_token_id,
        normalization_type="layer_norm",
    )


def _hidden(model: BERT, **extra) -> np.ndarray:
    inputs = {"input_ids": TOKENS, **extra}
    return np.array(model(inputs, training=False)["last_hidden_state"])


def test_the_setup_actually_contains_padding():
    """Anti-vacuity for the fixture itself: an all-real batch proves nothing."""
    mask = (TOKENS != PAD_REGION_TOKEN).astype("int32")
    assert mask.min() == 0, "setup: the batch has no padding region"
    assert mask.max() == 1, "setup: the batch is entirely padding"


def test_changing_pad_token_id_does_not_move_the_output_by_one_bit():
    """The inertness claim, stated as an exact bit-identity."""
    baseline = _hidden(_model(0))
    renamed = _hidden(_model(PAD_REGION_TOKEN))

    delta = float(np.abs(baseline - renamed).max())
    assert delta == 0.0, (
        f"pad_token_id is being READ: two models differing only in "
        f"pad_token_id (0 vs {PAD_REGION_TOKEN}) gave different outputs, "
        f"max|delta| = {delta} -- D-007 requires it to stay advisory"
    )


def test_a_maskless_call_is_exactly_the_all_ones_answer():
    """No mask is INFERRED, so padding is attended to like a real token.

    This is the second half of "advisory": a caller who omits
    ``attention_mask`` gets full attention over the padding region, not a
    silently derived mask. It is checked before the anti-vacuity arm below
    because an auto-deriving mutation fails both and the inequality would
    otherwise shadow it.
    """
    model = _model(PAD_REGION_TOKEN)
    no_mask = _hidden(model)
    all_ones = _hidden(model, attention_mask=np.ones_like(TOKENS))

    delta = float(np.abs(no_mask - all_ones).max())
    assert delta == 0.0, (
        f"a mask-less forward pass differs from an explicit all-ones one "
        f"(max|delta| = {delta}); a mask is being derived from pad_token_id, "
        "which D-007 forbids"
    )

    # The anti-vacuity arm. Without it BOTH zeros above are satisfied by a
    # model that ignores attention masking entirely. Measured at this config:
    # 0.02156496.
    real_mask = (TOKENS != PAD_REGION_TOKEN).astype("int32")
    masked = _hidden(model, attention_mask=real_mask)
    moved = float(np.abs(masked - no_mask).max())
    assert moved > 1e-4, (
        f"an explicit attention_mask does not change the output "
        f"(max|delta| = {moved}), so the two inertness assertions above are "
        "vacuous: they would pass against a model that ignores masking "
        "entirely"
    )

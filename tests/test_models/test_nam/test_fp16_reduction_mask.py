"""
`NAMCell`'s reduction scorer must survive `mixed_float16`.

`models/neural_computer/nam/cell.py` masked its reduction scores with

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

`mixed_bfloat16`, and why it was missed
---------------------------------------
The house dtype fixture is parametrized over ``("float32", "mixed_float16",
"float64")`` -- ``tests/test_layers/conftest.py:28``. bfloat16 is in NO arm of
it, anywhere in this suite, which is exactly why `NAMCell.call` could RAISE
under `mixed_bfloat16` for the whole life of the fp16 work above while 111
`test_nam` tests stayed green.

MEASURED at `1276b4ddc`, `mixed_bfloat16`, CPU, the same cell as `_run` builds::

    ValueError: data type <class 'ml_dtypes.bfloat16'> not inexact
      models/neural_computer/nam/cell.py:612
        halt_eps = max(1e-9, float(np.finfo(self.compute_dtype).tiny))

The mechanism is NOT a numeric one. Under a bfloat16 policy `self.compute_dtype`
is the string ``"bfloat16"``; numpy 2.0.2 resolves that name through the
`ml_dtypes` registration to ``ml_dtypes.bfloat16``, and then `np.finfo.__new__`
rejects it as "not inexact" because numpy does not classify the extension dtype
as a floating type. So the hand-rolled `max(literal, finfo(...).tiny)` epsilon --
the pattern this plan created `utils.dtype_policy` to stop N files from copying --
does not merely return a wrong number under bfloat16; it cannot be evaluated at
all. `dtype_policy._dtype_facts` routes around the same numpy fact through
`ml_dtypes.finfo` (decisions.md D-006), which is why the mask sentinel two dozen
lines above this site was already bfloat16-safe.

The site is a DIVISOR with an ADDED epsilon (`x / (sum(mask) + eps)`), not an
`ops.maximum` clamp, so D-014's rule routes it to `accumulation_dtype` promotion
rather than to `stability_floor` in the compute dtype. Both casts are identities
at float32/float64, so the float32 control below is unchanged bit-for-bit.

TWO SIBLING SITES ARE STILL LIVE, and this file does not cover them. The same
hand-rolled `max(<literal>, float(np.finfo(self.compute_dtype).tiny))` -- also
an ADDED divisor epsilon, also carrying the D-050 comment -- remains at
`models/neural_computer/nam/model.py:450` (`NAM.call`'s masked pooled mean) and
`models/vision/keypoints/superpoint/model.py:351` (descriptor L2 normalization).
Both raise under `mixed_bfloat16` for the identical numpy reason. They are
REPORTED, NOT FIXED: this completion fix is scoped to the site the review named,
and neither is reached by `NAMCell.call`, so neither can hide behind the arms
below. `tests/test_the_mask_sentinel_population_is_closed.py` cannot see them
either -- it censuses mask sentinels, not stability floors, and the review
separately measured 74 unguarded floor sites in 43 files.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.neural_computer.nam import NAMCell, NAMConfig


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


@pytest.fixture
def mixed_bfloat16():
    """Set the GLOBAL policy for one test, then always restore and verify it.

    Same shape as ``mixed_float16`` above. Deliberately a SEPARATE fixture
    rather than a parametrization of it: the two policies fail differently --
    float16 by numeric overflow, bfloat16 by a numpy type-classification error
    that raises before any arithmetic happens -- and merging them would hide
    which regime a red arm belongs to.
    """
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    try:
        yield "mixed_bfloat16"
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


class TestReductionMaskUnderMixedBfloat16:
    """`NAMCell.call` must RUN under `mixed_bfloat16`, not just under float16.

    The first arm is the one that matters and it is not a numeric assertion at
    all: before the completion fix, `cell.py:612` raised
    ``ValueError: data type <class 'ml_dtypes.bfloat16'> not inexact`` and the
    forward pass produced nothing to measure. The remaining arms mirror the
    float16 class so that a repair which makes the call RUN but silently breaks
    the mask, or the arithmetic, still fails here.
    """

    def test_the_forward_pass_completes_at_all(
        self, cell_config, mixed_bfloat16
    ) -> None:
        """RED before the fix with a raise, not with a bad number.

        `np.finfo("bfloat16")` is a `ValueError` under this environment's
        numpy 2.0.2 / ml_dtypes 0.4.1, so every hand-rolled
        `max(literal, np.finfo(compute_dtype).tiny)` in the tree is a landmine
        for exactly one dtype policy. Routed through
        `utils.dtype_policy.accumulation_dtype` / `stability_floor`, which
        resolve bfloat16 via `ml_dtypes.finfo`.
        """
        out = _run(cell_config, "bfloat16")
        assert set(out) == {"reduction_weights", "result"}

    def test_the_kept_positions_are_finite(self, cell_config, mixed_bfloat16) -> None:
        out = _run(cell_config, "bfloat16")
        kept = out["reduction_weights"][0, :7]
        assert np.all(np.isfinite(kept)), (
            f"kept positions are not finite under mixed_bfloat16: {kept}"
        )

    def test_a_fully_masked_row_is_finite(self, cell_config, mixed_bfloat16) -> None:
        out = _run(cell_config, "bfloat16")
        row = out["reduction_weights"][1]
        assert np.all(np.isfinite(row)), (
            f"softmax over an end-to-end masked row is not finite: {row}"
        )

    def test_the_result_is_finite(self, cell_config, mixed_bfloat16) -> None:
        out = _run(cell_config, "bfloat16")
        assert np.all(np.isfinite(out["result"])), (
            f"the cell's arithmetic result is not finite: {out['result']}"
        )

    def test_the_masked_positions_still_get_no_weight(
        self, cell_config, mixed_bfloat16
    ) -> None:
        """`mask_sentinel('bfloat16')` is -9984.0, a snapped grid point.

        It is asserted against itself in `test_utils/test_dtype_policy.py`; this
        is the only arm in the suite that measures whether it actually masks.
        """
        out = _run(cell_config, "bfloat16")
        padded = out["reduction_weights"][0, 7:]
        assert float(np.max(padded)) < 1e-3, (
            f"padded positions still carry reduction weight: max={np.max(padded)}"
        )


class TestTheHaltDivisorEpsilonIsDtypeResolvable:
    """The narrow, mechanism-level pin for `cell.py:612`.

    The forward-pass arms above would also go green if somebody deleted the
    epsilon outright, or replaced it with a bare `1e-9` that is `0.0` in
    float16. This asserts the two properties the site actually needs, at every
    dtype policy the cell can be run under, without running the cell: the
    epsilon must be COMPUTABLE (bfloat16 is where the hand-rolled version
    raised) and it must be STRICTLY POSITIVE once materialized (float16 is
    where a bare `1e-9` reads as protection and provides none).
    """

    @pytest.mark.parametrize(
        "compute_dtype", ["float32", "float64", "float16", "bfloat16"]
    )
    def test_the_epsilon_is_computable_and_nonzero_in_every_dtype(
        self, compute_dtype: str
    ) -> None:
        import ml_dtypes

        from dl_techniques.utils.dtype_policy import (
            accumulation_dtype,
            stability_floor,
        )

        accum = accumulation_dtype(compute_dtype)
        epsilon = stability_floor(accum, 1e-9)
        assert epsilon > 0.0
        materialized = np.array(
            epsilon, dtype=getattr(ml_dtypes, accum, None) or np.dtype(accum)
        )
        assert float(materialized) > 0.0, (
            f"the halt divisor's epsilon materializes as {float(materialized)} "
            f"in {accum}; it protects nothing there"
        )

    def test_the_hand_rolled_form_it_replaced_still_raises_under_bfloat16(
        self,
    ) -> None:
        """Pin the numpy fact, so the fix cannot be 'simplified' back.

        If a future numpy teaches `finfo` about `ml_dtypes.bfloat16`, this test
        fails and the reader learns that the constraint has lifted -- which is a
        far better outcome than the reverted one-liner sitting there looking
        harmless.
        """
        with pytest.raises(ValueError, match="not inexact"):
            np.finfo("bfloat16")


class TestFloat32IsTheControl:
    """The float32 arm is green both before and after -- this is a dtype defect."""

    def test_everything_is_finite_in_float32(self, cell_config) -> None:
        out = _run(cell_config, "float32")
        assert np.all(np.isfinite(out["reduction_weights"]))
        assert np.all(np.isfinite(out["result"]))

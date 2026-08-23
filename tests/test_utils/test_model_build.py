"""Unit contract for ``dl_techniques.utils.model_build.concretize_axes``.

``concretize_axes`` was added by plan ``plan-2026-08-23T091307-9a110062`` (D-420)
for exactly two callers -- ``Qwen3Next.build`` and ``SCUNet.build`` -- and both
exercise it only through the one shape their own model uses. Its docstring makes
three promises those two paths never touch: a CONCRETE axis is left alone, an
axis a shape is too short to have is ignored, and a negative axis index raises.
Each is asserted here, so a "simplification" of the walk fails on the clause it
broke rather than on a model test three packages away.

The substitution itself is measured where it matters, on the models:
``tests/test_models/test_the_explicit_build_materializes_the_model.py``.
"""

import pytest

from dl_techniques.utils.model_build import concretize_axes


def test_a_none_axis_becomes_the_requested_extent():
    assert concretize_axes((None, None), {1: 1}) == (None, 1)


def test_a_concrete_axis_is_left_alone():
    """The clause both callers depend on and neither one proves.

    ``Qwen3Next`` is built from ``(None, 16)`` by its own test harness and from
    ``(None, None)`` by the functional factories; only the second substitutes.
    A version that overwrote the concrete 16 with the probe value would still
    materialize 97 weights and pass every model test.
    """
    assert concretize_axes((None, 16), {1: 1}) == (None, 16)
    assert concretize_axes((None, 96, 128, 3), {1: 64, 2: 64}) == (None, 96, 128, 3)


def test_an_axis_the_shape_is_too_short_to_have_is_ignored():
    assert concretize_axes((None,), {1: 1, 2: 1}) == (None,)


def test_the_batch_axis_is_substitutable_like_any_other():
    """No axis is privileged -- axis 0 is not special-cased away."""
    assert concretize_axes((None, None), {0: 4}) == (4, None)


def test_a_dict_nest_is_walked_and_the_input_is_not_mutated():
    given = {"input_ids": (None, None), "attention_mask": (None, None)}
    got = concretize_axes(given, {1: 1})
    assert got == {"input_ids": (None, 1), "attention_mask": (None, 1)}
    assert given == {"input_ids": (None, None), "attention_mask": (None, None)}


def test_a_list_nest_is_walked():
    got = concretize_axes([(None, 16, 3), (None, None, 3)], {1: 8})
    assert got == [(None, 16, 3), (None, 8, 3)]


def test_no_replacements_is_a_structure_preserving_identity():
    assert concretize_axes((None, None, 3), {}) == (None, None, 3)


def test_a_negative_axis_raises_rather_than_wrapping():
    """``-1`` would silently mean "last axis" to Python indexing.

    ``list.__setitem__`` accepts it, so without the guard
    ``concretize_axes((None, None), {-1: 1})`` would substitute the CHANNEL
    axis while the caller meant the batch axis, and nothing downstream would
    notice.
    """
    with pytest.raises(ValueError, match="non-negative"):
        concretize_axes((None, None), {-1: 1})

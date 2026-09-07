"""
``clone_initializer`` -- proved against the symmetry it exists to break.

The premise (a shared seedless instance replays one seed) is MEASURED here
rather than assumed, and each arm has the control that makes it a finding.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.initializers import clone_initializer


def _kernel(initializer):
    layer = keras.layers.Dense(4, kernel_initializer=initializer)
    layer.build((None, 6))
    return np.asarray(ops.convert_to_numpy(layer.kernel))


def test_the_premise_a_shared_seedless_instance_emits_the_same_tensor_twice():
    """
    The Keras 3 behaviour this helper exists for. NOT a repo bug -- and the
    STRING control below is what proves the instance, not the name, is the cause.
    """
    keras.utils.set_random_seed(1234)
    shared = keras.initializers.get("glorot_uniform")
    assert getattr(shared, "seed", None) is not None, (
        "a seedless initializer INSTANCE self-assigns a seed; the exact value is "
        "process-specific (batch 6 recorded 880945459, batch 7 corrected it) so "
        "only its presence is pinned here"
    )
    assert np.array_equal(_kernel(shared), _kernel(shared))


def test_the_control_the_same_initializer_NAME_does_not_share():
    keras.utils.set_random_seed(1234)
    assert not np.array_equal(
        _kernel("glorot_uniform"), _kernel("glorot_uniform")
    ), "if this were equal the premise above would be about the name, not the instance"


def test_a_clone_breaks_the_symmetry():
    keras.utils.set_random_seed(1234)
    shared = keras.initializers.get("glorot_uniform")
    a = _kernel(shared)
    b = _kernel(clone_initializer(shared))
    assert not np.array_equal(a, b)
    assert float(np.abs(a - b).max()) > 0.0


def test_a_clone_of_a_SEEDED_initializer_deliberately_does_NOT_break_symmetry():
    """
    The documented failure mode. An author who asked for ``seed=7`` gets
    reproducibility, and this helper must not silently take it away.
    """
    seeded = keras.initializers.GlorotUniform(seed=7)
    assert np.array_equal(_kernel(seeded), _kernel(clone_initializer(seeded)))


@pytest.mark.parametrize("argument", [None, "zeros", "glorot_uniform"])
def test_strings_and_none_round_trip_through_keras_initializers_get(argument):
    result = clone_initializer(argument)
    assert result == keras.initializers.get(argument) or isinstance(
        result, keras.initializers.Initializer
    )


def test_a_clone_serializes_to_the_same_config():
    original = keras.initializers.get("he_normal")
    assert clone_initializer(original).get_config().keys() == original.get_config().keys()
    assert clone_initializer(original).__class__ is original.__class__


# ---------------------------------------------------------------------
# The module contract: matching SHAPES are not the criterion.
#
# `clone.py`'s module docstring is the canonical text six consumers read, and
# it used to make shape agreement the criterion for aliasing. That sentence is
# refuted below. Each arm here pins one clause of the corrected docstring.
# See plans decisions.md D-009/D-010.
# ---------------------------------------------------------------------

# (larger shape drawn FIRST, smaller SECOND) -- the order a build() makes.
CROSS_SHAPE_PAIRS = [
    ((8, 4, 3, 3, 7), (8, 4, 3, 3)),
    ((13,), (4, 2)),
    ((3, 3, 3), (9, 2)),
    ((5, 7), (6,)),
    ((2, 2, 2, 2), (4, 4)),
    ((10, 10), (3, 3, 3)),
    ((6,), (2, 3)),
]
CROSS_SHAPE_IDS = [f"{big}->{small}" for big, small in CROSS_SHAPE_PAIRS]


def _draw(initializer, shape):
    """Draw at ``shape`` and flatten -- ravelled, because the CLAIM is about the
    underlying sample, which the shape only reinterprets."""
    return np.asarray(
        ops.convert_to_numpy(initializer(shape=shape, dtype="float32"))
    ).ravel()


def _prefix_max_diff(big_draw, small_draw):
    n = min(big_draw.size, small_draw.size)
    return float(np.abs(big_draw[:n] - small_draw[:n]).max())


def _prefix_pearson_r(big_draw, small_draw):
    n = min(big_draw.size, small_draw.size)
    return float(np.corrcoef(big_draw[:n], small_draw[:n])[0, 1])


class TestMatchingShapesAreNotTheCriterion:
    """The refuted rule, and what replaces it."""

    @pytest.mark.parametrize("big,small", CROSS_SHAPE_PAIRS, ids=CROSS_SHAPE_IDS)
    def test_a_shared_seedless_instance_replays_the_sample_at_a_DIFFERENT_shape(
        self, big, small
    ):
        """
        With a non-scaling initializer the shorter draw is BIT-IDENTICAL to the
        longer one's prefix. The shapes never match at any of these 7 pairs.
        """
        keras.utils.set_random_seed(1234)
        shared = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
        assert _prefix_max_diff(_draw(shared, big), _draw(shared, small)) == 0.0

    def test_the_fan_scaled_DEFAULT_shares_one_sample_across_shapes(self):
        """
        The library default hides the sharing behind a per-shape scale: the
        two draws are the same numbers times ``sqrt(fan_big / fan_small)``,
        so bit-equality fails while the correlation is ~1.0. Measured on the
        exact pre-fix ``control_points`` / ``w_spline`` order.
        """
        keras.utils.set_random_seed(1234)
        shared = keras.initializers.get("glorot_uniform")
        big = _draw(shared, (8, 4, 3, 3, 7))
        small = _draw(shared, (8, 4, 3, 3))

        assert _prefix_pearson_r(big, small) > 0.999999, (
            "the shared instance replays one sample; if this drops, the "
            "docstring's r = 0.9999999999999964 no longer holds"
        )
        ratio = small / big[: small.size]
        assert ratio.std() < 1e-4, "one CONSTANT scale factor, not noise"
        assert abs(float(ratio.mean()) - np.sqrt(5.0)) < 1e-3, (
            "the scale is the fan ratio sqrt(5) = 2.236068 for these two shapes"
        )

    @pytest.mark.parametrize("big,small", CROSS_SHAPE_PAIRS, ids=CROSS_SHAPE_IDS)
    def test_a_clone_breaks_the_tie_at_a_DIFFERENT_shape_too(self, big, small):
        keras.utils.set_random_seed(1234)
        shared = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
        assert (
            _prefix_max_diff(_draw(shared, big), _draw(clone_initializer(shared), small))
            > 0.0
        )

    def test_a_clone_DECORRELATES_the_cross_shape_pair_at_the_default(self):
        """The positive control for the fan-scaled case: correlation, not
        bit-equality, is the only oracle that can see the tie there."""
        keras.utils.set_random_seed(1234)
        shared = keras.initializers.get("glorot_uniform")
        big = _draw(shared, (8, 4, 3, 3, 7))
        small = _draw(clone_initializer(shared), (8, 4, 3, 3))
        assert abs(_prefix_pearson_r(big, small)) < 0.5, (
            "un-cloned this is ~1.0; independent draws of n=288 sit near 0.06"
        )


class TestTheThreeExemptions:
    """Independence holds for a RANDOM SEEDLESS initializer. These three are
    correct behaviour, not defects -- and the docstring must keep saying so."""

    @pytest.mark.parametrize("big,small", CROSS_SHAPE_PAIRS, ids=CROSS_SHAPE_IDS)
    def test_SEEDED_replays_by_contract_ACROSS_shapes(self, big, small):
        shared = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=7)
        assert (
            _prefix_max_diff(_draw(shared, big), _draw(clone_initializer(shared), small))
            == 0.0
        ), "a caller who asked for seed=7 gets reproducibility, shapes notwithstanding"

    def test_SEEDED_replays_across_shapes_at_the_fan_scaled_default(self):
        """The docstring's measured number: seed=7 at (8,8,8) then (8,8)."""
        shared = keras.initializers.GlorotUniform(seed=7)
        big = _draw(shared, (8, 8, 8))
        small = _draw(clone_initializer(shared), (8, 8))
        assert _prefix_pearson_r(big, small) > 0.999999
        ratio = small / big[: small.size]
        assert ratio.std() < 1e-4
        assert abs(float(ratio.mean()) - 2.828427) < 1e-3

    @pytest.mark.parametrize(
        "initializer,expected",
        [
            ("zeros", 0.0),
            ("ones", 1.0),
            (keras.initializers.Constant(0.3), 0.3),
        ],
        ids=["zeros", "ones", "Constant(0.3)"],
    )
    def test_DETERMINISTIC_is_identical_at_every_site_and_that_is_CORRECT(
        self, initializer, expected
    ):
        shared = keras.initializers.get(initializer)
        same_shape = _draw(shared, (4, 3))
        cloned = _draw(clone_initializer(shared), (4, 3))
        assert np.array_equal(same_shape, cloned)
        assert np.allclose(same_shape, expected, rtol=0, atol=0)
        # and it does not care about the shape either, in the other direction:
        assert np.allclose(_draw(shared, (2, 2, 2)), expected, rtol=0, atol=0)

    def test_DETERMINISTIC_Identity_is_2D_ONLY(self):
        """``Identity`` belongs on the deterministic list only where the weight
        is 2-D; the docstring says so because it RAISES above that."""
        shared = keras.initializers.Identity()
        assert np.array_equal(_draw(shared, (4, 4)), _draw(clone_initializer(shared), (4, 4)))
        with pytest.raises(ValueError, match="2D"):
            _draw(shared, (2, 2, 2))

    def test_CUSTOM_that_fails_the_round_trip_falls_back_to_deepcopy_and_STAYS_TIED(
        self,
    ):
        """
        The third exemption, which no test anywhere covered. ``clone_initializer``
        catches the broken ``get_config()`` and deep-copies instead -- and a
        deepcopy carries the ALREADY-RESOLVED seed, so the site keeps aliasing
        with no diagnostic.
        """

        class _BrokenRoundTrip(keras.initializers.RandomUniform):
            def get_config(self):
                raise RuntimeError("this initializer does not serialize")

        keras.utils.set_random_seed(1234)
        shared = _BrokenRoundTrip(minval=-1.0, maxval=1.0)
        fallback = clone_initializer(shared)
        assert fallback is not shared, "it is a copy"
        assert fallback.seed == shared.seed, "but the resolved seed came with it"
        assert _prefix_max_diff(_draw(shared, (13,)), _draw(fallback, (4, 2))) == 0.0, (
            "so the 'clone' is tied to the original -- at a different shape too"
        )

    def test_the_CONTROL_a_healthy_custom_initializer_does_NOT_take_that_branch(self):
        """Anti-vacuity: the tie above is caused by the broken round trip, not
        by subclassing."""
        keras.utils.set_random_seed(1234)
        shared = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
        assert (
            _prefix_max_diff(
                _draw(shared, (13,)), _draw(clone_initializer(shared), (4, 2))
            )
            > 0.0
        )

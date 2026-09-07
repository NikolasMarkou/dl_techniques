"""`_SEWeights`'s two convolutions must carry explicit names.

`_SEWeights.build()` creates `conv_reduce` and `conv_restore` -- the only
`build()`-created sub-layers in `tripse_attention.py` that were constructed
without a `name=`. Keras then falls back to the process-global auto-increment
counter (`conv2d`, `conv2d_1`, ...), so the SAME weight gets a DIFFERENT path
in every instance built in a process.

MEASURED before the fix, two `TripSE4` instances in one process (one
explicitly built via `.build(shape)`, one lazily via `__call__`): equal weight
COUNTS (23 vs 23) but disagreeing paths for all six `se_logits_*` weights --

    se_logits_hw/conv2d_8/kernel   vs  se_logits_hw/conv2d_14/kernel
    se_logits_cw/conv2d_10/kernel  vs  se_logits_cw/conv2d_16/kernel
    se_logits_hc/conv2d_12/kernel  vs  se_logits_hc/conv2d_18/kernel

The drift is not a fixed offset: it depends on how many unnamed `Conv2D`
layers the process happened to create earlier, so any real training script
with other layers ahead of it gets different paths again. `.keras` full-model
save/load is unaffected (that path is graph-position-based, measured `0.0`
delta), which is exactly why the defect survived: the existing suites compare
weight counts, and a count is blind to a rename.

This is the trap the v2 guide's §8.3 names: "Parity requires every sub-layer
to carry an explicit `name=`... A parity failure is a naming problem before it
is a build problem." The fix is two `name=` arguments; this file is the guard
that keeps them.

Weight-path SETS are compared, not counts.
"""

import pytest
import numpy as np
import keras

from dl_techniques.layers.attention.tripse_attention import TripSE4, _SEWeights

# ---------------------------------------------------------------------

_SHAPE = (2, 8, 8, 4)


def _inputs():
    return np.zeros(_SHAPE, dtype='float32')


def _paths(layer):
    """Weight paths with the top-level layer name stripped.

    The OUTER layer name legitimately differs between two instances in one
    process (`trip_se4` vs `trip_se4_1`); that is Keras naming the layer, not
    the layer failing to name its children.
    """
    return sorted("/".join(w.path.split("/")[1:]) for w in layer.weights)


class TestTheSEWeightsConvsAreNamed:
    """The direct claim: both convolutions carry the names the fix gives them."""

    def test_the_bottleneck_convs_have_explicit_names(self):
        layer = _SEWeights()
        layer(_inputs())
        assert layer.conv_reduce.name == 'conv_reduce'
        assert layer.conv_restore.name == 'conv_restore'

    def test_no_weight_path_carries_an_auto_incremented_conv_name(self):
        """`conv2d`, `conv2d_1`, ... is the tell that a `name=` is missing."""
        layer = _SEWeights()
        layer(_inputs())
        offenders = [p for p in _paths(layer) if p.split('/')[0].startswith('conv2d')]
        assert offenders == [], (
            f"{offenders} carry Keras' auto-increment name, so their path depends "
            f"on how many unnamed Conv2D layers this process created earlier"
        )


class TestTheSEWeightsBuildMatchesTheLazyBuild:
    """Explicit `.build(shape)` and lazy `__call__` must agree on weight paths."""

    @pytest.mark.parametrize("cls", [_SEWeights, TripSE4], ids=['_SEWeights', 'TripSE4'])
    def test_the_weight_paths_agree(self, cls):
        explicit = cls()
        explicit.build(_SHAPE)

        lazy = cls()
        lazy(_inputs())

        assert explicit.built and lazy.built
        assert _paths(explicit) == _paths(lazy), (
            "an explicitly-built and a lazily-built instance disagree on weight "
            "paths, so a build()-created sub-layer is missing an explicit name="
        )

    @pytest.mark.parametrize("cls", [_SEWeights, TripSE4], ids=['_SEWeights', 'TripSE4'])
    def test_a_third_instance_built_after_the_first_two_still_agrees(self, cls):
        """Keras' auto-increment counter is process-global.

        A sub-layer missing `name=` drifts further with every instance built in
        the process -- the MEASURED pre-fix paths were `conv2d_8` and
        `conv2d_14` for the second and third `TripSE4` in one process -- so the
        claim is checked against an instance created third, not just second.
        """
        first = cls()
        first(_inputs())
        second = cls()
        second.build(_SHAPE)
        third = cls()
        third(_inputs())

        assert _paths(first) == _paths(second) == _paths(third)

    @pytest.mark.parametrize("cls", [_SEWeights, TripSE4], ids=['_SEWeights', 'TripSE4'])
    def test_the_weight_counts_agree_and_are_non_zero(self, cls):
        """A path-set comparison is the real guard; this pins that the class built.

        A count is deliberately NOT the guard -- pre-fix the counts already
        agreed (23 vs 23) while six paths disagreed. It is here only so a class
        that silently built nothing fails instead of passing the path
        comparison with two empty sets.
        """
        explicit = cls()
        explicit.build(_SHAPE)

        lazy = cls()
        lazy(_inputs())

        assert len(explicit.weights) == len(lazy.weights)
        assert len(explicit.weights) > 0


class TestTheNamingFixDidNotChangeTheWeights:
    """Naming a sub-layer must not move a shape or break a round trip."""

    @pytest.mark.parametrize("cls", [_SEWeights, TripSE4], ids=['_SEWeights', 'TripSE4'])
    def test_the_config_round_trip_preserves_the_weight_shapes(self, cls):
        layer = cls()
        layer(_inputs())
        restored = cls.from_config(layer.get_config())
        restored(_inputs())
        assert (
            [tuple(w.shape) for w in layer.weights]
            == [tuple(w.shape) for w in restored.weights]
        )

    def test_a_keras_round_trip_of_a_trip_se4_model_is_exact(self, tmp_path):
        """`.keras` save/load is graph-position-based, so it was never broken.

        Asserted anyway, because the fix changes weight-path STRINGS and this
        is the mechanism that could have been sensitive to them.
        """
        inputs = keras.Input(shape=_SHAPE[1:])
        model = keras.Model(inputs, TripSE4()(inputs))
        x = np.random.default_rng(0).standard_normal(_SHAPE).astype('float32')
        before = keras.ops.convert_to_numpy(model(x))

        path = tmp_path / 'trip_se4.keras'
        model.save(path)
        after = keras.ops.convert_to_numpy(keras.models.load_model(path)(x))

        assert float(np.max(np.abs(before - after))) == 0.0

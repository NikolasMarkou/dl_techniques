"""Tests for RandomSigns (uniform +/- 1 ensemble scaling-vector init).

Contract under test (read off `initializers/random_signs.py`, not assumed): the
initializer accepts ANY shape and returns a tensor whose every element is
exactly ``-1`` or ``+1``, drawn by thresholding a stock
``RandomUniform(-1, +1)`` at ``u >= 0``; ``dtype=None`` means
``keras.config.floatx()``; an explicit ``seed`` makes the draw repeatable across
instances.

Seeding follows the ORDINARY Keras 3 initializer-instance contract, not
`IdentityPlusNoise`'s. This class delegates to a stock
``keras.initializers.RandomUniform`` *instance* held on ``self._uniform``, and a
seedless Keras 3 initializer instance self-assigns a seed and REPLAYS it at every
matching shape (`initializers/clone.py` documents the measured behaviour). So one
seedless ``RandomSigns`` instance called twice at one shape returns the SAME
tensor -- reach for ``clone_initializer`` when two weights must start
differently. `IdentityPlusNoise` calls ``keras.random.normal`` directly and is
therefore stateful instead; the two are asserted separately on purpose.

The +/- 1 assertion is the load-bearing one. The whole reason the TabM paper
picks this initializer over ``'normal'`` is that every ensemble member starts at
a UNIT-magnitude perturbation, so a silent regression to the raw uniform draw
(dropping the threshold) would still produce a plausible-looking tensor in
``[-1, 1]`` and would be invisible to a shape/dtype/seed check alone.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.initializers import RandomSigns


def _np(x):
    return keras.ops.convert_to_numpy(x)


class TestConstruction:
    def test_defaults(self):
        assert RandomSigns().seed is None

    def test_seed_is_retained_verbatim(self):
        assert RandomSigns(seed=1234).seed == 1234

    def test_it_is_a_keras_initializer(self):
        assert isinstance(RandomSigns(), keras.initializers.Initializer)


class TestValues:
    def test_every_element_is_exactly_plus_or_minus_one(self):
        """The load-bearing claim: the draw is Rademacher, not uniform.

        Asserted by exact set membership, not by a magnitude tolerance -- the
        threshold either ran or it did not.
        """
        w = _np(RandomSigns(seed=0)((64, 64)))
        assert set(np.unique(w).tolist()) == {-1.0, 1.0}

    def test_no_intermediate_magnitudes_survive(self):
        """Explicit negative form of the same claim: dropping the threshold
        would leave |w| spread over [0, 1], which this catches."""
        w = _np(RandomSigns(seed=3)((128, 32)))
        np.testing.assert_array_equal(np.abs(w), np.ones_like(w))

    def test_both_signs_appear(self):
        """Not a constant initializer: a large draw contains both signs, and
        the split is near even (binomial, so a wide band is used)."""
        w = _np(RandomSigns(seed=11)((100, 100)))
        frac_positive = float((w > 0).mean())
        assert 0.45 < frac_positive < 0.55

    def test_a_scalar_shaped_draw_is_still_a_sign(self):
        w = _np(RandomSigns(seed=1)((1,)))
        assert w.shape == (1,)
        assert w[0] in (-1.0, 1.0)


class TestShapeContract:
    @pytest.mark.parametrize(
        "shape", [(8,), (4, 8), (2, 4, 8), (2, 3, 4, 5)]
    )
    def test_any_rank_is_accepted_and_the_shape_is_passed_through(self, shape):
        """Unlike `IdentityPlusNoise`, this initializer imposes NO rank or
        squareness constraint -- it is used on rank-1 and rank-2 scaling
        vectors and must not acquire one."""
        assert _np(RandomSigns(seed=0)(shape)).shape == shape


class TestDeterminism:
    def test_same_seed_gives_identical_draws(self):
        a = _np(RandomSigns(seed=5)((16, 16)))
        b = _np(RandomSigns(seed=5)((16, 16)))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_draws(self):
        a = _np(RandomSigns(seed=5)((16, 16)))
        b = _np(RandomSigns(seed=6)((16, 16)))
        assert not np.array_equal(a, b)

    def test_a_seedless_instance_replays_its_own_draw(self):
        """MEASURED, and the opposite of `IdentityPlusNoise`: this class
        delegates to a stock `RandomUniform` INSTANCE, which self-assigns a
        seed and replays it at every matching shape.

        Asserted so that swapping the stock initializer for a direct
        `keras.random.uniform` call -- which would make the draw stateful, and
        would silently de-correlate two weights an author expected to share a
        draw -- is caught.
        """
        init = RandomSigns(seed=None)
        np.testing.assert_array_equal(
            _np(init((16, 16))), _np(init((16, 16)))
        )

    def test_two_seedless_instances_draw_differently(self):
        """The flip side: replay is per-INSTANCE, so two seedless instances
        resolve independent seeds and do not collide."""
        a = _np(RandomSigns()((16, 16)))
        b = _np(RandomSigns()((16, 16)))
        assert not np.array_equal(a, b)

    def test_the_global_keras_seed_controls_a_seedless_instance(self):
        """`keras.utils.set_random_seed` is what makes a seedless instance
        reproducible across processes."""
        keras.utils.set_random_seed(1234)
        a = _np(RandomSigns()((16, 16)))
        keras.utils.set_random_seed(1234)
        b = _np(RandomSigns()((16, 16)))
        np.testing.assert_array_equal(a, b)


class TestDtype:
    def test_default_dtype_is_floatx(self):
        expected = np.dtype(keras.config.floatx())
        assert _np(RandomSigns(seed=0)((4, 4))).dtype == expected

    def test_explicit_dtype_is_honoured(self):
        w = _np(RandomSigns(seed=0)((4, 4), dtype="float64"))
        assert w.dtype == np.float64
        # And the values are still exactly +/- 1 in the wider dtype.
        assert set(np.unique(w).tolist()) == {-1.0, 1.0}

    def test_float16_is_honoured(self):
        w = _np(RandomSigns(seed=0)((4, 4), dtype="float16"))
        assert w.dtype == np.float16
        assert set(np.unique(w).tolist()) == {-1.0, 1.0}


class TestSerialization:
    def test_get_config_keys(self):
        assert RandomSigns(seed=11).get_config() == {"seed": 11}

    def test_get_config_roundtrip_reproduces_the_draw(self):
        init = RandomSigns(seed=11)
        restored = RandomSigns.from_config(init.get_config())
        assert restored.seed == 11
        np.testing.assert_array_equal(_np(init((8, 8))), _np(restored((8, 8))))

    def test_get_config_roundtrip_survives_a_none_seed(self):
        restored = RandomSigns.from_config(RandomSigns().get_config())
        assert restored.seed is None

    def test_registered_name_resolves(self, registration_contract):
        """The `custom_objects` key Keras looks a CLASS up by is its registered
        name, not its bare class name.

        `registration_contract` (tests/conftest.py) asserts both halves: the new
        key is package-qualified and owned by `dl_techniques`, and the legacy
        `Custom>RandomSigns` alias still resolves to the SAME object. The alias
        half is load-bearing here because this class MOVED packages -- it was
        registered under the `dl_techniques.layers.tabular` package before.
        """
        key = registration_contract(RandomSigns)
        assert key == "dl_techniques.initializers.random_signs>RandomSigns"

    def test_serialized_object_carries_the_registered_name(self):
        blob = keras.saving.serialize_keras_object(RandomSigns(seed=3))
        assert blob["registered_name"] == keras.saving.get_registered_name(
            RandomSigns
        )
        assert blob["config"] == {"seed": 3}

    def test_keras_serialize_deserialize(self):
        init = RandomSigns(seed=3)
        restored = keras.saving.deserialize_keras_object(
            keras.saving.serialize_keras_object(init)
        )
        assert isinstance(restored, RandomSigns)
        assert restored.seed == 3
        np.testing.assert_array_equal(_np(init((8, 8))), _np(restored((8, 8))))


class TestUseInLayer:
    def test_dense_kernel_save_load_preserves_values(self):
        """A Dense kernel initialized by RandomSigns survives a `.keras` round
        trip with its VALUES intact (not merely its shape).

        `training=False` is passed explicitly on both calls -- `training=None`
        is not inference in this repo.
        """
        inputs = keras.Input(shape=(8,))
        out = keras.layers.Dense(
            8, use_bias=False, kernel_initializer=RandomSigns(seed=17)
        )(inputs)
        model = keras.Model(inputs, out)

        x = np.random.RandomState(0).randn(4, 8).astype("float32")
        before = _np(model(x, training=False))
        kernel_before = model.layers[1].get_weights()[0]
        assert set(np.unique(kernel_before).tolist()) == {-1.0, 1.0}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

        after = _np(loaded(x, training=False))
        kernel_after = loaded.layers[1].get_weights()[0]

        np.testing.assert_allclose(kernel_before, kernel_after, rtol=0, atol=0)
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)

    def test_loaded_initializer_config_survives(self):
        inputs = keras.Input(shape=(8,))
        out = keras.layers.Dense(
            8, kernel_initializer=RandomSigns(seed=17)
        )(inputs)
        model = keras.Model(inputs, out)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        init = loaded.layers[1].kernel_initializer
        assert isinstance(init, RandomSigns)
        assert init.seed == 17


class TestTabmConsumer:
    def test_the_ensemble_scaling_resolver_still_returns_it(self):
        """`layers/tabular/_ensemble_scaling.py` imports this class after the
        move; `init_distribution='random-signs'` must still resolve to it.

        That private module is the single resolver the four TabM ensemble layers
        share, so this is the runtime edge that keeps `RandomSigns` reachable
        from `layers/` at all.
        """
        from dl_techniques.layers.tabular._ensemble_scaling import (
            _ensemble_scaling_initializer,
        )

        assert isinstance(
            _ensemble_scaling_initializer("random-signs"), RandomSigns
        )

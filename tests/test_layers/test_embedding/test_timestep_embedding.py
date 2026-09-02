"""`TimestepEmbedding`: the ladder, the concat order, and the machinery.

Two numerics here are invisible to every shape, finiteness, config and
round-trip assertion, and each gets its own value-level arm that fails
INDEPENDENTLY of the other:

1. **The frequency ladder divides by ``half``, not ``half - 1``.**
   ``TestTheFrequencyLadder`` reads the ``freqs`` weight directly against a
   closed form transcribed from ``reference/models.py:43-46``, and pins the
   discriminating entry: the LAST frequency, which is ``1 / max_period``
   exactly under the ``half - 1`` convention and strictly larger under
   upstream's. This arm does not look at the concat order at all.
2. **The basis is ``concat([cos, sin])``, cos first.** ``TestTheConcatOrder``
   evaluates at ``t = 0``, where ``cos(0) == 1`` and ``sin(0) == 0`` for EVERY
   frequency, so the assertion is ``[1...1, 0...0]`` and is blind to the ladder
   by construction. A sin-first layer reads ``[0...0, 1...1]``.

Both were proven RED by injection before this file was committed; the readings
are in the step's commit message.

A third arm cross-checks the whole layer, elementwise at ``atol=0``, against
its bit-exact sibling ``DiTXATimestepEmbedder`` in
``models/vision_language/bit_diffusion/blocks.py`` -- the deliberate,
recorded duplication of plan ``plan-2026-09-02T170923-1285ed83`` D-001. If the
two copies ever drift, that arm is what says so.
"""

import math

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding import TimestepEmbedding
from dl_techniques.layers.embedding.factory import (
    EMBEDDING_REGISTRY,
    create_embedding_layer,
)
from dl_techniques.models.vision_language.bit_diffusion.blocks import (
    DiTXATimestepEmbedder,
)


def _np(x) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(x))


def _built(hidden_size=16, frequency_embedding_size=32, **kwargs):
    layer = TimestepEmbedding(
        hidden_size=hidden_size,
        frequency_embedding_size=frequency_embedding_size,
        **kwargs,
    )
    layer.build((None,))
    return layer


def _basis(layer, t: np.ndarray) -> np.ndarray:
    return _np(layer.timestep_embedding(keras.ops.convert_to_tensor(t)))


class _Parent(keras.layers.Layer):
    """Minimal parent whose ``call()`` is the only path that builds the child."""

    def __init__(self, child: keras.layers.Layer, **kwargs) -> None:
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs):
        return self.child(inputs)


class TestTheFrequencyLadder:
    """The denominator is ``half``. Nothing about the concat order is read here."""

    @pytest.mark.parametrize("freq_size", [8, 32, 256])
    @pytest.mark.parametrize("max_period", [10000.0, 1000.0])
    def test_the_ladder_matches_the_closed_form(self, freq_size, max_period):
        # Closed form transcribed from `reference/models.py:43-46`, NOT obtained
        # by calling the module under test.
        half = freq_size // 2
        expected = np.exp(
            -math.log(max_period) * np.arange(half, dtype="float32") / half
        )
        layer = _built(
            frequency_embedding_size=freq_size, max_period=max_period
        )
        np.testing.assert_allclose(
            _np(layer.freqs), expected, rtol=0.0, atol=0.0
        )

    @pytest.mark.parametrize("freq_size", [8, 32, 256])
    def test_the_last_frequency_never_reaches_one_over_max_period(
        self, freq_size
    ):
        # THE discriminating entry. Under the `half - 1` convention the last
        # frequency lands exactly on 1 / max_period; under upstream's `half` it
        # is strictly larger and never gets there. Everything else about the two
        # ladders -- shape, dtype, monotonicity, first entry, finiteness -- is
        # identical, so this is the one place the two can be told apart by a
        # scalar.
        half = freq_size // 2
        max_period = 10000.0
        layer = _built(
            frequency_embedding_size=freq_size, max_period=max_period
        )
        freqs = _np(layer.freqs)

        floor = np.float32(1.0 / max_period)
        assert freqs[-1] > floor, (
            f"freqs[-1] == {freqs[-1]!r} reached the 1/max_period floor "
            f"{floor!r}: the ladder was divided by half - 1, not half."
        )
        np.testing.assert_allclose(
            freqs[-1],
            np.exp(-math.log(max_period) * (half - 1) / half).astype(
                "float32"
            ),
            rtol=0.0,
            atol=1e-9,
        )

    def test_the_first_frequency_is_one(self):
        # True under BOTH conventions -- recorded here as the anti-vacuity
        # control for the arm above: `freqs[0]` cannot discriminate, `freqs[-1]`
        # can.
        assert _np(_built(frequency_embedding_size=32).freqs)[0] == np.float32(
            1.0
        )


class TestTheConcatOrder:
    """The basis is cos first. Evaluated at ``t = 0``, where the ladder is mute."""

    @pytest.mark.parametrize("freq_size", [8, 32, 256])
    def test_at_t_zero_the_first_half_is_ones_and_the_second_half_is_zeros(
        self, freq_size
    ):
        half = freq_size // 2
        layer = _built(frequency_embedding_size=freq_size)
        basis = _basis(layer, np.array([0.0], dtype="float32"))[0]

        assert basis.shape == (freq_size,)
        # cos(0 * anything) == 1 and sin(0 * anything) == 0 for EVERY frequency,
        # so this assertion cannot be satisfied or defeated by the ladder.
        np.testing.assert_array_equal(basis[:half], np.ones(half, "float32"))
        np.testing.assert_array_equal(basis[half:], np.zeros(half, "float32"))

    def test_the_two_halves_are_cos_then_sin_at_a_general_timestep(self):
        # A second, ladder-dependent form of the same claim: whatever the
        # frequencies are, the first half must be their cosines.
        freq_size = 16
        layer = _built(frequency_embedding_size=freq_size)
        t = np.array([0.3, 7.0, 999.0], dtype="float32")
        args = t[:, None] * _np(layer.freqs)[None, :]
        basis = _basis(layer, t)
        np.testing.assert_allclose(
            basis[:, : freq_size // 2], np.cos(args), rtol=0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            basis[:, freq_size // 2 :], np.sin(args), rtol=0.0, atol=1e-6
        )

    def test_an_odd_width_pads_one_trailing_zero_column(self):
        layer = _built(frequency_embedding_size=7)
        basis = _basis(layer, np.array([1.0, 2.0], dtype="float32"))
        assert basis.shape == (2, 7)
        # half == 3, so 6 real columns plus one ZERO pad -- upstream pads, it
        # does not drop a frequency.
        np.testing.assert_array_equal(basis[:, 6], np.zeros(2, "float32"))


class TestItAgreesWithTheBitDiffusionSibling:
    """The recorded duplication (D-001) is pinned elementwise, not by prose."""

    @pytest.mark.parametrize(
        "hidden_size,freq_size", [(16, 32), (64, 256), (8, 7)]
    )
    def test_the_basis_is_identical_at_atol_zero(self, hidden_size, freq_size):
        t = np.array([0.0, 1.0, 17.0, 999.0], dtype="float32")
        mine = _built(hidden_size, freq_size)
        sibling = DiTXATimestepEmbedder(
            hidden_size=hidden_size, frequency_embedding_size=freq_size
        )
        sibling.build((None,))
        np.testing.assert_array_equal(_basis(mine, t), _basis(sibling, t))

    def test_the_whole_layer_is_identical_under_the_same_seed(self):
        t = np.array([0.0, 250.0, 999.0], dtype="float32")
        keras.utils.set_random_seed(0)
        mine = TimestepEmbedding(hidden_size=32, frequency_embedding_size=64)
        a = _np(mine(keras.ops.convert_to_tensor(t)))
        keras.utils.set_random_seed(0)
        sibling = DiTXATimestepEmbedder(
            hidden_size=32, frequency_embedding_size=64
        )
        b = _np(sibling(keras.ops.convert_to_tensor(t)))
        np.testing.assert_array_equal(a, b)


class TestTheTableIsAWeightAndSurvivesTheStatelessBuild:

    def test_freqs_is_a_non_trainable_weight(self):
        layer = _built()
        # Membership by IDENTITY: `variable in [variables]` runs an elementwise
        # `==` on a Keras variable and raises on the ambiguous truth value.
        assert any(w is layer.freqs for w in layer.non_trainable_weights)
        assert not any(w is layer.freqs for w in layer.trainable_weights)
        assert layer.freqs.trainable is False

    def test_the_ladder_holds_its_values_when_built_through_a_parent(self):
        # Keras 3 runs the symbolic build inside a StatelessScope whenever a
        # sublayer is first reached from a PARENT's call(), and that scope
        # RECORDS and then DISCARDS an `.assign()`. A test that calls
        # `.build(...)` directly is exactly the test that misses it.
        child = TimestepEmbedding(hidden_size=16, frequency_embedding_size=32)
        parent = _Parent(child)
        parent(keras.Input(shape=(), dtype="float32"))
        freqs = _np(child.freqs)
        assert freqs[0] == np.float32(1.0)  # an all-zero table reads 0 here
        assert np.all(freqs > 0.0)

    def test_the_ladder_does_not_move_under_an_optimizer_step(self):
        layer = TimestepEmbedding(hidden_size=8, frequency_embedding_size=16)
        model = keras.Sequential([keras.Input(shape=()), layer])
        model.compile(optimizer=keras.optimizers.SGD(1.0), loss="mse")
        before = _np(layer.freqs).copy()
        model.fit(
            np.array([0.0, 1.0, 2.0, 3.0], dtype="float32"),
            np.ones((4, 8), dtype="float32"),
            epochs=1,
            batch_size=4,
            verbose=0,
        )
        np.testing.assert_array_equal(_np(layer.freqs), before)


class TestTheMechanics:

    def test_forward_shape_and_finiteness(self):
        layer = TimestepEmbedding(hidden_size=16, frequency_embedding_size=32)
        out = _np(layer(keras.ops.convert_to_tensor(
            np.arange(5, dtype="float32")
        )))
        assert out.shape == (5, 16)
        assert np.all(np.isfinite(out))

    def test_rank_1_and_rank_2_inputs_agree(self):
        layer = TimestepEmbedding(hidden_size=12, frequency_embedding_size=16)
        t = np.array([0.0, 5.0, 900.0], dtype="float32")
        flat = _np(layer(keras.ops.convert_to_tensor(t)))
        column = _np(layer(keras.ops.convert_to_tensor(t[:, None])))
        np.testing.assert_array_equal(flat, column)

    def test_integer_timesteps_are_accepted(self):
        layer = TimestepEmbedding(hidden_size=12, frequency_embedding_size=16)
        ints = keras.ops.convert_to_tensor(np.array([0, 7, 999], dtype="int32"))
        floats = keras.ops.convert_to_tensor(
            np.array([0.0, 7.0, 999.0], dtype="float32")
        )
        np.testing.assert_array_equal(_np(layer(ints)), _np(layer(floats)))

    @pytest.mark.parametrize("shape", [(None,), (4,), (None, 1), (4, 1)])
    def test_compute_output_shape(self, shape):
        layer = TimestepEmbedding(hidden_size=13)
        assert layer.compute_output_shape(shape) == (shape[0], 13)

    def test_build_materializes_exactly_what_call_runs(self):
        layer = TimestepEmbedding(hidden_size=16, frequency_embedding_size=32)
        layer.build((None,))
        built_paths = {w.path for w in layer.weights}
        # Anti-vacuity: the set is non-empty and names all three sites.
        assert len(built_paths) == 5  # freqs + 2 kernels + 2 biases
        layer(keras.ops.convert_to_tensor(np.zeros((3,), "float32")))
        assert {w.path for w in layer.weights} == built_paths

    def test_two_layers_do_not_share_an_initializer_instance(self):
        layer = TimestepEmbedding(hidden_size=8, frequency_embedding_size=16)
        assert layer.mlp_in.kernel_initializer is not (
            layer.mlp_out.kernel_initializer
        )

    def test_the_dense_kernels_are_random_normal_at_the_declared_stddev(self):
        layer = TimestepEmbedding(
            hidden_size=8, frequency_embedding_size=16, kernel_stddev=0.05
        )
        for dense in (layer.mlp_in, layer.mlp_out):
            init = dense.kernel_initializer
            assert isinstance(init, keras.initializers.RandomNormal)
            assert init.stddev == 0.05

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(hidden_size=0),
            dict(hidden_size=-4),
            dict(hidden_size=8, frequency_embedding_size=1),
            dict(hidden_size=8, frequency_embedding_size=0),
            dict(hidden_size=8, max_period=1.0),
            dict(hidden_size=8, max_period=0.5),
            dict(hidden_size=8, kernel_stddev=0.0),
            dict(hidden_size=8, kernel_stddev=-0.02),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            TimestepEmbedding(**kwargs)


class TestSerialization:

    def test_config_round_trip_preserves_every_knob(self):
        layer = TimestepEmbedding(
            hidden_size=24,
            frequency_embedding_size=64,
            max_period=5000.0,
            kernel_stddev=0.03,
        )
        restored = TimestepEmbedding.from_config(layer.get_config())
        assert restored.hidden_size == 24
        assert restored.frequency_embedding_size == 64
        assert restored.max_period == 5000.0
        assert restored.kernel_stddev == 0.03
        assert restored.half == 32

    def test_get_config_names_every_constructor_argument(self):
        import inspect

        declared = {
            name
            for name, p in inspect.signature(
                TimestepEmbedding.__init__
            ).parameters.items()
            if p.kind is not inspect.Parameter.VAR_KEYWORD and name != "self"
        }
        config = TimestepEmbedding(hidden_size=8).get_config()
        assert declared <= set(config), declared - set(config)

    def test_keras_round_trip_preserves_values(self, tmp_path):
        inputs = keras.Input(shape=(), dtype="float32")
        outputs = TimestepEmbedding(
            hidden_size=16, frequency_embedding_size=32, name="t_emb"
        )(inputs)
        model = keras.Model(inputs, outputs)
        t = keras.ops.convert_to_tensor(
            np.array([0.0, 13.0, 999.0], dtype="float32")
        )
        before = _np(model(t, training=False))

        path = tmp_path / "temb.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)

        # Weight VALUES, at atol=0.0, BEFORE the loaded model's first call.
        for w_old, w_new in zip(model.weights, reloaded.weights):
            np.testing.assert_array_equal(_np(w_old), _np(w_new))

        after = _np(reloaded(t, training=False))
        np.testing.assert_allclose(before, after, rtol=0.0, atol=0.0)
        assert reloaded.get_layer("t_emb").frequency_embedding_size == 32


class TestPrecisionArms:

    @pytest.mark.parametrize(
        "dtype_policy", ["float32", "mixed_float16", "float64"], indirect=True
    )
    def test_it_runs_and_stays_finite_under_each_policy(self, dtype_policy):
        layer = TimestepEmbedding(hidden_size=16, frequency_embedding_size=32)
        out = _np(layer(keras.ops.convert_to_tensor(
            np.array([0.0, 500.0, 999.0], dtype="float32")
        )))
        assert out.shape == (3, 16)
        assert np.all(np.isfinite(out))

    def test_the_ladder_stays_float32_under_mixed_float16(
        self, mixed_float16_policy
    ):
        # The ladder is a CONSTANT table; narrowing it to float16 would quantize
        # the smallest frequencies toward each other with no shape symptom.
        layer = _built()
        assert layer.freqs.dtype == "float32"


class TestTheFactoryRegistration:

    def test_the_key_is_registered(self):
        assert "timestep" in EMBEDDING_REGISTRY
        entry = EMBEDDING_REGISTRY["timestep"]
        assert entry["class"] is TimestepEmbedding
        assert entry["required_params"] == ["hidden_size"]
        assert set(entry["optional_params"]) == {
            "frequency_embedding_size",
            "max_period",
            "kernel_stddev",
        }

    def test_the_factory_builds_it(self):
        layer = create_embedding_layer(
            "timestep", hidden_size=32, frequency_embedding_size=64
        )
        assert isinstance(layer, TimestepEmbedding)
        assert layer.frequency_embedding_size == 64

    def test_the_factory_defaults_match_the_constructor_defaults(self):
        from_factory = create_embedding_layer("timestep", hidden_size=32)
        direct = TimestepEmbedding(hidden_size=32)
        assert (
            from_factory.frequency_embedding_size
            == direct.frequency_embedding_size
        )
        assert from_factory.max_period == direct.max_period
        assert from_factory.kernel_stddev == direct.kernel_stddev

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(hidden_size=0),
            dict(hidden_size=8, frequency_embedding_size=1),
            dict(hidden_size=8, max_period=1.0),
            dict(hidden_size=8, kernel_stddev=-1.0),
        ],
    )
    def test_the_factory_validates(self, kwargs):
        with pytest.raises(ValueError):
            create_embedding_layer("timestep", **kwargs)

    def test_the_factory_rejects_an_undeclared_keyword(self):
        # `dim` is `scalar_sinusoidal`'s spelling. Silently dropping it would
        # leave the caller at the 256 default while they believe they asked for
        # something else.
        with pytest.raises(ValueError, match="unsupported parameter"):
            create_embedding_layer("timestep", hidden_size=8, dim=64)

    def test_it_is_re_exported_from_the_package(self):
        import dl_techniques.layers.embedding as pkg

        assert "TimestepEmbedding" in pkg.__all__
        assert pkg.TimestepEmbedding is TimestepEmbedding

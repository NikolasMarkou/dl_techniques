"""Tests for ConvStem: the contract a caller can state through its arguments.

The stem is a configurable Conv -> [norm] -> [activation] (-> second conv ...)
-> [pool] composer. This file pins what the arguments alone decide: weights
that exist after ``build()``, invalid arguments, shapes and pool geometry, an
absent norm, the norm epsilon, bias initializer / regularizer, every accepted
activation form, ``get_config``, gradient flow, a bit-exact ``.keras`` round
trip and registration. The traced forward (``jit_compile=True`` and an unknown
batch and spatial size) is compared against the eager one.

Three claims have a file of their own, because each is a different kind of
guard:

* ``test_the_stem_follows_its_own_dtype_policy.py`` -- a per-layer ``dtype=``
  reaches every sub-layer, for every normalization type the factory can build;
* ``test_a_stem_activation_survives_the_config_round_trip.py`` -- the activation
  comes back from ``get_config`` as what the caller wrote;
* ``test_the_two_convs_of_a_stem_draw_independently.py`` -- the second kernel
  is not the first kernel's draw.

The helpers they share are in ``stem_instruments.py``.
"""

import inspect
import json

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.activations.exact_activations import exact_gelu
from dl_techniques.layers.conv_blocks.stem import ConvStem
from tests.norm_epsilon_oracle import collect_norm_epsilons
from tests.numerics import reassociation_atol

from . import stem_instruments
from .stem_instruments import (
    B, EPS32, random_input, randomize_weights, roundtrip_exact, to_numpy)

# The module scopes TF32 off so the exact-equality comparisons below are made
# in true float32 on a TF32-capable GPU as well as on CPU.
pytestmark = pytest.mark.usefixtures("tf32_disabled")

# The ``rng`` fixture is owned by ``stem_instruments``; binding it here is what
# makes pytest find it in this module.
rng = stem_instruments.rng


# ---------------------------------------------------------------------
# Weights exist after build() alone
# ---------------------------------------------------------------------

class TestConvStemWeightsBeforeFirstCall:
    def test_build_alone_creates_every_weight_with_exact_values(self, rng):
        layer = ConvStem(filters=6, kernel_size=3, num_convs=2,
                         kernel_initializer="zeros", name="preweights")
        layer.build((B, 8, 8, 4))
        assert layer.built
        for sub in layer._flatten_layers(include_self=False, recursive=False):
            assert sub.built, f"{sub.name} was not built by ConvStem.build()"
        before = [w.path for w in layer.weights]
        # 2 convs x (kernel, bias) + 2 BN x (gamma, beta, moving mean, moving var)
        assert len(before) == 12

        np.testing.assert_allclose(to_numpy(layer.conv1.kernel), 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(to_numpy(layer.conv2.kernel), 0.0, rtol=0.0, atol=0.0)
        assert layer.conv1.kernel.shape == (3, 3, 4, 6)
        assert layer.conv2.kernel.shape == (3, 3, 6, 6)
        for conv in (layer.conv1, layer.conv2):
            np.testing.assert_allclose(to_numpy(conv.bias), 0.0, rtol=0.0, atol=0.0)
        for norm in (layer.norm1, layer.norm2):
            np.testing.assert_allclose(to_numpy(norm.gamma), 1.0, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(to_numpy(norm.beta), 0.0, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(to_numpy(norm.moving_mean), 0.0, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(to_numpy(norm.moving_variance), 1.0, rtol=0.0, atol=0.0)

        layer(random_input(rng, (B, 8, 8, 4)))
        assert [w.path for w in layer.weights] == before, (
            "the first call created weights that build() did not")

    def test_sublayers_exist_before_build(self):
        layer = ConvStem(filters=4, num_convs=2, use_pooling=True)
        assert not layer.built
        assert layer.conv1 is not None and layer.conv2 is not None
        assert layer.norm1 is not None and layer.pool is not None
        assert not layer.conv1.built


# ---------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------

class TestConvStemInvalidArgs:
    @pytest.mark.parametrize("ctor, message", [
        (lambda: ConvStem(filters=0), "filters"),
        (lambda: ConvStem(filters=4, num_convs=3), "num_convs"),
        (lambda: ConvStem(filters=4, padding="reflect"), "padding must be"),
        (lambda: ConvStem(filters=4, pool_type="min"), "pool_type must be"),
        (lambda: ConvStem(filters=4, use_pooling=True, pool_padding="causal"),
         "pool_padding must be"),
        (lambda: ConvStem(filters=4, num_convs=2, conv2_filters=0), "conv2_filters"),
        (lambda: ConvStem(filters=4, normalization_type="not_a_norm"), "not_a_norm"),
        (lambda: ConvStem(filters=4, activation_type="not_an_activation"),
         "not_an_activation"),
        (lambda: ConvStem(filters=4, normalization_type=None,
                          normalization_kwargs={"epsilon": 1e-3}),
         "normalization_kwargs"),
        (lambda: ConvStem(filters=4, normalization_kwargs={"not_a_kwarg": 1}),
         "not_a_kwarg"),
    ], ids=[
        "filters_zero", "num_convs", "padding", "pool_type", "pool_padding",
        "conv2_filters", "unknown_norm", "unknown_activation",
        "norm_kwargs_without_norm", "unknown_norm_kwarg",
    ])
    def test_invalid_args_raise(self, ctor, message):
        """The message is matched: ``Layer.__init__`` also raises ``ValueError``
        for an unrecognised keyword, so a bare ``ValueError`` would pass against
        a stem that has never heard of the argument."""
        with pytest.raises(ValueError, match=message):
            ctor()

    @pytest.mark.parametrize("activation", [
        "sigmoid", "linear", None, exact_gelu, keras.layers.LeakyReLU(0.1),
    ], ids=["keras_name", "linear", "none", "callable", "layer"])
    def test_activation_kwargs_are_refused_off_the_registry(self, activation):
        """``resolve_activation_layer`` drops kwargs silently off the registry."""
        with pytest.raises(ValueError, match="activation_kwargs"):
            ConvStem(filters=4, activation_type=activation,
                     activation_kwargs={"negative_slope": 0.2})

    def test_an_undeclared_registry_kwarg_is_refused(self):
        # The match names the factory and the registry key: a bare ValueError
        # also passes for any other refusal ``Layer.__init__`` raises.
        with pytest.raises(
                ValueError, match=r"create_activation_layer\('relu'\).*not_a_kwarg"):
            ConvStem(filters=4, activation_type="relu",
                     activation_kwargs={"not_a_kwarg": 1})


# ---------------------------------------------------------------------
# Shapes and pool geometry
# ---------------------------------------------------------------------

class TestConvStemShapes:
    def test_single_conv_forward_and_shape(self, rng):
        layer = ConvStem(filters=8, kernel_size=3, strides=2, name="single")
        out = layer(random_input(rng))
        assert tuple(out.shape) == (B, 8, 8, 8)
        assert layer.compute_output_shape((B, 16, 16, 3)) == (B, 8, 8, 8)
        assert np.isfinite(to_numpy(out)).all()

    def test_two_convs_apply_the_stride_to_each_conv(self, rng):
        layer = ConvStem(filters=8, num_convs=2, conv2_filters=12, strides=2,
                         name="two")
        out = layer(random_input(rng))
        assert tuple(out.shape) == (B, 4, 4, 12)
        assert layer.conv1.filters == 8 and layer.conv2.filters == 12

    def test_conv2_filters_defaults_to_filters(self):
        layer = ConvStem(filters=8, num_convs=2)
        assert layer.conv2.filters == 8

    def test_sublayers_carry_the_documented_names(self):
        layer = ConvStem(filters=4, num_convs=2, use_pooling=True, name="stem")
        layer.build((B, 8, 8, 3))
        assert layer.conv1.name == "stem_conv1"
        assert layer.norm1.name == "stem_norm1"
        assert layer.act1.name == "stem_act1"
        assert layer.conv2.name == "stem_conv2"
        assert layer.norm2.name == "stem_norm2"
        assert layer.act2.name == "stem_act2"
        assert layer.pool.name == "stem_pool"

    def test_a_stems_attributes_carry_the_names_of_its_layers(self):
        # Every sub-layer attribute is named after itself under the stem's own
        # name (``stem.act1`` is ``{name}_act1``), so a weight path and a Python
        # attribute spell the same word. Two names are used so a hard-coded
        # prefix cannot pass.
        for stem_name in ("stem", "patch_embed"):
            layer = ConvStem(filters=4, num_convs=2, use_pooling=True, name=stem_name)
            roles = ("conv1", "norm1", "act1", "conv2", "norm2", "act2", "pool")
            for role in roles:
                sublayer = getattr(layer, role)
                assert sublayer.name == f"{stem_name}_{role}", (stem_name, role)
            held = {
                attr for attr, value in vars(layer).items()
                if isinstance(value, keras.layers.Layer)
            }
            assert held == set(roles), sorted(held ^ set(roles))

    def test_final_activation_false_builds_no_second_activation(self, rng):
        on = ConvStem(filters=6, num_convs=2, final_activation=True)
        off = ConvStem(filters=6, num_convs=2, final_activation=False)
        assert on.act2 is not None
        assert off.act2 is None
        assert off.act1 is not None
        x = random_input(rng, (B, 8, 8, 3))
        assert float(np.min(to_numpy(on(x)))) >= 0.0
        # No trailing ReLU: BN output of a random conv has negative entries.
        assert float(np.min(to_numpy(off(x)))) < 0.0

    def test_resnet_pool_geometry_is_stride_two_same(self, rng):
        """7x7/2 same conv then 3x3/2 same max-pool: 224 -> 112 -> 56."""
        layer = ConvStem(
            filters=16, kernel_size=7, strides=2, padding="same",
            use_pooling=True, pool_size=3, pool_strides=2, pool_padding="same",
        )
        assert layer.pool.strides == (2, 2)
        assert layer.pool.padding == "same"
        assert layer.conv1.compute_output_shape((1, 224, 224, 3)) == (1, 112, 112, 16)
        assert layer.compute_output_shape((1, 224, 224, 3)) == (1, 56, 56, 16)
        out = layer(rng.standard_normal((1, 224, 224, 3)).astype("float32"))
        assert tuple(out.shape) == (1, 56, 56, 16)

    def test_squeezenet_pool_geometry_is_stride_two_valid(self, rng):
        """3x3/2 valid conv then 3x3/2 valid max-pool: 224 -> 111 -> 55."""
        layer = ConvStem(
            filters=16, kernel_size=3, strides=2, padding="valid",
            use_pooling=True, pool_size=3, pool_strides=2, pool_padding="valid",
        )
        assert layer.compute_output_shape((1, 224, 224, 3)) == (1, 55, 55, 16)
        out = layer(rng.standard_normal((1, 224, 224, 3)).astype("float32"))
        assert tuple(out.shape) == (1, 55, 55, 16)

    def test_pool_defaults_are_keras_defaults(self):
        """``pool_strides=None`` keeps stride == pool size, ``pool_padding='valid'``."""
        layer = ConvStem(filters=4, use_pooling=True, pool_size=3)
        assert layer.pool.strides == (3, 3)
        assert layer.pool.padding == "valid"
        # 112 -> floor((112 - 3) / 3) + 1 = 37
        assert layer.compute_output_shape((1, 112, 112, 3)) == (1, 37, 37, 4)

    def test_average_pool_takes_the_same_geometry(self):
        layer = ConvStem(
            filters=4, use_pooling=True, pool_type="avg", pool_size=3,
            pool_strides=2, pool_padding="same",
        )
        assert isinstance(layer.pool, keras.layers.AveragePooling2D)
        assert layer.pool.strides == (2, 2) and layer.pool.padding == "same"

    @pytest.mark.parametrize("kwargs", [
        dict(kernel_size=7, strides=2, padding="same", use_pooling=True,
             pool_size=3, pool_strides=2, pool_padding="same"),
        dict(kernel_size=3, strides=2, padding="valid", use_pooling=True,
             pool_size=3, pool_strides=2, pool_padding="valid"),
        dict(kernel_size=3, strides=1, padding="same", num_convs=2,
             use_pooling=True, pool_size=2),
        dict(kernel_size=(3, 5), strides=(2, 1), padding="valid", num_convs=2),
    ], ids=["resnet", "squeezenet", "two_conv_pool", "asymmetric"])
    def test_compute_output_shape_matches_the_real_forward(self, kwargs, rng):
        layer = ConvStem(filters=6, **kwargs)
        reported = layer.compute_output_shape((B, 32, 32, 3))
        # Asking for a shape must not build the layer: Keras calls it on an
        # unbuilt layer, and a build here would create weights early.
        assert not layer.built
        out = layer(random_input(rng, (B, 32, 32, 3)))
        assert reported == tuple(out.shape)

    @pytest.mark.parametrize("input_shape", [
        (None, None, None, 3), (None, 224, 224, 3), (None, None, 64, 3)])
    def test_unknown_spatial_dims_pass_through_compute_output_shape(self, input_shape):
        layer = ConvStem(
            filters=16, kernel_size=7, strides=2, padding="same",
            use_pooling=True, pool_size=3, pool_strides=2, pool_padding="same",
        )
        assert not layer.built
        got = layer.compute_output_shape(input_shape)
        assert not layer.built
        expected = tuple(
            None if d is None else (d + 3) // 4 for d in input_shape[1:3])
        assert got == (None,) + expected + (16,)


# ---------------------------------------------------------------------
# normalization_type=None
# ---------------------------------------------------------------------

class TestConvStemNoNormalization:
    def test_no_norm_layer_and_no_norm_weights(self):
        layer = ConvStem(filters=6, num_convs=2, normalization_type=None)
        layer.build((B, 8, 8, 3))
        assert layer.norm1 is None and layer.norm2 is None
        sublayers = list(layer._flatten_layers(include_self=False, recursive=True))
        assert not any("norm" in type(s).__name__.lower() for s in sublayers)
        assert collect_norm_epsilons([layer]) == []
        # Two kernels and two biases, nothing else.
        assert len(layer.weights) == 4
        assert all(("kernel" in w.path or "bias" in w.path) for w in layer.weights)

    def test_forward_is_conv_then_activation(self, rng):
        layer = ConvStem(filters=6, normalization_type=None, activation_type="relu")
        x = random_input(rng, (B, 8, 8, 3))
        out = to_numpy(layer(x))
        expected = np.maximum(to_numpy(layer.conv1(x)), 0.0)
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)

    def test_two_conv_forward_without_norm(self, rng):
        layer = ConvStem(filters=6, num_convs=2, normalization_type=None,
                         activation_type="relu")
        x = random_input(rng, (B, 8, 8, 3))
        out = to_numpy(layer(x))
        mid = np.maximum(to_numpy(layer.conv1(x)), 0.0)
        expected = np.maximum(to_numpy(layer.conv2(mid)), 0.0)
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)

    def test_the_config_round_trips_a_missing_norm(self):
        layer = ConvStem(filters=6, normalization_type=None)
        config = json.loads(json.dumps(layer.get_config()))
        assert config["normalization_type"] is None
        rebuilt = ConvStem.from_config(config)
        assert rebuilt.norm1 is None


# ---------------------------------------------------------------------
# Normalization epsilon and the training flag
# ---------------------------------------------------------------------

class TestConvStemNormalization:
    def test_default_epsilon_is_the_factorys(self):
        layer = ConvStem(filters=4, num_convs=2)
        assert [e for _, _, e in collect_norm_epsilons([layer])] == [1e-6, 1e-6]

    @pytest.mark.parametrize("num_convs", [1, 2])
    def test_epsilon_passes_through_and_tracks_the_knob(self, num_convs):
        seen = {}
        for epsilon in (1e-3, 1e-5):
            layer = ConvStem(
                filters=4, num_convs=num_convs,
                normalization_kwargs={"epsilon": epsilon, "momentum": 0.9})
            rows = collect_norm_epsilons([layer])
            assert len(rows) == num_convs
            assert {e for _, _, e in rows} == {epsilon}
            assert all(n.momentum == 0.9 for n in (layer.norm1, layer.norm2) if n)
            seen[epsilon] = rows
        assert seen[1e-3] != seen[1e-5]

    def test_training_reaches_every_batch_norm(self, rng):
        layer = ConvStem(filters=4, num_convs=2,
                         normalization_kwargs={"momentum": 0.5})
        x = random_input(rng, (B, 8, 8, 3))
        layer(x, training=False)
        before = [to_numpy(n.moving_mean).copy() for n in (layer.norm1, layer.norm2)]
        layer(x, training=False)
        for norm, old in zip((layer.norm1, layer.norm2), before):
            np.testing.assert_allclose(to_numpy(norm.moving_mean), old, rtol=0.0, atol=0.0)
        layer(x, training=True)
        for norm, old in zip((layer.norm1, layer.norm2), before):
            assert np.max(np.abs(to_numpy(norm.moving_mean) - old)) > 1e-4, (
                f"{norm.name} statistics did not move under training=True")


# ---------------------------------------------------------------------
# Bias initializer / regularizer
# ---------------------------------------------------------------------

class TestConvStemBias:
    def test_default_bias_is_zeros(self):
        layer = ConvStem(filters=4, num_convs=2)
        layer.build((B, 8, 8, 3))
        for conv in (layer.conv1, layer.conv2):
            np.testing.assert_allclose(to_numpy(conv.bias), 0.0, rtol=0.0, atol=0.0)

    def test_bias_initializer_reaches_both_convs(self):
        layer = ConvStem(filters=4, num_convs=2, bias_initializer="ones")
        layer.build((B, 8, 8, 3))
        for conv in (layer.conv1, layer.conv2):
            np.testing.assert_allclose(to_numpy(conv.bias), 1.0, rtol=0.0, atol=0.0)

    def test_bias_regularizer_reaches_both_convs_and_is_summed_into_losses(self, rng):
        layer = ConvStem(filters=4, num_convs=2, bias_initializer="ones",
                         bias_regularizer=keras.regularizers.L2(0.5))
        layer(random_input(rng, (B, 8, 8, 3)))
        for conv in (layer.conv1, layer.conv2):
            assert isinstance(conv.bias.regularizer, keras.regularizers.L2)
        # L2(0.5) of a length-4 ones vector is 0.5 * 4 = 2.0, once per conv.
        total = float(sum(to_numpy(loss) for loss in layer.losses))
        np.testing.assert_allclose(total, 4.0, rtol=0.0, atol=1e-6)

    def test_kernel_regularizer_reaches_both_convs(self):
        layer = ConvStem(filters=4, num_convs=2,
                         kernel_regularizer=keras.regularizers.L2(1e-3))
        layer.build((B, 8, 8, 3))
        for conv in (layer.conv1, layer.conv2):
            assert isinstance(conv.kernel.regularizer, keras.regularizers.L2)

    def test_bias_settings_are_serialized_and_rebuilt(self):
        layer = ConvStem(
            filters=4, num_convs=2, bias_initializer="ones",
            bias_regularizer=keras.regularizers.L2(0.25),
            kernel_regularizer=keras.regularizers.L1(0.125))
        config = json.loads(json.dumps(layer.get_config()))
        assert config["bias_regularizer"]["class_name"] == "L2"
        rebuilt = ConvStem.from_config(config)
        rebuilt.build((B, 8, 8, 3))
        np.testing.assert_allclose(to_numpy(rebuilt.conv2.bias), 1.0, rtol=0.0, atol=0.0)
        assert float(rebuilt.conv1.bias.regularizer.l2) == 0.25
        assert float(rebuilt.conv1.kernel.regularizer.l1) == 0.125

    def test_use_bias_false_builds_no_bias(self):
        layer = ConvStem(filters=4, num_convs=2, use_bias=False,
                         bias_initializer="ones")
        layer.build((B, 8, 8, 3))
        assert layer.conv1.bias is None and layer.conv2.bias is None


# ---------------------------------------------------------------------
# Activation forms
# ---------------------------------------------------------------------

def _leaky(x, slope):
    return np.where(x >= 0.0, x, slope * x)


class TestConvStemActivationForms:
    """Every accepted spelling builds, computes the right function and serializes."""

    @staticmethod
    def _run(activation, rng, **kwargs):
        layer = ConvStem(filters=6, normalization_type=None,
                         activation_type=activation, **kwargs)
        x = random_input(rng, (B, 8, 8, 3))
        layer.build(x.shape)  # a sub-layer called before the stem is built cannot be built again
        pre = to_numpy(layer.conv1(x))
        return layer, pre, to_numpy(layer(x))

    def test_registry_key_with_kwargs(self, rng):
        layer, pre, out = self._run("relu", rng, activation_kwargs={"negative_slope": 0.25})
        assert float(np.min(pre)) < 0.0
        np.testing.assert_allclose(out, _leaky(pre, 0.25), rtol=0.0, atol=EPS32)

    @pytest.mark.parametrize("name, reference", [
        ("relu", lambda v: np.maximum(v, 0.0)),
        ("relu6", lambda v: np.clip(v, 0.0, 6.0)),
        ("tanh", np.tanh),
        ("leaky_relu", lambda v: _leaky(v, 0.2)),
    ])
    def test_string_names(self, name, reference, rng):
        layer, pre, out = self._run(name, rng)
        assert layer.act1 is not None
        np.testing.assert_allclose(out, reference(pre), rtol=0.0, atol=4 * EPS32)

    @pytest.mark.parametrize("name", ["gelu", "silu", "hard_swish"])
    def test_registry_names_match_their_keras_function(self, name, rng):
        layer, pre, out = self._run(name, rng)
        reference = {
            "gelu": exact_gelu,
            "silu": keras.activations.silu,
            "hard_swish": keras.activations.hard_swish,
        }[name](tf.constant(pre))
        peak = max(1.0, float(np.max(np.abs(pre))))
        np.testing.assert_allclose(out, to_numpy(reference), rtol=0.0, atol=8 * EPS32 * peak)

    def test_callable_is_applied(self, rng):
        layer, pre, out = self._run(exact_gelu, rng)
        np.testing.assert_allclose(
            out, to_numpy(exact_gelu(tf.constant(pre))), rtol=0.0, atol=0.0)

    def test_layer_instance_is_applied(self, rng):
        layer, pre, out = self._run(keras.layers.LeakyReLU(0.1), rng)
        np.testing.assert_allclose(out, _leaky(pre, 0.1), rtol=0.0, atol=EPS32)

    def test_a_layer_instance_is_built_with_the_stem_not_in_call(self):
        """An activation layer with weights must exist after ``build()`` alone."""
        layer = ConvStem(filters=6, normalization_type=None,
                         activation_type=keras.layers.PReLU())
        layer.build((B, 8, 8, 3))
        assert [w.path for w in layer.weights if "alpha" in w.path], layer.weights

    def test_each_activation_site_owns_its_own_layer_instance(self):
        """Two sites sharing one PReLU would share one ``alpha``."""
        layer = ConvStem(filters=6, num_convs=2, activation_type=keras.layers.PReLU())
        layer.build((B, 8, 8, 3))
        assert layer.act1 is not layer.act2
        alphas = [w for w in layer.weights if "alpha" in w.path]
        assert len(alphas) == 2

    @pytest.mark.parametrize("activation", ["linear", None])
    def test_linear_and_none_build_no_activation(self, activation, rng):
        layer = ConvStem(filters=6, num_convs=2, normalization_type=None,
                         activation_type=activation)
        assert layer.act1 is None and layer.act2 is None
        x = random_input(rng, (B, 8, 8, 3))
        out = to_numpy(layer(x))
        np.testing.assert_allclose(
            out, to_numpy(layer.conv2(layer.conv1(x))), rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------
# get_config
# ---------------------------------------------------------------------

_ACTIVATIONS = [
    pytest.param("relu", id="str"),
    pytest.param("gelu", id="gelu_str"),
    pytest.param(exact_gelu, id="exact_gelu_callable"),
    pytest.param(keras.layers.LeakyReLU(0.1), id="layer_instance"),
    pytest.param("linear", id="linear"),
    pytest.param(None, id="none"),
]


class TestConvStemGetConfig:
    @pytest.mark.parametrize("activation", _ACTIVATIONS)
    def test_config_is_json_serializable_and_round_trips(self, activation, rng):
        layer = ConvStem(
            filters=6, kernel_size=(3, 3), strides=2, num_convs=2,
            conv2_filters=8, use_pooling=True, pool_size=3, pool_strides=2,
            pool_padding="same", activation_type=activation,
            normalization_kwargs={"momentum": 0.9, "epsilon": 1e-3},
            bias_initializer="ones", bias_regularizer=keras.regularizers.L2(1e-4),
            kernel_regularizer=keras.regularizers.L2(1e-3), name="cfg",
        )
        text = json.dumps(layer.get_config())
        config = json.loads(text)
        rebuilt = ConvStem.from_config(config)
        assert json.dumps(rebuilt.get_config(), sort_keys=True) == json.dumps(
            layer.get_config(), sort_keys=True)
        x = random_input(rng, (B, 16, 16, 3))
        layer.build(x.shape)
        rebuilt.build(x.shape)
        rebuilt.set_weights(layer.get_weights())
        np.testing.assert_allclose(
            to_numpy(rebuilt(x)), to_numpy(layer(x)), rtol=0.0, atol=4 * EPS32)

    def test_the_activation_serializes_to_the_documented_form(self):
        assert ConvStem(filters=4, activation_type="relu").get_config()[
            "activation_type"] == "relu"
        # An exact callable is written as the registered function, NOT as the
        # stock name (``test_a_stem_activation_survives_the_config_round_trip.py``).
        exact_config = ConvStem(filters=4, activation_type=exact_gelu).get_config()[
            "activation_type"]
        assert exact_config["class_name"] == "function"
        assert exact_config["config"].endswith(">exact_gelu")
        assert ConvStem(filters=4, activation_type=None).get_config()[
            "activation_type"] is None
        layer_config = ConvStem(
            filters=4, activation_type=keras.layers.LeakyReLU(0.1)
        ).get_config()["activation_type"]
        assert layer_config["class_name"] == "LeakyReLU"
        assert layer_config["config"]["negative_slope"] == 0.1

    def test_a_reloaded_layer_activation_keeps_its_slope(self, rng):
        layer = ConvStem(filters=4, normalization_type=None,
                         activation_type=keras.layers.LeakyReLU(0.3))
        rebuilt = ConvStem.from_config(json.loads(json.dumps(layer.get_config())))
        x = random_input(rng, (B, 8, 8, 3))
        rebuilt.build(x.shape)
        pre = to_numpy(rebuilt.conv1(x))
        np.testing.assert_allclose(to_numpy(rebuilt(x)), _leaky(pre, 0.3), rtol=0.0, atol=EPS32)

    def test_every_constructor_argument_is_in_the_config(self):
        params = set(inspect.signature(ConvStem.__init__).parameters)
        params -= {"self", "kwargs"}
        config = ConvStem(filters=4).get_config()
        assert params <= set(config), params - set(config)

    def test_get_config_reports_pool_and_bias_settings(self):
        config = ConvStem(
            filters=4, use_pooling=True, pool_size=3, pool_strides=2,
            pool_padding="same", bias_initializer="ones").get_config()
        assert config["pool_strides"] == 2
        assert config["pool_padding"] == "same"
        assert config["bias_initializer"]["class_name"] == "Ones"
        assert config["bias_regularizer"] is None


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------

class TestConvStemGradientFlow:
    @pytest.mark.parametrize("kwargs", [
        # A bias in front of a training-mode BatchNorm has an identically zero
        # gradient (the batch mean absorbs it), so the BN arm drops the bias.
        dict(use_bias=False),
        dict(normalization_type=None, use_bias=True),
    ], ids=["batch_norm_no_bias", "no_norm_with_bias"])
    def test_every_trainable_weight_receives_a_gradient(self, kwargs, rng):
        layer = ConvStem(filters=6, num_convs=2, use_pooling=True, pool_size=2,
                         **kwargs)
        x = tf.constant(random_input(rng, (B, 8, 8, 3)))
        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = keras.ops.mean(keras.ops.square(y))
        grads = tape.gradient(loss, layer.trainable_weights)
        assert len(layer.trainable_weights) > 0
        for weight, grad in zip(layer.trainable_weights, grads):
            assert grad is not None, f"{weight.path} received no gradient"
            grad_np = to_numpy(grad)
            assert np.isfinite(grad_np).all(), f"{weight.path} gradient non-finite"
            assert np.max(np.abs(grad_np)) > 0.0, f"{weight.path} gradient is zero"


# ---------------------------------------------------------------------
# The traced forward is the eager forward
# ---------------------------------------------------------------------

# One stem per branch of ``call`` that changes the graph: no pool, a pool, and
# two convs with a pool.
_TRACED_STEMS = {
    "unpooled": dict(filters=8, kernel_size=3, strides=2),
    "pooled": dict(filters=8, kernel_size=7, strides=2, padding="same",
                   use_pooling=True, pool_size=3, pool_strides=2,
                   pool_padding="same"),
    "two_conv_pooled": dict(filters=8, num_convs=2, conv2_filters=12,
                            kernel_size=3, strides=1, padding="same",
                            use_pooling=True, pool_size=2),
}


def _randomized_stem(config, rng, input_shape):
    """A built stem whose every weight (BN statistics included) is non-trivial."""
    layer = ConvStem(name="traced", **config)
    layer.build(input_shape)
    randomize_weights(layer, rng)
    return layer


def _contraction_lengths(layer):
    """Length of each convolution's reduction: ``kh * kw * in_channels``."""
    return [int(np.prod(conv.kernel.shape[:-1]))
            for conv in (layer.conv1, layer.conv2) if conv is not None]


class TestConvStemTracedForward:
    """``fit()`` runs a traced graph, and ``call`` must not need eager values.

    A ``.numpy()`` or a Python branch on a tensor value inside ``call`` is
    correct in the notebook and fails here: ``jit_compile=True`` lowers the
    graph through XLA, and a ``TensorSpec`` with unknown batch and spatial
    sizes traces it without any concrete shape.

    The bound for both arms is ``tests.numerics.reassociation_atol`` over the
    stem's own convolution lengths (batch norm at inference adds no reduction).
    Measured on CPU with TF32 off, over 8 seeds per stem, ``max|eager - xla|``
    is 4.8e-07 to 9.5e-07 for ``unpooled`` (bound 1.9e-05 to 3.1e-05), 9.5e-07 to
    2.9e-06 for ``pooled`` (bound 9.3e-05 to 1.9e-04) and 3.8e-06 to 1.1e-05 for
    ``two_conv_pooled`` (bound 2.5e-04 to 4.4e-04): the bound sits 19x to 50x
    above the noise it models. The unknown-size trace (no XLA) was bit-identical
    to eager in every one of 48 comparisons, so its bound is a device margin,
    not a measured need. The planted control shows the comparison sees a 1e-3
    error in every kernel entry.
    """

    @pytest.mark.parametrize("name", sorted(_TRACED_STEMS))
    def test_the_jit_compiled_forward_matches_the_eager_one(self, name, rng):
        x = random_input(rng, (B, 32, 32, 3))
        layer = _randomized_stem(_TRACED_STEMS[name], rng, x.shape)
        eager = to_numpy(layer(x, training=False))
        compiled = tf.function(
            lambda t: layer(t, training=False), jit_compile=True)
        traced = to_numpy(compiled(tf.constant(x)))
        assert traced.shape == eager.shape
        assert np.isfinite(traced).all()
        bound = reassociation_atol(
            _contraction_lengths(layer), 1, float(np.max(np.abs(eager))))
        np.testing.assert_allclose(traced, eager, rtol=0.0, atol=bound)

    def test_the_jit_bound_is_far_below_a_real_weight_error(self, rng):
        """Control: the same comparison sees a 1e-3 error in every kernel entry."""
        x = random_input(rng, (B, 32, 32, 3))
        config = _TRACED_STEMS["unpooled"]
        layer = _randomized_stem(config, rng, x.shape)
        twin = ConvStem(name="traced_twin", **config)
        twin.build(x.shape)
        twin.set_weights(layer.get_weights())
        twin.conv1.kernel.assign(twin.conv1.kernel + 1e-3)
        eager = to_numpy(layer(x, training=False))
        compiled = tf.function(
            lambda t: twin(t, training=False), jit_compile=True)
        gap = float(np.max(np.abs(to_numpy(compiled(tf.constant(x))) - eager)))
        bound = reassociation_atol(
            _contraction_lengths(layer), 1, float(np.max(np.abs(eager))))
        assert gap > 10.0 * bound, (gap, bound)

    @pytest.mark.parametrize("name", sorted(_TRACED_STEMS))
    def test_a_trace_with_unknown_batch_and_spatial_size_runs_at_any_size(
            self, name, rng):
        layer = _randomized_stem(
            _TRACED_STEMS[name], rng, (None, None, None, 3))
        forward = tf.function(
            lambda t: layer(t, training=False),
            input_signature=[tf.TensorSpec([None, None, None, 3], "float32")])
        concrete = forward.get_concrete_function()
        assert tuple(concrete.output_shapes.as_list()) == layer.compute_output_shape(
            (None, None, None, 3))
        for shape in ((B, 32, 32, 3), (3, 40, 56, 3)):
            x = random_input(rng, shape)
            eager = to_numpy(layer(x, training=False))
            traced = to_numpy(forward(tf.constant(x)))
            assert traced.shape == eager.shape
            bound = reassociation_atol(
                _contraction_lengths(layer), 1, float(np.max(np.abs(eager))))
            np.testing.assert_allclose(traced, eager, rtol=0.0, atol=bound)


# ---------------------------------------------------------------------
# .keras round trip at rtol=0
# ---------------------------------------------------------------------

class TestConvStemKerasRoundTripExact:
    @pytest.mark.parametrize("kwargs", [
        dict(),
        dict(kernel_size=7, strides=2, use_pooling=True, pool_size=3,
             pool_strides=2, pool_padding="same", activation_type="relu"),
        dict(num_convs=2, conv2_filters=8, activation_type=exact_gelu,
             bias_initializer="ones", bias_regularizer=keras.regularizers.L2(1e-4)),
        dict(normalization_type=None, activation_type=keras.layers.LeakyReLU(0.1),
             num_convs=2),
        dict(normalization_type=None, activation_type=None),
    ], ids=["default", "resnet_pool", "two_conv_exact_gelu", "no_norm_layer_act",
            "no_norm_linear"])
    def test_round_trip_is_bit_identical(self, kwargs, tmp_path, rng):
        layer = ConvStem(filters=6, name="exact_rt", **kwargs)
        x = random_input(rng, (B, 16, 16, 3))
        layer.build(x.shape)
        randomize_weights(layer, rng)
        loaded = roundtrip_exact(layer, (16, 16, 3), x, "exact_rt", tmp_path)
        reloaded_stem = [
            m for m in loaded.layers if isinstance(m, ConvStem)][0]
        assert json.dumps(reloaded_stem.get_config(), sort_keys=True) == json.dumps(
            layer.get_config(), sort_keys=True)


# ---------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------

class TestConvStemRegistration:
    def test_registration_contract(self, registration_contract):
        key = registration_contract(
            ConvStem, expected_package="dl_techniques.layers.conv_blocks.stem")
        assert key == "dl_techniques.layers.conv_blocks.stem>ConvStem"

    def test_the_package_exports_the_same_class(self):
        import dl_techniques.layers.conv_blocks as package
        assert package.__all__ == ["ConvStem"]
        assert package.ConvStem is ConvStem

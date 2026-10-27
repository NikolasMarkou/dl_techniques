"""A ``ConvStem`` activation survives ``get_config`` / ``from_config`` as what it is.

The stem stores its activation as a config and rebuilds one layer per site, so
the form the caller wrote must come back as the same form:

* an exact (``substitute_exact``) callable reloads as the registered function,
  not as the stock string, so a reloaded float64 stem computes the same numbers;
* a stock Keras built-in is written as its string and reloads as a layer
  (the type changes, and the gap is measured);
* an unregistered module-level callable fails at reload, not at ``get_config``;
* a Layer activation must be one the stem can clone (a class Keras can locate,
  holding no weights), and is refused at construction otherwise;
* a registered Layer that fails for its OWN reason keeps its own message.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.layers.activations.exact_activations import (
    EXACT_ACTIVATIONS, substitute_exact)
from dl_techniques.layers.conv_blocks.stem import ConvStem

from .stem_instruments import B, to_numpy, unlocatable_activation

# The module scopes TF32 off so the exact-equality comparisons are made in true
# float32 on a TF32-capable GPU as well as on CPU.
pytestmark = pytest.mark.usefixtures("tf32_disabled")


# ---------------------------------------------------------------------
# An exact activation callable survives the config round trip
# ---------------------------------------------------------------------

class TestAnExactActivationSurvivesTheConfigRoundTrip:
    """``substitute_exact("leaky_relu")`` is a CALLABLE, and ConvStem keeps it one.

    Measured before the fix: ``get_config`` wrote the exact form as the string
    ``"leaky_relu"``, and ``from_config`` then built the stock
    ``Activation("leaky_relu")``, so a reloaded float64 stem differed from the
    original by ``max|delta| = 2.97e-08``. A string in ``activation_type`` means
    the stock form (``stem.py`` docstring), so the exact form is written as the
    registered function it is, and reloads as that same function object.
    """

    @staticmethod
    def _stem(activation, name):
        return ConvStem(filters=6, normalization_type=None,
                        activation_type=activation, name=name)

    @staticmethod
    def _data():
        # Negative-heavy: the two forms differ only where the slope applies.
        return np.random.default_rng(7).uniform(-2.0, 2.0, (B, 8, 8, 3))

    def test_the_reloaded_stem_computes_the_exact_form_at_float64(self, float64_policy):
        stem = self._stem(substitute_exact("leaky_relu"), "exact_rt")
        data = self._data()
        stem.build(data.shape)
        rebuilt = ConvStem.from_config(json.loads(json.dumps(stem.get_config())))
        rebuilt.build(data.shape)
        rebuilt.set_weights(stem.get_weights())
        assert stem.conv1.compute_dtype == "float64"
        np.testing.assert_allclose(
            to_numpy(rebuilt(data)), to_numpy(stem(data)), rtol=0.0, atol=0.0)

    def test_the_stock_string_stays_stock_and_differs_from_the_exact_form(
            self, float64_policy):
        """Two-armed control: the instrument sees the 2.97e-08 gap at all.

        Without this arm the round-trip test above could pass vacuously on a
        policy under which the two forms are bit-identical (float32).
        """
        data = self._data()
        exact = self._stem(substitute_exact("leaky_relu"), "exact_ctl")
        stock = self._stem("leaky_relu", "stock_ctl")
        exact.build(data.shape)
        stock.build(data.shape)
        stock.set_weights(exact.get_weights())
        assert stock.get_config()["activation_type"] == "leaky_relu"
        reloaded = ConvStem.from_config(json.loads(json.dumps(stock.get_config())))
        reloaded.build(data.shape)
        reloaded.set_weights(exact.get_weights())
        np.testing.assert_allclose(
            to_numpy(reloaded(data)), to_numpy(stock(data)), rtol=0.0, atol=0.0)
        gap = float(np.max(np.abs(to_numpy(stock(data)) - to_numpy(exact(data)))))
        assert gap > 1e-9, f"stock and exact leaky_relu agree to {gap:.2e}; vacuous"

    @pytest.mark.parametrize("name", ["leaky_relu", "gelu"])
    def test_the_config_holds_the_registered_function_not_the_stock_name(self, name):
        spec = self._stem(substitute_exact(name), "cfg_exact").get_config()[
            "activation_type"]
        assert isinstance(spec, dict) and spec["class_name"] == "function", spec
        restored = ConvStem.from_config(json.loads(json.dumps(
            self._stem(substitute_exact(name), "cfg_exact").get_config())))
        assert restored.activation_type is EXACT_ACTIVATIONS[name]


# ---------------------------------------------------------------------
# A Layer activation must be one the stem can clone
# ---------------------------------------------------------------------

class TestALayerActivationMustBeCloneable:
    """The stem stores a Layer activation as its config and rebuilds it per site.

    Measured before the fix: a Layer Keras cannot locate by name raised
    ``TypeError: Could not locate class`` from deep inside ``keras.layers.
    deserialize``, and a built Layer holding weights was cloned fresh with its
    weights silently dropped. Both are now a ``ValueError`` at construction that
    names the problem. A locatable, unbuilt Layer still works
    (``TestActivationForms``).
    """

    def test_an_unlocatable_layer_is_refused_by_name(self):
        with pytest.raises(ValueError, match="UnlocatableActivation") as caught:
            ConvStem(filters=4, activation_type=unlocatable_activation())
        assert "registered" in str(caught.value)

    def test_a_layer_holding_weights_is_refused(self):
        trained = keras.layers.PReLU()
        trained.build((None, 4))
        assert trained.weights, "control: the PReLU holds an alpha weight"
        with pytest.raises(ValueError, match="PReLU.*weights"):
            ConvStem(filters=4, activation_type=trained)

    def test_a_registered_unbuilt_layer_is_still_accepted(self):
        stem = ConvStem(filters=4, activation_type=keras.layers.PReLU())
        assert isinstance(stem.act1, keras.layers.PReLU)


# ---------------------------------------------------------------------
# What a config round trip does to each kind of callable
# ---------------------------------------------------------------------

def an_unregistered_activation(x):
    """Module level on purpose, and registered with nobody."""
    return keras.ops.relu(x) * 2.0


class TestWhatAConfigRoundTripDoesToACallable:
    """Measured 2026-10-26 (CPU, keras 3.8); the ``stem.py`` docstring states it."""

    def test_an_unregistered_callable_fails_at_reload_not_at_get_config(self):
        stem = ConvStem(filters=4, normalization_type=None,
                        activation_type=an_unregistered_activation)
        config = stem.get_config()  # does not raise
        assert config["activation_type"]["config"] == "an_unregistered_activation"
        assert config["activation_type"]["class_name"] == "function"
        reloadable = json.loads(json.dumps(config))
        with pytest.raises(TypeError, match="an_unregistered_activation"):
            ConvStem.from_config(reloadable)

    def test_an_unregistered_callable_fails_in_load_model_too(self, tmp_path):
        stem = ConvStem(filters=4, normalization_type=None,
                        activation_type=an_unregistered_activation, name="unreg")
        inputs = keras.Input((8, 8, 3))
        path = str(tmp_path / "unregistered.keras")
        keras.Model(inputs, stem(inputs)).save(path)
        with pytest.raises(TypeError, match="an_unregistered_activation"):
            keras.models.load_model(path)

    def test_a_keras_builtin_is_written_as_its_string_and_reloads_as_a_layer(
            self, float64_policy):
        data = np.random.default_rng(7).uniform(-2.0, 2.0, (B, 8, 8, 3))
        stem = ConvStem(filters=6, normalization_type=None,
                        activation_type=keras.activations.gelu, name="builtin")
        stem.build(data.shape)
        assert stem.get_config()["activation_type"] == "gelu"
        assert type(stem.act1) is keras.layers.Activation
        rebuilt = ConvStem.from_config(json.loads(json.dumps(stem.get_config())))
        rebuilt.build(data.shape)
        rebuilt.set_weights(stem.get_weights())
        # The type changes with the string: Activation -> the registry GELU.
        assert type(rebuilt.act1).__name__ == "GELU"
        assert type(rebuilt.act1) is not type(stem.act1)
        delta = float(np.max(np.abs(to_numpy(rebuilt(data)) - to_numpy(stem(data)))))
        # Measured 5.023e-09 (float64, this data); matched by an independent measurement of 5.0e-09.
        assert delta == pytest.approx(5.0e-09, rel=0.1), delta

    def test_the_gap_is_gelus_alone_relu_reloads_exactly(self, float64_policy):
        """Control: the instrument reads 0.0 where the string and the callable agree."""
        data = np.random.default_rng(7).uniform(-2.0, 2.0, (B, 8, 8, 3))
        stem = ConvStem(filters=6, normalization_type=None,
                        activation_type=keras.activations.relu, name="builtin_relu")
        stem.build(data.shape)
        rebuilt = ConvStem.from_config(json.loads(json.dumps(stem.get_config())))
        rebuilt.build(data.shape)
        rebuilt.set_weights(stem.get_weights())
        assert type(rebuilt.act1) is keras.layers.ReLU
        np.testing.assert_allclose(
            to_numpy(rebuilt(data)), to_numpy(stem(data)), rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------
# A registered Layer that fails for its OWN reason keeps its own message
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="test_stem_layer_activation")
class ActivationWithAStrayConfigKey(keras.layers.Layer):
    """Registered, but ``get_config`` emits a key ``__init__`` rejects."""

    def call(self, inputs):
        return keras.ops.relu(inputs)

    def get_config(self):
        return {**super().get_config(), "a_stray_config_key": 1}


@keras.saving.register_keras_serializable(package="test_stem_layer_activation")
class ActivationWhoseFromConfigFails(keras.layers.Layer):
    """Registered, but ``from_config`` raises its own ``ValueError``."""

    def call(self, inputs):
        return keras.ops.relu(inputs)

    @classmethod
    def from_config(cls, config):
        raise ValueError("this activation refuses its own config: sentinel-7731")


class TestARegisteredLayerKeepsItsOwnError:
    """Only Keras' "Could not locate class" is relabelled by the stem.

    Measured before the fix: every ``TypeError`` / ``ValueError`` from
    ``keras.layers.deserialize`` became "cannot clone ... Register it", so a
    registered Layer with a bad config key told the caller to register a class
    that was registered already. ``TestALayerActivationMustBeCloneable`` keeps
    the other direction: an unlocatable class IS relabelled.
    """

    def test_a_stray_config_key_keeps_its_own_typeerror(self):
        with pytest.raises(TypeError, match="a_stray_config_key") as caught:
            ConvStem(filters=4, activation_type=ActivationWithAStrayConfigKey())
        assert "cannot clone" not in str(caught.value)

    def test_a_failing_from_config_keeps_its_own_message(self):
        with pytest.raises(ValueError, match="sentinel-7731") as caught:
            ConvStem(filters=4, activation_type=ActivationWhoseFromConfigFails())
        assert "cannot clone" not in str(caught.value)

    def test_both_layers_are_registered_so_the_error_is_not_a_locate_failure(self):
        """Control: Keras CAN locate both classes, so neither is the relabelled case."""
        for cls in (ActivationWithAStrayConfigKey, ActivationWhoseFromConfigFails):
            assert keras.saving.get_registered_object(
                keras.saving.get_registered_name(cls)) is cls

"""
Comprehensive test suite for expanded activation functions.

This module provides thorough testing of all activation layers following
the modern Keras 3 testing patterns from the dl-techniques framework guide.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict, Type
import tensorflow as tf

from dl_techniques.layers.activations.expanded_activations import (
    BaseActivation, GELU, SiLU, ExpandedActivation,
    xATLU, xGELU, xSiLU, EluPlusOne, get_activation,
    elu_plus_one_plus_epsilon
)


class TestBaseActivation:
    """Test suite for BaseActivation class."""

    @pytest.fixture
    def base_activation(self) -> BaseActivation:
        """Create a BaseActivation instance for testing."""
        return BaseActivation()

    def test_initialization(self, base_activation):
        """Test BaseActivation initialization."""
        assert hasattr(base_activation, 'trainable')
        assert base_activation.trainable is True
        assert not base_activation.built

    def test_config_serialization(self, base_activation):
        """Test configuration serialization."""
        config = base_activation.get_config()
        assert isinstance(config, dict)
        assert 'name' in config
        assert 'trainable' in config


class TestSimpleActivations:
    """Test suite for simple activation functions (GELU, SiLU)."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        return keras.random.normal(shape=(4, 32))

    @pytest.mark.parametrize("activation_class", [GELU, SiLU, EluPlusOne])
    def test_initialization(self, activation_class: Type[BaseActivation]):
        """Test activation initialization."""
        layer = activation_class()
        assert hasattr(layer, 'trainable')
        assert not layer.built

    @pytest.mark.parametrize("activation_class", [GELU, SiLU, EluPlusOne])
    def test_forward_pass(self, activation_class: Type[BaseActivation], sample_input):
        """Test forward pass and building."""
        layer = activation_class()
        output = layer(sample_input)

        assert layer.built
        assert output.shape == sample_input.shape
        assert keras.ops.all(keras.ops.isfinite(output))

    @pytest.mark.parametrize("activation_class", [GELU, SiLU, EluPlusOne])
    def test_serialization_cycle(self, activation_class: Type[BaseActivation], sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = activation_class()(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    @pytest.mark.parametrize("activation_class", [GELU, SiLU, EluPlusOne])
    def test_config_completeness(self, activation_class: Type[BaseActivation]):
        """Test that get_config contains all necessary parameters."""
        layer = activation_class()
        config = layer.get_config()

        # Basic Layer configuration should be present
        assert 'name' in config
        assert 'trainable' in config
        assert 'dtype' in config


class TestExpandedActivations:
    """Test suite for expanded activation functions with trainable alpha."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        return keras.random.normal(shape=(4, 32))

    @pytest.fixture
    def activation_configs(self) -> Dict[str, Dict[str, Any]]:
        """Configuration for different expanded activations."""
        return {
            'default': {},
            'with_regularizer': {
                'alpha_regularizer': keras.regularizers.L2(1e-4)
            },
            'with_constraint': {
                'alpha_constraint': keras.constraints.NonNeg()
            },
            'custom_init': {
                'alpha_initializer': keras.initializers.Constant(0.1)
            }
        }

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    def test_initialization(self, activation_class: Type[ExpandedActivation]):
        """Test expanded activation initialization."""
        layer = activation_class()

        assert hasattr(layer, 'alpha_initializer')
        assert hasattr(layer, 'alpha_regularizer')
        assert hasattr(layer, 'alpha_constraint')
        assert layer.alpha is None  # Not built yet
        assert not layer.built

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    def test_forward_pass(self, activation_class: Type[ExpandedActivation], sample_input):
        """Test forward pass and building."""
        layer = activation_class()
        output = layer(sample_input)

        assert layer.built
        assert layer.alpha is not None  # Alpha weight created
        assert output.shape == sample_input.shape
        assert keras.ops.all(keras.ops.isfinite(output))

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    @pytest.mark.parametrize("config_name", ['default', 'with_regularizer', 'custom_init'])
    def test_serialization_cycle(
            self,
            activation_class: Type[ExpandedActivation],
            activation_configs,
            config_name: str,
            sample_input
    ):
        """CRITICAL TEST: Full serialization cycle with different configurations."""
        config = activation_configs[config_name]

        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = activation_class(**config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Predictions differ after serialization for config {config_name}"
            )

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    def test_config_completeness(self, activation_class: Type[ExpandedActivation]):
        """Test that get_config contains all __init__ parameters."""
        layer = activation_class(
            alpha_regularizer=keras.regularizers.L2(1e-4),
            alpha_constraint=keras.constraints.NonNeg()
        )
        config = layer.get_config()

        # Check all custom parameters are present
        assert 'alpha_initializer' in config
        assert 'alpha_regularizer' in config
        assert 'alpha_constraint' in config

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    def test_alpha_learning(self, activation_class: Type[ExpandedActivation], sample_input):
        """Test that alpha parameter can be updated during training."""
        layer = activation_class()

        # Build layer
        output = layer(sample_input)

        # Get initial alpha value
        initial_alpha = keras.ops.convert_to_numpy(layer.alpha)

        # Create a simple training setup
        with tf.GradientTape() as tape:
            output = layer(sample_input, training=True)
            loss = keras.ops.mean(keras.ops.square(output))

        # Compute and apply gradients
        gradients = tape.gradient(loss, [layer.alpha])
        assert gradients[0] is not None

        # Apply a simple gradient step manually
        layer.alpha.assign_sub(0.01 * gradients[0])

        # Verify alpha changed
        updated_alpha = keras.ops.convert_to_numpy(layer.alpha)
        assert not np.allclose(initial_alpha, updated_alpha, rtol=1e-7, atol=1e-7)

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
            self,
            activation_class: Type[ExpandedActivation],
            sample_input,
            training: bool
    ):
        """Test behavior in different training modes."""
        layer = activation_class()

        output = layer(sample_input, training=training)
        assert output.shape == sample_input.shape


class TestFactoryFunction:
    """Test suite for get_activation factory function."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        return keras.random.normal(shape=(4, 32))

    @pytest.mark.parametrize("activation_name,expected_class", [
        ('gelu', GELU),
        ('silu', SiLU),
        ('xatlu', xATLU),
        ('xgelu', xGELU),
        ('xsilu', xSiLU),
        ('elu_plus_one', EluPlusOne),
    ])
    def test_factory_creation(self, activation_name: str, expected_class: Type[BaseActivation]):
        """Test factory function creates correct activation."""
        activation = get_activation(activation_name)
        assert isinstance(activation, expected_class)

    @pytest.mark.parametrize("activation_name", [
        'GELU', 'SiLU', 'xGELU',  # Test case insensitive
        ' gelu ', ' silu '  # Test whitespace handling
    ])
    def test_case_insensitive_and_whitespace(self, activation_name: str):
        """Test factory handles case insensitive input and whitespace."""
        activation = get_activation(activation_name)
        assert isinstance(activation, BaseActivation)

    def test_unknown_activation_raises_error(self):
        """Test that unknown activation names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("unknown_activation")

    def test_factory_serialization(self, sample_input):
        """Test that factory-created activations can be serialized."""
        activation = get_activation("xgelu")

        # Create model
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = activation(inputs)
        model = keras.Model(inputs, outputs)

        # Test serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'factory_test.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)

            original_pred = model(sample_input)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6
            )


class TestHelperFunctions:
    """Test suite for helper functions."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        return keras.random.normal(shape=(4, 32))

    def test_elu_plus_one_plus_epsilon_positive_output(self, sample_input):
        """Test that elu_plus_one_plus_epsilon always produces positive outputs."""
        output = elu_plus_one_plus_epsilon(sample_input)

        # All outputs should be positive
        assert keras.ops.all(output > 0)

    def test_elu_plus_one_plus_epsilon_shape(self, sample_input):
        """Test that elu_plus_one_plus_epsilon preserves input shape."""
        output = elu_plus_one_plus_epsilon(sample_input)
        assert output.shape == sample_input.shape


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_large_input_values(self):
        """Test activations with very large input values."""
        large_input = keras.ops.ones((2, 16)) * 100.0

        activations = [GELU(), SiLU(), xGELU(), xSiLU(), xATLU()]

        for activation in activations:
            output = activation(large_input)
            assert keras.ops.all(keras.ops.isfinite(output))

    def test_small_input_values(self):
        """Test activations with very small input values."""
        small_input = keras.ops.ones((2, 16)) * 1e-8

        activations = [GELU(), SiLU(), xGELU(), xSiLU(), xATLU()]

        for activation in activations:
            output = activation(small_input)
            assert keras.ops.all(keras.ops.isfinite(output))

    def test_negative_input_values(self):
        """Test activations with negative input values."""
        negative_input = keras.ops.ones((2, 16)) * -10.0

        activations = [GELU(), SiLU(), xGELU(), xSiLU(), xATLU()]

        for activation in activations:
            output = activation(negative_input)
            assert keras.ops.all(keras.ops.isfinite(output))

    def test_zero_input_values(self):
        """Test activations with zero input values."""
        zero_input = keras.ops.zeros((2, 16))

        activations = [GELU(), SiLU(), xGELU(), xSiLU(), xATLU()]

        for activation in activations:
            output = activation(zero_input)
            assert keras.ops.all(keras.ops.isfinite(output))


class TestIntegrationWithModels:
    """Integration tests with common model architectures."""

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample classification data."""
        x = keras.random.normal(shape=(100, 32))
        y = keras.ops.cast(
            keras.random.uniform(shape=(100,), minval=0, maxval=2),
            dtype='int32'
        )
        return x, y

    @pytest.mark.parametrize("activation_name", ['gelu', 'xgelu', 'silu', 'xsilu'])
    def test_in_dense_model(self, activation_name: str, sample_data):
        """Test activations in a simple dense model."""
        x, y = sample_data
        activation = get_activation(activation_name)

        model = keras.Sequential([
            keras.layers.Dense(64),
            activation,
            keras.layers.Dense(32),
            activation,
            keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test training
        history = model.fit(x, y, epochs=1, verbose=0)
        assert len(history.history['loss']) == 1

        # Test prediction
        predictions = model.predict(x[:10], verbose=0)
        assert predictions.shape == (10, 2)
        assert keras.ops.all(keras.ops.sum(predictions, axis=-1) - 1.0 < 1e-5)  # Softmax sums to 1


def debug_layer_serialization(layer_class, layer_config, sample_input):
    """
    Debug helper for layer serialization issues.

    This function helps identify issues with custom layer serialization
    by testing each step of the process.
    """
    from dl_techniques.utils.logger import logger

    try:
        # Test basic functionality
        layer = layer_class(**layer_config)
        output = layer(sample_input)
        logger.info(f"✅ Forward pass successful: {output.shape}")

        # Test configuration
        config = layer.get_config()
        logger.info(f"✅ Configuration keys: {list(config.keys())}")

        # Test serialization
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = layer_class(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(os.path.join(tmpdir, 'test.keras'))
            loaded = keras.models.load_model(os.path.join(tmpdir, 'test.keras'))
            logger.info("✅ Serialization test passed")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

# ---------------------------------------------------------------------
# Mechanism oracles -- plan-2026-08-27T103353-60745fe0 / iter-2/step-9.
#
# The iter-2/step-8 mutation probe neutered ALL SIX classes to the identity
# in one pass (GELU and SiLU returned `inputs`; xATLU/xGELU/xSiLU had their
# gates replaced by `keras.ops.ones_like(inputs)`, which at the shipped
# default `alpha_initializer='zeros'` gives `x * (1*1 - 0) == x`; EluPlusOne
# returned `inputs`). That moved the output by max|delta| = 2.42281 on all
# 288 values and ALL 67 pre-existing tests in this file still passed: the
# suite was BLIND. It is the worst result in the whole probe.
#
# Two recorded traps this section is written against:
#
#   * With `alpha_initializer='zeros'` -- the shipped default -- the alpha
#     widening `gate*(1 + 2a) - a` is an EXACT no-op. Any alpha-focused arm
#     run at the default is INERT and proves nothing. Every `x*` oracle below
#     therefore constructs the layer with `Constant(_ALPHA)`, a non-zero
#     value, and the knob arm compares against alpha=0.
#   * `test_elu_plus_one_plus_epsilon_positive_output` above would have caught
#     the EluPlusOne mutation, except it calls the module-level FUNCTION
#     `elu_plus_one_plus_epsilon`, which the mutation left intact -- not the
#     `EluPlusOne.call` that was gutted. The oracle below goes through the
#     LAYER on purpose.
#
# The gate references are transcribed from each class's published definition
# (SciPy `erf` / `expit` for the exact forms), not from
# `expanded_activations.py`.
# ---------------------------------------------------------------------

from scipy.special import erf as _scipy_erf, expit as _scipy_expit

#: Non-zero alpha for the widened variants. Chosen away from 0 (where the
#: widening is an exact identity) and away from 0.5 (where `1 + 2a == 2`
#: could hide a factor-of-two error in the widening).
_ALPHA = 0.3


def _oracle_inputs() -> np.ndarray:
    """`(6, 16)` standard normals scaled by 2.0, seed 7.

    The scale of 2.0 pushes samples into both saturated tails of every gate,
    so an oracle here is not evaluated only on the near-linear region around
    zero where all six activations look alike.
    """
    rng = np.random.default_rng(7)
    return (rng.standard_normal((6, 16)) * 2.0).astype("float32")


def _gate_reference(name: str, x: np.ndarray) -> np.ndarray:
    """Published gate of each widened activation, in float64 NumPy."""
    if name == "xATLU":
        return (np.arctan(x) + np.pi / 2.0) / np.pi
    if name == "xGELU":
        return 0.5 * (1.0 + _scipy_erf(x / np.sqrt(2.0)))
    if name == "xSiLU":
        return _scipy_expit(x)
    raise AssertionError(f"no gate reference for {name}")


class TestExpandedActivationsAgainstClosedForms:
    """Closed-form equality for all six classes, plus a live-alpha proof."""

    def test_gelu_matches_the_exact_erf_form(self) -> None:
        """`GELU(x) == 0.5 * x * (1 + erf(x / sqrt(2)))`, via SciPy.

        Tolerance atol=1e-6, rtol=0. Derivation: measured max absolute error
        is 2.036e-07 on outputs of magnitude <= ~6, which is under one float32
        ulp at that magnitude (ulp(6.0) = 4.77e-07).
        """
        x = _oracle_inputs()
        y = keras.ops.convert_to_numpy(GELU()(x))
        v = x.astype(np.float64)
        np.testing.assert_allclose(
            y, 0.5 * v * (1.0 + _scipy_erf(v / np.sqrt(2.0))), atol=1e-6, rtol=0.0
        )

    def test_silu_matches_x_times_sigmoid(self) -> None:
        """`SiLU(x) == x * sigmoid(x)`, with SciPy's `expit` as the sigmoid.

        Tolerance atol=1e-6, rtol=0. Derivation: measured max absolute error
        is 1.337e-07, about one float32 ulp of 1.0.
        """
        x = _oracle_inputs()
        y = keras.ops.convert_to_numpy(SiLU()(x))
        v = x.astype(np.float64)
        np.testing.assert_allclose(y, v * _scipy_expit(v), atol=1e-6, rtol=0.0)

    def test_elu_plus_one_layer_matches_elu_plus_one_plus_epsilon(self) -> None:
        """`EluPlusOne(x) == ELU(x) + 1 + epsilon`, through the LAYER.

        Deliberately exercises `EluPlusOne.call`, not the module-level
        function -- see the trap recorded at the top of this section.

        Tolerance atol=1e-6, rtol=0. Derivation: measured max absolute error
        is 2.192e-07 on outputs of magnitude <= ~7, under one float32 ulp at
        that magnitude (ulp(7.0) = 4.77e-07).
        """
        x = _oracle_inputs()
        y = keras.ops.convert_to_numpy(EluPlusOne()(x))
        v = x.astype(np.float64)
        reference = np.where(v > 0.0, v, np.expm1(v)) + 1.0 + keras.backend.epsilon()
        np.testing.assert_allclose(y, reference, atol=1e-6, rtol=0.0)

    @pytest.mark.parametrize("cls", [xATLU, xGELU, xSiLU])
    def test_widened_variant_matches_its_closed_form_at_nonzero_alpha(
            self, cls: Type[ExpandedActivation]
    ) -> None:
        """`f(x) == x * (gate(x) * (1 + 2a) - a)` at a non-zero `a`.

        Tolerance atol=1e-5, rtol=0. Derivation: measured max absolute error
        is 5.620e-07 (xGELU) on outputs of magnitude <= ~8; one float32 ulp at
        8.0 is 9.54e-07, so the measurement is already sub-ulp and 1e-5 leaves
        ~10x headroom for the extra rounding the `(1 + 2a)` scaling adds.
        It is looser than the 1e-6 used above because these outputs are larger
        and pass through two more float32 multiplies.
        """
        x = _oracle_inputs()
        layer = cls(alpha_initializer=keras.initializers.Constant(_ALPHA))
        y = keras.ops.convert_to_numpy(layer(x))

        v = x.astype(np.float64)
        gate = _gate_reference(cls.__name__, v)
        reference = v * (gate * (1.0 + 2.0 * _ALPHA) - _ALPHA)

        np.testing.assert_allclose(y, reference, atol=1e-5, rtol=0.0)

    @pytest.mark.parametrize("cls", [xATLU, xGELU, xSiLU])
    def test_alpha_actually_widens_the_gate(
            self, cls: Type[ExpandedActivation]
    ) -> None:
        """A non-zero `alpha` must change the output relative to `alpha = 0`.

        The knob-effect arm. It is stated against `alpha=0` on purpose,
        because the default IS zero and an arm that only exercised the default
        would compare the layer to itself.

        Threshold 0.1. Derivation: the measured `max|delta|` between
        alpha=0.3 and alpha=0.0 on this input is 1.3215 (xATLU), 1.5101
        (xGELU) and 1.4905 (xSiLU). 0.1 is >13x below the smallest.
        """
        x = _oracle_inputs()
        at_zero = keras.ops.convert_to_numpy(
            cls(alpha_initializer="zeros")(x)
        )
        at_alpha = keras.ops.convert_to_numpy(
            cls(alpha_initializer=keras.initializers.Constant(_ALPHA))(x)
        )
        assert np.abs(at_alpha - at_zero).max() > 0.1

    @pytest.mark.parametrize("cls", [GELU, SiLU, xATLU, xGELU, xSiLU])
    def test_is_not_the_identity(self, cls: Type[BaseActivation]) -> None:
        """None of the five gated activations may pass its input through.

        The cheapest possible non-degeneracy oracle, stated separately from
        the closed forms so a failure names the failure mode directly: the
        step-8 mutation reduced every one of these to `f(x) == x`.
        `EluPlusOne` is excluded -- it is an affine shift of ELU and its
        positive branch legitimately IS `x + 1 + eps`, so "not the identity"
        is the wrong claim for it; its closed form above covers it instead.

        Threshold 0.1. Derivation: the measured `max|f(x) - x|` on this input
        is 5.0335 (GELU), 5.0009 (SiLU), 6.0408 (xATLU), 6.5436 (xGELU) and
        6.4914 (xSiLU); the smallest is 5.00. 0.1 is 50x below that minimum.
        Note this arm is RED for GELU and SiLU under the step-8 mutation but
        NOT for the three `x*` variants: with the gate pinned to 1 they become
        `x * (1 + 2a) - a*x`, i.e. `1.3 * x` at this alpha, which is not the
        identity either. The closed-form arm above is what catches those.
        """
        x = _oracle_inputs()
        y = keras.ops.convert_to_numpy(
            cls(alpha_initializer=keras.initializers.Constant(_ALPHA))(x)
            if issubclass(cls, ExpandedActivation) else cls()(x)
        )
        assert np.abs(y - x).max() > 0.1

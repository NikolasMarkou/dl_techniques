"""
Test suite for the EnergyLayerNorm layer (layers/norms/energy_layer_norm.py).

EnergyLayerNorm is the Energy Transformer's layer norm (arXiv:2302.07253 eq. 1-2). Its
defining property — and the reason stock ``keras.layers.LayerNormalization`` cannot be
reused — is the parameterization:

    gamma is a SCALAR   (shape == ())
    delta is a VECTOR   (shape == (D,))

``test_build_creates_weights`` is the TRIPWIRE for that property (plan
plan_2026-07-13_57c9833e, success criterion S1): a reviewer "fixing" gamma into a
per-feature vector must make it RED.

Coverage: instantiation, invalid config, forward pass, weight shapes, normalization
correctness vs a numpy reference, .keras serialization round-trip, get_config
completeness, from_config, compute_output_shape (incl. PRE-build), variable batch size,
gradient flow, and norms-factory integration (S5).

Run:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \\
        tests/test_layers/test_norms/test_energy_layer_norm.py -q
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.norms.energy_layer_norm import EnergyLayerNorm
from dl_techniques.layers.norms.factory import create_normalization_layer


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def input_shape():
    """(batch, tokens, embed_dim)."""
    return (4, 12, 32)


@pytest.fixture
def sample_input(input_shape):
    rng = np.random.default_rng(42)
    # Non-zero mean and non-unit variance so centering/scaling are both exercised.
    return (rng.standard_normal(size=input_shape) * 2.5 + 1.5).astype("float32")


@pytest.fixture
def layer():
    return EnergyLayerNorm()


# ---------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------

class TestEnergyLayerNorm:
    """Full layer contract for EnergyLayerNorm."""

    def test_instantiation(self):
        default = EnergyLayerNorm()
        assert default.epsilon == 1e-5
        assert isinstance(default.gamma_initializer, keras.initializers.Ones)
        assert isinstance(default.delta_initializer, keras.initializers.Zeros)
        assert default.gamma is None
        assert default.delta is None
        assert not default.built

        custom = EnergyLayerNorm(
            epsilon=1e-3,
            gamma_initializer="zeros",
            delta_initializer="ones",
        )
        assert custom.epsilon == 1e-3
        assert isinstance(custom.gamma_initializer, keras.initializers.Zeros)
        assert isinstance(custom.delta_initializer, keras.initializers.Ones)

    @pytest.mark.parametrize("bad_epsilon", [0.0, -1e-5, -1.0])
    def test_invalid_config(self, bad_epsilon):
        with pytest.raises(ValueError, match="epsilon must be a positive number"):
            EnergyLayerNorm(epsilon=bad_epsilon)

    def test_forward_pass(self, layer, sample_input, input_shape):
        y = layer(sample_input, training=False)
        assert tuple(y.shape) == input_shape
        y_np = keras.ops.convert_to_numpy(y)
        assert np.all(np.isfinite(y_np))
        assert not np.any(np.isnan(y_np))

    # -----------------------------------------------------------------
    # S1 — THE TRIPWIRE: scalar gamma, vector delta.
    # -----------------------------------------------------------------

    def test_build_creates_weights(self, layer, sample_input, input_shape):
        """gamma MUST be a scalar and delta MUST be a (D,) vector (S1).

        This is the entire reason the class exists. Stock LayerNormalization has a
        VECTOR gamma; a vector gamma here breaks the identity g = dL/dx and destroys
        the Energy Transformer's descent guarantee.
        """
        layer.build(input_shape)
        d = input_shape[-1]

        assert tuple(layer.gamma.shape) == (), (
            f"gamma MUST be a SCALAR (shape ()), got {tuple(layer.gamma.shape)}. "
            "A per-feature gamma is NOT the ET layer norm."
        )
        assert tuple(layer.delta.shape) == (d,), (
            f"delta MUST be a VECTOR of shape ({d},), got {tuple(layer.delta.shape)}"
        )
        assert len(layer.trainable_weights) == 2, (
            f"expected exactly 2 trainable weights (gamma, delta), "
            f"got {[w.name for w in layer.trainable_weights]}"
        )
        assert len(layer.non_trainable_weights) == 0

    # -----------------------------------------------------------------
    # Numerical correctness
    # -----------------------------------------------------------------

    def test_normalization_correctness(self, sample_input):
        """With gamma=1, delta=0 the output is zero-mean / unit-variance on axis -1,
        and matches a hand-computed numpy reference."""
        eps = 1e-5
        layer = EnergyLayerNorm(epsilon=eps)
        y = keras.ops.convert_to_numpy(layer(sample_input, training=False))

        # Statistics along the last axis.
        np.testing.assert_allclose(
            y.mean(axis=-1), np.zeros(y.shape[:-1]), atol=1e-5,
            err_msg="output should be zero-mean along the last axis",
        )
        np.testing.assert_allclose(
            y.var(axis=-1), np.ones(y.shape[:-1]), atol=1e-4,
            err_msg="output should be unit-variance along the last axis",
        )

        # Hand-computed reference: gamma*(x - xbar)/sqrt(var + eps) + delta.
        x = sample_input.astype("float64")
        xbar = x.mean(axis=-1, keepdims=True)
        centered = x - xbar
        var = (centered ** 2).mean(axis=-1, keepdims=True)
        ref = 1.0 * centered / np.sqrt(var + eps) + 0.0
        np.testing.assert_allclose(y, ref.astype("float32"), rtol=1e-5, atol=1e-5)

    def test_normalization_correctness_nondefault_weights(self, sample_input):
        """Reference check with a non-unit gamma and a non-zero delta, so a dropped
        gamma or delta cannot pass by accident."""
        eps = 1e-5
        d = sample_input.shape[-1]
        layer = EnergyLayerNorm(epsilon=eps)
        layer.build(sample_input.shape)

        gamma_val = 1.7
        delta_val = np.linspace(-1.0, 1.0, d).astype("float32")
        layer.gamma.assign(keras.ops.convert_to_tensor(gamma_val, dtype="float32"))
        layer.delta.assign(keras.ops.convert_to_tensor(delta_val))

        y = keras.ops.convert_to_numpy(layer(sample_input, training=False))

        x = sample_input.astype("float64")
        xbar = x.mean(axis=-1, keepdims=True)
        centered = x - xbar
        var = (centered ** 2).mean(axis=-1, keepdims=True)
        ref = gamma_val * centered / np.sqrt(var + eps) + delta_val.astype("float64")
        np.testing.assert_allclose(y, ref.astype("float32"), rtol=1e-5, atol=1e-5)

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def test_serialization_cycle(self, sample_input, input_shape):
        inputs = keras.Input(shape=input_shape[1:])
        outputs = EnergyLayerNorm(epsilon=1e-4, delta_initializer="ones")(inputs)
        model = keras.Model(inputs, outputs)

        y_before = keras.ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "energy_layer_norm.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            y_after = keras.ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            y_before, y_after, rtol=1e-6, atol=1e-6,
            err_msg="EnergyLayerNorm outputs differ after a .keras round-trip",
        )

    def test_get_config_complete(self):
        layer = EnergyLayerNorm(
            epsilon=1e-4,
            gamma_initializer="ones",
            delta_initializer="zeros",
        )
        config = layer.get_config()
        for key in ("epsilon", "gamma_initializer", "delta_initializer"):
            assert key in config, f"{key} missing from get_config()"
        assert config["epsilon"] == 1e-4

    def test_from_config_reconstruction(self, sample_input):
        original = EnergyLayerNorm(epsilon=1e-4, delta_initializer="ones")
        rebuilt = EnergyLayerNorm.from_config(original.get_config())

        assert rebuilt.epsilon == original.epsilon
        assert isinstance(rebuilt.delta_initializer, keras.initializers.Ones)

        # Fresh weights (ones-gamma / ones-delta) => identical forward output.
        y0 = keras.ops.convert_to_numpy(original(sample_input, training=False))
        y1 = keras.ops.convert_to_numpy(rebuilt(sample_input, training=False))
        np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)

    # -----------------------------------------------------------------
    # Shapes
    # -----------------------------------------------------------------

    def test_compute_output_shape(self, layer, input_shape):
        layer.build(input_shape)
        assert layer.compute_output_shape(input_shape) == input_shape

    def test_compute_output_shape_before_build(self, input_shape):
        """compute_output_shape must work on an UNBUILT layer (no weight shapes read)."""
        unbuilt = EnergyLayerNorm()
        assert not unbuilt.built
        assert unbuilt.compute_output_shape(input_shape) == input_shape
        assert unbuilt.compute_output_shape((None, 7, 16)) == (None, 7, 16)
        assert not unbuilt.built

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_variable_batch_size(self, batch_size):
        layer = EnergyLayerNorm()
        rng = np.random.default_rng(batch_size)
        x = rng.standard_normal(size=(batch_size, 6, 16)).astype("float32")
        y = layer(x, training=False)
        assert tuple(y.shape) == (batch_size, 6, 16)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))

    # -----------------------------------------------------------------
    # Gradients
    # -----------------------------------------------------------------

    def test_gradient_flow(self, sample_input, input_shape):
        import tensorflow as tf

        layer = EnergyLayerNorm()
        layer.build(input_shape)
        x = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(y))

        grads = tape.gradient(loss, layer.trainable_weights)
        assert len(grads) == 2
        for g, w in zip(grads, layer.trainable_weights):
            assert g is not None, f"no gradient for {w.name}"
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g))), (
                f"non-finite gradient for {w.name}"
            )

    # -----------------------------------------------------------------
    # Factory (S5)
    # -----------------------------------------------------------------

    def test_factory_integration(self, sample_input, input_shape):
        layer = create_normalization_layer("energy_layer_norm", name="eln")
        assert isinstance(layer, EnergyLayerNorm)

        y = layer(sample_input, training=False)
        assert tuple(y.shape) == input_shape
        assert tuple(layer.gamma.shape) == ()
        assert tuple(layer.delta.shape) == (input_shape[-1],)


if __name__ == "__main__":
    pytest.main([__file__])

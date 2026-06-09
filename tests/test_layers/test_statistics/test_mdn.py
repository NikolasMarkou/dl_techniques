"""
Test suite for the modern Mixture Density Network Layer.

Tests core functionality including layer construction, shape handling,
forward pass, loss computation, sampling, serialization, and gradient flow,
following modern Keras 3 best practices.
"""

import os
import keras
import pytest
import numpy as np
import tensorflow as tf
import tempfile
from typing import Dict, Any

from dl_techniques.layers.statistics.mdn_layer import (
    MDNLayer,
    get_point_estimate,
    get_uncertainty,
    get_prediction_intervals
)
from dl_techniques.utils.logger import logger


class TestMDNLayer:
    """Test suite for the MDNLayer class, following Keras 3 best practices."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for the MDNLayer."""
        return {
            "output_dimension": 2,
            "num_mixtures": 3,
            "intermediate_units": 16,
            "diversity_regularizer_strength": 0.01,
            "use_batch_norm": True,
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Provides a sample input tensor for testing."""
        return keras.ops.convert_to_tensor(np.random.normal(size=(8, 10)), dtype="float32")

    def test_initialization(self, layer_config):
        """Tests that the layer initializes correctly."""
        layer = MDNLayer(**layer_config)

        assert layer.output_dim == layer_config["output_dimension"]
        assert layer.num_mix == layer_config["num_mixtures"]
        assert not layer.built, "Layer should not be built on initialization."

        # Verify sub-layers are created but not built, per the "Golden Rule"
        assert layer.mdn_mus is not None, "Sub-layer mdn_mus should be created."
        assert not layer.mdn_mus.built, "Sub-layer mdn_mus should not be built."
        assert layer.intermediate_mu_dense is not None
        assert not layer.intermediate_mu_dense.built
        logger.info("MDN layer initialization test passed.")

    def test_forward_pass_and_build(self, layer_config, sample_input):
        """Tests the forward pass and ensures the layer builds correctly."""
        layer = MDNLayer(**layer_config)
        output = layer(sample_input)

        assert layer.built, "Layer should be marked as built after the first call."

        # Check output shape
        expected_params = (2 * layer_config["output_dimension"] * layer_config["num_mixtures"]) + layer_config["num_mixtures"]
        expected_shape = (sample_input.shape[0], expected_params)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}."
        logger.info("MDN forward pass and build test passed.")

    def test_config_completeness(self, layer_config):
        """Tests that get_config returns all essential __init__ parameters."""
        layer = MDNLayer(**layer_config)
        config = layer.get_config()

        # Check that all keys from the original config are present
        for key in layer_config:
            assert key in config, f"Key '{key}' is missing from the layer's config."
        logger.info("MDN config completeness test passed.")

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Verifies a full serialization and deserialization cycle."""
        # 1. Create a model with the custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MDNLayer(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # 2. Get a prediction from the original model
        original_prediction = model(sample_input)

        # 3. Save and load the model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_mdn_model.keras")
            model.save(filepath)
            # No custom_objects needed due to the registration decorator
            loaded_model = keras.models.load_model(filepath)

        # 4. Get a prediction from the loaded model
        loaded_prediction = loaded_model(sample_input)

        # 5. Verify the predictions are identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_prediction),
            keras.ops.convert_to_numpy(loaded_prediction),
            rtol=1e-6, atol=1e-6,
            err_msg="Predictions differ after serialization cycle."
        )
        logger.info("MDN serialization cycle test passed.")

    def test_gradients_flow(self, layer_config, sample_input):
        """Tests that gradients can flow through the layer."""
        layer = MDNLayer(**layer_config)
        y_true = keras.ops.convert_to_tensor(np.random.normal(size=(sample_input.shape[0], layer_config["output_dimension"])), dtype="float32")

        with tf.GradientTape() as tape:
            # The layer needs to be built to have trainable variables
            y_pred = layer(sample_input, training=True)
            loss = layer.loss_func(y_true, y_pred)

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0, "No gradients were computed."
        assert all(g is not None for g in gradients), "Some gradients are None, indicating a disconnected path."
        logger.info("MDN gradients flow test passed.")

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Tests that the layer runs in all training modes."""
        layer = MDNLayer(**layer_config)
        output = layer(sample_input, training=training)
        assert output.shape[0] == sample_input.shape[0]
        # This test passes if no error is raised
        logger.info(f"MDN training mode test passed for training={training}.")

    def test_split_mixture_params(self, layer_config):
        """Tests the split_mixture_params method."""
        layer = MDNLayer(**layer_config)
        batch_size = 16
        total_params = (2 * layer.output_dim * layer.num_mix) + layer.num_mix
        mock_output = keras.ops.convert_to_tensor(np.random.normal(size=(batch_size, total_params)), dtype="float32")

        mu, sigma, pi = layer.split_mixture_params(mock_output)

        assert mu.shape == (batch_size, layer.num_mix, layer.output_dim)
        assert sigma.shape == (batch_size, layer.num_mix, layer.output_dim)
        assert pi.shape == (batch_size, layer.num_mix)
        logger.info("MDN split_mixture_params test passed.")

    def test_sampling(self, layer_config, sample_input):
        """Tests the sampling method."""
        layer = MDNLayer(**layer_config)
        predictions = layer(sample_input)

        samples = layer.sample(predictions)
        assert samples.shape == (sample_input.shape[0], layer.output_dim)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(samples)))

        # Test with temperature
        samples_temp = layer.sample(predictions, temperature=0.5)
        assert samples_temp.shape == (sample_input.shape[0], layer.output_dim)
        logger.info("MDN sampling test passed.")

    def test_parameter_validation(self):
        """Tests that invalid __init__ parameters raise ValueErrors."""
        with pytest.raises(ValueError, match="output_dimension must be positive"):
            MDNLayer(output_dimension=0, num_mixtures=3)
        with pytest.raises(ValueError, match="num_mixtures must be positive"):
            MDNLayer(output_dimension=2, num_mixtures=0)
        with pytest.raises(ValueError, match="intermediate_units must be positive"):
            MDNLayer(output_dimension=2, num_mixtures=3, intermediate_units=0)
        with pytest.raises(ValueError, match="diversity_regularizer_strength must be non-negative"):
            MDNLayer(output_dimension=2, num_mixtures=3, diversity_regularizer_strength=-0.1)
        logger.info("MDN parameter validation test passed.")

    def test_training_convergence(self):
        """An integration test to ensure the layer can fit a simple dataset."""
        n_samples, input_dim, output_dim, num_mixes = 500, 1, 1, 3
        np.random.seed(42)
        X = np.random.uniform(-5, 5, (n_samples, input_dim)).astype("float32")
        Y = (np.sin(X) + np.random.normal(0, 0.2, (n_samples, output_dim))).astype("float32")

        # FIX: Use the correct variable `output_dim`
        mdn_layer = MDNLayer(output_dim, num_mixes)
        inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(32, activation="relu")(inputs)
        outputs = mdn_layer(x)
        model = keras.Model(inputs, outputs)

        model.compile(optimizer="adam", loss=mdn_layer.loss_func)
        history = model.fit(X, Y, epochs=20, batch_size=32, verbose=0)

        initial_loss = history.history["loss"][0]
        final_loss = history.history["loss"][-1]
        assert final_loss < initial_loss, "Loss did not decrease during training."
        loss_reduction = (initial_loss - final_loss) / initial_loss
        assert loss_reduction > 0.2, f"Loss reduction was only {loss_reduction:.2%}, expected > 20%."
        logger.info(f"MDN training convergence test passed with {loss_reduction:.2%} loss reduction.")


class TestMDNUtilities:
    """Test suite for the MDN helper/utility functions."""

    @pytest.fixture
    def trained_mdn_model(self) -> tuple[keras.Model, MDNLayer]:
        """Provides a simple, trained MDN model and its layer instance."""
        input_dim, output_dim, num_mixes = 1, 1, 3
        mdn_layer = MDNLayer(output_dim, num_mixes)
        inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(16, activation="relu")(inputs)
        outputs = mdn_layer(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss=mdn_layer.loss_func)
        return model, mdn_layer

    def test_utility_functions(self, trained_mdn_model):
        """Tests get_point_estimate, get_uncertainty, and get_prediction_intervals."""
        model, mdn_layer = trained_mdn_model
        batch_size, input_dim, output_dim = 8, 1, 1
        x_test = np.random.normal(size=(batch_size, input_dim)).astype("float32")

        # Test point estimates
        point_estimates = get_point_estimate(model, x_test, mdn_layer)
        assert point_estimates.shape == (batch_size, output_dim)

        # Test uncertainty calculation
        total_variance, aleatoric_variance = get_uncertainty(model, x_test, mdn_layer, point_estimates)
        assert total_variance.shape == (batch_size, output_dim)
        assert aleatoric_variance.shape == (batch_size, output_dim)
        assert np.all(total_variance >= aleatoric_variance)
        assert np.all(total_variance >= 0)

        # Test prediction intervals
        lower, upper = get_prediction_intervals(point_estimates, total_variance)
        assert lower.shape == (batch_size, output_dim)
        assert upper.shape == (batch_size, output_dim)
        assert np.all(upper >= lower)
        logger.info("MDN utility functions test passed.")


class TestMDNNumerics:
    """Correctness tests for the log-space NLL and logits-based pi (Step 4)."""

    def _build_y_pred(self, layer, mu, sigma, pi_logits):
        """Assemble a [batch, total_params] tensor in the [mu, sigma, pi] layout
        that split_mixture_params expects.

        mu, sigma: [batch, num_mix, output_dim]; pi_logits: [batch, num_mix].
        """
        batch = mu.shape[0]
        mu_flat = mu.reshape(batch, -1)
        sigma_flat = sigma.reshape(batch, -1)
        y_pred = np.concatenate([mu_flat, sigma_flat, pi_logits], axis=-1)
        return keras.ops.convert_to_tensor(y_pred.astype("float32"))

    def test_nll_matches_analytic_ground_truth(self):
        """loss_func equals a hand-computed log-space mixture NLL (atol 1e-5).

        Tiny config: num_mix=2, output_dim=1. Hand-chosen mu/sigma/pi-logits and
        target y; the analytic mixture NLL is computed independently in numpy.
        """
        layer = MDNLayer(output_dimension=1, num_mixtures=2,
                         intermediate_units=4, use_batch_norm=False)

        mu = np.array([[[0.5], [-1.0]]], dtype="float32")        # [1,2,1]
        sigma = np.array([[[1.0], [0.5]]], dtype="float32")      # [1,2,1]
        pi_logits = np.array([[0.3, -0.7]], dtype="float32")     # [1,2]
        y = np.array([[0.2]], dtype="float32")                   # [1,1]

        # Analytic ground-truth (log-space, single softmax over pi logits).
        w = np.exp(pi_logits[0] - np.max(pi_logits[0]))
        w = w / w.sum()                                          # mixture weights
        comp_density = (1.0 / (np.sqrt(2.0 * np.pi) * sigma[0, :, 0])) * \
            np.exp(-0.5 * ((y[0, 0] - mu[0, :, 0]) / sigma[0, :, 0]) ** 2)
        nll_ref = -np.log(np.sum(w * comp_density))

        y_pred = self._build_y_pred(layer, mu, sigma, pi_logits)
        loss = keras.ops.convert_to_numpy(
            layer.loss_func(keras.ops.convert_to_tensor(y), y_pred))

        np.testing.assert_allclose(loss, nll_ref, atol=1e-5,
                                   err_msg="log-space NLL diverges from analytic ground-truth")
        logger.info(f"MDN NLL ground-truth test passed (loss={loss:.6f}, ref={nll_ref:.6f}).")

    def test_pi_treated_as_logits_softmax_sums_to_one(self):
        """The pi slice is raw logits; softmax over it sums to 1 (atol 1e-6).

        Drives the model's own pi path (forward pass) and applies the same
        softmax the loss/sampling use, verifying single-softmax semantics.
        """
        layer = MDNLayer(output_dimension=2, num_mixtures=4,
                         intermediate_units=8, use_batch_norm=True)
        x = keras.ops.convert_to_tensor(
            np.random.normal(size=(5, 7)).astype("float32"))
        y_pred = layer(x)

        _, _, pi_logits = layer.split_mixture_params(y_pred)
        weights = keras.ops.convert_to_numpy(
            keras.activations.softmax(pi_logits, axis=-1))

        sums = weights.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-6)
        # And the raw pi slice is NOT already a probability distribution
        # (would be the symptom of the old softplus double-activation).
        raw_pi = keras.ops.convert_to_numpy(pi_logits)
        assert np.any(raw_pi < 0.0), \
            "pi slice should be raw logits (can be negative), not softplus-positive."
        logger.info("MDN pi-sums-to-1 / logits test passed.")

    def test_high_output_dim_nll_is_finite(self):
        """Log-space NLL is finite at high output_dim where prob-space underflows.

        Old prob-space path: prod over output_dim of small Gaussian densities ->
        underflow to 0 -> log(0) = -inf. Log-space sum stays finite.
        """
        output_dim, num_mix, batch = 16, 3, 4
        layer = MDNLayer(output_dimension=output_dim, num_mixtures=num_mix,
                         intermediate_units=8, use_batch_norm=False)

        rng = np.random.default_rng(0)
        mu = rng.normal(size=(batch, num_mix, output_dim)).astype("float32")
        sigma = (0.05 + 0.01 * rng.random((batch, num_mix, output_dim))).astype("float32")
        pi_logits = rng.normal(size=(batch, num_mix)).astype("float32")
        # Target offset from every mean so densities are tiny per-dim.
        y = (mu[:, 0, :] + 0.3).astype("float32")

        y_pred = self._build_y_pred(layer, mu, sigma, pi_logits)
        loss = keras.ops.convert_to_numpy(
            layer.loss_func(keras.ops.convert_to_tensor(y), y_pred))

        assert np.isfinite(loss), f"NLL not finite at output_dim={output_dim}: {loss}"
        logger.info(f"MDN high-output_dim stability test passed (loss={loss:.4f}).")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
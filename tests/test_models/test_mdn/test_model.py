"""
Test suite for the MDNModel (subclassed keras.Model).

These tests exercise MODEL-level wiring — construction, forward-pass shape,
compile/loss auto-injection, fit, sampling, uncertainty decomposition,
serialization round-trip, gradient flow, parameter validation, and the
sample_seed reproducibility fix. Layer numerics (NLL correctness, logits-pi)
are covered in tests/test_layers/test_statistics/test_mdn.py and are NOT
re-tested here.

Conventions:
- Class-based: class TestMDNModel
- Synthetic np.random.normal data (no external dependency)
- model(dummy, training=False) BEFORE any .save (subclassed-model build requirement)
- compile receives NO loss= arg (MDNModel.compile force-injects mdn_layer.loss_func)
- atol=1e-6 for float comparisons (repo convention)
"""

import os
import keras
import pytest
import tempfile
import numpy as np
import tensorflow as tf
from typing import Dict, Any

from dl_techniques.models.mdn.model import MDNModel
from dl_techniques.layers.statistics.mdn_layer import MDNLayer
from dl_techniques.utils.logger import logger


class TestMDNModel:
    """Model-level test suite for MDNModel."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Standard MDNModel configuration for testing."""
        return {
            "hidden_layers": [32, 16],
            "output_dimension": 2,
            "num_mixtures": 3,
        }

    @pytest.fixture
    def input_dim(self) -> int:
        return 5

    @pytest.fixture
    def sample_input(self, input_dim) -> np.ndarray:
        """Synthetic input batch."""
        np.random.seed(0)
        return np.random.normal(size=(8, input_dim)).astype("float32")

    @pytest.fixture
    def synthetic_data(self, input_dim):
        """Synthetic (X, Y) pair for fit/grad tests. output_dimension=2."""
        np.random.seed(42)
        n = 256
        X = np.random.normal(size=(n, input_dim)).astype("float32")
        # Simple deterministic-ish target with mild noise; 2D output.
        Y = np.stack(
            [np.sin(X[:, 0]) + 0.1 * np.random.normal(size=n),
             np.cos(X[:, 1]) + 0.1 * np.random.normal(size=n)],
            axis=-1,
        ).astype("float32")
        return X, Y

    def _expected_params(self, config) -> int:
        return (2 * config["output_dimension"] * config["num_mixtures"]) + config["num_mixtures"]

    # -- 1 -----------------------------------------------------------------
    def test_initialization_stores_config(self, model_config):
        """All __init__ attrs stored; model not built before first call."""
        model = MDNModel(**model_config)

        assert model.hidden_layers_sizes == model_config["hidden_layers"]
        assert model.output_dim == model_config["output_dimension"]
        assert model.num_mix == model_config["num_mixtures"]
        assert model.use_batch_norm is False
        assert model.dropout_rate is None
        assert model.mdn_layer is None        # created in build()
        assert model.feature_layers == []
        assert not model.built
        logger.info("MDNModel initialization test passed.")

    # -- 2 -----------------------------------------------------------------
    def test_forward_pass_output_shape(self, model_config, sample_input):
        """Forward pass yields raw mixture-param tensor; model built after call."""
        model = MDNModel(**model_config)
        output = model(sample_input, training=False)

        assert model.built
        expected = self._expected_params(model_config)
        assert output.shape == (sample_input.shape[0], expected)
        assert model.mdn_layer is not None
        assert isinstance(model.mdn_layer, MDNLayer)
        logger.info("MDNModel forward-pass shape test passed.")

    # -- 3 -----------------------------------------------------------------
    def test_compile_wires_mdn_loss(self, model_config, sample_input):
        """compile (no loss=) force-injects mdn_layer.loss_func as the loss."""
        model = MDNModel(**model_config)
        # Build so mdn_layer exists (compile references self.mdn_layer.loss_func).
        model(sample_input, training=False)

        model.compile(optimizer="adam")
        # model.loss is the bound mdn_layer.loss_func. Keras stores it as a
        # bound method; compare by equality (== on bound methods compares the
        # underlying __func__ + __self__), not identity (a fresh bound-method
        # object can be a distinct reference).
        assert model.loss == model.mdn_layer.loss_func
        assert getattr(model.loss, "__func__", None) is MDNLayer.loss_func
        logger.info("MDNModel compile-wires-loss test passed.")

    # -- 4 (F3 GATE) -------------------------------------------------------
    def test_fit_one_step_loss_decreases(self, model_config, synthetic_data):
        """fit for 5 epochs: final loss < initial loss (compile/loss wiring sane).

        This is the F3 falsification gate: if loss does not decrease, the
        compile/loss injection is broken.
        """
        X, Y = synthetic_data
        model = MDNModel(**model_config)
        model(X[:1], training=False)  # build before compile
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2))

        history = model.fit(X, Y, epochs=5, batch_size=32, verbose=0)
        initial_loss = history.history["loss"][0]
        final_loss = history.history["loss"][-1]

        assert np.isfinite(initial_loss) and np.isfinite(final_loss)
        assert final_loss < initial_loss, (
            f"F3: loss did not decrease (initial={initial_loss:.4f}, "
            f"final={final_loss:.4f}) — compile/loss wiring broken."
        )
        logger.info(
            f"MDNModel fit-one-step test passed "
            f"(initial={initial_loss:.4f} -> final={final_loss:.4f})."
        )

    # -- 5 -----------------------------------------------------------------
    def test_sample_output_shape(self, model_config, sample_input):
        """sample(x, num_samples=K) -> (batch, K, out_dim); no NaN."""
        model = MDNModel(**model_config)
        model(sample_input, training=False)

        k = 10
        samples = model.sample(sample_input, num_samples=k)
        assert samples.shape == (sample_input.shape[0], k, model_config["output_dimension"])
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(samples)))
        logger.info("MDNModel sample-shape test passed.")

    # -- 6 -----------------------------------------------------------------
    def test_predict_with_uncertainty_keys_and_shapes(self, model_config, sample_input):
        """predict_with_uncertainty returns the 6-key dict with correct shapes."""
        model = MDNModel(**model_config)
        model(sample_input, training=False)

        result = model.predict_with_uncertainty(sample_input, confidence_level=0.95)

        expected_keys = {
            "point_estimates", "total_variance", "aleatoric_variance",
            "epistemic_variance", "lower_bound", "upper_bound",
        }
        assert set(result.keys()) == expected_keys

        batch = sample_input.shape[0]
        out_dim = model_config["output_dimension"]
        for key in expected_keys:
            arr = keras.ops.convert_to_numpy(result[key])
            assert arr.shape == (batch, out_dim), f"{key} has shape {arr.shape}"

        total = keras.ops.convert_to_numpy(result["total_variance"])
        aleatoric = keras.ops.convert_to_numpy(result["aleatoric_variance"])
        # Law of total variance: total = aleatoric + epistemic, epistemic >= 0.
        assert np.all(total >= aleatoric - 1e-6)
        assert np.all(total >= -1e-6)
        upper = keras.ops.convert_to_numpy(result["upper_bound"])
        lower = keras.ops.convert_to_numpy(result["lower_bound"])
        assert np.all(upper >= lower)
        logger.info("MDNModel predict_with_uncertainty test passed.")

    # -- 7 -----------------------------------------------------------------
    def test_serialization_round_trip(self, model_config, sample_input):
        """Build via forward pass, save .keras, reload, predictions match (atol 1e-6)."""
        model = MDNModel(**model_config)
        original = model(sample_input, training=False)  # build before save

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "mdn_model.keras")
            model.save(filepath)
            # MDNModel/MDNLayer are @register_keras_serializable -> no custom_objects.
            loaded = keras.models.load_model(filepath)
            reloaded = loaded(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original),
            keras.ops.convert_to_numpy(reloaded),
            rtol=1e-6, atol=1e-6,
            err_msg="Predictions differ after serialization round-trip.",
        )
        logger.info("MDNModel serialization round-trip test passed.")

    # -- 8 -----------------------------------------------------------------
    def test_gradient_flow(self, model_config, sample_input):
        """GradientTape over forward + loss: all grads present and finite."""
        model = MDNModel(**model_config)
        y_true = keras.ops.convert_to_tensor(
            np.random.normal(size=(sample_input.shape[0], model_config["output_dimension"])).astype("float32")
        )

        with tf.GradientTape() as tape:
            y_pred = model(sample_input, training=True)
            loss = model.mdn_layer.loss_func(y_true, y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) > 0
        assert all(g is not None for g in grads), "Some gradients are None."
        assert all(np.all(np.isfinite(keras.ops.convert_to_numpy(g))) for g in grads), \
            "Some gradients are non-finite."
        logger.info("MDNModel gradient-flow test passed.")

    # -- 9 -----------------------------------------------------------------
    def test_parameter_validation(self):
        """Constructor raises ValueError on invalid params."""
        with pytest.raises(ValueError, match="hidden_layers must be a non-empty"):
            MDNModel(hidden_layers=[], output_dimension=2, num_mixtures=3)
        with pytest.raises(ValueError, match="hidden_layers must be a non-empty"):
            MDNModel(hidden_layers=[32, -1], output_dimension=2, num_mixtures=3)
        with pytest.raises(ValueError, match="output_dimension must be a positive"):
            MDNModel(hidden_layers=[32], output_dimension=0, num_mixtures=3)
        with pytest.raises(ValueError, match="num_mixtures must be a positive"):
            MDNModel(hidden_layers=[32], output_dimension=2, num_mixtures=0)
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            MDNModel(hidden_layers=[32], output_dimension=2, num_mixtures=3, dropout_rate=1.0)
        logger.info("MDNModel parameter-validation test passed.")

    # -- 10 ----------------------------------------------------------------
    def test_optional_features_batch_norm_dropout(self, input_dim, sample_input):
        """Construct with batch_norm + dropout enabled; forward pass succeeds."""
        config = {
            "hidden_layers": [32, 16],
            "output_dimension": 2,
            "num_mixtures": 3,
            "use_batch_norm": True,
            "dropout_rate": 0.2,
        }
        model = MDNModel(**config)
        output = model(sample_input, training=True)

        expected = self._expected_params(config)
        assert output.shape == (sample_input.shape[0], expected)
        assert model.use_batch_norm is True
        assert model.dropout_rate == 0.2
        # Inference mode disables dropout -> deterministic forward.
        out_a = keras.ops.convert_to_numpy(model(sample_input, training=False))
        out_b = keras.ops.convert_to_numpy(model(sample_input, training=False))
        np.testing.assert_allclose(out_a, out_b, atol=1e-6)
        logger.info("MDNModel optional-features (batch_norm + dropout) test passed.")

    # -- 11 (sample_seed bug fix) -----------------------------------------
    def test_sample_seed_reproducible(self, model_config, sample_input):
        """Same seed -> identical samples; different seed -> different samples.

        Verifies the sample_seed fix: MDNModel.sample now forwards the per-sample
        seed into MDNLayer.sample (previously computed and discarded -> no-op).
        """
        model = MDNModel(**model_config)
        model(sample_input, training=False)

        s1 = keras.ops.convert_to_numpy(model.sample(sample_input, num_samples=4, seed=123))
        s2 = keras.ops.convert_to_numpy(model.sample(sample_input, num_samples=4, seed=123))
        s3 = keras.ops.convert_to_numpy(model.sample(sample_input, num_samples=4, seed=999))

        np.testing.assert_allclose(
            s1, s2, atol=1e-6,
            err_msg="Same seed must produce identical samples (seed not forwarded).",
        )
        assert not np.allclose(s1, s3, atol=1e-6), \
            "Different seeds should produce different samples."
        logger.info("MDNModel sample_seed reproducibility test passed.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
CCNets Test Suite - Comprehensive Testing for Cooperative Learning

This module provides comprehensive tests for the refined CCNets implementation,
verifying cooperative learning behavior, gradient isolation, and serialization.
"""

import pytest
import keras
import numpy as np
import tempfile
import os

from dl_techniques.models.ccnets.models import create_ccnets_model, CCNetsModel, CCNetsLoss
from dl_techniques.models.ccnets.base import (
    ExplainerNetwork, ReasonerNetwork, ProducerNetwork,
    create_explainer_network, create_reasoner_network, create_producer_network
)


class TestCCNetsLoss:
    """Test the loss computation utilities."""

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for loss testing."""
        batch_size, dim = 4, 10
        reconstructed = keras.random.normal((batch_size, dim))
        generated = keras.random.normal((batch_size, dim))
        input_obs = keras.random.normal((batch_size, dim))
        return reconstructed, generated, input_obs

    def test_inference_loss_computation(self, sample_tensors):
        """Test inference loss computation."""
        reconstructed, generated, _ = sample_tensors

        loss = CCNetsLoss.compute_inference_loss(reconstructed, generated)

        # Should return scalar
        assert loss.shape == ()
        assert loss >= 0  # Loss should be non-negative

        # Test with identical tensors
        identical_loss = CCNetsLoss.compute_inference_loss(generated, generated)
        assert identical_loss < 1e-6  # Should be near zero

    def test_generation_loss_computation(self, sample_tensors):
        """Test generation loss computation."""
        _, generated, input_obs = sample_tensors

        loss = CCNetsLoss.compute_generation_loss(generated, input_obs)

        assert loss.shape == ()
        assert loss >= 0

    def test_reconstruction_loss_computation(self, sample_tensors):
        """Test reconstruction loss computation."""
        reconstructed, _, input_obs = sample_tensors

        loss = CCNetsLoss.compute_reconstruction_loss(reconstructed, input_obs)

        assert loss.shape == ()
        assert loss >= 0

    def test_network_errors_computation(self, sample_tensors):
        """Test network error computation for cooperative objectives."""
        reconstructed, generated, input_obs = sample_tensors

        # Compute individual losses
        inf_loss = CCNetsLoss.compute_inference_loss(reconstructed, generated)
        gen_loss = CCNetsLoss.compute_generation_loss(generated, input_obs)
        rec_loss = CCNetsLoss.compute_reconstruction_loss(reconstructed, input_obs)

        # Compute network errors
        exp_error, rea_error, prod_error = CCNetsLoss.compute_network_errors(
            inf_loss, gen_loss, rec_loss
        )

        # All should be scalars
        assert exp_error.shape == ()
        assert rea_error.shape == ()
        assert prod_error.shape == ()

        # Verify cooperative objective formulas
        expected_exp = inf_loss + gen_loss - rec_loss
        expected_rea = rec_loss + inf_loss - gen_loss
        expected_prod = gen_loss + rec_loss - inf_loss

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(exp_error),
            keras.ops.convert_to_numpy(expected_exp),
            rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(rea_error),
            keras.ops.convert_to_numpy(expected_rea),
            rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(prod_error),
            keras.ops.convert_to_numpy(expected_prod),
            rtol=1e-6, atol=1e-6
        )


class TestCCNetsBaseNetworks:
    """Test individual network components."""

    @pytest.fixture
    def network_params(self):
        """Common parameters for network testing."""
        return {
            'input_dim': 20,
            'explanation_dim': 8,
            'output_dim': 5,
            'hidden_dims': [16, 12],
            'dropout_rate': 0.1
        }

    def test_explainer_network_creation(self, network_params):
        """Test ExplainerNetwork creation and forward pass."""
        explainer = ExplainerNetwork(
            input_dim=network_params['input_dim'],
            explanation_dim=network_params['explanation_dim'],
            hidden_dims=network_params['hidden_dims'],
            dropout_rate=network_params['dropout_rate']
        )

        # Test forward pass
        batch_size = 4
        inputs = keras.random.normal((batch_size, network_params['input_dim']))
        outputs = explainer(inputs, training=True)

        assert outputs.shape == (batch_size, network_params['explanation_dim'])
        # Outputs should be bounded [-1, 1] due to tanh activation
        assert keras.ops.all(outputs >= -1.0)
        assert keras.ops.all(outputs <= 1.0)

    def test_reasoner_network_creation(self, network_params):
        """Test ReasonerNetwork creation and forward pass."""
        reasoner = ReasonerNetwork(
            explanation_dim=network_params['explanation_dim'],
            output_dim=network_params['output_dim'],
            hidden_dims=network_params['hidden_dims'],
            dropout_rate=network_params['dropout_rate']
        )

        # Test forward pass
        batch_size = 4
        explanations = keras.random.normal((batch_size, network_params['explanation_dim']))
        outputs = reasoner(explanations, training=True)

        assert outputs.shape == (batch_size, network_params['output_dim'])
        # Outputs should sum to 1 (softmax)
        output_sums = keras.ops.sum(outputs, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_sums),
            np.ones(batch_size),
            rtol=1e-5, atol=1e-5
        )

    def test_producer_network_creation(self, network_params):
        """Test ProducerNetwork creation and forward pass."""
        producer = ProducerNetwork(
            label_dim=network_params['output_dim'],
            explanation_dim=network_params['explanation_dim'],
            output_dim=network_params['input_dim'],
            hidden_dims=network_params['hidden_dims'],
            dropout_rate=network_params['dropout_rate']
        )

        # Test forward pass
        batch_size = 4
        labels = keras.random.normal((batch_size, network_params['output_dim']))
        explanations = keras.random.normal((batch_size, network_params['explanation_dim']))
        outputs = producer([labels, explanations], training=True)

        assert outputs.shape == (batch_size, network_params['input_dim'])
        # Outputs should be in [0, 1] due to sigmoid activation
        assert keras.ops.all(outputs >= 0.0)
        assert keras.ops.all(outputs <= 1.0)

    def test_factory_functions(self, network_params):
        """Test network factory functions."""
        explainer = create_explainer_network(
            input_dim=network_params['input_dim'],
            explanation_dim=network_params['explanation_dim']
        )

        reasoner = create_reasoner_network(
            explanation_dim=network_params['explanation_dim'],
            output_dim=network_params['output_dim']
        )

        producer = create_producer_network(
            label_dim=network_params['output_dim'],
            explanation_dim=network_params['explanation_dim'],
            output_dim=network_params['input_dim']
        )

        assert isinstance(explainer, ExplainerNetwork)
        assert isinstance(reasoner, ReasonerNetwork)
        assert isinstance(producer, ProducerNetwork)


class TestCCNetsModel:
    """Test the complete CCNets model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size = 8
        input_dim = 20
        output_dim = 5

        observations = keras.random.normal((batch_size, input_dim))
        labels = keras.utils.to_categorical(
            np.random.randint(0, output_dim, batch_size),
            output_dim
        ).astype(np.float32)

        return observations, labels

    @pytest.fixture
    def ccnets_model(self):
        """Create CCNets model for testing."""
        model = create_ccnets_model(
            input_dim=20,
            explanation_dim=8,
            output_dim=5,
            explainer_kwargs={'hidden_dims': [16, 12], 'dropout_rate': 0.1},
            reasoner_kwargs={'hidden_dims': [16, 12], 'dropout_rate': 0.1},
            producer_kwargs={'hidden_dims': [12, 16], 'dropout_rate': 0.1},
            loss_weights=[1.0, 1.0, 1.0, 1.0]  # Corrected: 4 weights
        )
        return model

    def test_model_creation(self, ccnets_model):
        """Test model creation."""
        assert isinstance(ccnets_model, CCNetsModel)
        assert hasattr(ccnets_model, 'explainer_network')
        assert hasattr(ccnets_model, 'reasoner_network')
        assert hasattr(ccnets_model, 'producer_network')
        assert ccnets_model.loss_weights == [1.0, 1.0, 1.0, 1.0]

    def test_forward_pass(self, ccnets_model, sample_data):
        """Test forward pass through complete model."""
        observations, labels = sample_data

        # Test with labels (training mode)
        outputs = ccnets_model([observations, labels], training=False)

        required_keys = [
            'explanation_vector', 'inferred_label',
            'generated_observation', 'reconstructed_observation'
        ]

        for key in required_keys:
            assert key in outputs

        # Check shapes
        batch_size = observations.shape[0]
        assert outputs['explanation_vector'].shape == (batch_size, 8)
        assert outputs['inferred_label'].shape == (batch_size, 5)
        assert outputs['generated_observation'].shape == (batch_size, 20)
        assert outputs['reconstructed_observation'].shape == (batch_size, 20)

        # Test without labels (prediction mode)
        pred_outputs = ccnets_model(observations, training=False)
        assert 'explanation_vector' in pred_outputs
        assert 'inferred_label' in pred_outputs

    def test_loss_computation(self, ccnets_model, sample_data):
        """Test loss computation."""
        observations, labels = sample_data

        outputs = ccnets_model([observations, labels], training=False)
        losses = ccnets_model.compute_losses(observations, labels, outputs)

        required_loss_keys = [
            'classification_loss', 'inference_loss', 'generation_loss',
            'reconstruction_loss', 'total_loss'
        ]

        for key in required_loss_keys:
            assert key in losses
            assert losses[key].shape == ()  # Should be scalar

    def test_training_step(self, ccnets_model, sample_data):
        """Test training step with cooperative learning."""
        observations, labels = sample_data

        # Compile model
        ccnets_model.compile(optimizer='adam')

        # Prepare data in correct format
        data = ((observations, labels), None)

        # Training step
        metrics = ccnets_model.train_step(data)

        # Check returned metrics
        expected_metrics = [
            'classification_loss', 'inference_loss',
            'generation_loss', 'reconstruction_loss'
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert not np.isnan(keras.ops.convert_to_numpy(metrics[metric]))

    def test_prediction_step(self, ccnets_model, sample_data):
        """Test prediction step."""
        observations, _ = sample_data

        predictions = ccnets_model.predict_step(observations)

        expected_keys = ['predictions', 'explanations', 'reconstructions']
        for key in expected_keys:
            assert key in predictions

        assert predictions['predictions'].shape == (observations.shape[0], 5)
        assert predictions['explanations'].shape == (observations.shape[0], 8)
        assert predictions['reconstructions'].shape == observations.shape

    def test_gradient_isolation(self, ccnets_model, sample_data):
        """Test that gradients are properly isolated between networks."""
        observations, labels = sample_data

        # Get initial weights
        explainer_vars = ccnets_model.explainer_network.trainable_variables
        reasoner_vars = ccnets_model.reasoner_network.trainable_variables
        producer_vars = ccnets_model.producer_network.trainable_variables

        initial_explainer_weights = [keras.ops.convert_to_numpy(v) for v in explainer_vars]
        initial_reasoner_weights = [keras.ops.convert_to_numpy(v) for v in reasoner_vars]
        initial_producer_weights = [keras.ops.convert_to_numpy(v) for v in producer_vars]

        # Compile and do one training step
        ccnets_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1))
        data = ((observations, labels), None)
        ccnets_model.train_step(data)

        # Check that weights changed (networks were updated)
        final_explainer_weights = [keras.ops.convert_to_numpy(v) for v in explainer_vars]
        final_reasoner_weights = [keras.ops.convert_to_numpy(v) for v in reasoner_vars]
        final_producer_weights = [keras.ops.convert_to_numpy(v) for v in producer_vars]

        # At least some weights should have changed
        explainer_changed = any(
            not np.allclose(init, final, atol=1e-6)
            for init, final in zip(initial_explainer_weights, final_explainer_weights)
        )
        reasoner_changed = any(
            not np.allclose(init, final, atol=1e-6)
            for init, final in zip(initial_reasoner_weights, final_reasoner_weights)
        )
        producer_changed = any(
            not np.allclose(init, final, atol=1e-6)
            for init, final in zip(initial_producer_weights, final_producer_weights)
        )

        # All networks should be learning
        assert explainer_changed, "Explainer network weights did not change"
        assert reasoner_changed, "Reasoner network weights did not change"
        assert producer_changed, "Producer network weights did not change"


class TestCCNetsSerialization:
    """Test serialization and deserialization."""

    @pytest.fixture
    def ccnets_model(self):
        """Create model for serialization testing."""
        return create_ccnets_model(
            input_dim=10,
            explanation_dim=4,
            output_dim=3,
            explainer_kwargs={'hidden_dims': [8], 'dropout_rate': 0.0},
            reasoner_kwargs={'hidden_dims': [8], 'dropout_rate': 0.0},
            producer_kwargs={'hidden_dims': [8], 'dropout_rate': 0.0},
            loss_weights=[1.0, 1.0, 1.0, 1.0] # Corrected: 4 weights
        )

    def test_model_serialization(self, ccnets_model):
        """Test full model serialization and deserialization."""
        # Create test data
        observations = keras.random.normal((4, 10))
        labels = keras.utils.to_categorical(np.random.randint(0, 3, 4), 3).astype(np.float32)

        # Get original predictions
        original_outputs = ccnets_model([observations, labels], training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'ccnets_model.keras')
            ccnets_model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            # Test loaded model
            loaded_outputs = loaded_model([observations, labels], training=False)

            # Compare outputs
            for key in original_outputs.keys():
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(original_outputs[key]),
                    keras.ops.convert_to_numpy(loaded_outputs[key]),
                    rtol=1e-6, atol=1e-6,
                    err_msg=f"Mismatch in {key} after serialization"
                )

    def test_config_serialization(self, ccnets_model):
        """Test configuration serialization."""
        config = ccnets_model.get_config()

        # Check essential config keys
        assert 'loss_weights' in config
        assert 'use_mixed_precision' in config

        # Test config completeness
        assert config['loss_weights'] == [1.0, 1.0, 1.0, 1.0]
        assert config['use_mixed_precision'] is False


def run_cooperative_learning_validation():
    """
    Run a comprehensive validation of cooperative learning behavior.

    This test verifies that the three networks actually cooperate and
    that the cooperative objectives lead to meaningful learning.
    """
    print("Running Cooperative Learning Validation...")

    # Create larger model for meaningful learning
    model = create_ccnets_model(
        input_dim=50,
        explanation_dim=16,
        output_dim=8,
        explainer_kwargs={'hidden_dims': [64, 32], 'dropout_rate': 0.1},
        reasoner_kwargs={'hidden_dims': [64, 32], 'dropout_rate': 0.1},
        producer_kwargs={'hidden_dims': [32, 64], 'dropout_rate': 0.1},
        loss_weights=[1.0, 0.5, 0.5, 0.5] # Emphasize classification
    )

    # Generate meaningful synthetic data
    n_samples = 200
    X = np.random.randn(n_samples, 50)
    # Create structured labels based on input patterns
    y_indices = (np.sum(X[:, :8], axis=1) > 0).astype(int) * 4 + \
                (np.sum(X[:, 8:16], axis=1) > 0).astype(int) * 2 + \
                (np.sum(X[:, 16:24], axis=1) > 0).astype(int)
    y = keras.utils.to_categorical(y_indices % 8, 8).astype(np.float32)

    X = X.astype(np.float32)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))

    # Track losses over training
    losses_history = []

    # Training for several steps
    batch_size = 32
    for epoch in range(10):
        epoch_losses = []

        for i in range(0, len(X), batch_size):
            batch_x = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            data = ((batch_x, batch_y), None)
            metrics = model.train_step(data)
            epoch_losses.append({
                'classification': keras.ops.convert_to_numpy(metrics['classification_loss']),
                'generation': keras.ops.convert_to_numpy(metrics['generation_loss']),
                'reconstruction': keras.ops.convert_to_numpy(metrics['reconstruction_loss'])
            })

        # Average losses for epoch
        avg_losses = {
            'classification': np.mean([l['classification'] for l in epoch_losses]),
            'generation': np.mean([l['generation'] for l in epoch_losses]),
            'reconstruction': np.mean([l['reconstruction'] for l in epoch_losses])
        }
        losses_history.append(avg_losses)

        print(f"Epoch {epoch + 1}: "
              f"cls={avg_losses['classification']:.4f}, "
              f"gen={avg_losses['generation']:.4f}, "
              f"rec={avg_losses['reconstruction']:.4f}")

    # Validate learning occurred
    initial_losses = losses_history[0]
    final_losses = losses_history[-1]

    # All losses should have decreased
    assert final_losses['classification'] < initial_losses['classification'], \
        "Classification loss should decrease during training"
    assert final_losses['generation'] < initial_losses['generation'], \
        "Generation loss should decrease during training"
    assert final_losses['reconstruction'] < initial_losses['reconstruction'], \
        "Reconstruction loss should decrease during training"

    # Test cooperative behavior
    test_x = X[:10]
    test_y = y[:10]

    outputs = model([test_x, test_y], training=False)

    # Explanations should be meaningful (not all zeros or ones)
    explanations = outputs['explanation_vector']
    assert np.std(keras.ops.convert_to_numpy(explanations)) > 0.1, \
        "Explanations should have meaningful variance"

    # Predictions should be reasonable
    predictions = outputs['inferred_label']
    pred_classes = np.argmax(keras.ops.convert_to_numpy(predictions), axis=1)
    true_classes = np.argmax(test_y, axis=1)
    accuracy = np.mean(pred_classes == true_classes)

    print(f"Final accuracy: {accuracy:.2f}")
    assert accuracy > 0.2, "Model should achieve some learning on structured data"

    print("Cooperative Learning Validation: PASSED")
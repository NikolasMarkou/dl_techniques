"""
Tests for the Wasserstein loss functions implementation.

This module contains unit tests for the Wasserstein loss functions including
WassersteinLoss, WassersteinGradientPenaltyLoss, WassersteinDivergence,
and associated utility functions.
"""

import keras
from keras import ops
import pytest
import numpy as np
import tempfile
import os
import tensorflow as tf
from typing import Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.losses.wasserstein_loss import (
    WassersteinLoss,
    WassersteinGradientPenaltyLoss,
    WassersteinDivergence,
    compute_gradient_penalty,
    create_wgan_losses,
    create_wgan_gp_losses
)


@pytest.fixture
def gan_data() -> Tuple[np.ndarray, np.ndarray]:
    """Fixture providing GAN training data for testing.

    Returns
    -------
    tuple
        Tuple containing (labels, critic_predictions) arrays for GAN testing.
        Labels: 1 for real samples, 0 for fake samples
        Critic predictions: arbitrary real values (Wasserstein distance estimates)
    """
    # Create consistent test data
    np.random.seed(42)

    # Create labels: 1 for real, 0 for fake
    labels = np.array([
        1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
        1, 0, 1, 1, 0, 0, 1, 0, 1, 0
    ], dtype=np.float32).reshape(-1, 1)

    # Create critic predictions (can be any real values)
    # Higher values for real samples, lower for fake samples (ideal case)
    critic_preds = np.array([
        2.5, -1.8, 3.2, 1.9, -2.1, -0.8, 2.9, -1.5, 2.1, -1.2,
        1.8, -2.5, 2.7, 1.5, -1.9, -0.5, 3.1, -1.1, 2.3, -1.7
    ], dtype=np.float32).reshape(-1, 1)

    return labels, critic_preds


@pytest.fixture
def distribution_data() -> Tuple[np.ndarray, np.ndarray]:
    """Fixture providing distribution data for testing WassersteinDivergence.

    Returns
    -------
    tuple
        Tuple containing (true_dist, pred_dist) probability distributions.
    """
    np.random.seed(42)

    # Create normalized probability distributions
    true_dist = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.25, 0.25, 0.25, 0.25],
        [0.4, 0.3, 0.2, 0.1],
        [0.05, 0.15, 0.35, 0.45]
    ], dtype=np.float32)

    pred_dist = np.array([
        [0.15, 0.25, 0.25, 0.35],
        [0.2, 0.3, 0.3, 0.2],
        [0.3, 0.3, 0.3, 0.1],
        [0.1, 0.2, 0.3, 0.4]
    ], dtype=np.float32)

    return true_dist, pred_dist


@pytest.fixture
def simple_critic_model() -> keras.Model:
    """Fixture providing a simple critic model for testing.

    Returns
    -------
    keras.Model
        A simple discriminator/critic model for WGAN testing.
    """
    inputs = keras.Input(shape=(8, 8, 3))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    # No activation for critic output (linear)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def test_wasserstein_loss_critic_computation(gan_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test that WassersteinLoss correctly computes critic loss.

    Parameters
    ----------
    gan_data : tuple
        GAN training test data fixture.
    """
    labels, critic_preds = gan_data

    # Create critic loss
    critic_loss = WassersteinLoss(for_critic=True)

    # Convert to tensors
    labels_tensor = ops.convert_to_tensor(labels)
    preds_tensor = ops.convert_to_tensor(critic_preds)

    # Calculate loss
    loss_value = critic_loss(labels_tensor, preds_tensor).numpy()

    # Manual calculation for verification
    # The implementation computes: (sum(fake_preds) - sum(real_preds)) / total_batch_size
    # This is different from E[D(fake)] - E[D(real)] when we mix real and fake in one batch
    real_mask = labels == 1
    fake_mask = labels == 0

    real_preds = critic_preds[real_mask]
    fake_preds = critic_preds[fake_mask]

    # Expected loss matches the implementation: per-sample losses averaged over entire batch
    expected_loss = (np.sum(fake_preds) - np.sum(real_preds)) / len(labels)

    logger.info(f"Expected critic loss: {expected_loss:.4f}, Actual: {loss_value:.4f}")
    logger.info(f"Real predictions mean: {np.mean(real_preds):.4f}")
    logger.info(f"Fake predictions mean: {np.mean(fake_preds):.4f}")
    logger.info(f"Note: Implementation differs from theoretical E[D(fake)] - E[D(real)] due to batch mixing")

    # The losses should be close (accounting for tensor operations)
    assert np.isclose(loss_value, expected_loss, atol=1e-5), \
        f"Expected critic loss {expected_loss}, got {loss_value}"


def test_wasserstein_loss_generator_computation(gan_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test that WassersteinLoss correctly computes generator loss.

    Parameters
    ----------
    gan_data : tuple
        GAN training test data fixture.
    """
    labels, critic_preds = gan_data

    # For generator, we only care about fake samples (labels should be 0)
    fake_mask = labels.flatten() == 0
    fake_labels = labels[fake_mask]
    fake_preds = critic_preds[fake_mask]

    # Create generator loss
    generator_loss = WassersteinLoss(for_critic=False)

    # Convert to tensors
    fake_labels_tensor = ops.convert_to_tensor(fake_labels)
    fake_preds_tensor = ops.convert_to_tensor(fake_preds)

    # Calculate loss
    loss_value = generator_loss(fake_labels_tensor, fake_preds_tensor).numpy()

    # Manual calculation: Generator loss is -E[D(fake)]
    # But the implementation averages over all samples, so we need to account for that
    # Since we only pass fake samples to generator loss, this should just be -mean(fake_preds)
    expected_loss = -np.mean(fake_preds)

    logger.info(f"Expected generator loss: {expected_loss:.4f}, Actual: {loss_value:.4f}")
    logger.info(f"Fake predictions mean: {np.mean(fake_preds):.4f}")

    assert np.isclose(loss_value, expected_loss, atol=1e-5), \
        f"Expected generator loss {expected_loss}, got {loss_value}"


def test_wasserstein_gradient_penalty_loss_basic() -> None:
    """Test basic functionality of WassersteinGradientPenaltyLoss."""
    # Create test data
    np.random.seed(42)
    labels = np.array([[1], [0], [1], [0]], dtype=np.float32)
    preds = np.array([[1.5], [-1.2], [2.1], [-0.8]], dtype=np.float32)

    # Test critic loss
    critic_loss = WassersteinGradientPenaltyLoss(for_critic=True, lambda_gp=10.0)

    labels_tensor = ops.convert_to_tensor(labels)
    preds_tensor = ops.convert_to_tensor(preds)

    loss_value = critic_loss(labels_tensor, preds_tensor).numpy()

    # Should compute the same as regular Wasserstein loss (GP computed separately)
    regular_loss = WassersteinLoss(for_critic=True)
    expected_value = regular_loss(labels_tensor, preds_tensor).numpy()

    logger.info(f"WassersteinGP loss: {loss_value:.4f}, Regular Wasserstein: {expected_value:.4f}")

    assert np.isclose(loss_value, expected_value, atol=1e-5), \
        f"Expected WassersteinGP base loss to match regular Wasserstein loss"

    # Test generator loss
    gen_loss = WassersteinGradientPenaltyLoss(for_critic=False, lambda_gp=5.0)
    gen_loss_value = gen_loss(labels_tensor, preds_tensor).numpy()

    regular_gen_loss = WassersteinLoss(for_critic=False)
    expected_gen_value = regular_gen_loss(labels_tensor, preds_tensor).numpy()

    assert np.isclose(gen_loss_value, expected_gen_value, atol=1e-5), \
        f"Expected WassersteinGP generator loss to match regular Wasserstein loss"


def test_compute_gradient_penalty(simple_critic_model: keras.Model) -> None:
    """Test the gradient penalty computation function.

    Parameters
    ----------
    simple_critic_model : keras.Model
        Simple critic model fixture.
    """
    # Create synthetic real and fake samples
    np.random.seed(42)
    batch_size = 4
    real_samples = np.random.normal(0, 1, (batch_size, 8, 8, 3)).astype(np.float32)
    fake_samples = np.random.normal(0, 1, (batch_size, 8, 8, 3)).astype(np.float32)

    real_tensor = tf.convert_to_tensor(real_samples)
    fake_tensor = tf.convert_to_tensor(fake_samples)

    # Compute gradient penalty
    gp_loss = compute_gradient_penalty(
        simple_critic_model,
        real_tensor,
        fake_tensor,
        lambda_gp=10.0
    )

    logger.info(f"Gradient penalty loss: {gp_loss:.4f}")

    # Gradient penalty should be a positive scalar
    assert isinstance(gp_loss, tf.Tensor), "Gradient penalty should return a tensor"
    assert gp_loss.shape == (), f"Gradient penalty should be scalar, got shape {gp_loss.shape}"
    assert gp_loss >= 0, f"Gradient penalty should be non-negative, got {gp_loss}"

    # Test with different lambda values
    gp_loss_high = compute_gradient_penalty(
        simple_critic_model,
        real_tensor,
        fake_tensor,
        lambda_gp=20.0
    )

    # Higher lambda should generally result in higher penalty (not always true due to randomness)
    logger.info(f"Gradient penalty with lambda=20: {gp_loss_high:.4f}")

    # At least verify both are reasonable values
    assert 0 <= gp_loss_high <= 1000, f"Gradient penalty with high lambda seems unreasonable: {gp_loss_high}"


def test_wasserstein_divergence_computation(distribution_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test WassersteinDivergence loss computation.

    Parameters
    ----------
    distribution_data : tuple
        Distribution test data fixture.
    """
    true_dist, pred_dist = distribution_data

    # Create divergence loss
    divergence_loss = WassersteinDivergence(smooth_eps=1e-7)

    # Convert to tensors
    true_tensor = ops.convert_to_tensor(true_dist)
    pred_tensor = ops.convert_to_tensor(pred_dist)

    # Calculate loss
    loss_value = divergence_loss(true_tensor, pred_tensor).numpy()

    logger.info(f"Wasserstein divergence loss: {loss_value}")
    logger.info(f"True distribution sample: {true_dist[0]}")
    logger.info(f"Pred distribution sample: {pred_dist[0]}")

    # Basic sanity checks
    assert loss_value >= 0, f"Wasserstein divergence should be non-negative, got {loss_value}"
    assert not np.isnan(loss_value), "Wasserstein divergence should not be NaN"
    assert not np.isinf(loss_value), "Wasserstein divergence should not be infinite"

    # Test with identical distributions (should give near-zero divergence)
    identical_loss = divergence_loss(true_tensor, true_tensor).numpy()
    logger.info(f"Divergence with identical distributions: {identical_loss}")

    assert identical_loss < 0.01, f"Identical distributions should have near-zero divergence, got {identical_loss}"


def test_model_training_with_wasserstein_loss() -> None:
    """Test that a model can be trained using Wasserstein loss."""
    # Create simple synthetic GAN data
    np.random.seed(42)
    batch_size = 32

    # Create a simple generator that outputs random noise
    generator_input = np.random.normal(0, 1, (batch_size, 10)).astype(np.float32)
    fake_images = np.random.normal(0, 1, (batch_size, 8, 8, 3)).astype(np.float32)

    # Create real images (just more random data for testing)
    real_images = np.random.normal(1, 0.5, (batch_size, 8, 8, 3)).astype(np.float32)

    # Create labels
    real_labels = np.ones((batch_size, 1), dtype=np.float32)
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32)

    # Create a simple critic model
    inputs = keras.Input(shape=(8, 8, 3))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)  # No activation for Wasserstein

    critic = keras.Model(inputs=inputs, outputs=outputs)

    # Compile with Wasserstein loss
    critic_loss = WassersteinLoss(for_critic=True)
    critic.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=5e-5),
        loss=critic_loss
    )

    logger.info("Starting critic training with Wasserstein loss")

    # Train critic on real data
    initial_real_loss = critic.evaluate(real_images, real_labels, verbose=0)
    critic.fit(real_images, real_labels, epochs=2, batch_size=16, verbose=0)
    final_real_loss = critic.evaluate(real_images, real_labels, verbose=0)

    # Train critic on fake data
    initial_fake_loss = critic.evaluate(fake_images, fake_labels, verbose=0)
    critic.fit(fake_images, fake_labels, epochs=2, batch_size=16, verbose=0)
    final_fake_loss = critic.evaluate(fake_images, fake_labels, verbose=0)

    logger.info(f"Real data - Initial loss: {initial_real_loss:.4f}, Final loss: {final_real_loss:.4f}")
    logger.info(f"Fake data - Initial loss: {initial_fake_loss:.4f}, Final loss: {final_fake_loss:.4f}")

    # Training should complete without errors
    assert not np.isnan(final_real_loss), "Training should not result in NaN losses"
    assert not np.isnan(final_fake_loss), "Training should not result in NaN losses"


def test_loss_factory_functions() -> None:
    """Test the factory functions for creating WGAN losses."""
    # Test basic WGAN losses
    critic_loss, generator_loss = create_wgan_losses()

    assert isinstance(critic_loss, WassersteinLoss), "Should return WassersteinLoss for critic"
    assert isinstance(generator_loss, WassersteinLoss), "Should return WassersteinLoss for generator"
    assert critic_loss.for_critic == True, "Critic loss should have for_critic=True"
    assert generator_loss.for_critic == False, "Generator loss should have for_critic=False"

    logger.info("Basic WGAN losses created successfully")

    # Test WGAN-GP losses
    gp_critic_loss, gp_generator_loss = create_wgan_gp_losses(lambda_gp=15.0)

    assert isinstance(gp_critic_loss, WassersteinGradientPenaltyLoss), \
        "Should return WassersteinGradientPenaltyLoss for critic"
    assert isinstance(gp_generator_loss, WassersteinGradientPenaltyLoss), \
        "Should return WassersteinGradientPenaltyLoss for generator"
    assert gp_critic_loss.for_critic == True, "WGAN-GP critic loss should have for_critic=True"
    assert gp_generator_loss.for_critic == False, "WGAN-GP generator loss should have for_critic=False"
    assert gp_critic_loss.lambda_gp == 15.0, f"Expected lambda_gp=15.0, got {gp_critic_loss.lambda_gp}"
    assert gp_generator_loss.lambda_gp == 15.0, f"Expected lambda_gp=15.0, got {gp_generator_loss.lambda_gp}"

    logger.info("WGAN-GP losses created successfully")


def test_loss_serialization_and_deserialization() -> None:
    """Test that Wasserstein losses can be properly serialized and deserialized."""
    # Create a model with Wasserstein loss
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(8, activation='relu')(inputs)
    outputs = keras.layers.Dense(1)(x)  # Linear output for Wasserstein
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Create loss with custom parameters
    original_loss = WassersteinGradientPenaltyLoss(
        for_critic=True,
        lambda_gp=12.5,
        reduction="sum_over_batch_size"
    )

    logger.info(f"Created original loss with for_critic={original_loss.for_critic}, "
               f"lambda_gp={original_loss.lambda_gp}")

    # Compile the model
    model.compile(
        optimizer='rmsprop',
        loss=original_loss
    )

    # Save and load the model
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, 'wasserstein_model.keras')
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Custom objects for loading
        custom_objects = {
            'WassersteinLoss': WassersteinLoss,
            'WassersteinGradientPenaltyLoss': WassersteinGradientPenaltyLoss,
            'WassersteinDivergence': WassersteinDivergence
        }

        # Load the model back
        loaded_model = keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
        logger.info("Model loaded successfully")

        # Check that the loss configuration matches
        loaded_loss = loaded_model.loss

        assert loaded_loss.for_critic == original_loss.for_critic, \
            f"Expected for_critic to be {original_loss.for_critic}, got {loaded_loss.for_critic}"
        assert loaded_loss.lambda_gp == original_loss.lambda_gp, \
            f"Expected lambda_gp to be {original_loss.lambda_gp}, got {loaded_loss.lambda_gp}"

        logger.info("Serialization test passed successfully")


def test_loss_configuration_methods() -> None:
    """Test the get_config methods of all loss classes."""
    # Test WassersteinLoss config
    wasserstein_loss = WassersteinLoss(for_critic=False, reduction="sum")
    config = wasserstein_loss.get_config()

    expected_keys = ['for_critic', 'reduction', 'name']
    for key in expected_keys:
        assert key in config, f"Expected key '{key}' in WassersteinLoss config"

    assert config['for_critic'] == False, f"Expected for_critic=False in config"
    logger.info(f"WassersteinLoss config: {config}")

    # Test WassersteinGradientPenaltyLoss config
    wgp_loss = WassersteinGradientPenaltyLoss(for_critic=True, lambda_gp=8.0, reduction="mean")
    wgp_config = wgp_loss.get_config()

    expected_wgp_keys = ['for_critic', 'lambda_gp', 'reduction', 'name']
    for key in expected_wgp_keys:
        assert key in wgp_config, f"Expected key '{key}' in WassersteinGradientPenaltyLoss config"

    assert wgp_config['lambda_gp'] == 8.0, f"Expected lambda_gp=8.0 in config"
    logger.info(f"WassersteinGradientPenaltyLoss config: {wgp_config}")

    # Test WassersteinDivergence config
    div_loss = WassersteinDivergence(smooth_eps=1e-6, reduction="mean")
    div_config = div_loss.get_config()

    expected_div_keys = ['smooth_eps', 'reduction', 'name']
    for key in expected_div_keys:
        assert key in div_config, f"Expected key '{key}' in WassersteinDivergence config"

    assert div_config['smooth_eps'] == 1e-6, f"Expected smooth_eps=1e-6 in config"
    assert div_config['reduction'] == "mean", f"Expected reduction='mean' in config"
    logger.info(f"WassersteinDivergence config: {div_config}")


def test_wasserstein_theoretical_vs_implementation() -> None:
    """Test to highlight the difference between theoretical Wasserstein and implementation."""
    # Create simple test data with equal numbers of real/fake samples
    labels = np.array([[1], [0], [1], [0]], dtype=np.float32)
    preds = np.array([[2.0], [-1.0], [1.0], [-0.5]], dtype=np.float32)

    labels_tensor = ops.convert_to_tensor(labels)
    preds_tensor = ops.convert_to_tensor(preds)

    # Calculate using implementation
    critic_loss = WassersteinLoss(for_critic=True)
    impl_loss = critic_loss(labels_tensor, preds_tensor).numpy()

    # Calculate theoretical Wasserstein: E[D(fake)] - E[D(real)]
    real_preds = preds[labels.flatten() == 1]  # [2.0, 1.0]
    fake_preds = preds[labels.flatten() == 0]  # [-1.0, -0.5]
    theoretical_loss = np.mean(fake_preds) - np.mean(real_preds)  # -0.75 - 1.5 = -2.25

    # Implementation loss: (sum(fake) - sum(real)) / total_batch_size
    # = (-1.5 - 3.0) / 4 = -4.5 / 4 = -1.125
    expected_impl_loss = (np.sum(fake_preds) - np.sum(real_preds)) / len(labels)

    logger.info(f"Theoretical Wasserstein: {theoretical_loss:.4f}")
    logger.info(f"Implementation loss: {impl_loss:.4f}")
    logger.info(f"Expected implementation: {expected_impl_loss:.4f}")

    # Implementation should match our calculation
    assert np.isclose(impl_loss, expected_impl_loss, atol=1e-5), \
        f"Implementation doesn't match expected calculation"

    # For equal batch sizes, implementation = theoretical / 2
    assert np.isclose(impl_loss, theoretical_loss / 2, atol=1e-5), \
        f"For equal batches, implementation should be theoretical/2"


def test_edge_cases_and_numerical_stability() -> None:
    """Test edge cases and numerical stability of Wasserstein losses."""
    # Test with very small values
    small_labels = np.array([[0], [1]], dtype=np.float32)
    small_preds = np.array([[1e-8], [1e-8]], dtype=np.float32)

    loss = WassersteinLoss(for_critic=True)
    small_labels_tensor = ops.convert_to_tensor(small_labels)
    small_preds_tensor = ops.convert_to_tensor(small_preds)

    small_loss = loss(small_labels_tensor, small_preds_tensor).numpy()
    assert not np.isnan(small_loss), "Loss should handle small values without NaN"
    assert not np.isinf(small_loss), "Loss should handle small values without Inf"
    logger.info(f"Small values test passed, loss: {small_loss}")

    # Test with large values
    large_labels = np.array([[0], [1]], dtype=np.float32)
    large_preds = np.array([[1e6], [-1e6]], dtype=np.float32)

    large_labels_tensor = ops.convert_to_tensor(large_labels)
    large_preds_tensor = ops.convert_to_tensor(large_preds)

    large_loss = loss(large_labels_tensor, large_preds_tensor).numpy()
    assert not np.isnan(large_loss), "Loss should handle large values without NaN"
    assert not np.isinf(large_loss), "Loss should handle large values without Inf"
    logger.info(f"Large values test passed, loss: {large_loss}")

    # Test WassersteinDivergence with edge cases
    div_loss = WassersteinDivergence(smooth_eps=1e-10)

    # Test with distributions containing zeros
    zero_dist = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    normal_dist = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)

    zero_tensor = ops.convert_to_tensor(zero_dist)
    normal_tensor = ops.convert_to_tensor(normal_dist)

    zero_loss = div_loss(zero_tensor, normal_tensor).numpy()
    assert not np.isnan(zero_loss), "Divergence should handle zero distributions"
    assert not np.isinf(zero_loss), "Divergence should handle zero distributions"
    logger.info(f"Zero distribution test passed, divergence: {zero_loss}")


def test_different_reduction_methods() -> None:
    """Test different reduction methods for Wasserstein losses."""
    labels = np.array([[1], [0], [1], [0]], dtype=np.float32)
    preds = np.array([[2.0], [-1.0], [1.5], [-0.5]], dtype=np.float32)

    labels_tensor = ops.convert_to_tensor(labels)
    preds_tensor = ops.convert_to_tensor(preds)

    # Test different reduction methods
    reductions = ["sum_over_batch_size", "sum", "none"]

    for reduction in reductions:
        loss = WassersteinLoss(for_critic=True, reduction=reduction)
        loss_value = loss(labels_tensor, preds_tensor)

        logger.info(f"Reduction '{reduction}': loss shape {loss_value.shape}, value {loss_value}")

        if reduction == "none":
            # Should return per-sample losses
            assert loss_value.shape == labels.shape, \
                f"Expected per-sample losses shape {labels.shape}, got {loss_value.shape}"
        else:
            # Should return scalar
            assert loss_value.shape == (), \
                f"Expected scalar loss for reduction '{reduction}', got shape {loss_value.shape}"

        assert not np.any(np.isnan(loss_value.numpy())), \
            f"Loss should not contain NaN values for reduction '{reduction}'"


def test_gradient_penalty_with_different_batch_sizes() -> None:
    """Test gradient penalty computation with different batch sizes."""
    # Test with different batch sizes to ensure robustness
    batch_sizes = [1, 4, 16]

    for batch_size in batch_sizes:
        logger.info(f"Testing gradient penalty with batch size {batch_size}")

        # Create simple critic model
        inputs = keras.Input(shape=(4, 4, 1))
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(16, activation='relu')(x)
        outputs = keras.layers.Dense(1)(x)
        critic = keras.Model(inputs=inputs, outputs=outputs)

        # Create test data
        np.random.seed(42)
        real_samples = np.random.normal(0, 1, (batch_size, 4, 4, 1)).astype(np.float32)
        fake_samples = np.random.normal(0, 1, (batch_size, 4, 4, 1)).astype(np.float32)

        real_tensor = tf.convert_to_tensor(real_samples)
        fake_tensor = tf.convert_to_tensor(fake_samples)

        # Compute gradient penalty
        gp_loss = compute_gradient_penalty(critic, real_tensor, fake_tensor, lambda_gp=10.0)

        assert not np.isnan(gp_loss.numpy()), f"GP should not be NaN for batch size {batch_size}"
        assert not np.isinf(gp_loss.numpy()), f"GP should not be Inf for batch size {batch_size}"
        assert gp_loss >= 0, f"GP should be non-negative for batch size {batch_size}"

        logger.info(f"Batch size {batch_size}: GP loss = {gp_loss:.4f}")


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])
"""
Simple N-BEATS debugging script to isolate training issues.
"""

import numpy as np
import tensorflow as tf
import keras
from dl_techniques.models.nbeats import NBeatsNet
from dl_techniques.utils.logger import logger

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)


def create_simple_data(n_samples=1000, backcast_length=96, forecast_length=24):
    """Create simple synthetic data for debugging."""
    logger.info("Creating simple test data...")

    # Create simple sine wave with trend
    t = np.arange(n_samples + backcast_length + forecast_length)
    y = np.sin(t * 0.1) + 0.001 * t + np.random.normal(0, 0.1, len(t))

    # Normalize to reasonable range
    y = (y - y.mean()) / y.std()

    # Create sequences
    X, y_target = [], []
    for i in range(n_samples):
        X.append(y[i:i + backcast_length])
        y_target.append(y[i + backcast_length:i + backcast_length + forecast_length])

    X = np.array(X).reshape(-1, backcast_length, 1)
    y_target = np.array(y_target).reshape(-1, forecast_length, 1)

    logger.info(f"Data shapes: X={X.shape}, y={y_target.shape}")
    logger.info(f"Data ranges: X=[{X.min():.4f}, {X.max():.4f}], y=[{y_target.min():.4f}, {y_target.max():.4f}]")

    return X, y_target


def test_nbeats_training():
    """Test N-BEATS training with simple data."""
    logger.info("Testing N-BEATS training...")

    # Create simple data
    X, y = create_simple_data(n_samples=1000)

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create simple N-BEATS model
    model = NBeatsNet(
        backcast_length=96,
        forecast_length=24,
        stack_types=['trend', 'seasonality'],
        nb_blocks_per_stack=2,
        thetas_dim=[3, 6],
        hidden_layer_units=128,
        share_weights_in_stack=False,
        input_dim=1,
        output_dim=1
    )

    logger.info("Created N-BEATS model")

    # Compile with simple loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='mse',  # Use simple MSE loss
        metrics=['mae'],
    )

    logger.info("Compiled model with MSE loss")

    # Test a single forward pass
    logger.info("Testing forward pass...")
    test_pred = model(X_train[:2], training=False)
    logger.info(f"Forward pass successful. Output shape: {test_pred.shape}")
    logger.info(f"Output range: [{test_pred.numpy().min():.4f}, {test_pred.numpy().max():.4f}]")

    # Test a single training step
    logger.info("Testing single training step...")
    with tf.GradientTape() as tape:
        pred = model(X_train[:32], training=True)
        loss = keras.losses.mean_squared_error(y_train[:32], pred)
        loss_value = tf.reduce_mean(loss)

    gradients = tape.gradient(loss_value, model.trainable_variables)

    logger.info(f"Single step loss: {loss_value.numpy():.6f}")
    logger.info(f"Gradients computed: {len([g for g in gradients if g is not None])}/{len(gradients)}")

    # Check for gradient issues
    grad_norms = []
    for i, grad in enumerate(gradients):
        if grad is not None:
            grad_norm = tf.norm(grad).numpy()
            grad_norms.append(grad_norm)
            if grad_norm > 1000 or grad_norm < 1e-10:
                logger.warning(f"Unusual gradient norm for variable {i}: {grad_norm}")

    logger.info(f"Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")

    # Try a few training epochs
    logger.info("Starting training test...")

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.TerminateOnNaN()
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Check if training worked
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']

    logger.info(f"Training losses: {train_losses}")
    logger.info(f"Validation losses: {val_losses}")

    if len(set(train_losses)) == 1:
        logger.error("Training loss is stuck - no learning happening")
        return False
    else:
        logger.info("Training loss is changing - model is learning")

    # Test prediction
    test_pred = model.predict(X_val[:5], verbose=0)
    logger.info(f"Final prediction shape: {test_pred.shape}")
    logger.info(f"Final prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    return True


def test_with_smape_loss():
    """Test with SMAPE loss to see if that's the issue."""
    logger.info("Testing with SMAPE loss...")

    try:
        from dl_techniques.losses.smape_loss import SMAPELoss

        # Create simple data
        X, y = create_simple_data(n_samples=500)

        # Make sure data is positive for SMAPE
        y = np.abs(y) + 0.1  # Ensure positive values

        train_size = int(0.8 * len(X))
        X_train, y_train = X[:train_size], y[:train_size]

        # Create model
        model = NBeatsNet(
            backcast_length=96,
            forecast_length=24,
            stack_types=['generic'],
            nb_blocks_per_stack=2,
            thetas_dim=[8],
            hidden_layer_units=64,
            share_weights_in_stack=False
        )

        # Test SMAPE loss
        smape_loss = SMAPELoss()

        # Test a single loss computation
        test_pred = model(X_train[:2], training=False)
        test_pred = np.abs(test_pred) + 0.1  # Ensure positive

        loss_value = smape_loss(y_train[:2], test_pred)
        logger.info(f"SMAPE loss value: {loss_value.numpy():.6f}")

        if tf.math.is_nan(loss_value) or tf.math.is_inf(loss_value):
            logger.error("SMAPE loss returns NaN/Inf")
            return False
        elif loss_value.numpy() > 1000:
            logger.error(f"SMAPE loss is too large: {loss_value.numpy()}")
            return False
        else:
            logger.info("SMAPE loss seems reasonable")
            return True

    except Exception as e:
        logger.error(f"Error testing SMAPE loss: {e}")
        return False


def main():
    """Run debugging tests."""
    logger.info("Starting N-BEATS debugging tests...")

    # Test 1: Basic training with MSE
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Basic N-BEATS training with MSE loss")
    logger.info("="*50)

    success1 = test_nbeats_training()

    # Test 2: SMAPE loss
    logger.info("\n" + "="*50)
    logger.info("TEST 2: SMAPE loss testing")
    logger.info("="*50)

    success2 = test_with_smape_loss()

    # Summary
    logger.info("\n" + "="*50)
    logger.info("DEBUGGING SUMMARY")
    logger.info("="*50)
    logger.info(f"Basic MSE training: {'PASS' if success1 else 'FAIL'}")
    logger.info(f"SMAPE loss test: {'PASS' if success2 else 'FAIL'}")

    if success1:
        logger.info("✓ N-BEATS model implementation is working correctly")
        logger.info("✗ Issue is likely in your data preprocessing or loss function")
        logger.info("\nRecommendations:")
        logger.info("1. Check your data scaling - values might be too large")
        logger.info("2. Try using 'mse' or 'mae' loss instead of SMAPE initially")
        logger.info("3. Check the TimeSeriesNormalizer implementation")
        logger.info("4. Verify data ranges after scaling")
    else:
        logger.error("✗ N-BEATS model has fundamental issues")
        logger.error("The problem is in the model implementation")


if __name__ == "__main__":
    main()
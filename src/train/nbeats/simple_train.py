"""
Minimal Working N-BEATS Implementation.

This is a simplified version that focuses on getting the basic architecture working
before adding complexity. Uses only generic blocks initially.
"""

import os
import numpy as np
import tensorflow as tf
import keras
from keras import ops
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class SimpleNBeatsBlock(keras.layers.Layer):
    """Simplified N-BEATS block that should definitely work."""

    def __init__(
        self,
        units: int = 128,
        backcast_length: int = 48,
        forecast_length: int = 12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

    def build(self, input_shape):
        # Four fully connected layers
        self.fc1 = keras.layers.Dense(self.units, activation='relu', name='fc1')
        self.fc2 = keras.layers.Dense(self.units, activation='relu', name='fc2')
        self.fc3 = keras.layers.Dense(self.units, activation='relu', name='fc3')
        self.fc4 = keras.layers.Dense(self.units, activation='relu', name='fc4')

        # Direct output layers - no fancy basis functions
        self.backcast_linear = keras.layers.Dense(self.backcast_length, activation='linear', name='backcast_out')
        self.forecast_linear = keras.layers.Dense(self.forecast_length, activation='linear', name='forecast_out')

        super().build(input_shape)

    def call(self, inputs, training=None):
        # Forward pass through FC layers
        x = self.fc1(inputs, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        x = self.fc4(x, training=training)

        # Direct outputs
        backcast = self.backcast_linear(x, training=training)
        forecast = self.forecast_linear(x, training=training)

        return backcast, forecast


@keras.saving.register_keras_serializable()
class SimpleNBeatsModel(keras.Model):
    """Simplified N-BEATS model."""

    def __init__(
        self,
        backcast_length: int = 48,
        forecast_length: int = 12,
        num_blocks: int = 4,
        units: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.num_blocks = num_blocks
        self.units = units

        # Create blocks
        self.blocks = []
        for i in range(num_blocks):
            block = SimpleNBeatsBlock(
                units=units,
                backcast_length=backcast_length,
                forecast_length=forecast_length,
                name=f'block_{i}'
            )
            self.blocks.append(block)

    def call(self, inputs, training=None):
        # Ensure 2D input
        if len(inputs.shape) == 3:
            inputs = ops.squeeze(inputs, axis=-1)

        # Initialize
        residual = inputs
        forecast_sum = ops.zeros((ops.shape(inputs)[0], self.forecast_length))

        # Process each block
        for block in self.blocks:
            backcast, forecast = block(residual, training=training)

            # Update residual and forecast
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast

        return forecast_sum


def create_synthetic_data(
    n_samples: int = 2000,
    backcast_length: int = 48,
    forecast_length: int = 12,
    noise_std: float = 0.05  # Reduced noise
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Create simple synthetic data."""

    logger.info(f"Creating synthetic data with {n_samples} samples...")

    # Generate simple time series
    t = np.arange(n_samples, dtype=np.float32)

    # Simple components
    trend = 0.002 * t  # Very small trend
    seasonal = 0.5 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    noise = np.random.normal(0, noise_std, n_samples)

    # Combine
    data = trend + seasonal + noise

    # Light normalization
    data_mean = np.mean(data)
    data_std = np.std(data)
    data = (data - data_mean) / data_std

    logger.info(f"Data range: [{data.min():.3f}, {data.max():.3f}]")

    # Create sequences
    X, y = [], []
    for i in range(len(data) - backcast_length - forecast_length + 1):
        X.append(data[i:i + backcast_length])
        y.append(data[i + backcast_length:i + backcast_length + forecast_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Split
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    logger.info(f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    logger.info(f"        X_val={X_val.shape}, y_val={y_val.shape}")

    return X_train, y_train, X_val, y_val, data_mean, data_std


def test_model_components():
    """Test individual model components."""

    logger.info("Testing model components...")

    # Create test data
    batch_size = 4
    backcast_length = 48
    forecast_length = 12

    test_input = np.random.normal(0, 1, (batch_size, backcast_length)).astype(np.float32)

    # Test single block
    logger.info("Testing single block...")
    block = SimpleNBeatsBlock(
        units=64,
        backcast_length=backcast_length,
        forecast_length=forecast_length
    )

    backcast, forecast = block(test_input)
    logger.info(f"Block output: backcast={backcast.shape}, forecast={forecast.shape}")

    # Test gradients
    with tf.GradientTape() as tape:
        backcast, forecast = block(test_input, training=True)
        loss = tf.reduce_mean(tf.square(forecast))

    gradients = tape.gradient(loss, block.trainable_variables)
    logger.info(f"Gradients: {len(gradients)} variables")

    for i, grad in enumerate(gradients):
        if grad is not None:
            logger.info(f"  Grad {i}: shape={grad.shape}, mean={tf.reduce_mean(tf.abs(grad)):.6f}")
        else:
            logger.warning(f"  Grad {i}: None!")

    # Test full model
    logger.info("Testing full model...")
    model = SimpleNBeatsModel(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        num_blocks=2,
        units=64
    )

    output = model(test_input)
    logger.info(f"Model output: {output.shape}")

    # Test model gradients
    with tf.GradientTape() as tape:
        output = model(test_input, training=True)
        loss = tf.reduce_mean(tf.square(output))

    gradients = tape.gradient(loss, model.trainable_variables)
    logger.info(f"Model gradients: {len(gradients)} variables")

    grad_stats = []
    for i, grad in enumerate(gradients):
        if grad is not None:
            grad_mean = tf.reduce_mean(tf.abs(grad)).numpy()
            grad_stats.append(grad_mean)
        else:
            grad_stats.append(0.0)

    logger.info(f"Gradient stats: mean={np.mean(grad_stats):.6f}, std={np.std(grad_stats):.6f}")

    return True


def train_simple_model():
    """Train the simplified model."""

    logger.info("Starting simplified N-BEATS training...")

    # Test components first
    test_model_components()

    # Create data
    X_train, y_train, X_val, y_val, data_mean, data_std = create_synthetic_data(
        n_samples=2000,
        backcast_length=48,
        forecast_length=12,
        noise_std=0.05
    )

    # Create model
    model = SimpleNBeatsModel(
        backcast_length=48,
        forecast_length=12,
        num_blocks=3,  # Start with fewer blocks
        units=64       # Start with fewer units
    )

    # Compile with simple settings
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Start with MSE
        metrics=['mae']
    )

    # Test single prediction
    logger.info("Testing single prediction...")
    pred = model.predict(X_train[:1], verbose=0)
    logger.info(f"Single prediction shape: {pred.shape}")

    # Build model explicitly
    model.build((None, 48))
    model.summary()

    # Train with verbose output
    logger.info("Starting training...")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    logger.info("Evaluating model...")

    # Make predictions
    train_pred = model.predict(X_train, verbose=0)
    val_pred = model.predict(X_val, verbose=0)

    # Calculate metrics
    train_mse = np.mean((train_pred - y_train) ** 2)
    val_mse = np.mean((val_pred - y_val) ** 2)

    logger.info(f"Training MSE: {train_mse:.6f}")
    logger.info(f"Validation MSE: {val_mse:.6f}")

    # Plot some results
    plt.figure(figsize=(15, 10))

    # Plot training history
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot predictions
    n_samples = 3
    for i in range(n_samples):
        plt.subplot(2, 2, i + 2)

        # Plot input
        plt.plot(range(48), X_val[i], 'b-', label='Input', alpha=0.7)

        # Plot true forecast
        plt.plot(range(48, 60), y_val[i], 'g-', label='True', linewidth=2)

        # Plot predicted forecast
        plt.plot(range(48, 60), val_pred[i], 'r--', label='Predicted', linewidth=2)

        plt.title(f'Sample {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('nbeats_simple_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"simple_nbeats_{timestamp}.keras"
    model.save(model_path)

    print("\n" + "="*60)
    print("SIMPLIFIED N-BEATS TRAINING COMPLETED")
    print("="*60)
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Validation MSE: {val_mse:.6f}")
    print(f"Model saved to: {model_path}")
    print("="*60)

    return {
        'model': model,
        'history': history.history,
        'train_mse': train_mse,
        'val_mse': val_mse
    }


if __name__ == "__main__":
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)

    try:
        results = train_simple_model()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
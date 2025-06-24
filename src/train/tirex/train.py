"""
TiRex Example Usage and Training Script.

This script demonstrates how to use the TiRex model for time series forecasting,
including data preparation, model training, and evaluation.
"""

import keras
import numpy as np
from keras import ops
from typing import List, Tuple
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.tirex import create_tirex_model

# ---------------------------------------------------------------------


def quantile_loss(quantile_levels: List[float]):
    """
    Create quantile loss function for training.

    Args:
        quantile_levels: List of quantile levels.

    Returns:
        Quantile loss function.
    """

    def loss_fn(y_true, y_pred):
        """
        Quantile loss function.

        Args:
            y_true: True values [batch_size, prediction_length].
            y_pred: Predicted quantiles [batch_size, num_quantiles, prediction_length].

        Returns:
            Quantile loss.
        """
        # Expand y_true to match y_pred shape
        y_true_expanded = ops.expand_dims(y_true, axis=1)  # [batch_size, 1, prediction_length]
        y_true_expanded = ops.repeat(y_true_expanded, len(quantile_levels), axis=1)

        # Calculate quantile loss for each quantile level
        quantiles = ops.convert_to_tensor(quantile_levels, dtype=y_pred.dtype)
        quantiles = ops.reshape(quantiles, (1, -1, 1))

        errors = y_true_expanded - y_pred
        loss = ops.maximum(quantiles * errors, (quantiles - 1) * errors)

        return ops.mean(loss)

    return loss_fn

# ---------------------------------------------------------------------


def create_synthetic_time_series(
        num_samples: int = 1000,
        sequence_length: int = 100,
        prediction_length: int = 20,
        noise_level: float = 0.1,
        seasonal_period: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic time series data for testing.

    Args:
        num_samples: Number of time series samples.
        sequence_length: Length of input sequences.
        prediction_length: Length of prediction targets.
        noise_level: Amount of random noise to add.
        seasonal_period: Period of seasonal component.

    Returns:
        Tuple of (input_sequences, target_sequences).
    """
    np.random.seed(42)

    inputs = []
    targets = []

    for i in range(num_samples):
        # Create base trend
        t = np.arange(sequence_length + prediction_length)
        trend = 0.01 * t + np.random.randn() * 0.5

        # Add seasonal component
        seasonal = 2 * np.sin(2 * np.pi * t / seasonal_period) + \
                   1 * np.cos(2 * np.pi * t / (seasonal_period * 2))

        # Add noise
        noise = np.random.randn(len(t)) * noise_level

        # Combine components
        series = trend + seasonal + noise

        # Split into input and target
        input_seq = series[:sequence_length]
        target_seq = series[sequence_length:sequence_length + prediction_length]

        inputs.append(input_seq)
        targets.append(target_seq)

    return np.array(inputs), np.array(targets)

# ---------------------------------------------------------------------


def train_tirex_model():
    """
    Train a TiRex model on synthetic time series data.
    """
    logger.info("Starting TiRex model training example")

    # Create synthetic data
    sequence_length = 96
    prediction_length = 24

    logger.info("Generating synthetic time series data...")
    X_train, y_train = create_synthetic_time_series(
        num_samples=1000,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        noise_level=0.1
    )

    X_val, y_val = create_synthetic_time_series(
        num_samples=200,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        noise_level=0.1
    )

    logger.info(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")


    # Create model
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    model = create_tirex_model(
        input_length=sequence_length,
        prediction_length=prediction_length,
        patch_size=12,
        embed_dim=128,
        num_blocks=4,
        num_heads=8,
        quantile_levels=quantile_levels,
        block_types=['mixed', 'lstm', 'transformer', 'mixed'],
        dropout_rate=0.1
    )

    logger.info("Model created successfully")
    model.summary()

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=quantile_loss(quantile_levels),
        metrics=['mae']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train model
    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    logger.info("Evaluating model...")
    quantile_preds, mean_preds = model.predict_quantiles(X_val)

    # Calculate evaluation metrics
    mae = np.mean(np.abs(mean_preds - y_val))
    rmse = np.sqrt(np.mean((mean_preds - y_val) ** 2))

    logger.info(f"Evaluation Results:")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")

    # Save model
    model.save("tirex_model.keras")
    logger.info("Model saved to tirex_model.keras")

    return model, history, (X_val, y_val), (quantile_preds, mean_preds)

# ---------------------------------------------------------------------


def plot_predictions(
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        quantile_preds: np.ndarray,
        mean_preds: np.ndarray,
        sample_idx: int = 0
):
    """
    Plot predictions for a sample time series.

    Args:
        model: Trained TiRex model.
        X_test: Test input sequences.
        y_test: Test target sequences.
        quantile_preds: Quantile predictions.
        mean_preds: Mean predictions.
        sample_idx: Index of sample to plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot input sequence
    input_seq = X_test[sample_idx]
    target_seq = y_test[sample_idx]
    pred_seq = mean_preds[sample_idx]

    # Time indices
    input_time = np.arange(len(input_seq))
    target_time = np.arange(len(input_seq), len(input_seq) + len(target_seq))

    # Plot input and target
    plt.plot(input_time, input_seq, label='Historical', color='blue', linewidth=2)
    plt.plot(target_time, target_seq, label='Actual', color='green', linewidth=2)
    plt.plot(target_time, pred_seq, label='Predicted', color='red', linewidth=2)

    # Plot quantile intervals
    quantile_levels = model.quantile_levels
    if len(quantile_levels) >= 5:
        # Plot 80% prediction interval (10th and 90th percentiles)
        q10_idx = quantile_levels.index(0.1) if 0.1 in quantile_levels else 0
        q90_idx = quantile_levels.index(0.9) if 0.9 in quantile_levels else -1

        plt.fill_between(
            target_time,
            quantile_preds[sample_idx, q10_idx],
            quantile_preds[sample_idx, q90_idx],
            alpha=0.3,
            color='red',
            label='80% Prediction Interval'
        )

    plt.axvline(x=len(input_seq), color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'TiRex Prediction Example (Sample {sample_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------


def analyze_model_performance(
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        quantile_preds: np.ndarray,
        mean_preds: np.ndarray
):
    """
    Analyze model performance across different metrics.

    Args:
        model: Trained TiRex model.
        X_test: Test input sequences.
        y_test: Test target sequences.
        quantile_preds: Quantile predictions.
        mean_preds: Mean predictions.
    """
    logger.info("Analyzing model performance...")

    # Point forecast metrics
    mae = np.mean(np.abs(mean_preds - y_test))
    rmse = np.sqrt(np.mean((mean_preds - y_test) ** 2))
    mape = np.mean(np.abs((y_test - mean_preds) / (y_test + 1e-8))) * 100

    logger.info(f"Point Forecast Metrics:")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")

    # Quantile forecast metrics
    quantile_levels = model.quantile_levels

    # Calculate coverage for different prediction intervals
    for i, (q_low, q_high) in enumerate([(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]):
        if q_low in quantile_levels and q_high in quantile_levels:
            low_idx = quantile_levels.index(q_low)
            high_idx = quantile_levels.index(q_high)

            coverage = np.mean(
                (y_test >= quantile_preds[:, low_idx]) &
                (y_test <= quantile_preds[:, high_idx])
            )
            expected_coverage = q_high - q_low

            logger.info(f"{int(expected_coverage * 100)}% Prediction Interval:")
            logger.info(f"  Expected Coverage: {expected_coverage:.1%}")
            logger.info(f"  Actual Coverage: {coverage:.1%}")
            logger.info(f"  Coverage Difference: {abs(coverage - expected_coverage):.1%}")

# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Run training example
    try:
        model, history, test_data, predictions = train_tirex_model()
        X_test, y_test = test_data
        quantile_preds, mean_preds = predictions

        # Analyze performance
        analyze_model_performance(model, X_test, y_test, quantile_preds, mean_preds)

        # Plot sample predictions
        plot_predictions(model, X_test, y_test, quantile_preds, mean_preds, sample_idx=0)

        logger.info("TiRex training example completed successfully!")

    except Exception as e:
        logger.error(f"Error in TiRex training example: {str(e)}")
        raise
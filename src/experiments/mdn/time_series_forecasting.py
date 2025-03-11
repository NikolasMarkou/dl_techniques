"""
Time Series Forecasting with Mixture Density Networks

This experiment demonstrates how MDNs can be used for time series forecasting
with uncertainty quantification, with improved numerical stability for the loss function.

Python 3.11
Keras 3.8.0
TensorFlow 2.18.0 as backend
"""

import os
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from keras.api import layers, regularizers
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, cast

# Assume MDN Layer implementation is imported, but we'll create a custom loss function
from dl_techniques.layers.mdn_layer import (
    MDNLayer,
    get_uncertainty,
    get_point_estimate,
    get_prediction_intervals
)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def ensure_numpy(tensor_or_array):
    """Convert tensor to numpy array if it's a TensorFlow tensor."""
    if isinstance(tensor_or_array, tf.Tensor):
        return tensor_or_array.numpy()
    return tensor_or_array

# ---------------------------------------------------------------------
# Enhanced MDN Loss Function with Numerical Stability
# ---------------------------------------------------------------------

def mdn_negative_log_likelihood(y_true: tf.Tensor, y_pred: tf.Tensor, mdn_layer: MDNLayer) -> tf.Tensor:
    """
    Enhanced MDN loss function with additional numerical stability measures.

    This function computes the negative log likelihood of the target data under
    the mixture model, with safeguards to prevent negative loss values.

    Parameters
    ----------
    y_true : tf.Tensor
        Target values tensor of shape [batch_size, output_dim]
    y_pred : tf.Tensor
        Predicted parameters tensor from the MDN layer
    mdn_layer : MDNLayer
        The MDN layer used to split the mixture parameters

    Returns
    -------
    tf.Tensor
        Non-negative loss value
    """
    with tf.name_scope('MDN_Loss_Enhanced'):
        # Reshape target if needed
        y_true = tf.reshape(y_true, [-1, mdn_layer.output_dim], name='reshape_ytrue')

        # Split the parameters
        out_mu, out_sigma, out_pi = mdn_layer.split_mixture_params(y_pred)

        # Ensure sigma values are positive and have a minimum value for numerical stability
        out_sigma = tf.math.maximum(out_sigma, 1e-6)

        # Compute mixture weights using Keras softmax with improved numerical stability
        # Using log_softmax for better numerical properties
        log_mix_weights = tf.nn.log_softmax(out_pi, axis=-1)  # More stable than direct softmax

        # Expand y_true for broadcasting with mixture components
        y_true_expanded = tf.expand_dims(y_true, 1)  # [batch, 1, output_dim]

        # Compute log of Gaussian probabilities directly (more stable than computing prob then log)
        # log N(y|μ,σ) = -0.5*log(2π) - log(σ) - 0.5*((y-μ)/σ)²
        log_2pi = tf.math.log(tf.constant(2.0 * np.pi, dtype=tf.float32))
        z = (y_true_expanded - out_mu) / out_sigma
        log_component_probs = -0.5 * log_2pi - tf.math.log(out_sigma) - 0.5 * tf.square(z)

        # Sum log probs across output dimensions (assuming independence)
        log_component_probs = tf.reduce_sum(log_component_probs, axis=-1)  # [batch, num_mixes]

        # Combine mixture weights and component probs (in log space, addition replaces multiplication)
        log_weighted_probs = log_mix_weights + log_component_probs  # [batch, num_mixes]

        # Use logsumexp for numerical stability (replaces log(sum(exp(x))))
        log_prob = tf.reduce_logsumexp(log_weighted_probs, axis=-1)  # [batch]

        # Calculate negative log likelihood and ensure it's non-negative
        # Note: log probabilities are always <= 0, so negative log likelihood is always >= 0
        # But we add a safeguard just in case
        negative_log_likelihood = -log_prob
        loss = tf.reduce_mean(negative_log_likelihood)

        # Final numerical stability safeguard
        loss = tf.maximum(loss, 0.0)

        # Verify there are no NaN values
        loss = tf.debugging.check_numerics(loss, "NaN or Inf in loss calculation")

        return loss


# ---------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------

@dataclass
class MDNTimeSeriesConfig:
    """Configuration for MDN time series forecasting experiments.

    Attributes:
        # General experiment config
        result_dir: Directory to save experiment results
        save_results: Whether to save results and plots

        # Data config
        n_samples: Number of data points to generate for each time series
        train_ratio: Fraction of data to use for training
        val_ratio: Fraction of data to use for validation

        # Windowing config
        window_size: Number of time steps to use as input
        pred_horizon: Number of time steps ahead to predict
        stride: Step size between consecutive windows

        # Model config
        num_mixtures: Number of Gaussian mixture components
        hidden_units: List of hidden unit counts for LSTM layers
        dropout_rate: Dropout rate for regularization
        kernel_initializer: Initializer for the kernel weights matrix
        l2_regularization: L2 regularization coefficient
        use_enhanced_loss: Whether to use the enhanced loss function with better numerical stability

        # Training config
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        use_early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        learning_rate: Learning rate for optimizer
        clip_gradients: Whether to clip gradients during training
        gradient_clip_value: Value to clip gradients to if enabled

        # Prediction config
        confidence_level: Confidence level for prediction intervals
        multi_step_forecast_steps: Number of steps for multi-step forecasting
        multi_step_samples: Number of sample trajectories for multi-step forecasting

        # Visualization config
        max_plot_points: Maximum number of points to show in plots

        # Time series generation config
        sine_freq: Frequency of sine wave
        sine_phase: Phase of sine wave
        noisy_sine_noise_level: Noise level for noisy sine wave
        damped_sine_damping: Damping coefficient for damped sine wave
        stock_volatility: Volatility for stock price simulation
        stock_drift: Drift for stock price simulation
        stock_seasonality: Seasonality strength for stock price simulation
    """
    # General experiment config
    result_dir: str = "mdn_forecast_results"
    save_results: bool = True

    # Data config
    n_samples: int = 1000
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Windowing config
    window_size: int = 20
    pred_horizon: int = 1
    stride: int = 1

    # Model config
    num_mixtures: int = 5
    hidden_units: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout_rate: float = 0.2
    kernel_initializer: str = "glorot_uniform"
    l2_regularization: float = 0.001
    use_enhanced_loss: bool = True

    # Training config
    epochs: int = 100
    batch_size: int = 32
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    learning_rate: float = 0.001
    clip_gradients: bool = True
    gradient_clip_value: float = 1.0

    # Prediction config
    confidence_level: float = 0.95
    multi_step_forecast_steps: int = 50
    multi_step_samples: int = 10

    # Visualization config
    max_plot_points: int = 200

    # Time series generation config
    sine_freq: float = 0.1
    sine_phase: float = 0.0
    noisy_sine_noise_level: float = 0.2
    damped_sine_damping: float = 0.005
    stock_volatility: float = 0.01
    stock_drift: float = 0.0002
    stock_seasonality: float = 0.1

    @property
    def kernel_regularizer(self) -> Optional[keras.regularizers.Regularizer]:
        """Return L2 kernel regularizer using the configured value."""
        return regularizers.l2(self.l2_regularization) if self.l2_regularization > 0 else None


# ---------------------------------------------------------------------
# Data Generation Functions for different time series scenarios
# ---------------------------------------------------------------------

def generate_sine_wave(config: MDNTimeSeriesConfig) -> np.ndarray:
    """
    Generate a basic sine wave time series.

    Parameters
    ----------
    config : MDNTimeSeriesConfig
        Configuration parameters

    Returns
    -------
    np.ndarray
        Time series data of shape [n_samples, 1]
    """
    x = np.linspace(0, 10 * np.pi, config.n_samples)
    y = np.sin(config.sine_freq * x + config.sine_phase)
    return y.reshape(-1, 1)

def generate_noisy_sine_wave(config: MDNTimeSeriesConfig) -> np.ndarray:
    """
    Generate a sine wave with Gaussian noise.

    Parameters
    ----------
    config : MDNTimeSeriesConfig
        Configuration parameters

    Returns
    -------
    np.ndarray
        Time series data of shape [n_samples, 1]
    """
    x = np.linspace(0, 10 * np.pi, config.n_samples)
    y = np.sin(config.sine_freq * x) + np.random.normal(0, config.noisy_sine_noise_level, config.n_samples)
    return y.reshape(-1, 1)

def generate_damped_sine_wave(config: MDNTimeSeriesConfig) -> np.ndarray:
    """
    Generate a damped sine wave with decreasing amplitude over time.

    Parameters
    ----------
    config : MDNTimeSeriesConfig
        Configuration parameters

    Returns
    -------
    np.ndarray
        Time series data of shape [n_samples, 1]
    """
    x = np.linspace(0, 10 * np.pi, config.n_samples)
    y = np.exp(-config.damped_sine_damping * x) * np.sin(config.sine_freq * x)
    return y.reshape(-1, 1)

def generate_variable_freq_sine(config: MDNTimeSeriesConfig) -> np.ndarray:
    """
    Generate a sine wave with changing frequency.

    Parameters
    ----------
    config : MDNTimeSeriesConfig
        Configuration parameters

    Returns
    -------
    np.ndarray
        Time series data of shape [n_samples, 1]
    """
    x = np.linspace(0, 10 * np.pi, config.n_samples)

    # Create segments with different frequencies
    segment_size = config.n_samples // 4
    y = np.zeros(config.n_samples)

    # First segment: low frequency
    y[:segment_size] = np.sin(0.5 * x[:segment_size])

    # Second segment: medium frequency
    y[segment_size:2*segment_size] = np.sin(1.0 * x[segment_size:2*segment_size])

    # Third segment: high frequency
    y[2*segment_size:3*segment_size] = np.sin(2.0 * x[2*segment_size:3*segment_size])

    # Fourth segment: mixed frequency
    y[3*segment_size:] = np.sin(1.5 * x[3*segment_size:])

    return y.reshape(-1, 1)

def generate_stock_price_sim(config: MDNTimeSeriesConfig) -> np.ndarray:
    """
    Simulate stock price data with trend, seasonality, and random shocks.

    Parameters
    ----------
    config : MDNTimeSeriesConfig
        Configuration parameters

    Returns
    -------
    np.ndarray
        Time series data of shape [n_samples, 1]
    """
    # Initialize price at 100
    price = 100
    prices = [price]

    for i in range(1, config.n_samples):
        # Random shock
        shock = np.random.normal(0, config.stock_volatility)

        # Seasonal component (weekly pattern)
        season = config.stock_seasonality * np.sin(2 * np.pi * i / 20)

        # Drift component (long-term trend)
        trend = config.stock_drift * i

        # Calculate new price
        price = price * (1 + shock + season + trend)
        prices.append(price)

    return np.array(prices).reshape(-1, 1)

# ---------------------------------------------------------------------
# Data Preprocessing Functions
# ---------------------------------------------------------------------

def create_windows(data: np.ndarray, config: MDNTimeSeriesConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series prediction.

    Parameters
    ----------
    data : np.ndarray
        Time series data of shape [n_samples, n_features]
    config : MDNTimeSeriesConfig
        Configuration parameters

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X: Input windows of shape [n_windows, window_size, n_features]
        y: Target values of shape [n_windows, n_features]
    """
    n_samples, n_features = data.shape
    X, y = [], []

    for i in range(0, n_samples - config.window_size - config.pred_horizon + 1, config.stride):
        X.append(data[i:i+config.window_size])
        y.append(data[i+config.window_size+config.pred_horizon-1])

    return np.array(X), np.array(y)

def normalize_data(train: np.ndarray,
                  test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize time series data using min-max scaling based on training set.

    Parameters
    ----------
    train : np.ndarray
        Training data
    test : Optional[np.ndarray]
        Test data

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        train_scaled: Normalized training data
        test_scaled: Normalized test data
        min_vals: Minimum values used for scaling
        max_vals: Maximum values used for scaling
    """
    min_vals = train.min(axis=0)
    max_vals = train.max(axis=0)

    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    train_scaled = (train - min_vals) / range_vals

    if test is not None:
        test_scaled = (test - min_vals) / range_vals
        return train_scaled, test_scaled, min_vals, max_vals

    return train_scaled, None, min_vals, max_vals

def inverse_normalize(scaled_data: np.ndarray,
                     min_vals: np.ndarray,
                     max_vals: np.ndarray) -> np.ndarray:
    """
    Convert normalized data back to original scale.

    Parameters
    ----------
    scaled_data : np.ndarray
        Normalized data
    min_vals : np.ndarray
        Minimum values used for scaling
    max_vals : np.ndarray
        Maximum values used for scaling

    Returns
    -------
    np.ndarray
        Data in original scale
    """
    # Ensure all inputs are numpy arrays
    scaled_data = ensure_numpy(scaled_data)
    min_vals = ensure_numpy(min_vals)
    max_vals = ensure_numpy(max_vals)

    range_vals = max_vals - min_vals
    return scaled_data * range_vals + min_vals

# ---------------------------------------------------------------------
# Model Building Function
# ---------------------------------------------------------------------

def build_mdn_model(input_shape: Tuple[int, int],
                   output_dimension: int,
                   config: MDNTimeSeriesConfig) -> Tuple[keras.Model, MDNLayer]:
    """
    Build an LSTM-based Mixture Density Network model for time series forecasting.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of input time windows (window_size, n_features)
    output_dimension : int
        Dimensionality of the output
    config : MDNTimeSeriesConfig
        Configuration parameters

    Returns
    -------
    Tuple[keras.Model, MDNLayer]
        Compiled MDN model for time series forecasting and the MDN layer instance
    """
    # Input layer
    inputs = keras.Input(shape=input_shape)

    # LSTM layers
    x = layers.LSTM(config.hidden_units[0],
                   return_sequences=len(config.hidden_units) > 1,
                   kernel_initializer=config.kernel_initializer,
                   kernel_regularizer=config.kernel_regularizer)(inputs)
    x = layers.Dropout(config.dropout_rate)(x)

    for i, units in enumerate(config.hidden_units[1:]):
        return_seq = i < len(config.hidden_units) - 2
        x = layers.LSTM(units,
                      return_sequences=return_seq,
                      kernel_initializer=config.kernel_initializer,
                      kernel_regularizer=config.kernel_regularizer)(x)
        x = layers.Dropout(config.dropout_rate)(x)

    # Dense layer before MDN
    x = layers.Dense(config.hidden_units[-1] // 2,
                   activation='relu',
                   kernel_initializer=config.kernel_initializer,
                   kernel_regularizer=config.kernel_regularizer)(x)

    # MDN Layer
    mdn_layer = MDNLayer(
        output_dimension=output_dimension,
        num_mixtures=config.num_mixtures,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=config.kernel_regularizer
    )
    outputs = mdn_layer(x)

    # Build model
    model = keras.Model(inputs, outputs)

    # Compile model with either the default or enhanced loss function
    if config.use_enhanced_loss:
        # Create a custom loss function that captures the mdn_layer
        def custom_loss(y_true, y_pred):
            return mdn_negative_log_likelihood(y_true, y_pred, mdn_layer)

        # Configure optimizer with gradient clipping if enabled
        if config.clip_gradients:
            optimizer = keras.optimizers.Adam(
                learning_rate=config.learning_rate,
                clipvalue=config.gradient_clip_value
            )
        else:
            optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

        model.compile(optimizer=optimizer, loss=custom_loss)
    else:
        # Use the default loss function from MDNLayer
        model.compile(optimizer='adam', loss=mdn_layer.loss_func)

    return model, mdn_layer

# ---------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------

def plot_time_series_with_predictions(y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     lower_bound: np.ndarray,
                                     upper_bound: np.ndarray,
                                     title: str,
                                     config: MDNTimeSeriesConfig,
                                     start_idx: int = 0,
                                     save_path: Optional[str] = None) -> None:
    """
    Plot time series data with predictions and uncertainty intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True values in original scale
    y_pred : np.ndarray
        Predicted values in original scale
    lower_bound : np.ndarray
        Lower bound of prediction interval in original scale
    upper_bound : np.ndarray
        Upper bound of prediction interval in original scale
    title : str
        Plot title
    config : MDNTimeSeriesConfig
        Configuration parameters
    start_idx : int
        Starting index for plotting
    save_path : Optional[str]
        Path to save the figure, if None the figure is displayed
    """
    plt.figure(figsize=(12, 6))

    # Ensure we don't exceed array bounds
    end_idx = min(start_idx + config.max_plot_points, len(y_true))
    x_indices = np.arange(start_idx, end_idx)

    # Plot true values
    plt.plot(x_indices, y_true[start_idx:end_idx], 'b-', label='True values', alpha=0.7)

    # Plot predictions
    plt.plot(x_indices, y_pred[start_idx:end_idx], 'r-', label='Predictions', alpha=0.7)

    # Plot prediction interval
    # Convert tensors to numpy arrays if needed
    if isinstance(lower_bound, tf.Tensor):
        lower_bound_np = lower_bound.numpy()
    else:
        lower_bound_np = lower_bound

    if isinstance(upper_bound, tf.Tensor):
        upper_bound_np = upper_bound.numpy()
    else:
        upper_bound_np = upper_bound

    plt.fill_between(x_indices,
                    lower_bound_np.reshape(-1)[start_idx:end_idx],
                    upper_bound_np.reshape(-1)[start_idx:end_idx],
                    color='red', alpha=0.2,
                    label=f'{int(config.confidence_level*100)}% Prediction Interval')

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_uncertainty_components(aleatoric_variance: np.ndarray,
                               epistemic_variance: np.ndarray,
                               title: str,
                               config: MDNTimeSeriesConfig,
                               start_idx: int = 0,
                               save_path: Optional[str] = None) -> None:
    """
    Plot decomposition of uncertainty into aleatoric and epistemic components.

    Parameters
    ----------
    aleatoric_variance : np.ndarray
        Aleatoric uncertainty component
    epistemic_variance : np.ndarray
        Epistemic uncertainty component
    title : str
        Plot title
    config : MDNTimeSeriesConfig
        Configuration parameters
    start_idx : int
        Starting index for plotting
    save_path : Optional[str]
        Path to save the figure, if None the figure is displayed
    """
    plt.figure(figsize=(12, 6))

    # Convert tensors to numpy if needed
    aleatoric_variance_np = ensure_numpy(aleatoric_variance)
    epistemic_variance_np = ensure_numpy(epistemic_variance)

    # Ensure we don't exceed array bounds
    end_idx = min(start_idx + config.max_plot_points, len(aleatoric_variance_np))
    x_indices = np.arange(start_idx, end_idx)

    total_variance = aleatoric_variance_np + epistemic_variance_np

    plt.plot(x_indices, aleatoric_variance_np[start_idx:end_idx], 'b-',
             label='Aleatoric Uncertainty (Data Noise)', alpha=0.7)
    plt.plot(x_indices, epistemic_variance_np[start_idx:end_idx], 'g-',
             label='Epistemic Uncertainty (Model Uncertainty)', alpha=0.7)
    plt.plot(x_indices, total_variance[start_idx:end_idx], 'r-',
             label='Total Uncertainty', alpha=0.7)

    plt.title(f"{title} - Uncertainty Decomposition")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time Step")
    plt.ylabel("Variance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_multi_step_samples(model: keras.Model,
                           mdn_layer: MDNLayer,
                           x_seed: np.ndarray,
                           config: MDNTimeSeriesConfig,
                           title: str = "Multi-step prediction samples",
                           save_path: Optional[str] = None) -> np.ndarray:
    """
    Generate and plot multiple multi-step prediction samples from the MDN model.

    Parameters
    ----------
    model : keras.Model
        Trained MDN model
    mdn_layer : MDNLayer
        The MDN layer instance used in the model
    x_seed : np.ndarray
        Initial window to start prediction from, shape [1, window_size, n_features]
    config : MDNTimeSeriesConfig
        Configuration parameters
    title : str
        Plot title
    save_path : Optional[str]
        Path to save the figure, if None the figure is displayed

    Returns
    -------
    np.ndarray
        Generated samples of shape [n_samples, n_steps, n_features]
    """
    # Create storage for samples
    window_size = x_seed.shape[1]
    n_features = x_seed.shape[2]
    n_steps = config.multi_step_forecast_steps
    n_samples = config.multi_step_samples

    samples = np.zeros((n_samples, n_steps, n_features))

    # Generate samples for each trajectory
    for i in range(n_samples):
        # Start with the seed window
        current_window = x_seed.copy()

        for t in range(n_steps):
            # Get prediction from current window
            pred = model.predict(current_window, verbose=0)

            # Sample from the predicted distribution
            sample = mdn_layer.sample(pred, seed=i*1000+t).numpy()

            # Store the sample
            samples[i, t] = sample

            # Update window for next prediction (roll and replace last value with new sample)
            current_window = np.roll(current_window, -1, axis=1)
            current_window[0, -1] = sample

    # Plot the samples
    plt.figure(figsize=(12, 6))

    # Time steps on x-axis
    x_indices = np.arange(n_steps)

    # Plot each sample trajectory
    for i in range(n_samples):
        plt.plot(x_indices, samples[i, :, 0], alpha=0.5, linewidth=1)

    # Plot the mean trajectory
    mean_trajectory = np.mean(samples, axis=0)
    plt.plot(x_indices, mean_trajectory[:, 0], 'r-', linewidth=2, label='Mean Trajectory')

    # Calculate prediction intervals from samples
    lower_bound = np.percentile(samples, (1 - config.confidence_level) * 50, axis=0)
    upper_bound = np.percentile(samples, (1 + config.confidence_level) * 50, axis=0)

    # Plot prediction intervals
    plt.fill_between(x_indices, lower_bound[:, 0], upper_bound[:, 0],
                    color='red', alpha=0.2,
                    label=f'{int(config.confidence_level*100)}% Prediction Interval')

    plt.title(title)
    plt.xlabel("Future Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return samples

# Function to monitor loss values during training
class LossMonitorCallback(keras.callbacks.Callback):
    """
    Callback to monitor loss values during training and check for negative values.
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Check if loss is negative
        if logs.get('loss', 0) < 0:
            print(f"\nWarning: Negative loss detected in epoch {epoch+1}: {logs.get('loss', 0)}")

        # Check if validation loss is negative
        if logs.get('val_loss', 0) < 0:
            print(f"\nWarning: Negative validation loss detected in epoch {epoch+1}: {logs.get('val_loss', 0)}")

# ---------------------------------------------------------------------
# Main Experiment Function
# ---------------------------------------------------------------------

def run_mdn_forecast_experiment(data_generator: Callable[[MDNTimeSeriesConfig], np.ndarray],
                               data_name: str,
                               config: MDNTimeSeriesConfig) -> Dict[str, Any]:
    """
    Run a complete MDN time series forecasting experiment.

    Parameters
    ----------
    data_generator : Callable
        Function to generate time series data
    data_name : str
        Name of the dataset for reporting
    config : MDNTimeSeriesConfig
        Configuration parameters

    Returns
    -------
    Dict[str, Any]
        Dictionary containing experiment results
    """
    print(f"\n{'='*50}")
    print(f"Running MDN experiment on {data_name} dataset")
    print(f"{'='*50}")

    # Create results directory if needed
    if config.save_results:
        os.makedirs(config.result_dir, exist_ok=True)
        exp_dir = os.path.join(config.result_dir, f"{data_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(exp_dir, exist_ok=True)

    # 1. Generate data
    data = data_generator(config)

    # 2. Split data into train/test/val
    train_size = int(config.train_ratio * len(data))
    val_size = int(config.val_ratio * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    # 3. Normalize data
    train_scaled, val_scaled, min_vals, max_vals = normalize_data(train_data, val_data)
    _, test_scaled, _, _ = normalize_data(train_data, test_data)

    # 4. Create windowed datasets
    X_train, y_train = create_windows(train_scaled, config)
    X_val, y_val = create_windows(val_scaled, config)
    X_test, y_test = create_windows(test_scaled, config)

    # 5. Build and compile model
    input_shape = (config.window_size, data.shape[1])
    output_dimension = data.shape[1]

    model, mdn_layer = build_mdn_model(
        input_shape=input_shape,
        output_dimension=output_dimension,
        config=config
    )

    # Display model architecture
    model.summary()

    # 6. Set up callbacks
    callbacks = [LossMonitorCallback()]  # Always add loss monitoring

    if config.use_early_stopping:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

    if config.save_results:
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(exp_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True
        )
        callbacks.append(model_checkpoint)

    # 7. Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Check if loss values went negative during training
    if any(loss < 0 for loss in history.history['loss']):
        print(f"\nWarning: Negative loss values detected during training on {data_name}")
        if not config.use_enhanced_loss:
            print("Consider setting use_enhanced_loss=True in the config to use the numerically stable loss function")

    # 8. Make predictions on test set
    y_pred = model.predict(X_test)

    # 9. Get point estimates
    point_estimates = get_point_estimate(model, X_test, mdn_layer)

    # 10. Get uncertainty estimates
    total_variance, aleatoric_variance = get_uncertainty(model, X_test, mdn_layer, point_estimates)

    # Calculate epistemic variance
    epistemic_variance = total_variance - aleatoric_variance

    # 11. Get prediction intervals
    lower_bound, upper_bound = get_prediction_intervals(point_estimates, total_variance,
                                                       confidence_level=config.confidence_level)

    # 12. Convert predictions back to original scale
    # First ensure all tensors are converted to numpy arrays
    y_test_np = ensure_numpy(y_test)
    point_estimates_np = ensure_numpy(point_estimates)
    lower_bound_np = ensure_numpy(lower_bound)
    upper_bound_np = ensure_numpy(upper_bound)

    y_test_orig = inverse_normalize(y_test_np, min_vals, max_vals)
    point_estimates_orig = inverse_normalize(point_estimates_np, min_vals, max_vals)
    lower_bound_orig = inverse_normalize(lower_bound_np, min_vals, max_vals)
    upper_bound_orig = inverse_normalize(upper_bound_np, min_vals, max_vals)

    # 13. Calculate metrics
    # Ensure all values are numpy arrays for metric calculations
    y_test_orig = ensure_numpy(y_test_orig)
    point_estimates_orig = ensure_numpy(point_estimates_orig)
    lower_bound_orig = ensure_numpy(lower_bound_orig)
    upper_bound_orig = ensure_numpy(upper_bound_orig)

    mse = np.mean((y_test_orig - point_estimates_orig) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_orig - point_estimates_orig))

    # Prediction interval coverage
    in_interval = np.logical_and(
        y_test_orig >= lower_bound_orig,
        y_test_orig <= upper_bound_orig
    )
    coverage = np.mean(in_interval)

    # Mean prediction interval width
    interval_width = np.mean(upper_bound_orig - lower_bound_orig)

    # 14. Visualize results
    if config.save_results:
        # Training history plot
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f"{data_name} - Training History")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(exp_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Predictions with uncertainty plot
        plot_time_series_with_predictions(
            y_test_orig, point_estimates_orig,
            lower_bound_orig, upper_bound_orig,
            f"{data_name} - Forecasting with {int(config.confidence_level*100)}% Prediction Intervals",
            config,
            save_path=os.path.join(exp_dir, 'predictions.png')
        )

        # Uncertainty decomposition plot
        plot_uncertainty_components(
            aleatoric_variance, epistemic_variance,
            data_name,
            config,
            save_path=os.path.join(exp_dir, 'uncertainty.png')
        )

        # Multi-step prediction samples
        plot_multi_step_samples(
            model, mdn_layer,
            X_test[:1], config,
            title=f"{data_name} - Multi-step prediction samples",
            save_path=os.path.join(exp_dir, 'multi_step_samples.png')
        )

        # Save metrics
        metrics = {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "Prediction Interval Coverage": float(coverage),
            "Mean Interval Width": float(interval_width),
            "Average Aleatoric Uncertainty": float(np.mean(aleatoric_variance)),
            "Average Epistemic Uncertainty": float(np.mean(epistemic_variance)),
            "Average Total Uncertainty": float(np.mean(total_variance))
        }

        with open(os.path.join(exp_dir, 'metrics.txt'), 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

    # Print metrics
    print(f"\nResults for {data_name} dataset:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Prediction Interval Coverage ({int(config.confidence_level*100)}%): {coverage:.2%}")
    print(f"Mean Prediction Interval Width: {interval_width:.6f}")
    print(f"Average Aleatoric Uncertainty: {np.mean(aleatoric_variance):.6f}")
    print(f"Average Epistemic Uncertainty: {np.mean(epistemic_variance):.6f}")

    # 15. Collect results
    results = {
        "data_name": data_name,
        "model": model,
        "mdn_layer": mdn_layer,
        "point_estimates": point_estimates_orig,
        "lower_bound": lower_bound_orig,
        "upper_bound": upper_bound_orig,
        "aleatoric_variance": aleatoric_variance,
        "epistemic_variance": epistemic_variance,
        "metrics": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "coverage": coverage,
            "interval_width": interval_width
        },
        "history": history.history
    }

    return results

# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main() -> None:
    """Main function to run the MDN time series forecasting experiments."""
    # Create configuration with enhanced numerical stability
    config = MDNTimeSeriesConfig(
        # Set the enhanced loss to true to address negative loss issues
        use_enhanced_loss=True,
        # Add gradient clipping for stability
        clip_gradients=True,
        gradient_clip_value=1.0,
        # Adjust learning rate
        learning_rate=0.001
    )

    # Define dataset generators (from simple to complex)
    datasets = [
        (lambda cfg: generate_sine_wave(cfg), "Sine_Wave"),
        (lambda cfg: generate_noisy_sine_wave(cfg), "Noisy_Sine_Wave"),
        (lambda cfg: generate_damped_sine_wave(cfg), "Damped_Sine_Wave"),
        (lambda cfg: generate_variable_freq_sine(cfg), "Variable_Frequency_Sine"),
        (lambda cfg: generate_stock_price_sim(cfg), "Stock_Price_Simulation")
    ]

    # Run experiments for each dataset
    all_results = {}

    for data_generator, data_name in datasets:
        results = run_mdn_forecast_experiment(
            data_generator=data_generator,
            data_name=data_name,
            config=config
        )

        all_results[data_name] = results

    # Compare results across datasets
    comparison_metrics = ["rmse", "mae", "coverage", "interval_width"]
    comparison_df = pd.DataFrame(
        {
            dataset_name: {
                metric: results["metrics"][metric]
                for metric in comparison_metrics
            }
            for dataset_name, results in all_results.items()
        }
    )

    print("\nComparison of results across datasets:")
    print(comparison_df)

    # Save comparison table
    comparison_df.to_csv(os.path.join(config.result_dir, "comparison_results.csv"))

    # Create comparison plot
    plt.figure(figsize=(12, 10))

    for i, metric in enumerate(comparison_metrics):
        plt.subplot(2, 2, i+1)
        bars = plt.bar(comparison_df.columns, comparison_df.loc[metric])
        plt.title(f"Comparison of {metric.upper()}")
        plt.xticks(rotation=45)
        plt.ylabel(metric.upper())
        plt.grid(True, alpha=0.3)

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(config.result_dir, "comparison_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nAll experiments completed. Results saved to {config.result_dir}")

if __name__ == "__main__":
    main()
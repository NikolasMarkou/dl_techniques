"""
Multi-Task Time Series Forecasting with a Single MDN Model

This experiment demonstrates training a single MDN model on multiple time series tasks
simultaneously. The model learns to handle different types of patterns (sine waves,
noisy data, stock prices) through multi-task learning, where task information is
embedded as additional features.

Key Features:
- Single MDN model trained on multiple tasks
- Task embeddings to help the model distinguish between different data types
- Simultaneous training with balanced sampling across tasks
- Individual evaluation on each task to assess multi-task performance
- Comprehensive uncertainty quantification for each task type
"""

import os
import sys
from pathlib import Path

# Suppress excessive TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass, field
from keras.api import regularizers
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from collections import defaultdict

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))

from dl_techniques.models.mdn_model import MDNModel
from dl_techniques.utils.logger import logger

# Set random seeds for reproducibility
keras.utils.set_random_seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class MultiTaskMDNConfig:
    """Configuration for multi-task MDN time series forecasting."""
    # General experiment config
    result_dir: str = "multitask_mdn_results"
    save_results: bool = True

    # Data config
    n_samples: int = 1000
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Windowing config
    window_size: int = 30
    pred_horizon: int = 1
    stride: int = 1

    # Model config
    num_mixtures: int = 5
    hidden_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    l2_regularization: float = 1e-5
    task_embedding_dim: int = 8  # Embedding dimension for task identifiers

    # Training config
    epochs: int = 150
    batch_size: int = 128
    early_stopping_patience: int = 20
    learning_rate: float = 0.001
    optimizer: str = 'adamw'

    # Multi-task training config
    task_balance_sampling: bool = True  # Whether to balance samples across tasks
    task_weight_decay: float = 0.95     # Decay factor for task weights

    # Prediction config
    confidence_level: float = 0.95
    num_forecast_samples: int = 50

    # Visualization config
    max_plot_points: int = 200

    # Time series generation config
    sine_freq: float = 0.1
    noisy_sine_noise_level: float = 0.15
    stock_initial_price: float = 100.0
    stock_drift: float = 0.05
    stock_volatility: float = 0.2

@dataclass
class TaskMetrics:
    """Metrics for a specific task."""
    task_name: str
    mse: float
    rmse: float
    mae: float
    coverage: float
    interval_width: float
    avg_aleatoric: float
    avg_epistemic: float
    samples_count: int

# ---------------------------------------------------------------------
# Multi-Task Data Utilities
# ---------------------------------------------------------------------

class TaskScaler:
    """Scaler that handles multiple tasks with separate scaling parameters."""

    def __init__(self):
        self.scalers = {}

    def fit(self, data: Dict[str, np.ndarray]):
        """Fit scalers for each task."""
        for task_name, task_data in data.items():
            scaler = Scaler()
            scaler.fit(task_data)
            self.scalers[task_name] = scaler

    def transform(self, task_name: str, data: np.ndarray) -> np.ndarray:
        """Transform data for a specific task."""
        return self.scalers[task_name].transform(data)

    def inverse_transform(self, task_name: str, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform data for a specific task."""
        return self.scalers[task_name].inverse_transform(scaled_data)

class Scaler:
    """Simple min-max scaler."""

    def __init__(self):
        self.min_, self.max_, self.range_ = None, None, None

    def fit(self, data: np.ndarray):
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min_) / self.range_

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        return scaled_data * self.range_ + self.min_

# ---------------------------------------------------------------------
# Data Generation Functions
# ---------------------------------------------------------------------

def generate_sine_wave(config: MultiTaskMDNConfig) -> np.ndarray:
    """Generate clean sine wave."""
    x = np.linspace(0, 20 * np.pi, config.n_samples)
    return np.sin(config.sine_freq * x).reshape(-1, 1)

def generate_noisy_sine_wave(config: MultiTaskMDNConfig) -> np.ndarray:
    """Generate noisy sine wave."""
    x = np.linspace(0, 20 * np.pi, config.n_samples)
    y = np.sin(config.sine_freq * x) + np.random.normal(0, config.noisy_sine_noise_level, config.n_samples)
    return y.reshape(-1, 1)

def generate_stock_price_gbm(config: MultiTaskMDNConfig) -> np.ndarray:
    """Generate stock price using Geometric Brownian Motion."""
    dt = 1 / 252
    mu = (config.stock_drift - 0.5 * config.stock_volatility**2) * dt
    sigma = config.stock_volatility * np.sqrt(dt)
    prices = np.zeros(config.n_samples)
    prices[0] = config.stock_initial_price
    shocks = np.random.normal(0, 1, config.n_samples)
    for t in range(1, config.n_samples):
        prices[t] = prices[t-1] * np.exp(mu + sigma * shocks[t])
    return prices.reshape(-1, 1)

# ---------------------------------------------------------------------
# Multi-Task Data Processing
# ---------------------------------------------------------------------

def create_windows_with_tasks(
    data: np.ndarray,
    task_id: int,
    config: MultiTaskMDNConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create windowed data with task identifiers."""
    X, y, task_ids = [], [], []

    for i in range(len(data) - config.window_size - config.pred_horizon + 1):
        X.append(data[i : i + config.window_size])
        y.append(data[i + config.window_size + config.pred_horizon - 1])
        task_ids.append(task_id)

    return np.array(X), np.array(y), np.array(task_ids)

def combine_task_data(
    task_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: MultiTaskMDNConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Combine data from multiple tasks."""
    all_X, all_y, all_task_ids = [], [], []
    task_indices = {}

    start_idx = 0
    for task_name, (X, y, task_ids) in task_data.items():
        all_X.append(X)
        all_y.append(y)
        all_task_ids.append(task_ids)

        # Store indices for this task
        end_idx = start_idx + len(X)
        task_indices[task_name] = np.arange(start_idx, end_idx)
        start_idx = end_idx

    combined_X = np.concatenate(all_X, axis=0)
    combined_y = np.concatenate(all_y, axis=0)
    combined_task_ids = np.concatenate(all_task_ids, axis=0)

    return combined_X, combined_y, combined_task_ids, task_indices

# ---------------------------------------------------------------------
# Multi-Task MDN Model
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiTaskMDNModel(keras.Model):
    """Multi-task MDN model with task embeddings."""

    def __init__(
        self,
        num_tasks: int,
        task_embedding_dim: int,
        hidden_layers: List[int],
        output_dimension: int,
        num_mixtures: int,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim
        self.hidden_layers_sizes = hidden_layers
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.kernel_regularizer = kernel_regularizer

        # Task embedding layer
        self.task_embedding = keras.layers.Embedding(
            input_dim=num_tasks,
            output_dim=task_embedding_dim,
            name="task_embedding"
        )

        # Feature extraction layers
        self.feature_layers = []
        self.mdn_model = None
        self._build_input_shape = None

        logger.info(f"Initialized MultiTaskMDNModel with {num_tasks} tasks")

    def build(self, input_shape: Tuple[Optional[int], ...]):
        """Build the multi-task model."""
        # input_shape: [(batch, window_size, features), (batch,)]
        sequence_shape, task_shape = input_shape
        self._build_input_shape = input_shape

        # Build task embedding
        self.task_embedding.build(task_shape)

        # Calculate combined feature size
        sequence_features = sequence_shape[-1] * sequence_shape[-2]  # Flattened sequence
        combined_features = sequence_features + self.task_embedding_dim

        # Create MDN model for the combined features
        self.mdn_model = MDNModel(
            hidden_layers=self.hidden_layers_sizes,
            output_dimension=self.output_dim,
            num_mixtures=self.num_mix,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            kernel_regularizer=self.kernel_regularizer
        )

        # Build MDN model with combined feature shape
        self.mdn_model.build((None, combined_features))

        super().build(input_shape)
        logger.info(f"MultiTaskMDNModel built with input shapes: {input_shape}")

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor], training: Optional[bool] = None):
        """Forward pass with task-aware processing."""
        sequence_input, task_input = inputs

        # Flatten sequence input
        batch_size = keras.ops.shape(sequence_input)[0]
        sequence_flat = keras.ops.reshape(sequence_input, (batch_size, -1))

        # Get task embeddings
        task_emb = self.task_embedding(task_input)

        # Combine sequence features with task embeddings
        combined_features = keras.ops.concatenate([sequence_flat, task_emb], axis=-1)

        # Pass through MDN model
        return self.mdn_model(combined_features, training=training)

    def get_mdn_layer(self):
        """Get the MDN layer for loss computation."""
        return self.mdn_model.mdn_layer

    def sample(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor], num_samples: int = 1, temp: float = 1.0):
        """Generate samples from the predicted distribution."""
        predictions = self(inputs, training=False)
        return self.mdn_model.mdn_layer.sample(predictions, temperature=temp)

    def get_config(self):
        """Get model configuration."""
        return {
            "num_tasks": self.num_tasks,
            "task_embedding_dim": self.task_embedding_dim,
            "hidden_layers": self.hidden_layers_sizes,
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        }

# ---------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------

class MultiTaskTrainer:
    """Trainer for multi-task MDN model."""

    def __init__(self, config: MultiTaskMDNConfig):
        self.config = config
        self.task_names = ["sine_wave", "noisy_sine", "stock_price"]
        self.task_generators = {
            "sine_wave": generate_sine_wave,
            "noisy_sine": generate_noisy_sine_wave,
            "stock_price": generate_stock_price_gbm
        }
        self.task_scalers = TaskScaler()

    def prepare_data(self) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Prepare multi-task training data."""
        logger.info("Preparing multi-task data...")

        # Generate raw data for all tasks
        raw_data = {}
        for task_name, generator in self.task_generators.items():
            raw_data[task_name] = generator(self.config)

        # Fit scalers
        self.task_scalers.fit(raw_data)

        # Split and scale data for each task
        task_data = {}
        for task_id, (task_name, data) in enumerate(raw_data.items()):
            # Split data
            train_size = int(self.config.train_ratio * len(data))
            val_size = int(self.config.val_ratio * len(data))

            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]

            # Scale data
            train_scaled = self.task_scalers.transform(task_name, train_data)
            val_scaled = self.task_scalers.transform(task_name, val_data)
            test_scaled = self.task_scalers.transform(task_name, test_data)

            # Create windows
            train_windows = create_windows_with_tasks(train_scaled, task_id, self.config)
            val_windows = create_windows_with_tasks(val_scaled, task_id, self.config)
            test_windows = create_windows_with_tasks(test_scaled, task_id, self.config)

            task_data[task_name] = {
                "train": train_windows,
                "val": val_windows,
                "test": test_windows
            }

        return task_data

    def create_model(self) -> MultiTaskMDNModel:
        """Create the multi-task MDN model."""
        model = MultiTaskMDNModel(
            num_tasks=len(self.task_names),
            task_embedding_dim=self.config.task_embedding_dim,
            hidden_layers=self.config.hidden_units,
            output_dimension=1,  # Single output dimension for time series
            num_mixtures=self.config.num_mixtures,
            dropout_rate=self.config.dropout_rate,
            use_batch_norm=self.config.use_batch_norm,
            kernel_regularizer=regularizers.l2(self.config.l2_regularization)
        )

        return model

    def train_model(self, model: MultiTaskMDNModel, task_data: Dict) -> Dict[str, Any]:
        """Train the multi-task model."""
        logger.info("Training multi-task MDN model...")

        # Combine training data
        train_data = {name: data["train"] for name, data in task_data.items()}
        val_data = {name: data["val"] for name, data in task_data.items()}

        X_train, y_train, task_ids_train, train_indices = combine_task_data(train_data, self.config)
        X_val, y_val, task_ids_val, val_indices = combine_task_data(val_data, self.config)

        # Build model
        model.build([(None, self.config.window_size, 1), (None,)])

        # Compile model
        optimizer = keras.optimizers.get(self.config.optimizer)
        optimizer.learning_rate = self.config.learning_rate

        model.compile(
            optimizer=optimizer,
            loss=model.get_mdn_layer().loss_func
        )

        # Print model summary
        model.summary(print_fn=lambda x: logger.info(x))

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]

        # Train model
        history = model.fit(
            [X_train, task_ids_train],
            y_train,
            validation_data=([X_val, task_ids_val], y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return {"history": history.history, "train_indices": train_indices, "val_indices": val_indices}

    def evaluate_model(self, model: MultiTaskMDNModel, task_data: Dict) -> Dict[str, TaskMetrics]:
        """Evaluate model on each task separately."""
        logger.info("Evaluating multi-task model...")

        task_metrics = {}

        for task_id, (task_name, data) in enumerate(task_data.items()):
            logger.info(f"Evaluating on task: {task_name}")

            X_test, y_test, task_ids_test = data["test"]

            # Make predictions
            predictions = model.predict([X_test, task_ids_test])

            # Get uncertainty estimates
            mdn_layer = model.get_mdn_layer()
            mu, sigma, pi = mdn_layer.split_mixture_params(predictions)

            # Convert to probabilities
            pi_probs = keras.activations.softmax(pi, axis=-1)

            # Calculate point estimates (weighted mean)
            pi_expanded = keras.ops.expand_dims(pi_probs, axis=-1)
            point_estimates = keras.ops.sum(mu * pi_expanded, axis=1)

            # Calculate variances
            point_expanded = keras.ops.expand_dims(point_estimates, axis=1)
            aleatoric_var = keras.ops.sum(pi_expanded * sigma**2, axis=1)
            epistemic_var = keras.ops.sum(pi_expanded * (mu - point_expanded)**2, axis=1)
            total_var = aleatoric_var + epistemic_var

            # Convert to numpy and inverse transform
            point_estimates_np = keras.ops.convert_to_numpy(point_estimates)
            total_var_np = keras.ops.convert_to_numpy(total_var)
            aleatoric_var_np = keras.ops.convert_to_numpy(aleatoric_var)
            epistemic_var_np = keras.ops.convert_to_numpy(epistemic_var)

            # Inverse transform predictions
            point_estimates_orig = self.task_scalers.inverse_transform(task_name, point_estimates_np)
            y_test_orig = self.task_scalers.inverse_transform(task_name, y_test)

            # Scale variances
            scale_factor = self.task_scalers.scalers[task_name].range_**2
            total_var_orig = total_var_np * scale_factor
            aleatoric_var_orig = aleatoric_var_np * scale_factor
            epistemic_var_orig = epistemic_var_np * scale_factor

            # Calculate prediction intervals
            z_score = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
            std_dev = np.sqrt(total_var_orig)
            lower_bound = point_estimates_orig - z_score * std_dev
            upper_bound = point_estimates_orig + z_score * std_dev

            # Calculate metrics
            mse = np.mean((y_test_orig - point_estimates_orig)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test_orig - point_estimates_orig))
            coverage = np.mean((y_test_orig >= lower_bound) & (y_test_orig <= upper_bound))
            interval_width = np.mean(upper_bound - lower_bound)

            task_metrics[task_name] = TaskMetrics(
                task_name=task_name,
                mse=mse,
                rmse=rmse,
                mae=mae,
                coverage=coverage,
                interval_width=interval_width,
                avg_aleatoric=np.mean(aleatoric_var_orig),
                avg_epistemic=np.mean(epistemic_var_orig),
                samples_count=len(y_test_orig)
            )

            logger.info(f"Task {task_name} - RMSE: {rmse:.4f}, Coverage: {coverage:.4f}")

        return task_metrics

    def visualize_results(self, model: MultiTaskMDNModel, task_data: Dict, save_dir: str):
        """Create visualizations for each task."""
        logger.info("Creating visualizations...")

        os.makedirs(save_dir, exist_ok=True)

        for task_id, (task_name, data) in enumerate(task_data.items()):
            X_test, y_test, task_ids_test = data["test"]

            # Make predictions
            predictions = model.predict([X_test, task_ids_test])

            # Get point estimates and uncertainties
            mdn_layer = model.get_mdn_layer()
            mu, sigma, pi = mdn_layer.split_mixture_params(predictions)
            pi_probs = keras.activations.softmax(pi, axis=-1)
            pi_expanded = keras.ops.expand_dims(pi_probs, axis=-1)
            point_estimates = keras.ops.sum(mu * pi_expanded, axis=1)

            # Calculate total variance
            point_expanded = keras.ops.expand_dims(point_estimates, axis=1)
            total_var = keras.ops.sum(pi_expanded * (sigma**2 + (mu - point_expanded)**2), axis=1)

            # Convert to numpy and inverse transform
            point_estimates_np = keras.ops.convert_to_numpy(point_estimates)
            total_var_np = keras.ops.convert_to_numpy(total_var)

            point_estimates_orig = self.task_scalers.inverse_transform(task_name, point_estimates_np)
            y_test_orig = self.task_scalers.inverse_transform(task_name, y_test)

            # Scale variance
            scale_factor = self.task_scalers.scalers[task_name].range_**2
            total_var_orig = total_var_np * scale_factor

            # Calculate prediction intervals
            z_score = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
            std_dev = np.sqrt(total_var_orig)
            lower_bound = point_estimates_orig - z_score * std_dev
            upper_bound = point_estimates_orig + z_score * std_dev

            # Create plot
            plt.figure(figsize=(15, 8))

            # Limit plot points for clarity
            plot_len = min(len(y_test_orig), self.config.max_plot_points)
            x_indices = np.arange(plot_len)

            plt.plot(x_indices, y_test_orig[:plot_len], 'b-', label='True Values', linewidth=2)
            plt.plot(x_indices, point_estimates_orig[:plot_len], 'r-', label='Predictions', linewidth=2)
            plt.fill_between(
                x_indices,
                lower_bound[:plot_len].flatten(),
                upper_bound[:plot_len].flatten(),
                color='red', alpha=0.2, label=f'{int(self.config.confidence_level*100)}% Prediction Interval'
            )

            plt.title(f'Multi-Task MDN Predictions - {task_name.replace("_", " ").title()}', fontsize=16)
            plt.xlabel('Time Steps', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            plt.savefig(os.path.join(save_dir, f'{task_name}_predictions.png'), dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved visualization for {task_name}")

# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def main():
    """Main function to run the multi-task MDN experiment."""
    config = MultiTaskMDNConfig()

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(config.result_dir, f"multitask_mdn_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"Starting multi-task MDN experiment. Results will be saved to: {exp_dir}")

    # Initialize trainer
    trainer = MultiTaskTrainer(config)

    try:
        # Prepare data
        task_data = trainer.prepare_data()
        logger.info(f"Data prepared for {len(task_data)} tasks")

        # Create model
        model = trainer.create_model()
        logger.info("Multi-task MDN model created")

        # Train model
        training_results = trainer.train_model(model, task_data)
        logger.info("Model training completed")

        # Evaluate model on each task
        task_metrics = trainer.evaluate_model(model, task_data)

        # Create visualizations
        trainer.visualize_results(model, task_data, exp_dir)

        # Print results
        print("\n" + "="*80)
        print("MULTI-TASK MDN RESULTS")
        print("="*80)

        # Create results dataframe
        results_data = []
        for task_name, metrics in task_metrics.items():
            results_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'RMSE': metrics.rmse,
                'MAE': metrics.mae,
                'Coverage': metrics.coverage,
                'Interval Width': metrics.interval_width,
                'Aleatoric Unc.': metrics.avg_aleatoric,
                'Epistemic Unc.': metrics.avg_epistemic,
                'Samples': metrics.samples_count
            })

        results_df = pd.DataFrame(results_data)
        print("\nPer-Task Performance:")
        print(results_df.to_string(index=False, float_format='%.4f'))

        # Calculate aggregate metrics
        total_samples = sum(m.samples_count for m in task_metrics.values())
        weighted_rmse = sum(m.rmse * m.samples_count for m in task_metrics.values()) / total_samples
        weighted_coverage = sum(m.coverage * m.samples_count for m in task_metrics.values()) / total_samples
        avg_aleatoric = np.mean([m.avg_aleatoric for m in task_metrics.values()])
        avg_epistemic = np.mean([m.avg_epistemic for m in task_metrics.values()])

        print(f"\nAggregate Metrics:")
        print(f"Weighted RMSE: {weighted_rmse:.4f}")
        print(f"Weighted Coverage: {weighted_coverage:.4f}")
        print(f"Average Aleatoric Uncertainty: {avg_aleatoric:.4f}")
        print(f"Average Epistemic Uncertainty: {avg_epistemic:.4f}")

        # Save results
        if config.save_results:
            # Save metrics
            results_df.to_csv(os.path.join(exp_dir, 'task_metrics.csv'), index=False)

            # Save aggregate metrics
            with open(os.path.join(exp_dir, 'aggregate_metrics.txt'), 'w') as f:
                f.write(f"Weighted RMSE: {weighted_rmse:.4f}\n")
                f.write(f"Weighted Coverage: {weighted_coverage:.4f}\n")
                f.write(f"Average Aleatoric Uncertainty: {avg_aleatoric:.4f}\n")
                f.write(f"Average Epistemic Uncertainty: {avg_epistemic:.4f}\n")

            # Save model
            model.save(os.path.join(exp_dir, 'multitask_mdn_model.keras'))

            # Save training history
            history_df = pd.DataFrame(training_results['history'])
            history_df.to_csv(os.path.join(exp_dir, 'training_history.csv'), index=False)

            logger.info(f"Results saved to {exp_dir}")

        # Plot training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(training_results['history']['loss'], label='Training Loss')
        plt.plot(training_results['history']['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.bar(results_df['Task'], results_df['RMSE'])
        plt.title('RMSE by Task')
        plt.xlabel('Task')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'training_summary.png'), dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
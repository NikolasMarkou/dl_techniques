"""
Orchestrate end-to-end training and evaluation of the Multi-Task MDN framework.

Architecture
------------
The Multi-Task MDN (Mixture Density Network) framework combines a task-aware
embedding system with a deep feature extractor and a probabilistic output layer.
It allows for:
1.  Probabilistic Forecasting (predicting distribution parameters)
2.  Multi-Task Learning (handling distinct time series patterns simultaneously)
3.  Uncertainty Quantification (Aleatoric vs Epistemic)

This script implements a robust training pipeline supporting:
1.  Infinite streaming data generation.
2.  In-memory validation caching for performance.
3.  Evaluation of probabilistic calibration.

Integration
-----------
Uses the unified `dl_techniques.datasets.time_series` module for
synthetic data generation and normalization.

References
----------
1.  Mixture Density Networks (Bishop, 1994)
2.  Deep Multi-Task Learning for Time Series
3.  Uncertainty in Deep Learning (Gal, 2016)
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from scipy import stats

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.mdn_model.model import MDNModel

from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator,
    TimeSeriesGeneratorConfig,
    TimeSeriesNormalizer,
    NormalizationMethod
)

# ---------------------------------------------------------------------

plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for major libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class MDNTrainingConfig:
    """Configuration dataclass for Multi-Task MDN training."""
    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "mdn_multitask"

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Sequence Configuration
    window_size: int = 120
    pred_horizon: int = 1  # MDN typically predicts next step(s) distribution
    stride: int = 1

    # Model Architecture
    num_mixtures: int = 12
    hidden_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    task_embedding_dim: int = 32
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    use_attention: bool = True
    attention_heads: int = 4
    attention_dim: int = 64

    # Calibration
    use_temperature_scaling: bool = True
    initial_temperature: float = 1.0
    calibration_weight: float = 0.1

    # Training configuration
    epochs: int = 100
    batch_size: int = 256
    steps_per_epoch: int = 200
    learning_rate: float = 5e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adamw'
    weight_decay: float = 1e-4

    # Pattern selection and Normalization
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 10
    normalize_per_instance: bool = True

    # Prediction / Viz
    confidence_level: float = 0.95
    num_forecast_samples: int = 100
    visualize_every_n_epochs: int = 5
    plot_top_k_patterns: int = 9

    def __post_init__(self) -> None:
        """Validate configuration."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")

        if self.window_size <= 0:
            raise ValueError("window_size must be positive")


# ---------------------------------------------------------------------
# Wrapper Model for Multi-Task Support
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiTaskMDNModel(keras.Model):
    """
    Multi-task wrapper around the core MDNModel.

    Handles:
    1. Task Embeddings
    2. Sequence feature extraction (Conv1D + Attention)
    3. Injection into the MDN head
    """
    def __init__(
        self,
        num_tasks: int,
        config: MDNTrainingConfig,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.num_tasks = num_tasks

        # 1. Task Embedding
        self.task_embedding = keras.layers.Embedding(
            input_dim=num_tasks,
            output_dim=config.task_embedding_dim,
            name="task_embedding"
        )

        # 2. Sequence Processing
        self.conv1 = keras.layers.Conv1D(64, 7, padding="same", activation="gelu")
        self.norm1 = keras.layers.LayerNormalization()
        self.conv2 = keras.layers.Conv1D(128, 5, padding="same", activation="gelu")
        self.norm2 = keras.layers.LayerNormalization()

        # 3. Attention (Optional)
        if config.use_attention:
            self.attention = keras.layers.MultiHeadAttention(
                num_heads=config.attention_heads,
                key_dim=config.attention_dim,
                name="seq_attention"
            )
            self.att_norm = keras.layers.LayerNormalization()

        self.flatten = keras.layers.Flatten()

        # 4. Core MDN
        # We calculate input dimension dynamically in build, but MDNModel takes feature vector
        self.mdn_core = MDNModel(
            hidden_layers=config.hidden_units,
            output_dimension=1,  # Univariate target per task
            num_mixtures=config.num_mixtures,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm
        )

        # 5. Calibration
        if config.use_temperature_scaling:
            self.temperature = self.add_weight(
                name="temperature",
                shape=(),
                initializer=keras.initializers.Constant(config.initial_temperature),
                trainable=True
            )

    def build(self, input_shape):
        # input_shape is ((batch, win, 1), (batch, 1))
        # Note: input_shape may come in as a list or tuple
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: tuple of (sequence, task_id)
        sequence_input, task_input = inputs

        # A. Process Sequence
        x = self.conv1(sequence_input)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.config.use_attention:
            att = self.attention(x, x)
            x = self.att_norm(x + att)

        seq_features = self.flatten(x)

        # B. Process Task
        # task_input shape: (batch, 1) or (batch,)
        task_emb = self.task_embedding(task_input)

        # If task_input was (batch, 1), output is (batch, 1, dim) -> needs squeeze
        # If task_input was (batch,), output is (batch, dim) -> no squeeze needed
        # We handle this robustly by checking rank.
        if len(task_emb.shape) == 3:
            task_emb = keras.ops.squeeze(task_emb, axis=1) # Ensure flat vector (batch, dim)

        # C. Combine
        combined = keras.ops.concatenate([seq_features, task_emb], axis=-1)

        # D. MDN Prediction
        outputs = self.mdn_core(combined, training=training)

        return outputs

    def get_mdn_layer(self):
        return self.mdn_core.mdn_layer


# ---------------------------------------------------------------------
# Data Processor
# ---------------------------------------------------------------------

class MDNDataProcessor:
    """
    Robust data processor for Multi-Task MDN.

    Features:
    - Maps diverse patterns to specific Task IDs.
    - Infinite streaming generator for Training.
    - Pre-computed in-memory arrays for Validation/Test.
    - Robust normalization.
    """

    def __init__(
        self,
        config: MDNTrainingConfig,
        generator: TimeSeriesGenerator,
        selected_patterns: List[str]
    ):
        self.config = config
        self.ts_generator = generator
        self.selected_patterns = selected_patterns

        # Create Task ID mapping
        self.pattern_to_id = {name: i for i, name in enumerate(selected_patterns)}
        self.id_to_pattern = {i: name for name, i in self.pattern_to_id.items()}
        self.num_tasks = len(selected_patterns)

        logger.info(f"Initialized Processor with {self.num_tasks} tasks.")

    def _safe_normalize(self, series: np.ndarray) -> np.ndarray:
        """Robust normalization and clipping."""
        # Clip infinities
        series = np.clip(series, -1e6, 1e6)

        if self.config.normalize_per_instance:
            normalizer = TimeSeriesNormalizer(method=NormalizationMethod.ROBUST)
            if np.isnan(series).any():
                series = _fill_nans(series)
            series = normalizer.fit_transform(series)

        # NN stability clipping
        series = np.clip(series, -10.0, 10.0)
        return series.astype(np.float32)

    def _training_generator(self) -> Generator[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray], None, None]:
        """
        Infinite generator yielding ((sequence, task_id), target).
        """
        patterns_to_mix = 50
        windows_per_pattern = 10
        buffer = []

        total_len = self.config.window_size + self.config.pred_horizon

        while True:
            if not buffer:
                # Select random mix of tasks
                batch_patterns = random.choices(self.selected_patterns, k=patterns_to_mix)

                for pattern_name in batch_patterns:
                    task_id = self.pattern_to_id[pattern_name]

                    # Generate Data
                    try:
                        data = self.ts_generator.generate_task_data(pattern_name)
                    except Exception:
                        continue # Skip failed generations

                    if len(data) < total_len * 2:
                        continue

                    # Train split only
                    train_size = int(self.config.train_ratio * len(data))
                    train_data = data[:train_size]

                    if len(train_data) < total_len:
                        continue

                    # Extract windows
                    max_start = len(train_data) - total_len
                    for _ in range(windows_per_pattern):
                        start_idx = random.randint(0, max_start)
                        window_slice = train_data[start_idx : start_idx + total_len]

                        # Normalize
                        window_slice = self._safe_normalize(window_slice)

                        # Split inputs/targets
                        # X: [0 : window_size]
                        # Y: [window_size] (assuming horizon 1)
                        x_seq = window_slice[:self.config.window_size].reshape(-1, 1)
                        y_target = window_slice[self.config.window_size:].reshape(-1) # shape (1,)

                        # Yield task_id as array of shape (1,)
                        buffer.append(((x_seq, np.array([task_id], dtype=np.int32)), y_target))

                random.shuffle(buffer)

            if buffer:
                yield buffer.pop()

    def _generate_fixed_dataset(self, split: str, num_samples: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Pre-compute dataset for Val/Test."""
        logger.info(f"Pre-computing {split} dataset ({num_samples} samples)...")

        x_seqs, x_tasks, y_targets = [], [], []
        total_len = self.config.window_size + self.config.pred_horizon

        samples_collected = 0
        pattern_cycle = 0

        while samples_collected < num_samples:
            # Deterministic pattern cycling
            pattern_idx = pattern_cycle % self.num_tasks
            pattern_name = self.selected_patterns[pattern_idx]
            pattern_cycle += 1
            task_id = self.pattern_to_id[pattern_name]

            try:
                data = self.ts_generator.generate_task_data(pattern_name)
            except Exception:
                continue

            # Splitting
            train_end = int(self.config.train_ratio * len(data))
            val_end = train_end + int(self.config.val_ratio * len(data))

            if split == 'val':
                split_data = data[train_end:val_end]
            else:
                split_data = data[val_end:]

            if len(split_data) < total_len:
                continue

            max_start = len(split_data) - total_len
            start_idx = random.randint(0, max_start)

            window_slice = split_data[start_idx : start_idx + total_len]
            window_slice = self._safe_normalize(window_slice)

            x_seq = window_slice[:self.config.window_size].reshape(-1, 1)
            y_tgt = window_slice[self.config.window_size:].reshape(-1)

            x_seqs.append(x_seq)
            # Ensure task ID is a list/vector [task_id] so slicing returns (1,)
            x_tasks.append([task_id])
            y_targets.append(y_tgt)
            samples_collected += 1

        # Return numpy arrays.
        # x_tasks will be (N, 1) to match training generator's (Batch, 1) behavior after batching
        return (np.array(x_seqs, dtype=np.float32), np.array(x_tasks, dtype=np.int32)), np.array(y_targets, dtype=np.float32)

    def prepare_datasets(self) -> Dict[str, Any]:
        """Create tf.data.Datasets."""

        # Output Signature: ((Sequence, TaskID), Target)
        output_sig = (
            (
                tf.TensorSpec(shape=(self.config.window_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.int32)
            ),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )

        train_ds = tf.data.Dataset.from_generator(
            self._training_generator, output_signature=output_sig
        ).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        validation_steps = max(50, self.num_tasks)
        test_steps = max(20, self.num_tasks)

        # Generate validation data
        val_inputs, val_y = self._generate_fixed_dataset('val', validation_steps * self.config.batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_y))
        val_ds = val_ds.batch(self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        # Generate test data
        test_inputs, test_y = self._generate_fixed_dataset('test', test_steps * self.config.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((test_inputs, test_y))
        test_ds = test_ds.batch(self.config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        return {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
            'validation_steps': validation_steps,
            'test_steps': test_steps,
            'test_data_raw': (test_inputs, test_y) # Store for visualization
        }

def _fill_nans(data: np.ndarray) -> np.ndarray:
    mask = np.isnan(data)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return data[idx]


# ---------------------------------------------------------------------
# Performance Callback & Visualization
# ---------------------------------------------------------------------

class MDNPerformanceCallback(keras.callbacks.Callback):
    """
    Custom callback for tracking and visualizing MDN performance.
    Plots probabilistic forecasts (intervals) and learning curves.
    """

    def __init__(
            self,
            config: MDNTrainingConfig,
            save_dir: str,
            viz_data: Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
    ):
        super().__init__()
        self.config = config
        self.save_dir = save_dir
        self.viz_inputs, self.viz_targets = viz_data

        os.makedirs(self.save_dir, exist_ok=True)

        self.history_log = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}
        self.history_log['loss'].append(logs.get('loss', 0))
        self.history_log['val_loss'].append(logs.get('val_loss', 0))

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(f"Creating visualizations for epoch {epoch + 1}...")
            self._plot_learning_curves(epoch)
            self._plot_probabilistic_predictions(epoch)

    def _plot_learning_curves(self, epoch: int) -> None:
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.history_log['loss']) + 1)
        plt.plot(epochs, self.history_log['loss'], label='Train Loss')
        plt.plot(epochs, self.history_log['val_loss'], label='Val Loss')
        plt.title('MDN Negative Log-Likelihood')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (NLL)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'learning_curves_epoch_{epoch+1:03d}.png'))
        plt.close()

    def _plot_probabilistic_predictions(self, epoch: int) -> None:
        """
        Visualize MDN outputs: Context, Target, and Predicted Distribution (Median + Interval).
        """
        # Select random subset for viz
        total_samples = len(self.viz_targets)
        indices = np.random.choice(total_samples, min(self.config.plot_top_k_patterns, total_samples), replace=False)

        # Prepare batch
        # viz_inputs is ((N, win, 1), (N, 1))
        sample_seq = self.viz_inputs[0][indices]
        sample_task = self.viz_inputs[1][indices]
        sample_target = self.viz_targets[indices]

        # Predict
        # Output is MDN params: [mu..., sigma..., pi...]
        params = self.model.predict((sample_seq, sample_task), verbose=0)

        mdn_layer = self.model.get_mdn_layer()

        # Decompose parameters
        mus, sigmas, pis = mdn_layer.split_mixture_params(params)

        # Convert to numpy
        mus = keras.ops.convert_to_numpy(mus)     # (batch, num_mix)
        sigmas = keras.ops.convert_to_numpy(sigmas) # (batch, num_mix)
        pis = keras.ops.convert_to_numpy(pis)       # (batch, num_mix)
        pis = np.exp(pis) / np.sum(np.exp(pis), axis=1, keepdims=True) # Softmax

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= len(indices): break

            # 1. Plot Context
            ctx = sample_seq[i].flatten()
            tgt = sample_target[i]
            time_steps = np.arange(len(ctx))
            future_step = len(ctx)

            ax.plot(time_steps, ctx, label='Context', color='blue', alpha=0.6)
            ax.scatter([future_step], [tgt], label='True Target', color='green', marker='x', s=100, zorder=5)

            # 2. Reconstruct Density
            # Create a grid of y values to evaluate PDF
            y_min = min(ctx.min(), float(tgt)) - 2.0
            y_max = max(ctx.max(), float(tgt)) + 2.0
            y_grid = np.linspace(y_min, y_max, 200)

            # Calculate Mixture PDF: sum(pi * N(y|mu, sigma))
            pdf_values = np.zeros_like(y_grid)
            for k in range(self.config.num_mixtures):
                w = pis[i, k]
                mu = mus[i, k]
                sigma = sigmas[i, k]
                pdf_values += w * stats.norm.pdf(y_grid, mu, sigma)

            # Normalize PDF for plotting scale (arbitrary scaling for visual fit)
            max_pdf = pdf_values.max()
            if max_pdf > 1e-6:
                scale_factor = (ctx.max() - ctx.min()) * 0.5 / max_pdf
            else:
                scale_factor = 1.0

            # Plot PDF 'sideways'
            # To visualize density at the prediction step, we plot probability mass
            # We can just plot percentiles to keep it clean.

            # Calculate CDF to find percentiles
            cdf_values = np.cumsum(pdf_values)
            if cdf_values[-1] > 0:
                cdf_values /= cdf_values[-1]

                idx_05 = np.searchsorted(cdf_values, 0.05)
                idx_50 = np.searchsorted(cdf_values, 0.50)
                idx_95 = np.searchsorted(cdf_values, 0.95)

                # Bounds check
                idx_05 = min(max(idx_05, 0), len(y_grid)-1)
                idx_50 = min(max(idx_50, 0), len(y_grid)-1)
                idx_95 = min(max(idx_95, 0), len(y_grid)-1)

                lower_bound = y_grid[idx_05]
                median_pred = y_grid[idx_50]
                upper_bound = y_grid[idx_95]

                # Plot Forecast
                ax.scatter([future_step], [median_pred], label='Median Pred', color='red', alpha=0.8)
                ax.errorbar([future_step], [median_pred],
                            yerr=[[median_pred - lower_bound], [upper_bound - median_pred]],
                            fmt='none', ecolor='red', alpha=0.3, capsize=5, label='90% CI')

            ax.set_title(f'Sample {i} (Task {sample_task[i][0]})')
            if i == 0: ax.legend(loc='upper left', fontsize='small')

        plt.suptitle(f'MDN Probabilistic Forecasts (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch+1:03d}.png'))
        plt.close()


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------

class MDNTrainer:
    """Trainer for Multi-Task MDN using robust data handling."""

    def __init__(
        self,
        config: MDNTrainingConfig,
        generator_config: TimeSeriesGeneratorConfig
    ) -> None:
        self.config = config
        self.generator = TimeSeriesGenerator(generator_config)

        # Pattern Selection
        all_patterns = self.generator.get_task_names()

        # Filter logic similar to PRISM (pick subset if needed)
        if config.max_patterns:
            self.selected_patterns = random.sample(all_patterns, config.max_patterns)
        else:
            self.selected_patterns = all_patterns

        self.processor = MDNDataProcessor(config, self.generator, self.selected_patterns)
        self.model: Optional[MultiTaskMDNModel] = None

    def run_experiment(self) -> Dict[str, Any]:
        logger.info("Starting Multi-Task MDN experiment")
        self.exp_dir = self._create_experiment_dir()

        # 1. Data
        data_pipeline = self.processor.prepare_datasets()

        # 2. Model
        self.model = self._build_model(self.processor.num_tasks)

        # 3. Train
        training_results = self._train_model(data_pipeline, self.exp_dir)

        if self.config.save_results:
            self._save_results(training_results, self.exp_dir)

        return {
            'config': self.config,
            'experiment_dir': self.exp_dir,
            'results': training_results
        }

    def _create_experiment_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config.experiment_name}_{timestamp}"
        exp_dir = os.path.join(self.config.result_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def _build_model(self, num_tasks: int) -> MultiTaskMDNModel:
        logger.info(f"Building Multi-Task MDN model for {num_tasks} tasks")

        model = MultiTaskMDNModel(num_tasks, self.config)

        # Build to init weights
        # Ensure inputs match the pipeline types/shapes
        dummy_seq = tf.zeros((1, self.config.window_size, 1))
        dummy_task = tf.zeros((1, 1), dtype=tf.int32)
        model((dummy_seq, dummy_task))

        # Optimizer
        optimizer = keras.optimizers.get(self.config.optimizer)
        optimizer.learning_rate = self.config.learning_rate
        if self.config.gradient_clip_norm:
            optimizer.clipnorm = self.config.gradient_clip_norm

        # Loss: The MDN Core layer has the loss function
        # We need to construct a wrapper function that might handle calibration
        def mdn_loss_wrapper(y_true, y_pred):
            # Standard NLL
            base_loss = model.get_mdn_layer().loss_func(y_true, y_pred)

            # Calibration penalty (Temperature)
            if self.config.use_temperature_scaling:
                temp_penalty = keras.ops.square(model.temperature - 1.0) * self.config.calibration_weight
                return base_loss + temp_penalty
            return base_loss

        model.compile(
            optimizer=optimizer,
            loss=mdn_loss_wrapper
        )

        model.summary(print_fn=logger.info)
        return model

    def _train_model(self, data_pipeline: Dict[str, Any], exp_dir: str) -> Dict[str, Any]:
        viz_dir = os.path.join(exp_dir, 'visualizations')

        callbacks = [
            MDNPerformanceCallback(self.config, viz_dir, data_pipeline['test_data_raw']),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(exp_dir, 'best_model.keras'),
                monitor='val_loss', save_best_only=True, verbose=1
            ),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            )
        ]

        history = self.model.fit(
            data_pipeline['train_ds'],
            validation_data=data_pipeline['val_ds'],
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_steps=data_pipeline['validation_steps'],
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Evaluating on test set...")
        test_metrics = self.model.evaluate(
            data_pipeline['test_ds'],
            steps=data_pipeline['test_steps'],
            return_dict=True
        )

        return {
            'history': history.history,
            'test_metrics': test_metrics
        }

    def _save_results(self, results: Dict, exp_dir: str) -> None:
        def json_convert(o):
            if isinstance(o, (np.floating, np.integer)): return str(o)
            return str(o)

        serializable = {
            'history': results['history'],
            'test_metrics': {k: float(v) for k, v in results['test_metrics'].items()},
            'config': self.config.__dict__
        }

        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=json_convert)


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Task MDN Training")

    parser.add_argument("--window_size", type=int, default=120)
    parser.add_argument("--num_mixtures", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--visualize_every_n_epochs", type=int, default=5)

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    config = MDNTrainingConfig(
        window_size=args.window_size,
        num_mixtures=args.num_mixtures,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        visualize_every_n_epochs=args.visualize_every_n_epochs
    )

    gen_config = TimeSeriesGeneratorConfig(
        n_samples=5000,
        random_seed=42,
        default_noise_level=0.1
    )

    try:
        trainer = MDNTrainer(config, gen_config)
        results = trainer.run_experiment()
        logger.info(f"Completed. Results: {results['experiment_dir']}")

        keras.backend.clear_session()
        sys.stdout.flush()

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

    os._exit(0)

if __name__ == "__main__":
    main()
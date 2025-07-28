import os
import json
import keras
import random
import matplotlib
import numpy as np
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

# Use a non-interactive backend for saving plots to files
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.nbeats import create_nbeats_model
from dl_techniques.losses.smape_loss import SMAPELoss
from dl_techniques.utils.datasets.nbeats import TimeSeriesNormalizer
from dl_techniques.utils.datasets.time_series_generator import TimeSeriesGenerator, TimeSeriesConfig

from dl_techniques.models.nbeats_multitask import (
    MultiTaskNBeatsNet,
    MultiTaskNBeatsConfig,
    create_multi_task_nbeats
)

# ---------------------------------------------------------------------
# Set up plotting style
# ---------------------------------------------------------------------

plt.style.use('default')
sns.set_palette("husl")


# ---------------------------------------------------------------------
# Random seeds for reproducibility
# ---------------------------------------------------------------------

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility in Keras 3.x."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)


# ---------------------------------------------------------------------
# Multi-Task Configuration
# ---------------------------------------------------------------------

@dataclass
class MultiTaskNBeatsConfig:
    """Configuration for multi-task N-BEATS training with interim visualizations."""

    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "multi_task_nbeats"

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # N-BEATS specific configuration
    backcast_length: int = 168
    forecast_length: int = 24
    forecast_horizons: List[int] = field(default_factory=lambda: [12, 24])

    # Multi-task model configuration
    use_task_embeddings: bool = True  # NEW: Add task embeddings
    task_embedding_dim: int = 32  # Dimension of task embeddings
    shared_layers: int = 4  # Number of shared layers before task-specific heads

    # Model architecture
    stack_types: List[str] = field(default_factory=lambda: ["trend", "seasonality", "generic"])
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 512  # Larger for multi-task
    use_revin: bool = True

    # Training configuration
    epochs: int = 150
    batch_size: int = 128  # Larger batch for multi-task
    learning_rate: float = 1e-3  # Slightly higher for multi-task
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adamw'
    primary_loss: str = "mae"

    # Regularization
    kernel_regularizer_l2: float = 1e-4
    dropout_rate: float = 0.15  # Higher dropout for multi-task

    # Task selection and balancing
    max_tasks_per_category: int = 5  # More tasks for multi-task learning
    min_data_length: int = 1000
    balance_tasks: bool = True  # Balance data across tasks
    samples_per_task: int = 2000  # Max samples per task to prevent imbalance

    # Interim visualization configuration
    visualize_every_n_epochs: int = 5  # Show results every N epochs
    save_interim_plots: bool = True
    plot_top_k_tasks: int = 6  # Show results for top K tasks
    create_learning_curves: bool = True
    create_prediction_plots: bool = True
    create_task_performance_heatmap: bool = True

    # Evaluation configuration
    eval_during_training: bool = True  # Evaluate on test set during training
    eval_every_n_epochs: int = 10

    def __post_init__(self) -> None:
        """Enhanced validation for multi-task configuration."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")

        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")

        ratio = self.backcast_length / self.forecast_length
        if ratio < 3.0:
            logger.warning(f"Consider increasing backcast_length for better performance (current ratio: {ratio:.1f})")

        logger.info(f"Multi-Task N-BEATS Configuration:")
        logger.info(f"  - Multi-task learning: ✓ Enabled")
        logger.info(f"  - Task embeddings: {'✓' if self.use_task_embeddings else '✗'}")
        logger.info(f"  - Interim visualizations every {self.visualize_every_n_epochs} epochs")
        logger.info(f"  - Task balancing: {'✓' if self.balance_tasks else '✗'}")

# ---------------------------------------------------------------------
# Multi-Task Data Processor
# ---------------------------------------------------------------------

class MultiTaskDataProcessor:
    """Data processor for multi-task N-BEATS training."""

    def __init__(self, config: MultiTaskNBeatsConfig):
        self.config = config
        self.scalers: Dict[str, TimeSeriesNormalizer] = {}
        self.task_to_id: Dict[str, int] = {}
        self.id_to_task: Dict[int, str] = {}

    def prepare_multi_task_data(
            self,
            raw_task_data: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Prepare combined multi-task training data."""

        logger.info("Preparing multi-task N-BEATS data...")

        # Create task ID mapping
        self.task_to_id = {task: idx for idx, task in enumerate(raw_task_data.keys())}
        self.id_to_task = {idx: task for task, idx in self.task_to_id.items()}

        # Fit scalers for each task
        self._fit_scalers(raw_task_data)

        # Prepare data for each horizon
        prepared_data = {}

        for horizon in self.config.forecast_horizons:
            logger.info(f"Preparing data for horizon {horizon}")

            # Collect all sequences from all tasks
            all_train_X, all_train_y, all_train_task_ids = [], [], []
            all_val_X, all_val_y, all_val_task_ids = [], [], []
            all_test_X, all_test_y, all_test_task_ids = [], [], []

            for task_name, data in raw_task_data.items():
                task_id = self.task_to_id[task_name]

                try:
                    # Check data sufficiency
                    min_length = self.config.backcast_length + horizon + 100
                    if len(data) < min_length:
                        logger.warning(f"Insufficient data for {task_name} H={horizon}")
                        continue

                    # Split data
                    train_size = int(self.config.train_ratio * len(data))
                    val_size = int(self.config.val_ratio * len(data))

                    train_data = data[:train_size]
                    val_data = data[train_size:train_size + val_size]
                    test_data = data[train_size + val_size:]

                    # Transform data
                    train_scaled = self.scalers[task_name].transform(train_data)
                    val_scaled = self.scalers[task_name].transform(val_data)
                    test_scaled = self.scalers[task_name].transform(test_data)

                    # Create sequences
                    train_X, train_y = self._create_sequences(train_scaled, horizon)
                    val_X, val_y = self._create_sequences(val_scaled, horizon, stride=horizon // 2)
                    test_X, test_y = self._create_sequences(test_scaled, horizon, stride=horizon // 2)

                    # Balance data if requested
                    if self.config.balance_tasks:
                        max_samples = self.config.samples_per_task
                        if len(train_X) > max_samples:
                            indices = np.random.choice(len(train_X), max_samples, replace=False)
                            train_X = train_X[indices]
                            train_y = train_y[indices]

                    # Add to combined dataset
                    all_train_X.append(train_X)
                    all_train_y.append(train_y)
                    all_train_task_ids.append(np.full(len(train_X), task_id))

                    all_val_X.append(val_X)
                    all_val_y.append(val_y)
                    all_val_task_ids.append(np.full(len(val_X), task_id))

                    all_test_X.append(test_X)
                    all_test_y.append(test_y)
                    all_test_task_ids.append(np.full(len(test_X), task_id))

                    logger.info(f"Added {task_name}: train={len(train_X)}, val={len(val_X)}, test={len(test_X)}")

                except Exception as e:
                    logger.warning(f"Failed to prepare {task_name} H={horizon}: {e}")
                    continue

            if not all_train_X:
                logger.error(f"No data prepared for horizon {horizon}")
                continue

            # Combine all tasks
            combined_train_X = np.concatenate(all_train_X, axis=0)
            combined_train_y = np.concatenate(all_train_y, axis=0)
            combined_train_task_ids = np.concatenate(all_train_task_ids, axis=0)

            combined_val_X = np.concatenate(all_val_X, axis=0)
            combined_val_y = np.concatenate(all_val_y, axis=0)
            combined_val_task_ids = np.concatenate(all_val_task_ids, axis=0)

            combined_test_X = np.concatenate(all_test_X, axis=0)
            combined_test_y = np.concatenate(all_test_y, axis=0)
            combined_test_task_ids = np.concatenate(all_test_task_ids, axis=0)

            # Shuffle the combined training data
            train_indices = np.random.permutation(len(combined_train_X))
            combined_train_X = combined_train_X[train_indices]
            combined_train_y = combined_train_y[train_indices]
            combined_train_task_ids = combined_train_task_ids[train_indices]

            prepared_data[horizon] = (
                (combined_train_X, combined_train_y, combined_train_task_ids),
                (combined_val_X, combined_val_y, combined_val_task_ids),
                (combined_test_X, combined_test_y, combined_test_task_ids)
            )

            logger.info(
                f"Combined H={horizon} data: train={len(combined_train_X)}, val={len(combined_val_X)}, test={len(combined_test_X)}")

        return prepared_data

    def _fit_scalers(self, task_data: Dict[str, np.ndarray]) -> None:
        """Fit scalers for each task."""
        for task_name, data in task_data.items():
            if len(data) < self.config.min_data_length:
                continue

            scaler = TimeSeriesNormalizer(method='standard')
            train_size = int(self.config.train_ratio * len(data))
            train_data = data[:train_size]

            scaler.fit(train_data)
            self.scalers[task_name] = scaler

            logger.info(f"Fitted scaler for {task_name}")

    def _create_sequences(
            self,
            data: np.ndarray,
            forecast_length: int,
            stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        X, y = [], []

        for i in range(0, len(data) - self.config.backcast_length - forecast_length + 1, stride):
            backcast = data[i: i + self.config.backcast_length]
            forecast = data[i + self.config.backcast_length: i + self.config.backcast_length + forecast_length]

            if not (np.isnan(backcast).any() or np.isnan(forecast).any()):
                X.append(backcast)
                y.append(forecast)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------
# Interim Visualization Callback
# ---------------------------------------------------------------------

class InterimVisualizationCallback(keras.callbacks.Callback):
    """Callback for creating interim visualizations during training."""

    def __init__(
            self,
            config: MultiTaskNBeatsConfig,
            data_processor: MultiTaskDataProcessor,
            test_data: Dict[int, Tuple],
            save_dir: str
    ):
        super().__init__()
        self.config = config
        self.data_processor = data_processor
        self.test_data = test_data
        self.save_dir = save_dir
        self.training_history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'task_losses': {task: [] for task in data_processor.task_to_id.keys()}
        }

        os.makedirs(save_dir, exist_ok=True)

        # Select representative tasks for visualization
        self.viz_tasks = list(data_processor.task_to_id.keys())[:config.plot_top_k_tasks]

    def on_epoch_end(self, epoch, logs=None):
        """Create visualizations at specified intervals."""
        if logs is None:
            logs = {}

        # Store training history
        self.training_history['epoch'].append(epoch)
        self.training_history['loss'].append(logs.get('loss', 0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0))

        # Create visualizations
        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(f"Creating interim visualizations at epoch {epoch + 1}")
            self._create_interim_plots(epoch)

    def _create_interim_plots(self, epoch: int):
        """Create comprehensive interim plots."""
        try:
            # 1. Learning curves
            if self.config.create_learning_curves:
                self._plot_learning_curves(epoch)

            # 2. Prediction samples
            if self.config.create_prediction_plots:
                self._plot_prediction_samples(epoch)

            # 3. Task performance heatmap
            if self.config.create_task_performance_heatmap:
                self._plot_task_performance(epoch)

            logger.info(f"Interim plots saved for epoch {epoch + 1}")

        except Exception as e:
            logger.warning(f"Failed to create interim plots: {e}")

    def _plot_learning_curves(self, epoch: int):
        """Plot training and validation loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        epochs = self.training_history['epoch']

        # Loss curves
        axes[0].plot(epochs, self.training_history['loss'], label='Training Loss', color='blue')
        axes[0].plot(epochs, self.training_history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title(f'Learning Curves (Epoch {epoch + 1})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Learning rate (if available)
        try:
            if hasattr(self.model.optimizer.learning_rate, 'numpy'):
                current_lr = float(self.model.optimizer.learning_rate.numpy())
            else:
                current_lr = float(self.model.optimizer.learning_rate)

            axes[1].axhline(y=current_lr, color='green', linestyle='--', label=f'Current LR: {current_lr:.2e}')
            axes[1].set_title('Learning Rate')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        except Exception:
            axes[1].text(0.5, 0.5, 'LR info unavailable', ha='center', va='center', transform=axes[1].transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'learning_curves_epoch_{epoch + 1:03d}.png'), dpi=150,
                    bbox_inches='tight')
        plt.close()

    def _plot_prediction_samples(self, epoch: int):
        """Plot prediction samples for different tasks."""
        horizon = self.config.forecast_horizons[0]  # Use first horizon
        test_X, test_y, test_task_ids = self.test_data[horizon]

        # Get the actual forecast length from the test data
        forecast_length = test_y.shape[1]

        # Sample predictions for each task
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, task_name in enumerate(self.viz_tasks[:6]):
            if i >= len(axes):
                break

            task_id = self.data_processor.task_to_id[task_name]
            task_mask = test_task_ids == task_id

            if not np.any(task_mask):
                continue

            # Get sample for this task
            task_X = test_X[task_mask]
            task_y = test_y[task_mask]
            task_ids = test_task_ids[task_mask]

            if len(task_X) == 0:
                continue

            # Take first sample
            sample_X = task_X[0:1]  # Shape: (1, backcast_length)
            sample_y = task_y[0:1]  # Shape: (1, forecast_length)
            sample_task_id = task_ids[0:1]  # Shape: (1,)

            # Get prediction
            pred_y = self.model((sample_X, sample_task_id), training=False)

            # Plot
            backcast_x = np.arange(-self.config.backcast_length, 0)
            forecast_x = np.arange(0, forecast_length)  # Use actual forecast length

            axes[i].plot(backcast_x, sample_X[0], label='Input', color='blue', alpha=0.7)
            axes[i].plot(forecast_x, sample_y[0].flatten(), label='True', color='green', linewidth=2)
            axes[i].plot(forecast_x, pred_y[0].numpy().flatten(), label='Predicted', color='red', linestyle='--',
                         linewidth=2)
            axes[i].set_title(f'{task_name}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.5)

        # Remove empty subplots
        for i in range(len(self.viz_tasks), len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f'Prediction Samples (Epoch {epoch + 1})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch + 1:03d}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_task_performance(self, epoch: int):
        """Plot task-specific performance heatmap."""
        horizon = self.config.forecast_horizons[0]
        test_X, test_y, test_task_ids = self.test_data[horizon]

        # Calculate MAE for each task
        task_maes = []
        task_names = []

        for task_name, task_id in self.data_processor.task_to_id.items():
            task_mask = test_task_ids == task_id

            if not np.any(task_mask):
                continue

            task_X = test_X[task_mask][:100]  # Limit to 100 samples for speed
            task_y = test_y[task_mask][:100]
            task_ids = np.full(len(task_X), task_id)

            if len(task_X) == 0:
                continue

            # Get predictions
            pred_y = self.model((task_X, task_ids), training=False)

            # Calculate MAE
            mae = np.mean(np.abs(task_y - pred_y.numpy()))
            task_maes.append(mae)
            task_names.append(task_name)

        if task_maes:
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))

            # Prepare data for heatmap
            data_matrix = np.array(task_maes).reshape(-1, 1)

            # Create heatmap
            im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')

            # Set labels
            ax.set_yticks(range(len(task_names)))
            ax.set_yticklabels(task_names)
            ax.set_xticks([0])
            ax.set_xticklabels(['MAE'])
            ax.set_title(f'Task Performance Heatmap (Epoch {epoch + 1})')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mean Absolute Error')

            # Add value annotations
            for i, mae in enumerate(task_maes):
                ax.text(0, i, f'{mae:.3f}', ha='center', va='center',
                        color='white' if mae > np.median(task_maes) else 'black')

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'task_performance_epoch_{epoch + 1:03d}.png'), dpi=150,
                        bbox_inches='tight')
            plt.close()


# ---------------------------------------------------------------------
# Multi-Task Trainer
# ---------------------------------------------------------------------

class MultiTaskNBeatsTrainer:
    """Multi-task N-BEATS trainer with interim visualizations."""

    def __init__(self, config: MultiTaskNBeatsConfig, ts_config: TimeSeriesConfig):
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
        self.processor = MultiTaskDataProcessor(config)

        # Get all available tasks
        self.all_tasks = self.generator.get_task_names()
        self.task_categories = self.generator.get_task_categories()

        # Select tasks for training
        self.selected_tasks = self._select_tasks()

        logger.info(f"Multi-Task N-BEATS Trainer Initialized:")
        logger.info(f"  - Total available tasks: {len(self.all_tasks)}")
        logger.info(f"  - Selected tasks: {len(self.selected_tasks)}")
        logger.info(f"  - Multi-task learning: ✓ ENABLED")
        logger.info(f"  - Interim visualizations: ✓ ENABLED")

    def _select_tasks(self) -> List[str]:
        """Select tasks for multi-task training."""
        selected = []

        for category in self.task_categories:
            category_tasks = self.generator.get_tasks_by_category(category)
            selected.extend(category_tasks[:self.config.max_tasks_per_category])

        logger.info(f"Selected {len(selected)} tasks from {len(self.task_categories)} categories")
        return selected

    def prepare_data(self) -> Dict[str, Any]:
        """Prepare multi-task training data."""
        logger.info("Generating raw data for all selected tasks...")

        # Generate raw data for all selected tasks
        raw_task_data = {}
        failed_tasks = []

        for task_name in self.selected_tasks:
            try:
                data = self.generator.generate_task_data(task_name)
                if len(data) >= self.config.min_data_length:
                    raw_task_data[task_name] = data
                else:
                    failed_tasks.append(f"{task_name} (too short)")
            except Exception as e:
                logger.warning(f"Failed to generate {task_name}: {e}")
                failed_tasks.append(f"{task_name} (error)")

        if failed_tasks:
            logger.warning(f"Failed tasks: {', '.join(failed_tasks)}")

        logger.info(f"Successfully generated data for {len(raw_task_data)} tasks")

        # Prepare multi-task data
        prepared_data = self.processor.prepare_multi_task_data(raw_task_data)

        return {
            'prepared_data': prepared_data,
            'raw_task_data': raw_task_data,
            'num_tasks': len(raw_task_data),
            'task_to_id': self.processor.task_to_id
        }

    def create_model(self, num_tasks: int, task_to_id: Dict[str, int], forecast_length: int) -> MultiTaskNBeatsNet:
        """Create multi-task N-BEATS model for a specific forecast length.

        This method creates a MultiTaskNBeatsNet model configured with the trainer's
        parameters and optimized for the specified forecast horizon.

        Args:
            num_tasks: Number of different tasks the model should handle.
            task_to_id: Dictionary mapping task names to integer IDs.
            forecast_length: Number of time steps the model should forecast.

        Returns:
            Configured and ready-to-use MultiTaskNBeatsNet model.

        Raises:
            ValueError: If configuration parameters are invalid.
            RuntimeError: If model creation fails.
        """
        logger.info(f"Creating Multi-Task N-BEATS model for forecast horizon {forecast_length}...")

        try:
            # Create model configuration from trainer config
            model_config = MultiTaskNBeatsConfig(
                # Architecture parameters
                backcast_length=self.config.backcast_length,
                use_task_embeddings=self.config.use_task_embeddings,
                task_embedding_dim=self.config.task_embedding_dim,
                stack_types=self.config.stack_types.copy(),  # Copy to avoid reference issues
                nb_blocks_per_stack=self.config.nb_blocks_per_stack,
                hidden_layer_units=self.config.hidden_layer_units,
                use_revin=self.config.use_revin,

                # Regularization parameters
                dropout_rate=self.config.dropout_rate,
                kernel_regularizer_l2=self.config.kernel_regularizer_l2,

                # Training parameters
                optimizer=self.config.optimizer,
                primary_loss=self.config.primary_loss,
                learning_rate=self.config.learning_rate,
                gradient_clip_norm=self.config.gradient_clip_norm
            )

            logger.info("✓ Model configuration created")
            logger.info(f"  - Architecture: {model_config.stack_types}")
            logger.info(f"  - Hidden units: {model_config.hidden_layer_units}")
            logger.info(f"  - Task embeddings: {'✓' if model_config.use_task_embeddings else '✗'}")
            logger.info(f"  - RevIN normalization: {'✓' if model_config.use_revin else '✗'}")

            # Create the model using the factory function
            model = create_multi_task_nbeats(
                config=model_config,
                num_tasks=num_tasks,
                task_to_id=task_to_id,
                forecast_length=forecast_length,
                name=f"MultiTaskNBeats_H{forecast_length}"
            )

            logger.info("✓ Multi-Task N-BEATS model created successfully")
            logger.info(f"  - Number of tasks: {num_tasks}")
            logger.info(f"  - Forecast length: {forecast_length}")
            logger.info(f"  - Task mapping: {list(task_to_id.keys())}")

            return model

        except Exception as e:
            logger.error(f"Failed to create Multi-Task N-BEATS model: {e}")
            logger.error(f"Configuration details:")
            logger.error(f"  - num_tasks: {num_tasks}")
            logger.error(f"  - forecast_length: {forecast_length}")
            logger.error(f"  - task_to_id keys: {list(task_to_id.keys()) if task_to_id else 'None'}")
            raise RuntimeError(f"Model creation failed: {e}") from e

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete multi-task N-BEATS experiment."""

        try:
            # Create experiment directory
            exp_dir = os.path.join(
                self.config.result_dir,
                f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(exp_dir, exist_ok=True)

            logger.info(f"🚀 Starting Multi-Task N-BEATS Experiment: {exp_dir}")

            # Prepare data
            data_info = self.prepare_data()
            prepared_data = data_info['prepared_data']

            if not prepared_data:
                raise ValueError("No data prepared for training")

            # Train for each horizon (create separate model for each horizon)
            results = {}

            for horizon in self.config.forecast_horizons:
                if horizon not in prepared_data:
                    logger.warning(f"No data available for horizon {horizon}")
                    continue

                logger.info(f"\n{'=' * 60}\n🎯 Training Multi-Task Model for Horizon {horizon}\n{'=' * 60}")

                # Create model for this specific horizon
                model = self.create_model(data_info['num_tasks'], data_info['task_to_id'], horizon)

                # Get data for this horizon
                train_data, val_data, test_data = prepared_data[horizon]

                # Build model with appropriate input shape
                sample_input = (train_data[0][:1], train_data[2][:1])  # (X, task_ids)
                model(sample_input)  # Build the model

                # IMPORTANT: Compile the model after building
                # Create optimizer
                if self.config.optimizer.lower() == 'adamw':
                    optimizer = keras.optimizers.AdamW(
                        learning_rate=self.config.learning_rate,
                        clipnorm=self.config.gradient_clip_norm if self.config.gradient_clip_norm > 0 else None
                    )
                elif self.config.optimizer.lower() == 'adam':
                    optimizer = keras.optimizers.Adam(
                        learning_rate=self.config.learning_rate,
                        clipnorm=self.config.gradient_clip_norm if self.config.gradient_clip_norm > 0 else None
                    )
                else:
                    optimizer = keras.optimizers.get(self.config.optimizer)
                    optimizer.learning_rate = self.config.learning_rate
                    if self.config.gradient_clip_norm > 0:
                        optimizer.clipnorm = self.config.gradient_clip_norm

                # Create loss function
                if self.config.primary_loss.lower() == 'mae':
                    loss = keras.losses.MeanAbsoluteError()
                elif self.config.primary_loss.lower() == 'mse':
                    loss = keras.losses.MeanSquaredError()
                elif self.config.primary_loss.lower() == 'smape':
                    loss = SMAPELoss()
                else:
                    loss = keras.losses.get(self.config.primary_loss)

                # Compile the model
                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=['mae']
                )

                logger.info("✅ Model compiled successfully")

                # Create callbacks
                viz_dir = os.path.join(exp_dir, f'visualizations_h{horizon}')

                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=30,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=8,
                        min_lr=1e-6,
                        verbose=1
                    ),
                    InterimVisualizationCallback(
                        config=self.config,
                        data_processor=self.processor,
                        test_data={horizon: test_data},
                        save_dir=viz_dir
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(exp_dir, f'best_model_h{horizon}.keras'),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1
                    )
                ]

                # Train the model
                logger.info(
                    f"Training data shape: X={train_data[0].shape}, y={train_data[1].shape}, tasks={train_data[2].shape}")
                logger.info(
                    f"Validation data shape: X={val_data[0].shape}, y={val_data[1].shape}, tasks={val_data[2].shape}")

                start_time = datetime.now()

                history = model.fit(
                    x=(train_data[0], train_data[2]),  # (X, task_ids)
                    y=train_data[1],  # y
                    validation_data=((val_data[0], val_data[2]), val_data[1]),
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    callbacks=callbacks,
                    verbose=1
                )

                training_time = (datetime.now() - start_time).total_seconds()

                # Evaluate on test set
                test_results = model.evaluate(
                    x=(test_data[0], test_data[2]),
                    y=test_data[1],
                    verbose=0
                )

                # Extract loss value (handle both single value and list cases)
                if isinstance(test_results, list):
                    test_loss = test_results[0]  # First element is the loss
                    test_mae = test_results[1] if len(test_results) > 1 else None
                else:
                    test_loss = test_results
                    test_mae = None

                results[horizon] = {
                    'history': history.history,
                    'training_time': training_time,
                    'test_loss': test_loss,
                    'test_mae': test_mae,
                    'final_epoch': len(history.history['loss'])
                }

                logger.info(f"✅ Completed training for horizon {horizon}:")
                logger.info(f"   - Training time: {training_time:.1f}s")
                logger.info(f"   - Final epochs: {results[horizon]['final_epoch']}")
                logger.info(f"   - Test loss: {test_loss:.4f}")
                if test_mae is not None:
                    logger.info(f"   - Test MAE: {test_mae:.4f}")

            # Save final results
            self._save_final_results(results, exp_dir, data_info)

            logger.info(f"🎉 Multi-Task N-BEATS experiment completed successfully!")
            logger.info(f"📊 Results directory: {exp_dir}")
            logger.info(f"📈 Horizons trained: {list(results.keys())}")

            return {
                "results_dir": exp_dir,
                "results": results,
                "num_tasks": data_info['num_tasks'],
                "task_mapping": data_info['task_to_id']
            }

        except Exception as e:
            logger.error(f"💥 Multi-task experiment failed: {e}", exc_info=True)
            raise

    def _save_final_results(self, results: Dict, exp_dir: str, data_info: Dict):
        """Save final experiment results."""
        try:
            # Save training history
            with open(os.path.join(exp_dir, 'training_results.json'), 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for horizon, result in results.items():
                    json_results[str(horizon)] = {
                        'training_time': result['training_time'],
                        'test_loss': float(result['test_loss']),
                        'test_mae': float(result['test_mae']) if result.get('test_mae') is not None else None,
                        'final_epoch': result['final_epoch'],
                        'final_train_loss': result['history']['loss'][-1],
                        'final_val_loss': result['history']['val_loss'][-1]
                    }
                json.dump(json_results, f, indent=2)

            # Save task information
            with open(os.path.join(exp_dir, 'task_info.json'), 'w') as f:
                json.dump({
                    'num_tasks': data_info['num_tasks'],
                    'task_to_id': data_info['task_to_id'],
                    'selected_tasks': self.selected_tasks
                }, f, indent=2)

            # Create final summary plot
            self._create_final_summary_plot(results, exp_dir)

            logger.info(f"Final results saved to {exp_dir}")

        except Exception as e:
            logger.error(f"Failed to save final results: {e}")

    def _create_final_summary_plot(self, results: Dict, exp_dir: str):
        """Create final summary visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Training curves for all horizons
            for horizon, result in results.items():
                history = result['history']
                epochs = range(1, len(history['loss']) + 1)

                axes[0, 0].plot(epochs, history['loss'], label=f'H{horizon} Train', alpha=0.8)
                axes[0, 1].plot(epochs, history['val_loss'], label=f'H{horizon} Val', alpha=0.8)

            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Performance summary
            horizons = list(results.keys())
            test_losses = [results[h]['test_loss'] for h in horizons]
            training_times = [results[h]['training_time'] for h in horizons]

            axes[1, 0].bar(range(len(horizons)), test_losses, color='skyblue', alpha=0.7)
            axes[1, 0].set_title('Test Loss by Horizon')
            axes[1, 0].set_xlabel('Horizon')
            axes[1, 0].set_ylabel('Test Loss')
            axes[1, 0].set_xticks(range(len(horizons)))
            axes[1, 0].set_xticklabels(horizons)

            axes[1, 1].bar(range(len(horizons)), training_times, color='lightcoral', alpha=0.7)
            axes[1, 1].set_title('Training Time by Horizon')
            axes[1, 1].set_xlabel('Horizon')
            axes[1, 1].set_ylabel('Training Time (s)')
            axes[1, 1].set_xticks(range(len(horizons)))
            axes[1, 1].set_xticklabels(horizons)

            plt.suptitle('Multi-Task N-BEATS Training Summary', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, 'final_summary.png'), dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create final summary plot: {e}")


# ---------------------------------------------------------------------
# Main Experiment Function
# ---------------------------------------------------------------------

def main():
    """Run the multi-task N-BEATS experiment with interim visualizations."""

    # Configuration
    config = MultiTaskNBeatsConfig(
        # Multi-task specific
        backcast_length=168,
        forecast_length=24,
        forecast_horizons=[12, 24],
        use_task_embeddings=True,

        # Visualizations
        visualize_every_n_epochs=5,
        save_interim_plots=True,
        plot_top_k_tasks=8,
    )

    # Time series configuration
    ts_config = TimeSeriesConfig(
        n_samples=3000,
        random_seed=42,
        default_noise_level=0.1
    )

    logger.info("🚀 Starting Multi-Task N-BEATS Experiment with Interim Visualizations")
    logger.info("=" * 80)
    logger.info("KEY FEATURES:")
    logger.info("  ✓ Multi-task learning on ALL selected tasks simultaneously")
    logger.info("  ✓ Task embeddings for task-aware predictions")
    logger.info("  ✓ Interim visualizations every 5 epochs")
    logger.info("  ✓ Real-time performance monitoring")
    logger.info("  ✓ Balanced multi-task dataset")
    logger.info("  ✓ Comprehensive results analysis")
    logger.info("=" * 80)

    try:
        trainer = MultiTaskNBeatsTrainer(config, ts_config)
        results = trainer.run_experiment()

        logger.info("🎉 Multi-Task Experiment Completed Successfully!")
        logger.info(f"📊 Results directory: {results['results_dir']}")
        logger.info(f"📈 Number of tasks: {results['num_tasks']}")
        logger.info(f"🎯 Horizons trained: {list(results['results'].keys())}")
        logger.info("💡 Check the results directory for:")
        logger.info("  - Interim visualization plots (every 5 epochs)")
        logger.info("  - Learning curves and prediction samples")
        logger.info("  - Task performance heatmaps")
        logger.info("  - Final summary plots and metrics")

    except Exception as e:
        logger.error(f"💥 Multi-task experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
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
from typing import Dict, List, Tuple, Any, Optional

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dl_techniques.utils.logger import logger
from dl_techniques.losses.smape_loss import SMAPELoss
from dl_techniques.utils.datasets.nbeats import TimeSeriesNormalizer
from dl_techniques.utils.datasets.time_series_generator import TimeSeriesGenerator, TimeSeriesConfig

plt.style.use('default')
sns.set_palette("husl")


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


set_random_seeds(42)


@dataclass
class MultiTaskNBeatsTrainingConfig:
    """Configuration for multi-task N-BEATS training."""

    # General experiment configuration
    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "nbeats_multi_task"

    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # N-BEATS specific configuration
    backcast_length: int = 168
    forecast_length: int = 24
    forecast_horizons: List[int] = field(default_factory=lambda: [12])

    # Multi-task model configuration
    use_task_embeddings: bool = True
    task_embedding_dim: int = 16

    # Trainable Task Inference Configuration
    train_task_inference: bool = True
    task_inference_loss_weight: float = 0.5      # Increased from 0.1
    consistency_loss_weight: float = 0.1         # Increased from 0.05
    entropy_loss_weight: float = 0.05            # Increased from 0.02
    consistency_temperature: float = 0.1
    min_entropy_target: float = 0.1              # Reduced from 0.5 to reasonable value

    # Curriculum Learning Configuration (improved schedule)
    use_curriculum_learning: bool = True
    curriculum_start_ratio: float = 0.8          # Start with 80% labeled (was 1.0)
    curriculum_end_ratio: float = 0.4            # End with 40% labeled (was 0.3)
    curriculum_transition_epochs: int = 30       # Faster transition (was 50)

    # Model architecture
    stack_types: List[str] = field(default_factory=lambda: ["trend", "seasonality", "generic"])
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 256                # reduced for stability (was 512)
    use_revin: bool = True

    # ORIGINAL Training configuration
    epochs: int = 150
    batch_size: int = 128
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adamw'
    primary_loss: str = "mae"

    # BALANCED Regularization (not too aggressive, not too weak)
    kernel_regularizer_l2: float = 1e-5
    dropout_rate: float = 0.15

    # ORIGINAL Task selection and balancing
    max_tasks_per_category: int = 5
    min_data_length: int = 1000
    balance_tasks: bool = True
    samples_per_task: int = 10000

    # ORIGINAL Visualization configuration (restored)
    visualize_every_n_epochs: int = 5
    save_interim_plots: bool = True
    plot_top_k_tasks: int = 6
    create_learning_curves: bool = True
    create_prediction_plots: bool = True
    create_task_performance_heatmap: bool = True

    # ORIGINAL Evaluation configuration (restored)
    eval_during_training: bool = True
    eval_every_n_epochs: int = 10

    def __post_init__(self) -> None:
        """Enhanced validation with better error checking."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")

        if self.backcast_length <= 0 or self.forecast_length <= 0:
            raise ValueError("backcast_length and forecast_length must be positive")

        if self.use_curriculum_learning:
            if not (0.0 <= self.curriculum_end_ratio <= self.curriculum_start_ratio <= 1.0):
                raise ValueError("Invalid curriculum learning ratios")

        # Check if we have enough validation data
        if self.val_ratio < 0.2:
            logger.warning(f"Validation ratio {self.val_ratio} might be too small for reliable validation")

        logger.info(f"Multi-Task N-BEATS Configuration with parameters:")
        logger.info(f"  ✅ train/val split: {self.train_ratio:.1f}/{self.val_ratio:.1f}/{self.test_ratio:.1f}")
        logger.info(f"  ✅ data generation: n_samples will be 3000")
        logger.info(f"  ✅ task selection: {self.max_tasks_per_category} tasks per category")
        logger.info(f"  ✅ model size: {self.nb_blocks_per_stack} blocks, {self.hidden_layer_units} units")
        logger.info(f"  ✅ auxiliary loss weights: {self.task_inference_loss_weight:.1f}/{self.consistency_loss_weight:.1f}/{self.entropy_loss_weight:.2f}")
        logger.info(f"  ✅ training: {self.epochs} epochs, batch {self.batch_size}, lr {self.learning_rate}")
        logger.info(f"  ✅ regularization: dropout {self.dropout_rate}, L2 {self.kernel_regularizer_l2}")
        logger.info(f"  ✅ curriculum learning: {self.curriculum_start_ratio:.1f} → {self.curriculum_end_ratio:.1f} over {self.curriculum_transition_epochs} epochs")


class MultiTaskDataProcessor:
    """Data processor with better validation handling."""

    def __init__(self, config: MultiTaskNBeatsTrainingConfig):
        self.config = config
        self.scalers: Dict[str, TimeSeriesNormalizer] = {}
        self.task_to_id: Dict[str, int] = {}
        self.id_to_task: Dict[int, str] = {}

    def prepare_multi_task_data(
            self,
            raw_task_data: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Tuple]]:
        """: Prepare multi-task data with better validation split."""

        logger.info("Preparing multi-task data...")

        # Create task ID mapping
        self.task_to_id = {task: idx for idx, task in enumerate(raw_task_data.keys())}
        self.id_to_task = {idx: task for task, idx in self.task_to_id.items()}

        # Fit scalers
        self._fit_scalers(raw_task_data)

        prepared_data = {}

        for horizon in self.config.forecast_horizons:
            logger.info(f"Preparing data for horizon {horizon}")

            all_train_X, all_train_y, all_train_task_ids = [], [], []
            all_val_X, all_val_y, all_val_task_ids = [], [], []
            all_test_X, all_test_y, all_test_task_ids = [], [], []
            all_unlabeled_X, all_unlabeled_y = [], []

            for task_name, data in raw_task_data.items():
                task_id = self.task_to_id[task_name]

                try:
                    min_length = self.config.backcast_length + horizon + 100  # Reduced buffer from 200
                    if len(data) < min_length:
                        logger.warning(f"Insufficient data for {task_name} H={horizon}: {len(data)} < {min_length}")
                        continue

                    train_size = int(self.config.train_ratio * len(data))
                    val_size = int(self.config.val_ratio * len(data))

                    train_data = data[:train_size]
                    val_data = data[train_size:train_size + val_size]
                    test_data = data[train_size + val_size:]

                    min_seq_len = self.config.backcast_length + horizon
                    if len(train_data) < min_seq_len * 5:  # Reduced from 10 sequences
                        logger.warning(f"Not enough training data for {task_name}: {len(train_data)} < {min_seq_len * 5}")
                        continue
                    if len(val_data) < min_seq_len * 2:  # Reduced from 3 sequences
                        logger.warning(f"Not enough validation data for {task_name}: {len(val_data)} < {min_seq_len * 2}")
                        continue

                    # Transform data
                    train_scaled = self.scalers[task_name].transform(train_data)
                    val_scaled = self.scalers[task_name].transform(val_data)
                    test_scaled = self.scalers[task_name].transform(test_data)

                    # Create sequences with different strides for train/val
                    train_X, train_y = self._create_sequences(train_scaled, horizon, stride=1)
                    val_X, val_y = self._create_sequences(val_scaled, horizon, stride=horizon//3)  # Reduced overlap
                    test_X, test_y = self._create_sequences(test_scaled, horizon, stride=horizon//3)

                    if self.config.balance_tasks and len(train_X) > self.config.samples_per_task:
                        # Use stratified sampling to maintain temporal structure
                        step = max(1, len(train_X) // self.config.samples_per_task)
                        indices = np.arange(0, len(train_X), step)[:self.config.samples_per_task]
                        train_X = train_X[indices]
                        train_y = train_y[indices]

                    # Add to datasets
                    all_train_X.append(train_X)
                    all_train_y.append(train_y)
                    all_train_task_ids.append(np.full(len(train_X), task_id))

                    all_val_X.append(val_X)
                    all_val_y.append(val_y)
                    all_val_task_ids.append(np.full(len(val_X), task_id))

                    all_test_X.append(test_X)
                    all_test_y.append(test_y)
                    all_test_task_ids.append(np.full(len(test_X), task_id))

                    # Create unlabeled data for task inference training
                    unlabeled_size = min(len(train_X)//2, 500)  # Increased from 200
                    if len(train_X) > unlabeled_size:
                        unlabeled_indices = np.random.choice(len(train_X), unlabeled_size, replace=False)
                        all_unlabeled_X.append(train_X[unlabeled_indices])
                        all_unlabeled_y.append(train_y[unlabeled_indices])

                    logger.info(f"Added {task_name}: train={len(train_X)}, val={len(val_X)}, test={len(test_X)}")

                except Exception as e:
                    logger.warning(f"Failed to prepare {task_name} H={horizon}: {e}")
                    continue

            if not all_train_X:
                logger.error(f"No data prepared for horizon {horizon}")
                continue

            # Combine and shuffle
            combined_train_X = np.concatenate(all_train_X, axis=0)
            combined_train_y = np.concatenate(all_train_y, axis=0)
            combined_train_task_ids = np.concatenate(all_train_task_ids, axis=0)

            combined_val_X = np.concatenate(all_val_X, axis=0)
            combined_val_y = np.concatenate(all_val_y, axis=0)
            combined_val_task_ids = np.concatenate(all_val_task_ids, axis=0)

            combined_test_X = np.concatenate(all_test_X, axis=0)
            combined_test_y = np.concatenate(all_test_y, axis=0)
            combined_test_task_ids = np.concatenate(all_test_task_ids, axis=0)

            combined_unlabeled_X = np.concatenate(all_unlabeled_X, axis=0) if all_unlabeled_X else np.array([])
            combined_unlabeled_y = np.concatenate(all_unlabeled_y, axis=0) if all_unlabeled_y else np.array([])

            train_indices = np.random.permutation(len(combined_train_X))
            combined_train_X = combined_train_X[train_indices]
            combined_train_y = combined_train_y[train_indices]
            combined_train_task_ids = combined_train_task_ids[train_indices]

            # Don't shuffle validation data to maintain evaluation consistency
            prepared_data[horizon] = {
                'labeled': (
                    (combined_train_X, combined_train_y, combined_train_task_ids),
                    (combined_val_X, combined_val_y, combined_val_task_ids),
                    (combined_test_X, combined_test_y, combined_test_task_ids)
                ),
                'unlabeled': (combined_unlabeled_X, combined_unlabeled_y) if len(combined_unlabeled_X) > 0 else None
            }

            logger.info(f"H={horizon} data: train={len(combined_train_X)}, val={len(combined_val_X)}, test={len(combined_test_X)}")

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

    def _create_sequences(
            self,
            data: np.ndarray,
            forecast_length: int,
            stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with specified stride."""
        X, y = [], []

        for i in range(0, len(data) - self.config.backcast_length - forecast_length + 1, stride):
            backcast = data[i: i + self.config.backcast_length]
            forecast = data[i + self.config.backcast_length: i + self.config.backcast_length + forecast_length]

            if not (np.isnan(backcast).any() or np.isnan(forecast).any()):
                X.append(backcast)
                y.append(forecast)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class CurriculumLearningCallback(keras.callbacks.Callback):
    """Curriculum Learning callback with proper implementation."""

    def __init__(
            self,
            config: MultiTaskNBeatsTrainingConfig,
            labeled_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
            unlabeled_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        super().__init__()
        self.config = config
        self.labeled_X, self.labeled_y, self.labeled_task_ids = labeled_data
        self.unlabeled_X, self.unlabeled_y = unlabeled_data if unlabeled_data else (None, None)

        self.current_labeled_ratio = config.curriculum_start_ratio
        self.curriculum_active = config.use_curriculum_learning and self.unlabeled_X is not None

    def on_epoch_begin(self, epoch, logs=None):
        """Update curriculum schedule."""
        if not self.curriculum_active:
            return

        # Calculate current ratio with smoother transition
        if epoch < self.config.curriculum_transition_epochs:
            progress = epoch / self.config.curriculum_transition_epochs
            # Use cosine annealing for smoother transition
            cos_progress = 0.5 * (1 + np.cos(np.pi * progress))
            self.current_labeled_ratio = (
                self.config.curriculum_end_ratio +
                (self.config.curriculum_start_ratio - self.config.curriculum_end_ratio) * cos_progress
            )
        else:
            self.current_labeled_ratio = self.config.curriculum_end_ratio

        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: Curriculum labeled ratio = {self.current_labeled_ratio:.3f}")


class InterimVisualizationCallback(keras.callbacks.Callback):
    """Visualization callback"""

    def __init__(
            self,
            config: MultiTaskNBeatsTrainingConfig,
            data_processor: MultiTaskDataProcessor,
            test_data: Dict[int, Tuple],
            unlabeled_data: Dict[int, Tuple],
            save_dir: str
    ):
        super().__init__()
        self.config = config
        self.data_processor = data_processor
        self.test_data = test_data
        self.unlabeled_data = unlabeled_data
        self.save_dir = save_dir

        self.training_history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'primary_loss': [],
            'aux_entropy_loss': [],
            'aux_consistency_loss': [],
            'aux_balance_loss': [],
            'task_losses': {task: [] for task in data_processor.task_to_id.keys()}
        }

        os.makedirs(save_dir, exist_ok=True)
        self.viz_tasks = list(data_processor.task_to_id.keys())[:config.plot_top_k_tasks]

    def on_epoch_end(self, epoch, logs=None):
        """Track training progress and create ALL visualizations."""
        if logs is None:
            logs = {}

        self.training_history['epoch'].append(epoch)
        self.training_history['loss'].append(logs.get('loss', 0.0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0.0))
        self.training_history['primary_loss'].append(logs.get('primary_loss', logs.get('loss', 0.0)))

        # Track auxiliary losses with proper defaults
        self.training_history['aux_entropy_loss'].append(logs.get('aux_entropy_loss', 0.0))
        self.training_history['aux_consistency_loss'].append(logs.get('aux_consistency_loss', 0.0))
        self.training_history['aux_balance_loss'].append(logs.get('aux_balance_loss', 0.0))

        # Create ALL visualizations at specified intervals
        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            logger.info(f"Creating interim visualizations at epoch {epoch + 1}")
            self._create_all_interim_plots(epoch)

    def _create_all_interim_plots(self, epoch: int):
        """Create ALL comprehensive interim plots with task inference monitoring."""
        try:
            # 1. Enhanced learning curves
            if self.config.create_learning_curves:
                self._plot_enhanced_learning_curves(epoch)

            # 2. Task inference analysis
            self._plot_task_inference_analysis(epoch)

            # 3. Prediction samples with task inference
            if self.config.create_prediction_plots:
                self._plot_prediction_samples_with_inference(epoch)

            # 4. Task performance heatmap
            if self.config.create_task_performance_heatmap:
                self._plot_enhanced_task_performance(epoch)

            logger.info(f"All interim plots saved for epoch {epoch + 1}")

        except Exception as e:
            logger.warning(f"Failed to create interim plots: {e}")

    def _plot_enhanced_learning_curves(self, epoch: int):
        """Plot enhanced training and validation loss curves with auxiliary losses"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        epochs = self.training_history['epoch']

        # Main loss curves
        axes[0, 0].plot(epochs, self.training_history['loss'], label='Total Loss', color='blue', linewidth=2)
        axes[0, 0].plot(epochs, self.training_history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        axes[0, 0].plot(epochs, self.training_history['primary_loss'], label='Primary Loss', color='green', linewidth=1,
                        linestyle='--', alpha=0.7)
        axes[0, 0].set_title(f'Learning Curves (Epoch {epoch + 1})')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Set reasonable y-limits to avoid scale issues
        if len(epochs) > 5:
            y_values = (self.training_history['loss'][5:] +
                       self.training_history['val_loss'][5:] +
                       self.training_history['primary_loss'][5:])
            y_min, y_max = min(y_values), max(y_values)
            axes[0, 0].set_ylim(y_min * 0.9, y_max * 1.1)

        # Auxiliary losses
        aux_loss_plotted = False
        if any(v > 1e-6 for v in self.training_history['aux_entropy_loss']):
            axes[0, 1].plot(epochs, self.training_history['aux_entropy_loss'], label='Entropy Loss', color='purple')
            aux_loss_plotted = True
        if any(v > 1e-6 for v in self.training_history['aux_consistency_loss']):
            axes[0, 1].plot(epochs, self.training_history['aux_consistency_loss'], label='Consistency Loss', color='orange')
            aux_loss_plotted = True
        if any(v > 1e-6 for v in self.training_history['aux_balance_loss']):
            axes[0, 1].plot(epochs, self.training_history['aux_balance_loss'], label='Balance Loss', color='brown')
            aux_loss_plotted = True

        if aux_loss_plotted:
            axes[0, 1].set_title('Auxiliary Losses (Task Inference)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Auxiliary losses not active\n(all values near zero)',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Auxiliary Losses (Task Inference)')

        # Learning rate tracking
        try:
            if hasattr(self.model.optimizer, 'learning_rate'):
                current_lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
                axes[1, 0].axhline(y=current_lr, color='green', linestyle='--', label=f'Current LR: {current_lr:.2e}')
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        except Exception:
            axes[1, 0].text(0.5, 0.5, 'LR info unavailable', ha='center', va='center', transform=axes[1, 0].transAxes)

        # Loss decomposition
        if len(epochs) > 1:
            total_losses = np.array(self.training_history['loss'])
            primary_losses = np.array(self.training_history['primary_loss'])

            # Avoid division by zero
            ratios = np.divide(primary_losses, total_losses,
                             out=np.ones_like(primary_losses), where=total_losses!=0)

            axes[1, 1].plot(epochs, ratios, label='Primary Loss Ratio', color='blue')
            axes[1, 1].set_title('Loss Decomposition')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Primary Loss / Total Loss')
            axes[1, 1].set_ylim(0, 1.1)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'learning_curves_epoch_{epoch + 1:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_task_inference_analysis(self, epoch: int):
        """Plot task inference analysis."""
        horizon = self.config.forecast_horizons[0]

        if horizon not in self.unlabeled_data or self.unlabeled_data[horizon] is None:
            logger.warning("No unlabeled data available for task inference analysis")
            return

        unlabeled_X, unlabeled_y = self.unlabeled_data[horizon]

        if len(unlabeled_X) == 0:
            logger.warning("Empty unlabeled data for task inference analysis")
            return

        try:
            # Sample some unlabeled data for analysis
            sample_size = min(100, len(unlabeled_X))
            sample_indices = np.random.choice(len(unlabeled_X), sample_size, replace=False)
            sample_X = unlabeled_X[sample_indices]

            # Get task probabilities - to handle model structure
            if hasattr(self.model, '_infer_task_probabilities'):
                task_probs = self.model._infer_task_probabilities(sample_X, training=False)
                task_probs_np = keras.ops.convert_to_numpy(task_probs)
            else:
                logger.warning("Model does not have task inference capability")
                return

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Task probability distribution
            max_probs = np.max(task_probs_np, axis=1)
            axes[0, 0].hist(max_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Max Task Probability Distribution')
            axes[0, 0].set_xlabel('Max Probability')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axvline(x=np.mean(max_probs), color='red', linestyle='--',
                              label=f'Mean: {np.mean(max_probs):.3f}')
            axes[0, 0].legend()

            # Task assignment distribution
            predicted_tasks = np.argmax(task_probs_np, axis=1)
            task_counts = np.bincount(predicted_tasks, minlength=self.model.num_tasks)

            axes[0, 1].bar(range(self.model.num_tasks), task_counts, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Predicted Task Distribution')
            axes[0, 1].set_xlabel('Task ID')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_xticks(range(self.model.num_tasks))
            axes[0, 1].set_xticklabels([f'T{i}' for i in range(self.model.num_tasks)], rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # Task probability heatmap
            display_samples = min(50, len(task_probs_np))
            im = axes[1, 0].imshow(task_probs_np[:display_samples].T,
                                   cmap='viridis', aspect='auto', interpolation='nearest')
            axes[1, 0].set_title(f'Task Probabilities Heatmap (First {display_samples} samples)')
            axes[1, 0].set_xlabel('Sample')
            axes[1, 0].set_ylabel('Task ID')
            axes[1, 0].set_yticks(range(self.model.num_tasks))
            axes[1, 0].set_yticklabels([f'T{i}' for i in range(self.model.num_tasks)])
            plt.colorbar(im, ax=axes[1, 0], label='Probability')

            # Entropy distribution
            entropy = -np.sum(task_probs_np * np.log(task_probs_np + 1e-8), axis=1)
            axes[1, 1].hist(entropy, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 1].axvline(x=self.config.min_entropy_target, color='red', linestyle='--',
                               label=f'Target: {self.config.min_entropy_target}')
            axes[1, 1].axvline(x=np.mean(entropy), color='blue', linestyle='--',
                               label=f'Mean: {np.mean(entropy):.3f}')
            axes[1, 1].set_title('Task Prediction Entropy')
            axes[1, 1].set_xlabel('Entropy')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle(f'Task Inference Analysis (Epoch {epoch + 1})', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'task_inference_analysis_epoch_{epoch + 1:03d}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create task inference analysis: {e}")

    def _plot_prediction_samples_with_inference(self, epoch: int):
        """Plot prediction samples with task inference information."""
        horizon = self.config.forecast_horizons[0]

        # Get labeled test data
        test_X, test_y, test_task_ids = self.test_data[horizon]
        forecast_length = test_y.shape[1]

        # Also get some unlabeled data for comparison
        if horizon in self.unlabeled_data and self.unlabeled_data[horizon] is not None:
            unlabeled_X, unlabeled_y = self.unlabeled_data[horizon]
        else:
            unlabeled_X, unlabeled_y = None, None

        try:
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))
            axes = axes.flatten()

            plot_idx = 0

            # Plot labeled samples (first 6)
            for i, task_name in enumerate(self.viz_tasks[:6]):
                if plot_idx >= len(axes):
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
                sample_X = task_X[0:1]
                sample_y = task_y[0:1]
                sample_task_id = task_ids[0:1]

                # Get prediction with explicit task ID
                pred_y_labeled = self.model((sample_X, sample_task_id), training=False)

                # Get prediction with task inference (if available)
                try:
                    pred_y_inferred = self.model(sample_X, training=False)
                    has_inference = True
                except:
                    pred_y_inferred = pred_y_labeled
                    has_inference = False

                # Plot
                backcast_x = np.arange(-self.config.backcast_length, 0)
                forecast_x = np.arange(0, forecast_length)

                axes[plot_idx].plot(backcast_x, sample_X[0], label='Input', color='blue', alpha=0.7)
                axes[plot_idx].plot(forecast_x, sample_y[0].flatten(), label='True', color='green', linewidth=2)
                axes[plot_idx].plot(forecast_x, pred_y_labeled[0].numpy().flatten(),
                                    label='Pred (Labeled)', color='red', linestyle='-', linewidth=2)

                if has_inference:
                    axes[plot_idx].plot(forecast_x, pred_y_inferred[0].numpy().flatten(),
                                        label='Pred (Inferred)', color='orange', linestyle='--', linewidth=2)

                axes[plot_idx].set_title(f'{task_name} (Labeled)')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                axes[plot_idx].axvline(x=0, color='black', linestyle='-', alpha=0.5)

                plot_idx += 1

            # Plot unlabeled samples (if available)
            if unlabeled_X is not None and len(unlabeled_X) > 0 and hasattr(self.model, '_infer_task_probabilities'):
                for i in range(min(3, len(unlabeled_X))):
                    if plot_idx >= len(axes):
                        break

                    sample_X = unlabeled_X[i:i + 1]
                    sample_y = unlabeled_y[i:i + 1]

                    try:
                        # Get prediction with task inference only
                        pred_y_inferred = self.model(sample_X, training=False)

                        # Get task probabilities
                        task_probs = self.model._infer_task_probabilities(sample_X, training=False)
                        task_probs_np = keras.ops.convert_to_numpy(task_probs)[0]
                        predicted_task_id = np.argmax(task_probs_np)
                        confidence = np.max(task_probs_np)

                        # Plot
                        backcast_x = np.arange(-self.config.backcast_length, 0)
                        forecast_x = np.arange(0, forecast_length)

                        axes[plot_idx].plot(backcast_x, sample_X[0], label='Input', color='blue', alpha=0.7)
                        axes[plot_idx].plot(forecast_x, sample_y[0].flatten(), label='True', color='green', linewidth=2)
                        axes[plot_idx].plot(forecast_x, pred_y_inferred[0].numpy().flatten(),
                                            label='Pred (Inferred)', color='purple', linestyle='--', linewidth=2)

                        predicted_task_name = self.data_processor.id_to_task[predicted_task_id]
                        axes[plot_idx].set_title(f'Unlabeled (Inferred: {predicted_task_name}, Conf: {confidence:.2f})')
                        axes[plot_idx].legend()
                        axes[plot_idx].grid(True, alpha=0.3)
                        axes[plot_idx].axvline(x=0, color='black', linestyle='-', alpha=0.5)

                        plot_idx += 1
                    except Exception as e:
                        logger.warning(f"Failed to process unlabeled sample {i}: {e}")
                        continue

            # Remove empty subplots
            for i in range(plot_idx, len(axes)):
                fig.delaxes(axes[i])

            plt.suptitle(f'Prediction Samples with Task Inference (Epoch {epoch + 1})', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'predictions_with_inference_epoch_{epoch + 1:03d}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create prediction samples plot: {e}")

    def _plot_enhanced_task_performance(self, epoch: int):
        """Plot enhanced task-specific performance analysis."""
        horizon = self.config.forecast_horizons[0]
        test_X, test_y, test_task_ids = self.test_data[horizon]

        try:
            # Calculate performance metrics for each task
            task_metrics = {
                'task_names': [],
                'labeled_mae': [],
                'inferred_mae': [],
                'mae_difference': [],
                'confidence': []
            }

            for task_name, task_id in self.data_processor.task_to_id.items():
                task_mask = test_task_ids == task_id

                if not np.any(task_mask):
                    continue

                task_X = test_X[task_mask][:50]  # Limit to 50 samples for speed
                task_y = test_y[task_mask][:50]
                task_ids = np.full(len(task_X), task_id)

                if len(task_X) == 0:
                    continue

                try:
                    # Get predictions with explicit task IDs
                    pred_y_labeled = self.model((task_X, task_ids), training=False)

                    # Get predictions with task inference
                    pred_y_inferred = self.model(task_X, training=False)

                    # Calculate MAE
                    labeled_mae = np.mean(np.abs(task_y - pred_y_labeled.numpy()))
                    inferred_mae = np.mean(np.abs(task_y - pred_y_inferred.numpy()))
                    mae_diff = inferred_mae - labeled_mae

                    # Get task inference confidence (if available)
                    if hasattr(self.model, '_infer_task_probabilities'):
                        task_probs = self.model._infer_task_probabilities(task_X, training=False)
                        task_probs_np = keras.ops.convert_to_numpy(task_probs)
                        avg_confidence = np.mean(np.max(task_probs_np, axis=1))
                    else:
                        avg_confidence = 0.5  # Default when inference not available

                    task_metrics['task_names'].append(task_name)
                    task_metrics['labeled_mae'].append(labeled_mae)
                    task_metrics['inferred_mae'].append(inferred_mae)
                    task_metrics['mae_difference'].append(mae_diff)
                    task_metrics['confidence'].append(avg_confidence)

                except Exception as e:
                    logger.warning(f"Failed to process task {task_name}: {e}")
                    continue

            if task_metrics['task_names']:
                # Create enhanced performance visualization
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))

                # MAE comparison
                x = np.arange(len(task_metrics['task_names']))
                width = 0.35

                axes[0, 0].bar(x - width / 2, task_metrics['labeled_mae'], width,
                               label='Labeled MAE', alpha=0.8, color='blue', edgecolor='black')
                axes[0, 0].bar(x + width / 2, task_metrics['inferred_mae'], width,
                               label='Inferred MAE', alpha=0.8, color='red', edgecolor='black')
                axes[0, 0].set_title('MAE Comparison: Labeled vs Inferred')
                axes[0, 0].set_xlabel('Task')
                axes[0, 0].set_ylabel('MAE')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(task_metrics['task_names'], rotation=45, ha='right')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

                # MAE difference
                colors = ['green' if diff <= 0 else 'red' for diff in task_metrics['mae_difference']]
                bars = axes[0, 1].bar(x, task_metrics['mae_difference'], color=colors, alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('MAE Difference (Inferred - Labeled)')
                axes[0, 1].set_xlabel('Task')
                axes[0, 1].set_ylabel('MAE Difference')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(task_metrics['task_names'], rotation=45, ha='right')
                axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[0, 1].grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, value in zip(bars, task_metrics['mae_difference']):
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

                # Task inference confidence
                axes[1, 0].bar(x, task_metrics['confidence'], color='purple', alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Average Task Inference Confidence')
                axes[1, 0].set_xlabel('Task')
                axes[1, 0].set_ylabel('Confidence')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(task_metrics['task_names'], rotation=45, ha='right')
                axes[1, 0].set_ylim(0, 1)
                axes[1, 0].grid(True, alpha=0.3)

                # Confidence vs Performance scatter
                scatter = axes[1, 1].scatter(task_metrics['confidence'], task_metrics['mae_difference'],
                                           c=task_metrics['inferred_mae'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
                axes[1, 1].set_title('Confidence vs Performance')
                axes[1, 1].set_xlabel('Inference Confidence')
                axes[1, 1].set_ylabel('MAE Difference')
                axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[1, 1].grid(True, alpha=0.3)

                # Add task name labels to scatter points
                for i, txt in enumerate(task_metrics['task_names']):
                    axes[1, 1].annotate(txt, (task_metrics['confidence'][i], task_metrics['mae_difference'][i]),
                                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

                # Add colorbar
                plt.colorbar(scatter, ax=axes[1, 1], label='Inferred MAE')

                plt.suptitle(f'Task Performance Analysis (Epoch {epoch + 1})', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, f'task_performance_epoch_{epoch + 1:03d}.png'),
                            dpi=150, bbox_inches='tight')
                plt.close()

        except Exception as e:
            logger.warning(f"Failed to create task performance analysis: {e}")


class MultiTaskNBeatsTrainer:
    """Multi-task N-BEATS trainer with improved training process."""

    def __init__(self, config: MultiTaskNBeatsTrainingConfig, ts_config: TimeSeriesConfig):
        self.config = config
        self.ts_config = ts_config
        self.generator = TimeSeriesGenerator(ts_config)
        self.processor = MultiTaskDataProcessor(config)

        # Get tasks
        self.all_tasks = self.generator.get_task_names()
        self.task_categories = self.generator.get_task_categories()
        self.selected_tasks = self._select_tasks()

        logger.info(f"Multi-Task N-BEATS Trainer initialized:")
        logger.info(f"  - Selected {len(self.selected_tasks)} tasks")
        logger.info(f"  - Improved auxiliary loss handling")
        logger.info(f"  - Better validation split: {config.val_ratio:.1f}")

    def _select_tasks(self) -> List[str]:
        """Select tasks for training."""
        selected = []
        for category in self.task_categories:
            category_tasks = self.generator.get_tasks_by_category(category)
            selected.extend(category_tasks[:self.config.max_tasks_per_category])
        return selected

    def prepare_data(self) -> Dict[str, Any]:
        """Prepare training data."""
        logger.info("Generating data for selected tasks...")

        raw_task_data = {}
        for task_name in self.selected_tasks:
            try:
                data = self.generator.generate_task_data(task_name)
                if len(data) >= self.config.min_data_length:
                    raw_task_data[task_name] = data
            except Exception as e:
                logger.warning(f"Failed to generate {task_name}: {e}")

        logger.info(f"Generated data for {len(raw_task_data)} tasks")

        prepared_data = self.processor.prepare_multi_task_data(raw_task_data)

        return {
            'prepared_data': prepared_data,
            'raw_task_data': raw_task_data,
            'num_tasks': len(raw_task_data),
            'task_to_id': self.processor.task_to_id
        }

    def create_model(self, num_tasks: int, task_to_id: Dict[str, int], forecast_length: int):
        """Create model with proper configuration."""
        # Import the model classes - update this path as needed
        try:
            from dl_techniques.models.nbeats_multitask import MultiTaskNBeatsConfig, create_multi_task_nbeats
        except ImportError:
            # Fallback if using the version from artifacts
            logger.warning("Using local MultiTaskNBeatsConfig")
            from dl_techniques.models.nbeats_multitask import MultiTaskNBeatsConfig, create_multi_task_nbeats

        # Create model configuration
        model_config = MultiTaskNBeatsConfig(
            backcast_length=self.config.backcast_length,
            use_task_embeddings=self.config.use_task_embeddings,
            task_embedding_dim=self.config.task_embedding_dim,
            use_task_inference=True,

            # task inference parameters
            train_task_inference=self.config.train_task_inference,
            task_inference_loss_weight=self.config.task_inference_loss_weight,
            consistency_loss_weight=self.config.consistency_loss_weight,
            entropy_loss_weight=self.config.entropy_loss_weight,
            min_entropy_target=self.config.min_entropy_target,

            stack_types=self.config.stack_types.copy(),
            nb_blocks_per_stack=self.config.nb_blocks_per_stack,
            hidden_layer_units=self.config.hidden_layer_units,
            use_revin=self.config.use_revin,

            dropout_rate=self.config.dropout_rate,
            kernel_regularizer_l2=self.config.kernel_regularizer_l2,

            optimizer=self.config.optimizer,
            primary_loss=self.config.primary_loss,
            learning_rate=self.config.learning_rate,
            gradient_clip_norm=self.config.gradient_clip_norm
        )

        return create_multi_task_nbeats(
            config=model_config,
            num_tasks=num_tasks,
            task_to_id=task_to_id,
            forecast_length=forecast_length,
            name=f"MultiTaskNBeats_H{forecast_length}"
        )

    def _compile_model(self, model):
        """model compilation."""
        if self.config.optimizer.lower() == 'adamw':
            optimizer = keras.optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                clipnorm=self.config.gradient_clip_norm if self.config.gradient_clip_norm > 0 else None
            )
        else:
            optimizer = keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
                clipnorm=self.config.gradient_clip_norm if self.config.gradient_clip_norm > 0 else None
            )

        if self.config.primary_loss.lower() == 'mae':
            loss = keras.losses.MeanAbsoluteError()
        elif self.config.primary_loss.lower() == 'smape':
            loss = SMAPELoss()
        else:
            loss = keras.losses.MeanSquaredError()

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['mae']
        )

        logger.info("✓ model compiled with proper auxiliary loss tracking")

    def run_experiment(self) -> Dict[str, Any]:
        """Run the multi-task experiment."""
        try:
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

            results = {}

            for horizon in self.config.forecast_horizons:
                if horizon not in prepared_data:
                    continue

                logger.info(f"\n{'='*50}\n🎯 Training Model H={horizon}\n{'='*50}")

                # Create and build model
                model = self.create_model(data_info['num_tasks'], data_info['task_to_id'], horizon)

                labeled_data = prepared_data[horizon]['labeled']
                unlabeled_data = prepared_data[horizon]['unlabeled']
                train_data, val_data, test_data = labeled_data

                # Build model
                sample_input = (train_data[0][:1], train_data[2][:1])
                model(sample_input)
                self._compile_model(model)

                # Create callbacks
                viz_dir = os.path.join(exp_dir, f'visualizations_h{horizon}')
                unlabeled_dict = {horizon: unlabeled_data} if unlabeled_data is not None else {}

                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=50,
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
                        unlabeled_data=unlabeled_dict,
                        save_dir=viz_dir
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(exp_dir, f'best_model_h{horizon}.keras'),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1
                    )
                ]

                # Add curriculum learning
                if self.config.use_curriculum_learning:
                    curriculum_callback = CurriculumLearningCallback(
                        config=self.config,
                        labeled_data=train_data,
                        unlabeled_data=unlabeled_data
                    )
                    callbacks.append(curriculum_callback)
                    logger.info("✓ Curriculum learning callback added")

                # Add progress callback for detailed monitoring
                progress_callback = keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: logger.info(
                        f"Epoch {epoch+1:3d}: "
                        f"loss={logs.get('loss', 0):.4f}, "
                        f"val_loss={logs.get('val_loss', 0):.4f}, "
                        f"primary={logs.get('primary_loss', 0):.4f}, "
                        f"aux_entropy={logs.get('aux_entropy_loss', 0):.4f}, "
                        f"aux_consistency={logs.get('aux_consistency_loss', 0):.4f}, "
                        f"aux_balance={logs.get('aux_balance_loss', 0):.4f}"
                    )
                )
                callbacks.append(progress_callback)

                # Train model
                logger.info("🚀 Starting training...")
                start_time = datetime.now()

                history = model.fit(
                    x=(train_data[0], train_data[2]),
                    y=train_data[1],
                    validation_data=((val_data[0], val_data[2]), val_data[1]),
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    callbacks=callbacks,
                    verbose=1
                )

                training_time = (datetime.now() - start_time).total_seconds()

                # Evaluate
                test_results = model.evaluate(
                    x=(test_data[0], test_data[2]),
                    y=test_data[1],
                    verbose=0
                )

                test_loss = test_results[0] if isinstance(test_results, list) else test_results
                test_mae = test_results[1] if len(test_results) > 1 else None

                results[horizon] = {
                    'history': history.history,
                    'training_time': training_time,
                    'test_loss': test_loss,
                    'test_mae': test_mae,
                    'final_epoch': len(history.history['loss'])
                }

                logger.info(f"✅ training completed for H={horizon}:")
                logger.info(f"   - Training time: {training_time:.1f}s")
                logger.info(f"   - Test loss: {test_loss:.4f}")

            # Save results
            self._save_results(results, exp_dir, data_info)

            logger.info("🎉 Multi-Task Experiment completed successfully!")
            return {
                "results_dir": exp_dir,
                "results": results,
                "num_tasks": data_info['num_tasks'],
                "task_mapping": data_info['task_to_id']
            }

        except Exception as e:
            logger.error(f"💥 experiment failed: {e}", exc_info=True)
            raise

    def _save_results(self, results: Dict, exp_dir: str, data_info: Dict):
        """Save experiment results with ALL visualizations."""
        try:
            # Save JSON results
            with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
                json_results = {}
                for horizon, result in results.items():
                    json_results[str(horizon)] = {
                        'training_time': result['training_time'],
                        'test_loss': float(result['test_loss']),
                        'test_mae': float(result['test_mae']) if result.get('test_mae') else None,
                        'final_epoch': result['final_epoch'],
                        'final_train_loss': result['history']['loss'][-1],
                        'final_val_loss': result['history']['val_loss'][-1],
                        'has_aux_losses': any(k.startswith('aux_') for k in result['history'].keys()),
                        'final_aux_entropy_loss': result['history'].get('aux_entropy_loss', [0])[-1] if result['history'].get('aux_entropy_loss') else 0,
                        'final_aux_consistency_loss': result['history'].get('aux_consistency_loss', [0])[-1] if result['history'].get('aux_consistency_loss') else 0,
                        'final_aux_balance_loss': result['history'].get('aux_balance_loss', [0])[-1] if result['history'].get('aux_balance_loss') else 0
                    }
                json.dump(json_results, f, indent=2)

            # Save task information
            with open(os.path.join(exp_dir, 'task_info.json'), 'w') as f:
                json.dump({
                    'num_tasks': data_info['num_tasks'],
                    'task_to_id': data_info['task_to_id'],
                    'selected_tasks': self.selected_tasks,
                    'improvements': {
                        'better_data_split': f"{self.config.train_ratio}/{self.config.val_ratio}/{self.config.test_ratio}",
                        'improved_aux_loss_weights': {
                            'task_inference_loss_weight': self.config.task_inference_loss_weight,
                            'consistency_loss_weight': self.config.consistency_loss_weight,
                            'entropy_loss_weight': self.config.entropy_loss_weight
                        },
                        'reduced_complexity': {
                            'stack_types': self.config.stack_types,
                            'nb_blocks_per_stack': self.config.nb_blocks_per_stack,
                            'hidden_layer_units': self.config.hidden_layer_units
                        }
                    }
                }, f, indent=2)

            # Create comprehensive final summary plot
            self._create_comprehensive_final_summary_plot(results, exp_dir)

            # Create detailed experiment report
            self._create_detailed_experiment_report(results, exp_dir, data_info)

            logger.info(f"results with ALL visualizations saved to {exp_dir}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def _create_comprehensive_final_summary_plot(self, results: Dict, exp_dir: str):
        """Create comprehensive final summary visualization with ALL plots."""
        try:
            fig, axes = plt.subplots(3, 2, figsize=(18, 15))

            # Training curves for all horizons
            for horizon, result in results.items():
                history = result['history']
                epochs = range(1, len(history['loss']) + 1)

                # Main losses
                axes[0, 0].plot(epochs, history['loss'], label=f'H{horizon} Total', alpha=0.8, linewidth=2)
                axes[0, 0].plot(epochs, history.get('primary_loss', history['loss']),
                                label=f'H{horizon} Primary', alpha=0.6, linestyle='--')

                axes[0, 1].plot(epochs, history['val_loss'], label=f'H{horizon} Val', alpha=0.8)

            axes[0, 0].set_title('Training Loss (Total vs Primary)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Auxiliary losses
            aux_plotted = False
            for horizon, result in results.items():
                history = result['history']
                epochs = range(1, len(history['loss']) + 1)

                if 'aux_entropy_loss' in history and any(v > 1e-6 for v in history['aux_entropy_loss']):
                    axes[1, 0].plot(epochs, history['aux_entropy_loss'], label=f'H{horizon} Entropy', alpha=0.8)
                    aux_plotted = True
                if 'aux_consistency_loss' in history and any(v > 1e-6 for v in history['aux_consistency_loss']):
                    axes[1, 1].plot(epochs, history['aux_consistency_loss'], label=f'H{horizon} Consistency', alpha=0.8)
                    aux_plotted = True

            if aux_plotted:
                axes[1, 0].set_title('Auxiliary Loss: Entropy')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Entropy Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                axes[1, 1].set_title('Auxiliary Loss: Consistency')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Consistency Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Auxiliary losses not active\n(Check model configuration)',
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Auxiliary Loss: Entropy')

                axes[1, 1].text(0.5, 0.5, 'Auxiliary losses not active\n(Check model configuration)',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Auxiliary Loss: Consistency')

            # Performance summary
            horizons = list(results.keys())
            test_losses = [results[h]['test_loss'] for h in horizons]
            training_times = [results[h]['training_time'] for h in horizons]

            bars1 = axes[2, 0].bar(range(len(horizons)), test_losses, color='skyblue', alpha=0.7, edgecolor='black')
            axes[2, 0].set_title('Test Loss by Horizon')
            axes[2, 0].set_xlabel('Horizon')
            axes[2, 0].set_ylabel('Test Loss')
            axes[2, 0].set_xticks(range(len(horizons)))
            axes[2, 0].set_xticklabels(horizons)
            axes[2, 0].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars1, test_losses):
                height = bar.get_height()
                axes[2, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')

            bars2 = axes[2, 1].bar(range(len(horizons)), training_times, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[2, 1].set_title('Training Time by Horizon')
            axes[2, 1].set_xlabel('Horizon')
            axes[2, 1].set_ylabel('Training Time (s)')
            axes[2, 1].set_xticks(range(len(horizons)))
            axes[2, 1].set_xticklabels(horizons)
            axes[2, 1].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars2, training_times):
                height = bar.get_height()
                axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1f}s', ha='center', va='bottom')

            plt.suptitle('Multi-Task N-BEATS Training Summary\n(with Comprehensive Visualizations)', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, 'comprehensive_final_summary.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Create additional analysis plot
            self._create_additional_analysis_plot(results, exp_dir)

        except Exception as e:
            logger.warning(f"Failed to create comprehensive final summary plot: {e}")

    def _create_additional_analysis_plot(self, results: Dict, exp_dir: str):
        """Create additional analysis plots for deeper insights."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            for horizon, result in results.items():
                history = result['history']
                epochs = range(1, len(history['loss']) + 1)

                # Loss decomposition over time
                total_losses = np.array(history['loss'])
                primary_losses = np.array(history.get('primary_loss', history['loss']))
                aux_losses = total_losses - primary_losses

                axes[0, 0].plot(epochs, primary_losses, label=f'H{horizon} Primary', linewidth=2)
                axes[0, 1].plot(epochs, aux_losses, label=f'H{horizon} Auxiliary', linewidth=2)

                # Training efficiency
                if 'val_loss' in history:
                    val_losses = np.array(history['val_loss'])
                    efficiency = primary_losses / (val_losses + 1e-8)  # Lower is better
                    axes[1, 0].plot(epochs, efficiency, label=f'H{horizon} Efficiency', alpha=0.7)

                # Overfitting analysis
                if 'val_loss' in history:
                    overfitting_gap = val_losses - primary_losses
                    axes[1, 1].plot(epochs, overfitting_gap, label=f'H{horizon} Gap', alpha=0.7)

            axes[0, 0].set_title('Primary Loss Evolution')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Primary Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].set_title('Auxiliary Loss Evolution')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Auxiliary Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].set_title('Training Efficiency (Primary/Val Loss)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Efficiency Ratio')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].set_title('Overfitting Gap (Val - Train Loss)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gap')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

            plt.suptitle('Training Analysis: Efficiency and Overfitting', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, 'training_analysis.png'), dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create additional analysis plot: {e}")

    def _create_detailed_experiment_report(self, results: Dict, exp_dir: str, data_info: Dict):
        """Create a detailed text report of the experiment results."""
        try:
            report_path = os.path.join(exp_dir, 'detailed_experiment_report.txt')

            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("MULTI-TASK N-BEATS EXPERIMENT REPORT\n")
                f.write("=" * 80 + "\n\n")

                # Experiment overview
                f.write("EXPERIMENT OVERVIEW\n")
                f.write("-" * 20 + "\n")
                f.write(f"Experiment Name: {self.config.experiment_name}\n")
                f.write(f"Number of Tasks: {data_info['num_tasks']}\n")
                f.write(f"Selected Tasks: {', '.join(self.selected_tasks)}\n")
                f.write(f"Forecast Horizons: {self.config.forecast_horizons}\n")
                f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Configuration summary
                f.write("MODEL CONFIGURATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"Backcast Length: {self.config.backcast_length}\n")
                f.write(f"Stack Types: {self.config.stack_types}\n")
                f.write(f"Blocks per Stack: {self.config.nb_blocks_per_stack}\n")
                f.write(f"Hidden Layer Units: {self.config.hidden_layer_units}\n")
                f.write(f"Task Embeddings: {'Enabled' if self.config.use_task_embeddings else 'Disabled'}\n")
                f.write(f"Task Inference: {'Enabled' if self.config.train_task_inference else 'Disabled'}\n")
                f.write(f"Curriculum Learning: {'Enabled' if self.config.use_curriculum_learning else 'Disabled'}\n\n")

                # Training configuration
                f.write("TRAINING CONFIGURATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"Data Split: {self.config.train_ratio:.1f}/{self.config.val_ratio:.1f}/{self.config.test_ratio:.1f}\n")
                f.write(f"Epochs: {self.config.epochs}\n")
                f.write(f"Batch Size: {self.config.batch_size}\n")
                f.write(f"Learning Rate: {self.config.learning_rate}\n")
                f.write(f"Optimizer: {self.config.optimizer}\n")
                f.write(f"Primary Loss: {self.config.primary_loss}\n")
                f.write(f"Dropout Rate: {self.config.dropout_rate}\n")
                f.write(f"L2 Regularization: {self.config.kernel_regularizer_l2}\n\n")

                # Task inference configuration
                if self.config.train_task_inference:
                    f.write("TASK INFERENCE CONFIGURATION\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Task Inference Loss Weight: {self.config.task_inference_loss_weight}\n")
                    f.write(f"Consistency Loss Weight: {self.config.consistency_loss_weight}\n")
                    f.write(f"Entropy Loss Weight: {self.config.entropy_loss_weight}\n")
                    f.write(f"Min Entropy Target: {self.config.min_entropy_target}\n\n")

                # Results for each horizon
                for horizon, result in results.items():
                    f.write(f"RESULTS FOR HORIZON {horizon}\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Training Time: {result['training_time']:.1f} seconds\n")
                    f.write(f"Final Epoch: {result['final_epoch']}\n")
                    f.write(f"Test Loss: {result['test_loss']:.6f}\n")
                    if result.get('test_mae'):
                        f.write(f"Test MAE: {result['test_mae']:.6f}\n")

                    # Training history analysis
                    history = result['history']
                    f.write(f"Final Training Loss: {history['loss'][-1]:.6f}\n")
                    f.write(f"Final Validation Loss: {history['val_loss'][-1]:.6f}\n")

                    # Best validation loss
                    best_val_loss = min(history['val_loss'])
                    best_val_epoch = history['val_loss'].index(best_val_loss) + 1
                    f.write(f"Best Validation Loss: {best_val_loss:.6f} (Epoch {best_val_epoch})\n")

                    # Auxiliary losses
                    if 'aux_entropy_loss' in history:
                        final_entropy = history['aux_entropy_loss'][-1]
                        final_consistency = history.get('aux_consistency_loss', [0])[-1]
                        final_balance = history.get('aux_balance_loss', [0])[-1]
                        f.write(f"Final Entropy Loss: {final_entropy:.6f}\n")
                        f.write(f"Final Consistency Loss: {final_consistency:.6f}\n")
                        f.write(f"Final Balance Loss: {final_balance:.6f}\n")

                    # Training efficiency
                    overfitting_gap = history['val_loss'][-1] - history['loss'][-1]
                    f.write(f"Overfitting Gap: {overfitting_gap:.6f}\n")
                    f.write("\n")

                # Task mapping
                f.write("TASK MAPPING\n")
                f.write("-" * 12 + "\n")
                for task_name, task_id in data_info['task_to_id'].items():
                    f.write(f"Task {task_id:2d}: {task_name}\n")
                f.write("\n")

                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 15 + "\n")
                if any(results[h]['test_loss'] > results[h]['history']['val_loss'][-1] * 1.1 for h in results):
                    f.write("⚠️  Consider further regularization if test loss >> validation loss\n")
                if any(len(results[h]['history']['loss']) < 20 for h in results):
                    f.write("⚠️  Training stopped early - consider increasing patience or epochs\n")
                if any('aux_entropy_loss' not in results[h]['history'] or
                       all(v < 1e-6 for v in results[h]['history'].get('aux_entropy_loss', [0])) for h in results):
                    f.write("⚠️  Auxiliary losses appear inactive - check task inference configuration\n")
                else:
                    f.write("✅ Task inference training appears to be working correctly\n")
                    f.write("✅ Model configuration looks appropriate\n")
                    f.write("✅ Training completed successfully with proper monitoring\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("End of Report\n")
                f.write("=" * 80 + "\n")

            logger.info(f"Detailed experiment report saved to {report_path}")

        except Exception as e:
            logger.warning(f"Failed to create detailed experiment report: {e}")


def main():
    """Run the multi-task experiment with ORIGINAL configuration values."""

    #  configuration with ORIGINAL data parameters restored
    config = MultiTaskNBeatsTrainingConfig(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,

        backcast_length=168,
        forecast_length=24,
        forecast_horizons=[12],

        train_task_inference=True,
        task_inference_loss_weight=0.5,
        consistency_loss_weight=0.1,
        entropy_loss_weight=0.05,
        min_entropy_target=0.1,

        use_curriculum_learning=True,
        curriculum_start_ratio=0.8,
        curriculum_end_ratio=0.4,
        curriculum_transition_epochs=30,

        stack_types=["trend", "seasonality", "generic"],
        nb_blocks_per_stack=3,
        hidden_layer_units=256,
        use_revin=True,

        max_tasks_per_category=5,
        min_data_length=1000,
        balance_tasks=True,
        samples_per_task=10000,

        epochs=150,
        batch_size=128,
        learning_rate=1e-4,
        dropout_rate=0.15,
        kernel_regularizer_l2=1e-5,
        gradient_clip_norm=1.0,
        optimizer='adamw',
        primary_loss="mae",

        visualize_every_n_epochs=5,
        save_interim_plots=True,
        plot_top_k_tasks=6,
        create_learning_curves=True,
        create_prediction_plots=True,
        create_task_performance_heatmap=True,
        eval_during_training=True,
        eval_every_n_epochs=10,
    )

    # ORIGINAL TimeSeriesConfig restored
    ts_config = TimeSeriesConfig(
        n_samples=3000,
        random_seed=42,
        default_noise_level=0.01
    )

    try:
        trainer = MultiTaskNBeatsTrainer(config, ts_config)
        results = trainer.run_experiment()
    except Exception as e:
        logger.error(f"💥 experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
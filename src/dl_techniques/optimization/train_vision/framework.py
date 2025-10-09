"""
Model-agnostic training framework for vision tasks with integrated
visualization and analysis capabilities.

This framework provides a complete solution for training, monitoring,
and analyzing computer vision models with minimal boilerplate code.
It integrates with the dl_techniques optimization module for advanced
optimizer and learning rate schedule configuration.
"""

import gc
import json
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from typing import (
    Tuple, List, Optional, Dict, Any, Callable
)

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    # Training and Performance
    TrainingCurvesVisualization,
    LearningRateScheduleVisualization,
    ModelComparisonBarChart,
    PerformanceRadarChart,
    ConvergenceAnalysis,
    OverfittingAnalysis,
    PerformanceDashboard,
    # Classification
    ConfusionMatrixVisualization,
    ROCPRCurves,
    ClassificationReportVisualization,
    PerClassAnalysis,
    ErrorAnalysisDashboard,
    ClassificationResults,
    # Data and Neural Network
    DataDistributionAnalysis,
    ClassBalanceVisualization,
    NetworkArchitectureVisualization,
    ActivationVisualization,
    WeightVisualization,
    FeatureMapVisualization,
    GradientVisualization,
    GradientTopologyVisualization,
    GradientTopologyData
)
from dl_techniques.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
    DataInput,
)

from .. import (
    optimizer_builder,
    learning_rate_schedule_builder as schedule_builder
)

# =============================================================================
# 1. CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Centralized configuration for training vision models.

    This dataclass encapsulates all training parameters, making experiments
    reproducible and easy to configure. It integrates with the dl_techniques
    optimization module for advanced schedule and optimizer configuration.

    Attributes:
        input_shape: Input image shape (H, W, C).
        num_classes: Number of output classes.
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.

        # Learning Rate Schedule Configuration
        lr_schedule_type: Type of LR schedule ('cosine_decay', 'exponential_decay',
                         'cosine_decay_restarts', 'constant').
        learning_rate: Initial learning rate.
        decay_steps: Steps over which to apply decay (auto-calculated if None).
        decay_rate: Decay rate for exponential schedule.
        alpha: Minimum learning rate as fraction of initial (for cosine schedules).
        t_mul: Period multiplier for cosine restarts.
        m_mul: LR multiplier for cosine restarts.
        warmup_steps: Number of warmup steps (0 disables warmup).
        warmup_start_lr: Starting learning rate for warmup phase.

        # Optimizer Configuration
        optimizer_type: Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop', 'adadelta').
        weight_decay: L2 regularization coefficient (for AdamW).
        beta_1: First moment decay rate (Adam/AdamW).
        beta_2: Second moment decay rate (Adam/AdamW).
        epsilon: Numerical stability constant.
        amsgrad: Use AMSGrad variant (Adam/AdamW).
        rho: Decay factor (RMSprop/Adadelta).
        momentum: Momentum factor (RMSprop/SGD).
        centered: Use centered RMSprop.
        nesterov: Use Nesterov momentum (SGD).

        # Gradient Clipping Configuration
        gradient_clipping_value: Clip gradients by absolute value.
        gradient_clipping_norm_local: Clip gradients by local L2 norm.
        gradient_clipping_norm_global: Clip gradients by global L2 norm.

        # Loss Configuration
        from_logits: Whether model outputs logits (True) or probabilities (False).

        # Training Control
        steps_per_epoch: Override automatic calculation of steps per epoch.
        validation_steps: Override automatic calculation of validation steps.
        early_stopping_patience: Patience for early stopping callback.
        monitor_metric: Metric to monitor for callbacks ('val_accuracy', 'val_loss').
        monitor_mode: Mode for monitoring ('max' for accuracy, 'min' for loss).

        # Output & Logging
        output_dir: Base directory for experiment outputs.
        experiment_name: Name of the experiment. Auto-generated if None.

        # Visualization & Analysis
        enable_visualization: Enable automatic visualization during training.
        enable_analysis: Enable model analysis after training.
        visualization_frequency: Create visualizations every N epochs.
        enable_convergence_analysis: Enable convergence analysis dashboard.
        enable_overfitting_analysis: Enable overfitting analysis dashboard.
        enable_gradient_tracking: Track gradients for gradient flow visualization.
        enable_classification_viz: Enable classification-specific visualizations.
        create_final_dashboard: Create comprehensive dashboard at training end.

        # Model-Specific Arguments
        model_args: Additional model-specific arguments.
    """
    # --- Data & Model Configuration ---
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 1000

    # --- Training Hyperparameters ---
    epochs: int = 100
    batch_size: int = 64

    # --- Learning Rate Schedule Configuration ---
    lr_schedule_type: str = 'cosine_decay'
    learning_rate: float = 1e-3
    decay_steps: Optional[int] = None
    decay_rate: float = 0.9
    alpha: float = 0.0001
    t_mul: float = 2.0
    m_mul: float = 0.9
    warmup_steps: int = 1000
    warmup_start_lr: float = 1e-8

    # --- Optimizer Configuration ---
    optimizer_type: str = 'adamw'
    weight_decay: float = 1e-4
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    amsgrad: bool = False
    rho: float = 0.9
    momentum: float = 0.9
    centered: bool = False
    nesterov: bool = True

    # --- Gradient Clipping Configuration ---
    gradient_clipping_value: Optional[float] = None
    gradient_clipping_norm_local: Optional[float] = None
    gradient_clipping_norm_global: Optional[float] = 1.0

    # --- Loss Configuration ---
    from_logits: bool = False

    # --- Training Control ---
    steps_per_epoch: Optional[int] = None
    validation_steps: Optional[int] = None
    early_stopping_patience: int = 25
    monitor_metric: str = 'val_accuracy'
    monitor_mode: str = 'max'

    # --- Output & Logging ---
    output_dir: str = 'results'
    experiment_name: Optional[str] = None

    # --- Visualization & Analysis ---
    enable_visualization: bool = True
    enable_analysis: bool = True
    visualization_frequency: int = 10
    enable_convergence_analysis: bool = True
    enable_overfitting_analysis: bool = True
    enable_gradient_tracking: bool = False
    enable_gradient_topology_viz: bool = False
    enable_classification_viz: bool = True
    create_final_dashboard: bool = True

    # --- Model-Specific Arguments ---
    model_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate experiment name if not provided."""
        if self.experiment_name is None:
            model_name = self.model_args.get('variant', 'model')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{model_name}_{timestamp}"

    def to_schedule_config(self, total_steps: int) -> Dict[str, Any]:
        """
        Convert training config to schedule builder config.

        Args:
            total_steps: Total training steps for decay calculation.

        Returns:
            Configuration dictionary for schedule_builder.
        """
        # Calculate decay_steps if not specified
        decay_steps = self.decay_steps
        if decay_steps is None:
            decay_steps = total_steps

        config = {
            "type": self.lr_schedule_type,
            "learning_rate": self.learning_rate,
            "decay_steps": decay_steps,
            "warmup_steps": self.warmup_steps,
            "warmup_start_lr": self.warmup_start_lr,
        }

        # Add schedule-specific parameters
        if self.lr_schedule_type == 'exponential_decay':
            config["decay_rate"] = self.decay_rate
        elif self.lr_schedule_type in ['cosine_decay', 'cosine_decay_restarts']:
            config["alpha"] = self.alpha
            if self.lr_schedule_type == 'cosine_decay_restarts':
                config["t_mul"] = self.t_mul
                config["m_mul"] = self.m_mul

        return config

    def to_optimizer_config(self) -> Dict[str, Any]:
        """
        Convert training config to optimizer builder config.

        Returns:
            Configuration dictionary for optimizer_builder.
        """
        config = {
            "type": self.optimizer_type,
            "gradient_clipping_by_value": self.gradient_clipping_value,
            "gradient_clipping_by_norm_local": self.gradient_clipping_norm_local,
            "gradient_clipping_by_norm": self.gradient_clipping_norm_global,
        }

        # Add optimizer-specific parameters
        opt_type = self.optimizer_type.lower()

        if opt_type in ['adam', 'adamw']:
            config.update({
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            })
            if opt_type == 'adamw':
                config["weight_decay"] = self.weight_decay

        elif opt_type == 'rmsprop':
            config.update({
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
            })

        elif opt_type == 'adadelta':
            config.update({
                "rho": self.rho,
                "epsilon": self.epsilon,
            })

        elif opt_type == 'sgd':
            config.update({
                "momentum": self.momentum,
                "nesterov": self.nesterov,
            })

        return config

    def save(self, file_path: Path) -> None:
        """
        Save configuration to JSON file.

        Args:
            file_path: Path where to save the configuration.
        """
        logger.info(f"Saving configuration to {file_path}")
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, file_path: Path) -> 'TrainingConfig':
        """
        Load configuration from JSON file.

        Args:
            file_path: Path to the configuration file.

        Returns:
            Loaded TrainingConfig instance.
        """
        logger.info(f"Loading configuration from {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# 2. ABSTRACT DATASET BUILDER
# =============================================================================

class DatasetBuilder(ABC):
    """
    Abstract base class for creating training and validation datasets.

    This class defines the interface for data handling. By creating concrete
    subclasses, you can easily switch between different data sources without
    changing the main training loop.

    Attributes:
        config: The TrainingConfig object for this experiment.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the dataset builder.

        Args:
            config: Training configuration object.
        """
        self.config = config

    @abstractmethod
    def build(self) -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        Optional[int],
        Optional[int]
    ]:
        """
        Construct training and validation datasets.

        Returns:
            Tuple containing:
                - train_dataset: Configured training dataset.
                - val_dataset: Configured validation dataset.
                - steps_per_epoch: Calculated steps for one epoch.
                - validation_steps: Calculated steps for validation.
        """
        pass

    def get_test_data(self) -> Optional[DataInput]:
        """
        Get test data for model analysis.

        Returns:
            DataInput object containing test data, or None if not available.
        """
        return None

    def get_class_names(self) -> Optional[List[str]]:
        """
        Get class names for the dataset.

        Returns:
            List of class names, or None if not available.
        """
        return None


# =============================================================================
# 3. ENHANCED VISUALIZATION CALLBACK
# =============================================================================

class EnhancedVisualizationCallback(keras.callbacks.Callback):
    """
    Enhanced callback for creating comprehensive visualizations during training.

    This callback integrates with the VisualizationManager to create
    real-time visualizations of training progress, learning rate schedules,
    gradient flow (if enabled), and other metrics.

    Attributes:
        viz_manager: VisualizationManager instance.
        config: Training configuration.
        frequency: Create visualizations every N epochs.
        lr_schedule: Learning rate schedule for visualization.
        track_gradients: Whether to track gradient norms.
        sample_batch: A sample data batch for gradient computation.
    """

    def __init__(
            self,
            viz_manager: VisualizationManager,
            config: TrainingConfig,
            frequency: int = 10,
            lr_schedule: Optional[Any] = None,
            track_gradients: bool = False,
            sample_batch: Optional[Tuple[tf.Tensor, tf.Tensor]] = None
    ):
        """
        Initialize the enhanced visualization callback.

        Args:
            viz_manager: VisualizationManager instance.
            config: Training configuration.
            frequency: Visualization frequency in epochs.
            lr_schedule: Learning rate schedule object for plotting.
            track_gradients: Whether to track and visualize gradient norms.
            sample_batch: A sample batch (x, y) for gradient computation.
        """
        super().__init__()
        self.viz_manager = viz_manager
        self.config = config
        self.frequency = frequency
        self.lr_schedule = lr_schedule
        self.track_gradients = track_gradients
        # CHANGED: Store the sample batch.
        self.sample_batch = sample_batch
        self.history_data = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {},
            'learning_rates': [],
            'grad_norms': []
        }
        self.last_raw_gradients: Optional[List[tf.Tensor]] = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """
        Called at the end of each epoch to update and visualize metrics.

        Args:
            epoch: Current epoch number.
            logs: Dictionary of metrics from this epoch.
        """
        if logs is None:
            return

        # Collect history data
        self.history_data['epochs'].append(epoch)
        self.history_data['train_loss'].append(logs.get('loss', 0.0))
        self.history_data['val_loss'].append(logs.get('val_loss', 0.0))

        # Collect metrics
        for key, value in logs.items():
            if key.startswith('val_') and key != 'val_loss':
                metric_name = key.replace('val_', '')
                if metric_name not in self.history_data['val_metrics']:
                    self.history_data['val_metrics'][metric_name] = []
                    self.history_data['train_metrics'][metric_name] = []
                self.history_data['val_metrics'][metric_name].append(value)
                self.history_data['train_metrics'][metric_name].append(
                    logs.get(metric_name, 0.0)
                )

        # Collect learning rate
        try:
            lr = float(keras.ops.convert_to_numpy(
                self.model.optimizer.learning_rate
            ))
            self.history_data['learning_rates'].append(lr)
        except Exception:
            pass

        # Collect gradient norms if enabled
        if self.track_gradients:
            try:
                grad_norm, raw_gradients = self._compute_gradients()
                if grad_norm is not None:
                    self.history_data['grad_norms'].append(grad_norm)
                # Store the raw gradients for use in _create_visualizations
                self.last_raw_gradients = raw_gradients
            except Exception as e:
                logger.warning(f"Failed to compute gradients: {e}")

        # Create visualizations at specified frequency
        if (epoch + 1) % self.frequency == 0:
            self._create_visualizations()

    def _compute_gradients(self) -> Tuple[Optional[float], Optional[List[tf.Tensor]]]:
        """
        Compute the global gradient norm and return the raw gradients.
        """
        if not self.track_gradients or self.sample_batch is None:
            return None, None

        try:
            x, y_true = self.sample_batch

            with tf.GradientTape() as tape:
                y_pred = self.model(x, training=False)
                loss = self.model.compute_loss(x=x, y=y_true, y_pred=y_pred)

            # Compute gradients of the total loss w.r.t. the variables
            gradients = tape.gradient(loss, self.model.trainable_variables)

            # Filter out potential None gradients for unconnected layers
            valid_gradients = [g for g in gradients if g is not None]
            if not valid_gradients:
                logger.warning("All computed gradients were None. Check model connectivity and loss function.")
                return None, None

            global_norm = tf.linalg.global_norm(valid_gradients)

            return float(global_norm.numpy()), gradients

        except Exception as e:
            logger.warning(f"Could not compute gradients: {e}", exc_info=True)
            return None, None

    def _create_visualizations(self):
        """Create and save current visualizations."""
        try:
            # Create training history visualization
            history = TrainingHistory(
                epochs=self.history_data['epochs'],
                train_loss=self.history_data['train_loss'],
                val_loss=self.history_data['val_loss'],
                train_metrics=self.history_data['train_metrics'],
                val_metrics=self.history_data['val_metrics']
            )

            self.viz_manager.visualize(
                data=history,
                plugin_name='training_curves',
                smooth_factor=0.1,
                show=False
            )

            # Create LR schedule visualization if available
            if self.history_data['learning_rates']:
                self.viz_manager.visualize(
                    data={'learning_rate': self.history_data['learning_rates']},
                    plugin_name='lr_schedule',
                    show=False
                )

            # Create convergence analysis if enabled
            if self.config.enable_convergence_analysis:
                try:
                    # Add grad_norms to history if available
                    history_dict = {
                        'epochs': self.history_data['epochs'],
                        'train_loss': self.history_data['train_loss'],
                        'val_loss': self.history_data['val_loss'],
                        'train_metrics': self.history_data['train_metrics'],
                        'val_metrics': self.history_data['val_metrics']
                    }
                    if self.history_data['grad_norms']:
                        history_dict['grad_norms'] = self.history_data['grad_norms']

                    self.viz_manager.visualize(
                        data=history_dict,
                        plugin_name='convergence_analysis',
                        show=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to create convergence analysis: {e}")

            if self.config.enable_gradient_topology_viz and self.last_raw_gradients:
                logger.info("Creating gradient topology visualization...")
                try:
                    topo_data = GradientTopologyData(
                        model=self.model,
                        gradients=self.last_raw_gradients,
                        model_name=self.config.experiment_name
                    )
                    self.viz_manager.visualize(
                        data=topo_data,
                        plugin_name='gradient_topology',
                        show=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to create gradient topology visualization: {e}")

            # Create overfitting analysis if enabled
            if self.config.enable_overfitting_analysis:
                try:
                    self.viz_manager.visualize(
                        data=history,
                        plugin_name='overfitting_analysis',
                        patience=self.config.early_stopping_patience,
                        show=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to create overfitting analysis: {e}")

            logger.info("Training visualizations updated")

        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")

    def on_train_end(self, logs: Optional[Dict[str, float]] = None):
        """Create final visualizations when training ends."""
        self._create_visualizations()

    def get_history_data(self) -> Dict[str, Any]:
        """
        Get the collected history data.

        Returns:
            Dictionary containing all collected metrics and history.
        """
        return self.history_data


# =============================================================================
# 4. MODEL BUILDER TYPE
# =============================================================================

ModelBuilder = Callable[[TrainingConfig], keras.Model]


# =============================================================================
# 5. CORE TRAINING PIPELINE
# =============================================================================

class TrainingPipeline:
    """
    Orchestrates end-to-end model training with visualization and analysis.

    This class takes a configuration, model builder, and dataset builder
    to run a complete training experiment with automatic visualization,
    monitoring, and post-training analysis. It integrates with the
    dl_techniques optimization module for advanced schedule and optimizer
    configuration.

    Attributes:
        config: Training configuration object.
        experiment_dir: Directory for saving experiment outputs.
        viz_manager: VisualizationManager for creating plots.
        training_history: Storage for training metrics.
        viz_callback: Visualization callback instance.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the training pipeline.

        Args:
            config: TrainingConfig object for the experiment.
        """
        self.config = config
        self.experiment_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.viz_manager: Optional[VisualizationManager] = None
        self.training_history: Optional[keras.callbacks.History] = None
        self.viz_callback: Optional[EnhancedVisualizationCallback] = None

    def _setup_environment(self):
        """
        Set up the training environment.

        Creates output directories, saves configuration, and configures
        GPU settings for optimal performance.
        """
        logger.info(f"Setting up experiment: {self.config.experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(self.experiment_dir / 'config.json')

        # Configure GPU settings
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                logger.error(f"GPU setup error: {e}")
        else:
            logger.info("No GPUs found, using CPU")

    def _setup_visualization(self):
        """
        Initialize the visualization manager with all available templates.

        Sets up the visualization system with appropriate configuration
        and registers all relevant visualization templates.
        """
        if not self.config.enable_visualization:
            return

        logger.info("Setting up comprehensive visualization manager")

        # Create plot configuration
        plot_config = PlotConfig(
            style=PlotStyle.PUBLICATION,
            color_scheme=ColorScheme(
                primary="#2E86AB",
                secondary="#A23B72"
            ),
            title_fontsize=14,
            save_format='png',
            dpi=300
        )

        # Initialize visualization manager
        self.viz_manager = VisualizationManager(
            experiment_name=self.config.experiment_name,
            output_dir=str(self.experiment_dir / 'visualizations'),
            config=plot_config
        )

        # Register training and performance visualization templates
        self.viz_manager.register_template(
            'training_curves',
            TrainingCurvesVisualization
        )
        self.viz_manager.register_template(
            'lr_schedule',
            LearningRateScheduleVisualization
        )
        self.viz_manager.register_template(
            'model_comparison_bars',
            ModelComparisonBarChart
        )
        self.viz_manager.register_template(
            'performance_radar',
            PerformanceRadarChart
        )
        self.viz_manager.register_template(
            'convergence_analysis',
            ConvergenceAnalysis
        )
        self.viz_manager.register_template(
            'overfitting_analysis',
            OverfittingAnalysis
        )
        self.viz_manager.register_template(
            'performance_dashboard',
            PerformanceDashboard
        )

        # Register classification visualization templates
        if self.config.enable_classification_viz:
            self.viz_manager.register_template(
                'confusion_matrix',
                ConfusionMatrixVisualization
            )
            self.viz_manager.register_template(
                'roc_pr_curves',
                ROCPRCurves
            )
            self.viz_manager.register_template(
                'classification_report',
                ClassificationReportVisualization
            )
            self.viz_manager.register_template(
                'per_class_analysis',
                PerClassAnalysis
            )
            self.viz_manager.register_template(
                'error_analysis',
                ErrorAnalysisDashboard
            )

        # Register data and neural network visualization templates
        self.viz_manager.register_template(
            'data_distribution',
            DataDistributionAnalysis
        )
        self.viz_manager.register_template(
            'class_balance',
            ClassBalanceVisualization
        )
        self.viz_manager.register_template(
            'network_architecture',
            NetworkArchitectureVisualization
        )
        self.viz_manager.register_template(
            'activations',
            ActivationVisualization
        )
        self.viz_manager.register_template(
            'weights',
            WeightVisualization
        )
        self.viz_manager.register_template(
            'feature_maps',
            FeatureMapVisualization
        )
        self.viz_manager.register_template(
            'gradients',
            GradientVisualization
        )
        self.viz_manager.register_template(
            'gradient_topology',
            GradientTopologyVisualization
        )

        logger.info(f"Visualization manager ready with {len(self.viz_manager.templates)} templates")

    def _compile_model(
            self,
            model: keras.Model,
            total_steps: int
    ) -> None:
        """
        Compile the model with configured optimizer, loss, and metrics.

        Uses the dl_techniques optimization module for advanced schedule
        and optimizer configuration with warmup support.

        Args:
            model: Keras model to compile.
            total_steps: Total number of training steps for LR schedule.
        """
        logger.info("Compiling model using dl_techniques optimization module")
        logger.info(f"Loss configured with from_logits={self.config.from_logits}")

        # Handle constant learning rate (no schedule)
        if self.config.lr_schedule_type == 'constant':
            logger.info("Using constant learning rate (no schedule)")
            lr_schedule = self.config.learning_rate
        else:
            # Build learning rate schedule using schedule_builder
            schedule_config = self.config.to_schedule_config(total_steps)
            logger.info(f"Building LR schedule with config: {schedule_config}")
            lr_schedule = schedule_builder(schedule_config)

        # Build optimizer using optimizer_builder
        optimizer_config = self.config.to_optimizer_config()
        logger.info(f"Building optimizer [{self.config.optimizer_type}] with config: {optimizer_config}")
        optimizer = optimizer_builder(optimizer_config, lr_schedule)

        # Define loss (using configured from_logits)
        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.config.from_logits
        )

        # Define metrics (assuming classification)
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(
                k=5,
                name='top5_accuracy'
            )
        ]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        logger.info("Model compilation complete")

    # CHANGED: Added `train_ds` parameter to fetch a sample batch.
    def _create_callbacks(
            self,
            lr_schedule: Optional[Any] = None,
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None,
            train_ds: Optional[tf.data.Dataset] = None
    ) -> List[keras.callbacks.Callback]:
        """
        Create standard and custom callbacks for training.

        Args:
            lr_schedule: Learning rate schedule for visualization.
            custom_callbacks: Additional user-provided callbacks.
            train_ds: Training dataset to fetch a sample batch from for gradient tracking.
        """
        logger.info("Creating training callbacks")

        callbacks: List[keras.callbacks.Callback] = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.experiment_dir / 'best_model.keras'),
                monitor=self.config.monitor_metric,
                save_best_only=True,
                mode=self.config.monitor_mode,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor=self.config.monitor_metric,
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                mode=self.config.monitor_mode,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                filename=str(self.experiment_dir / 'training_log.csv')
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(self.experiment_dir / 'tensorboard_logs'),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]

        # Add enhanced visualization callback
        if self.config.enable_visualization and self.viz_manager is not None:
            # CHANGED: Fetch a sample batch for gradient norm calculation if enabled.
            sample_batch = None
            if self.config.enable_gradient_tracking and train_ds is not None:
                try:
                    # Take one batch from the training dataset
                    sample_batch = next(iter(train_ds))
                    logger.info("Fetched a sample batch for gradient norm tracking.")
                except StopIteration:
                    logger.warning("Could not fetch a sample batch from train_ds for gradient tracking.")

            self.viz_callback = EnhancedVisualizationCallback(
                viz_manager=self.viz_manager,
                config=self.config,
                frequency=self.config.visualization_frequency,
                lr_schedule=lr_schedule,
                track_gradients=self.config.enable_gradient_tracking,
                # CHANGED: Pass the fetched batch to the callback.
                sample_batch=sample_batch
            )
            callbacks.append(self.viz_callback)

        # Add custom callbacks
        if custom_callbacks:
            callbacks.extend(custom_callbacks)

        logger.info(f"Created {len(callbacks)} callbacks")
        return callbacks

    def _create_classification_visualizations(
            self,
            model: keras.Model,
            test_data: DataInput,
            class_names: Optional[List[str]] = None
    ):
        """
        Create classification-specific visualizations after training.

        Args:
            model: Trained model.
            test_data: Test data for evaluation.
            class_names: List of class names.
        """
        if not self.config.enable_classification_viz or not self.config.enable_visualization:
            return

        logger.info("Creating classification visualizations")

        try:
            # Get predictions
            x_test = test_data.x_data
            y_true = test_data.y_data

            # Make predictions
            y_prob = model.predict(x_test, verbose=0)
            y_pred = np.argmax(y_prob, axis=-1)

            # Create ClassificationResults
            results = ClassificationResults(
                y_true=y_true.flatten() if len(y_true.shape) > 1 else y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                class_names=class_names,
                model_name=self.config.experiment_name
            )

            # Confusion Matrix
            self.viz_manager.visualize(
                data=results,
                plugin_name='confusion_matrix',
                normalize='true',
                show=False
            )

            # ROC and PR Curves
            try:
                self.viz_manager.visualize(
                    data=results,
                    plugin_name='roc_pr_curves',
                    plot_type='both',
                    show=False
                )
            except Exception as e:
                logger.warning(f"Failed to create ROC/PR curves: {e}")

            # Classification Report
            self.viz_manager.visualize(
                data=results,
                plugin_name='classification_report',
                show=False
            )

            # Per-Class Analysis Dashboard
            try:
                self.viz_manager.visualize(
                    data=results,
                    plugin_name='per_class_analysis',
                    show=False
                )
            except Exception as e:
                logger.warning(f"Failed to create per-class analysis: {e}")

            # Error Analysis Dashboard
            try:
                self.viz_manager.visualize(
                    data=results,
                    plugin_name='error_analysis',
                    x_data=x_test,
                    show=False
                )
            except Exception as e:
                logger.warning(f"Failed to create error analysis: {e}")

            logger.info("Classification visualizations created successfully")

        except Exception as e:
            logger.error(f"Failed to create classification visualizations: {e}", exc_info=True)

    def _create_final_dashboard(self):
        """
        Create a comprehensive final dashboard combining multiple visualizations.
        """
        if not self.config.create_final_dashboard or not self.config.enable_visualization:
            return

        logger.info("Creating final comprehensive dashboard")

        try:
            # Prepare dashboard data
            dashboard_data = {}

            # Add training curves if available
            if self.viz_callback is not None:
                history_data = self.viz_callback.get_history_data()
                if history_data['epochs']:
                    history = TrainingHistory(
                        epochs=history_data['epochs'],
                        train_loss=history_data['train_loss'],
                        val_loss=history_data['val_loss'],
                        train_metrics=history_data['train_metrics'],
                        val_metrics=history_data['val_metrics']
                    )
                    dashboard_data['training_curves'] = history

            # Create the dashboard
            if dashboard_data:
                self.viz_manager.create_dashboard(
                    data=dashboard_data,
                    show=False
                )
                logger.info("Final dashboard created successfully")
            else:
                logger.warning("No data available for dashboard creation")

        except Exception as e:
            logger.error(f"Failed to create final dashboard: {e}", exc_info=True)

    def _run_model_analysis(
            self,
            model: keras.Model,
            test_data: Optional[DataInput] = None
    ):
        """
        Run comprehensive model analysis after training.

        Args:
            model: Trained model to analyze.
            test_data: Test data for evaluation.
        """
        if not self.config.enable_analysis:
            return

        logger.info("Starting model analysis")

        try:
            # Configure analysis
            analysis_config = AnalysisConfig(
                analyze_weights=True,
                analyze_calibration=True,
                analyze_information_flow=True,
                analyze_training_dynamics=True,
                compute_weight_pca=True,
                save_plots=True,
                save_format='png',
                dpi=300,
                plot_style='publication'
            )

            # Prepare training history for analyzer
            training_history = None
            if self.training_history is not None:
                training_history = {
                    self.config.experiment_name: self.training_history.history
                }

            # Initialize analyzer
            analyzer = ModelAnalyzer(
                models={self.config.experiment_name: model},
                training_history=training_history,
                config=analysis_config,
                output_dir=str(self.experiment_dir / 'analysis')
            )

            # Run analysis
            results = analyzer.analyze(data=test_data)

            # Log summary statistics
            summary = analyzer.get_summary_statistics()
            logger.info("Model analysis complete")
            logger.info(f"Analysis results saved to {self.experiment_dir / 'analysis'}")

            # Log key metrics
            if 'calibration_summary' in summary:
                cal_metrics = summary['calibration_summary'].get(
                    self.config.experiment_name,
                    {}
                )
                if 'ece' in cal_metrics:
                    logger.info(f"Expected Calibration Error: {cal_metrics['ece']:.4f}")

        except Exception as e:
            logger.error(f"Model analysis failed: {e}", exc_info=True)

    def run(
            self,
            model_builder: ModelBuilder,
            dataset_builder: DatasetBuilder,
            custom_callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> Tuple[keras.Model, keras.callbacks.History]:
        """
        Execute the complete training pipeline.

        This method orchestrates the entire training process including:
        - Environment setup
        - Dataset preparation
        - Model creation and compilation (with dl_techniques optimization)
        - Training with callbacks
        - Comprehensive visualization and analysis

        Args:
            model_builder: Function that creates and returns a Keras model.
            dataset_builder: Instance of DatasetBuilder for data loading.
            custom_callbacks: Optional additional callbacks for training.

        Returns:
            Tuple of (trained_model, training_history).
        """
        # Setup
        self._setup_environment()
        self._setup_visualization()

        # Build datasets
        logger.info("Building datasets")
        train_ds, val_ds, steps_per_epoch, val_steps = dataset_builder.build()

        # Use config overrides if provided
        steps_per_epoch = self.config.steps_per_epoch or steps_per_epoch
        val_steps = self.config.validation_steps or val_steps

        if steps_per_epoch is None:
            raise ValueError(
                "steps_per_epoch must be defined in config or by DatasetBuilder"
            )

        # Build model
        logger.info("Building model")
        model = model_builder(self.config)

        # Log model architecture
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info, expand_nested=True)

        # Visualize model architecture
        if self.config.enable_visualization and self.viz_manager is not None:
            try:
                self.viz_manager.visualize(
                    model,
                    plugin_name='network_architecture',
                    show=False
                )
            except Exception as e:
                logger.warning(f"Could not visualize architecture: {e}")

        # Visualize class balance if available
        try:
            test_data = dataset_builder.get_test_data()
            if test_data is not None and self.viz_manager is not None:
                self.viz_manager.visualize(
                    data=(test_data.x, test_data.y),
                    plugin_name='class_balance',
                    show=False
                )
        except Exception as e:
            logger.warning(f"Could not visualize class balance: {e}")

        # Compile model
        total_steps = steps_per_epoch * self.config.epochs
        self._compile_model(model, total_steps)

        # Get LR schedule for callbacks (if not constant)
        lr_schedule = None
        if self.config.lr_schedule_type != 'constant':
            schedule_config = self.config.to_schedule_config(total_steps)
            lr_schedule = schedule_builder(schedule_config)

        # Create callbacks
        # CHANGED: Pass train_ds to _create_callbacks to allow it to fetch a sample batch.
        callbacks = self._create_callbacks(lr_schedule, custom_callbacks, train_ds)

        # Train model
        logger.info("Starting model training")
        logger.info(
            f"Epochs: {self.config.epochs}, "
            f"Steps per epoch: {steps_per_epoch}"
        )

        self.training_history = model.fit(
            train_ds,
            epochs=self.config.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training finished")

        # Save final model
        final_model_path = self.experiment_dir / 'final_model.keras'
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        # Create classification visualizations
        test_data = dataset_builder.get_test_data()
        if test_data is not None:
            class_names = dataset_builder.get_class_names()
            self._create_classification_visualizations(
                model, test_data, class_names
            )

        # Create final comprehensive dashboard
        self._create_final_dashboard()

        # Run model analysis
        self._run_model_analysis(model, test_data)

        # Clean up memory
        gc.collect()
        keras.backend.clear_session()

        return model, self.training_history


# =============================================================================
# 6. UTILITY FUNCTIONS
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for command-line configuration.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Train vision models with comprehensive visualization and analysis'
    )

    # Data arguments
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs=3,
        default=[224, 224, 3],
        help='Input image shape (H W C)'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=1000,
        help='Number of output classes'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Training batch size'
    )

    # Learning rate schedule arguments
    parser.add_argument(
        '--lr-schedule',
        type=str,
        default='cosine_decay',
        choices=['cosine_decay', 'exponential_decay', 'cosine_decay_restarts', 'constant'],
        help='Learning rate schedule type'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=1000,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.0001,
        help='Minimum learning rate fraction (cosine schedules)'
    )

    # Optimizer arguments
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adadelta'],
        help='Optimizer type'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay for regularization (AdamW)'
    )
    parser.add_argument(
        '--gradient-clip',
        type=float,
        default=1.0,
        help='Gradient clipping by global norm'
    )

    # Loss arguments
    parser.add_argument(
        '--from-logits',
        action='store_true',
        help='Model outputs logits instead of probabilities'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Base output directory'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )

    # Feature flags
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Disable automatic visualization'
    )
    parser.add_argument(
        '--no-analysis',
        action='store_true',
        help='Disable model analysis'
    )
    parser.add_argument(
        '--no-convergence-analysis',
        action='store_true',
        help='Disable convergence analysis'
    )
    parser.add_argument(
        '--no-overfitting-analysis',
        action='store_true',
        help='Disable overfitting analysis'
    )
    parser.add_argument(
        '--enable-gradient-tracking',
        action='store_true',
        help='Enable gradient norm tracking (may slow training)'
    )
    parser.add_argument(
        '--no-classification-viz',
        action='store_true',
        help='Disable classification-specific visualizations'
    )
    parser.add_argument(
        '--no-final-dashboard',
        action='store_true',
        help='Disable final comprehensive dashboard'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Load configuration from JSON file'
    )

    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """
    Create TrainingConfig from command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        TrainingConfig instance.
    """
    # If config file provided, load it and override with CLI args
    if args.config:
        config = TrainingConfig.load(Path(args.config))
        # Override with CLI arguments that were explicitly set
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        return config

    return TrainingConfig(
        input_shape=tuple(args.input_shape),
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_schedule_type=args.lr_schedule,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        alpha=args.alpha,
        optimizer_type=args.optimizer,
        weight_decay=args.weight_decay,
        gradient_clipping_norm_global=args.gradient_clip,
        from_logits=args.from_logits,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        enable_visualization=not args.no_visualization,
        enable_analysis=not args.no_analysis,
        enable_convergence_analysis=not args.no_convergence_analysis,
        enable_overfitting_analysis=not args.no_overfitting_analysis,
        enable_gradient_tracking=args.enable_gradient_tracking,
        enable_classification_viz=not args.no_classification_viz,
        create_final_dashboard=not args.no_final_dashboard
    )
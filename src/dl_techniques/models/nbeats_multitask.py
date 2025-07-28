"""
Multi-Task N-BEATS Model Implementation with Trainable Task Inference - GRAPH MODE FIXED.

This module provides a multi-task variant of the N-BEATS architecture that can
simultaneously learn multiple time series forecasting tasks with task-specific
embeddings and trainable task inference. The model can learn to automatically
infer appropriate task adjustments even when task IDs are not provided during training.

Key improvements:
- Trainable task inference with auxiliary losses
- Convolutional feature extractor for improved task inference
- Consistency regularization for task inference
- Entropy regularization for confident task predictions
- Curriculum learning support (labeled → unlabeled)
- Mixed training with and without task IDs
- FIXED: Graph-compatible conditional operations using keras.ops.cond

References:
    - Oreshkin, B. N., et al. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." ICLR 2020.
    - Multi-task learning principles adapted for time series forecasting
"""

import keras
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any, Tuple

from dl_techniques.utils.logger import logger
from .nbeats import create_nbeats_model


@dataclass
class MultiTaskNBeatsConfig:
    """Configuration class for Multi-Task N-BEATS model with trainable task inference."""

    # Architecture parameters
    backcast_length: int = 168
    use_task_embeddings: bool = True
    task_embedding_dim: int = 32
    use_task_inference: bool = True
    task_inference_hidden_dim: int = 128
    task_inference_dropout: float = 0.1
    task_inference_activation: str = 'gelu'

    # Task inference convolutional feature extractor parameters
    task_inference_use_conv: bool = True
    task_inference_conv_filters: List[int] = None
    task_inference_conv_kernels: List[int] = None
    task_inference_conv_activation: str = 'gelu'

    # Task inference training parameters - FIXED WEIGHTS
    train_task_inference: bool = True
    task_inference_loss_weight: float = 0.5
    consistency_loss_weight: float = 0.1
    entropy_loss_weight: float = 0.05
    consistency_temperature: float = 0.1
    min_entropy_target: float = 0.1
    task_adjustment: float = 0.1

    use_bias: bool = False
    stack_types: List[str] = None
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 512
    use_revin: bool = True

    # Regularization parameters
    dropout_rate: float = 0.1
    kernel_regularizer_l2: float = 1e-5

    # Training parameters
    optimizer: str = 'adamw'
    primary_loss: str = 'mae'
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.stack_types is None:
            self.stack_types = ["trend", "seasonality", "generic"]
        if self.task_inference_conv_filters is None:
            self.task_inference_conv_filters = [16, 32, 64]
        if self.task_inference_conv_kernels is None:
            self.task_inference_conv_kernels = [7, 5, 3]


@keras.saving.register_keras_serializable()
class MultiTaskNBeatsNet(keras.Model):
    """Multi-task N-BEATS model with graph-compatible trainable task inference."""

    def __init__(
            self,
            config: MultiTaskNBeatsConfig,
            num_tasks: int,
            task_to_id: Dict[str, int],
            forecast_length: int,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.config = config
        self.num_tasks = num_tasks
        self.task_to_id = task_to_id
        self.id_to_task = {v: k for k, v in task_to_id.items()}
        self.forecast_length = forecast_length

        # Task embedding layer
        self.task_embedding = None
        if config.use_task_embeddings:
            self.task_embedding = keras.layers.Embedding(
                input_dim=num_tasks,
                output_dim=config.task_embedding_dim,
                name='task_embedding'
            )

        # Task inference network components
        self.task_inference_layers = {}

        # Main N-BEATS model (created in build)
        self.nbeats_model = None

        # Task-specific adjustment layer
        self.task_adjustment_layer = None

        # Track auxiliary losses for monitoring
        self.aux_loss_tracker = {
            'entropy_loss': keras.metrics.Mean(name='aux_entropy_loss'),
            'consistency_loss': keras.metrics.Mean(name='aux_consistency_loss'),
            'balance_loss': keras.metrics.Mean(name='aux_balance_loss')
        }

        logger.info(f"Initialized Graph-Compatible MultiTaskNBeatsNet:")
        logger.info(f"  - Number of tasks: {num_tasks}")
        logger.info(f"  - Task inference enabled: {config.use_task_inference}")
        logger.info(f"  - Task inference loss weights: {config.task_inference_loss_weight}")

    def build(self, input_shape: Union[Tuple, List[Tuple]]) -> None:
        """Build the multi-task N-BEATS model."""
        logger.info("Building Graph-Compatible Multi-Task N-BEATS model...")

        if isinstance(input_shape, list):
            time_series_shape = input_shape[0]
        else:
            time_series_shape = input_shape

        # Create the base N-BEATS model
        try:
            self.nbeats_model = create_nbeats_model(
                backcast_length=self.config.backcast_length,
                forecast_length=self.forecast_length,
                stack_types=self.config.stack_types,
                nb_blocks_per_stack=self.config.nb_blocks_per_stack,
                hidden_layer_units=self.config.hidden_layer_units,
                use_revin=self.config.use_revin,
                dropout_rate=self.config.dropout_rate,
                kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2),
                optimizer=self.config.optimizer,
                loss=self.config.primary_loss,
                learning_rate=self.config.learning_rate,
                gradient_clip_norm=self.config.gradient_clip_norm
            )
            logger.info("✓ Base N-BEATS model created")
        except Exception as e:
            logger.error(f"Failed to create base N-BEATS model: {e}")
            raise

        # Add task-specific adjustment layer
        if self.config.use_task_embeddings:
            self.task_adjustment_layer = keras.layers.Dense(
                units=self.forecast_length,
                activation='linear',
                use_bias=self.config.use_bias,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros' if self.config.use_bias else None,
                name='task_adjustment'
            )
            logger.info("✓ Task adjustment layer created")

        # Build task inference network
        if self.config.use_task_inference and self.config.use_task_embeddings:
            self._build_improved_task_inference_network(time_series_shape)
            logger.info("✓ Improved task inference network created")

        super().build(input_shape)
        logger.info("✓ Graph-Compatible Multi-Task N-BEATS model built successfully")

    def _build_improved_task_inference_network(self, time_series_shape: Tuple) -> None:
        """Build improved task inference network with better architecture."""

        # Simplified and more robust feature extraction
        # 1. Global pooling for basic statistics
        self.task_inference_layers['global_avg_pool'] = keras.layers.GlobalAveragePooling1D(
            name='task_inference_global_avg'
        )
        self.task_inference_layers['global_max_pool'] = keras.layers.GlobalMaxPooling1D(
            name='task_inference_global_max'
        )

        # 2. Optional conv layers for pattern extraction
        if self.config.task_inference_use_conv:
            self.task_inference_layers['conv1'] = keras.layers.Conv1D(
                filters=32,
                kernel_size=7,
                padding='same',
                activation=self.config.task_inference_conv_activation,
                kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2),
                name='task_inference_conv1'
            )
            self.task_inference_layers['conv2'] = keras.layers.Conv1D(
                filters=16,
                kernel_size=3,
                padding='same',
                activation=self.config.task_inference_conv_activation,
                kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2),
                name='task_inference_conv2'
            )

        # 3. Dense layers for task classification
        self.task_inference_layers['dense1'] = keras.layers.Dense(
            self.config.task_inference_hidden_dim,
            activation=self.config.task_inference_activation,
            kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2),
            name='task_inference_dense1'
        )

        self.task_inference_layers['dropout1'] = keras.layers.Dropout(
            self.config.task_inference_dropout,
            name='task_inference_dropout1'
        )

        self.task_inference_layers['dense2'] = keras.layers.Dense(
            self.config.task_inference_hidden_dim // 2,
            activation=self.config.task_inference_activation,
            kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2),
            name='task_inference_dense2'
        )

        self.task_inference_layers['dropout2'] = keras.layers.Dropout(
            self.config.task_inference_dropout,
            name='task_inference_dropout2'
        )

        # 4. Task probability prediction
        self.task_inference_layers['task_probs'] = keras.layers.Dense(
            self.num_tasks,
            activation='softmax',
            kernel_initializer='glorot_uniform',
            name='task_inference_probs'
        )

    def _infer_task_probabilities(
            self,
            time_series_data: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Improved task probability inference."""

        # Extract basic statistical features
        global_avg = self.task_inference_layers['global_avg_pool'](time_series_data)
        global_max = self.task_inference_layers['global_max_pool'](time_series_data)

        # Compute additional statistical features
        time_std = keras.ops.std(time_series_data, axis=1)
        time_min = keras.ops.min(time_series_data, axis=1)

        # Start with statistical features
        features = keras.ops.concatenate([global_avg, global_max, time_std, time_min], axis=-1)

        # Add convolutional features if enabled
        if self.config.task_inference_use_conv and 'conv1' in self.task_inference_layers:
            conv_features = self.task_inference_layers['conv1'](time_series_data, training=training)
            conv_features = self.task_inference_layers['conv2'](conv_features, training=training)
            conv_features = self.task_inference_layers['global_avg_pool'](conv_features)
            features = keras.ops.concatenate([features, conv_features], axis=-1)

        # Process through dense network
        x = self.task_inference_layers['dense1'](features, training=training)
        x = self.task_inference_layers['dropout1'](x, training=training)
        x = self.task_inference_layers['dense2'](x, training=training)
        x = self.task_inference_layers['dropout2'](x, training=training)

        task_probs = self.task_inference_layers['task_probs'](x, training=training)
        return task_probs

    def _compute_weighted_task_adjustment(
            self,
            task_probs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Compute weighted task adjustment based on task probabilities."""
        # Get all task embeddings
        all_task_ids = keras.ops.arange(self.num_tasks)
        all_task_embeddings = self.task_embedding(all_task_ids, training=training)

        # Compute task adjustments for all tasks
        all_task_adjustments = self.task_adjustment_layer(
            all_task_embeddings, training=training
        )

        # Weight task adjustments by probabilities
        weighted_adjustment = keras.ops.einsum(
            'bn,nf->bf',
            task_probs,
            all_task_adjustments
        )

        # Add feature dimension
        weighted_adjustment = keras.ops.expand_dims(weighted_adjustment, axis=-1)
        return weighted_adjustment

    def _compute_task_inference_losses(
            self,
            time_series_data: keras.KerasTensor,
            task_probs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Compute auxiliary losses for task inference training - GRAPH MODE FIXED."""
        losses = {}

        if not training or not self.config.train_task_inference:
            return losses

        # 1. FIXED Entropy regularization loss
        # Encourage confident predictions but not too extreme
        entropy = -keras.ops.sum(task_probs * keras.ops.log(task_probs + 1e-8), axis=-1)

        # We want entropy to be low (confident predictions) but not zero
        # Penalty when entropy is too high (uncertain predictions)
        entropy_loss = keras.ops.mean(keras.ops.maximum(0.0, entropy - self.config.min_entropy_target))
        losses['entropy_loss'] = entropy_loss

        # 2. FIXED Consistency regularization loss - GRAPH COMPATIBLE
        batch_size = keras.ops.shape(time_series_data)[0]

        def compute_consistency_loss():
            """Compute consistency loss when batch_size > 1."""
            # Compute pairwise similarities between time series
            flat_series = keras.ops.reshape(time_series_data, (batch_size, -1))

            # L2 normalize for cosine similarity
            flat_series_norm = flat_series / (keras.ops.norm(flat_series, axis=1, keepdims=True) + 1e-8)
            similarity_matrix = keras.ops.matmul(flat_series_norm, keras.ops.transpose(flat_series_norm))

            # Convert similarities to probabilities
            similarity_probs = keras.ops.softmax(similarity_matrix / self.config.consistency_temperature)

            # Compute task probability similarities
            task_similarity_matrix = keras.ops.matmul(task_probs, keras.ops.transpose(task_probs))
            task_similarity_probs = keras.ops.softmax(task_similarity_matrix / self.config.consistency_temperature)

            # KL divergence between time series similarity and task probability similarity
            kl_div = keras.ops.sum(
                similarity_probs * (
                            keras.ops.log(similarity_probs + 1e-8) - keras.ops.log(task_similarity_probs + 1e-8)),
                axis=-1
            )
            return keras.ops.mean(kl_div)

        def no_consistency_loss():
            """Return zero loss when batch_size <= 1."""
            return keras.ops.convert_to_tensor(0.0, dtype=task_probs.dtype)

        # FIXED: Use keras.ops.cond instead of Python if for graph compatibility
        consistency_loss = keras.ops.cond(
            batch_size > 1,
            compute_consistency_loss,
            no_consistency_loss
        )
        losses['consistency_loss'] = consistency_loss

        # 3. FIXED Balance regularization loss
        # Encourage balanced task predictions across the batch
        mean_task_probs = keras.ops.mean(task_probs, axis=0)
        uniform_distribution = keras.ops.ones_like(mean_task_probs) / keras.ops.cast(self.num_tasks,
                                                                                     dtype=mean_task_probs.dtype)

        # Use KL divergence instead of MSE for better balance
        balance_loss = keras.ops.sum(
            mean_task_probs * keras.ops.log(mean_task_probs / (uniform_distribution + 1e-8) + 1e-8)
        )
        losses['balance_loss'] = balance_loss

        return losses

    def call(
            self,
            inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]],
            training: Optional[bool] = None,
            return_aux_losses: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, Dict[str, keras.KerasTensor]]]:
        """Forward pass with improved task-aware processing."""
        aux_losses = {}

        # Handle input formats
        if isinstance(inputs, tuple) and len(inputs) == 2:
            time_series_data, task_ids = inputs
            has_task_ids = task_ids is not None
        elif isinstance(inputs, (tuple, list)) and len(inputs) != 2:
            raise ValueError(f"Expected 2 inputs for multi-task format, got {len(inputs)}")
        else:
            time_series_data = inputs
            task_ids = None
            has_task_ids = False

        # Get base prediction
        base_prediction = self.nbeats_model(time_series_data, training=training)

        # Apply task adjustments if using task embeddings
        if self.config.use_task_embeddings and self.task_adjustment_layer is not None:
            if has_task_ids:
                # Use provided task IDs
                task_emb = self.task_embedding(task_ids, training=training)
                task_adjustment = self.task_adjustment_layer(task_emb, training=training)
                task_adjustment = keras.ops.expand_dims(task_adjustment, axis=-1)

                # Also compute task inference for auxiliary losses
                if training and self.config.use_task_inference and self.config.train_task_inference:
                    task_probs = self._infer_task_probabilities(time_series_data, training=training)
                    aux_losses = self._compute_task_inference_losses(time_series_data, task_probs, training=training)

            elif self.config.use_task_inference:
                # Use task inference
                task_probs = self._infer_task_probabilities(time_series_data, training=training)
                task_adjustment = self._compute_weighted_task_adjustment(task_probs, training=training)

                # Compute auxiliary losses during training
                if training and self.config.train_task_inference:
                    aux_losses = self._compute_task_inference_losses(time_series_data, task_probs, training=training)
            else:
                task_adjustment = 0.0

            final_prediction = base_prediction + (self.config.task_adjustment * task_adjustment)
        else:
            final_prediction = base_prediction

        return (final_prediction, aux_losses) if return_aux_losses else final_prediction

    def train_step(self, data):
        """FIXED custom training step with proper auxiliary loss handling."""
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("Expected data to be a tuple of (x, y)")

        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass with auxiliary losses
            predictions, aux_losses = self(x, training=True, return_aux_losses=True)

            # Compute primary loss
            primary_loss = self.compute_loss(y=y, y_pred=predictions)

            # Initialize total loss with primary loss
            total_loss = primary_loss

            # Add auxiliary losses with proper weights
            if self.config.train_task_inference and aux_losses:
                for loss_name, loss_value in aux_losses.items():
                    # Ensure loss value is a scalar
                    if isinstance(loss_value, tf.Tensor):
                        loss_value = tf.reduce_mean(loss_value)

                    # Apply appropriate weight
                    if loss_name == 'entropy_loss':
                        weight = self.config.entropy_loss_weight
                    elif loss_name == 'consistency_loss':
                        weight = self.config.consistency_loss_weight
                    else:  # balance_loss and others
                        weight = self.config.task_inference_loss_weight

                    weighted_loss = weight * loss_value
                    total_loss += weighted_loss

                    # Update auxiliary loss trackers
                    if loss_name in self.aux_loss_tracker:
                        self.aux_loss_tracker[loss_name].update_state(loss_value)

        # Compute gradients and apply
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Apply gradient clipping if specified
        if self.config.gradient_clip_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.gradient_clip_norm)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, predictions)

        # Prepare results dictionary
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = total_loss
        results['primary_loss'] = primary_loss

        # Add auxiliary loss results
        if aux_losses:
            for loss_name in self.aux_loss_tracker:
                results[f'aux_{loss_name}'] = self.aux_loss_tracker[loss_name].result()

        return results

    def test_step(self, data):
        """Custom test step - unchanged."""
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("Expected data to be a tuple of (x, y)")
        x, y = data

        predictions = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=predictions)
        self.compiled_metrics.update_state(y, predictions)

        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results

    @property
    def metrics(self):
        """Include auxiliary loss trackers in metrics."""
        metrics = super().metrics
        if self.config.train_task_inference:
            metrics.extend(list(self.aux_loss_tracker.values()))
        return metrics

    def reset_metrics(self):
        """Reset all metrics including auxiliary loss trackers."""
        super().reset_metrics()
        for tracker in self.aux_loss_tracker.values():
            tracker.reset_state()

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        base_config = super().get_config()
        base_config.update({
            'config': asdict(self.config),
            'num_tasks': self.num_tasks,
            'task_to_id': self.task_to_id,
            'forecast_length': self.forecast_length,
        })
        return base_config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MultiTaskNBeatsNet':
        """Create model from configuration dictionary."""
        config_data = config.pop('config')
        mt_config = MultiTaskNBeatsConfig(**config_data)
        return cls(config=mt_config, **config)

    # Utility methods remain the same...
    def get_task_id(self, task_name: str) -> int:
        """Get task ID for a given task name."""
        if task_name not in self.task_to_id:
            raise KeyError(f"Task '{task_name}' not found. Available tasks: {list(self.task_to_id.keys())}")
        return self.task_to_id[task_name]

    def get_task_name(self, task_id: int) -> str:
        """Get task name for a given task ID."""
        if task_id not in self.id_to_task:
            raise KeyError(f"Task ID {task_id} not found. Available IDs: {list(self.id_to_task.keys())}")
        return self.id_to_task[task_id]

    def predict_single_task(self, x: np.ndarray, task_name: str, **kwargs) -> np.ndarray:
        """Make predictions for a single task."""
        task_id = self.get_task_id(task_name)
        task_ids = np.full(x.shape[0], task_id)
        return self.predict((x, task_ids), **kwargs)

    def predict_with_task_inference(
            self, x: np.ndarray, return_task_probs: bool = False, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using automatic task inference."""
        if not self.config.use_task_inference:
            raise ValueError("Task inference is not enabled. Set use_task_inference=True in config.")

        predictions = self.predict(x, **kwargs)
        if return_task_probs:
            task_probs = self._infer_task_probabilities(tf.convert_to_tensor(x), training=False)
            return predictions, keras.ops.convert_to_numpy(task_probs)
        else:
            return predictions

    def summary(self, **kwargs) -> None:
        """Print model summary including task information."""
        print("=" * 80)
        print("GRAPH-COMPATIBLE MULTI-TASK N-BEATS MODEL SUMMARY")
        print("=" * 80)
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Forecast length: {self.forecast_length}")
        print(f"Task embeddings: {'Enabled' if self.config.use_task_embeddings else 'Disabled'}")
        print(f"Task inference: {'Enabled' if self.config.use_task_inference else 'Disabled'}")
        print(f"Trainable task inference: {'Enabled' if self.config.train_task_inference else 'Disabled'}")

        if self.config.train_task_inference:
            print(f"\nTask Inference Training Parameters:")
            print(f"  Task inference loss weight: {self.config.task_inference_loss_weight}")
            print(f"  Consistency loss weight: {self.config.consistency_loss_weight}")
            print(f"  Entropy loss weight: {self.config.entropy_loss_weight}")
            print(f"  Min entropy target: {self.config.min_entropy_target}")

        print("\nTask Mapping:")
        for task_name, task_id in self.task_to_id.items():
            print(f"  {task_id:2d}: {task_name}")
        print("=" * 80)

        if self.built:
            super().summary(**kwargs)
        else:
            print("\nModel not built yet. Call model.build() or model() with sample data first.")


def create_multi_task_nbeats(
        config: MultiTaskNBeatsConfig,
        num_tasks: int,
        task_to_id: Dict[str, int],
        forecast_length: int,
        name: Optional[str] = None
) -> MultiTaskNBeatsNet:
    """Create a Graph-Compatible Multi-Task N-BEATS model."""
    logger.info("Creating Graph-Compatible Multi-Task N-BEATS model...")

    model = MultiTaskNBeatsNet(
        config=config,
        num_tasks=num_tasks,
        task_to_id=task_to_id,
        forecast_length=forecast_length,
        name=name
    )

    logger.info("✓ Graph-Compatible Multi-Task N-BEATS model created successfully")
    return model


def create_multi_task_nbeats_from_tasks(
        task_names: List[str],
        forecast_length: int,
        config: Optional[MultiTaskNBeatsConfig] = None,
        name: Optional[str] = None
) -> MultiTaskNBeatsNet:
    """Create a Graph-Compatible Multi-Task N-BEATS model from a list of task names."""
    if config is None:
        config = MultiTaskNBeatsConfig()

    task_to_id = {task: idx for idx, task in enumerate(task_names)}
    logger.info(f"Creating Graph-Compatible Multi-Task N-BEATS for {len(task_names)} tasks: {task_names}")

    return create_multi_task_nbeats(
        config=config,
        num_tasks=len(task_names),
        task_to_id=task_to_id,
        forecast_length=forecast_length,
        name=name
    )
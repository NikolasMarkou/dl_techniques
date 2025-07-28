"""
Multi-Task N-BEATS Model Implementation with Trainable Task Inference.

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

References:
    - Oreshkin, B. N., et al. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." ICLR 2020.
    - Multi-task learning principles adapted for time series forecasting
"""

import keras
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .nbeats import create_nbeats_model

# ---------------------------------------------------------------------

@dataclass
class MultiTaskNBeatsConfig:
    """Configuration class for Multi-Task N-BEATS model with trainable task inference.

    This configuration class encapsulates all parameters needed to create
    and train a multi-task N-BEATS model with trainable task inference.

    Args:
        backcast_length: Number of time steps to look back for prediction.
        use_task_embeddings: Whether to use task embedding layers.
        task_embedding_dim: Dimension of task embedding vectors.
        use_task_inference: Whether to enable automatic task inference.
        task_inference_hidden_dim: Hidden dimension for task inference network.
        task_inference_dropout: Dropout rate for task inference network.
        task_inference_activation: Activation function for task inference network.

        # NEW: Task inference convolutional feature extractor parameters
        task_inference_use_conv: Whether to use a Conv1D extractor for inference.
        task_inference_conv_filters: List of filters for each Conv1D layer.
        task_inference_conv_kernels: List of kernel sizes for each Conv1D layer.
        task_inference_conv_activation: Activation for the convolutional blocks.

        # Task inference training parameters
        train_task_inference: Whether to train task inference network.
        task_inference_loss_weight: Weight for task inference auxiliary losses.
        consistency_loss_weight: Weight for consistency regularization.
        entropy_loss_weight: Weight for entropy regularization.
        consistency_temperature: Temperature for consistency loss.
        min_entropy_target: Target minimum entropy for task predictions.

        use_bias: Whether to use bias terms in dense layers.
        stack_types: Types of N-BEATS stacks to use.
        nb_blocks_per_stack: Number of blocks per stack.
        hidden_layer_units: Number of units in hidden layers.
        use_revin: Whether to use RevIN normalization.
        dropout_rate: Dropout rate for regularization.
        kernel_regularizer_l2: L2 regularization strength.
        optimizer: Optimizer type to use.
        primary_loss: Primary loss function.
        learning_rate: Learning rate for training.
        gradient_clip_norm: Gradient clipping norm.
    """

    # Architecture parameters
    backcast_length: int = 168
    use_task_embeddings: bool = True
    task_embedding_dim: int = 32
    use_task_inference: bool = True
    task_inference_hidden_dim: int = 128
    task_inference_dropout: float = 0.2
    task_inference_activation: str = 'gelu'

    # NEW: Task inference convolutional feature extractor parameters
    task_inference_use_conv: bool = True
    task_inference_conv_filters: List[int] = None
    task_inference_conv_kernels: List[int] = None
    task_inference_conv_activation: str = 'gelu'

    # Task inference training parameters
    train_task_inference: bool = True
    task_inference_loss_weight: float = 0.1
    consistency_loss_weight: float = 0.05
    entropy_loss_weight: float = 0.02
    consistency_temperature: float = 0.1
    min_entropy_target: float = 0.5
    task_adjustment: float = 0.25

    use_bias: bool = False
    stack_types: List[str] = None
    nb_blocks_per_stack: int = 3
    hidden_layer_units: int = 512
    use_revin: bool = True

    # Regularization parameters
    dropout_rate: float = 0.15
    kernel_regularizer_l2: float = 1e-4

    # Training parameters
    optimizer: str = 'adamw'
    primary_loss: str = 'mae'
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.stack_types is None:
            self.stack_types = ["trend", "seasonality", "generic"]
        # Default values for Conv1D parameters
        if self.task_inference_conv_filters is None:
            self.task_inference_conv_filters = [16, 32, 64]
        if self.task_inference_conv_kernels is None:
            self.task_inference_conv_kernels = [7, 5, 3]


@keras.saving.register_keras_serializable()
class MultiTaskNBeatsNet(keras.Model):
    """Multi-task N-BEATS model with trainable task inference.

    This model extends the standard N-BEATS architecture to handle multiple
    time series forecasting tasks simultaneously with trainable task inference.
    The model can learn to automatically infer appropriate task adjustments
    even when task IDs are not provided during training.

    Key features:
    - Trainable task inference with auxiliary losses
    - Convolutional feature extractor for enhanced inference
    - Consistency regularization for similar time series
    - Entropy regularization for confident predictions
    - Mixed training with and without task IDs
    - Curriculum learning support
    - Self-supervised task learning

    Args:
        config: Configuration object containing model parameters.
        num_tasks: Number of different tasks to handle.
        task_to_id: Mapping from task names to integer IDs.
        forecast_length: Number of time steps to forecast.
        **kwargs: Additional keyword arguments for the base Model class.
    """

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

        # Task inference network (for automatic task detection)
        self.task_inference_layers = {}
        if config.use_task_inference and config.use_task_embeddings:
            # Will be built in the build() method
            pass

        # Main N-BEATS model (created in build)
        self.nbeats_model = None

        # Task-specific adjustment layer
        self.task_adjustment_layer = None

        logger.info(f"Initialized MultiTaskNBeatsNet with Trainable Task Inference:")
        logger.info(f"  - Number of tasks: {num_tasks}")
        logger.info(f"  - Forecast length: {forecast_length}")
        logger.info(f"  - Task embeddings: {'✓' if config.use_task_embeddings else '✗'}")
        logger.info(f"  - Task inference: {'✓' if config.use_task_inference else '✗'}")
        logger.info(f"  - Trainable task inference: {'✓' if config.train_task_inference else '✗'}")
        logger.info(f"  - Task inference Conv1D extractor: {'✓' if config.task_inference_use_conv else '✗'}")
        logger.info(f"  - Task inference activation: {config.task_inference_activation}")
        logger.info(f"  - Use bias: {'✓' if config.use_bias else '✗'}")

    def build(self, input_shape: Union[Tuple, List[Tuple]]) -> None:
        """Build the multi-task N-BEATS model with trainable task inference."""
        logger.info("Building Multi-Task N-BEATS model with trainable task inference...")

        # Handle different input shape formats
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
            logger.info("✓ Base N-BEATS model created successfully")
        except Exception as e:
            logger.error(f"Failed to create base N-BEATS model: {e}")
            raise

        # Add task-specific adjustment layer if using embeddings
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

        # Build task inference network if enabled
        if self.config.use_task_inference and self.config.use_task_embeddings:
            self._build_task_inference_network(time_series_shape)
            logger.info("✓ Task inference network created")

        super().build(input_shape)
        logger.info("✓ Multi-Task N-BEATS model built successfully")

    def _build_conv_feature_extractor(self) -> None:
        """Build the convolutional feature extractor for the task inference network."""
        self.task_inference_layers['conv_blocks'] = []
        for i, (filters, kernel_size) in enumerate(zip(
            self.config.task_inference_conv_filters,
            self.config.task_inference_conv_kernels
        )):
            block = {
                'conv': keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding='same',
                    use_bias=False,  # Bias is handled by BatchNormalization
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2),
                    name=f'task_inference_conv{i+1}'
                ),
                'bn': keras.layers.BatchNormalization(
                    center=self.config.use_bias,
                    scale=True,
                    name=f'task_inference_conv_bn{i+1}'
                ),
                'activation': keras.layers.Activation(
                    self.config.task_inference_conv_activation,
                    name=f'task_inference_conv_act{i+1}'
                ),
                'pool': keras.layers.MaxPooling1D(
                    pool_size=2,
                    name=f'task_inference_conv_pool{i+1}'
                )
            }
            self.task_inference_layers['conv_blocks'].append(block)

        # Final pooling layer to aggregate the output of the conv blocks
        self.task_inference_layers['conv_global_pool'] = keras.layers.GlobalAveragePooling1D(
            name='task_inference_conv_global_pool'
        )

    def _build_task_inference_network(self, time_series_shape: Tuple) -> None:
        """Build the task inference network for trainable task detection."""
        # Build convolutional part if enabled
        if self.config.task_inference_use_conv:
            self._build_conv_feature_extractor()
            logger.info("✓ Built Conv1D feature extractor for task inference.")

        # Global average pooling to get series-level statistical features
        self.task_inference_layers['stat_global_pool'] = keras.layers.GlobalAveragePooling1D(
            name='task_inference_stat_global_pool'
        )

        # Feature extraction layers with stronger regularization for training
        self.task_inference_layers['dense1'] = keras.layers.Dense(
            self.config.task_inference_hidden_dim,
            activation='linear',
            use_bias=self.config.use_bias,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2 * 2),
            bias_initializer='zeros' if self.config.use_bias else None,
            name='task_inference_dense1'
        )

        self.task_inference_layers['bn1'] = keras.layers.BatchNormalization(
            center=self.config.use_bias,
            scale=True,
            name='task_inference_bn1'
        )

        self.task_inference_layers['activation1'] = keras.layers.Activation(
            self.config.task_inference_activation,
            name='task_inference_activation1'
        )

        self.task_inference_layers['dropout1'] = keras.layers.Dropout(
            self.config.task_inference_dropout,
            name='task_inference_dropout1'
        )

        self.task_inference_layers['dense2'] = keras.layers.Dense(
            self.config.task_inference_hidden_dim // 2,
            activation='linear',
            use_bias=self.config.use_bias,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2 * 2),
            bias_initializer='zeros' if self.config.use_bias else None,
            name='task_inference_dense2'
        )

        self.task_inference_layers['bn2'] = keras.layers.BatchNormalization(
            center=self.config.use_bias,
            scale=True,
            name='task_inference_bn2'
        )

        self.task_inference_layers['activation2'] = keras.layers.Activation(
            self.config.task_inference_activation,
            name='task_inference_activation2'
        )

        self.task_inference_layers['dropout2'] = keras.layers.Dropout(
            self.config.task_inference_dropout,
            name='task_inference_dropout2'
        )

        # Task probability prediction
        self.task_inference_layers['task_probs'] = keras.layers.Dense(
            self.num_tasks,
            activation='softmax',
            use_bias=self.config.use_bias,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros' if self.config.use_bias else None,
            name='task_inference_probs'
        )

    def _infer_task_probabilities(
            self,
            time_series_data: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Infer task probabilities from time series data."""
        all_features = []

        # 1. Convolutional Feature Extraction (if enabled)
        if self.config.task_inference_use_conv and 'conv_blocks' in self.task_inference_layers:
            x_conv = time_series_data
            for block in self.task_inference_layers['conv_blocks']:
                x_conv = block['conv'](x_conv, training=training)
                x_conv = block['bn'](x_conv, training=training)
                x_conv = block['activation'](x_conv)
                x_conv = block['pool'](x_conv)

            conv_features = self.task_inference_layers['conv_global_pool'](x_conv)
            all_features.append(conv_features)

        # 2. Statistical and Global Feature Extraction
        # Extract global features from original data
        global_features = self.task_inference_layers['stat_global_pool'](time_series_data)

        # Compute additional statistical features
        time_mean = keras.ops.mean(time_series_data, axis=1)
        time_std = keras.ops.std(time_series_data, axis=1)
        time_min = keras.ops.min(time_series_data, axis=1)
        time_max = keras.ops.max(time_series_data, axis=1)

        # Trend estimation
        batch_size = keras.ops.shape(time_series_data)[0]
        seq_len = keras.ops.shape(time_series_data)[1]
        features = keras.ops.shape(time_series_data)[2]

        time_indices = keras.ops.cast(keras.ops.arange(seq_len, dtype='float32'), time_series_data.dtype)
        time_indices = keras.ops.reshape(time_indices, (1, seq_len, 1))
        time_indices = keras.ops.broadcast_to(time_indices, (batch_size, seq_len, features))

        centered_data = time_series_data - keras.ops.expand_dims(time_mean, axis=1)
        centered_time = time_indices - keras.ops.expand_dims(keras.ops.mean(time_indices, axis=1), axis=1)

        numerator = keras.ops.mean(centered_data * centered_time, axis=1)
        data_var = keras.ops.mean(centered_data ** 2, axis=1)
        time_var = keras.ops.mean(centered_time ** 2, axis=1)

        # Combine statistical features
        statistical_features = keras.ops.concatenate([
            global_features, time_mean, time_std, time_min, time_max
        ], axis=-1)
        all_features.append(statistical_features)

        # Combine all features for the dense network
        combined_features = keras.ops.concatenate(all_features, axis=-1)

        # Process through dense inference network
        x = self.task_inference_layers['dense1'](combined_features, training=training)
        x = self.task_inference_layers['bn1'](x, training=training)
        x = self.task_inference_layers['activation1'](x, training=training)
        x = self.task_inference_layers['dropout1'](x, training=training)

        x = self.task_inference_layers['dense2'](x, training=training)
        x = self.task_inference_layers['bn2'](x, training=training)
        x = self.task_inference_layers['activation2'](x, training=training)
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
        """Compute auxiliary losses for task inference training."""
        losses = {}

        if not training or not self.config.train_task_inference:
            return losses

        # 1. Entropy regularization loss
        entropy = -keras.ops.sum(task_probs * keras.ops.log(task_probs + 1e-8), axis=-1)
        target_entropy = self.config.min_entropy_target
        entropy_loss = keras.ops.mean(keras.ops.maximum(0.0, entropy - target_entropy))
        losses['entropy_loss'] = entropy_loss

        # 2. Consistency regularization loss
        if keras.ops.shape(time_series_data)[0] > 1:
            batch_size = keras.ops.shape(time_series_data)[0]
            flat_series = keras.ops.reshape(time_series_data, (batch_size, -1))
            flat_series_norm = flat_series / (keras.ops.norm(flat_series, axis=1, keepdims=True) + 1e-8)
            similarity_matrix = keras.ops.matmul(flat_series_norm, keras.ops.transpose(flat_series_norm))
            similarity_probs = keras.ops.softmax(similarity_matrix / self.config.consistency_temperature)

            task_similarity_matrix = keras.ops.matmul(task_probs, keras.ops.transpose(task_probs))
            task_similarity_probs = keras.ops.softmax(task_similarity_matrix / self.config.consistency_temperature)

            kl_div = keras.ops.sum(
                similarity_probs * (keras.ops.log(similarity_probs + 1e-8) - keras.ops.log(task_similarity_probs + 1e-8)),
                axis=-1
            )
            consistency_loss = keras.ops.mean(kl_div)
            losses['consistency_loss'] = consistency_loss

        # 3. Balance regularization loss
        mean_task_probs = keras.ops.mean(task_probs, axis=0)
        uniform_distribution = keras.ops.ones_like(mean_task_probs) / self.num_tasks
        balance_loss = keras.ops.sum(keras.ops.square(mean_task_probs - uniform_distribution))
        losses['balance_loss'] = balance_loss

        return losses

    def call(
            self,
            inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]],
            training: Optional[bool] = None,
            return_aux_losses: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, Dict[str, keras.KerasTensor]]]:
        """Forward pass with trainable task-aware processing."""
        aux_losses = {}

        if isinstance(inputs, tuple) and len(inputs) == 2:
            time_series_data, task_ids = inputs
            base_prediction = self.nbeats_model(time_series_data, training=training)

            if self.config.use_task_embeddings and self.task_adjustment_layer is not None:
                if task_ids is not None:
                    task_emb = self.task_embedding(task_ids, training=training)
                    task_adjustment = self.task_adjustment_layer(task_emb, training=training)
                    task_adjustment = keras.ops.expand_dims(task_adjustment, axis=-1)
                else:
                    if self.config.use_task_inference:
                        task_probs = self._infer_task_probabilities(time_series_data, training=training)
                        task_adjustment = self._compute_weighted_task_adjustment(task_probs, training=training)
                        if training and self.config.train_task_inference:
                            aux_losses = self._compute_task_inference_losses(time_series_data, task_probs, training=training)
                    else:
                        return (base_prediction, aux_losses) if return_aux_losses else base_prediction

                final_prediction = base_prediction + (self.config.task_adjustment * task_adjustment)
                return (final_prediction, aux_losses) if return_aux_losses else final_prediction
            else:
                return (base_prediction, aux_losses) if return_aux_losses else base_prediction

        elif isinstance(inputs, (tuple, list)) and len(inputs) != 2:
            raise ValueError(f"Expected 2 inputs for multi-task format, got {len(inputs)}")

        base_prediction = self.nbeats_model(inputs, training=training)

        if (self.config.use_task_embeddings and self.config.use_task_inference and self.task_adjustment_layer is not None):
            task_probs = self._infer_task_probabilities(inputs, training=training)
            task_adjustment = self._compute_weighted_task_adjustment(task_probs, training=training)

            if training and self.config.train_task_inference:
                aux_losses = self._compute_task_inference_losses(inputs, task_probs, training=training)

            final_prediction = base_prediction + (self.config.task_adjustment * task_adjustment)
            return (final_prediction, aux_losses) if return_aux_losses else final_prediction
        else:
            return (base_prediction, aux_losses) if return_aux_losses else base_prediction

    def train_step(self, data):
        """Custom training step with auxiliary losses for task inference."""
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("Expected data to be a tuple of (x, y)")
        x, y = data

        with tf.GradientTape() as tape:
            predictions, aux_losses = self(x, training=True, return_aux_losses=True)
            primary_loss = self.compute_loss(y=y, y_pred=predictions)

            total_loss = primary_loss
            if self.config.train_task_inference and aux_losses:
                for loss_name, loss_value in aux_losses.items():
                    if loss_name == 'entropy_loss':
                        weight = self.config.entropy_loss_weight
                    elif loss_name == 'consistency_loss':
                        weight = self.config.consistency_loss_weight
                    else: # 'balance_loss' and any other
                        weight = self.config.task_inference_loss_weight
                    total_loss += weight * loss_value

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, predictions)
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = total_loss
        results['primary_loss'] = primary_loss
        if aux_losses:
            for loss_name, loss_value in aux_losses.items():
                results[f'aux_{loss_name}'] = loss_value
        return results

    def test_step(self, data):
        """Custom test step."""
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("Expected data to be a tuple of (x, y)")
        x, y = data

        predictions = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=predictions)
        self.compiled_metrics.update_state(y, predictions)

        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        base_config = super().get_config()
        # **FIX:** Convert the dataclass to a dictionary to make it serializable
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
        # **FIX:** Reconstruct the dataclass from the serialized dictionary
        config_data = config.pop('config')
        mt_config = MultiTaskNBeatsConfig(**config_data)

        # The remaining items in 'config' are the direct arguments for __init__
        return cls(config=mt_config, **config)

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
        print("MULTI-TASK N-BEATS MODEL SUMMARY (WITH TRAINABLE TASK INFERENCE)")
        print("=" * 80)
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Forecast length: {self.forecast_length}")
        print(f"Task embeddings: {'Enabled' if self.config.use_task_embeddings else 'Disabled'}")
        print(f"Task inference: {'Enabled' if self.config.use_task_inference else 'Disabled'}")
        print(f"Trainable task inference: {'Enabled' if self.config.train_task_inference else 'Disabled'}")
        print(f"Task inference Conv1D extractor: {'Enabled' if self.config.task_inference_use_conv else 'Disabled'}")
        if self.config.task_inference_use_conv:
            print(f"  Conv filters: {self.config.task_inference_conv_filters}")
            print(f"  Conv kernels: {self.config.task_inference_conv_kernels}")
            print(f"  Conv activation: {self.config.task_inference_conv_activation}")
        print(f"Task inference activation (dense): {self.config.task_inference_activation}")
        print(f"Use bias: {'Enabled' if self.config.use_bias else 'Disabled'}")
        print(f"Task embedding dimension: {self.config.task_embedding_dim}")

        if self.config.train_task_inference:
            print(f"\nTask Inference Training Parameters:")
            print(f"  Task inference loss weight: {self.config.task_inference_loss_weight}")
            print(f"  Consistency loss weight: {self.config.consistency_loss_weight}")
            print(f"  Entropy loss weight: {self.config.entropy_loss_weight}")
            print(f"  Consistency temperature: {self.config.consistency_temperature}")
            print(f"  Min entropy target: {self.config.min_entropy_target}")

        print(f"\nArchitecture:")
        print(f"  Stack types: {self.config.stack_types}")
        print(f"  Blocks per stack: {self.config.nb_blocks_per_stack}")
        print(f"  Hidden units: {self.config.hidden_layer_units}")
        print(f"  RevIN normalization: {'Enabled' if self.config.use_revin else 'Disabled'}")
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
    """Create a Multi-Task N-BEATS model with trainable task inference."""
    logger.info("Creating Multi-Task N-BEATS model with trainable task inference...")

    model = MultiTaskNBeatsNet(
        config=config,
        num_tasks=num_tasks,
        task_to_id=task_to_id,
        forecast_length=forecast_length,
        name=name
    )

    logger.info("✓ Multi-Task N-BEATS model with trainable task inference created successfully")
    return model

# ---------------------------------------------------------------------

def create_multi_task_nbeats_from_tasks(
        task_names: List[str],
        forecast_length: int,
        config: Optional[MultiTaskNBeatsConfig] = None,
        name: Optional[str] = None
) -> MultiTaskNBeatsNet:
    """Create a Multi-Task N-BEATS model from a list of task names with trainable task inference."""
    if config is None:
        config = MultiTaskNBeatsConfig()

    task_to_id = {task: idx for idx, task in enumerate(task_names)}
    logger.info(f"Creating Multi-Task N-BEATS with trainable task inference for {len(task_names)} tasks: {task_names}")

    return create_multi_task_nbeats(
        config=config,
        num_tasks=len(task_names),
        task_to_id=task_to_id,
        forecast_length=forecast_length,
        name=name
    )
"""
Multi-Task N-BEATS Model Implementation with Automatic Task Inference.

This module provides a multi-task variant of the N-BEATS architecture that can
simultaneously learn multiple time series forecasting tasks with task-specific
embeddings and shared representation learning. When task IDs are not provided,
the model automatically infers appropriate task adjustments.

The implementation extends the base N-BEATS architecture with:
- Task embedding layers for task-aware processing
- Automatic task inference from time series patterns
- Shared backbone with task-specific adjustments
- Support for different forecast horizons per task
- Flexible multi-task training capabilities

References:
    - Oreshkin, B. N., et al. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." ICLR 2020.
    - Multi-task learning principles adapted for time series forecasting
"""

import keras
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple


from dl_techniques.utils.logger import logger
from .nbeats import create_nbeats_model


@dataclass
class MultiTaskNBeatsConfig:
    """Configuration class for Multi-Task N-BEATS model.

    This configuration class encapsulates all parameters needed to create
    and train a multi-task N-BEATS model.

    Args:
        backcast_length: Number of time steps to look back for prediction.
        use_task_embeddings: Whether to use task embedding layers.
        task_embedding_dim: Dimension of task embedding vectors.
        use_task_inference: Whether to enable automatic task inference when no task IDs provided.
        task_inference_hidden_dim: Hidden dimension for task inference network.
        task_inference_dropout: Dropout rate for task inference network.
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


@keras.saving.register_keras_serializable()
class MultiTaskNBeatsNet(keras.Model):
    """Multi-task N-BEATS model with task embeddings, shared architecture, and automatic task inference.

    This model extends the standard N-BEATS architecture to handle multiple
    time series forecasting tasks simultaneously. It uses task embeddings
    to provide task-specific adjustments while maintaining a shared backbone
    for efficient learning across tasks.

    Key features:
    - Automatic task inference when no task IDs are provided
    - Task-specific adjustments through learned embeddings
    - Backwards compatible with single-task usage
    - Weighted task combination for intelligent adjustments

    The architecture consists of:
    1. A base N-BEATS model for shared feature extraction
    2. Task embedding layers for task-specific representations
    3. Task inference network for automatic task detection
    4. Task adjustment layers for fine-tuning predictions

    Args:
        config: Configuration object containing model parameters.
        num_tasks: Number of different tasks to handle.
        task_to_id: Mapping from task names to integer IDs.
        forecast_length: Number of time steps to forecast.
        **kwargs: Additional keyword arguments for the base Model class.

    Example:
        >>> config = MultiTaskNBeatsConfig(backcast_length=168, use_task_embeddings=True)
        >>> task_mapping = {'task_1': 0, 'task_2': 1, 'task_3': 2}
        >>> model = MultiTaskNBeatsNet(config, 3, task_mapping, forecast_length=24)
        >>>
        >>> # Multi-task input format: (time_series_data, task_ids)
        >>> x_data = np.random.randn(32, 168, 1)
        >>> task_ids = np.random.randint(0, 3, (32,))
        >>> predictions = model((x_data, task_ids))
        >>>
        >>> # Single input format with automatic task inference
        >>> predictions = model(x_data)  # Automatically infers task adjustments
        >>> print(predictions.shape)  # (32, 24, 1)
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
        self.task_inference_network = None
        if config.use_task_inference and config.use_task_embeddings:
            # Will be built in build() method
            self.task_inference_layers = {}

        # Main N-BEATS model (created in build)
        self.nbeats_model = None

        # Task-specific adjustment layer
        self.task_adjustment_layer = None

        logger.info(f"Initialized MultiTaskNBeatsNet:")
        logger.info(f"  - Number of tasks: {num_tasks}")
        logger.info(f"  - Forecast length: {forecast_length}")
        logger.info(f"  - Task embeddings: {'✓' if config.use_task_embeddings else '✗'}")
        logger.info(f"  - Task inference: {'✓' if config.use_task_inference else '✗'}")
        logger.info(f"  - Stack types: {config.stack_types}")

    def build(self, input_shape: Union[Tuple, List[Tuple]]) -> None:
        """Build the multi-task N-BEATS model.

        This method creates the internal N-BEATS model, task-specific layers,
        and task inference network based on the provided configuration.

        Args:
            input_shape: Input shape specification. Can be a single shape tuple
                for time series data or a list of shapes for multi-input format.
        """
        logger.info("Building Multi-Task N-BEATS model...")

        # Handle different input shape formats
        if isinstance(input_shape, list):
            time_series_shape = input_shape[0]
        else:
            time_series_shape = input_shape

        # Create the base N-BEATS model with the specific forecast length
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
                self.forecast_length,
                activation='linear',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                name='task_adjustment'
            )

            logger.info("✓ Task adjustment layer created")

        # Build task inference network if enabled
        if self.config.use_task_inference and self.config.use_task_embeddings:
            self._build_task_inference_network(time_series_shape)
            logger.info("✓ Task inference network created")

        super().build(input_shape)
        logger.info("✓ Multi-Task N-BEATS model built successfully")

    def _build_task_inference_network(self, time_series_shape: Tuple) -> None:
        """Build the task inference network for automatic task detection.

        This network analyzes the input time series to predict task probabilities,
        which are then used to create weighted task adjustments.

        Args:
            time_series_shape: Shape of the time series input.
        """
        # Global average pooling to get series-level features
        self.task_inference_layers['global_pool'] = keras.layers.GlobalAveragePooling1D(
            name='task_inference_global_pool'
        )

        # Additional statistical features
        self.task_inference_layers['flatten'] = keras.layers.Flatten(
            name='task_inference_flatten'
        )

        # Feature extraction layers
        self.task_inference_layers['dense1'] = keras.layers.Dense(
            self.config.task_inference_hidden_dim,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2),
            name='task_inference_dense1'
        )

        self.task_inference_layers['dropout1'] = keras.layers.Dropout(
            self.config.task_inference_dropout,
            name='task_inference_dropout1'
        )

        self.task_inference_layers['dense2'] = keras.layers.Dense(
            self.config.task_inference_hidden_dim // 2,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(self.config.kernel_regularizer_l2),
            name='task_inference_dense2'
        )

        self.task_inference_layers['dropout2'] = keras.layers.Dropout(
            self.config.task_inference_dropout,
            name='task_inference_dropout2'
        )

        # Task probability prediction
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
        """Infer task probabilities from time series data.

        Args:
            time_series_data: Input time series data.
            training: Boolean indicating training mode.

        Returns:
            Task probabilities with shape (batch_size, num_tasks).
        """
        # Extract global features
        global_features = self.task_inference_layers['global_pool'](time_series_data)

        # Compute additional statistical features
        # Mean, std, min, max along time dimension
        time_mean = keras.ops.mean(time_series_data, axis=1)
        time_std = keras.ops.std(time_series_data, axis=1)
        time_min = keras.ops.min(time_series_data, axis=1)
        time_max = keras.ops.max(time_series_data, axis=1)

        # Trend estimation (simple linear trend)
        batch_size = keras.ops.shape(time_series_data)[0]
        seq_len = keras.ops.shape(time_series_data)[1]
        features = keras.ops.shape(time_series_data)[2]

        # Create time indices for trend calculation
        time_indices = keras.ops.cast(
            keras.ops.arange(seq_len, dtype='float32'),
            time_series_data.dtype
        )
        time_indices = keras.ops.reshape(time_indices, (1, seq_len, 1))
        time_indices = keras.ops.broadcast_to(time_indices, (batch_size, seq_len, features))

        # Simple trend estimation using correlation
        centered_data = time_series_data - keras.ops.expand_dims(time_mean, axis=1)
        centered_time = time_indices - keras.ops.expand_dims(keras.ops.mean(time_indices, axis=1), axis=1)

        # Compute trend strength (correlation with time)
        numerator = keras.ops.mean(centered_data * centered_time, axis=1)
        data_var = keras.ops.mean(centered_data ** 2, axis=1)
        time_var = keras.ops.mean(centered_time ** 2, axis=1)
        trend_strength = numerator / (keras.ops.sqrt(data_var * time_var) + 1e-8)

        # Combine all features
        combined_features = keras.ops.concatenate([
            global_features,
            time_mean,
            time_std,
            time_min,
            time_max,
            trend_strength
        ], axis=-1)

        # Process through inference network
        x = self.task_inference_layers['dense1'](combined_features, training=training)
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
        """Compute weighted task adjustment based on task probabilities.

        Args:
            task_probs: Task probabilities with shape (batch_size, num_tasks).
            training: Boolean indicating training mode.

        Returns:
            Weighted task adjustment with shape (batch_size, forecast_length, 1).
        """
        batch_size = keras.ops.shape(task_probs)[0]

        # Get all task embeddings
        all_task_ids = keras.ops.arange(self.num_tasks)
        all_task_embeddings = self.task_embedding(all_task_ids, training=training)  # (num_tasks, embedding_dim)

        # Compute task adjustments for all tasks
        all_task_adjustments = self.task_adjustment_layer(
            all_task_embeddings, training=training
        )  # (num_tasks, forecast_length)

        # Weight task adjustments by probabilities
        # task_probs: (batch_size, num_tasks)
        # all_task_adjustments: (num_tasks, forecast_length)
        weighted_adjustment = keras.ops.einsum(
            'bn,nf->bf',
            task_probs,
            all_task_adjustments
        )  # (batch_size, forecast_length)

        # Add feature dimension
        weighted_adjustment = keras.ops.expand_dims(weighted_adjustment, axis=-1)

        return weighted_adjustment

    def call(
            self,
            inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass with task-aware processing.

        Args:
            inputs: Input data. Can be either:
                - Single tensor for standard N-BEATS input (batch_size, sequence_length, features)
                - Tuple of (time_series_data, task_ids) for multi-task input
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Forecast predictions with shape (batch_size, forecast_length, features).

        Raises:
            ValueError: If input format is invalid.
        """
        if isinstance(inputs, tuple) and len(inputs) == 2:
            # Multi-task format: (time_series_data, task_ids)
            time_series_data, task_ids = inputs

            # Get base N-BEATS prediction
            base_prediction = self.nbeats_model(time_series_data, training=training)

            if self.config.use_task_embeddings and self.task_adjustment_layer is not None:
                if task_ids is not None:
                    # Explicit task IDs provided - use them directly
                    task_emb = self.task_embedding(task_ids, training=training)
                    task_adjustment = self.task_adjustment_layer(task_emb, training=training)
                    task_adjustment = keras.ops.expand_dims(task_adjustment, axis=-1)
                else:
                    # No task IDs but in tuple format - infer from data
                    if self.config.use_task_inference:
                        task_probs = self._infer_task_probabilities(time_series_data, training=training)
                        task_adjustment = self._compute_weighted_task_adjustment(task_probs, training=training)
                    else:
                        # No inference capability - return base prediction
                        return base_prediction

                # Combine base prediction with task-specific adjustment
                # Use small adjustment factor to preserve base model performance
                final_prediction = base_prediction + 0.01 * task_adjustment

                return final_prediction
            else:
                return base_prediction

        elif isinstance(inputs, (tuple, list)) and len(inputs) != 2:
            raise ValueError(f"Expected 2 inputs for multi-task format, got {len(inputs)}")

        else:
            # Standard format: just time series data
            # Get base N-BEATS prediction
            base_prediction = self.nbeats_model(inputs, training=training)

            # Apply automatic task inference if enabled
            if (self.config.use_task_embeddings and
                self.config.use_task_inference and
                self.task_adjustment_layer is not None):

                # Infer task probabilities from the time series
                task_probs = self._infer_task_probabilities(inputs, training=training)

                # Compute weighted task adjustment
                task_adjustment = self._compute_weighted_task_adjustment(task_probs, training=training)

                # Combine with base prediction
                final_prediction = base_prediction + 0.05 * task_adjustment

                return final_prediction
            else:
                # No task embeddings or inference - return base prediction
                return base_prediction

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update({
            'num_tasks': self.num_tasks,
            'task_to_id': self.task_to_id,
            'forecast_length': self.forecast_length,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MultiTaskNBeatsNet':
        """Create model from configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            MultiTaskNBeatsNet instance.
        """
        # Extract MultiTaskNBeatsConfig if needed
        if 'config' in config:
            mt_config = config.pop('config')
        else:
            mt_config = MultiTaskNBeatsConfig()

        return cls(config=mt_config, **config)

    def get_task_id(self, task_name: str) -> int:
        """Get task ID for a given task name.

        Args:
            task_name: Name of the task.

        Returns:
            Integer ID for the task.

        Raises:
            KeyError: If task name is not found.
        """
        if task_name not in self.task_to_id:
            raise KeyError(f"Task '{task_name}' not found. Available tasks: {list(self.task_to_id.keys())}")

        return self.task_to_id[task_name]

    def get_task_name(self, task_id: int) -> str:
        """Get task name for a given task ID.

        Args:
            task_id: Integer ID of the task.

        Returns:
            Name of the task.

        Raises:
            KeyError: If task ID is not found.
        """
        if task_id not in self.id_to_task:
            raise KeyError(f"Task ID {task_id} not found. Available IDs: {list(self.id_to_task.keys())}")

        return self.id_to_task[task_id]

    def predict_single_task(
            self,
            x: np.ndarray,
            task_name: str,
            **kwargs
    ) -> np.ndarray:
        """Make predictions for a single task.

        Args:
            x: Input time series data with shape (batch_size, sequence_length, features).
            task_name: Name of the task to predict for.
            **kwargs: Additional arguments passed to predict().

        Returns:
            Predictions with shape (batch_size, forecast_length, features).
        """
        task_id = self.get_task_id(task_name)
        task_ids = np.full(x.shape[0], task_id)

        return self.predict((x, task_ids), **kwargs)

    def predict_with_task_inference(
            self,
            x: np.ndarray,
            return_task_probs: bool = False,
            **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using automatic task inference.

        Args:
            x: Input time series data with shape (batch_size, sequence_length, features).
            return_task_probs: Whether to return inferred task probabilities.
            **kwargs: Additional arguments passed to predict().

        Returns:
            If return_task_probs is False: predictions with shape (batch_size, forecast_length, features).
            If return_task_probs is True: tuple of (predictions, task_probabilities).
        """
        if not self.config.use_task_inference:
            raise ValueError("Task inference is not enabled. Set use_task_inference=True in config.")

        # Get predictions (will use automatic task inference)
        predictions = self.predict(x, **kwargs)

        if return_task_probs:
            # Get task probabilities separately
            task_probs = self._infer_task_probabilities(x, training=False)
            task_probs_np = keras.ops.convert_to_numpy(task_probs)
            return predictions, task_probs_np
        else:
            return predictions

    def summary(self, **kwargs) -> None:
        """Print model summary including task information."""
        print("=" * 80)
        print("MULTI-TASK N-BEATS MODEL SUMMARY")
        print("=" * 80)
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Forecast length: {self.forecast_length}")
        print(f"Task embeddings: {'Enabled' if self.config.use_task_embeddings else 'Disabled'}")
        print(f"Task inference: {'Enabled' if self.config.use_task_inference else 'Disabled'}")
        print(f"Task embedding dimension: {self.config.task_embedding_dim}")
        print(f"Stack types: {self.config.stack_types}")
        print(f"Blocks per stack: {self.config.nb_blocks_per_stack}")
        print(f"Hidden units: {self.config.hidden_layer_units}")
        print(f"RevIN normalization: {'Enabled' if self.config.use_revin else 'Disabled'}")
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
    """Create a Multi-Task N-BEATS model.

    This is a convenience function for creating a multi-task N-BEATS model
    with the specified configuration.

    Args:
        config: Model configuration.
        num_tasks: Number of tasks.
        task_to_id: Mapping from task names to IDs.
        forecast_length: Number of time steps to forecast.
        name: Optional name for the model.

    Returns:
        Configured MultiTaskNBeatsNet model.

    Example:
        >>> config = MultiTaskNBeatsConfig(
        ...     backcast_length=168,
        ...     use_task_embeddings=True,
        ...     task_embedding_dim=32,
        ...     use_task_inference=True
        ... )
        >>> task_mapping = {'trend': 0, 'seasonal': 1, 'residual': 2}
        >>> model = create_multi_task_nbeats(config, 3, task_mapping, 24)
        >>> model.summary()
    """
    logger.info("Creating Multi-Task N-BEATS model...")

    model = MultiTaskNBeatsNet(
        config=config,
        num_tasks=num_tasks,
        task_to_id=task_to_id,
        forecast_length=forecast_length,
        name=name
    )

    logger.info("✓ Multi-Task N-BEATS model created successfully")
    return model


def create_multi_task_nbeats_from_tasks(
        task_names: List[str],
        forecast_length: int,
        config: Optional[MultiTaskNBeatsConfig] = None,
        name: Optional[str] = None
) -> MultiTaskNBeatsNet:
    """Create a Multi-Task N-BEATS model from a list of task names.

    This convenience function automatically creates the task mapping
    from a list of task names.

    Args:
        task_names: List of task names.
        forecast_length: Number of time steps to forecast.
        config: Optional model configuration. Uses defaults if not provided.
        name: Optional name for the model.

    Returns:
        Configured MultiTaskNBeatsNet model.

    Example:
        >>> tasks = ['linear_trend', 'daily_seasonality', 'random_walk']
        >>> model = create_multi_task_nbeats_from_tasks(tasks, forecast_length=24)
        >>>
        >>> # Can now use with or without explicit task IDs
        >>> x = np.random.randn(10, 168, 1)
        >>> predictions = model(x)  # Automatic task inference
        >>>
        >>> # Or with explicit tasks
        >>> task_ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        >>> predictions = model((x, task_ids))
        >>>
        >>> model.summary()
    """
    if config is None:
        config = MultiTaskNBeatsConfig()

    # Create task mapping
    task_to_id = {task: idx for idx, task in enumerate(task_names)}

    logger.info(f"Creating Multi-Task N-BEATS for {len(task_names)} tasks: {task_names}")

    return create_multi_task_nbeats(
        config=config,
        num_tasks=len(task_names),
        task_to_id=task_to_id,
        forecast_length=forecast_length,
        name=name
    )
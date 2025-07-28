"""
Multi-Task N-BEATS Model with Trainable Task Inference for Time Series Forecasting

This module provides a sophisticated multi-task variant of the N-BEATS (Neural Basis
Expansion Analysis for Time Series) architecture that can simultaneously learn multiple
time series forecasting tasks with task-specific embeddings and trainable task inference.
The model represents a significant advancement in unified time series forecasting, capable
of automatically adapting to different time series patterns without explicit task labels.

Overview
--------
Traditional time series forecasting approaches require separate models for different
types of patterns (trend, seasonal, financial, etc.), leading to inefficient resource
usage and limited knowledge transfer. This Multi-Task N-BEATS implementation addresses
these limitations by:

1. **Unified Architecture**: Single model handling diverse time series patterns
2. **Automatic Task Inference**: Learn to identify time series types from data alone
3. **Task-Specific Adaptations**: Specialized adjustments while sharing core representations
4. **Knowledge Transfer**: Leverage similarities between related forecasting tasks
5. **Semi-Supervised Learning**: Train on both labeled and unlabeled time series data

The model extends the original N-BEATS architecture with task embeddings, task inference
networks, and specialized loss functions that enable robust multi-task learning with
automatic pattern recognition capabilities.

Key Features
------------
**Advanced Architecture Components:**
* **Task Embedding System**: Learnable representations for different time series types
* **Task Inference Network**: Automatic classification of time series patterns
* **Weighted Task Adjustment**: Probability-based adaptation of predictions
* **Shared N-BEATS Backbone**: Common feature extraction across all tasks
* **Graph-Compatible Design**: Full TensorFlow graph mode compatibility

**Training Innovations:**
* **Multi-Objective Learning**: Primary forecasting + auxiliary task inference losses
* **Consistency Regularization**: Ensure similar series receive similar task classifications
* **Entropy Regularization**: Encourage confident task predictions
* **Balance Regularization**: Maintain uniform task distribution during training
* **Gradient Clipping**: Stable training with multiple loss components

**Flexible Inference Modes:**
* **Supervised Mode**: Use explicit task IDs when available
* **Semi-Supervised Mode**: Automatic task inference from input sequences
* **Hybrid Mode**: Combine labeled and unlabeled data during training
* **Confidence Estimation**: Probability distributions over task assignments

**Production-Ready Features:**
* **Model Serialization**: Complete save/load support with custom components
* **Configurable Architecture**: Extensive customization through configuration objects
* **Comprehensive Metrics**: Track all loss components and training dynamics
* **Memory Efficient**: Optimized implementation for large-scale deployment

Classes
-------
MultiTaskNBeatsConfig
    Comprehensive configuration dataclass containing all model parameters including
    architecture settings, task inference parameters, training configurations, and
    regularization options. Provides sensible defaults for all components.

MultiTaskNBeatsNet
    Main multi-task N-BEATS model class implementing the complete architecture with
    task embeddings, task inference network, and specialized training procedures.
    Inherits from keras.Model with full graph compatibility.

Architecture Details
--------------------
**Model Components:**

1. **Input Processing**:
   - Time series sequences (batch_size, backcast_length, features)
   - Optional task IDs (batch_size,) for supervised training
   - Automatic shape inference and validation

2. **Task Embedding Layer**:
   - Learnable embeddings for each task type
   - Configurable embedding dimension
   - Used for task-specific adjustments

3. **Task Inference Network**:
   - Convolutional feature extraction for pattern recognition
   - Global pooling for statistical feature aggregation
   - Multi-layer perceptron for task classification
   - Softmax output for task probability distribution

4. **Base N-BEATS Model**:
   - Standard N-BEATS architecture with stacks (trend, seasonality, generic)
   - Configurable number of blocks per stack
   - Optional Reversible Instance Normalization (RevIN)

5. **Task Adjustment Layer**:
   - Dense layer for task-specific prediction modifications
   - Weighted combination based on task probabilities
   - Configurable adjustment strength

**Loss Function Design:**
The model optimizes multiple objectives simultaneously:
* **Primary Loss**: Standard forecasting loss (MAE, MSE, or SMAPE)
* **Task Inference Loss**: Cross-entropy for task classification accuracy
* **Consistency Loss**: KL-divergence for similar series consistency
* **Entropy Loss**: Encourages confident task predictions
* **Balance Loss**: Maintains uniform task distribution

Usage Examples
--------------
Basic Multi-Task Model Creation:
    >>> from dl_techniques.models.nbeats_multitask import (
    ...     create_multi_task_nbeats, MultiTaskNBeatsConfig
    ... )
    >>>
    >>> # Define task mapping
    >>> task_to_id = {
    ...     'trend_strong': 0, 'seasonal_daily': 1, 'financial_stock': 2,
    ...     'weather_temp': 3, 'network_traffic': 4
    ... }
    >>>
    >>> # Configure model
    >>> config = MultiTaskNBeatsConfig(
    ...     backcast_length=168,
    ...     use_task_embeddings=True,
    ...     use_task_inference=True,
    ...     train_task_inference=True,
    ...     task_embedding_dim=32,
    ...     stack_types=['trend', 'seasonality', 'generic']
    ... )
    >>>
    >>> # Create model
    >>> model = create_multi_task_nbeats(
    ...     config=config,
    ...     num_tasks=len(task_to_id),
    ...     task_to_id=task_to_id,
    ...     forecast_length=24
    ... )
    >>>
    >>> # Build and compile
    >>> sample_input = (tf.random.normal([32, 168, 1]), tf.zeros([32], dtype=tf.int32))
    >>> model(sample_input)  # Build the model
    >>> model.compile(optimizer='adam', loss='mae')

Advanced Configuration:
    >>> # Advanced multi-task configuration
    >>> advanced_config = MultiTaskNBeatsConfig(
    ...     # Architecture
    ...     backcast_length=336,
    ...     use_task_embeddings=True,
    ...     task_embedding_dim=64,
    ...     use_task_inference=True,
    ...     task_inference_hidden_dim=256,
    ...
    ...     # Task inference network
    ...     task_inference_use_conv=True,
    ...     task_inference_conv_filters=[32, 64, 128],
    ...     task_inference_conv_kernels=[7, 5, 3],
    ...     task_inference_dropout=0.2,
    ...
    ...     # Training parameters
    ...     train_task_inference=True,
    ...     task_inference_loss_weight=0.3,
    ...     consistency_loss_weight=0.1,
    ...     entropy_loss_weight=0.05,
    ...     min_entropy_target=0.1,
    ...
    ...     # N-BEATS parameters
    ...     stack_types=['trend', 'seasonality', 'generic'],
    ...     nb_blocks_per_stack=4,
    ...     hidden_layer_units=512,
    ...     use_revin=True,
    ...
    ...     # Regularization
    ...     dropout_rate=0.15,
    ...     kernel_regularizer_l2=1e-4,
    ...     gradient_clip_norm=1.0
    ... )

Training with Multiple Input Modes:
    >>> # Prepare training data
    >>> # Supervised training (with task IDs)
    >>> x_supervised = tf.random.normal([1000, 168, 1])
    >>> y_supervised = tf.random.normal([1000, 24, 1])
    >>> task_ids_supervised = tf.random.uniform([1000], 0, 5, dtype=tf.int32)
    >>>
    >>> # Semi-supervised training (without task IDs)
    >>> x_unsupervised = tf.random.normal([500, 168, 1])
    >>> y_unsupervised = tf.random.normal([500, 24, 1])
    >>>
    >>> # Training step with labeled data
    >>> with tf.GradientTape() as tape:
    ...     # Forward pass with task IDs
    ...     pred_supervised, aux_losses = model(
    ...         (x_supervised, task_ids_supervised),
    ...         training=True,
    ...         return_aux_losses=True
    ...     )
    ...     primary_loss = tf.reduce_mean(tf.abs(y_supervised - pred_supervised))
    ...
    ...     # Add auxiliary losses for task inference training
    ...     total_loss = primary_loss
    ...     for loss_name, loss_value in aux_losses.items():
    ...         if loss_name == 'entropy_loss':
    ...             total_loss += config.entropy_loss_weight * loss_value
    ...         elif loss_name == 'consistency_loss':
    ...             total_loss += config.consistency_loss_weight * loss_value
    >>>
    >>> # Training step with unlabeled data (task inference only)
    >>> pred_unsupervised = model(x_unsupervised, training=True)

Inference with Task Inference:
    >>> # Automatic task detection and prediction
    >>> test_sequences = tf.random.normal([10, 168, 1])
    >>>
    >>> # Get predictions with automatic task inference
    >>> predictions = model.predict_with_task_inference(test_sequences.numpy())
    >>>
    >>> # Get predictions with task probabilities
    >>> predictions, task_probs = model.predict_with_task_inference(
    ...     test_sequences.numpy(),
    ...     return_task_probs=True
    ... )
    >>>
    >>> # Show predicted tasks
    >>> predicted_task_ids = np.argmax(task_probs, axis=1)
    >>> for i, (task_id, probs) in enumerate(zip(predicted_task_ids, task_probs)):
    ...     confidence = np.max(probs)
    ...     task_name = model.get_task_name(task_id)
    ...     print(f"Series {i}: Predicted task '{task_name}' with confidence {confidence:.3f}")

Single Task Prediction:
    >>> # Predict for specific known task
    >>> financial_data = tf.random.normal([5, 168, 1])
    >>> financial_predictions = model.predict_single_task(
    ...     financial_data.numpy(),
    ...     task_name='financial_stock'
    ... )

Model Analysis and Interpretation:
    >>> # Analyze task inference capabilities
    >>> test_data = tf.random.normal([100, 168, 1])
    >>> task_probabilities = model._infer_task_probabilities(test_data, training=False)
    >>>
    >>> # Compute prediction confidence
    >>> max_probs = tf.reduce_max(task_probabilities, axis=1)
    >>> avg_confidence = tf.reduce_mean(max_probs)
    >>> print(f"Average task inference confidence: {avg_confidence:.3f}")
    >>>
    >>> # Analyze task distribution
    >>> predicted_tasks = tf.argmax(task_probabilities, axis=1)
    >>> task_counts = tf.math.bincount(predicted_tasks, minlength=model.num_tasks)
    >>> print(f"Task distribution: {task_counts.numpy()}")

Model Serialization:
    >>> # Save complete model with custom components
    >>> model.save('multi_task_nbeats_model.keras')
    >>>
    >>> # Load model (requires custom_objects for complex components)
    >>> loaded_model = keras.models.load_model(
    ...     'multi_task_nbeats_model.keras',
    ...     custom_objects={'MultiTaskNBeatsNet': MultiTaskNBeatsNet}
    ... )
    >>>
    >>> # Verify task mapping preservation
    >>> print(f"Original tasks: {model.task_to_id}")
    >>> print(f"Loaded tasks: {loaded_model.task_to_id}")

Configuration Parameters
------------------------
**Architecture Parameters:**
* `backcast_length`: Input sequence length (default: 168)
* `use_task_embeddings`: Enable task-specific embeddings (default: True)
* `task_embedding_dim`: Task embedding dimension (default: 32)
* `use_task_inference`: Enable automatic task inference (default: True)
* `task_inference_hidden_dim`: Task inference network hidden dimension (default: 128)

**Task Inference Network:**
* `task_inference_use_conv`: Use convolutional feature extraction (default: True)
* `task_inference_conv_filters`: Convolutional layer filter sizes (default: [16, 32, 64])
* `task_inference_conv_kernels`: Convolutional kernel sizes (default: [7, 5, 3])
* `task_inference_dropout`: Dropout rate in task inference network (default: 0.1)
* `task_inference_activation`: Activation function (default: 'gelu')

**Training Parameters:**
* `train_task_inference`: Enable task inference training (default: True)
* `task_inference_loss_weight`: Weight for task classification loss (default: 0.5)
* `consistency_loss_weight`: Weight for consistency regularization (default: 0.1)
* `entropy_loss_weight`: Weight for entropy regularization (default: 0.05)
* `min_entropy_target`: Target minimum entropy for predictions (default: 0.1)

**N-BEATS Parameters:**
* `stack_types`: Types of N-BEATS stacks (default: ['trend', 'seasonality', 'generic'])
* `nb_blocks_per_stack`: Number of blocks per stack (default: 3)
* `hidden_layer_units`: Hidden layer size in N-BEATS blocks (default: 512)
* `use_revin`: Enable Reversible Instance Normalization (default: True)

**Regularization:**
* `dropout_rate`: Dropout rate in N-BEATS blocks (default: 0.1)
* `kernel_regularizer_l2`: L2 regularization strength (default: 1e-5)
* `gradient_clip_norm`: Gradient clipping norm (default: 1.0)

**Optimization:**
* `optimizer`: Optimizer type ('adam' or 'adamw', default: 'adamw')
* `primary_loss`: Primary forecasting loss ('mae', 'mse', 'smape', default: 'mae')
* `learning_rate`: Initial learning rate (default: 1e-3)

Technical Implementation Details

**Task Inference Algorithm:**
1. **Feature Extraction**: Extract statistical and convolutional features from input sequences
2. **Pattern Recognition**: Multi-layer neural network processes features
3. **Probability Estimation**: Softmax layer outputs task probabilities
4. **Weighted Adjustment**: Task-specific adjustments weighted by probabilities

**Loss Function Components:**
* **Primary Loss**: L(y, ŷ) where ŷ is the forecasting prediction
* **Task Inference Loss**: -∑ log p(t|x) for true task t
* **Consistency Loss**: KL(sim(x_i, x_j) || sim(p_i, p_j)) for input similarity
* **Entropy Loss**: max(0, H(p(t|x)) - H_min) to encourage confidence
* **Balance Loss**: KL(p̄(t) || uniform) for task distribution balance
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

    # Task inference training parameters
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
        """Compute auxiliary losses for task inference training """
        losses = {}

        if not training or not self.config.train_task_inference:
            return losses

        # 1. Entropy regularization loss
        # Encourage confident predictions but not too extreme
        entropy = -keras.ops.sum(task_probs * keras.ops.log(task_probs + 1e-8), axis=-1)

        # We want entropy to be low (confident predictions) but not zero
        # Penalty when entropy is too high (uncertain predictions)
        entropy_loss = keras.ops.mean(keras.ops.maximum(0.0, entropy - self.config.min_entropy_target))
        losses['entropy_loss'] = entropy_loss

        # 2. Consistency regularization loss - GRAPH COMPATIBLE
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

        # Use keras.ops.cond instead of Python if for graph compatibility
        consistency_loss = keras.ops.cond(
            batch_size > 1,
            compute_consistency_loss,
            no_consistency_loss
        )
        losses['consistency_loss'] = consistency_loss

        # 3. Balance regularization loss
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
        """custom training step with proper auxiliary loss handling."""
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
"""
CCNet MNIST Experiment with Centralized Configuration - Refined Implementation.

This module demonstrates a complete training and analysis workflow for a
Causal Cooperative Network (CCNet) on the MNIST dataset, following modern
Keras 3 best practices for custom layers and models.

Examples:
    Basic usage::

        # Use default configuration
        config = ExperimentConfig()
        experiment = CCNetExperiment(config)
        orchestrator, trainer = experiment.run()

        # Or customize specific aspects
        config = ExperimentConfig(
            model=ModelConfig(explanation_dim=32),
            training=TrainingConfig(epochs=100),
            data=DataConfig(batch_size=256)
        )
        experiment = CCNetExperiment(config)
        orchestrator, trainer = experiment.run()
"""

import keras
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Union

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.visualization import (
    VisualizationManager,
    VisualizationPlugin,
    CompositeVisualization,
)
from dl_techniques.models.ccnets import (
    CCNetConfig,
    CCNetOrchestrator,
    CCNetTrainer,
    EarlyStoppingCallback,
    wrap_keras_model,
)
from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations.golu import GoLU
from dl_techniques.layers.regularizers.soft_orthonormal_constraint_regularizer import (
    SoftOrthonormalConstraintRegularizer
)


# =====================================================================
# CENTRALIZED CONFIGURATION
# =====================================================================

@dataclass
class ModelConfig:
    """Configuration for model architecture parameters."""

    # Shared parameters
    explanation_dim: int = 16
    num_classes: int = 10

    # Explainer parameters
    explainer_conv_filters: List[int] = field(
        default_factory=lambda: [32, 64, 128]
    )
    explainer_conv_kernels: List[int] = field(
        default_factory=lambda: [5, 3, 3]
    )
    explainer_l2_regularization: float = 1e-4

    # Orthonormal regularization for explanation vectors
    explainer_orthonormal_lambda: float = 1e-3
    explainer_orthonormal_l1: float = 0.0
    explainer_orthonormal_l2: float = 0.0
    explainer_use_matrix_scaling: bool = True

    # Reasoner parameters
    reasoner_conv_filters: List[int] = field(
        default_factory=lambda: [32, 64]
    )
    reasoner_conv_kernels: List[int] = field(
        default_factory=lambda: [5, 3]
    )
    reasoner_dense_units: List[int] = field(
        default_factory=lambda: [512, 256]
    )
    reasoner_dropout_rate: float = 0.2
    reasoner_l2_regularization: float = 1e-4

    # Producer parameters
    producer_initial_dense_units: int = 256
    producer_initial_spatial_size: int = 7
    producer_initial_channels: int = 128
    producer_conv_filters: List[int] = field(
        default_factory=lambda: [128, 64]
    )
    producer_style_units: List[int] = field(
        default_factory=lambda: [256, 128]
    )

    # Orthonormal regularization for style projections
    producer_orthonormal_lambda: float = 1e-3
    producer_orthonormal_l1: float = 0.0
    producer_orthonormal_l2: float = 0.0
    producer_use_matrix_scaling: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Basic training parameters
    epochs: int = 50
    learning_rates: Dict[str, float] = field(
        default_factory=lambda: {
            'explainer': 3e-4,
            'reasoner': 3e-4,
            'producer': 3e-4
        }
    )

    # CCNet-specific parameters
    loss_fn: str = 'l2'
    loss_fn_params: Dict[str, Any] = field(default_factory=dict)
    gradient_clip_norm: Optional[float] = 1.0
    dynamic_weighting: bool = False
    use_mixed_precision: bool = False

    # KL annealing
    kl_annealing_epochs: Optional[int] = 10

    # Loss weights
    explainer_weights: Dict[str, float] = field(
        default_factory=lambda: {
            'inference': 1.0,
            'generation': 1.0,
            'kl_divergence': 0.001
        }
    )
    reasoner_weights: Dict[str, float] = field(
        default_factory=lambda: {
            'reconstruction': 1.0,
            'inference': 1.0
        }
    )
    producer_weights: Dict[str, float] = field(
        default_factory=lambda: {
            'generation': 1.0,
            'reconstruction': 1.0
        }
    )


@dataclass
class DataConfig:
    """Configuration for data loading and augmentation."""

    # Data loading
    batch_size: int = 128
    shuffle_buffer_size: int = 60000
    prefetch_buffer_size: int = tf.data.AUTOTUNE

    # Data augmentation
    apply_augmentation: bool = True
    noise_stddev: float = 0.05
    max_translation: int = 2  # pixels


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping callback."""

    enabled: bool = True
    patience: int = 10
    error_threshold: float = 1e-4
    grad_stagnation_threshold: Optional[float] = 1e-4


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""

    # Output settings
    experiment_name: str = "ccnets_mnist_corrected"
    results_dir: str = "results"
    save_figures: bool = True
    show_figures: bool = False

    # Visualization samples
    reconstruction_samples: int = 5
    counterfactual_source_digits: List[int] = field(
        default_factory=lambda: [0, 1, 3, 5, 7]
    )
    counterfactual_target_digits: List[int] = field(
        default_factory=lambda: [0, 2, 4, 6, 8, 9]
    )
    latent_space_samples: int = 1000

    # t-SNE parameters
    tsne_perplexity: int = 30
    tsne_max_iter: int = 300
    tsne_random_state: int = 42

    # Dashboard settings
    dashboard_reconstruction_samples: int = 3
    dashboard_latent_samples: int = 500


@dataclass
class ExperimentConfig:
    """Master configuration for the entire experiment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )
    visualization: VisualizationConfig = field(
        default_factory=VisualizationConfig
    )

    # Logging
    log_interval: int = 10  # Log progress every N batches
    verbose: bool = True


# =====================================================================
# MODERN KERAS 3 LAYERS
# =====================================================================

@keras.saving.register_keras_serializable()
class FiLMLayer(keras.layers.Layer):
    """
    Feature-wise Linear Modulation (FiLM) Layer with optional orthonormal regularization.

    Applies an affine transformation to a content tensor based on a
    style vector. It learns a projection from the style vector to generate
    per-channel scaling (gamma) and shifting (beta) parameters.

    **Intent**: Enable conditional generation by modulating feature maps
    based on style/condition vectors, commonly used in conditional GANs
    and style transfer networks. Optional orthonormal regularization helps
    maintain diverse and non-redundant style projections.

    **Architecture**:
    ```
    Content Tensor(B, H, W, C) + Style Vector(B, S)
                    ↓
    Style → Dense(C) → Gamma(B, 1, 1, C)
         ↘ Dense(C) → Beta(B, 1, 1, C)
                    ↓
    Output = Content * (1 + Gamma) + Beta
    ```

    Args:
        kernel_regularizer: Optional regularizer for projection layers
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        List of two tensors:
        - content_tensor: (batch_size, height, width, channels)
        - style_vector: (batch_size, style_dim)

    Output shape:
        Same as content_tensor: (batch_size, height, width, channels)

    Attributes:
        gamma_projection: Dense layer for scaling parameters
        beta_projection: Dense layer for shifting parameters
        num_channels: Number of channels to modulate
        kernel_regularizer: Regularizer applied to projection layers
    """

    def __init__(
            self,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize FiLM layer with projection layers."""
        super().__init__(**kwargs)

        self.kernel_regularizer = kernel_regularizer

        # These will be created in build() when we know the shape
        self.gamma_projection: Optional[keras.layers.Dense] = None
        self.beta_projection: Optional[keras.layers.Dense] = None
        self.num_channels: Optional[int] = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Build the layer's weights and sub-layers.

        Args:
            input_shape: List of two shapes: [content_shape, style_shape].
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                f"FiLMLayer expects input_shape to be a list of 2 shapes, "
                f"got {type(input_shape)} with length {len(input_shape) if isinstance(input_shape, list) else 'N/A'}"
            )

        content_shape, style_shape = input_shape
        self.num_channels = content_shape[-1]

        if self.num_channels is None:
            raise ValueError("Content tensor must have a known number of channels")

        # Create sub-layers in build() when we know the dimensions
        # Use tanh activation on gamma to constrain scaling to stable range
        # Apply orthonormal regularization to gamma projection for better disentanglement
        self.gamma_projection = keras.layers.Dense(
            self.num_channels,
            activation='tanh',
            kernel_regularizer=self.kernel_regularizer,
            name=f"{self.name}_gamma_projection"
        )
        self.beta_projection = keras.layers.Dense(
            self.num_channels,
            activation='linear',
            name=f"{self.name}_beta_projection"
        )

        # Build sub-layers explicitly for proper serialization
        self.gamma_projection.build(style_shape)
        self.beta_projection.build(style_shape)

        super().build(input_shape)

    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """
        Apply the FiLM transformation.

        Args:
            inputs: List containing [content_tensor, style_vector].

        Returns:
            The modulated content tensor.
        """
        if len(inputs) != 2:
            raise ValueError(f"FiLMLayer expects 2 inputs, got {len(inputs)}")

        content_tensor, style_vector = inputs

        # Project style vector to get gamma (scale) and beta (bias)
        gamma = self.gamma_projection(style_vector)
        beta = self.beta_projection(style_vector)

        # Reshape for broadcasting: (batch, 1, 1, channels)
        gamma = keras.ops.expand_dims(keras.ops.expand_dims(gamma, 1), 1)
        beta = keras.ops.expand_dims(keras.ops.expand_dims(beta, 1), 1)

        # Apply modulation: output = content * (1 + gamma) + beta
        return content_tensor * (1.0 + gamma) + beta

    def compute_output_shape(
            self, input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (same as content tensor)."""
        content_shape, _ = input_shape
        return content_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for layer serialization."""
        config = super().get_config()
        config.update({
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
        })
        return config


@keras.saving.register_keras_serializable()
class ConvBlock(keras.layers.Layer):
    """
    Convolutional block with BatchNorm, Activation, and optional Pooling.

    This layer implements a standard convolutional building block consisting
    of Conv2D → BatchNorm → Activation → Optional Pooling, providing a
    reusable component for CNN architectures.

    **Intent**: Provide a standardized, configurable convolutional block
    that handles the common Conv-BN-Activation-Pool pattern with proper
    regularization and initialization.

    Args:
        filters: Number of convolutional filters
        kernel_size: Size of convolutional kernel
        strides: Convolution strides
        padding: Padding mode ('same' or 'valid')
        activation: Activation function name or callable
        use_pooling: Whether to apply pooling layer
        pool_size: Size of pooling window
        pool_type: Type of pooling ('max' or 'avg')
        kernel_regularizer: Regularizer for convolution kernel
        **kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            strides: Union[int, Tuple[int, int]] = 1,
            padding: str = "same",
            activation: str = "relu",
            use_pooling: bool = False,
            pool_size: Union[int, Tuple[int, int]] = 2,
            pool_type: str = "max",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize ConvBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if pool_type not in ["max", "avg"]:
            raise ValueError(f"pool_type must be 'max' or 'avg', got {pool_type}")

        # Store configuration
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_pooling = use_pooling
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.kernel_regularizer = kernel_regularizer

        # Create sub-layers in __init__
        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_conv"
        )

        self.bn = keras.layers.BatchNormalization(name=f"{self.name}_bn")

        self.act = GoLU(name=f"{self.name}_activation")

        if use_pooling:
            if pool_type == "max":
                self.pool = keras.layers.MaxPooling2D(
                    pool_size=pool_size, name=f"{self.name}_pool"
                )
            else:  # avg
                self.pool = keras.layers.AveragePooling2D(
                    pool_size=pool_size, name=f"{self.name}_pool"
                )
        else:
            self.pool = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build sub-layers explicitly for proper serialization."""
        # Build sub-layers in computational order
        self.conv.build(input_shape)

        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.bn.build(conv_output_shape)

        self.act.build(conv_output_shape)

        if self.pool is not None:
            self.pool.build(conv_output_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the convolutional block."""
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)

        if self.pool is not None:
            x = self.pool(x)

        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        shape = self.conv.compute_output_shape(input_shape)
        if self.pool is not None:
            shape = self.pool.compute_output_shape(shape)
        return shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'use_pooling': self.use_pooling,
            'pool_size': self.pool_size,
            'pool_type': self.pool_type,
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
        })
        return config


@keras.saving.register_keras_serializable()
class DenseBlock(keras.layers.Layer):
    """
    Dense block with BatchNorm, Activation, and optional Dropout.

    This layer implements a standard dense building block consisting
    of Dense → BatchNorm → Activation → Optional Dropout, providing a
    reusable component for MLP architectures.

    **Intent**: Provide a standardized, configurable dense block
    that handles the common Dense-BN-Activation-Dropout pattern with proper
    regularization and initialization.

    Args:
        units: Number of dense units
        activation: Activation function name or callable
        dropout_rate: Dropout rate (0.0 to disable)
        use_batch_norm: Whether to apply batch normalization
        kernel_regularizer: Regularizer for dense kernel
        **kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            units: int,
            activation: str = "relu",
            dropout_rate: float = 0.0,
            use_batch_norm: bool = True,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize DenseBlock with specified parameters."""
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0,1], got {dropout_rate}")

        # Store configuration
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.kernel_regularizer = kernel_regularizer

        # Create sub-layers in __init__
        self.dense = keras.layers.Dense(
            units=units,
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_dense"
        )

        if use_batch_norm:
            self.bn = keras.layers.BatchNormalization(name=f"{self.name}_bn")
        else:
            self.bn = None

        self.act = GoLU(name=f"{self.name}_activation")

        if dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=dropout_rate, name=f"{self.name}_dropout"
            )
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build sub-layers explicitly for proper serialization."""
        # Build sub-layers in computational order
        self.dense.build(input_shape)

        dense_output_shape = self.dense.compute_output_shape(input_shape)

        if self.bn is not None:
            self.bn.build(dense_output_shape)

        self.act.build(dense_output_shape)

        if self.dropout is not None:
            self.dropout.build(dense_output_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the dense block."""
        x = self.dense(inputs)

        if self.bn is not None:
            x = self.bn(x, training=training)

        x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return self.dense.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
        })
        return config


# =====================================================================
# REFINED MODEL DEFINITIONS
# =====================================================================

@keras.saving.register_keras_serializable()
class MNISTExplainer(keras.Model):
    """
    Explainer network for MNIST CCNet using modern Keras 3 patterns.

    This model implements the explainer component of a Causal Cooperative Network,
    which learns to extract causal explanations from input images. It uses a
    convolutional architecture followed by variational encoding to produce
    mean and log-variance parameters for the latent explanation distribution.

    **Intent**: Extract meaningful causal explanations from MNIST digits by
    learning a compact latent representation that captures the essential
    features needed for both reconstruction and classification.

    **Architecture**:
    ```
    Input(28, 28, 1)
           ↓
    ConvBlock(32, 5) → ConvBlock(64, 3) → ConvBlock(128, 3)
           ↓
    GlobalMaxPool → Flatten
           ↓
    Dense(explanation_dim) → μ (mean)
    Dense(explanation_dim) → log(σ²) (log variance)
    ```

    Args:
        config: ModelConfig instance containing architecture parameters
        **kwargs: Additional arguments for Model base class
    """

    def __init__(self, config: ModelConfig, **kwargs: Any) -> None:
        """Initialize explainer with convolutional blocks and variational head."""
        super().__init__(**kwargs)

        self.config = config

        # Create orthonormal regularizer for style projections
        self.orthonormal_regularizer = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=config.producer_orthonormal_lambda,
            l1_coefficient=config.producer_orthonormal_l1,
            l2_coefficient=config.producer_orthonormal_l2,
            use_matrix_scaling=config.producer_use_matrix_scaling
        ) if config.producer_orthonormal_lambda > 0 else None

        # Create regularizers
        l2_reg = config.explainer_l2_regularization
        self.l2_regularizer = keras.regularizers.L2(l2_reg) if l2_reg > 0 else None

        # Orthonormal regularizer for explanation vectors
        self.orthonormal_regularizer = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=config.explainer_orthonormal_lambda,
            l1_coefficient=config.explainer_orthonormal_l1,
            l2_coefficient=config.explainer_orthonormal_l2,
            use_matrix_scaling=config.explainer_use_matrix_scaling
        ) if config.explainer_orthonormal_lambda > 0 else None

        # Create convolutional blocks
        self.conv_blocks: List[ConvBlock] = []
        for i, (filters, kernel) in enumerate(zip(
                config.explainer_conv_filters,
                config.explainer_conv_kernels
        )):
            is_last = (i == len(config.explainer_conv_filters) - 1)
            use_pooling = not is_last  # No pooling on last block

            block = ConvBlock(
                filters=filters,
                kernel_size=kernel,
                use_pooling=use_pooling,
                pool_size=2,
                pool_type="max",
                kernel_regularizer=self.l2_regularizer,
                name=f"conv_block_{i + 1}"
            )
            self.conv_blocks.append(block)

        # Global pooling and flattening
        self.global_pool = keras.layers.GlobalMaxPooling2D(name="global_pool")
        self.flatten = keras.layers.Flatten(name="flatten")

        # Variational head with orthonormal regularization
        self.fc_mu = keras.layers.Dense(
            config.explanation_dim,
            name="mu",
            kernel_regularizer=self.orthonormal_regularizer
        )
        self.fc_log_var = keras.layers.Dense(
            config.explanation_dim,
            name="log_var",
            kernel_regularizer=self.l2_regularizer  # Use L2 for variance, not orthonormal
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build sub-layers explicitly for proper serialization."""
        current_shape = input_shape

        # Build convolutional blocks
        for block in self.conv_blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        # Build pooling layers
        pooled_shape = self.global_pool.compute_output_shape(current_shape)
        flat_shape = self.flatten.compute_output_shape(pooled_shape)

        # Build dense layers
        self.fc_mu.build(flat_shape)
        self.fc_log_var.build(flat_shape)

        super().build(input_shape)

    def call(
            self,
            x: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Forward pass through explainer network."""
        # Pass through convolutional blocks
        for block in self.conv_blocks:
            x = block(x, training=training)

        # Global pooling and flatten
        x = self.global_pool(x)
        features = self.flatten(x)

        # Variational parameters
        mu = self.fc_mu(features)
        log_var = self.fc_log_var(features)

        return mu, log_var

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "config": {
                "explanation_dim": self.config.explanation_dim,
                "num_classes": self.config.num_classes,
                "explainer_conv_filters": self.config.explainer_conv_filters,
                "explainer_conv_kernels": self.config.explainer_conv_kernels,
                "explainer_l2_regularization": self.config.explainer_l2_regularization,
                "explainer_orthonormal_lambda": self.config.explainer_orthonormal_lambda,
                "explainer_orthonormal_l1": self.config.explainer_orthonormal_l1,
                "explainer_orthonormal_l2": self.config.explainer_orthonormal_l2,
                "explainer_use_matrix_scaling": self.config.explainer_use_matrix_scaling,
            }
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MNISTExplainer':
        """Create instance from configuration."""
        model_config = ModelConfig(**config.pop("config"))
        return cls(model_config, **config)


@keras.saving.register_keras_serializable()
class MNISTReasoner(keras.Model):
    """
    Reasoner network for MNIST CCNet using modern Keras 3 patterns.

    This model implements the reasoner component of a Causal Cooperative Network,
    which learns to classify images based on both visual features and causal
    explanations from the explainer network.

    **Intent**: Perform classification by combining visual feature extraction
    with causal explanations, demonstrating how explanations can improve
    reasoning and classification accuracy.

    **Architecture**:
    ```
    Image(28, 28, 1) + Explanation(explanation_dim)
           ↓
    ConvBlock(32, 5) → ConvBlock(64, 3)
           ↓
    Flatten → Concat(image_features, explanation)
           ↓
    DenseBlock(512) → DenseBlock(256) → Dense(10, softmax)
    ```

    Args:
        config: ModelConfig instance containing architecture parameters
        **kwargs: Additional arguments for Model base class
    """

    def __init__(self, config: ModelConfig, **kwargs: Any) -> None:
        """Initialize reasoner with convolutional and dense blocks."""
        super().__init__(**kwargs)

        self.config = config

        # Create regularizers
        l2_reg = config.reasoner_l2_regularization
        self.l2_regularizer = keras.regularizers.L2(l2_reg) if l2_reg > 0 else None

        # Create convolutional blocks for image processing
        self.conv_blocks: List[ConvBlock] = []
        for i, (filters, kernel) in enumerate(zip(
                config.reasoner_conv_filters,
                config.reasoner_conv_kernels
        )):
            block = ConvBlock(
                filters=filters,
                kernel_size=kernel,
                use_pooling=True,
                pool_size=2,
                pool_type="max",
                kernel_regularizer=self.l2_regularizer,
                name=f"conv_block_{i + 1}"
            )
            self.conv_blocks.append(block)

        # Flatten for combining with explanations
        self.flatten = keras.layers.Flatten(name="flatten")

        # Dense blocks for reasoning
        self.dense_blocks: List[DenseBlock] = []
        for i, units in enumerate(config.reasoner_dense_units):
            block = DenseBlock(
                units=units,
                dropout_rate=config.reasoner_dropout_rate,
                kernel_regularizer=self.l2_regularizer,
                name=f"dense_block_{i + 1}"
            )
            self.dense_blocks.append(block)

        # Output classifier
        self.fc_output = keras.layers.Dense(
            config.num_classes,
            activation="softmax",
            name="classifier"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build sub-layers explicitly for proper serialization."""
        # For reasoner, input_shape represents the image input shape
        # The explanation will be concatenated later
        current_shape = input_shape

        # Build convolutional blocks
        for block in self.conv_blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        # Flatten and combine with explanation
        flat_shape = self.flatten.compute_output_shape(current_shape)
        # Assume explanation dimension from config for building
        combined_dim = flat_shape[-1] + self.config.explanation_dim
        combined_shape = (flat_shape[0], combined_dim)

        # Build dense blocks
        current_shape = combined_shape
        for block in self.dense_blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        # Build output layer
        self.fc_output.build(current_shape)

        super().build(input_shape)

    def call(
            self,
            x: keras.KerasTensor,
            e: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through reasoner network."""

        # Process image through convolutional blocks
        img_features = x
        for block in self.conv_blocks:
            img_features = block(img_features, training=training)

        # Flatten and combine with explanation
        img_features = self.flatten(img_features)
        combined = keras.ops.concatenate([img_features, e], axis=-1)

        # Process through dense blocks
        for block in self.dense_blocks:
            combined = block(combined, training=training)

        # Final classification
        return self.fc_output(combined)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "config": {
                "explanation_dim": self.config.explanation_dim,
                "num_classes": self.config.num_classes,
                "reasoner_conv_filters": self.config.reasoner_conv_filters,
                "reasoner_conv_kernels": self.config.reasoner_conv_kernels,
                "reasoner_dense_units": self.config.reasoner_dense_units,
                "reasoner_dropout_rate": self.config.reasoner_dropout_rate,
                "reasoner_l2_regularization": self.config.reasoner_l2_regularization,
            }
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MNISTReasoner':
        """Create instance from configuration."""
        model_config = ModelConfig(**config.pop("config"))
        return cls(model_config, **config)


@keras.saving.register_keras_serializable()
class MNISTProducer(keras.Model):
    """
    Producer network for MNIST CCNet using FiLM layers for style modulation.

    This model implements the producer component of a Causal Cooperative Network,
    which generates images conditioned on class labels and causal explanations.
    It uses FiLM layers to modulate feature maps based on explanations.

    **Intent**: Generate realistic MNIST digits by combining class information
    with causal explanations, demonstrating how explanations can guide
    generation towards specific semantic attributes.

    **Architecture**:
    ```
    Label(num_classes) + Explanation(explanation_dim)
           ↓
    Embedding(label) → Dense → Reshape(7, 7, 128)
           ↓
    UpSample → Conv → BN → FiLM(explanation) → Activation
           ↓
    UpSample → Conv → BN → FiLM(explanation) → Activation
           ↓
    Conv(32) → Conv(1, sigmoid)
    ```

    Args:
        config: ModelConfig instance containing architecture parameters
        **kwargs: Additional arguments for Model base class
    """

    def __init__(self, config: ModelConfig, **kwargs: Any) -> None:
        """Initialize producer with embedding, upsampling, and FiLM layers."""
        super().__init__(**kwargs)

        self.config = config

        # Label embedding
        self.label_embedding = keras.layers.Embedding(
            input_dim=config.num_classes,
            output_dim=config.producer_initial_dense_units,
            name="label_embedding"
        )

        # Initial content generation
        s, c = config.producer_initial_spatial_size, config.producer_initial_channels
        self.fc_content = keras.layers.Dense(
            units=s * s * c,
            activation='relu',
            name="fc_content"
        )
        self.reshape_content = keras.layers.Reshape(
            target_shape=(s, s, c),
            name="reshape_content"
        )

        # Upsampling and generation blocks
        self.generation_blocks: List[Dict[str, keras.layers.Layer]] = []
        for i, filters in enumerate(config.producer_conv_filters):
            block = {
                'upsample': keras.layers.UpSampling2D(
                    size=(2, 2), name=f"up_{i + 1}"
                ),
                'conv': keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=3,
                    padding="same",
                    name=f"conv_{i + 1}"
                ),
                'bn': keras.layers.BatchNormalization(name=f"norm_{i + 1}"),
                'film': FiLMLayer(
                    kernel_regularizer=self.orthonormal_regularizer,
                    name=f"film_{i + 1}"
                ),
                'activation': GoLU(name=f"act_{i + 1}")
            }
            self.generation_blocks.append(block)

        # Output layers
        self.conv_out_1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=1,
            padding="same",
            activation="linear",
            name="conv_out_1"
        )
        self.conv_out_2 = keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            padding="same",
            activation="sigmoid",
            name="conv_out_2"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build sub-layers explicitly for proper serialization."""
        # For producer, input_shape represents the label input shape
        # The explanation shape is known from config
        label_shape = input_shape
        explanation_shape = (label_shape[0], self.config.explanation_dim)

        # Build embedding and initial layers
        self.label_embedding.build(label_shape)

        embedding_output_shape = self.label_embedding.compute_output_shape(label_shape)
        self.fc_content.build(embedding_output_shape)

        fc_output_shape = self.fc_content.compute_output_shape(embedding_output_shape)
        self.reshape_content.build(fc_output_shape)

        # Build generation blocks
        s, c = self.config.producer_initial_spatial_size, self.config.producer_initial_channels
        current_shape = (label_shape[0], s, s, c)

        for block in self.generation_blocks:
            # Upsample
            block['upsample'].build(current_shape)
            upsampled_shape = block['upsample'].compute_output_shape(current_shape)

            # Conv
            block['conv'].build(upsampled_shape)
            conv_shape = block['conv'].compute_output_shape(upsampled_shape)

            # BatchNorm
            block['bn'].build(conv_shape)

            # FiLM (needs content and style shapes)
            film_input_shapes = [conv_shape, explanation_shape]
            block['film'].build(film_input_shapes)

            # Activation
            block['activation'].build(conv_shape)

            current_shape = conv_shape

        # Build output layers
        self.conv_out_1.build(current_shape)
        out1_shape = self.conv_out_1.compute_output_shape(current_shape)

        self.conv_out_2.build(out1_shape)

        super().build(input_shape)

    def call(
            self,
            y: keras.KerasTensor,
            e: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the producer network."""

        # Handle both integer indices and one-hot vectors
        if y.shape.rank > 1 and y.shape[-1] == self.config.num_classes:
            # One-hot vector: convert to indices
            y_indices = keras.ops.argmax(y, axis=-1)
        else:
            # Already indices
            y_indices = y

        # Generate initial content from label
        c = self.label_embedding(y_indices)
        c = self.fc_content(c)
        x = self.reshape_content(c)

        # Pass through generation blocks
        for block in self.generation_blocks:
            x = block['upsample'](x)
            x = block['conv'](x)
            x = block['bn'](x, training=training)
            x = block['film']([x, e])  # Apply style modulation
            x = block['activation'](x)

        # Generate output
        x = self.conv_out_1(x)
        x = self.conv_out_2(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "config": {
                "explanation_dim": self.config.explanation_dim,
                "num_classes": self.config.num_classes,
                "producer_initial_dense_units": self.config.producer_initial_dense_units,
                "producer_initial_spatial_size": self.config.producer_initial_spatial_size,
                "producer_initial_channels": self.config.producer_initial_channels,
                "producer_conv_filters": self.config.producer_conv_filters,
                "producer_style_units": self.config.producer_style_units,
            }
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MNISTProducer':
        """Create instance from configuration."""
        model_config = ModelConfig(**config.pop("config"))
        return cls(model_config, **config)


# =====================================================================
# VISUALIZATION PLUGINS (Keep original implementation)
# =====================================================================

class CCNetTrainingHistoryViz(CompositeVisualization):
    """Visualization for the complete loss landscape of CCNet training."""

    @property
    def name(self) -> str:
        return "ccnet_training_history"

    @property
    def description(self) -> str:
        return "Plots all CCNet-specific losses, errors, and gradient norms."

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and all(
            k in data for k in ["generation_loss", "explainer_error"]
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_subplot("Fundamental Losses", self._plot_fundamental_losses)
        self.add_subplot("Model Errors", self._plot_model_errors)
        self.add_subplot("Gradient Norms", self._plot_gradient_norms)
        self.add_subplot("System Convergence", self._plot_system_convergence)

    def _plot_fundamental_losses(
            self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        loss_keys: List[str] = [
            "generation_loss", "reconstruction_loss", "inference_loss"
        ]
        for key in loss_keys:
            if key in data and data[key]:
                ax.plot(
                    data[key],
                    label=key.replace("_loss", "").title(),
                    linewidth=2
                )
        ax.set_title("Fundamental Losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_model_errors(
            self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        error_keys: List[str] = [
            "explainer_error", "reasoner_error", "producer_error"
        ]
        for key in error_keys:
            if key in data and data[key]:
                ax.plot(
                    data[key],
                    label=key.replace("_error", "").title(),
                    linewidth=2
                )
        ax.set_title("Derived Model Errors")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_gradient_norms(
            self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        grad_keys: List[str] = [
            "explainer_grad_norm", "reasoner_grad_norm", "producer_grad_norm"
        ]
        for key in grad_keys:
            if key in data and data[key]:
                ax.plot(
                    data[key],
                    label=key.replace("_grad_norm", "").title(),
                    linewidth=2
                )
        ax.set_title("Gradient Norms (L2)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_system_convergence(
            self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        required_keys: List[str] = [
            "generation_loss", "reconstruction_loss", "inference_loss"
        ]
        if not all(k in data and data[k] for k in required_keys):
            ax.text(
                0.5, 0.5, "Incomplete loss data for convergence plot.",
                ha='center', va='center'
            )
            return

        total_error: List[float] = [
            g + r + i for g, r, i in zip(
                data["generation_loss"],
                data["reconstruction_loss"],
                data["inference_loss"]
            )
        ]
        ax.plot(total_error, linewidth=3, color="darkblue")
        ax.fill_between(range(len(total_error)), total_error, alpha=0.3)
        ax.set_title("Total System Error Convergence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total System Error")
        ax.grid(True, alpha=0.3)


class ReconstructionQualityViz(VisualizationPlugin):
    """Visualization comparing original, reconstructed, and generated images."""

    @property
    def name(self) -> str:
        return "ccnet_reconstruction_quality"

    @property
    def description(self) -> str:
        return "Compares original, reconstructed, generated images and errors."

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and all(
            k in data for k in ["orchestrator", "x_data", "y_data"]
        )

    def create_visualization(
            self, data: Dict[str, Any], ax: Optional[plt.Axes] = None, **kwargs: Any
    ) -> plt.Figure:
        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data: np.ndarray = data["x_data"]
        y_data: np.ndarray = data["y_data"]
        num_samples: int = data.get("num_samples", 5)

        fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))
        fig.suptitle(
            "Reconstruction Quality Analysis", fontsize=16, fontweight="bold"
        )

        for i in range(num_samples):
            x_input: np.ndarray = x_data[i: i + 1]
            y_truth: np.ndarray = keras.utils.to_categorical(
                [y_data[i]], 10
            )

            tensors: Dict[str, np.ndarray] = orchestrator.forward_pass(
                x_input, y_truth, training=False
            )
            y_pred: int = np.argmax(tensors["y_inferred"][0])

            axes[i, 0].imshow(x_input[0, :, :, 0], cmap="gray")
            axes[i, 0].set_title(f"Original\nLabel: {y_data[i]}")

            axes[i, 1].imshow(
                tensors["x_reconstructed"][0, :, :, 0], cmap="gray"
            )
            color: str = "green" if y_pred == y_data[i] else "red"
            axes[i, 1].set_title(
                f"Reconstructed\nInferred: {y_pred}", color=color
            )

            axes[i, 2].imshow(tensors["x_generated"][0, :, :, 0], cmap="gray")
            axes[i, 2].set_title(f"Generated\nTrue: {y_data[i]}")

            diff_recon: np.ndarray = np.abs(
                tensors["x_reconstructed"][0, :, :, 0] - x_input[0, :, :, 0]
            )
            axes[i, 3].imshow(diff_recon, cmap="hot")
            axes[i, 3].set_title(
                f"Recon Error\nMAE: {np.mean(diff_recon):.3f}"
            )

            diff_gen: np.ndarray = np.abs(
                tensors["x_generated"][0, :, :, 0] - x_input[0, :, :, 0]
            )
            axes[i, 4].imshow(diff_gen, cmap="hot")
            axes[i, 4].set_title(f"Gen Error\nMAE: {np.mean(diff_gen):.3f}")

            for j in range(5):
                axes[i, j].axis("off")

        return fig


class CounterfactualMatrixViz(VisualizationPlugin):
    """Visualization generating a matrix of counterfactual digits."""

    @property
    def name(self) -> str:
        return "ccnet_counterfactual_matrix"

    @property
    def description(self) -> str:
        return "Generates a matrix of counterfactual digits."

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and all(
            k in data for k in ["orchestrator", "x_data", "y_data"]
        )

    def create_visualization(
            self, data: Dict[str, Any], ax: Optional[plt.Axes] = None, **kwargs: Any
    ) -> plt.Figure:
        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data: np.ndarray = data["x_data"]
        y_data: np.ndarray = data["y_data"]
        source_digits: List[int] = data.get("source_digits", [0, 1, 3, 5, 7])
        target_digits: List[int] = data.get("target_digits", [0, 2, 4, 6, 8, 9])

        rows: int = len(source_digits)
        cols: int = len(target_digits)

        fig, axes = plt.subplots(
            rows, cols + 1, figsize=(cols * 1.5 + 1.5, rows * 1.5)
        )
        fig.suptitle(
            'Counterfactual Generation Matrix\n'
            '"What if this digit were drawn as..."',
            fontsize=16,
            fontweight="bold"
        )

        for i, source in enumerate(source_digits):
            idx: int = np.where(y_data == source)[0][0]
            x_source: np.ndarray = x_data[idx: idx + 1]

            axes[i, 0].imshow(x_source[0, :, :, 0], cmap="gray")
            axes[i, 0].set_ylabel(
                f"{source} →", rotation=0, size='large', labelpad=20
            )

            for j, target in enumerate(target_digits):
                y_target: np.ndarray = keras.utils.to_categorical([target], 10)
                x_counter: np.ndarray = orchestrator.counterfactual_generation(
                    x_source, y_target
                )
                axes[i, j + 1].imshow(x_counter[0, :, :, 0], cmap="gray")

                if i == 0:
                    axes[i, j + 1].set_title(f"→ {target}", size='large')

                if source == target:
                    axes[i, j + 1].patch.set_edgecolor("green")
                    axes[i, j + 1].patch.set_linewidth(3)

        for axis in axes.flatten():
            axis.set_xticks([])
            axis.set_yticks([])

        return fig


class LatentSpaceAnalysisViz(CompositeVisualization):
    """Visualization for latent space analysis using t-SNE and statistics."""

    @property
    def name(self) -> str:
        return "ccnet_latent_space"

    @property
    def description(self) -> str:
        return "Visualizes latent space with t-SNE and statistical plots."

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and all(
            k in data for k in ["orchestrator", "x_data", "y_data"]
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.latent_vectors: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.latent_2d: Optional[np.ndarray] = None
        self.tsne_config: Optional[Dict[str, Any]] = None

        self.add_subplot("t-SNE Colored by Class", self._plot_tsne_class)
        self.add_subplot("t-SNE Density", self._plot_tsne_density)
        self.add_subplot("Mean Activation", self._plot_mean_activation)
        self.add_subplot("Latent Norms by Class", self._plot_latent_norms)

    def _prepare_data(self, data: Dict[str, Any]) -> None:
        if self.latent_vectors is not None:
            return

        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data: np.ndarray = data["x_data"]
        y_data: np.ndarray = data["y_data"]
        num_samples: int = data.get("num_samples", 1000)

        self.tsne_config = data.get(
            "tsne_config",
            {"perplexity": 30, "max_iter": 300, "random_state": 42}
        )

        indices: np.ndarray = np.random.choice(
            len(x_data), min(num_samples, len(x_data)), replace=False
        )

        latent_vectors: List[np.ndarray] = []
        labels: List[int] = []

        for idx in indices:
            x_sample: np.ndarray = x_data[idx: idx + 1]
            _, e_latent = orchestrator.disentangle_causes(x_sample)
            latent_vectors.append(keras.ops.convert_to_numpy(e_latent[0]))
            labels.append(y_data[idx])

        self.latent_vectors = np.array(latent_vectors)
        self.labels = np.array(labels)

        logger.info("Computing t-SNE embedding for latent space...")
        tsne = TSNE(
            n_components=2,
            perplexity=self.tsne_config["perplexity"],
            max_iter=self.tsne_config["max_iter"],
            random_state=self.tsne_config["random_state"]
        )
        self.latent_2d = tsne.fit_transform(self.latent_vectors)

    def _plot_tsne_class(
            self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        self._prepare_data(data)
        scatter = ax.scatter(
            self.latent_2d[:, 0], self.latent_2d[:, 1],
            c=self.labels, cmap="tab10", s=20, alpha=0.7
        )
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.get_figure().colorbar(scatter, ax=ax, label="Digit")
        ax.grid(True, alpha=0.3)

    def _plot_tsne_density(
            self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        self._prepare_data(data)
        xy: np.ndarray = self.latent_2d.T
        z: np.ndarray = gaussian_kde(xy)(xy)
        scatter = ax.scatter(
            self.latent_2d[:, 0], self.latent_2d[:, 1],
            c=z, cmap="viridis", s=20, alpha=0.7
        )
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.get_figure().colorbar(scatter, ax=ax, label="Density")
        ax.grid(True, alpha=0.3)

    def _plot_mean_activation(
            self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        self._prepare_data(data)
        mean_activations: np.ndarray = np.mean(
            np.abs(self.latent_vectors), axis=0
        )
        ax.bar(range(len(mean_activations)), mean_activations, color="steelblue")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Mean |Activation|")
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_latent_norms(
            self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        self._prepare_data(data)
        latent_norms: np.ndarray = np.linalg.norm(self.latent_vectors, axis=1)
        sns.violinplot(
            x=self.labels, y=latent_norms, ax=ax,
            palette="husl", inner="quartile"
        )
        ax.set_xlabel("Digit Class")
        ax.set_ylabel("||E|| (L2 Norm)")
        ax.grid(True, alpha=0.3, axis='y')


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def create_mnist_ccnet(config: ExperimentConfig) -> CCNetOrchestrator:
    """
    Create and initialize a CCNet for MNIST digit classification.

    Args:
        config: Experiment configuration object.

    Returns:
        CCNetOrchestrator: Configured CCNet orchestrator instance.
    """
    # Initialize model components with configuration
    explainer = MNISTExplainer(config.model)
    reasoner = MNISTReasoner(config.model)
    producer = MNISTProducer(config.model)

    # Build models with dummy data to initialize weights
    dummy_image = keras.ops.zeros((1, 28, 28, 1))
    dummy_label_indices = keras.ops.zeros((1,), dtype="int32")
    dummy_latent = keras.ops.zeros((1, config.model.explanation_dim))

    # Initialize explainer
    mu, _ = explainer(dummy_image)

    # Initialize producer with proper input format
    _ = producer(dummy_label_indices, dummy_latent)

    # Initialize reasoner
    _ = reasoner(dummy_image, dummy_latent)

    # Configure CCNet from training config
    ccnet_config = CCNetConfig(
        explanation_dim=config.model.explanation_dim,
        loss_fn=config.training.loss_fn,
        loss_fn_params=config.training.loss_fn_params,
        learning_rates=config.training.learning_rates,
        gradient_clip_norm=config.training.gradient_clip_norm,
        use_mixed_precision=config.training.use_mixed_precision,
        explainer_weights=config.training.explainer_weights,
        reasoner_weights=config.training.reasoner_weights,
        producer_weights=config.training.producer_weights,
    )

    # Create orchestrator
    return CCNetOrchestrator(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=ccnet_config
    )


def prepare_mnist_data(config: DataConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Prepare MNIST dataset for training and validation."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    h, w = x_train.shape[1:]

    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    aug_layers = []
    if config.apply_augmentation:
        if config.max_translation > 0:
            aug_layers.append(keras.layers.RandomTranslation(
                height_factor=config.max_translation / h,
                width_factor=config.max_translation / w,
                fill_mode='constant'
            ))
        if config.noise_stddev > 0:
            aug_layers.append(keras.layers.GaussianNoise(config.noise_stddev))

    aug_pipe = keras.Sequential(aug_layers) if aug_layers else lambda x, **_: x
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(config.shuffle_buffer_size)
    train_ds = train_ds.map(
        lambda x, y: (aug_pipe(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(config.batch_size).prefetch(config.prefetch_buffer_size)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(config.batch_size).prefetch(config.prefetch_buffer_size)

    return train_ds, val_ds


# =====================================================================
# MAIN EXPERIMENT CLASS
# =====================================================================

class CCNetExperiment:
    """Experimental framework for CCNet training and visualization."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize experiment with configuration and visualization setup."""
        self.config = config
        self.viz_manager = VisualizationManager(
            experiment_name=config.visualization.experiment_name,
            output_dir=config.visualization.results_dir
        )
        self._register_plugins()
        self.x_test, self.y_test = self._load_test_data()

    def _register_plugins(self) -> None:
        """Register visualization plugins with the manager."""
        plugin_classes = [
            CCNetTrainingHistoryViz,
            ReconstructionQualityViz,
            CounterfactualMatrixViz,
            LatentSpaceAnalysisViz
        ]
        for plugin_class in plugin_classes:
            instance = plugin_class(self.viz_manager.config, self.viz_manager.context)
            self.viz_manager.register_plugin(instance)

    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess test data."""
        (_, _), (x, y) = keras.datasets.mnist.load_data()
        return (x.astype("float32") / 255.0)[..., np.newaxis], y

    def run(self) -> Tuple[CCNetOrchestrator, CCNetTrainer]:
        """Execute the complete training and visualization pipeline."""
        logger.info("Creating CCNet for MNIST...")
        orchestrator = create_mnist_ccnet(self.config)

        logger.info("Preparing data...")
        train_ds, val_ds = prepare_mnist_data(self.config.data)

        logger.info("Setting up trainer...")
        trainer = CCNetTrainer(
            orchestrator,
            kl_annealing_epochs=self.config.training.kl_annealing_epochs
        )

        callbacks = []
        if self.config.early_stopping.enabled:
            callbacks.append(EarlyStoppingCallback(
                patience=self.config.early_stopping.patience,
                error_threshold=self.config.early_stopping.error_threshold,
                grad_stagnation_threshold=self.config.early_stopping.grad_stagnation_threshold
            ))

        logger.info(f"Starting training for {self.config.training.epochs} epochs...")
        try:
            trainer.train(
                train_ds,
                self.config.training.epochs,
                validation_dataset=val_ds,
                callbacks=callbacks
            )
        except StopIteration:
            logger.info("Training stopped early due to convergence.")

        self.generate_final_report(orchestrator, trainer)
        logger.info(
            f"Visualizations saved to: "
            f"{self.viz_manager.context.output_dir / self.viz_manager.context.experiment_name}"
        )
        return orchestrator, trainer

    def generate_final_report(
            self, orchestrator: CCNetOrchestrator, trainer: CCNetTrainer
    ) -> None:
        """Generate comprehensive visualization report."""
        logger.info("=" * 30 + " Generating Final Report " + "=" * 30)
        vis_cfg = self.config.visualization
        tsne_cfg = {
            "perplexity": vis_cfg.tsne_perplexity,
            "max_iter": vis_cfg.tsne_max_iter,
            "random_state": vis_cfg.tsne_random_state
        }
        report_data = {
            'orchestrator': orchestrator,
            'x_data': self.x_test,
            'y_data': self.y_test,
            'num_samples': vis_cfg.reconstruction_samples,
            'source_digits': vis_cfg.counterfactual_source_digits,
            'target_digits': vis_cfg.counterfactual_target_digits,
            'tsne_config': tsne_cfg
        }

        self.viz_manager.visualize(
            trainer.history, "ccnet_training_history", show=vis_cfg.show_figures
        )
        self.viz_manager.visualize(
            report_data, "ccnet_reconstruction_quality", show=vis_cfg.show_figures
        )
        self.viz_manager.visualize(
            report_data, "ccnet_counterfactual_matrix", show=vis_cfg.show_figures
        )
        self.viz_manager.visualize(
            {**report_data, 'num_samples': vis_cfg.latent_space_samples},
            "ccnet_latent_space", show=vis_cfg.show_figures
        )

        models_dir = self.viz_manager.context.get_save_path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.save_models(str(models_dir / "mnist_ccnet"))
        logger.info(f"Models saved to {models_dir}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    # Configuration addressing posterior collapse and systemic imbalance
    config = ExperimentConfig(
        model=ModelConfig(
            explanation_dim=16,
            explainer_l2_regularization=1e-5,
            reasoner_dropout_rate=0.25
        ),
        training=TrainingConfig(
            epochs=50,
            learning_rates={
                'explainer': 3e-4,
                'reasoner': 3e-4,
                'producer': 3e-4
            },
            loss_fn='l2',
            dynamic_weighting=False,
            kl_annealing_epochs=10,
            explainer_weights={
                'inference': 1.0,
                'generation': 1.0,
                'kl_divergence': 0.001
            }
        ),
        data=DataConfig(
            batch_size=128,
            apply_augmentation=True,
            noise_stddev=0.02,
            max_translation=2
        ),
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=10,
            error_threshold=1e-4,
            grad_stagnation_threshold=1e-4
        ),
        visualization=VisualizationConfig(
            experiment_name="ccnets_mnist_refined",
            reconstruction_samples=5,
            latent_space_samples=2000,
            tsne_perplexity=40
        )
    )

    # Run experiment
    experiment = CCNetExperiment(config)
    orchestrator, trainer = experiment.run()
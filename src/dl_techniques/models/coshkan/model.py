"""
CoshKan (Complex Shearlet-Kolmogorov-Arnold Network) Implementation
================================================================

This module implements the CoshKan architecture, a hybrid neural network that combines
the multi-scale feature extraction capabilities of CoShNet with the learnable activation
functions of Kolmogorov-Arnold Networks for enhanced image classification.

Key Features:
------------
1. Hybrid Architecture:
   - Fixed shearlet transform frontend for multi-scale feature extraction
   - Complex-valued convolutional and dense layers for feature processing
   - KAN layers with learnable spline-based activations for final classification
   - Efficient parameter usage through fixed transform and learnable activations

2. Technical Advantages:
   - Multi-scale directional sensitivity from shearlet transforms
   - Complex-valued operations for better gradient flow and phase information
   - Learnable activation functions for improved expressiveness
   - Natural regularization through the combined architecture
   - Reduced parameter count compared to standard CNNs

3. Architecture Flow:
   Input → ShearletTransform → ComplexConv2D → ComplexDense → KANLinear → Softmax

Network Components:
-----------------
1. Frontend (CoShNet backbone):
   - ShearletTransform: Fixed multi-scale decomposition
   - ComplexConv2D: Complex-valued convolutions with pooling
   - ComplexDense: Complex-valued dense layers with dropout

2. Backend (KAN classifier):
   - Complex-to-Real conversion via magnitude operation
   - KANLinear layers with learnable B-spline activations
   - Final softmax classification

References:
----------
1. "CoShNet: A Hybrid Complex Valued Neural Network Using Shearlets"
2. "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
3. "Deep Complex Networks" (Trabelsi et al., 2018)
"""

import keras
from keras import layers, ops, initializers, regularizers
from typing import Optional, Tuple, List, Dict, Any, Sequence, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.kan_linear import KANLinear
from dl_techniques.layers.shearlet_transform import ShearletTransform
from dl_techniques.layers.complex_layers import ComplexDense, ComplexConv2D, ComplexReLU

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CoshKan(keras.Model):
    """
    Complex Shearlet-Kolmogorov-Arnold Network (CoshKan) implementation.

    CoshKan combines the multi-scale feature extraction of CoShNet with the learnable
    activation functions of KAN for enhanced image classification. The architecture
    uses complex-valued operations for better feature representation and KAN layers
    for improved classification expressiveness.

    **Architecture**:
    ```
    Input → ShearletTransform → ComplexConv2D + Pool → ComplexConv2D + Pool
          → Flatten → ComplexDense + Dropout → ComplexDense + Dropout
          → |·| (magnitude) → KANLinear → KANLinear → Dense(softmax)
    ```

    **Detailed Flow**:
    1. **Shearlet Transform**: Multi-scale directional decomposition
    2. **Complex Convolutions**: Feature extraction with complex-valued filters
    3. **Complex Dense**: High-level feature processing
    4. **Magnitude Conversion**: Complex → Real for KAN compatibility
    5. **KAN Classification**: Learnable activation functions for decision boundaries

    Args:
        # Input configuration
        input_shape: Shape of input images (height, width, channels). Default (32, 32, 3).
        num_classes: Number of output classes for classification. Default 10.

        # Architecture configuration
        conv_filters: List of filter counts for complex convolutional layers. Default [32, 64].
        dense_units: List of unit counts for complex dense layers. Default [512, 256].
        kan_units: List of unit counts for KAN layers. Default [128, 64].

        # Shearlet transform configuration
        shearlet_scales: Number of scales in shearlet transform. Default 4.
        shearlet_directions: Number of directions per scale. Default 8.

        # Layer configuration
        conv_kernel_size: Kernel size for convolutional layers. Default 5.
        conv_strides: Stride size for convolutional layers. Default 2.
        pool_size: Pooling size for average pooling layers. Default 2.

        # KAN configuration
        kan_grid_size: Grid size for KAN spline functions. Default 5.
        kan_spline_order: Spline order for KAN functions. Default 3.
        kan_activation: Base activation for KAN layers. Default 'swish'.

        # Regularization
        dropout_rate: Dropout rate for complex dense layers. Default 0.1.
        kernel_regularizer: Regularization for all layers. Default None.

        # Initialization
        kernel_initializer: Weight initialization method. Default 'glorot_uniform'.

        # Numerical stability
        epsilon: Small value for numerical stability. Default 1e-7.

        **kwargs: Additional arguments for Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        2D tensor with shape: `(batch_size, num_classes)` with softmax probabilities

    Example:
        ```python
        # Create model for CIFAR-10
        model = CoshKan(
            input_shape=(32, 32, 3),
            num_classes=10,
            conv_filters=[32, 64],
            dense_units=[512, 256],
            kan_units=[128, 64]
        )

        # Compile and build
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.build((None, 32, 32, 3))

        # Create compact variant
        compact_model = CoshKan(
            input_shape=(32, 32, 3),
            num_classes=10,
            conv_filters=[16, 32],
            dense_units=[256, 128],
            kan_units=[64, 32],
            kan_grid_size=3
        )
        ```
    """

    def __init__(
            self,
            # Input configuration
            input_shape: Tuple[int, int, int] = (32, 32, 3),
            num_classes: int = 10,
            # Architecture configuration
            conv_filters: Sequence[int] = (32, 64),
            dense_units: Sequence[int] = (512, 256),
            kan_units: Sequence[int] = (128, 64),
            # Shearlet transform configuration
            shearlet_scales: int = 4,
            shearlet_directions: int = 8,
            # Layer configuration
            conv_kernel_size: int = 5,
            conv_strides: int = 2,
            pool_size: int = 2,
            # KAN configuration
            kan_grid_size: int = 5,
            kan_spline_order: int = 3,
            kan_activation: str = 'swish',
            # Regularization
            dropout_rate: float = 0.1,
            kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            # Initialization
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            # Numerical stability
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(name="coshkan", **kwargs)

        # Store all configuration parameters
        self.input_shape_config = input_shape
        self.num_classes = num_classes
        self.conv_filters = list(conv_filters)
        self.dense_units = list(dense_units)
        self.kan_units = list(kan_units)
        self.shearlet_scales = shearlet_scales
        self.shearlet_directions = shearlet_directions
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.pool_size = pool_size
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_activation = kan_activation
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon

        # Handle regularizer serialization
        if isinstance(kernel_regularizer, str):
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self._kernel_regularizer_config = kernel_regularizer
        elif kernel_regularizer is None:
            self.kernel_regularizer = None
            self._kernel_regularizer_config = None
        else:
            self.kernel_regularizer = kernel_regularizer
            self._kernel_regularizer_config = regularizers.serialize(kernel_regularizer)

        # Handle initializer serialization
        if isinstance(kernel_initializer, str):
            self.kernel_initializer = initializers.get(kernel_initializer)
            self._kernel_initializer_config = kernel_initializer
        else:
            self.kernel_initializer = kernel_initializer
            self._kernel_initializer_config = initializers.serialize(kernel_initializer)

        # Validate configuration
        self._validate_config()

        # Create fixed shearlet transform layer
        self.shearlet = ShearletTransform(
            scales=self.shearlet_scales,
            directions=self.shearlet_directions,
            name='shearlet_transform'
        )

        # Create complex convolutional layers
        self.conv_layers: List[ComplexConv2D] = []
        self.pool_layers: List[layers.AveragePooling2D] = []

        for i, filters in enumerate(self.conv_filters):
            conv_layer = ComplexConv2D(
                filters=filters,
                kernel_size=self.conv_kernel_size,
                strides=self.conv_strides,
                padding="same",
                kernel_regularizer=self.kernel_regularizer,
                kernel_initializer=self.kernel_initializer,
                epsilon=self.epsilon,
                name=f'complex_conv_{i}'
            )
            self.conv_layers.append(conv_layer)

            pool_layer = layers.AveragePooling2D(
                pool_size=self.pool_size,
                name=f'avg_pool_{i}'
            )
            self.pool_layers.append(pool_layer)

        # Complex ReLU activation
        self.complex_activation = ComplexReLU(name='complex_relu')

        # Flatten layer
        self.flatten = layers.Flatten(name='flatten')

        # Create complex dense layers with dropout
        self.dense_layers: List[ComplexDense] = []
        self.dropout_layers: List[layers.Dropout] = []

        for i, units in enumerate(self.dense_units):
            dense_layer = ComplexDense(
                units=units,
                kernel_regularizer=self.kernel_regularizer,
                kernel_initializer=self.kernel_initializer,
                epsilon=self.epsilon,
                name=f'complex_dense_{i}'
            )
            self.dense_layers.append(dense_layer)

            dropout_layer = layers.Dropout(
                rate=self.dropout_rate,
                name=f'dropout_{i}'
            )
            self.dropout_layers.append(dropout_layer)

        # Create KAN layers
        self.kan_layers: List[KANLinear] = []

        for i, units in enumerate(self.kan_units):
            kan_layer = KANLinear(
                features=units,
                grid_size=self.kan_grid_size,
                spline_order=self.kan_spline_order,
                activation=self.kan_activation,
                name=f'kan_layer_{i}'
            )
            self.kan_layers.append(kan_layer)

        # Final classification layer
        self.classifier = layers.Dense(
            units=self.num_classes,
            activation="softmax",
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=self.kernel_initializer,
            name='classifier'
        )

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.info(f"CoshKan initialized with {sum(self.conv_filters)} conv filters, "
                    f"{sum(self.dense_units)} dense units, {sum(self.kan_units)} KAN units, "
                    f"{self.num_classes} classes")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Input shape validation
        if len(self.input_shape_config) != 3:
            raise ValueError(f"input_shape must be 3D (height, width, channels), got {self.input_shape_config}")
        if any(dim <= 0 for dim in self.input_shape_config):
            raise ValueError(f"All dimensions in input_shape must be positive, got {self.input_shape_config}")

        # Architecture validation
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if not self.conv_filters:
            raise ValueError("conv_filters cannot be empty")
        if any(f <= 0 for f in self.conv_filters):
            raise ValueError(f"All values in conv_filters must be positive, got {self.conv_filters}")
        if not self.dense_units:
            raise ValueError("dense_units cannot be empty")
        if any(u <= 0 for u in self.dense_units):
            raise ValueError(f"All values in dense_units must be positive, got {self.dense_units}")
        if not self.kan_units:
            raise ValueError("kan_units cannot be empty")
        if any(u <= 0 for u in self.kan_units):
            raise ValueError(f"All values in kan_units must be positive, got {self.kan_units}")

        # Shearlet validation
        if self.shearlet_scales <= 0:
            raise ValueError(f"shearlet_scales must be positive, got {self.shearlet_scales}")
        if self.shearlet_directions <= 0:
            raise ValueError(f"shearlet_directions must be positive, got {self.shearlet_directions}")

        # Layer configuration validation
        if self.conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {self.conv_kernel_size}")
        if self.conv_strides <= 0:
            raise ValueError(f"conv_strides must be positive, got {self.conv_strides}")
        if self.pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {self.pool_size}")

        # KAN validation
        if self.kan_grid_size < 3:
            raise ValueError(f"kan_grid_size must be >= 3, got {self.kan_grid_size}")
        if self.kan_spline_order <= 0:
            raise ValueError(f"kan_spline_order must be positive, got {self.kan_spline_order}")

        # Regularization validation
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")

        # Numerical validation
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the CoshKan model components."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Build shearlet transform
        self.shearlet.build(input_shape)

        # Compute shearlet output shape and track through layers
        current_shape = self.shearlet.compute_output_shape(input_shape)

        # Build conv and pool layers
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_layer.build(current_shape)
            conv_output_shape = conv_layer.compute_output_shape(current_shape)

            pool_layer.build(conv_output_shape)
            current_shape = pool_layer.compute_output_shape(conv_output_shape)

        # Build flatten
        self.flatten.build(current_shape)
        flatten_output_shape = self.flatten.compute_output_shape(current_shape)

        # Build complex dense and dropout layers
        current_shape = flatten_output_shape
        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            dense_layer.build(current_shape)
            dense_output_shape = dense_layer.compute_output_shape(current_shape)

            dropout_layer.build(dense_output_shape)
            current_shape = dense_output_shape

        # Build KAN layers - create them here when we know the input size
        # Convert complex shape to real shape (magnitude operation)
        kan_input_shape = (current_shape[0], current_shape[1])  # Remove complex dimension
        current_shape = kan_input_shape

        for i, units in enumerate(self.kan_units):
            kan_layer = KANLinear(
                features=units,
                grid_size=self.kan_grid_size,
                spline_order=self.kan_spline_order,
                activation=self.kan_activation,
                name=f'kan_layer_{i}'
            )
            self.kan_layers.append(kan_layer)
            kan_layer.build(current_shape)
            current_shape = kan_layer.compute_output_shape(current_shape)

        # Build classifier
        self.classifier.build(current_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the CoshKan network.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch_size, num_classes] with softmax probabilities
        """
        # Apply shearlet transform
        x = self.shearlet(inputs, training=training)

        # Convert to complex (real shearlet output → complex)
        x = ops.cast(x, 'complex64')

        # Complex convolutional layers with pooling
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            x = conv_layer(x, training=training)
            x = self.complex_activation(x, training=training)

        # Flatten
        x = self.flatten(x)

        # Complex dense layers with dropout
        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            x = dense_layer(x, training=training)
            x = self.complex_activation(x, training=training)
            if training:
                x = dropout_layer(x, training=training)

        # Convert complex to real for KAN (magnitude operation)
        x = ops.abs(x)

        # KAN layers with learnable activations
        for kan_layer in self.kan_layers:
            x = kan_layer(x, training=training)

        # Final classification
        x = self.classifier(x, training=training)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            # Input configuration
            'input_shape': self.input_shape_config,
            'num_classes': self.num_classes,
            # Architecture configuration
            'conv_filters': self.conv_filters,
            'dense_units': self.dense_units,
            'kan_units': self.kan_units,
            # Shearlet transform configuration
            'shearlet_scales': self.shearlet_scales,
            'shearlet_directions': self.shearlet_directions,
            # Layer configuration
            'conv_kernel_size': self.conv_kernel_size,
            'conv_strides': self.conv_strides,
            'pool_size': self.pool_size,
            # KAN configuration
            'kan_grid_size': self.kan_grid_size,
            'kan_spline_order': self.kan_spline_order,
            'kan_activation': self.kan_activation,
            # Regularization
            'dropout_rate': self.dropout_rate,
            'kernel_regularizer': self._kernel_regularizer_config,
            # Initialization
            'kernel_initializer': self._kernel_initializer_config,
            # Numerical stability
            'epsilon': self.epsilon,
        })
        return config


# ---------------------------------------------------------------------

def create_coshkan(
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        num_classes: int = 10,
        conv_filters: Sequence[int] = (32, 64),
        dense_units: Sequence[int] = (512, 256),
        kan_units: Sequence[int] = (128, 64),
        dropout_rate: float = 0.1,
        **kwargs: Any
) -> CoshKan:
    """
    Create a CoshKan model with specified configuration.

    Args:
        input_shape: Shape of input images (height, width, channels). Default (32, 32, 3).
        num_classes: Number of output classes. Default 10.
        conv_filters: Filter counts for complex convolutional layers. Default (32, 64).
        dense_units: Unit counts for complex dense layers. Default (512, 256).
        kan_units: Unit counts for KAN layers. Default (128, 64).
        dropout_rate: Dropout rate for regularization. Default 0.1.
        **kwargs: Additional configuration parameters.

    Returns:
        Configured CoshKan model

    Example:
        ```python
        # Create standard CoshKan for CIFAR-10
        model = create_coshkan(
            input_shape=(32, 32, 3),
            num_classes=10,
            conv_filters=(32, 64),
            dense_units=(512, 256),
            kan_units=(128, 64)
        )

        # Build and compile
        model.build((None, 32, 32, 3))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """
    logger.info("Creating CoshKan model with hybrid CoShNet-KAN architecture")
    return CoshKan(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_filters=conv_filters,
        dense_units=dense_units,
        kan_units=kan_units,
        dropout_rate=dropout_rate,
        **kwargs
    )


# ---------------------------------------------------------------------

def create_coshkan_variant(variant: str = "base") -> CoshKan:
    """
    Create predefined CoshKan model variants for different use cases.

    Args:
        variant: Model variant string. Options:
            - "micro": Ultra-compact model (minimal parameters)
            - "small": Compact model for resource-constrained environments
            - "base": Standard model for general classification tasks
            - "large": Larger model for complex datasets
            - "cifar10": Optimized for CIFAR-10 classification
            - "imagenet": Scaled for ImageNet-style inputs

    Returns:
        Configured CoshKan model optimized for the specified use case

    Example:
        ```python
        # Create micro variant for edge deployment
        micro_model = create_coshkan_variant("micro")

        # Create base variant for general use
        base_model = create_coshkan_variant("base")

        # Create ImageNet variant
        imagenet_model = create_coshkan_variant("imagenet")
        ```
    """
    variant_configs = {
        "micro": {
            "input_shape": (32, 32, 3),
            "num_classes": 10,
            "conv_filters": (8, 16),
            "dense_units": (64, 32),
            "kan_units": (16,),
            "shearlet_scales": 2,
            "shearlet_directions": 4,
            "kan_grid_size": 3,
            "dropout_rate": 0.2,
            "conv_kernel_size": 3,
            "pool_size": 2,
        },
        "small": {
            "input_shape": (32, 32, 3),
            "num_classes": 10,
            "conv_filters": (16, 32),
            "dense_units": (128, 64),
            "kan_units": (32, 16),
            "shearlet_scales": 3,
            "shearlet_directions": 6,
            "kan_grid_size": 3,
            "dropout_rate": 0.15,
            "conv_kernel_size": 3,
            "pool_size": 2,
        },
        "base": {
            "input_shape": (32, 32, 3),
            "num_classes": 10,
            "conv_filters": (32, 64),
            "dense_units": (512, 256),
            "kan_units": (128, 64),
            "shearlet_scales": 4,
            "shearlet_directions": 8,
            "kan_grid_size": 5,
            "dropout_rate": 0.1,
            "conv_kernel_size": 5,
            "pool_size": 2,
        },
        "large": {
            "input_shape": (32, 32, 3),
            "num_classes": 10,
            "conv_filters": (64, 128, 256),
            "dense_units": (1024, 512, 256),
            "kan_units": (256, 128, 64),
            "shearlet_scales": 5,
            "shearlet_directions": 12,
            "kan_grid_size": 7,
            "kan_spline_order": 4,
            "dropout_rate": 0.15,
            "conv_kernel_size": 5,
            "pool_size": 2,
        },
        "cifar10": {
            "input_shape": (32, 32, 3),
            "num_classes": 10,
            "conv_filters": (32, 64),
            "dense_units": (512, 256),
            "kan_units": (128, 64),
            "shearlet_scales": 4,
            "shearlet_directions": 8,
            "kan_grid_size": 5,
            "dropout_rate": 0.1,
            "conv_kernel_size": 5,
            "pool_size": 2,
        },
        "imagenet": {
            "input_shape": (224, 224, 3),
            "num_classes": 1000,
            "conv_filters": (64, 128, 256),
            "dense_units": (2048, 1024, 512),
            "kan_units": (512, 256, 128),
            "shearlet_scales": 5,
            "shearlet_directions": 16,
            "kan_grid_size": 8,
            "kan_spline_order": 4,
            "dropout_rate": 0.2,
            "conv_kernel_size": 7,
            "conv_strides": 2,
            "pool_size": 3,
        },
    }

    if variant not in variant_configs:
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {list(variant_configs.keys())}"
        )

    config_dict = variant_configs[variant]
    logger.info(f"Creating CoshKan {variant} variant")

    return create_coshkan(**config_dict)

# ---------------------------------------------------------------------
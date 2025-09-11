"""
CoShNet (Complex Shearlet Network) Implementation
================================================================

This module implements the CoShNet architecture, a hybrid complex-valued neural network
that combines fixed shearlet transforms with learnable complex-valued layers for
efficient image classification.

Key Features:
------------
1. Hybrid Architecture:
   - Fixed shearlet transform frontend for multi-scale feature extraction
   - Learnable complex-valued convolutional and dense layers
   - Global Average Pooling for spatial dimension reduction
   - Efficient parameter usage through fixed transform
   - Built-in multi-scale and directional sensitivity

2. Technical Advantages:
   - Fewer parameters than traditional CNNs
   - Faster training convergence
   - Better gradient flow through complex-valued operations
   - Natural handling of phase information
   - Self-regularizing behavior

3. Implementation Details:
   - Complex-valued operations with split real/imaginary implementation
   - Numerically stable complex arithmetic
   - Proper complex weight initialization
   - Efficient memory usage through global pooling
   - Modern Keras 3 patterns for serialization and configuration

Network Architecture:
-------------------
1. Input Layer:
   - ShearletTransform: Fixed transform for multi-scale decomposition
   - Scales: 4 (default)
   - Directions: 8 per scale (default)

2. Learnable Layers:
   - Complex Convolution: 2 layers (32 and 64 filters)
   - Complex Dense: 2 layers (1250 and 500 units)
   - Average Pooling: After each convolution
   - Global Average Pooling: Before dense layers
   - Complex ReLU: Non-linear activation
   - Final Real Classification Layer

Performance Characteristics:
-------------------------
1. Model Size:
   - Base Model: ~1.3M parameters (reduced with global pooling)
   - Tiny Model: ~49.9k parameters
   - Memory Efficient: Global pooling reduces spatial dimensions

2. Computational Efficiency:
   - Reduced FLOPs compared to traditional CNNs
   - Fast convergence in 20 epochs
   - Efficient forward pass through fixed transform

References:
----------
1. "CoShNet: A Hybrid Complex Valued Neural Network Using Shearlets"
2. "Deep Complex Networks" (Trabelsi et al., 2018)
3. "CoShRem: Faithful Digital Shearlet Transforms based on Compactly Supported Shearlets"
"""

import keras
from keras import layers, ops, initializers, regularizers
from typing import Optional, Tuple, List, Dict, Any, Sequence, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.shearlet_transform import ShearletTransform
from dl_techniques.layers.complex_layers import (
    ComplexDense,
    ComplexConv2D,
    ComplexReLU,
    ComplexDropout,
    ComplexAveragePooling2D,
    ComplexGlobalAveragePooling2D
)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CoShNet(keras.Model):
    """
    Refined Complex Shearlet Network (CoShNet) implementation.

    CoShNet combines fixed ShearletTransform with complex-valued layers for efficient
    image classification with built-in multi-scale and directional sensitivity.
    This refined version uses global average pooling for better parameter efficiency.

    **Architecture**:
    ```
    Input → ShearletTransform → Complex Conv Layers → Global Avg Pool → Complex Dense Layers → Classification

    Detailed Flow:
    Image → Shearlet(scales, directions) → ComplexConv2D + Pool → ComplexConv2D + Pool
          → GlobalAvgPool → ComplexDense + Dropout → ComplexDense + Dropout → Dense(softmax)
    ```

    Args:
        # Input configuration
        input_shape: Shape of input images (height, width, channels). Default (32, 32, 3).
        num_classes: Number of output classes for classification. Default 10.

        # Architecture configuration
        conv_filters: List of filter counts for convolutional layers. Default [32, 64].
        dense_units: List of unit counts for dense layers. Default [1250, 500].

        # Shearlet transform configuration
        shearlet_scales: Number of scales in shearlet transform. Default 4.
        shearlet_directions: Number of directions per scale in shearlet transform. Default 8.

        # Layer configuration
        conv_kernel_size: Kernel size for convolutional layers. Default 5.
        conv_strides: Stride size for convolutional layers. Default 2.
        pool_size: Pooling size for average pooling layers. Default 2.

        # Regularization
        dropout_rate: Dropout rate for regularization. Default 0.1.
        kernel_regularizer: Regularization to apply to kernel weights. Default None.

        # Initialization
        kernel_initializer: Initialization method for kernel weights. Default 'glorot_uniform'.

        # Numerical stability
        epsilon: Small value for numerical stability in complex operations. Default 1e-7.

        **kwargs: Additional arguments for Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        2D tensor with shape: `(batch_size, num_classes)` with softmax probabilities

    Example:
        ```python
        # Create model for CIFAR-10
        model = CoShNet(
            input_shape=(32, 32, 3),
            num_classes=10,
            conv_filters=[32, 64],
            dense_units=[1250, 500]
        )

        # Compile and train
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Build model
        model.build((None, 32, 32, 3))
        print(f"Total parameters: {model.count_params():,}")
        ```
    """

    def __init__(
            self,
            # Input configuration
            input_shape: Tuple[int, int, int] = (32, 32, 3),
            num_classes: int = 10,
            # Architecture configuration
            conv_filters: Sequence[int] = (32, 64),
            dense_units: Sequence[int] = (1250, 500),
            # Shearlet transform configuration
            shearlet_scales: int = 4,
            shearlet_directions: int = 8,
            # Layer configuration
            conv_kernel_size: int = 5,
            conv_strides: int = 2,
            # Regularization
            dropout_rate: float = 0.1,
            kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            # Initialization
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            # Numerical stability
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(name="coshnet", **kwargs)

        # Store all configuration parameters
        self.input_shape_config = input_shape
        self.num_classes = num_classes
        self.conv_filters = list(conv_filters)
        self.dense_units = list(dense_units)
        self.shearlet_scales = shearlet_scales
        self.shearlet_directions = shearlet_directions
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
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

        # Complex ReLU activation
        self.activation = ComplexReLU(name='complex_relu')

        # Global Average Pooling layer (replaces flatten)
        self.global_avg_pool = ComplexGlobalAveragePooling2D(
            keepdims=False,
            name='global_avg_pool'
        )

        # Create complex dense layers with dropout
        self.dense_layers: List[ComplexDense] = []
        self.dropout_layers: List[ComplexDropout] = []

        for i, units in enumerate(self.dense_units):
            dense_layer = ComplexDense(
                units=units,
                kernel_regularizer=self.kernel_regularizer,
                kernel_initializer=self.kernel_initializer,
                epsilon=self.epsilon,
                name=f'complex_dense_{i}'
            )
            self.dense_layers.append(dense_layer)

            dropout_layer = ComplexDropout(
                rate=self.dropout_rate,
                name=f'dropout_{i}'
            )
            self.dropout_layers.append(dropout_layer)

        # Final classification layer
        self.classifier = layers.Dense(
            units=self.num_classes,
            activation="softmax",
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=self.kernel_initializer,
            name='classifier'
        )

        logger.info(f"Refined CoShNet initialized with {sum(self.conv_filters)} conv filters, "
                    f"{sum(self.dense_units)} dense units, {self.num_classes} classes")
        logger.info("Using global average pooling for spatial dimension reduction")

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

        # Regularization validation
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")

        # Numerical validation
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the refined CoShNet model components."""
        if self.built:
            return

        # Build shearlet transform
        self.shearlet.build(input_shape)

        # Compute shearlet output shape
        shearlet_output_shape = self.shearlet.compute_output_shape(input_shape)
        current_shape = shearlet_output_shape

        # Build conv a layers
        for conv_layer in self.conv_layers:
            conv_layer.build(current_shape)
            current_shape = conv_layer.compute_output_shape(current_shape)

        # Build global average pooling
        self.global_avg_pool.build(current_shape)
        global_pool_output_shape = self.global_avg_pool.compute_output_shape(current_shape)

        # Build dense and dropout layers
        current_shape = global_pool_output_shape
        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            dense_layer.build(current_shape)
            dense_output_shape = dense_layer.compute_output_shape(current_shape)

            dropout_layer.build(dense_output_shape)
            current_shape = dense_output_shape

        # Build classifier - expects real-valued input (magnitude of complex)
        classifier_input_shape = (current_shape[0], current_shape[1])
        self.classifier.build(classifier_input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the refined network.

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

        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, training=training)
            x = self.activation(x, training=training)

        # Global average pooling (replaces flatten)
        x = self.global_avg_pool(x)

        # Dense layers with dropout
        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            x = dense_layer(x, training=training)
            x = self.activation(x, training=training)
            x = dropout_layer(x, training=training)

        # Final classification (convert to real by taking magnitude)
        x = ops.abs(x)
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
            # Shearlet transform configuration
            'shearlet_scales': self.shearlet_scales,
            'shearlet_directions': self.shearlet_directions,
            # Layer configuration
            'conv_kernel_size': self.conv_kernel_size,
            'conv_strides': self.conv_strides,
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
# Factory Functions
# ---------------------------------------------------------------------

def create_coshnet(
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        num_classes: int = 10,
        conv_filters: Sequence[int] = (32, 64),
        dense_units: Sequence[int] = (1250, 500),
        dropout_rate: float = 0.1,
        **kwargs: Any
) -> CoShNet:
    """
    Create a refined CoShNet model with global average pooling.

    Args:
        input_shape: Shape of input images (height, width, channels). Default (32, 32, 3).
        num_classes: Number of output classes. Default 10.
        conv_filters: Filter counts for convolutional layers. Default (32, 64).
        dense_units: Unit counts for dense layers. Default (1250, 500).
        dropout_rate: Dropout rate for regularization. Default 0.1.
        **kwargs: Additional configuration parameters.

    Returns:
        Configured refined CoShNet model

    Example:
        ```python
        # Create standard refined CoShNet for CIFAR-10
        model = create_coshnet(
            input_shape=(32, 32, 3),
            num_classes=10,
            conv_filters=(32, 64),
            dense_units=(1250, 500)
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
    logger.info("Creating refined CoShNet model with global average pooling")
    return CoShNet(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_filters=conv_filters,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        **kwargs
    )


def create_coshnet_variant(variant: str = "base") -> CoShNet:
    """
    Create predefined refined CoShNet model variants for different use cases.

    Args:
        variant: Model variant string. Options:
            - "tiny": Minimal model (~20k parameters)
            - "base": Standard model (~800k parameters)
            - "large": Larger model for complex datasets
            - "cifar10": Optimized for CIFAR-10
            - "imagenet": Scaled for ImageNet-style inputs

    Returns:
        Configured CoShNet model optimized for the specified use case

    Example:
        ```python
        # Create tiny variant for resource-constrained environments
        tiny_model = create_coshnet_variant("tiny")

        # Create base variant for general use
        base_model = create_coshnet_variant("base")

        # Create ImageNet variant
        imagenet_model = create_coshnet_variant("imagenet")
        ```
    """
    variant_configs = {
        "tiny": {
            "input_shape": (32, 32, 3),
            "num_classes": 10,
            "conv_filters": (16, 32),
            "dense_units": (256, 128),
            "shearlet_scales": 3,
            "shearlet_directions": 6,
            "dropout_rate": 0.2,
            "conv_kernel_size": 3,
        },
        "base": {
            "input_shape": (32, 32, 3),
            "num_classes": 10,
            "conv_filters": (32, 64),
            "dense_units": (1250, 500),
            "shearlet_scales": 4,
            "shearlet_directions": 8,
            "dropout_rate": 0.1,
            "conv_kernel_size": 5,
        },
        "large": {
            "input_shape": (32, 32, 3),
            "num_classes": 10,
            "conv_filters": (64, 128, 256),
            "dense_units": (2048, 1024, 512),
            "shearlet_scales": 5,
            "shearlet_directions": 12,
            "dropout_rate": 0.15,
            "conv_kernel_size": 5,
        },
        "cifar10": {
            "input_shape": (32, 32, 3),
            "num_classes": 10,
            "conv_filters": (32, 64),
            "dense_units": (800, 400),  # Reduced due to global pooling
            "shearlet_scales": 4,
            "shearlet_directions": 8,
            "dropout_rate": 0.1,
            "conv_kernel_size": 5,
        },
        "imagenet": {
            "input_shape": (224, 224, 3),
            "num_classes": 1000,
            "conv_filters": (64, 128, 256),
            "dense_units": (2048, 1024),  # Reduced due to global pooling
            "shearlet_scales": 5,
            "shearlet_directions": 16,
            "dropout_rate": 0.2,
            "conv_kernel_size": 7,
            "conv_strides": 2,
        },
    }

    if variant not in variant_configs:
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {list(variant_configs.keys())}"
        )

    config_dict = variant_configs[variant]
    logger.info(f"Creating refined CoShNet {variant} variant with global pooling")

    return create_coshnet(**config_dict)
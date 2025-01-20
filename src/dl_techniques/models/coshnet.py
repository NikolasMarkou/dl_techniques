"""
CoShNet (Complex Shearlet Network) Implementation
===============================================

This module implements the CoShNet architecture, a hybrid complex-valued neural network
that combines fixed shearlet transforms with learnable complex-valued layers for
efficient image classification.

Key Features:
------------
1. Hybrid Architecture:
   - Fixed shearlet transform frontend for multi-scale feature extraction
   - Learnable complex-valued convolutional and dense layers
   - Efficient parameter usage through fixed transform
   - Built-in multi-scale and directional sensitivity

2. Technical Advantages:
   - Fewer parameters than traditional CNNs (49.9k vs 11.18M for ResNet-18)
   - Faster training convergence (20 epochs vs 200 for standard CNNs)
   - Better gradient flow through complex-valued operations
   - Natural handling of phase information
   - Self-regularizing behavior

3. Implementation Details:
   - Complex-valued operations with split real/imaginary implementation
   - Numerically stable complex arithmetic
   - Proper complex weight initialization
   - Efficient memory usage through in-place operations
   - Configurable architecture through dataclass configuration

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
   - Complex ReLU: Non-linear activation
   - Final Real Classification Layer

Performance Characteristics:
-------------------------
1. Model Size:
   - Base Model: 1.3M parameters
   - Tiny Model: 49.9k parameters
   - Memory Efficient: No batch norm, minimal activation storage

2. Computational Efficiency:
   - 52× fewer FLOPs than ResNet-18
   - 93.06 MFLOPs vs 4.77 GFLOPs
   - Fast convergence in 20 epochs
   - Efficient forward pass through fixed transform

References:
----------
1. "CoShNet: A Hybrid Complex Valued Neural Network Using Shearlets"
2. "Deep Complex Networks" (Trabelsi et al., 2018)
3. "CoShRem: Faithful Digital Shearlet Transforms based on Compactly Supported Shearlets"
"""


import tensorflow as tf
from keras import Model
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict, Any
from keras.api.layers import Layer, Dense, AveragePooling2D

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.shearlet_transform import ShearletTransform
from dl_techniques.layers.complex_layers import \
    ComplexDense, ComplexConv2D, ComplexReLU


# ---------------------------------------------------------------------


@dataclass
class CoShNetConfig:
    """Configuration for CoShNet model.

    Attributes:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        conv_filters: List of filter counts for conv layers
        dense_units: List of unit counts for dense layers
        shearlet_scales: Number of scales in shearlet transform
        shearlet_directions: Number of directions per scale
        dropout_rate: Dropout rate for regularization
        kernel_regularizer: Optional kernel regularizer
        kernel_initializer: Kernel initializer for layers
        epsilon: Small value for numerical stability
    """
    input_shape: Tuple[int, int, int]
    num_classes: int
    conv_filters: List[int] = (32, 64)
    dense_units: List[int] = (1250, 500)
    shearlet_scales: int = 4
    shearlet_directions: int = 8
    dropout_rate: float = 0.1
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
    kernel_initializer: tf.keras.initializers.Initializer = tf.keras.initializers.GlorotUniform()
    epsilon: float = 1e-7


# ---------------------------------------------------------------------


class CoShNet(Model):
    """Complex Shearlet Network (CoShNet) implementation.

    Combines ShearletTransform with complex-valued layers for efficient
    image classification with built-in multi-scale and directional sensitivity.
    """

    def __init__(self, config: CoShNetConfig) -> None:
        """Initialize CoShNet model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Create layers
        self.shearlet = ShearletTransform(
            scales=config.shearlet_scales,
            directions=config.shearlet_directions,
            kernel_regularizer=config.kernel_regularizer
        )

        # Complex convolutional layers
        self.conv_layers = []
        for filters in config.conv_filters:
            self.conv_layers.append(
                ComplexConv2D(
                    filters=filters,
                    kernel_size=5,
                    strides=2,
                    kernel_regularizer=config.kernel_regularizer,
                    kernel_initializer=config.kernel_initializer
                )
            )

        self.pool = AveragePooling2D(2)
        self.activation = ComplexReLU()

        # Complex dense layers
        self.dense_layers = []
        for units in config.dense_units:
            self.dense_layers.append(
                ComplexDense(
                    units=units,
                    kernel_regularizer=config.kernel_regularizer,
                    kernel_initializer=config.kernel_initializer
                )
            )

        # Final classification layer
        self.classifier = Dense(
            config.num_classes,
            kernel_regularizer=config.kernel_regularizer,
            kernel_initializer=config.kernel_initializer
        )

        # Build model
        self.build((None, *config.input_shape))

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the network.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        # Apply shearlet transform
        x = self.shearlet(inputs)

        # Convert to complex
        x = tf.cast(x, tf.complex64)

        # Convolutional layers
        for conv in self.conv_layers:
            x = self.activation(conv(x))
            x = self.pool(x)

        # Flatten
        x = tf.reshape(x, (tf.shape(x)[0], -1))

        # Dense layers
        for dense in self.dense_layers:
            x = self.activation(dense(x))

        # Final classification
        x = tf.abs(x)  # Convert to real
        return self.classifier(x)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'config': self.config
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CoShNet':
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            CoShNet model instance
        """
        return cls(**config)


# ---------------------------------------------------------------------

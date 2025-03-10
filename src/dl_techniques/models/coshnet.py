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

import keras
import tensorflow as tf
from keras.api import Model
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any, Sequence
from keras.api.layers import Dense, AveragePooling2D, Dropout, Flatten

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.shearlet_transform import ShearletTransform
from dl_techniques.layers.complex_layers import (
    ComplexDense,
    ComplexConv2D,
    ComplexReLU
)


# ---------------------------------------------------------------------


@dataclass
class CoShNetConfig:
    """Configuration for CoShNet model.

    This dataclass provides a structured way to configure the CoShNet architecture,
    allowing for easy experimentation with different network designs.

    Attributes:
        input_shape (Tuple[int, int, int]): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes for classification
        conv_filters (Sequence[int]): List of filter counts for convolutional layers
        dense_units (Sequence[int]): List of unit counts for dense layers
        shearlet_scales (int): Number of scales in shearlet transform
        shearlet_directions (int): Number of directions per scale
        dropout_rate (float): Dropout rate for regularization
        kernel_regularizer (Optional[keras.regularizers.Regularizer]):
            Regularization to apply to kernel weights
        kernel_initializer (keras.initializers.Initializer):
            Initialization method for kernel weights
        conv_kernel_size (int): Kernel size for convolutional layers
        conv_strides (int): Stride size for convolutional layers
        pool_size (int): Pooling size for average pooling layers
        epsilon (float): Small value for numerical stability
    """
    input_shape: Tuple[int, int, int]
    num_classes: int
    conv_filters: Sequence[int] = field(default_factory=lambda: [32, 64])
    dense_units: Sequence[int] = field(default_factory=lambda: [1250, 500])
    shearlet_scales: int = 4
    shearlet_directions: int = 8
    dropout_rate: float = 0.1
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None
    kernel_initializer: keras.initializers.Initializer = keras.initializers.GlorotUniform()
    conv_kernel_size: int = 5
    conv_strides: int = 2
    pool_size: int = 2
    epsilon: float = 1e-7

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return asdict(self)


class CoShNet(Model):
    """Complex Shearlet Network (CoShNet) implementation.

    CoShNet combines fixed ShearletTransform with complex-valued layers for efficient
    image classification with built-in multi-scale and directional sensitivity.

    The architecture uses fewer parameters than traditional CNNs while maintaining
    competitive performance through its hybrid design of fixed transforms and
    learnable complex-valued neural network layers.
    """

    def __init__(self, config: CoShNetConfig) -> None:
        """Initialize CoShNet model.

        Args:
            config (CoShNetConfig): Model configuration parameters
        """
        super().__init__(name="coshnet")
        self.config = config

        # Initialize layer lists
        self.conv_layers: List[ComplexConv2D] = []
        self.pool_layers: List[AveragePooling2D] = []
        self.dense_layers: List[ComplexDense] = []
        self.dropout_layers: List[Dropout] = []

        # Create fixed shearlet transform layer
        self.shearlet = ShearletTransform(
            scales=config.shearlet_scales,
            directions=config.shearlet_directions,
            kernel_regularizer=config.kernel_regularizer
        )

        # Create complex convolutional layers
        for filters in config.conv_filters:
            self.conv_layers.append(
                ComplexConv2D(
                    filters=filters,
                    kernel_size=config.conv_kernel_size,
                    strides=config.conv_strides,
                    padding="same",  # Added padding for better feature extraction
                    kernel_regularizer=config.kernel_regularizer,
                    kernel_initializer=config.kernel_initializer
                )
            )
            self.pool_layers.append(AveragePooling2D(config.pool_size))

        # Complex ReLU activation
        self.activation = ComplexReLU()

        # Flatten layer
        self.flatten = Flatten()

        # Complex dense layers with dropout
        for units in config.dense_units:
            self.dense_layers.append(
                ComplexDense(
                    units=units,
                    kernel_regularizer=config.kernel_regularizer,
                    kernel_initializer=config.kernel_initializer
                )
            )
            self.dropout_layers.append(Dropout(config.dropout_rate))

        # Final classification layer
        self.classifier = Dense(
            config.num_classes,
            activation="softmax",  # Added softmax for classification output
            kernel_regularizer=config.kernel_regularizer,
            kernel_initializer=config.kernel_initializer
        )

        # Build model
        self.build((None, *config.input_shape))
        logger.info(f"CoShNet initialized with config: {config}")

    def call(self,
             inputs: tf.Tensor,
             training: bool = False) -> tf.Tensor:
        """Forward pass through the network.

        Args:
            inputs (tf.Tensor): Input tensor of shape [batch_size, height, width, channels]
            training (bool, optional): Whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        # Apply shearlet transform
        x = self.shearlet(inputs)

        # Convert to complex
        x = tf.cast(x, tf.complex64)

        # Convolutional layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.activation(x)
            x = self.pool_layers[i](x)

        # Flatten
        x = self.flatten(x)

        # Dense layers
        for i, dense in enumerate(self.dense_layers):
            x = dense(x)
            x = self.activation(x)
            if training:
                x = self.dropout_layers[i](x)

        # Final classification (convert to real by taking magnitude)
        x = tf.abs(x)
        return self.classifier(x)

    def build_graph(self) -> None:
        """Build computational graph visualization.

        Creates and displays a visual representation of the model's architecture
        using TensorFlow's summary functionality.
        """
        x = keras.Input(shape=self.config.input_shape)
        model = keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            'config': self.config.to_dict()
        }

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'CoShNet':
        """Create model from configuration.

        Args:
            config_dict (Dict[str, Any]): Configuration dictionary

        Returns:
            CoShNet: Model instance
        """
        config = CoShNetConfig(**config_dict['config'])
        return cls(config)

    def save_model(self, filepath: str) -> None:
        """Save model to .keras format.

        Args:
            filepath (str): Path to save the model
        """
        if not filepath.endswith('.keras'):
            filepath += '.keras'

        self.save(filepath, save_format='keras')
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'CoShNet':
        """Load model from file.

        Args:
            filepath (str): Path to load the model from

        Returns:
            CoShNet: Loaded model
        """
        model = keras.models.load_model(filepath)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is not a {cls.__name__}")
        return model


# ---------------------------------------------------------------------

"""
Squeeze-and-Excitation Layer
===========================

Implementation of the Squeeze-and-Excitation (SE) block as described in:
"Squeeze-and-Excitation Networks" (Hu et al., 2019)
https://arxiv.org/abs/1709.01507

Key Features:
------------
- Channel-wise attention mechanism
- Configurable reduction ratio
- Configurable activation function
- Learnable channel scaling
- Support for various kernel initializations
- Optional bias terms

Architecture:
------------
The SE block consists of:
1. Global average pooling (squeeze)
2. Two-layer bottleneck (excitation):
   - Dimensionality reduction (1x1 conv)
   - Configurable activation (default: ReLU)
   - Dimensionality restoration (1x1 conv)
   - Sigmoid activation
3. Channel-wise scaling

The computation flow is:
input -> global_pool -> conv_reduce -> activation -> conv_restore -> sigmoid -> scale -> output

References:
----------
Based on implementation from:
https://github.com/lvpeiqing/SAR-U-Net-liver-segmentation/blob/master/models/core/modules.py

Usage Example:
------------
```python
# Default configuration with ReLU activation
se_block = SqueezeExcitation(
    reduction_ratio=0.25,
    kernel_initializer='he_normal',
    kernel_regularizer=keras.regularizers.L2(l2=0.01)
)

# Custom activation
se_block = SqueezeExcitation(
    reduction_ratio=0.25,
    activation='swish',
    kernel_regularizer=keras.regularizers.L2(l2=0.01)
)
```
"""

import keras
import tensorflow as tf
from keras import layers, activations
from typing import Dict, Optional, Tuple, Union, Callable


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .layer_scale import LearnableMultiplier


# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class SqueezeExcitation(layers.Layer):
    """Squeeze-and-Excitation block for channel-wise feature recalibration.

    This layer implements the Squeeze-and-Excitation mechanism that adaptively
    recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.

    Args:
        reduction_ratio: Float between 0 and 1, determining the bottleneck width.
            Defaults to 0.25.
        kernel_initializer: Initializer for the kernel weights matrix.
            Defaults to "glorot_normal".
        kernel_regularizer: Optional regularizer function applied to kernel weights.
            Defaults to None.
        activation: Activation function to use after reduction convolution.
            Can be a string identifier or a callable. Defaults to "relu".
        use_bias: Boolean, whether the layer uses a bias vector.
            Defaults to False.

    Raises:
        ValueError: If reduction_ratio is not in the range (0, 1].
    """

    def __init__(
            self,
            reduction_ratio: float = 0.25,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activation: Union[str, Callable] = "relu",
            use_bias: bool = False,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if not 0 < reduction_ratio <= 1.0:
            raise ValueError(
                f"reduction_ratio must be in range (0, 1], got {reduction_ratio}"
            )

        # Layer parameters
        self.reduction_ratio = reduction_ratio
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.activation = activations.get(activation)

        # Layer components (initialized in build)
        self.input_channels: int = -1
        self.bottleneck_channels: int = -1
        self.global_pool: Optional[layers.Layer] = None
        self.conv_reduce: Optional[layers.Conv2D] = None
        self.conv_restore: Optional[layers.Conv2D] = None
        self.scale: Optional[LearnableMultiplier] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Builds the layer with given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        self.input_channels = input_shape[-1]
        self.bottleneck_channels = max(1, int(round(
            self.input_channels * self.reduction_ratio
        )))

        # Global pooling for squeeze operation
        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)

        # Channel reduction convolution
        self.conv_reduce = layers.Conv2D(
            filters=self.bottleneck_channels,
            kernel_size=(1, 1),
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        # Channel restoration convolution
        self.conv_restore = layers.Conv2D(
            filters=self.input_channels,
            kernel_size=(1, 1),
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        # Learnable channel scaling
        self.scale = LearnableMultiplier(
            capped=True,
            multiplier_type="channel"
        )

        super().build(input_shape)

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether in training mode.

        Returns:
            Output tensor after applying SE operations.
        """
        # Squeeze operation
        x = self.global_pool(inputs)

        # Excitation operation
        x = self.conv_reduce(x, training=training)
        x = self.activation(x)  # Using configured activation
        x = self.conv_restore(x, training=training)
        x = self.scale(x, training=training)
        x = tf.nn.sigmoid(x)

        # Scale input features
        return tf.math.multiply(inputs, x)

    def get_config(self) -> Dict:
        """Returns the config of the layer.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
        })
        return config

# ---------------------------------------------------------------------
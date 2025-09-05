"""
This module provides the `Universal Inverted Bottleneck` (UIB), a highly flexible
and configurable Keras layer that serves as a unified building block for a variety
of modern, efficient convolutional neural network architectures.

The UIB is not a new, distinct architecture itself, but rather a generalization that
can be configured to exactly replicate several successful designs, including the
Inverted Bottleneck (IB) from MobileNetV2, the ConvNeXt block, the Feed-Forward
Network (FFN) block from Transformers, and a variant with an extra depthwise
convolution (ExtraDW). This unification allows for easy architectural experimentation
and searching within a single, consistent framework.

Core Architectural Pattern:

The UIB follows a general "expand-process-project" pattern, common to many efficient
network designs:

1.  **Expansion Phase:**
    -   The input feature map, which has a relatively small number of channels, is
        first "expanded" to a much higher-dimensional space using a `1x1 Conv2D`
        layer. The degree of this expansion is controlled by the `expansion_factor`.

2.  **Processing Phase (Depthwise Convolutions):**
    -   The core processing happens in this high-dimensional space using one or two
        efficient `DepthwiseConv2D` layers. Depthwise convolutions process each
        channel independently, capturing spatial patterns without the high
        computational cost of standard convolutions.
    -   The UIB can be configured to use zero, one (`use_dw1=True`), or two
        (`use_dw1=True` and `use_dw2=True`) of these depthwise layers. The first one
        can optionally perform spatial downsampling via its `stride`.

3.  **Projection Phase:**
    -   After the spatial processing, the high-dimensional feature map is projected
        back down to the desired number of output channels using another `1x1 Conv2D`
        layer.

4.  **Residual Connection:**
    -   If the input and output dimensions match (i.e., `stride=1` and `input_filters=output_filters`),
        a residual "skip" connection adds the original input to the output of the
        block. This is crucial for enabling the training of very deep networks.

Emulating Other Architectures:

By toggling its boolean flags (`use_dw1`, `use_dw2`) and other parameters, the UIB
can mimic the following blocks:

-   **Inverted Bottleneck (IB) / MobileNetV2 block:**
    -   `use_dw1=True`, `use_dw2=False`. This creates the classic expand -> depthwise conv -> project structure.

-   **ConvNeXt block:**
    -   `use_dw1=True`, `use_dw2=False`, with `kernel_size=7` and specific normalization/activation
      ordering (note: this implementation uses a standard BN-ReLU order).

-   **Transformer Feed-Forward Network (FFN):**
    -   `use_dw1=False`, `use_dw2=False`. This reduces the block to two `1x1` convolutions
      (expand and project), which is the convolutional equivalent of the two `Dense`
      layers in a Transformer's FFN.

-   **Extra Depthwise (ExtraDW):**
    -   `use_dw1=True`, `use_dw2=True`. This adds a second depthwise convolution to the
      standard IB block, potentially increasing its capacity to learn complex spatial
      features.
"""

import keras
from typing import Tuple, Optional, Any, Dict, Union
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class UIB(keras.layers.Layer):
    """Universal Inverted Bottleneck (UIB) block.

    This block unifies and extends various efficient building blocks:
    Inverted Bottleneck (IB), ConvNext, Feed-Forward Network (FFN), and Extra Depthwise (ExtraDW).

    Args:
        filters: Number of output filters.
        expansion_factor: Expansion factor for the hidden dimension of the block.
        stride: Stride for the first depthwise convolution, used for downsampling.
        kernel_size: Kernel size for all depthwise convolutions.
        use_dw1: Whether to use the first depthwise convolution.
        use_dw2: Whether to use the second depthwise convolution.
        block_type: A string identifier for the block type ('IB', 'ConvNext',
            'ExtraDW', or 'FFN'). This is for configuration tracking and does
            not alter behavior, which is controlled by `use_dw1`/`use_dw2`.
        kernel_initializer: Initializer for the convolution kernels.
        kernel_regularizer: Regularizer for the convolution kernels.
        **kwargs: Additional arguments for the `Layer` base class.
    """

    def __init__(
            self,
            filters: int,
            expansion_factor: int = 4,
            stride: int = 1,
            kernel_size: int = 3,
            use_dw1: bool = True,
            use_dw2: bool = False,
            block_type: str = 'IB',
            kernel_initializer: Union[str, initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ):
        super().__init__(**kwargs)

        # 1. Store all configuration parameters
        self.filters = filters
        self.expansion_factor = expansion_factor
        self.stride = stride
        self.kernel_size = kernel_size
        self.use_dw1 = use_dw1
        self.use_dw2 = use_dw2
        self.block_type = block_type
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Shared configuration for convolution layers
        conv_config = {
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "padding": "same",
            "use_bias": False
        }

        # 2. CREATE all sub-layers in __init__ (they remain unbuilt)
        # Expansion phase
        # Note: The expanded filter count is determined from input_shape in build()
        self.expand_conv = layers.Conv2D(
            filters=1,  # Placeholder, will be set in build()
            kernel_size=1,
            **conv_config
        )
        self.bn1 = layers.BatchNormalization()
        self.activation = layers.ReLU()

        # Processing phase (depthwise convolutions)
        if self.use_dw1:
            self.dw1 = layers.DepthwiseConv2D(
                kernel_size=self.kernel_size,
                strides=self.stride,
                **conv_config
            )
            self.bn_dw1 = layers.BatchNormalization()
        else:
            self.dw1 = None
            self.bn_dw1 = None

        if self.use_dw2:
            self.dw2 = layers.DepthwiseConv2D(
                kernel_size=self.kernel_size,
                strides=1,
                **conv_config
            )
            self.bn_dw2 = layers.BatchNormalization()
        else:
            self.dw2 = None
            self.bn_dw2 = None

        # Projection phase
        self.project_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            **conv_config
        )
        self.bn2 = layers.BatchNormalization()

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer's weights and sub-layers."""
        input_filters = input_shape[-1]
        expanded_filters = input_filters * self.expansion_factor
        self.expand_conv.filters = expanded_filters

        # Build sub-layers sequentially, propagating shape information.
        # This ensures robust serialization and weight loading.
        current_shape = input_shape

        self.expand_conv.build(current_shape)
        current_shape = self.expand_conv.compute_output_shape(current_shape)

        self.bn1.build(current_shape)

        if self.dw1 is not None:
            self.dw1.build(current_shape)
            current_shape = self.dw1.compute_output_shape(current_shape)
            self.bn_dw1.build(current_shape)

        if self.dw2 is not None:
            self.dw2.build(current_shape)
            current_shape = self.dw2.compute_output_shape(current_shape)
            self.bn_dw2.build(current_shape)

        self.project_conv.build(current_shape)
        current_shape = self.project_conv.compute_output_shape(current_shape)

        self.bn2.build(current_shape)

        # Call the parent's build method at the end
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the UIB block."""
        shortcut = inputs

        # Expansion phase
        x = self.expand_conv(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)

        # Processing phase
        if self.dw1 is not None:
            x = self.dw1(x)
            x = self.bn_dw1(x, training=training)
            x = self.activation(x)

        if self.dw2 is not None:
            x = self.dw2(x)
            x = self.bn_dw2(x, training=training)
            x = self.activation(x)

        # Projection phase
        x = self.project_conv(x)
        x = self.bn2(x, training=training)

        # Residual connection
        if self.stride == 1 and ops.shape(inputs)[-1] == self.filters:
            return shortcut + x
        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer."""
        output_shape = list(input_shape)

        # Apply stride to spatial dimensions
        if self.stride > 1:
            if output_shape[1] is not None:
                output_shape[1] //= self.stride
            if output_shape[2] is not None:
                output_shape[2] //= self.stride

        # Update channel dimension
        output_shape[-1] = self.filters

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer's configuration for serialization."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "expansion_factor": self.expansion_factor,
            "stride": self.stride,
            "kernel_size": self.kernel_size,
            "use_dw1": self.use_dw1,
            "use_dw2": self.use_dw2,
            "block_type": self.block_type,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

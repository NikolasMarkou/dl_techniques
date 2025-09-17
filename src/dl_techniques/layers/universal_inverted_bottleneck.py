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
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.activations import create_activation_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class UniversalInvertedBottleneck(keras.layers.Layer):
    """
    Universal Inverted Bottleneck (UIB) - A highly configurable building block for efficient CNNs.

    This layer unifies and extends various efficient building blocks including Inverted Bottleneck (IB),
    ConvNeXt, Feed-Forward Network (FFN), and Extra Depthwise (ExtraDW) variants. It provides extensive
    configurability for activation functions, normalization methods, dropout, and architectural options
    while maintaining backward compatibility.

    **Intent**: Provide a single, unified building block that can replicate or extend multiple successful
    CNN architectures through configuration, enabling easy architectural experimentation and neural
    architecture search within a consistent framework.

    **Architecture**:
    ```
    Input(shape=[batch, H, W, C_in])
           ↓
    Expand: Conv2D(1x1) → Norm → Activation
           ↓
    Process: DepthwiseConv2D(optional) → Norm → Activation → Dropout(optional)
           ↓
    Process: DepthwiseConv2D(optional) → Norm → Activation → Dropout(optional)
           ↓
    SE Block(optional): Squeeze-and-Excitation
           ↓
    Project: Conv2D(1x1) → Norm
           ↓
    Residual: Add input if stride=1 and C_in=C_out
           ↓
    Output(shape=[batch, H//stride, W//stride, filters])
    ```

    **Architectural Variants**:
    - **Inverted Bottleneck (MobileNetV2)**: `use_dw1=True, use_dw2=False`
    - **ConvNeXt Block**: `use_dw1=True, use_dw2=False, kernel_size=7, activation_type='gelu'`
    - **FFN Block**: `use_dw1=False, use_dw2=False` (pure 1x1 convolutions)
    - **Extra Depthwise**: `use_dw1=True, use_dw2=True`

    Args:
        filters: Integer, number of output filters. Must be positive.
            Determines the final channel dimension of the block output.
        expansion_factor: Integer, expansion factor for the hidden dimension.
            The intermediate channel count becomes `input_filters * expansion_factor`.
            Must be positive. Defaults to 4.
        stride: Integer, stride for the first depthwise convolution.
            Used for spatial downsampling. Must be positive. Defaults to 1.
        kernel_size: Integer, kernel size for depthwise convolutions.
            Must be positive and odd for symmetric padding. Defaults to 3.
        use_dw1: Boolean, whether to use the first depthwise convolution.
            When False, skips the first depthwise processing stage. Defaults to True.
        use_dw2: Boolean, whether to use the second depthwise convolution.
            When False, skips the second depthwise processing stage. Defaults to False.
        activation_type: String, type of activation function to use.
            Supports all activations from dl_techniques activation factory.
            Defaults to 'relu' for backward compatibility.
        activation_args: Optional dictionary of activation-specific arguments.
            Passed to the activation factory for customization. Defaults to None.
        normalization_type: String, type of normalization to use.
            Supports all normalizations from dl_techniques normalization factory.
            Defaults to 'batch_norm' for backward compatibility.
        normalization_args: Optional dictionary of normalization-specific arguments.
            Passed to the normalization factory for customization. Defaults to None.
        dropout_rate: Float, dropout rate applied after depthwise convolutions.
            Must be between 0.0 and 1.0. When 0.0, no dropout is applied.
            Defaults to 0.0.
        use_squeeze_excitation: Boolean, whether to add SE block before projection.
            Adds channel attention mechanism for improved feature selection.
            Defaults to False.
        se_ratio: Float, reduction ratio for SE block.
            Controls the bottleneck size in SE block. Only used when use_squeeze_excitation=True.
            Must be positive. Defaults to 0.25.
        se_activation: String, activation function for SE block.
            Used for the first activation in SE block. Defaults to 'relu'.
        use_bias: Boolean, whether to use bias in convolution layers.
            Defaults to False for efficiency and since normalization follows.
        padding: String, padding type for convolutions.
            Must be 'same', 'valid', or 'causal'. Defaults to 'same'.
        block_type: String identifier for the block type.
            Used for configuration tracking and logging. Does not affect behavior.
            Defaults to 'UIB'.
        kernel_initializer: String or initializer, initializer for convolution kernels.
            Defaults to 'he_normal'.
        depthwise_initializer: String or initializer, initializer for depthwise kernels.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional regularizer for convolution kernels.
            Applied to all Conv2D layers. Defaults to None.
        depthwise_regularizer: Optional regularizer for depthwise kernels.
            Applied to all DepthwiseConv2D layers. Defaults to None.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        4D tensor with shape: `(batch_size, new_height, new_width, filters)`.
        `new_height` and `new_width` are reduced by `stride` if stride > 1.

    Raises:
        ValueError: If any parameter is invalid (e.g., non-positive filters, invalid dropout rate).
        ValueError: If activation_type or normalization_type is not supported.

    Example:
        ```python
        # Standard MobileNetV2 Inverted Bottleneck
        ib_block = UniversalInvertedBottleneck(
            filters=64,
            expansion_factor=6,
            stride=2,
            kernel_size=3,
            use_dw1=True,
            use_dw2=False
        )

        # ConvNeXt-style block with GELU activation
        convnext_block = UniversalInvertedBottleneck(
            filters=128,
            expansion_factor=4,
            kernel_size=7,
            use_dw1=True,
            use_dw2=False,
            activation_type='gelu',
            normalization_type='layer_norm'
        )

        # FFN block with custom activation and dropout
        ffn_block = UniversalInvertedBottleneck(
            filters=256,
            expansion_factor=4,
            use_dw1=False,
            use_dw2=False,
            activation_type='silu',
            dropout_rate=0.1,
            use_squeeze_excitation=True
        )

        # Extra Depthwise with advanced configuration
        extradw_block = UniversalInvertedBottleneck(
            filters=512,
            expansion_factor=8,
            use_dw1=True,
            use_dw2=True,
            activation_type='mish',
            normalization_type='rms_norm',
            dropout_rate=0.2,
            use_squeeze_excitation=True,
            se_ratio=0.125
        )
        ```

    Note:
        This implementation maintains backward compatibility with the original UIB layer.
        All new parameters have defaults that preserve the original behavior.
        The layer automatically handles residual connections when appropriate.
    """

    def __init__(
            self,
            filters: int,
            expansion_factor: int = 4,
            stride: int = 1,
            kernel_size: int = 3,
            use_dw1: bool = True,
            use_dw2: bool = False,
            activation_type: str = 'relu',
            activation_args: Optional[Dict[str, Any]] = None,
            normalization_type: str = 'batch_norm',
            normalization_args: Optional[Dict[str, Any]] = None,
            dropout_rate: float = 0.0,
            use_squeeze_excitation: bool = False,
            se_ratio: float = 0.25,
            se_activation: str = 'relu',
            use_bias: bool = False,
            padding: str = 'same',
            block_type: str = 'UIB',
            kernel_initializer: Union[str, initializers.Initializer] = "he_normal",
            depthwise_initializer: Union[str, initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            depthwise_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if expansion_factor <= 0:
            raise ValueError(f"expansion_factor must be positive, got {expansion_factor}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")
        if se_ratio <= 0:
            raise ValueError(f"se_ratio must be positive, got {se_ratio}")
        if padding not in ['same', 'valid', 'causal']:
            raise ValueError(f"padding must be 'same', 'valid', or 'causal', got {padding}")

        # Store all configuration parameters
        self.filters = filters
        self.expansion_factor = expansion_factor
        self.stride = stride
        self.kernel_size = kernel_size
        self.use_dw1 = use_dw1
        self.use_dw2 = use_dw2
        self.activation_type = activation_type
        self.activation_args = activation_args or {}
        self.normalization_type = normalization_type
        self.normalization_args = normalization_args or {}
        self.dropout_rate = dropout_rate
        self.use_squeeze_excitation = use_squeeze_excitation
        self.se_ratio = se_ratio
        self.se_activation = se_activation
        self.use_bias = use_bias
        self.padding = padding
        self.block_type = block_type
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)

        # Common configuration for layers
        self.conv_config = {
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "padding": self.padding,
            "use_bias": self.use_bias
        }

        self.depth_conv_config = {
            "depthwise_initializer": self.depthwise_initializer,
            "depthwise_regularizer": self.depthwise_regularizer,
            "padding": self.padding,
            "use_bias": self.use_bias
        }

        # CREATE all sub-layers in __init__ (they remain unbuilt)
        # Expansion phase
        self.expand_conv = layers.Conv2D(
            filters=1,  # Placeholder, will be set in build()
            kernel_size=1,
            name='expand_conv',
            **self.conv_config
        )

        self.expand_norm = self._create_normalization_layer('expand_norm')
        self.expand_activation = self._create_activation_layer('expand_activation')

        # Processing phase (depthwise convolutions)
        if self.use_dw1:
            self.dw1 = layers.DepthwiseConv2D(
                kernel_size=self.kernel_size,
                strides=self.stride,
                name='dw1',
                **self.depth_conv_config
            )
            self.dw1_norm = self._create_normalization_layer('dw1_norm')
            self.dw1_activation = self._create_activation_layer('dw1_activation')

            if self.dropout_rate > 0:
                self.dw1_dropout = layers.Dropout(self.dropout_rate, name='dw1_dropout')
            else:
                self.dw1_dropout = None
        else:
            self.dw1 = None
            self.dw1_norm = None
            self.dw1_activation = None
            self.dw1_dropout = None

        if self.use_dw2:
            self.dw2 = layers.DepthwiseConv2D(
                kernel_size=self.kernel_size,
                strides=1,
                name='dw2',
                **self.depth_conv_config
            )
            self.dw2_norm = self._create_normalization_layer('dw2_norm')
            self.dw2_activation = self._create_activation_layer('dw2_activation')

            if self.dropout_rate > 0:
                self.dw2_dropout = layers.Dropout(self.dropout_rate, name='dw2_dropout')
            else:
                self.dw2_dropout = None
        else:
            self.dw2 = None
            self.dw2_norm = None
            self.dw2_activation = None
            self.dw2_dropout = None

        # Squeeze-and-Excitation block
        if self.use_squeeze_excitation:
            self.se_squeeze = layers.GlobalAveragePooling2D(keepdims=True, name='se_squeeze')
            # SE layers will be created in build() since they need channel info
            self.se_reduce = None
            self.se_expand = None
            self.se_activation1 = create_activation_layer(self.se_activation, name='se_activation1')
            self.se_activation2 = layers.Activation('sigmoid', name='se_activation2')
        else:
            self.se_squeeze = None
            self.se_reduce = None
            self.se_expand = None
            self.se_activation1 = None
            self.se_activation2 = None

        # Projection phase
        self.project_conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            name='project_conv',
            **self.conv_config
        )
        self.project_norm = self._create_normalization_layer('project_norm')

    def _create_activation_layer(self, name: str) -> keras.layers.Layer:
        """Create activation layer using the factory."""
        try:
            return create_activation_layer(
                self.activation_type,
                name=name,
                **self.activation_args
            )
        except Exception as e:
            raise ValueError(f"Failed to create activation layer '{self.activation_type}': {e}")

    def _create_normalization_layer(self, name: str) -> keras.layers.Layer:
        """Create normalization layer using the factory."""
        try:
            return create_normalization_layer(
                self.normalization_type,
                name=name,
                **self.normalization_args
            )
        except Exception as e:
            raise ValueError(f"Failed to create normalization layer '{self.normalization_type}': {e}")

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer's weights and sub-layers."""
        input_filters = input_shape[-1]
        expanded_filters = input_filters * self.expansion_factor

        # Set the actual number of filters for expansion
        self.expand_conv.filters = expanded_filters

        # Build sub-layers sequentially, propagating shape information
        # This ensures robust serialization and weight loading
        current_shape = input_shape

        # Expansion phase
        self.expand_conv.build(current_shape)
        current_shape = self.expand_conv.compute_output_shape(current_shape)

        self.expand_norm.build(current_shape)
        # Activation doesn't change shape

        # Processing phase
        if self.dw1 is not None:
            self.dw1.build(current_shape)
            current_shape = self.dw1.compute_output_shape(current_shape)

            self.dw1_norm.build(current_shape)
            # Activation doesn't change shape
            # Dropout doesn't change shape

        if self.dw2 is not None:
            self.dw2.build(current_shape)
            current_shape = self.dw2.compute_output_shape(current_shape)

            self.dw2_norm.build(current_shape)
            # Activation doesn't change shape
            # Dropout doesn't change shape

        # Squeeze-and-Excitation block
        if self.use_squeeze_excitation:
            expanded_filters_se = current_shape[-1]
            se_filters = max(1, int(expanded_filters_se * self.se_ratio))

            # Create SE layers with proper channel dimensions
            self.se_reduce = layers.Conv2D(
                filters=se_filters,
                kernel_size=1,
                activation=None,
                name='se_reduce',
                **self.conv_config
            )

            self.se_expand = layers.Conv2D(
                filters=expanded_filters_se,
                kernel_size=1,
                activation=None,
                name='se_expand',
                **self.conv_config
            )

            # Build SE layers
            se_shape = (current_shape[0], 1, 1, expanded_filters_se)
            self.se_reduce.build(se_shape)
            se_reduced_shape = self.se_reduce.compute_output_shape(se_shape)
            self.se_expand.build(se_reduced_shape)

        # Projection phase
        self.project_conv.build(current_shape)
        current_shape = self.project_conv.compute_output_shape(current_shape)

        self.project_norm.build(current_shape)

        # Call the parent's build method at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the UIB block."""
        shortcut = inputs

        # Expansion phase
        x = self.expand_conv(inputs)
        x = self.expand_norm(x, training=training)
        x = self.expand_activation(x)

        # Processing phase
        if self.dw1 is not None:
            x = self.dw1(x)
            x = self.dw1_norm(x, training=training)
            x = self.dw1_activation(x)

            if self.dw1_dropout is not None:
                x = self.dw1_dropout(x, training=training)

        if self.dw2 is not None:
            x = self.dw2(x)
            x = self.dw2_norm(x, training=training)
            x = self.dw2_activation(x)

            if self.dw2_dropout is not None:
                x = self.dw2_dropout(x, training=training)

        # Squeeze-and-Excitation block
        if self.use_squeeze_excitation:
            se = self.se_squeeze(x)
            se = self.se_reduce(se)
            se = self.se_activation1(se)
            se = self.se_expand(se)
            se = self.se_activation2(se)
            x = x * se

        # Projection phase
        x = self.project_conv(x)
        x = self.project_norm(x, training=training)

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
                output_shape[1] = (output_shape[1] + self.stride - 1) // self.stride
            if output_shape[2] is not None:
                output_shape[2] = (output_shape[2] + self.stride - 1) // self.stride

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
            "activation_type": self.activation_type,
            "activation_args": self.activation_args,
            "normalization_type": self.normalization_type,
            "normalization_args": self.normalization_args,
            "dropout_rate": self.dropout_rate,
            "use_squeeze_excitation": self.use_squeeze_excitation,
            "se_ratio": self.se_ratio,
            "se_activation": self.se_activation,
            "use_bias": self.use_bias,
            "padding": self.padding,
            "block_type": self.block_type,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "depthwise_initializer": initializers.serialize(self.depthwise_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "depthwise_regularizer": regularizers.serialize(self.depthwise_regularizer),
        })
        return config


# Backward compatibility alias
UIB = UniversalInvertedBottleneck

# ---------------------------------------------------------------------
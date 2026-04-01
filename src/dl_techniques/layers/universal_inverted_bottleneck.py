"""
A `Universal Inverted Bottleneck` (UIB), a highly flexible
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
        layer. The degree of this expansion is controlled by the `expansion_factor`
        or by specifying the exact `expanded_channels`.

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

from dl_techniques.layers.norms import create_normalization_layer, NormalizationType
from dl_techniques.layers.activations import create_activation_layer, ActivationType

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class UniversalInvertedBottleneck(keras.layers.Layer):
    """
    Universal Inverted Bottleneck (UIB) for efficient CNNs.

    A highly configurable building block that unifies Inverted Bottleneck (IB),
    ConvNeXt, Feed-Forward Network (FFN), and Extra Depthwise (ExtraDW) variants
    through boolean flags and parameter selection. The block follows an
    expand-process-project pattern with optional depthwise convolutions, SE
    attention, and residual connections.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input [B, H, W, C_in]                   │
        └──────┬───────────────────────────┬───────┘
               ▼                           │ (residual)
        ┌──────────────────────┐           │
        │  Expand: Conv1x1     │           │
        │  → Norm → Act        │           │
        ├──────────────────────┤           │
        │  Opt. DW Conv (dw1)  │           │
        │  → Norm → Act → Drop │           │
        ├──────────────────────┤           │
        │  Opt. DW Conv (dw2)  │           │
        │  → Norm → Act → Drop │           │
        ├──────────────────────┤           │
        │  Opt. SE Block       │           │
        ├──────────────────────┤           │
        │  Project: Conv1x1    │           │
        │  → Norm              │           │
        └──────┬───────────────┘           │
               ▼                           ▼
        ┌──────────────────────────────────────────┐
        │  Add (if stride=1 and C_in=filters)      │
        └──────────────┬───────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────┐
        │  Output [B, H/s, W/s, filters]           │
        └──────────────────────────────────────────┘

    :param filters: Number of output filters. Must be positive.
    :type filters: int
    :param expansion_factor: Expansion factor for hidden dimension. Ignored if
        ``expanded_channels`` is provided. Defaults to 4.
    :type expansion_factor: int
    :param expanded_channels: Exact number of expansion channels (overrides
        ``expansion_factor``). Defaults to None.
    :type expanded_channels: int or None
    :param stride: Stride for the first depthwise convolution. Defaults to 1.
    :type stride: int
    :param kernel_size: Kernel size for depthwise convolutions. Defaults to 3.
    :type kernel_size: int
    :param use_dw1: Whether to use the first depthwise convolution. Defaults to True.
    :type use_dw1: bool
    :param use_dw2: Whether to use the second depthwise convolution. Defaults to False.
    :type use_dw2: bool
    :param activation_type: Activation function type. Defaults to ``'relu'``.
    :type activation_type: str
    :param activation_args: Additional activation arguments.
    :type activation_args: dict or None
    :param normalization_type: Normalization type. Defaults to ``'batch_norm'``.
    :type normalization_type: str
    :param normalization_args: Additional normalization arguments.
    :type normalization_args: dict or None
    :param dropout_rate: Dropout rate after depthwise convolutions. Defaults to 0.0.
    :type dropout_rate: float
    :param use_squeeze_excitation: Whether to add SE block. Defaults to False.
    :type use_squeeze_excitation: bool
    :param se_ratio: Reduction ratio for SE block. Defaults to 0.25.
    :type se_ratio: float
    :param se_activation: Activation for SE block. Defaults to ``'relu'``.
    :type se_activation: str
    :param use_bias: Whether to use bias in convolutions. Defaults to False.
    :type use_bias: bool
    :param padding: Padding type. Defaults to ``'same'``.
    :type padding: str
    :param block_type: String identifier for the block type. Defaults to ``'UIB'``.
    :type block_type: str
    :param kernel_initializer: Initializer for convolution kernels. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param depthwise_initializer: Initializer for depthwise kernels. Defaults to ``'he_normal'``.
    :type depthwise_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for convolution kernels.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param depthwise_regularizer: Optional regularizer for depthwise kernels.
    :type depthwise_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional arguments for the Layer base class.
    :type kwargs: Any
    """

    def __init__(
            self,
            filters: int,
            expansion_factor: int = 4,
            expanded_channels: Optional[int] = None,
            stride: int = 1,
            kernel_size: int = 3,
            use_dw1: bool = True,
            use_dw2: bool = False,
            activation_type: ActivationType = 'relu',
            activation_args: Optional[Dict[str, Any]] = None,
            normalization_type: NormalizationType = 'batch_norm',
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
        # --- CHANGE START ---
        # Added validation for the new expanded_channels parameter.
        if expansion_factor <= 0 and expanded_channels is None:
            raise ValueError(
                "Either expansion_factor must be positive or "
                "expanded_channels must be provided."
            )
        if expanded_channels is not None and expanded_channels <= 0:
            raise ValueError(
                f"expanded_channels must be positive, got {expanded_channels}"
            )
        # --- CHANGE END ---
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}"
            )
        if se_ratio <= 0:
            raise ValueError(f"se_ratio must be positive, got {se_ratio}")
        if padding not in ['same', 'valid', 'causal']:
            raise ValueError(
                f"padding must be 'same', 'valid', or 'causal', got {padding}"
            )

        # Store all configuration parameters
        self.filters = filters
        self.expansion_factor = expansion_factor
        # --- CHANGE START ---
        # Storing the new parameter.
        self.expanded_channels = expanded_channels
        # --- CHANGE END ---
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
                self.dw1_dropout = layers.Dropout(
                    self.dropout_rate, name='dw1_dropout'
                )
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
                self.dw2_dropout = layers.Dropout(
                    self.dropout_rate, name='dw2_dropout'
                )
            else:
                self.dw2_dropout = None
        else:
            self.dw2 = None
            self.dw2_norm = None
            self.dw2_activation = None
            self.dw2_dropout = None

        # Squeeze-and-Excitation block
        if self.use_squeeze_excitation:
            self.se_squeeze = layers.GlobalAveragePooling2D(
                keepdims=True, name='se_squeeze'
            )
            # SE layers will be created in build() since they need channel info
            self.se_reduce = None
            self.se_expand = None
            self.se_activation1 = create_activation_layer(
                self.se_activation, name='se_activation1'
            )
            self.se_activation2 = layers.Activation(
                'sigmoid', name='se_activation2'
            )
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
            raise ValueError(
                f"Failed to create activation layer '{self.activation_type}': {e}"
            )

    def _create_normalization_layer(self, name: str) -> keras.layers.Layer:
        """Create normalization layer using the factory."""
        try:
            return create_normalization_layer(
                self.normalization_type,
                name=name,
                **self.normalization_args
            )
        except Exception as e:
            raise ValueError(
                f"Failed to create normalization layer '{self.normalization_type}': {e}"
            )

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer's weights and sub-layers."""
        input_filters = input_shape[-1]

        # --- CHANGE START ---
        # Logic now prioritizes `expanded_channels` if provided, otherwise it
        # falls back to the original `expansion_factor` calculation. This makes
        # the change fully backward-compatible.
        if self.expanded_channels is not None:
            expanded_filters = self.expanded_channels
        else:
            expanded_filters = input_filters * self.expansion_factor
        # --- CHANGE END ---

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
            # --- CHANGE START ---
            # Added expanded_channels to the config for proper serialization.
            "expanded_channels": self.expanded_channels,
            # --- CHANGE END ---
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
            "kernel_initializer":
                initializers.serialize(self.kernel_initializer),
            "depthwise_initializer":
                initializers.serialize(self.depthwise_initializer),
            "kernel_regularizer":
                regularizers.serialize(self.kernel_regularizer),
            "depthwise_regularizer":
                regularizers.serialize(self.depthwise_regularizer),
        })
        return config

# ---------------------------------------------------------------------
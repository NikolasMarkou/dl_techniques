import keras
from typing import  Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .universal_inverted_bottleneck import UniversalInvertedBottleneck

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class InvertedResidualBlock(UniversalInvertedBottleneck):
    """Inverted residual block, the core building block for MobileNetV2.

    This block implements the inverted residual structure with a linear
    bottleneck, a key innovation of the MobileNetV2 architecture. It consists
    of three main stages: expansion via 1x1 convolution, depthwise spatial
    convolution, and linear projection back to the bottleneck dimension. The
    key insight is the use of a linear (no activation) projection layer, which
    prevents non-linearities from destroying information in the low-dimensional
    bottleneck: ``x_expanded = ReLU6(BN(Conv_1x1(input)))``,
    ``x_dw = ReLU6(BN(DWConv_3x3(x_expanded)))``,
    ``x_proj = BN(Conv_1x1(x_dw))``, ``output = input + x_proj`` if applicable.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────┐
        │       Input [H, W, C_in]          │
        └───────────┬───────────────────────┘
                    │
                    ├──────────────────────────┐
                    ▼                          │
        ┌───────────────────────────────┐      │
        │ 1x1 Conv → BN → ReLU6         │      │
        │ (Expand: C_in → C_expanded)   │      │
        └───────────┬───────────────────┘      │
                    ▼                          │
        ┌───────────────────────────────┐      │
        │ 3x3 DWConv → BN → ReLU6       │      │
        │ (Spatial: C_expanded)         │      │
        └───────────┬───────────────────┘      │
                    ▼                          │
        ┌───────────────────────────────┐      │
        │ 1x1 Conv → BN (LINEAR)        │      │
        │ (Project: C_expanded → C_out) │      │
        └───────────┬───────────────────┘      │
                    │                          │
                    ▼                          │
                ┌───┴───┐  (if stride=1        │
                │  Add  │◄─ and C_in==C_out)───┘
                └───┬───┘
                    ▼
        ┌───────────────────────────────────┐
        │      Output [H', W', C_out]       │
        └───────────────────────────────────┘

    :param filters: Number of output filters (channels).
    :type filters: int
    :param expansion_factor: Expansion ratio for the first layer.
        Determines the intermediate channel dimension.
    :type expansion_factor: int
    :param stride: Stride for the depthwise convolution (1 or 2).
    :type stride: int
    :param block_id: Unique identifier for the block for naming purposes.
    :type block_id: int
    :param skip_connection: Whether to use a residual connection. A residual
        connection is only added if ``stride=1`` and input/output channels match.
    :type skip_connection: bool
    :param kernel_initializer: Initializer for weight initialization.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            filters: int,
            expansion_factor: int = 6,
            stride: int = 1,
            block_id: int = 0,
            skip_connection: bool = True,
            kernel_initializer: Union[
                str, keras.initializers.Initializer
            ] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        # Store original arguments for get_config serialization.
        # These are the public API parameters for this specialized class.
        self._block_id = block_id
        self._skip_connection_arg = skip_connection
        self._kernel_initializer_arg = kernel_initializer
        self._kernel_regularizer_arg = kernel_regularizer

        # Call the parent UniversalInvertedBottleneck's __init__ with the
        # specific configuration for a MobileNetV2 block.
        super().__init__(
            filters=filters,
            expansion_factor=expansion_factor,
            stride=stride,
            kernel_size=3,
            use_dw1=True,
            use_dw2=False,
            activation_type="relu",
            activation_args={"max_value": 6},  # This creates ReLU6
            normalization_type="batch_norm",
            dropout_rate=0.0,
            use_squeeze_excitation=False,
            kernel_initializer=kernel_initializer,
            depthwise_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            depthwise_regularizer=kernel_regularizer,
            name=f"inverted_residual_block_{block_id}",
            **kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        This method ensures the layer is saved using its own simplified
        constructor arguments, not the full UIB configuration.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        # Start with the full configuration from the UIB parent.
        config = super().get_config()

        # Define the set of UIB-specific parameters that are hard-coded
        # in this class and should not be part of the saved config.
        params_to_remove = [
            "expanded_channels",
            "kernel_size",
            "use_dw1",
            "use_dw2",
            "activation_type",
            "activation_args",
            "normalization_type",
            "normalization_args",
            "dropout_rate",
            "use_squeeze_excitation",
            "se_ratio",
            "se_activation",
            "use_bias",
            "padding",
            "block_type",
            "depthwise_initializer",
            "depthwise_regularizer",
        ]
        for param in params_to_remove:
            config.pop(param, None)

        # Update the config with the original arguments of this class,
        # ensuring they are properly serialized.
        config.update(
            {
                "block_id": self._block_id,
                "skip_connection": self._skip_connection_arg,
                "kernel_initializer": keras.initializers.serialize(
                    self._kernel_initializer_arg
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self._kernel_regularizer_arg
                ),
            }
        )

        return config

# ---------------------------------------------------------------------
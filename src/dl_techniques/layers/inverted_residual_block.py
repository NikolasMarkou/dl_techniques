"""
InvertedResidualBlock, the MobileNetV2 inverted-residual building block.

A thin specialization of :class:`UniversalInvertedBottleneck` that fixes the
MobileNetV2 configuration: a 1x1 expansion, a 3x3 depthwise spatial
convolution, and a linear 1x1 projection (no activation on the bottleneck),
with ReLU6 activations and batch normalization throughout. A residual
connection is added when ``stride == 1`` and the input/output channel counts
match.

References:
    - Sandler et al., 2018. MobileNetV2: Inverted Residuals and Linear
      Bottlenecks. (https://arxiv.org/abs/1801.04381)
"""

import keras
from typing import  Optional, Dict, Any, Union

from .universal_inverted_bottleneck import UniversalInvertedBottleneck
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.inverted_residual_block")
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

    Architecture:

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
        # This class's own public constructor args, saved for get_config.
        self._block_id = block_id
        self._skip_connection_arg = skip_connection
        self._kernel_initializer_arg = kernel_initializer
        self._kernel_regularizer_arg = kernel_regularizer

        # An explicit `name` (e.g. restored from get_config) must win over
        # this default, or a round trip collides on both being set.
        kwargs.setdefault("name", f"inverted_residual_block_{block_id}")

        super().__init__(
            filters=filters,
            expansion_factor=expansion_factor,
            stride=stride,
            kernel_size=3,
            use_dw1=True,
            use_dw2=False,
            activation_type="relu",
            # max_value=6 turns this ReLU into ReLU6.
            activation_args={"max_value": 6},
            normalization_type="batch_norm",
            dropout_rate=0.0,
            use_squeeze_excitation=False,
            kernel_initializer=kernel_initializer,
            depthwise_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            depthwise_regularizer=kernel_regularizer,
            **kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        This method ensures the layer is saved using its own simplified
        constructor arguments, not the full UIB configuration.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()

        # These UIB parameters are hard-coded by this class and dropped
        # from the saved config.
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
"""
``PatchMerging`` halves the spatial resolution of a feature map and doubles
its channel depth, the downsampling step used between stages of the Swin
Transformer to build a hierarchical, multi-scale representation.

Unlike max or average pooling, it preserves every input value: it groups each
non-overlapping 2x2 patch into a `4*C`-channel vector by concatenation, then
projects that down to `2*C` channels with a normalized, trainable Dense
layer. Input shape ``(H, W, C)`` maps to ``(H/2, W/2, 2*C)``; odd spatial
dimensions are padded by one before extraction.

References:
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer
      using Shifted Windows. (https://arxiv.org/abs/2103.14030)

"""

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import ops, layers, initializers, regularizers
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.pooling.patch_merging")
class PatchMerging(keras.layers.Layer):
    """Patch merging layer for hierarchical downsampling in Swin Transformers.

    This layer performs spatial downsampling by extracting non-overlapping
    2x2 patches, concatenating them along the channel axis to produce
    ``4*C`` channels, normalising the result, and projecting down to
    ``2*C`` channels via a learned linear transformation. The operation
    halves each spatial dimension while doubling the feature depth,
    analogous to strided pooling in CNNs but fully learnable. For odd
    spatial dimensions the layer automatically pads before extraction.

    Architecture:

    .. code-block:: text

        ┌───────────────────────────────────┐
        │  Input [batch, H, W, C]           │
        └────────────────┬──────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  Extract 2x2 patches:             │
        │  x0 (top-left), x1 (bottom-left)  │
        │  x2 (top-right), x3 (bottom-right)│
        └────────────────┬──────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  Concatenate along channels       │
        │  [batch, H/2, W/2, 4*C]           │
        └────────────────┬──────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  LayerNormalization               │
        └────────────────┬──────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  Dense(2*C) linear projection     │
        └────────────────┬──────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  Output [batch, H/2, W/2, 2*C]    │
        └───────────────────────────────────┘

    :param dim: Number of input channels. Must be positive.
    :type dim: int
    :param use_bias: Whether to include bias in the linear projection.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the projection kernel.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for the projection bias.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for projection weights.
    :type kernel_regularizer: Optional[Union[str, regularizers.Regularizer]]
    :param bias_regularizer: Optional regularizer for the projection bias.
    :type bias_regularizer: Optional[Union[str, regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
            self,
            dim: int,
            use_bias: bool = False,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.norm = layers.LayerNormalization(
            epsilon=1e-5,
            name="norm"
        )

        self.reduction = layers.Dense(
            units=2 * dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="reduction"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build the sub-layers for robust serialization.

        After 2x2 patch extraction and channel concatenation the feature depth
        becomes ``4 * dim``; both the normalization and the linear reduction
        operate on that merged representation.

        :param input_shape: Shape tuple ``(batch, H, W, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        batch_size, height, width, _ = input_shape
        merged_h = None if height is None else (height + 1) // 2
        merged_w = None if width is None else (width + 1) // 2
        merged_shape = (batch_size, merged_h, merged_w, 4 * self.dim)

        self.norm.build(merged_shape)
        self.reduction.build(merged_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the patch merging operation.

        :param inputs: Input tensor of shape ``(batch, H, W, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Merged tensor of shape ``(batch, H//2, W//2, 2*dim)``.
        :rtype: keras.KerasTensor"""
        B, H, W, C = ops.shape(inputs)[0], ops.shape(inputs)[1], ops.shape(inputs)[2], ops.shape(inputs)[3]

        if H % 2 == 1 or W % 2 == 1:
            pad_h = H % 2
            pad_w = W % 2
            inputs = ops.pad(inputs, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
            H, W = H + pad_h, W + pad_w

        # Top-left, bottom-left, top-right, bottom-right of each 2x2 patch.
        x0 = inputs[:, 0::2, 0::2, :]
        x1 = inputs[:, 1::2, 0::2, :]
        x2 = inputs[:, 0::2, 1::2, :]
        x3 = inputs[:, 1::2, 1::2, :]

        x = ops.concatenate([x0, x1, x2, x3], axis=-1)

        x = self.norm(x, training=training)
        x = self.reduction(x, training=training)

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape for shape inference.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        batch_size, height, width, channels = input_shape
        output_height = None if height is None else (height + 1) // 2
        output_width = None if width is None else (width + 1) // 2
        output_channels = self.dim * 2
        return (batch_size, output_height, output_width, output_channels)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

"""
Downsample feature maps by merging patches to create a hierarchical representation.

This layer implements the patch merging strategy introduced in the Swin
Transformer, a key component for building hierarchical Vision Transformers.
While standard Vision Transformers maintain a fixed sequence length of
patches throughout the network, this layer provides a mechanism analogous to
pooling in Convolutional Neural Networks (CNNs), enabling the model to
create multi-scale feature maps. This hierarchical structure allows the
network to learn features at various granularities, from fine-grained details
to global context, which is crucial for complex vision tasks.

Architecturally, the layer performs a learned spatial downsampling. It takes an
input feature map of shape `(H, W, C)` and produces an output of shape
`(H/2, W/2, 2*C)`. This is accomplished through a two-step process:

1.  **Patch Concatenation:** The input feature map is first partitioned into
    non-overlapping 2x2 patches. The features from these four adjacent
    spatial locations are then concatenated along the channel dimension. For
    an input patch at `(2i, 2j)`, this combines information from its
    neighbors at `(2i+1, 2j)`, `(2i, 2j+1)`, and `(2i+1, 2j+1)`. This step
    halves the spatial dimensions while quadrupling the channel depth to `4*C`,
    critically preserving all the information from the input, unlike lossy
    pooling operations.

2.  **Linear Projection:** A trainable linear layer (a Dense layer) then
    projects the resulting `4*C`-dimensional feature vectors down to `2*C`
    dimensions. This projection allows the model to learn the most effective
    way to combine and summarize the features from the local 2x2 neighborhood.
    A Layer Normalization is applied before this projection to stabilize the
    training dynamics.

The overall operation is a powerful, data-driven alternative to fixed
downsampling functions like max or average pooling, forming the backbone of
the feature pyramid structure in the Swin Transformer architecture.

References:
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer
      using Shifted Windows. (https://arxiv.org/abs/2103.14030)

"""

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PatchMerging(keras.layers.Layer):
    """Patch merging layer for hierarchical downsampling in Swin Transformers.

    This layer performs spatial downsampling by extracting non-overlapping
    2x2 patches, concatenating them along the channel axis to produce
    ``4*C`` channels, normalising the result, and projecting down to
    ``2*C`` channels via a learned linear transformation. The operation
    halves each spatial dimension while doubling the feature depth,
    analogous to strided pooling in CNNs but fully learnable. For odd
    spatial dimensions the layer automatically pads before extraction.

    **Architecture Overview:**

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
        │  Output [batch, H/2, W/2, 2*C]   │
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

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        # Store ALL configuration parameters
        self.dim = dim
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
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

        # Handle odd dimensions by padding
        if H % 2 == 1 or W % 2 == 1:
            pad_h = H % 2
            pad_w = W % 2
            inputs = ops.pad(inputs, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
            H, W = H + pad_h, W + pad_w

        # Extract 2x2 patches and concatenate
        x0 = inputs[:, 0::2, 0::2, :]  # Top-left
        x1 = inputs[:, 1::2, 0::2, :]  # Bottom-left
        x2 = inputs[:, 0::2, 1::2, :]  # Top-right
        x3 = inputs[:, 1::2, 1::2, :]  # Bottom-right

        # Concatenate patches: (B, H//2, W//2, 4*C)
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)

        # Apply normalization and projection
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

"""
An efficient multi-query attention mechanism for mobile devices.

This layer implements Multi-Query Attention (MQA), a memory-efficient variant
of the standard Multi-Head Attention (MHA) mechanism. The primary motivation
behind MQA is to reduce the significant memory bandwidth overhead associated
with loading the Key (K) and Value (V) tensors in MHA, which is often a
bottleneck on mobile and edge hardware.

Architecturally, the key difference lies in how the K and V projections are
handled. In standard MHA, each attention head has its own independent linear
projections for Query (Q), Key, and Value. This means that for `h` heads,
`h` separate K and V tensors must be computed and stored. In MQA, this is
radically simplified:
-   **Multiple Queries:** Each head still has its own unique Q projection,
    allowing different heads to focus on different aspects of the input.
-   **Shared Key and Value:** A single, shared linear projection is used to
    create one K and one V tensor that are subsequently used by *all*
    attention heads.

The mathematical formulation remains largely the same as scaled dot-product
attention, but the K and V tensors are effectively broadcasted across the
head dimension during the attention score calculation. This design
dramatically reduces the size of the K and V tensors from `(batch, h, seq_len,
d_head)` to `(batch, 1, seq_len, d_head)`, leading to a significant reduction
in memory footprint and the I/O required to read them from memory during
computation.

Furthermore, this implementation introduces an optional spatial downsampling
step for the shared K and V tensors. When enabled, a strided depthwise
convolution is applied to the K and V feature maps before attention. This
further reduces the sequence length of the context that the queries attend
to, decreasing the `O(N^2)` complexity of the dot-product to `O(N*M)`, where `M`
is the downsampled sequence length. This allows the model to efficiently
aggregate information from a summarized, lower-resolution context, trading
some spatial granularity for a large gain in computational efficiency.

References:
    - Shazeer, 2019. Fast Transformer Decoding: One Write-Head is All You Need.
      (Introduced Multi-Query Attention)
    - Rombach et al., 2022. High-Resolution Image Synthesis with Latent
      Diffusion Models. (Used cross-attention with downsampled context)

"""

import keras
from keras import ops
from typing import Tuple, Optional, Any, Dict, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileMQA(keras.layers.Layer):
    """
    Mobile Multi-Query Attention (MQA) block.

    This block implements an efficient attention mechanism optimized for mobile accelerators.
    It uses shared keys and values across heads to reduce memory bandwidth requirements.

    The layer operates on 4D feature maps and applies multi-query attention where all heads
    share the same key and value projections, significantly reducing memory requirements
    compared to standard multi-head attention.

    Args:
        dim: Integer, dimension of the input and output tensors. Must be positive and
            divisible by num_heads.
        num_heads: Integer, number of attention heads. Must be positive. Defaults to 8.
        use_downsampling: Boolean, whether to use spatial downsampling for keys and values
            to further reduce computational cost. Defaults to False.
        kernel_initializer: String or initializer, initializer for the convolution kernels.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional regularizer for the convolution kernels.
            Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`
        where channels must equal dim.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, dim)`

    Attributes:
        q_proj: Dense layer for query projection.
        kv_proj: Dense layer for shared key-value projection.
        o_proj: Dense layer for output projection.
        downsample: Optional DepthwiseConv2D layer for spatial downsampling.
        lambda_param: Learnable parameter weight.

    Example:
        ```python
        # Basic usage
        layer = MobileMQA(dim=256, num_heads=8)

        # With downsampling for efficiency
        layer = MobileMQA(
            dim=512,
            num_heads=16,
            use_downsampling=True,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a model
        inputs = keras.Input(shape=(32, 32, 256))
        outputs = MobileMQA(dim=256, num_heads=8)(inputs)
        model = keras.Model(inputs, outputs)
        ```

    Raises:
        ValueError: If dim is not positive or not divisible by num_heads.
        ValueError: If num_heads is not positive.

    Note:
        This implementation is optimized for computer vision_heads tasks and operates on
        4D feature maps. The spatial dimensions are flattened for attention computation
        and then reshaped back to the original format.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_downsampling: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_downsampling = use_downsampling
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.scale = self.head_dim**-0.5

        # CREATE all sub-layers in __init__ (they are unbuilt)
        dense_config = {"kernel_initializer": self.kernel_initializer}

        # Only add regularizer if it's not None
        if self.kernel_regularizer is not None:
            dense_config["kernel_regularizer"] = self.kernel_regularizer

        self.q_proj = keras.layers.Dense(self.dim, name="q_proj", **dense_config)
        self.kv_proj = keras.layers.Dense(
            2 * self.head_dim, name="kv_proj", **dense_config
        )
        self.o_proj = keras.layers.Dense(self.dim, name="o_proj", **dense_config)

        if self.use_downsampling:
            # Configure downsample layer arguments
            downsample_args = {
                "kernel_size": 3,
                "strides": 2,
                "padding": "same",
                "depthwise_initializer": self.kernel_initializer,
                "name": "downsample",
            }

            # Only add regularizer if it's not None
            if self.kernel_regularizer is not None:
                downsample_args["depthwise_regularizer"] = self.kernel_regularizer

            self.downsample = keras.layers.DepthwiseConv2D(**downsample_args)
        else:
            self.downsample = None

        # Layer's own weights will be created in build()
        self.lambda_param = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer's own weights and explicitly build sub-layers.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Create layer's own weights using add_weight()
        self.lambda_param = self.add_weight(
            name="lambda", shape=(), initializer="ones", trainable=True
        )

        # Build sub-layers explicitly for robust serialization
        self.q_proj.build(input_shape)
        self.kv_proj.build(input_shape)
        self.o_proj.build(input_shape)

        if self.downsample is not None:
            # The input shape for the downsample layer is the output shape
            # of the kv_proj layer.
            kv_shape = list(input_shape)
            kv_shape[-1] = self.kv_proj.units
            self.downsample.build(tuple(kv_shape))


        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the MobileMQA block.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            attention_mask: Optional attention mask tensor.
            training: Boolean indicating training mode.

        Returns:
            Output tensor of shape (batch_size, height, width, dim).
        """
        input_shape = ops.shape(inputs)
        height = input_shape[1]
        width = input_shape[2]

        # Project to queries and key-values
        q = self.q_proj(inputs, training=training)
        kv = self.kv_proj(inputs, training=training)

        # Apply optional downsampling to key-values
        if self.downsample is not None:
            kv = self.downsample(kv, training=training)
            kv_shape = ops.shape(kv)
            kv_height, kv_width = kv_shape[1], kv_shape[2]
        else:
            kv_height, kv_width = height, width

        # Split kv into k and v using head_dim
        k = kv[..., : self.head_dim]
        v = kv[..., self.head_dim :]

        # Reshape for attention computation
        # q: (batch_size, num_heads, height * width, head_dim)
        q = ops.reshape(q, (-1, height * width, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))

        # k, v: (batch_size, 1, kv_height * kv_width, head_dim) - shared across heads
        k = ops.reshape(k, (-1, kv_height * kv_width, self.head_dim))
        v = ops.reshape(v, (-1, kv_height * kv_width, self.head_dim))
        k = ops.expand_dims(k, axis=1)
        v = ops.expand_dims(v, axis=1)

        # Compute scaled dot-product attention
        k_transposed = ops.transpose(k, axes=(0, 1, 3, 2))
        attn = ops.matmul(q, k_transposed) * self.scale
        attn = ops.nn.softmax(attn, axis=-1)

        # Apply attention to values
        out = ops.matmul(attn, v)

        # Reshape back to spatial format
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (-1, height, width, self.dim))

        # Final output projection
        attention_output = self.o_proj(out, training=training)

        # FIX: Add a scaled residual connection to use lambda_param and ensure gradients.
        return inputs + self.lambda_param * attention_output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple. Same as input shape for MobileMQA.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        Returns:
            Dictionary containing ALL __init__ parameters for reconstruction.
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "use_downsampling": self.use_downsampling,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

# ---------------------------------------------------------------------

"""
This module provides a `MobileMQA` layer, an implementation of Multi-Query Attention
that is highly optimized for efficiency on mobile and edge devices.

Standard Multi-Head Attention (MHA) can be memory-bandwidth intensive because it
requires separate, large Key (K) and Value (V) tensors to be projected and loaded for
each attention head. This layer mitigates that bottleneck by using the Multi-Query
Attention (MQA) strategy.

Core Concepts and Optimizations:

1.  **Shared Key and Value (The MQA Strategy):**
    -   In MQA, while each attention head gets its own unique Query (Q) projection,
        all heads *share a single Key and Value projection*.
    -   This is the defining feature of MQA. It dramatically reduces the memory
        footprint and I/O required for the K and V tensors, which is a major
        performance bottleneck on memory-constrained mobile accelerators. The layer
        implements this by having a single `kv_proj` that is shared across all
        `num_heads` query heads.

2.  **Optional Spatial Downsampling of Context:**
    -   The layer includes an optional `use_downsampling` mechanism that further
        reduces computational load.
    -   When enabled, it applies an efficient, strided `DepthwiseConv2D` to the
        Key and Value feature maps *before* the attention calculation.
    -   This reduces the spatial resolution of the context (K and V) that the
        queries attend to. The full-resolution queries can still attend to this
        coarser-grained context, effectively summarizing the key information from a
        larger receptive field at a lower cost.

3.  **Designed for 4D Image Tensors:**
    -   This implementation is tailored for computer vision tasks, operating directly
        on 4D feature maps of shape `(batch, height, width, channels)`. It internally
        flattens the spatial dimensions to perform attention and then reshapes the
        output back to the original 4D format.

Architectural Flow:

1.  An input feature map is passed through two parallel projections:
    a. `q_proj`: To create the multi-headed Query tensor.
    b. `kv_proj`: To create the *single*, shared Key and Value tensors.

2.  If `use_downsampling` is active, the combined KV tensor is spatially downsampled.

3.  The KV tensor is split into a single Key and a single Value.

4.  The Query tensor is reshaped to have `num_heads`. The Key and Value tensors are
    reshaped to have a head dimension of 1, ready for broadcasting.

5.  Standard scaled dot-product attention is performed. Due to broadcasting, all query
    heads attend to the same Key/Value pair.

6.  The output is reshaped back to its 4D spatial format and passed through a final
    output projection (`o_proj`) to produce the layer's result.
"""

import keras
from keras import ops
from typing import Tuple, Optional, Any, Dict, Union


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
        This implementation is optimized for computer vision tasks and operates on
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
            # FIX: Handle both list and tuple for input_shape during serialization.
            kv_shape = list(input_shape)
            kv_shape[-1] = 2 * self.head_dim
            self.downsample.build(tuple(kv_shape))

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self, x: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the MobileMQA block.

        Args:
            x: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating training mode.

        Returns:
            Output tensor of shape (batch_size, height, width, dim).
        """
        batch_size = ops.shape(x)[0]
        height = ops.shape(x)[1]
        width = ops.shape(x)[2]

        # Project to queries and key-values
        q = self.q_proj(x, training=training)
        kv = self.kv_proj(x, training=training)

        # Apply optional downsampling to key-values
        if self.downsample is not None:
            kv = self.downsample(kv, training=training)
            kv_height, kv_width = height // 2, width // 2
        else:
            kv_height, kv_width = height, width

        # Split kv into k and v using head_dim
        k = kv[..., : self.head_dim]
        v = kv[..., self.head_dim :]

        # Reshape for attention computation
        # q: (batch_size, num_heads, height * width, head_dim)
        q = ops.reshape(q, (batch_size, height * width, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))

        # k, v: (batch_size, 1, kv_height * kv_width, head_dim) - shared across heads
        k = ops.reshape(k, (batch_size, kv_height * kv_width, self.head_dim))
        v = ops.reshape(v, (batch_size, kv_height * kv_width, self.head_dim))
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
        out = ops.reshape(out, (batch_size, height, width, self.dim))

        # Final output projection
        attention_output = self.o_proj(out, training=training)

        # FIX: Add a scaled residual connection to use lambda_param and ensure gradients.
        return x + self.lambda_param * attention_output

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
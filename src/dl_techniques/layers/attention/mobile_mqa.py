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

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileMQA(keras.layers.Layer):
    """Mobile Multi-Query Attention (MQA) block.

    This block implements an efficient attention mechanism optimized for mobile accelerators.
    It uses shared keys and values across heads to reduce memory bandwidth requirements.

    Args:
        dim: Dimension of the input and output tensors
        num_heads: Number of attention heads
        use_downsampling: Whether to use spatial downsampling for keys and values
        kernel_initializer: Initializer for the convolution kernels
        kernel_regularizer: Regularizer for the convolution kernels
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            use_downsampling: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_downsampling = use_downsampling
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.scale = self.head_dim ** -0.5

        # Initialize layer attributes to None - will be built in build()
        self.q_proj = None
        self.kv_proj = None
        self.o_proj = None
        self.downsample = None
        self.lambda_param = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer weights and sublayers.

        Args:
            input_shape: Shape of the input tensor
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Layer configurations
        dense_config = {
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer
        }

        self.q_proj = keras.layers.Dense(self.dim, **dense_config)
        self.kv_proj = keras.layers.Dense(2 * self.dim, **dense_config)
        self.o_proj = keras.layers.Dense(self.dim, **dense_config)

        if self.use_downsampling:
            self.downsample = keras.layers.DepthwiseConv2D(
                3,
                strides=2,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )

        self.lambda_param = self.add_weight(
            "lambda",
            shape=(),
            initializer="ones",
            trainable=True
        )

        # Build sublayers explicitly
        self.q_proj.build(input_shape)
        self.kv_proj.build(input_shape)
        self.o_proj.build(input_shape)

        if self.use_downsampling:
            self.downsample.build(input_shape)

        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the MobileMQA block.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        batch_size = ops.shape(x)[0]
        height = ops.shape(x)[1]
        width = ops.shape(x)[2]

        q = self.q_proj(x)
        kv = self.kv_proj(x)

        if self.use_downsampling:
            kv = self.downsample(kv)
            kv_height, kv_width = height // 2, width // 2
        else:
            kv_height, kv_width = height, width

        # Split kv into k and v - using slice operation since tf.split isn't in keras.ops
        k = kv[..., :self.dim]
        v = kv[..., self.dim:]

        q = ops.reshape(q, (batch_size, height * width, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, kv_height * kv_width, 1, self.head_dim))
        v = ops.reshape(v, (batch_size, kv_height * kv_width, 1, self.head_dim))

        attn = ops.matmul(q, k, transpose_b=True) * self.scale
        attn = ops.nn.softmax(attn, axis=-1)

        out = ops.matmul(attn, v)
        out = ops.reshape(out, (batch_size, height, width, self.dim))

        return self.o_proj(out)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input

        Returns:
            Output shape
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "use_downsampling": self.use_downsampling,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------

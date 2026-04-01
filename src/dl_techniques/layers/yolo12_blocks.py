"""
YOLOv12 Core Building Blocks.

This module provides a collection of custom Keras layers and blocks that form the
fundamental components of the YOLOv12 object detection architecture. These blocks
are designed to be modular, efficient, and fully serializable following modern
Keras 3 best practices.

The key components include:

- **ConvBlock**: The standard base unit for all convolutions in the network,
  consisting of a Conv2D layer, followed by Batch Normalization and a SiLU
  (Sigmoid-weighted Linear Unit) activation function. This pattern promotes
  stable training and effective feature learning.

- **AreaAttention**: A specialized multi-head self-attention mechanism designed for
  2D feature maps. It can operate globally or on localized "areas" of the
  feature map, allowing the model to focus on relevant spatial regions with
  varying levels of granularity. This is crucial for capturing both local and
  global context.

- **AttentionBlock**: A transformer-style block that integrates the `AreaAttention`
  layer with a small feed-forward MLP (Multi-Layer Perceptron), both wrapped in
  residual connections. This allows the model to refine features by attending
  to different parts of the image and then processing them through a non-linear
  transformation.

- **Bottleneck**: A classic residual block used in many modern CNNs. It consists
  of two sequential `ConvBlock` layers with a shortcut (residual) connection
  that adds the input to the output. This helps to mitigate the vanishing
  gradient problem and allows for the construction of very deep networks.

- **C3k2Block**: A CSP (Cross-Stage Partial) inspired block that splits the input
  features into two paths. One path is processed through a series of `Bottleneck`
  layers, while the other remains unchanged. The two paths are then concatenated,
  fusing the processed and original features to enhance the gradient flow and
  learning capacity without a significant increase in computational cost.

- **A2C2fBlock**: An attention-enhanced feature fusion block inspired by ELAN
  (Efficient Layer Aggregation Network) principles. It processes an input through
  a series of `AttentionBlock` pairs, progressively concatenating the output of
  each stage. This creates a rich feature hierarchy, allowing the network to
  learn complex representations by combining features from different levels of
  abstraction.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvBlock(keras.layers.Layer):
    """
    Standard YOLOv12 convolution block: Conv2D, BatchNorm, SiLU.

    This block implements the base convolutional pattern used throughout YOLOv12,
    combining a 2D convolution with batch normalization and optional SiLU activation.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [B, H, W, C]            │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Conv2D(filters, kernel, stride)│
        │  BatchNormalization             │
        │  Optional SiLU Activation       │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Output [B, H', W', filters]   │
        └─────────────────────────────────┘

    :param filters: Number of output filters. Must be positive.
    :type filters: int
    :param kernel_size: Size of the convolution kernel. Defaults to 3.
    :type kernel_size: int
    :param strides: Stride of the convolution. Defaults to 1.
    :type strides: int
    :param padding: Padding mode (``'same'`` or ``'valid'``). Defaults to ``'same'``.
    :type padding: str
    :param groups: Number of groups for grouped convolution. Defaults to 1.
    :type groups: int
    :param activation: Whether to apply SiLU activation. Defaults to True.
    :type activation: bool
    :param use_bias: Whether to use bias in convolution. Defaults to False.
    :type use_bias: bool
    :param kernel_initializer: Weight initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional weight regularizer.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "same",
        groups: int = 1,
        activation: bool = True,
        use_bias: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if strides <= 0:
            raise ValueError(f"strides must be positive, got {strides}")
        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")
        if padding not in ["same", "valid"]:
            raise ValueError(f"padding must be 'same' or 'valid', got {padding}")

        # Store ALL configuration parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            groups=self.groups,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv"
        )

        self.bn = keras.layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.97,
            name="bn"
        )

        self.act = None
        if self.activation:
            self.act = keras.layers.Activation("silu", name="silu")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all its sub-layers.

            :param input_shape: Shape tuple of the input tensor.
            :type input_shape: tuple
        """
        # Build sub-layers in computational order for robust serialization
        self.conv.build(input_shape)

        # Compute intermediate shape for next layer
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.bn.build(conv_output_shape)

        if self.act is not None:
            self.act.build(conv_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the convolution block.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Boolean, whether the layer should behave in training mode.
            :type training: bool or None

            :return: Output tensor after convolution, batch normalization, and optional activation.
            :rtype: keras.KerasTensor
        """
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.act is not None:
            x = self.act(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

            :param input_shape: Shape tuple of input.
            :type input_shape: tuple

            :return: Output shape tuple.
            :rtype: tuple
        """
        return self.conv.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

            :return: Dictionary containing the layer configuration.
            :rtype: dict
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "groups": self.groups,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AreaAttention(keras.layers.Layer):
    """
    Area Attention mechanism for YOLOv12.

    This layer implements multi-head self-attention over 2D feature maps that can
    operate globally (``area=1``) or on localized spatial regions, enabling both
    local and global context modeling. Includes positional encoding via depthwise
    convolution and an output projection.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────┐
        │  Input [B, H, W, C]              │
        └──────────────┬────────────────────┘
                       ▼
        ┌───────────────────────────────────┐
        │  QK = ConvBlock 1x1 → [B,H,W,2D]│
        │  V  = ConvBlock 1x1 → [B,H,W,D] │
        │  PE = DW ConvBlock 5x5 on V      │
        └──────────────┬────────────────────┘
                       ▼
        ┌───────────────────────────────────┐
        │  Reshape to [B, areas, seq, D]   │
        │  Multi-head scaled dot-product   │
        │  attention within each area      │
        └──────────────┬────────────────────┘
                       ▼
        ┌───────────────────────────────────┐
        │  Add PE + projection ConvBlock   │
        │  Output [B, H, W, dim]           │
        └───────────────────────────────────┘

    :param dim: Number of feature dimensions. Must be divisible by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param area: Area size for attention grouping (1 for global). Defaults to 1.
    :type area: int
    :param kernel_initializer: Weight initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        area: int = 1,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.area = area
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.head_dim = dim // num_heads

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Query-Key projection
        self.qk_conv = ConvBlock(
            filters=self.dim * 2,
            kernel_size=1,
            activation=False,
            kernel_initializer=self.kernel_initializer,
            name="qk"
        )

        # Value projection
        self.v_conv = ConvBlock(
            filters=self.dim,
            kernel_size=1,
            activation=False,
            kernel_initializer=self.kernel_initializer,
            name="v"
        )

        # Position encoding
        self.pe_conv = ConvBlock(
            filters=self.dim,
            kernel_size=5,
            padding="same",
            groups=self.dim,
            activation=False,
            kernel_initializer=self.kernel_initializer,
            name="pe"
        )

        # Output projection
        self.proj_conv = ConvBlock(
            filters=self.dim,
            kernel_size=1,
            activation=False,
            kernel_initializer=self.kernel_initializer,
            name="proj"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the attention components and all sub-layers.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        """
        # Build sub-layers in computational order for robust serialization
        self.qk_conv.build(input_shape)
        self.v_conv.build(input_shape)

        # pe_conv and proj_conv operate on v/output tensors with channels=self.dim,
        # not on the raw input — build with the correct intermediate shape
        v_output_shape = self.v_conv.compute_output_shape(input_shape)
        self.pe_conv.build(v_output_shape)
        self.proj_conv.build(v_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through area attention.

            :param inputs: Input tensor of shape (batch_size, height, width, channels).
            :type inputs: keras.KerasTensor
            :param training: Boolean, whether in training mode.
            :type training: bool or None

            :return: Output tensor with attention applied.
            :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]
        height = ops.shape(inputs)[1]
        width = ops.shape(inputs)[2]
        channels = ops.shape(inputs)[3]

        # Generate query-key and value projections
        qk = self.qk_conv(inputs, training=training)
        v = self.v_conv(inputs, training=training)

        # Position encoding
        pe = self.pe_conv(v, training=training)

        # Reshape for attention computation
        seq_len = height * width
        qk = ops.reshape(qk, (batch_size, seq_len, self.dim * 2))
        v = ops.reshape(v, (batch_size, seq_len, self.dim))

        # Split query and key
        q, k = ops.split(qk, 2, axis=-1)

        # Apply area-based attention if specified
        if self.area > 1 and seq_len % self.area == 0:
            area_size = seq_len // self.area
            q = ops.reshape(q, (batch_size, self.area, area_size, self.dim))
            k = ops.reshape(k, (batch_size, self.area, area_size, self.dim))
            v = ops.reshape(v, (batch_size, self.area, area_size, self.dim))

            # Compute attention within each area
            attn_output = self._compute_attention(q, k, v)
            attn_output = ops.reshape(attn_output, (batch_size, seq_len, self.dim))
        else:
            # Global attention
            attn_output = self._compute_attention(
                ops.expand_dims(q, 1),
                ops.expand_dims(k, 1),
                ops.expand_dims(v, 1)
            )
            attn_output = ops.squeeze(attn_output, 1)

        # Reshape back to spatial dimensions
        attn_output = ops.reshape(attn_output, (batch_size, height, width, self.dim))

        # Add position encoding and apply final projection
        output = attn_output + pe
        return self.proj_conv(output, training=training)

    def _compute_attention(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute scaled dot-product attention.

            :param q: Query tensor.
            :type q: keras.KerasTensor
            :param k: Key tensor.
            :type k: keras.KerasTensor
            :param v: Value tensor.
            :type v: keras.KerasTensor

            :return: Attention output tensor.
            :rtype: keras.KerasTensor
        """
        # Reshape for multi-head attention
        batch_size = ops.shape(q)[0]
        num_areas = ops.shape(q)[1] if len(ops.shape(q)) > 3 else 1
        seq_len = ops.shape(q)[-2]

        q = ops.reshape(q, (batch_size, num_areas, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, num_areas, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, num_areas, seq_len, self.num_heads, self.head_dim))

        # Transpose for attention computation
        q = ops.transpose(q, (0, 1, 3, 2, 4))  # [batch, areas, heads, seq, head_dim]
        k = ops.transpose(k, (0, 1, 3, 2, 4))
        v = ops.transpose(v, (0, 1, 3, 2, 4))

        # Scaled dot-product attention
        scale = ops.cast(1.0 / ops.sqrt(ops.cast(self.head_dim, "float32")), q.dtype)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 2, 4, 3))) * scale
        attn_weights = ops.nn.softmax(scores, axis=-1)

        # Apply attention to values
        attn_output = ops.matmul(attn_weights, v)

        # Reshape and transpose back
        attn_output = ops.transpose(attn_output, (0, 1, 3, 2, 4))
        attn_output = ops.reshape(attn_output, (batch_size, num_areas, seq_len, self.dim))

        return attn_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

            :param input_shape: Shape tuple of input.
            :type input_shape: tuple

            :return: Output shape tuple.
            :rtype: tuple
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

            :return: Dictionary containing the layer configuration.
            :rtype: dict
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "area": self.area,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AttentionBlock(keras.layers.Layer):
    """
    Transformer-style block with Area Attention and MLP for YOLOv12.

    Combines area attention with a two-layer feed-forward network (1x1
    convolutions), both wrapped in residual connections. This allows the
    model to refine features through attention and non-linear transformation.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────┐
        │  Input [B, H, W, C]           │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  AreaAttention + Residual      │
        │  x = input + attn(input)       │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  MLP (Conv1x1 expand + shrink) │
        │  + Residual                    │
        │  x = x + mlp2(mlp1(x))        │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  Output [B, H, W, dim]        │
        └────────────────────────────────┘

    :param dim: Number of feature dimensions. Must be positive.
    :type dim: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param mlp_ratio: Expansion ratio for MLP hidden dimension. Defaults to 1.2.
    :type mlp_ratio: float
    :param area: Area size for attention grouping. Defaults to 1.
    :type area: int
    :param kernel_initializer: Weight initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 1.2,
        area: int = 1,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.area = area
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Attention layer
        self.attn = AreaAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            area=self.area,
            kernel_initializer=self.kernel_initializer,
            name="attn"
        )

        # MLP layers
        self.mlp1 = ConvBlock(
            filters=self.mlp_hidden_dim,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="mlp1"
        )

        self.mlp2 = ConvBlock(
            filters=self.dim,
            kernel_size=1,
            activation=False,
            kernel_initializer=self.kernel_initializer,
            name="mlp2"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the attention block components and all sub-layers.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        """
        # Build sub-layers in computational order for robust serialization
        self.attn.build(input_shape)
        self.mlp1.build(input_shape)

        # Compute intermediate shape for mlp2
        mlp1_output_shape = self.mlp1.compute_output_shape(input_shape)
        self.mlp2.build(mlp1_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through attention block.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Boolean, whether in training mode.
            :type training: bool or None

            :return: Output tensor after attention and MLP with residual connections.
            :rtype: keras.KerasTensor
        """
        # Attention with residual connection
        attn_out = self.attn(inputs, training=training)
        x = inputs + attn_out

        # MLP with residual connection
        mlp_out = self.mlp1(x, training=training)
        mlp_out = self.mlp2(mlp_out, training=training)
        x = x + mlp_out

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

            :param input_shape: Shape tuple of input.
            :type input_shape: tuple

            :return: Output shape tuple.
            :rtype: tuple
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

            :return: Dictionary containing the layer configuration.
            :rtype: dict
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "area": self.area,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Bottleneck(keras.layers.Layer):
    """
    Standard bottleneck block with optional residual connection for YOLOv12.

    Two sequential 3x3 ConvBlock layers with an optional shortcut connection
    that adds the input to the output when channels match.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │  Input [B, H, W, C]         │
        └──────┬───────────────┬───────┘
               ▼               │ (shortcut if C=filters)
        ┌──────────────┐       │
        │  ConvBlock3x3│       │
        ├──────────────┤       │
        │  ConvBlock3x3│       │
        └──────┬───────┘       │
               ▼               ▼
        ┌──────────────────────────────┐
        │  Add (if shortcut=True)      │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Output [B, H, W, filters]  │
        └──────────────────────────────┘

    :param filters: Number of output filters. Must be positive.
    :type filters: int
    :param shortcut: Whether to use residual connection. Defaults to True.
    :type shortcut: bool
    :param kernel_initializer: Weight initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        filters: int,
        shortcut: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")

        # Store ALL configuration parameters
        self.filters = filters
        self.shortcut = shortcut
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.cv1 = ConvBlock(
            filters=self.filters,
            kernel_size=3,
            kernel_initializer=self.kernel_initializer,
            name="cv1"
        )

        self.cv2 = ConvBlock(
            filters=self.filters,
            kernel_size=3,
            activation=False,
            kernel_initializer=self.kernel_initializer,
            name="cv2"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the bottleneck components and all sub-layers.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        """
        # Build sub-layers in computational order for robust serialization
        self.cv1.build(input_shape)

        # Compute intermediate shape for cv2
        cv1_output_shape = self.cv1.compute_output_shape(input_shape)
        self.cv2.build(cv1_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through bottleneck.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Boolean, whether in training mode.
            :type training: bool or None

            :return: Output tensor with optional residual connection.
            :rtype: keras.KerasTensor
        """
        x = self.cv1(inputs, training=training)
        x = self.cv2(x, training=training)

        if self.shortcut and ops.shape(inputs)[-1] == self.filters:
            x = inputs + x

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

            :param input_shape: Shape tuple of input.
            :type input_shape: tuple

            :return: Output shape tuple.
            :rtype: tuple
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.filters
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

            :return: Dictionary containing the layer configuration.
            :rtype: dict
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "shortcut": self.shortcut,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class C3k2Block(keras.layers.Layer):
    """
    CSP-like block with dual paths and Bottleneck layers for YOLOv12.

    Splits the input into two paths via 1x1 convolutions. One path is
    processed through ``n`` Bottleneck layers, while the other remains
    unchanged. Both paths are concatenated and fused through a final 1x1
    convolution.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, H, W, C]             │
        └──────┬───────────────┬───────────┘
               ▼               ▼
        ┌────────────┐  ┌────────────┐
        │  cv1 (1x1) │  │  cv2 (1x1) │
        ├────────────┤  └──────┬─────┘
        │  Bottleneck│         │
        │  x n       │         │
        └──────┬─────┘         │
               ▼               ▼
        ┌──────────────────────────────────┐
        │  Concatenate along channels      │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  cv3 (1x1) → Output [B,H,W,F]  │
        └──────────────────────────────────┘

    :param filters: Number of output filters. Must be positive.
    :type filters: int
    :param n: Number of bottleneck layers. Must be non-negative. Defaults to 1.
    :type n: int
    :param shortcut: Whether to use shortcuts in bottlenecks. Defaults to True.
    :type shortcut: bool
    :param kernel_initializer: Weight initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        filters: int,
        n: int = 1,
        shortcut: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")

        # Store ALL configuration parameters
        self.filters = filters
        self.n = n
        self.shortcut = shortcut
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.hidden_filters = filters // 2

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.cv1 = ConvBlock(
            filters=self.hidden_filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="cv1"
        )

        self.cv2 = ConvBlock(
            filters=self.hidden_filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="cv2"
        )

        # Create bottleneck layers - store as list attributes for proper serialization
        self.bottlenecks = []
        for i in range(self.n):
            bottleneck = Bottleneck(
                filters=self.hidden_filters,
                shortcut=self.shortcut,
                kernel_initializer=self.kernel_initializer,
                name=f"bottleneck_{i}"
            )
            self.bottlenecks.append(bottleneck)

        self.cv3 = ConvBlock(
            filters=self.filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="cv3"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the C3k2 block components and all sub-layers.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        """
        # Build sub-layers in computational order for robust serialization
        self.cv1.build(input_shape)
        self.cv2.build(input_shape)

        # Compute intermediate shape for bottlenecks
        cv1_output_shape = self.cv1.compute_output_shape(input_shape)

        # Build bottleneck layers sequentially
        current_shape = cv1_output_shape
        for bottleneck in self.bottlenecks:
            bottleneck.build(current_shape)
            current_shape = bottleneck.compute_output_shape(current_shape)

        # Compute shape for final convolution (concatenation of two paths)
        cv2_output_shape = self.cv2.compute_output_shape(input_shape)
        concat_shape = list(cv2_output_shape)
        concat_shape[-1] = current_shape[-1] + cv2_output_shape[-1]  # Concatenate channel dimension
        self.cv3.build(tuple(concat_shape))

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through C3k2 block.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Boolean, whether in training mode.
            :type training: bool or None

            :return: Output tensor after CSP processing.
            :rtype: keras.KerasTensor
        """
        y1 = self.cv1(inputs, training=training)
        y2 = self.cv2(inputs, training=training)

        # Apply bottleneck layers sequentially
        for bottleneck in self.bottlenecks:
            y1 = bottleneck(y1, training=training)

        # Concatenate and apply final convolution
        y = ops.concatenate([y1, y2], axis=-1)
        return self.cv3(y, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

            :param input_shape: Shape tuple of input.
            :type input_shape: tuple

            :return: Output shape tuple.
            :rtype: tuple
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.filters
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

            :return: Dictionary containing the layer configuration.
            :rtype: dict
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "n": self.n,
            "shortcut": self.shortcut,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class A2C2fBlock(keras.layers.Layer):
    """
    Attention-enhanced ELAN block with progressive feature extraction for YOLOv12.

    Processes input through a 1x1 convolution, then through ``n`` pairs of
    AttentionBlocks, progressively concatenating outputs from each stage to
    build a rich feature hierarchy. A final 1x1 convolution fuses all features.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, H, W, C]             │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  cv1 (1x1) → y0 [B,H,W,F/2]   │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  AttentionBlock pair 1 → y1     │
        │  AttentionBlock pair 2 → y2     │
        │  ...                             │
        │  AttentionBlock pair n → yn     │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Concat [y0, y1, ..., yn]       │
        │  cv2 (1x1) → Output [B,H,W,F]  │
        └──────────────────────────────────┘

    :param filters: Number of output filters. Must be positive.
    :type filters: int
    :param n: Number of attention block pairs. Must be non-negative. Defaults to 1.
    :type n: int
    :param area: Area size for attention mechanism. Defaults to 1.
    :type area: int
    :param kernel_initializer: Weight initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        filters: int,
        n: int = 1,
        area: int = 1,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")

        # Store ALL configuration parameters
        self.filters = filters
        self.n = n
        self.area = area
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.hidden_filters = filters // 2

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.cv1 = ConvBlock(
            filters=self.hidden_filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="cv1"
        )

        # Create attention blocks as individual list attributes for proper serialization
        # Each pair has two attention blocks: first and second
        self.attention_first_blocks = []
        self.attention_second_blocks = []

        for i in range(self.n):
            attn_block_1 = AttentionBlock(
                dim=self.hidden_filters,
                num_heads=max(1, self.hidden_filters // 32),
                area=self.area,
                kernel_initializer=self.kernel_initializer,
                name=f"attn_{i}_1"
            )
            attn_block_2 = AttentionBlock(
                dim=self.hidden_filters,
                num_heads=max(1, self.hidden_filters // 32),
                area=self.area,
                kernel_initializer=self.kernel_initializer,
                name=f"attn_{i}_2"
            )
            self.attention_first_blocks.append(attn_block_1)
            self.attention_second_blocks.append(attn_block_2)

        self.cv2 = ConvBlock(
            filters=self.filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="cv2"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the A2C2f block components and all sub-layers.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        """
        # Build sub-layers in computational order for robust serialization
        self.cv1.build(input_shape)

        # Compute intermediate shape for attention blocks
        cv1_output_shape = self.cv1.compute_output_shape(input_shape)

        # Build attention blocks sequentially
        current_shape = cv1_output_shape
        for i in range(self.n):
            # Build first attention block
            self.attention_first_blocks[i].build(current_shape)
            current_shape = self.attention_first_blocks[i].compute_output_shape(current_shape)

            # Build second attention block
            self.attention_second_blocks[i].build(current_shape)
            current_shape = self.attention_second_blocks[i].compute_output_shape(current_shape)

        # Compute shape for final convolution (concatenation of all features)
        # We have n+1 feature tensors (initial + n pairs)
        concat_channels = self.hidden_filters * (self.n + 1)
        concat_shape = list(cv1_output_shape)
        concat_shape[-1] = concat_channels
        self.cv2.build(tuple(concat_shape))

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through A2C2f block.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Boolean, whether in training mode.
            :type training: bool or None

            :return: Output tensor after progressive feature extraction and fusion.
            :rtype: keras.KerasTensor
        """
        y = self.cv1(inputs, training=training)

        # Collect features progressively
        features = [y]

        for i in range(self.n):
            # Apply two attention blocks sequentially
            y = self.attention_first_blocks[i](features[-1], training=training)
            y = self.attention_second_blocks[i](y, training=training)
            features.append(y)

        # Concatenate all features and apply final convolution
        y = ops.concatenate(features, axis=-1)
        return self.cv2(y, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

            :param input_shape: Shape tuple of input.
            :type input_shape: tuple

            :return: Output shape tuple.
            :rtype: tuple
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.filters
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

            :return: Dictionary containing the layer configuration.
            :rtype: dict
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "n": self.n,
            "area": self.area,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config

# ---------------------------------------------------------------------

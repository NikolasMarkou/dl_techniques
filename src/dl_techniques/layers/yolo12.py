"""
YOLOv12 Core Building Blocks.

This module provides a collection of custom Keras layers and blocks that form the
fundamental components of the YOLOv12 object detection architecture. These blocks
are designed to be modular, efficient, and fully serializable.

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
- learning capacity without a significant increase in computational cost.

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
    """Standard Convolution Block with BatchNorm and SiLU activation.

    This block implements the standard pattern used throughout YOLOv12:
    Conv2D -> BatchNormalization -> SiLU Activation

    Args:
        filters: Number of output filters.
        kernel_size: Size of the convolution kernel.
        strides: Stride of the convolution.
        padding: Padding mode ('same' or 'valid').
        groups: Number of groups for grouped convolution.
        activation: Whether to apply SiLU activation.
        use_bias: Whether to use bias in convolution.
        kernel_initializer: Weight initializer.
        kernel_regularizer: Weight regularizer.
        name: Layer name.
        **kwargs: Additional keyword arguments.
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
            name: Optional[str] = None,
            **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Layers will be created in build()
        self.conv = None
        self.bn = None
        self.act = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer components.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        super().build(input_shape)

        # Store for serialization
        self._build_input_shape = input_shape

        self.conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            groups=self.groups,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{self.name}_conv" if self.name else None
        )

        self.bn = keras.layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.97,
            name=f"{self.name}_bn" if self.name else None
        )

        if self.activation:
            self.act = keras.layers.Activation(
                "silu",
                name=f"{self.name}_silu" if self.name else None
            )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the convolution block.

        Args:
            inputs: Input tensor.
            training: Whether the layer should behave in training mode.

        Returns:
            Output tensor after convolution, batch normalization, and optional activation.
        """
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.activation:
            x = self.act(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AreaAttention(keras.layers.Layer):
    """Area Attention mechanism for YOLOv12.

    This implements the area-attention mechanism that allows the model
    to focus on different spatial regions with varying granularities.

    Args:
        dim: Number of feature dimensions.
        num_heads: Number of attention heads.
        area: Area size for attention grouping (1 for global attention).
        kernel_initializer: Weight initializer.
        name: Layer name.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            area: int = 1,
            kernel_initializer: str = "he_normal",
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.dim = dim
        self.num_heads = num_heads
        self.area = area
        self.kernel_initializer = kernel_initializer

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = dim // num_heads

        # Layers will be created in build()
        self.qk_conv = None
        self.v_conv = None
        self.pe_conv = None
        self.proj_conv = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the attention components."""
        super().build(input_shape)

        # Query-Key projection
        self.qk_conv = ConvBlock(
            filters=self.dim * 2,
            kernel_size=1,
            activation=False,
            name=f"{self.name}_qk"
        )

        # Value projection
        self.v_conv = ConvBlock(
            filters=self.dim,
            kernel_size=1,
            activation=False,
            name=f"{self.name}_v"
        )

        # Position encoding
        self.pe_conv = ConvBlock(
            filters=self.dim,
            kernel_size=5,
            padding="same",
            groups=self.dim,
            activation=False,
            name=f"{self.name}_pe"
        )

        # Output projection
        self.proj_conv = ConvBlock(
            filters=self.dim,
            kernel_size=1,
            activation=False,
            name=f"{self.name}_proj"
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through area attention."""
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
        """Compute scaled dot-product attention."""
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

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "area": self.area,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AttentionBlock(keras.layers.Layer):
    """Attention Block with Area Attention and MLP.

    This block combines area attention with a feed-forward network
    using residual connections.

    Args:
        dim: Number of feature dimensions.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for MLP.
        area: Area size for attention grouping.
        kernel_initializer: Weight initializer.
        name: Layer name.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 1.2,
            area: int = 1,
            kernel_initializer: str = "he_normal",
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.area = area
        self.kernel_initializer = kernel_initializer

        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # Layers will be created in build()
        self.attn = None
        self.mlp1 = None
        self.mlp2 = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the attention block components."""
        super().build(input_shape)

        # Attention layer
        self.attn = AreaAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            area=self.area,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_attn"
        )

        # MLP layers
        self.mlp1 = ConvBlock(
            filters=self.mlp_hidden_dim,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_mlp1"
        )

        self.mlp2 = ConvBlock(
            filters=self.dim,
            kernel_size=1,
            activation=False,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_mlp2"
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through attention block."""
        # Attention with residual connection
        attn_out = self.attn(inputs, training=training)
        x = inputs + attn_out

        # MLP with residual connection
        mlp_out = self.mlp1(x, training=training)
        mlp_out = self.mlp2(mlp_out, training=training)
        x = x + mlp_out

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "area": self.area,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Bottleneck(keras.layers.Layer):
    """Standard Bottleneck block with optional residual connection.

    Args:
        filters: Number of output filters.
        shortcut: Whether to use residual connection.
        kernel_initializer: Weight initializer.
        name: Layer name.
    """

    def __init__(
            self,
            filters: int,
            shortcut: bool = True,
            kernel_initializer: str = "he_normal",
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.filters = filters
        self.shortcut = shortcut
        self.kernel_initializer = kernel_initializer

        # Layers will be created in build()
        self.cv1 = None
        self.cv2 = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the bottleneck components."""
        super().build(input_shape)

        self.cv1 = ConvBlock(
            filters=self.filters,
            kernel_size=3,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_cv1"
        )

        self.cv2 = ConvBlock(
            filters=self.filters,
            kernel_size=3,
            activation=False,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_cv2"
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through bottleneck."""
        x = self.cv1(inputs, training=training)
        x = self.cv2(x, training=training)

        if self.shortcut and ops.shape(inputs)[-1] == self.filters:
            x = inputs + x

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "shortcut": self.shortcut,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class C3k2Block(keras.layers.Layer):
    """CSP-like block with 2 convolutions and Bottleneck layers.

    Args:
        filters: Number of output filters.
        n: Number of bottleneck layers.
        shortcut: Whether to use shortcuts in bottlenecks.
        kernel_initializer: Weight initializer.
        name: Layer name.
    """

    def __init__(
            self,
            filters: int,
            n: int = 1,
            shortcut: bool = True,
            kernel_initializer: str = "he_normal",
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.filters = filters
        self.n = n
        self.shortcut = shortcut
        self.kernel_initializer = kernel_initializer

        self.hidden_filters = filters // 2

        # Layers will be created in build()
        self.cv1 = None
        self.cv2 = None
        self.bottlenecks = None
        self.cv3 = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the C3k2 block components."""
        super().build(input_shape)

        self.cv1 = ConvBlock(
            filters=self.hidden_filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_cv1"
        )

        self.cv2 = ConvBlock(
            filters=self.hidden_filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_cv2"
        )

        # Create bottleneck layers
        self.bottlenecks = []
        for i in range(self.n):
            bottleneck = Bottleneck(
                filters=self.hidden_filters,
                shortcut=self.shortcut,
                kernel_initializer=self.kernel_initializer,
                name=f"{self.name}_bottleneck_{i}"
            )
            self.bottlenecks.append(bottleneck)

        self.cv3 = ConvBlock(
            filters=self.filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_cv3"
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through C3k2 block."""
        y1 = self.cv1(inputs, training=training)
        y2 = self.cv2(inputs, training=training)

        # Apply bottleneck layers sequentially
        for bottleneck in self.bottlenecks:
            y1 = bottleneck(y1, training=training)

        # Concatenate and apply final convolution
        y = ops.concatenate([y1, y2], axis=-1)
        return self.cv3(y, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "n": self.n,
            "shortcut": self.shortcut,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class A2C2fBlock(keras.layers.Layer):
    """Attention-enhanced R-ELAN block with progressive feature extraction.

    Args:
        filters: Number of output filters.
        n: Number of attention block pairs.
        area: Area size for attention mechanism.
        kernel_initializer: Weight initializer.
        name: Layer name.
    """

    def __init__(
            self,
            filters: int,
            n: int = 1,
            area: int = 1,
            kernel_initializer: str = "he_normal",
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.filters = filters
        self.n = n
        self.area = area
        self.kernel_initializer = kernel_initializer

        self.hidden_filters = filters // 2

        # Layers will be created in build()
        self.cv1 = None
        self.attention_blocks = None
        self.cv2 = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the A2C2f block components."""
        super().build(input_shape)

        self.cv1 = ConvBlock(
            filters=self.hidden_filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_cv1"
        )

        # Create attention blocks
        self.attention_blocks = []
        for i in range(self.n):
            # Each iteration has two attention blocks
            attn_block_1 = AttentionBlock(
                dim=self.hidden_filters,
                num_heads=max(1, self.hidden_filters // 32),
                area=self.area,
                kernel_initializer=self.kernel_initializer,
                name=f"{self.name}_attn_{i}_1"
            )
            attn_block_2 = AttentionBlock(
                dim=self.hidden_filters,
                num_heads=max(1, self.hidden_filters // 32),
                area=self.area,
                kernel_initializer=self.kernel_initializer,
                name=f"{self.name}_attn_{i}_2"
            )
            self.attention_blocks.append([attn_block_1, attn_block_2])

        self.cv2 = ConvBlock(
            filters=self.filters,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_cv2"
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through A2C2f block."""
        y = self.cv1(inputs, training=training)

        # Collect features progressively
        features = [y]

        for attn_blocks in self.attention_blocks:
            # Apply two attention blocks sequentially
            y = attn_blocks[0](features[-1], training=training)
            y = attn_blocks[1](y, training=training)
            features.append(y)

        # Concatenate all features and apply final convolution
        y = ops.concatenate(features, axis=-1)
        return self.cv2(y, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "n": self.n,
            "area": self.area,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

# ---------------------------------------------------------------------

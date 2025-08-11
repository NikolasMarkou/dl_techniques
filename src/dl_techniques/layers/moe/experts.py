"""
Expert network implementations for Mixture of Experts (MoE) models.

This module provides various expert network architectures that can be used
in MoE layers, including Feed-Forward Networks (FFN), Attention experts,
and Convolutional experts.
"""

import keras
from keras import ops
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Any, Dict
from keras import layers, initializers, regularizers, activations

class BaseExpert(layers.Layer, ABC):
    """
    Abstract base class for MoE expert networks.

    This class defines the interface that all expert implementations must follow,
    ensuring consistency across different expert types and enabling polymorphic
    usage in MoE layers.

    Args:
        name: Name for the expert layer.
        **kwargs: Additional keyword arguments for the base Layer class.

    Methods:
        call: Abstract method for forward computation.
        compute_output_shape: Abstract method for output shape computation.
    """

    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize the base expert layer."""
        super().__init__(name=name, **kwargs)
        self._built_input_shape = None

    @abstractmethod
    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward computation for the expert.

        Args:
            inputs: Input tensor.
            training: Whether the layer is in training mode.

        Returns:
            Expert output tensor.
        """
        pass

    @abstractmethod
    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the expert.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Shape of the output tensor.
        """
        pass

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._built_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the expert from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


class FFNExpert(BaseExpert):
    """
    Feed-Forward Network expert for MoE layers.

    This expert implements a standard FFN with configurable intermediate size,
    activation function, and regularization. It's the most common expert type
    in Transformer-based MoE models.

    Args:
        hidden_dim: Output dimension of the expert.
        intermediate_size: Intermediate layer dimension.
        activation: Activation function to use.
        dropout_rate: Dropout probability for regularization.
        use_bias: Whether to use bias terms.
        kernel_initializer: Weight initialization strategy.
        bias_initializer: Bias initialization strategy.
        kernel_regularizer: Regularization for weights.
        bias_regularizer: Regularization for biases.
        **kwargs: Additional keyword arguments.

    Example:
        ```python
        expert = FFNExpert(
            hidden_dim=768,
            intermediate_size=3072,
            activation='gelu',
            dropout_rate=0.1
        )
        ```
    """

    def __init__(
            self,
            hidden_dim: int,
            intermediate_size: int = 3072,
            activation: Union[str, callable] = 'gelu',
            dropout_rate: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the FFN expert."""
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Sublayers initialized in build()
        self.dense1 = None
        self.dense2 = None
        self.activation_fn = None
        self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the FFN expert layers."""
        self._built_input_shape = input_shape

        # First dense layer (expansion)
        self.dense1 = layers.Dense(
            units=self.intermediate_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='dense1'
        )

        # Second dense layer (projection)
        self.dense2 = layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='dense2'
        )

        # Activation function
        self.activation_fn = activations.get(self.activation)

        # Dropout layer
        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(rate=self.dropout_rate, name='dropout')

        # Build sublayers
        self.dense1.build(input_shape)
        intermediate_shape = self.dense1.compute_output_shape(input_shape)
        self.dense2.build(intermediate_shape)

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the FFN expert."""
        # First transformation with activation
        x = self.dense1(inputs)
        x = self.activation_fn(x)

        # Apply dropout if configured
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Second transformation (projection)
        outputs = self.dense2(x)

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape of the FFN expert."""
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'intermediate_size': self.intermediate_size,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer)
        })
        return config


class AttentionExpert(BaseExpert):
    """
    Multi-Head Attention expert for MoE layers.

    This expert implements a multi-head attention mechanism that can be used
    as an expert in Mixture-of-Attention (MoA) architectures, allowing for
    specialized attention patterns.

    Args:
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        dropout_rate: Dropout probability.
        use_bias: Whether to use bias terms.
        kernel_initializer: Weight initialization strategy.
        bias_initializer: Bias initialization strategy.
        **kwargs: Additional keyword arguments.

    Example:
        ```python
        expert = AttentionExpert(
            hidden_dim=768,
            num_heads=12,
            head_dim=64,
            dropout_rate=0.1
        )
        ```
    """

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 8,
            head_dim: Optional[int] = None,
            dropout_rate: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            **kwargs: Any
    ) -> None:
        """Initialize the attention expert."""
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Derived parameters
        self.all_head_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        # Sublayers initialized in build()
        self.query_dense = None
        self.key_dense = None
        self.value_dense = None
        self.output_dense = None
        self.attention_dropout = None
        self.output_dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the attention expert layers."""
        self._built_input_shape = input_shape

        # Query, Key, Value projections
        self.query_dense = layers.Dense(
            units=self.all_head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='query'
        )

        self.key_dense = layers.Dense(
            units=self.all_head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='key'
        )

        self.value_dense = layers.Dense(
            units=self.all_head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='value'
        )

        # Output projection
        self.output_dense = layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='output'
        )

        # Dropout layers
        if self.dropout_rate > 0:
            self.attention_dropout = layers.Dropout(rate=self.dropout_rate, name='attention_dropout')
            self.output_dropout = layers.Dropout(rate=self.dropout_rate, name='output_dropout')

        # Build sublayers
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)
        self.output_dense.build(input_shape[:-1] + (self.all_head_dim,))

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the attention expert."""
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Compute Q, K, V
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Reshape for multi-head attention
        query = ops.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ops.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dim))
        value = ops.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose for attention computation
        query = ops.transpose(query, (0, 2, 1, 3))  # [batch, heads, seq_len, head_dim]
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.transpose(value, (0, 2, 1, 3))

        # Scaled dot-product attention
        attention_scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * self.scale
        attention_weights = ops.softmax(attention_scores, axis=-1)

        # Apply attention dropout
        if self.attention_dropout is not None:
            attention_weights = self.attention_dropout(attention_weights, training=training)

        # Apply attention to values
        attention_output = ops.matmul(attention_weights, value)

        # Transpose back and reshape
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = ops.reshape(attention_output, (batch_size, seq_len, self.all_head_dim))

        # Final output projection
        outputs = self.output_dense(attention_output)

        # Apply output dropout
        if self.output_dropout is not None:
            outputs = self.output_dropout(outputs, training=training)

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape of the attention expert."""
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer)
        })
        return config


class Conv2DExpert(BaseExpert):
    """
    Convolutional expert for vision MoE models.

    This expert implements convolutional operations specialized for different
    visual patterns or features, enabling spatial specialization in vision
    transformer architectures.

    Args:
        filters: Number of convolutional filters.
        kernel_size: Size of the convolutional kernel.
        strides: Stride configuration for convolution.
        padding: Padding strategy for convolution.
        activation: Activation function to use.
        use_bias: Whether to use bias terms.
        kernel_initializer: Weight initialization strategy.
        bias_initializer: Bias initialization strategy.
        **kwargs: Additional keyword arguments.

    Example:
        ```python
        expert = Conv2DExpert(
            filters=256,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
        ```
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            strides: Union[int, Tuple[int, int]] = 1,
            padding: str = 'same',
            activation: Union[str, callable] = 'relu',
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            **kwargs: Any
    ) -> None:
        """Initialize the Conv2D expert."""
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Sublayers initialized in build()
        self.conv_layer = None
        self.activation_fn = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the Conv2D expert layers."""
        self._built_input_shape = input_shape

        # Convolutional layer
        self.conv_layer = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='conv2d'
        )

        # Activation function
        self.activation_fn = activations.get(self.activation)

        # Build sublayers
        self.conv_layer.build(input_shape)

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the Conv2D expert."""
        # Apply convolution
        x = self.conv_layer(inputs)

        # Apply activation
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape of the Conv2D expert."""
        return self.conv_layer.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer)
        })
        return config


def create_expert(expert_type: str, **kwargs) -> BaseExpert:
    """
    Factory function to create expert networks.

    Args:
        expert_type: Type of expert to create ('ffn', 'attention', 'conv2d').
        **kwargs: Configuration parameters for the expert.

    Returns:
        Configured expert network.

    Raises:
        ValueError: If expert_type is not supported.

    Example:
        ```python
        # Create FFN expert
        expert = create_expert('ffn', hidden_dim=768, intermediate_size=3072)

        # Create attention expert
        expert = create_expert('attention', hidden_dim=768, num_heads=12)
        ```
    """
    if expert_type == 'ffn':
        return FFNExpert(**kwargs)
    elif expert_type == 'attention':
        return AttentionExpert(**kwargs)
    elif expert_type == 'conv2d':
        return Conv2DExpert(**kwargs)
    else:
        raise ValueError(f"Unsupported expert type: {expert_type}")
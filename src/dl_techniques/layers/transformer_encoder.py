import keras
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .norms.rms_norm import RMSNorm
from .norms.band_rms import BandRMS

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TransformerEncoderLayer(keras.layers.Layer):
    """
    Generic transformer encoder layer with configurable normalization.

    This layer implements a standard transformer encoder block with:
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections
    - Configurable normalization

    Args:
        hidden_size: Integer, hidden size of the layer.
        num_heads: Integer, number of attention heads.
        intermediate_size: Integer, size of the intermediate (feed-forward) layer.
        normalization_type: String, type of normalization to use.
            Available options depend on installed dependencies:
            - 'layer_norm': Standard layer normalization (always available)
            - 'batch_norm': Batch normalization (always available)
            - 'rms_norm': Root Mean Square normalization (if RMSNorm is available)
            - 'band_rms': Band-constrained RMS normalization (if BandRMS is available)
            Defaults to 'layer_norm'.
        dropout_rate: Float, dropout rate. Defaults to 0.1.
        attention_dropout_rate: Float, attention-specific dropout rate. Defaults to 0.1.
        activation: String or callable, activation function for feed-forward network.
            Defaults to 'gelu'.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: String or initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`

    Example:
        >>> inputs = keras.Input(shape=(128, 768))
        >>> layer = TransformerEncoderLayer(
        ...     hidden_size=768,
        ...     num_heads=12,
        ...     intermediate_size=3072,
        ...     normalization_type='layer_norm'
        ... )
        >>> outputs = layer(inputs)
        >>> print(outputs.shape)
        (None, 128, 768)
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            normalization_type: str = 'layer_norm',
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            activation: Union[str, callable] = 'gelu',
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}")

        valid_norm_types = ['layer_norm', 'rms_norm', 'batch_norm', 'band_rms']
        if normalization_type not in valid_norm_types:
            raise ValueError(f"normalization_type must be one of {valid_norm_types}, got {normalization_type}")

        # Store configuration parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.normalization_type = normalization_type
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Initialize layers to None - will be created in build()
        self.attention = None
        self.attention_norm = None
        self.intermediate = None
        self.output_dense = None
        self.output_norm = None
        self.dropout = None
        self.attention_dropout = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def _create_normalization_layer(self, name: str) -> keras.layers.Layer:
        """Create a normalization layer based on the specified type.

        Args:
            name: Name for the normalization layer.

        Returns:
            A normalization layer instance.
        """
        if self.normalization_type == 'layer_norm':
            return keras.layers.LayerNormalization(
                epsilon=1e-12,
                name=name
            )
        elif self.normalization_type == 'rms_norm':
            return RMSNorm(name=name)
        elif self.normalization_type == 'band_rms':
            return BandRMS(max_band_width=0.1, name=name)
        elif self.normalization_type == 'batch_norm':
            return keras.layers.BatchNormalization(
                epsilon=1e-12,
                name=name
            )
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the transformer layer components.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {len(input_shape)}D: {input_shape}")

        if input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input feature dimension ({input_shape[-1]}) must match hidden_size ({self.hidden_size})"
            )

        # Multi-head attention
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_size // self.num_heads,
            dropout=self.attention_dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='attention'
        )

        # Attention layer normalization
        self.attention_norm = self._create_normalization_layer('attention_norm')

        # Feed-forward network
        self.intermediate = keras.layers.Dense(
            self.intermediate_size,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='intermediate'
        )

        self.output_dense = keras.layers.Dense(
            self.hidden_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='output'
        )

        # Output layer normalization
        self.output_norm = self._create_normalization_layer('output_norm')

        # Dropout layers
        self.dropout = keras.layers.Dropout(self.dropout_rate, name='dropout')
        self.attention_dropout = keras.layers.Dropout(
            self.attention_dropout_rate,
            name='attention_dropout'
        )

        # Build sublayers
        self.attention.build(query_shape=input_shape, value_shape=input_shape)
        self.attention_norm.build(input_shape)
        self.intermediate.build(input_shape)

        # Output layer takes intermediate output
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.intermediate_size
        self.output_dense.build(tuple(intermediate_shape))
        self.output_norm.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            attention_mask: Optional[Any] = None,
            training: Optional[bool] = None
    ) -> Any:
        """
        Forward pass of the transformer layer.

        Args:
            inputs: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor. Can be:
                - 2D tensor of shape (batch_size, seq_length) for padding mask
                - 3D tensor of shape (batch_size, seq_length, seq_length) for attention mask
                - 4D tensor of shape (batch_size, num_heads, seq_length, seq_length) for full mask
            training: Boolean indicating training mode

        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        # For now, let's handle only 3D masks to avoid shape issues
        # TODO: Add proper 2D mask processing later
        processed_mask = None
        if attention_mask is not None:
            if len(attention_mask.shape) == 3:
                # Use 3D mask as-is
                processed_mask = attention_mask
            elif len(attention_mask.shape) == 4:
                # Use 4D mask as-is
                processed_mask = attention_mask
            else:
                # Skip 2D masks for now to avoid shape issues
                logger.warning(f"Skipping 2D attention mask of shape {attention_mask.shape}. Use 3D mask instead.")
                processed_mask = None

        # Multi-head attention with residual connection
        attention_output = self.attention(
            query=inputs,
            value=inputs,  # value = query for self-attention
            key=inputs,  # key = query for self-attention
            attention_mask=processed_mask,
            training=training
        )
        attention_output = self.attention_dropout(attention_output, training=training)
        attention_output = self.attention_norm(attention_output + inputs, training=training)

        # Feed-forward network with residual connection
        intermediate_output = self.intermediate(attention_output, training=training)
        layer_output = self.output_dense(intermediate_output, training=training)
        layer_output = self.dropout(layer_output, training=training)
        layer_output = self.output_norm(layer_output + attention_output, training=training)

        return layer_output

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple.
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)

        # Output shape is the same as input shape for transformer layers
        return tuple(input_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'normalization_type': self.normalization_type,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing build configuration.
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

# ---------------------------------------------------------------------

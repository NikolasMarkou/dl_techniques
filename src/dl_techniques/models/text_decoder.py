import keras
from typing import Optional, Dict, Any, Tuple, Union, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .bert import BertEmbeddings, BertConfig
from ..layers.transformer import TransformerLayer

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

NormalizationType = Literal['layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'dynamic_tanh']
NormalizationPosition = Literal['post', 'pre']
AttentionType = Literal['multi_head_attention', 'window_attention', 'group_query_attention', 'differential_attention']
FFNType = Literal['mlp', 'swiglu', 'differential', 'glu', 'residual', 'swin_mlp']


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TextDecoder(keras.layers.Layer):
    """
    Text decoder using BERT-style embeddings with causal attention for text generation.

    This layer implements a transformer-based text decoder that combines BERT-style
    embeddings with configurable transformer blocks for autoregressive text generation.
    The decoder supports various attention mechanisms, normalization techniques, and
    feed-forward network architectures.

    Key features:
    - BERT-style token, position, and type embeddings
    - Configurable transformer layers with multiple attention types
    - Support for different normalization strategies (pre/post-norm)
    - Flexible feed-forward network architectures
    - Proper causal masking for autoregressive generation
    - Modern Keras 3 serialization support

    Args:
        vocab_size: Integer, size of the vocabulary. Must be positive. Defaults to 32000.
        hidden_dim: Integer, dimensionality of the model hidden states. Must be positive
            and divisible by num_heads. Defaults to 768.
        num_layers: Integer, number of transformer decoder layers. Must be positive.
            Defaults to 12.
        num_heads: Integer, number of attention heads in each layer. Must be positive
            and divide hidden_dim evenly. Defaults to 12.
        mlp_dim: Integer, dimensionality of the feed-forward network. Must be positive.
            Defaults to 3072.
        max_position_embeddings: Integer, maximum sequence length for position embeddings.
            Must be positive. Defaults to 512.
        type_vocab_size: Integer, vocabulary size for token type embeddings. Must be positive.
            Defaults to 2.
        dropout: Float, dropout rate applied throughout the model. Must be between 0 and 1.
            Defaults to 0.1.
        activation: String, activation function for feed-forward networks. Can be any
            valid Keras activation name. Defaults to 'gelu'.
        normalization_type: NormalizationType, type of normalization to use.
            Options: 'layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'dynamic_tanh'.
            Defaults to 'layer_norm'.
        normalization_position: NormalizationPosition, position of normalization layers.
            Options: 'post' (original Transformer), 'pre' (often more stable).
            Defaults to 'post'.
        attention_type: AttentionType, type of attention mechanism.
            Options: 'multi_head_attention', 'window_attention', 'group_query_attention',
            'differential_attention'. Defaults to 'multi_head_attention'.
        ffn_type: FFNType, type of feed-forward network.
            Options: 'mlp', 'swiglu', 'differential', 'glu', 'residual', 'swin_mlp'.
            Defaults to 'mlp'.
        use_bias: Boolean, whether to use bias terms in linear layers. Defaults to True.
        kernel_initializer: String or initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        initializer_range: Float, standard deviation for truncated normal initialization
            of transformer weights. Must be positive. Defaults to 0.02.
        layer_norm_eps: Float, epsilon value for layer normalization. Must be positive.
            Defaults to 1e-12.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, sequence_length)`
        Input should contain integer token IDs.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_dim)`

    Example:
        ```python
        # Basic usage
        decoder = TextDecoder(
            vocab_size=50000,
            hidden_dim=768,
            num_layers=12
        )

        # Advanced configuration with custom attention and normalization
        decoder = TextDecoder(
            vocab_size=50000,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            attention_type='differential_attention',
            normalization_type='rms_norm',
            normalization_position='pre',
            ffn_type='swiglu',
            dropout=0.1
        )

        # In a model
        inputs = keras.Input(shape=(512,), dtype='int32')
        hidden_states = decoder(inputs)
        logits = keras.layers.Dense(vocab_size)(hidden_states)
        model = keras.Model(inputs, logits)
        ```

    Raises:
        ValueError: If any parameter is invalid or incompatible.

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and Keras handles building automatically. This ensures
        proper serialization and eliminates common build errors.
    """

    def __init__(
            self,
            vocab_size: int = 32000,
            hidden_dim: int = 768,
            num_layers: int = 12,
            num_heads: int = 12,
            mlp_dim: int = 3072,
            max_position_embeddings: int = 512,
            type_vocab_size: int = 2,
            dropout: float = 0.1,
            activation: str = 'gelu',
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPosition = 'post',
            attention_type: AttentionType = 'multi_head_attention',
            ffn_type: FFNType = 'mlp',
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate configuration before storing
        self._validate_config_params(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dropout=dropout,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps
        )

        # Store ALL configuration parameters as instance attributes
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.dropout = dropout
        self.activation = activation
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        try:
            # Create BERT configuration for embeddings
            bert_config = BertConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_dim,
                num_heads=self.num_heads,
                max_position_embeddings=self.max_position_embeddings,
                type_vocab_size=self.type_vocab_size,
                hidden_dropout_prob=self.dropout,
                initializer_range=self.initializer_range,
                layer_norm_eps=self.layer_norm_eps,
                normalization_type=self.normalization_type
            )

            # Create embeddings layer
            self.embeddings = BertEmbeddings(
                config=bert_config,
                name='embeddings'
            )

            # Create transformer decoder layers
            self.decoder_layers = []
            for i in range(self.num_layers):
                decoder_layer = TransformerLayer(
                    hidden_size=self.hidden_dim,
                    num_heads=self.num_heads,
                    intermediate_size=self.mlp_dim,
                    normalization_type=self.normalization_type,
                    normalization_position=self.normalization_position,
                    attention_type=self.attention_type,
                    ffn_type=self.ffn_type,
                    dropout_rate=self.dropout,
                    attention_dropout_rate=self.dropout,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_initializer=keras.initializers.TruncatedNormal(
                        stddev=self.initializer_range
                    ),
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name=f'decoder_layer_{i}'
                )
                self.decoder_layers.append(decoder_layer)

            # Create final normalization layer
            self.final_norm = keras.layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name='final_norm'
            )

            logger.info(
                f"Created TextDecoder with {self.num_layers} layers, "
                f"hidden_dim={self.hidden_dim}, vocab_size={self.vocab_size}, "
                f"attention_type={self.attention_type}, ffn_type={self.ffn_type}"
            )

        except Exception as e:
            logger.error(f"Failed to create TextDecoder sub-layers: {e}")
            raise ValueError(
                f"Failed to create TextDecoder. This might be due to missing "
                f"dependencies or incompatible configurations. Original error: {e}"
            )

    def _validate_config_params(
            self,
            vocab_size: int,
            hidden_dim: int,
            num_layers: int,
            num_heads: int,
            mlp_dim: int,
            max_position_embeddings: int,
            type_vocab_size: int,
            dropout: float,
            initializer_range: float,
            layer_norm_eps: float
    ) -> None:
        """
        Validate configuration parameters.

        Args:
            vocab_size: Vocabulary size to validate
            hidden_dim: Hidden dimension to validate
            num_layers: Number of layers to validate
            num_heads: Number of attention heads to validate
            mlp_dim: MLP dimension to validate
            max_position_embeddings: Maximum position embeddings to validate
            type_vocab_size: Type vocabulary size to validate
            dropout: Dropout rate to validate
            initializer_range: Initializer range to validate
            layer_norm_eps: Layer norm epsilon to validate

        Raises:
            ValueError: If any parameter is invalid or incompatible
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if mlp_dim <= 0:
            raise ValueError(f"mlp_dim must be positive, got {mlp_dim}")

        if max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {max_position_embeddings}")

        if type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size must be positive, got {type_vocab_size}")

        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        if initializer_range <= 0:
            raise ValueError(f"initializer_range must be positive, got {initializer_range}")

        if layer_norm_eps <= 0:
            raise ValueError(f"layer_norm_eps must be positive, got {layer_norm_eps}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the text decoder layer.

        Args:
            input_shape: Shape tuple of the input tensor, including batch dimension.
                Expected to be (batch_size, sequence_length).

        Raises:
            ValueError: If input shape is invalid or incompatible.
        """
        if len(input_shape) != 2:
            raise ValueError(
                f"Expected 2D input shape (batch_size, sequence_length), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        # Let Keras know the build is complete
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            token_type_ids: Optional[keras.KerasTensor] = None,
            position_ids: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the text decoder.

        Args:
            inputs: Input tensor containing token IDs with shape (batch_size, seq_length).
            token_type_ids: Optional tensor for token type IDs with shape (batch_size, seq_length).
                Used to distinguish different segments in the input. If None, all tokens
                are treated as type 0.
            position_ids: Optional tensor for position IDs with shape (batch_size, seq_length).
                If None, positions are automatically generated as [0, 1, 2, ..., seq_length-1].
            attention_mask: Optional attention mask tensor. Can be:
                - 2D tensor of shape (batch_size, seq_length) for padding mask
                - 3D tensor of shape (batch_size, seq_length, seq_length) for attention mask
                - 4D tensor of shape (batch_size, num_heads, seq_length, seq_length)
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor with shape (batch_size, seq_length, hidden_dim) containing
            contextualized hidden states for each input token.
        """
        # Process embeddings
        embeddings = self.embeddings(
            input_ids=inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training
        )

        # Pass through transformer decoder layers
        hidden_states = embeddings
        for i, decoder_layer in enumerate(self.decoder_layers):
            hidden_states = decoder_layer(
                inputs=hidden_states,
                attention_mask=attention_mask,
                layer_idx=i,  # Useful for differential attention
                training=training
            )

        # Apply final normalization
        hidden_states = self.final_norm(hidden_states, training=training)

        return hidden_states

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor (batch_size, seq_length).

        Returns:
            Output shape tuple (batch_size, seq_length, hidden_dim).
        """
        # Convert to list for manipulation, then back to tuple
        output_shape = list(input_shape)
        output_shape.append(self.hidden_dim)
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__. Use keras serializers for complex objects.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'max_position_embeddings': self.max_position_embeddings,
            'type_vocab_size': self.type_vocab_size,
            'dropout': self.dropout,
            'activation': self.activation,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'attention_type': self.attention_type,
            'ffn_type': self.ffn_type,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_text_decoder_base(**kwargs: Any) -> TextDecoder:
    """
    Create base text decoder configuration.

    This function creates a TextDecoder with balanced performance and efficiency,
    suitable for most text generation tasks.

    Args:
        **kwargs: Additional arguments to override base configuration.
            Any parameter accepted by TextDecoder.__init__ can be overridden.

    Returns:
        TextDecoder instance with base configuration.

    Example:
        ```python
        # Default base configuration
        decoder = create_text_decoder_base()

        # Override specific parameters
        decoder = create_text_decoder_base(
            vocab_size=50000,
            attention_type='differential_attention'
        )
        ```
    """
    config = {
        'vocab_size': 32000,
        'hidden_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'mlp_dim': 3072,
        'dropout': 0.1,
    }
    config.update(kwargs)

    logger.info("Creating base text decoder")
    return TextDecoder(**config)


def create_text_decoder_small(**kwargs: Any) -> TextDecoder:
    """
    Create small text decoder configuration.

    This function creates a smaller, faster TextDecoder suitable for
    resource-constrained environments or when training speed is critical.

    Args:
        **kwargs: Additional arguments to override small configuration.
            Any parameter accepted by TextDecoder.__init__ can be overridden.

    Returns:
        TextDecoder instance with small configuration.

    Example:
        ```python
        # Default small configuration
        decoder = create_text_decoder_small()

        # Override with even smaller vocabulary
        decoder = create_text_decoder_small(vocab_size=16000)
        ```
    """
    config = {
        'vocab_size': 32000,
        'hidden_dim': 384,
        'num_layers': 6,
        'num_heads': 6,
        'mlp_dim': 1536,
        'dropout': 0.1,
    }
    config.update(kwargs)

    logger.info("Creating small text decoder")
    return TextDecoder(**config)


def create_text_decoder_large(**kwargs: Any) -> TextDecoder:
    """
    Create large text decoder configuration.

    This function creates a larger, more powerful TextDecoder suitable for
    high-quality text generation where computational resources are available.

    Args:
        **kwargs: Additional arguments to override large configuration.
            Any parameter accepted by TextDecoder.__init__ can be overridden.

    Returns:
        TextDecoder instance with large configuration.

    Example:
        ```python
        # Default large configuration
        decoder = create_text_decoder_large()

        # Override with custom normalization
        decoder = create_text_decoder_large(
            normalization_type='rms_norm',
            normalization_position='pre'
        )
        ```
    """
    config = {
        'vocab_size': 32000,
        'hidden_dim': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'mlp_dim': 4096,
        'dropout': 0.1,
    }
    config.update(kwargs)

    logger.info("Creating large text decoder")
    return TextDecoder(**config)
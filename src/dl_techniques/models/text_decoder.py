import keras
from typing import Optional, Dict, Any, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .bert import BertEmbeddings, BertConfig
from ..layers.transformer import TransformerLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TextDecoder(keras.layers.Layer):
    """
    Text decoder using BERT-style embeddings with causal attention for generation.

    This layer implements a transformer-based text decoder that uses BERT's
    embedding approach (word + position + token type embeddings) but with
    causal attention for autoregressive text generation. It's designed for
    vision-language models that need to generate text conditioned on visual input.

    The decoder consists of:
    - BERT-style embeddings (word, position, token type)
    - Stack of causal transformer decoder layers
    - Layer normalization and dropout for regularization

    Args:
        vocab_size: Size of the vocabulary for text embeddings. Must be positive.
        hidden_dim: Hidden dimension of the transformer layers. Must be positive
            and divisible by num_heads.
        num_layers: Number of transformer decoder layers. Must be positive.
        num_heads: Number of attention heads in each transformer layer. Must be positive.
        mlp_dim: Dimension of the feed-forward network. Must be positive.
        max_position_embeddings: Maximum sequence length for positional embeddings.
            Must be positive. Defaults to 512.
        type_vocab_size: Size of the token type vocabulary. Must be positive.
            Defaults to 2.
        dropout: Dropout rate applied throughout the model. Must be between 0 and 1.
            Defaults to 0.1.
        activation: Activation function for transformer layers. Defaults to 'gelu'.
        normalization_type: Type of normalization layer to use. Options include
            'layer_norm', 'rms_norm', etc. Defaults to 'layer_norm'.
        normalization_position: Position of normalization ('pre' or 'post').
            Defaults to 'post'.
        attention_type: Type of attention mechanism. Options include
            'multi_head_attention', 'window_attention', etc. Defaults to 'multi_head_attention'.
        ffn_type: Type of feed-forward network architecture. Options include
            'mlp', 'swiglu', 'differential', etc. Defaults to 'mlp'.
        use_bias: Whether to use bias in linear layers. Defaults to True.
        kernel_initializer: Initializer for kernel weights. Accepts string names
            or Initializer instances. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias vectors. Accepts string names
            or Initializer instances. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias vectors.
        initializer_range: Standard deviation for weight initialization.
            Must be positive. Defaults to 0.02.
        layer_norm_eps: Epsilon value for normalization layers. Must be positive.
            Defaults to 1e-12.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        2D tensor of shape (batch_size, sequence_length) containing token IDs

    Output shape:
        3D tensor of shape (batch_size, sequence_length, hidden_dim)

    Raises:
        ValueError: If hidden_dim is not positive or not divisible by num_heads.
        ValueError: If any dimension parameter is not positive.
        ValueError: If dropout is not between 0 and 1.

    Example:
        ```python
        # Basic usage
        decoder = TextDecoder(
            vocab_size=32000,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_dim=3072
        )

        # Advanced configuration with custom attention and FFN
        decoder = TextDecoder(
            vocab_size=50000,
            hidden_dim=1024,
            num_layers=24,
            num_heads=16,
            mlp_dim=4096,
            attention_type='window_attention',
            ffn_type='swiglu',
            normalization_position='pre',
            dropout=0.15
        )

        # In a model
        text_tokens = keras.Input(shape=(50,), dtype='int32')
        features = decoder(text_tokens)  # Shape: (batch_size, 50, 768)
        ```
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_dim: int,
            num_layers: int,
            num_heads: int,
            mlp_dim: int,
            max_position_embeddings: int = 512,
            type_vocab_size: int = 2,
            dropout: float = 0.1,
            activation: str = 'gelu',
            normalization_type: str = 'layer_norm',
            normalization_position: str = 'post',
            attention_type: str = 'multi_head_attention',
            ffn_type: str = 'mlp',
            use_bias: bool = True,
            kernel_initializer: str = 'glorot_uniform',
            bias_initializer: str = 'zeros',
            kernel_regularizer: Optional[str] = None,
            bias_regularizer: Optional[str] = None,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate configuration early
        self._validate_config(
            vocab_size, hidden_dim, num_layers, num_heads, mlp_dim,
            max_position_embeddings, type_vocab_size, dropout, initializer_range,
            layer_norm_eps
        )

        # Store ALL configuration parameters
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

        # CREATE all sublayers in __init__ (following modern Keras 3 pattern)
        try:
            # Create BERT embeddings configuration
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

            # Create decoder transformer layers
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

        except Exception as e:
            logger.error(f"Failed to create TextDecoder sublayers: {e}")
            raise ValueError(
                f"Failed to create TextDecoder. This might be due to missing dependencies "
                f"or incompatible configurations. Original error: {e}"
            )

        logger.info(f"Created TextDecoder with {num_layers} layers, "
                    f"hidden_dim={hidden_dim}, vocab_size={vocab_size}")

    def _validate_config(
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
            num_heads: Number of heads to validate
            mlp_dim: MLP dimension to validate
            max_position_embeddings: Max position embeddings to validate
            type_vocab_size: Type vocab size to validate
            dropout: Dropout rate to validate
            initializer_range: Initializer range to validate
            layer_norm_eps: Layer norm epsilon to validate

        Raises:
            ValueError: If any parameter is invalid
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if mlp_dim <= 0:
            raise ValueError(f"mlp_dim must be positive, got {mlp_dim}")
        if max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {max_position_embeddings}")
        if type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size must be positive, got {type_vocab_size}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        if initializer_range <= 0:
            raise ValueError(f"initializer_range must be positive, got {initializer_range}")
        if layer_norm_eps <= 0:
            raise ValueError(f"layer_norm_eps must be positive, got {layer_norm_eps}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the text decoder by explicitly building all sublayers.

        This is critical for proper serialization - we must build each sublayer
        so that weight variables exist before weight restoration during loading.

        Args:
            input_shape: Shape tuple of input token IDs (batch_size, seq_length)

        Raises:
            ValueError: If input shape is invalid
        """
        if self.built:
            return

        if len(input_shape) != 2:
            raise ValueError(
                f"Expected 2D input shape (batch_size, seq_length), "
                f"got {input_shape}"
            )

        logger.info(f"Building TextDecoder with input_shape: {input_shape}")

        # CRITICAL: Explicitly build each sublayer for robust serialization
        try:
            # Build embeddings layer
            self.embeddings.build(input_shape)

            # Compute shape after embeddings: (batch_size, seq_length, hidden_dim)
            embedding_output_shape = (*input_shape, self.hidden_dim)

            # Build each decoder layer
            for i, decoder_layer in enumerate(self.decoder_layers):
                decoder_layer.build(embedding_output_shape)
                logger.debug(f"Built decoder layer {i}")

            # Build final normalization layer
            self.final_norm.build(embedding_output_shape)

        except Exception as e:
            logger.error(f"Failed to build TextDecoder sublayers: {e}")
            raise ValueError(f"Failed to build TextDecoder sublayers: {e}")

        # Always call parent build at the end
        super().build(input_shape)
        logger.info("TextDecoder built successfully")

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
            inputs: Input token IDs of shape [batch_size, seq_length]
            token_type_ids: Token type IDs for segment embeddings of shape
                [batch_size, seq_length]. Optional.
            position_ids: Position IDs for custom positional encoding of shape
                [batch_size, seq_length]. Optional.
            attention_mask: Attention mask to avoid attention on padding tokens
                of shape [batch_size, seq_length] or [batch_size, seq_length, seq_length].
                Optional.
            training: Whether in training mode

        Returns:
            Hidden states of shape [batch_size, seq_length, hidden_dim]
        """
        # Get embeddings from BERT embedding layer
        embeddings = self.embeddings(
            input_ids=inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training
        )

        # Pass through decoder layers
        hidden_states = embeddings
        for i, decoder_layer in enumerate(self.decoder_layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_idx=i,  # For differential attention
                training=training
            )

        # Apply final normalization
        hidden_states = self.final_norm(hidden_states, training=training)

        return hidden_states

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        Args:
            input_shape: Input shape tuple (batch_size, seq_length)

        Returns:
            Output shape tuple (batch_size, seq_length, hidden_dim)
        """
        return (*input_shape, self.hidden_dim)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        This must include ALL parameters passed to __init__ for proper serialization.

        Returns:
            Dictionary containing complete layer configuration
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TextDecoder':
        """
        Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            TextDecoder instance
        """
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_text_decoder_base(**kwargs: Any) -> TextDecoder:
    """
    Create base text decoder configuration.

    Args:
        **kwargs: Additional arguments to override base configuration

    Returns:
        TextDecoder with base configuration

    Example:
        >>> decoder = create_text_decoder_base(vocab_size=50000)
        >>> # Creates decoder with base config but custom vocab_size
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
    Create small text decoder configuration for resource-constrained environments.

    Args:
        **kwargs: Additional arguments to override small configuration

    Returns:
        TextDecoder with small configuration

    Example:
        >>> decoder = create_text_decoder_small(num_layers=4)
        >>> # Creates smaller decoder with even fewer layers
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
    Create large text decoder configuration for high-performance applications.

    Args:
        **kwargs: Additional arguments to override large configuration

    Returns:
        TextDecoder with large configuration

    Example:
        >>> decoder = create_text_decoder_large(attention_type='differential_attention')
        >>> # Creates large decoder with advanced attention mechanism
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
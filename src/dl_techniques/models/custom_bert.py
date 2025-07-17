"""
Custom BERT Model with Configurable Normalization

This module implements a custom BERT model with configurable normalization layers,
designed to integrate with the dl_techniques library. The model supports different
normalization techniques including BandRMS, RMSNorm, and LayerNorm.

The implementation follows Keras best practices with proper serialization support,
backend-agnostic operations, and comprehensive configuration management.

Example usage:
    ```python
    from dl_techniques.models.custom_bert import CustomBertModel
    from dl_techniques.layers.norms.band_rms import BandRMS

    # Create normalization factory
    def norm_factory():
        return BandRMS(max_band_width=0.2, axis=-1)

    # Build model
    model = CustomBertModel(
        vocab_size=30522,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        normalization_factory=norm_factory,
        num_classes=3
    )

    # Compile and use
    model.compile(
        optimizer='adamw',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    ```
"""

import keras
from keras import ops
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.multi_head_attention import MultiHeadAttention


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

@dataclass
class BertConfig:
    """
    Configuration class for Custom BERT model.

    Args:
        vocab_size: Size of vocabulary.
        hidden_size: Hidden size of transformer layers.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        intermediate_size: Size of intermediate (feed-forward) layer.
        max_position_embeddings: Maximum position embeddings.
        type_vocab_size: Vocabulary size for token types.
        dropout_rate: Dropout rate.
        attention_dropout_rate: Attention dropout rate.
        activation: Activation function for feed-forward layers.
        layer_norm_epsilon: Epsilon for layer normalization.
        initializer_range: Range for weight initialization.
        use_bias: Whether to use bias in linear layers.
        num_classes: Number of output classes (for classification head).
    """
    vocab_size: int = 30522
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    activation: str = 'gelu'
    layer_norm_epsilon: float = 1e-12
    initializer_range: float = 0.02
    use_bias: bool = True
    num_classes: int = 2


# ---------------------------------------------------------------------
# NORMALIZATION FACTORY
# ---------------------------------------------------------------------

def create_normalization_factory(
    norm_type: str = 'layer_norm',
    norm_params: Optional[Dict[str, Any]] = None
) -> Callable[[], keras.layers.Layer]:
    """
    Create a normalization factory function.

    Args:
        norm_type: Type of normalization ('band_rms', 'rms_norm', 'layer_norm').
        norm_params: Parameters for the normalization layer.

    Returns:
        Factory function that creates normalization layers.
    """
    if norm_params is None:
        norm_params = {}

    def factory():
        if norm_type == 'band_rms':
            return BandRMS(
                axis=-1,
                max_band_width=norm_params.get('max_band_width', 0.1),
                epsilon=norm_params.get('epsilon', 1e-7),
                **{k: v for k, v in norm_params.items()
                   if k not in ['max_band_width', 'epsilon']}
            )
        elif norm_type == 'rms_norm':
            return RMSNorm(
                axis=-1,
                epsilon=norm_params.get('epsilon', 1e-6),
                **{k: v for k, v in norm_params.items() if k != 'epsilon'}
            )
        elif norm_type == 'layer_norm':
            return keras.layers.LayerNormalization(
                axis=-1,
                epsilon=norm_params.get('epsilon', 1e-12),
                **{k: v for k, v in norm_params.items() if k != 'epsilon'}
            )
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    return factory


# Multi-head attention implementation is now imported from:
# dl_techniques.layers.multi_head_attention


# ---------------------------------------------------------------------
# BERT TRANSFORMER LAYER
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BertTransformerLayer(keras.layers.Layer):
    """
    BERT transformer layer with configurable normalization.

    This layer implements a standard transformer encoder block with:
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections
    - Configurable normalization

    Args:
        hidden_size: Hidden size of the layer.
        num_heads: Number of attention heads.
        intermediate_size: Size of the intermediate (feed-forward) layer.
        normalization_factory: Factory function to create normalization layers.
        dropout_rate: Dropout rate.
        attention_dropout_rate: Attention-specific dropout rate.
        activation: Activation function for feed-forward network.
        use_bias: Whether to use bias in linear layers.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        normalization_factory: Callable[[], keras.layers.Layer],
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        activation: str = 'gelu',
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.normalization_factory = normalization_factory
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Initialize layers to None - will be created in build()
        self.attention = None
        self.attention_norm = None
        self.intermediate = None
        self.output = None
        self.output_norm = None
        self.dropout = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the transformer layer components."""
        self._build_input_shape = input_shape

        # Multi-head attention using existing implementation
        self.attention = MultiHeadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name='attention'
        )

        # Attention layer normalization
        self.attention_norm = self.normalization_factory()
        self.attention_norm._name = 'attention_norm'

        # Feed-forward network
        self.intermediate = keras.layers.Dense(
            self.intermediate_size,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='intermediate'
        )

        self.output = keras.layers.Dense(
            self.hidden_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='output'
        )

        # Output layer normalization
        self.output_norm = self.normalization_factory()
        self.output_norm._name = 'output_norm'

        # Dropout
        self.dropout = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, attention_mask=None, training=None):
        """
        Forward pass of the transformer layer.

        Args:
            inputs: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask
            training: Boolean indicating training mode

        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        # Multi-head attention with residual connection
        attention_output = self.attention(
            inputs, attention_mask=attention_mask, training=training
        )
        attention_output = self.dropout(attention_output, training=training)
        attention_output = self.attention_norm(attention_output + inputs)

        # Feed-forward network with residual connection
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output, training=training)
        layer_output = self.output_norm(layer_output + attention_output)

        return layer_output

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {'input_shape': self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])


# ---------------------------------------------------------------------
# BERT EMBEDDINGS LAYER
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BertEmbeddings(keras.layers.Layer):
    """
    BERT embeddings layer combining word, position, and token type embeddings.

    Args:
        config: BERT configuration object.
        normalization_factory: Factory function to create normalization layers.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        config: BertConfig,
        normalization_factory: Callable[[], keras.layers.Layer],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.normalization_factory = normalization_factory

        # Initialize layers to None - will be created in build()
        self.word_embeddings = None
        self.position_embeddings = None
        self.token_type_embeddings = None
        self.layer_norm = None
        self.dropout = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the embedding layers."""
        self._build_input_shape = input_shape

        # Word embeddings
        self.word_embeddings = keras.layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name='word_embeddings'
        )

        # Position embeddings
        self.position_embeddings = keras.layers.Embedding(
            input_dim=self.config.max_position_embeddings,
            output_dim=self.config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name='position_embeddings'
        )

        # Token type embeddings
        self.token_type_embeddings = keras.layers.Embedding(
            input_dim=self.config.type_vocab_size,
            output_dim=self.config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name='token_type_embeddings'
        )

        # Layer normalization and dropout
        self.layer_norm = self.normalization_factory()
        self.layer_norm._name = 'layer_norm'
        self.dropout = keras.layers.Dropout(self.config.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass of the embeddings layer.

        Args:
            inputs: Dictionary containing input_ids, token_type_ids, and position_ids
            training: Boolean indicating training mode

        Returns:
            Embedded input tensor
        """
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            token_type_ids = inputs.get('token_type_ids', None)
            position_ids = inputs.get('position_ids', None)
        else:
            input_ids = inputs
            token_type_ids = None
            position_ids = None

        seq_length = ops.shape(input_ids)[1]

        # Word embeddings
        word_embeddings = self.word_embeddings(input_ids)

        # Position embeddings
        if position_ids is None:
            position_ids = ops.arange(seq_length)[None, :]
        position_embeddings = self.position_embeddings(position_ids)

        # Token type embeddings
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Combine embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings

        # Layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        if isinstance(input_shape, dict):
            batch_size = input_shape['input_ids'][0]
            seq_length = input_shape['input_ids'][1]
        else:
            batch_size = input_shape[0]
            seq_length = input_shape[1]

        return (batch_size, seq_length, self.config.hidden_size)

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'config': {
                'vocab_size': self.config.vocab_size,
                'hidden_size': self.config.hidden_size,
                'max_position_embeddings': self.config.max_position_embeddings,
                'type_vocab_size': self.config.type_vocab_size,
                'dropout_rate': self.config.dropout_rate,
                'initializer_range': self.config.initializer_range,
            }
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {'input_shape': self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])


# ---------------------------------------------------------------------
# CUSTOM BERT MODEL
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CustomBertModel(keras.Model):
    """
    Custom BERT model with configurable normalization.

    This model implements the BERT architecture with the ability to use different
    normalization techniques including BandRMS, RMSNorm, and LayerNorm.

    Args:
        config: BERT configuration object.
        normalization_factory: Factory function to create normalization layers.
        add_pooling_layer: Whether to add pooling layer for classification.
        **kwargs: Additional keyword arguments for the Model base class.

    Example:
        ```python
        from dl_techniques.models.custom_bert import CustomBertModel, BertConfig
        from dl_techniques.layers.norms.band_rms import BandRMS

        # Create configuration
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            num_classes=3
        )

        # Create normalization factory
        def norm_factory():
            return BandRMS(max_band_width=0.2, axis=-1)

        # Build model
        model = CustomBertModel(
            config=config,
            normalization_factory=norm_factory,
            add_pooling_layer=True
        )

        # Compile
        model.compile(
            optimizer='adamw',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """

    def __init__(
        self,
        config: BertConfig,
        normalization_factory: Callable[[], keras.layers.Layer],
        add_pooling_layer: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.normalization_factory = normalization_factory
        self.add_pooling_layer = add_pooling_layer

        # Build embeddings
        self.embeddings = BertEmbeddings(
            config=config,
            normalization_factory=normalization_factory,
            name='embeddings'
        )

        # Build transformer layers
        self.transformer_layers = []
        for i in range(config.num_layers):
            layer = BertTransformerLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                normalization_factory=normalization_factory,
                dropout_rate=config.dropout_rate,
                attention_dropout_rate=config.attention_dropout_rate,
                activation=config.activation,
                use_bias=config.use_bias,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=config.initializer_range
                ),
                name=f'transformer_layer_{i}'
            )
            self.transformer_layers.append(layer)

        # Build pooling and classification layers
        if add_pooling_layer:
            self.pooler = keras.layers.Dense(
                config.hidden_size,
                activation='tanh',
                use_bias=config.use_bias,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=config.initializer_range
                ),
                name='pooler'
            )

            self.classifier_dropout = keras.layers.Dropout(config.dropout_rate)

            self.classifier = keras.layers.Dense(
                config.num_classes,
                activation='softmax',
                use_bias=config.use_bias,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=config.initializer_range
                ),
                name='classifier'
            )

    def call(self, inputs, attention_mask=None, training=None):
        """
        Forward pass of the BERT model.

        Args:
            inputs: Input dictionary containing input_ids, token_type_ids, etc.
                   or tensor of input_ids.
            attention_mask: Optional attention mask.
            training: Boolean indicating training mode.

        Returns:
            Model outputs (logits if classification head is added).
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            if attention_mask is None:
                attention_mask = inputs.get('attention_mask', None)
            embedding_inputs = inputs
        else:
            input_ids = inputs
            embedding_inputs = {'input_ids': input_ids}

        # Embeddings
        hidden_states = self.embeddings(embedding_inputs, training=training)

        # Transformer layers
        for layer in self.transformer_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )

        # Pooling and classification
        if self.add_pooling_layer:
            # Use [CLS] token (first token) for classification
            pooled_output = hidden_states[:, 0, :]
            pooled_output = self.pooler(pooled_output)
            pooled_output = self.classifier_dropout(pooled_output, training=training)
            logits = self.classifier(pooled_output)
            return logits
        else:
            return hidden_states

    def get_config(self):
        """Get model configuration for serialization."""
        config = {
            'config': {
                'vocab_size': self.config.vocab_size,
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'num_heads': self.config.num_heads,
                'intermediate_size': self.config.intermediate_size,
                'max_position_embeddings': self.config.max_position_embeddings,
                'type_vocab_size': self.config.type_vocab_size,
                'dropout_rate': self.config.dropout_rate,
                'attention_dropout_rate': self.config.attention_dropout_rate,
                'activation': self.config.activation,
                'layer_norm_epsilon': self.config.layer_norm_epsilon,
                'initializer_range': self.config.initializer_range,
                'use_bias': self.config.use_bias,
                'num_classes': self.config.num_classes,
            },
            'add_pooling_layer': self.add_pooling_layer,
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create model from configuration."""
        bert_config = BertConfig(**config['config'])

        # Default normalization factory (LayerNorm)
        def default_norm_factory():
            return keras.layers.LayerNormalization(
                epsilon=bert_config.layer_norm_epsilon,
                axis=-1
            )

        return cls(
            config=bert_config,
            normalization_factory=default_norm_factory,
            add_pooling_layer=config.get('add_pooling_layer', True)
        )


# ---------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ---------------------------------------------------------------------

def create_bert_for_classification(
    num_classes: int,
    normalization_type: str = 'layer_norm',
    normalization_params: Optional[Dict[str, Any]] = None,
    bert_config: Optional[BertConfig] = None,
    **kwargs
) -> CustomBertModel:
    """
    Create a BERT model for classification tasks.

    Args:
        num_classes: Number of output classes.
        normalization_type: Type of normalization to use.
        normalization_params: Parameters for normalization layer.
        bert_config: BERT configuration. If None, uses default.
        **kwargs: Additional arguments for BertConfig.

    Returns:
        Configured BERT model for classification.
    """
    if bert_config is None:
        bert_config = BertConfig(num_classes=num_classes, **kwargs)
    else:
        bert_config.num_classes = num_classes

    # Create normalization factory
    normalization_factory = create_normalization_factory(
        norm_type=normalization_type,
        norm_params=normalization_params or {}
    )

    # Create model
    model = CustomBertModel(
        config=bert_config,
        normalization_factory=normalization_factory,
        add_pooling_layer=True
    )

    return model


def create_bert_for_sequence_output(
    normalization_type: str = 'layer_norm',
    normalization_params: Optional[Dict[str, Any]] = None,
    bert_config: Optional[BertConfig] = None,
    **kwargs
) -> CustomBertModel:
    """
    Create a BERT model for sequence-to-sequence tasks.

    Args:
        normalization_type: Type of normalization to use.
        normalization_params: Parameters for normalization layer.
        bert_config: BERT configuration. If None, uses default.
        **kwargs: Additional arguments for BertConfig.

    Returns:
        Configured BERT model for sequence output.
    """
    if bert_config is None:
        bert_config = BertConfig(**kwargs)

    # Create normalization factory
    normalization_factory = create_normalization_factory(
        norm_type=normalization_type,
        norm_params=normalization_params or {}
    )

    # Create model
    model = CustomBertModel(
        config=bert_config,
        normalization_factory=normalization_factory,
        add_pooling_layer=False
    )

    return model

# ---------------------------------------------------------------------

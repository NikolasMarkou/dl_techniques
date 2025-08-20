"""
BERT (Bidirectional Encoder Representations from Transformers) Implementation

This module provides a comprehensive BERT implementation using the dl-techniques framework,
featuring configurable architectures, proper serialization, and integration with the
existing TransformerLayer component.

Key features:
- Full compatibility with Keras 3.x model lifecycle
- Configurable normalization types (LayerNorm, RMSNorm, BandRMS)
- Configurable attention mechanisms and FFN architectures
- Proper serialization and deserialization support
- Integration with dl-techniques TransformerLayer
- Support for both classification and sequence output tasks
"""

import keras
from keras import ops
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.norms.rms_norm import RMSNorm
from ..layers.norms.band_rms import BandRMS
from ..layers.transformer import TransformerLayer


# ---------------------------------------------------------------------

@dataclass
class BertConfig:
    """
    Configuration class for BERT model parameters.

    This dataclass contains all the hyperparameters and architectural choices
    for the BERT model, including support for advanced normalization types
    and attention mechanisms from the dl-techniques framework.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Hidden dimension of the transformer layers.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads in each transformer layer.
        intermediate_size: Size of the intermediate layer in the feed-forward network.
        hidden_act: Activation function for the feed-forward network.
        hidden_dropout_prob: Dropout probability for hidden layers.
        attention_probs_dropout_prob: Dropout probability for attention weights.
        max_position_embeddings: Maximum sequence length for positional embeddings.
        type_vocab_size: Size of the token type vocabulary (for segment embeddings).
        initializer_range: Standard deviation for weight initialization.
        layer_norm_eps: Epsilon value for normalization layers.
        pad_token_id: Token ID used for padding.
        position_embedding_type: Type of position embedding ('absolute' or 'relative').
        use_cache: Whether to use caching in attention layers.
        classifier_dropout: Dropout probability for classification head.
        normalization_type: Type of normalization layer to use.
        normalization_position: Position of normalization ('pre' or 'post').
        attention_type: Type of attention mechanism.
        ffn_type: Type of feed-forward network architecture.
        use_stochastic_depth: Whether to enable stochastic depth regularization.
        stochastic_depth_rate: Drop path rate for stochastic depth.
    """
    vocab_size: int = 30522
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    classifier_dropout: Optional[float] = None
    normalization_type: str = "layer_norm"
    normalization_position: str = "post"
    attention_type: str = "multi_head_attention"
    ffn_type: str = "mlp"
    use_stochastic_depth: bool = False
    stochastic_depth_rate: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BertConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if not (0.0 <= self.hidden_dropout_prob <= 1.0):
            raise ValueError(f"hidden_dropout_prob must be between 0 and 1")


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BertEmbeddings(keras.layers.Layer):
    """
    BERT embeddings layer combining word, position, and token type embeddings.

    This layer implements the embedding component of BERT, which combines:
    - Word embeddings: Map token IDs to dense vector representations
    - Position embeddings: Add positional information to sequence tokens
    - Token type embeddings: Distinguish between different sentence segments

    The embeddings are summed together, normalized, and passed through dropout
    for regularization. Supports configurable normalization types from the
    dl-techniques framework.

    Args:
        config: BERT configuration object containing all hyperparameters.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        - input_ids: (batch_size, sequence_length)
        - token_type_ids: (batch_size, sequence_length) - optional
        - position_ids: (batch_size, sequence_length) - optional

    Output shape:
        (batch_size, sequence_length, hidden_size)

    Example:
        ```python
        config = BertConfig(hidden_size=768, vocab_size=30522)
        embeddings = BertEmbeddings(config)

        input_ids = keras.ops.array([[101, 2023, 2003, 102]])  # [CLS] this is [SEP]
        embedded = embeddings(input_ids)
        ```
    """

    def __init__(
            self,
            config: BertConfig,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate configuration
        config.validate()
        self.config = config

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.word_embeddings = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            mask_zero=True,
            name="word_embeddings"
        )

        self.position_embeddings = keras.layers.Embedding(
            input_dim=config.max_position_embeddings,
            output_dim=config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            name="position_embeddings"
        )

        self.token_type_embeddings = keras.layers.Embedding(
            input_dim=config.type_vocab_size,
            output_dim=config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            name="token_type_embeddings"
        )

        # Create normalization layer based on config
        self.layer_norm = self._create_normalization_layer("layer_norm")

        self.dropout = keras.layers.Dropout(
            rate=config.hidden_dropout_prob,
            name="dropout"
        )

        logger.info(f"Created BertEmbeddings with hidden_size={config.hidden_size}, "
                    f"vocab_size={config.vocab_size}")

    def _create_normalization_layer(self, name: str) -> keras.layers.Layer:
        """
        Create a normalization layer based on the configuration type.

        Args:
            name: Name for the normalization layer.

        Returns:
            Configured normalization layer instance.

        Raises:
            ValueError: If normalization_type is not supported.
        """
        if self.config.normalization_type == 'layer_norm':
            return keras.layers.LayerNormalization(
                epsilon=self.config.layer_norm_eps,
                name=name
            )
        elif self.config.normalization_type == 'rms_norm':
            return RMSNorm(
                epsilon=self.config.layer_norm_eps,
                name=name
            )
        elif self.config.normalization_type == 'band_rms':
            return BandRMS(
                epsilon=self.config.layer_norm_eps,
                name=name
            )
        elif self.config.normalization_type == 'batch_norm':
            return keras.layers.BatchNormalization(
                epsilon=self.config.layer_norm_eps,
                name=name
            )
        else:
            raise ValueError(
                f"Unknown normalization type: {self.config.normalization_type}. "
                f"Supported types: layer_norm, rms_norm, band_rms, batch_norm"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the embeddings layer by explicitly building all sub-layers.

        This follows the modern Keras 3 pattern where the parent layer
        explicitly builds all child layers for robust serialization.

        Args:
            input_shape: Shape tuple for input_ids (batch_size, seq_length).

        Raises:
            ValueError: If input_shape is invalid.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape (batch_size, seq_length), "
                             f"got {len(input_shape)}D: {input_shape}")

        logger.info(f"Building BertEmbeddings with input_shape: {input_shape}")

        # CRITICAL: Explicitly build all sub-layers for robust serialization
        self.word_embeddings.build(input_shape)
        self.position_embeddings.build(input_shape)
        self.token_type_embeddings.build(input_shape)

        # Build normalization and dropout with embeddings output shape
        embeddings_output_shape = (*input_shape, self.config.hidden_size)
        self.layer_norm.build(embeddings_output_shape)
        self.dropout.build(embeddings_output_shape)

        super().build(input_shape)
        logger.info("BertEmbeddings built successfully")

    def call(
            self,
            input_ids: keras.KerasTensor,
            token_type_ids: Optional[keras.KerasTensor] = None,
            position_ids: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply embeddings to input tokens.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length).
            token_type_ids: Token type IDs of shape (batch_size, seq_length).
                If None, defaults to all zeros.
            position_ids: Position IDs of shape (batch_size, seq_length).
                If None, defaults to sequential positions.
            training: Whether the layer is in training mode.

        Returns:
            Embedded and normalized tokens of shape
            (batch_size, seq_length, hidden_size).
        """
        input_shape = ops.shape(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = ops.arange(seq_length, dtype="int32")
            position_ids = ops.expand_dims(position_ids, axis=0)
            position_ids = ops.broadcast_to(position_ids, (batch_size, seq_length))

        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids, dtype="int32")

        # Apply all embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Sum all embeddings
        embeddings = word_embeds + position_embeds + token_type_embeds

        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape given input shape."""
        return (*input_shape, self.config.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BertEmbeddings':
        """Create layer from configuration."""
        bert_config = BertConfig.from_dict(config['config'])
        return cls(config=bert_config)


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Bert(keras.Model):
    """
    BERT (Bidirectional Encoder Representations from Transformers) model.

    This implementation uses the dl-techniques TransformerLayer for the encoder
    blocks, providing full configurability of attention mechanisms, normalization
    types, and feed-forward architectures.

    The model consists of:
    - Embeddings layer (word + position + token type)
    - Stack of transformer encoder layers
    - Optional pooling layer for classification tasks

    Args:
        config: BERT configuration object containing all hyperparameters.
        add_pooling_layer: Whether to add a pooling layer for classification tasks.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        - input_ids: (batch_size, sequence_length)
        - attention_mask: (batch_size, sequence_length) - optional
        - token_type_ids: (batch_size, sequence_length) - optional
        - position_ids: (batch_size, sequence_length) - optional

    Output shape:
        - If add_pooling_layer=False: (batch_size, sequence_length, hidden_size)
        - If add_pooling_layer=True: tuple of sequence output and pooled output

    Example:
        ```python
        config = BertConfig(num_layers=12, hidden_size=768)
        model = Bert(config, add_pooling_layer=True)

        # Use the model
        input_ids = keras.random.uniform((2, 512), 0, 30522, dtype='int32')
        outputs = model(input_ids)
        ```
    """

    def __init__(
            self,
            config: BertConfig,
            add_pooling_layer: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate and store configuration
        config.validate()
        self.config = config
        self.add_pooling_layer = add_pooling_layer

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.embeddings = BertEmbeddings(
            config=config,
            name="embeddings"
        )

        # Create transformer encoder layers
        self.encoder_layers: List[TransformerLayer] = []
        for i in range(config.num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                normalization_type=config.normalization_type,
                normalization_position=config.normalization_position,
                attention_type=config.attention_type,
                ffn_type=config.ffn_type,
                dropout_rate=config.hidden_dropout_prob,
                attention_dropout_rate=config.attention_probs_dropout_prob,
                use_stochastic_depth=config.use_stochastic_depth,
                stochastic_depth_rate=config.stochastic_depth_rate,
                activation=config.hidden_act,
                use_bias=True,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=config.initializer_range
                ),
                bias_initializer="zeros",
                name=f"encoder_layer_{i}"
            )
            self.encoder_layers.append(transformer_layer)

        # Create pooler if needed
        self.pooler = None
        if add_pooling_layer:
            self.pooler = keras.layers.Dense(
                units=config.hidden_size,
                activation="tanh",
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=config.initializer_range
                ),
                name="pooler"
            )

        logger.info(f"Created BERT model with {config.num_layers} layers, "
                    f"hidden_size={config.hidden_size}, pooling={add_pooling_layer}")

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            attention_mask: Optional[keras.KerasTensor] = None,
            token_type_ids: Optional[keras.KerasTensor] = None,
            position_ids: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
            return_dict: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor], Dict[str, keras.KerasTensor]]:
        """
        Forward pass of the BERT model.

        Args:
            inputs: Input token IDs or dictionary containing multiple inputs.
            attention_mask: Mask to avoid attention on padding tokens.
            token_type_ids: Token type IDs for distinguishing sequences.
            position_ids: Position IDs for positional embeddings.
            training: Whether the model is in training mode.
            return_dict: Whether to return outputs as a dictionary.

        Returns:
            Model outputs. Format depends on return_dict and add_pooling_layer:
            - return_dict=False, no pooling: sequence_output
            - return_dict=False, with pooling: (sequence_output, pooled_output)
            - return_dict=True: dictionary with last_hidden_state and pooler_output
        """
        # Parse inputs
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
        else:
            input_ids = inputs

        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training
        )

        # Pass through encoder layers
        hidden_states = embedding_output
        for i, encoder_layer in enumerate(self.encoder_layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_idx=i,  # For differential attention
                training=training
            )

        sequence_output = hidden_states

        # Apply pooling if available
        pooled_output = None
        if self.pooler is not None:
            # Pool the representation of the first token (CLS token)
            first_token_tensor = sequence_output[:, 0]  # (batch_size, hidden_size)
            pooled_output = self.pooler(first_token_tensor)

        # Return in requested format
        if return_dict:
            outputs = {
                "last_hidden_state": sequence_output,
            }
            if pooled_output is not None:
                outputs["pooler_output"] = pooled_output
            return outputs
        else:
            if pooled_output is not None:
                return sequence_output, pooled_output
            else:
                return sequence_output

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'config': self.config.to_dict(),
            'add_pooling_layer': self.add_pooling_layer,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Bert':
        """Create model from configuration."""
        bert_config = BertConfig.from_dict(config['config'])
        return cls(
            config=bert_config,
            add_pooling_layer=config.get('add_pooling_layer', True)
        )


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_bert_for_classification(
        config: BertConfig,
        num_labels: int,
        classifier_dropout: Optional[float] = None
) -> keras.Model:
    """
    Create a BERT model for sequence classification tasks.

    This function creates a complete BERT model with a classification head,
    suitable for tasks like sentiment analysis, text classification, etc.

    Args:
        config: BERT configuration object.
        num_labels: Number of classification labels.
        classifier_dropout: Dropout rate for the classifier head.
            If None, uses config.classifier_dropout or config.hidden_dropout_prob.

    Returns:
        Complete BERT model for classification with proper input/output structure.

    Example:
        ```python
        config = create_bert_base_uncased()
        model = create_bert_for_classification(config, num_labels=2)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """
    logger.info(f"Creating BERT classification model with {num_labels} labels")

    # Create base BERT model with pooling
    bert = Bert(config=config, add_pooling_layer=True, name="bert")

    # Define inputs
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = keras.Input(shape=(None,), dtype="int32", name="token_type_ids")

    # Get BERT outputs
    bert_outputs = bert(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        return_dict=True
    )

    # Classification head
    pooled_output = bert_outputs["pooler_output"]

    # Apply classifier dropout
    if classifier_dropout is None:
        classifier_dropout = config.classifier_dropout or config.hidden_dropout_prob

    if classifier_dropout > 0.0:
        pooled_output = keras.layers.Dropout(
            classifier_dropout,
            name="classifier_dropout"
        )(pooled_output)

    # Final classification layer
    logits = keras.layers.Dense(
        units=num_labels,
        kernel_initializer=keras.initializers.TruncatedNormal(
            stddev=config.initializer_range
        ),
        name="classifier"
    )(pooled_output)

    # Create the final model
    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=logits,
        name="bert_for_classification"
    )

    logger.info(f"Created BERT classification model with {model.count_params()} parameters")
    return model


def create_bert_for_sequence_output(
        config: BertConfig
) -> keras.Model:
    """
    Create a BERT model for sequence-level output tasks.

    This function creates a BERT model that outputs sequence representations,
    suitable for tasks like token classification, question answering, etc.

    Args:
        config: BERT configuration object.

    Returns:
        BERT model outputting sequence-level representations.

    Example:
        ```python
        config = create_bert_base_uncased()
        model = create_bert_for_sequence_output(config)

        # For token classification, add a classification head
        num_tags = 9  # e.g., for NER
        sequence_output = model.output
        logits = keras.layers.Dense(num_tags)(sequence_output)
        token_classifier = keras.Model(model.input, logits)
        ```
    """
    logger.info("Creating BERT model for sequence output tasks")

    # Create base BERT model without pooling
    bert = Bert(config=config, add_pooling_layer=False, name="bert")

    # Define inputs
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = keras.Input(shape=(None,), dtype="int32", name="token_type_ids")

    # Get BERT sequence output
    sequence_output = bert(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    )

    # Create the final model
    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=sequence_output,
        name="bert_for_sequence_output"
    )

    logger.info(f"Created BERT sequence model with {model.count_params()} parameters")
    return model


def create_bert_base_uncased() -> BertConfig:
    """
    Create configuration for BERT-base-uncased model.

    Returns:
        BertConfig configured for the base model size.
    """
    return BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
    )


def create_bert_large_uncased() -> BertConfig:
    """
    Create configuration for BERT-large-uncased model.

    Returns:
        BertConfig configured for the large model size.
    """
    return BertConfig(
        vocab_size=30522,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096,
    )


def create_bert_with_rms_norm(
        size: str = "base",
        normalization_position: str = "pre"
) -> BertConfig:
    """
    Create BERT configuration with RMS normalization.

    Args:
        size: Model size ('base' or 'large').
        normalization_position: Position of normalization ('pre' or 'post').

    Returns:
        BERT configuration with RMS normalization settings.

    Raises:
        ValueError: If size is not supported.
    """
    if size == "base":
        config = create_bert_base_uncased()
    elif size == "large":
        config = create_bert_large_uncased()
    else:
        raise ValueError(f"Unsupported size: {size}. Use 'base' or 'large'")

    config.normalization_type = "rms_norm"
    config.normalization_position = normalization_position

    logger.info(f"Created BERT-{size} config with RMS normalization ({normalization_position})")
    return config


def create_bert_with_advanced_features(
        size: str = "base",
        normalization_type: str = "rms_norm",
        normalization_position: str = "pre",
        attention_type: str = "multi_head_attention",
        ffn_type: str = "swiglu",
        use_stochastic_depth: bool = True,
        stochastic_depth_rate: float = 0.1
) -> BertConfig:
    """
    Create BERT configuration with advanced dl-techniques features.

    Args:
        size: Model size ('base' or 'large').
        normalization_type: Type of normalization layer.
        normalization_position: Position of normalization.
        attention_type: Type of attention mechanism.
        ffn_type: Type of feed-forward network.
        use_stochastic_depth: Whether to use stochastic depth.
        stochastic_depth_rate: Drop path rate for stochastic depth.

    Returns:
        Advanced BERT configuration using dl-techniques features.
    """
    if size == "base":
        config = create_bert_base_uncased()
    elif size == "large":
        config = create_bert_large_uncased()
    else:
        raise ValueError(f"Unsupported size: {size}. Use 'base' or 'large'")

    # Apply advanced features
    config.normalization_type = normalization_type
    config.normalization_position = normalization_position
    config.attention_type = attention_type
    config.ffn_type = ffn_type
    config.use_stochastic_depth = use_stochastic_depth
    config.stochastic_depth_rate = stochastic_depth_rate

    logger.info(f"Created advanced BERT-{size} config with {normalization_type}, "
                f"{attention_type}, {ffn_type}")
    return config
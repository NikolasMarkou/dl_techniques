
import keras
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict, Tuple

from ..utils.logger import logger
from ..layers.transformer import TransformerLayer


@dataclass
class BertConfig:
    """Configuration class for BERT model.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Hidden size of the transformer layers.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        intermediate_size: Size of the intermediate layer in the FFN.
        hidden_act: Activation function for the FFN.
        hidden_dropout_prob: Dropout probability for hidden layers.
        attention_probs_dropout_prob: Dropout probability for attention weights.
        max_position_embeddings: Maximum sequence length for positional embeddings.
        type_vocab_size: Size of the token type vocabulary.
        initializer_range: Standard deviation for weight initialization.
        layer_norm_eps: Epsilon for layer normalization.
        pad_token_id: ID of the padding token.
        position_embedding_type: Type of position embedding ('absolute' or 'relative').
        use_cache: Whether to use caching in attention layers.
        classifier_dropout: Dropout probability for classification head.
        normalization_type: Type of normalization ('layer_norm', 'rms_norm', etc.).
        normalization_position: Position of normalization ('pre' or 'post').
        attention_type: Type of attention mechanism.
        ffn_type: Type of feed-forward network.
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BertConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@keras.saving.register_keras_serializable()
class BertEmbeddings(keras.layers.Layer):
    """BERT embeddings layer combining word, position, and token type embeddings.

    This layer combines three types of embeddings:
    - Word embeddings: Map token IDs to dense vectors
    - Position embeddings: Add positional information to tokens
    - Token type embeddings: Distinguish between different segments

    Args:
        config: BERT configuration object.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        config: BertConfig,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.config = config

        # Initialize embeddings to None - will be created in build()
        self.word_embeddings = None
        self.position_embeddings = None
        self.token_type_embeddings = None
        self.layer_norm = None
        self.dropout = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def _create_normalization_layer(self, name: str) -> keras.layers.Layer:
        """Create a normalization layer based on the configuration.

        Args:
            name: Name for the normalization layer.

        Returns:
            A normalization layer instance.
        """
        if self.config.normalization_type == 'layer_norm':
            return keras.layers.LayerNormalization(
                epsilon=self.config.layer_norm_eps,
                name=name
            )
        elif self.config.normalization_type == 'rms_norm':
            from dl_techniques.layers.norms.rms_norm import RMSNorm
            return RMSNorm(epsilon=self.config.layer_norm_eps, name=name)
        elif self.config.normalization_type == 'band_rms':
            from dl_techniques.layers.norms.band_rms import BandRMS
            return BandRMS(epsilon=self.config.layer_norm_eps, name=name)
        elif self.config.normalization_type == 'batch_norm':
            return keras.layers.BatchNormalization(
                epsilon=self.config.layer_norm_eps,
                name=name
            )
        else:
            raise ValueError(f"Unknown normalization type: {self.config.normalization_type}")

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the embeddings layer.

        Args:
            input_shape: Shape of input tensor.
        """
        self._build_input_shape = input_shape

        # Word embeddings
        self.word_embeddings = keras.layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            mask_zero=True,
            name="word_embeddings"
        )

        # Position embeddings
        self.position_embeddings = keras.layers.Embedding(
            input_dim=self.config.max_position_embeddings,
            output_dim=self.config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name="position_embeddings"
        )

        # Token type embeddings
        self.token_type_embeddings = keras.layers.Embedding(
            input_dim=self.config.type_vocab_size,
            output_dim=self.config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name="token_type_embeddings"
        )

        # Layer normalization and dropout
        self.layer_norm = self._create_normalization_layer("layer_norm")
        self.dropout = keras.layers.Dropout(
            rate=self.config.hidden_dropout_prob,
            name="dropout"
        )

        super().build(input_shape)

    def call(
        self,
        input_ids: keras.KerasTensor,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply embeddings to input tokens.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length).
            token_type_ids: Token type IDs of shape (batch_size, seq_length).
            position_ids: Position IDs of shape (batch_size, seq_length).
            training: Whether the layer is in training mode.

        Returns:
            Embedded tokens of shape (batch_size, seq_length, hidden_size).
        """
        input_shape = keras.ops.shape(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = keras.ops.arange(seq_length, dtype="int32")
            position_ids = keras.ops.expand_dims(position_ids, axis=0)
            position_ids = keras.ops.broadcast_to(position_ids, (batch_size, seq_length))

        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = keras.ops.zeros_like(input_ids, dtype="int32")

        # Apply embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Sum all embeddings
        embeddings = word_embeds + position_embeds + token_type_embeds

        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute output shape.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Output shape tuple.
        """
        return (*input_shape, self.config.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration.

        Returns:
            Build configuration dictionary.
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        Args:
            config: Build configuration dictionary.
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BertEmbeddings':
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary.

        Returns:
            BertEmbeddings layer instance.
        """
        bert_config = BertConfig.from_dict(config['config'])
        return cls(config=bert_config)


@keras.saving.register_keras_serializable()
class CustomBertModel(keras.Model):
    """Custom BERT model using existing TransformerEncoderLayer.

    This implementation leverages the existing TransformerEncoderLayer from
    dl-techniques instead of implementing a custom transformer layer.

    Args:
        config: BERT configuration object.
        add_pooling_layer: Whether to add a pooling layer for classification.
        **kwargs: Additional keyword arguments for the Model base class.
    """

    def __init__(
        self,
        config: BertConfig,
        add_pooling_layer: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.config = config
        self.add_pooling_layer = add_pooling_layer

        # Initialize layers to None - will be created in build()
        self.embeddings = None
        self.encoder_layers = []
        self.pooler = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]) -> None:
        """Build the BERT model.

        Args:
            input_shape: Shape of input tensor(s).
        """
        self._build_input_shape = input_shape

        # Handle different input shape formats
        if isinstance(input_shape, dict):
            # Multiple inputs
            input_ids_shape = input_shape.get('input_ids', input_shape.get('inputs', (None, None)))
        else:
            # Single input
            input_ids_shape = input_shape

        # Create embeddings layer
        self.embeddings = BertEmbeddings(
            config=self.config,
            name="embeddings"
        )
        self.embeddings.build(input_ids_shape)

        # Create transformer encoder layers using existing TransformerEncoderLayer
        self.encoder_layers = []
        for i in range(self.config.num_layers):
            layer = TransformerLayer(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                intermediate_size=self.config.intermediate_size,
                normalization_type=self.config.normalization_type,
                normalization_position=self.config.normalization_position,
                attention_type=self.config.attention_type,
                ffn_type=self.config.ffn_type,
                dropout_rate=self.config.hidden_dropout_prob,
                attention_dropout_rate=self.config.attention_probs_dropout_prob,
                activation=self.config.hidden_act,
                use_bias=True,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.config.initializer_range
                ),
                bias_initializer="zeros",
                name=f"encoder_layer_{i}"
            )

            # Build the transformer layer with embeddings output shape
            embeddings_output_shape = (*input_ids_shape, self.config.hidden_size)
            layer.build(embeddings_output_shape)

            self.encoder_layers.append(layer)

        # Create pooler if needed
        if self.add_pooling_layer:
            self.pooler = keras.layers.Dense(
                units=self.config.hidden_size,
                activation="tanh",
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.config.initializer_range
                ),
                name="pooler"
            )

        super().build(input_shape)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        return_dict: bool = False
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass of the BERT model.

        Args:
            inputs: Input token IDs or dictionary of inputs.
            attention_mask: Attention mask to avoid performing attention on padding tokens.
            token_type_ids: Token type IDs for distinguishing sequences.
            position_ids: Position IDs for positional embeddings.
            training: Whether the model is in training mode.
            return_dict: Whether to return outputs as a dictionary.

        Returns:
            Model outputs (last hidden states and optionally pooled output).
        """
        # Handle different input formats
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
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )

        # Get sequence output
        sequence_output = hidden_states

        # Get pooled output if pooler is available
        pooled_output = None
        if self.pooler is not None:
            # Pool the first token (CLS token)
            first_token_tensor = sequence_output[:, 0]  # Shape: (batch_size, hidden_size)
            pooled_output = self.pooler(first_token_tensor)

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
        """Get model configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'config': self.config.to_dict(),
            'add_pooling_layer': self.add_pooling_layer,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration.

        Returns:
            Build configuration dictionary.
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build model from configuration.

        Args:
            config: Build configuration dictionary.
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CustomBertModel':
        """Create model from configuration.

        Args:
            config: Model configuration dictionary.

        Returns:
            CustomBertModel instance.
        """
        bert_config = BertConfig.from_dict(config['config'])
        return cls(
            config=bert_config,
            add_pooling_layer=config.get('add_pooling_layer', True)
        )


def create_bert_for_classification(
    config: BertConfig,
    num_labels: int,
    classifier_dropout: Optional[float] = None
) -> keras.Model:
    """Create a BERT model for sequence classification.

    Args:
        config: BERT configuration object.
        num_labels: Number of classification labels.
        classifier_dropout: Dropout rate for the classifier.

    Returns:
        BERT model with classification head.
    """
    # Create base BERT model
    bert = CustomBertModel(config=config, add_pooling_layer=True)

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
        pooled_output = keras.layers.Dropout(classifier_dropout)(pooled_output)

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

    return model


def create_bert_for_sequence_output(
    config: BertConfig
) -> keras.Model:
    """Create a BERT model that outputs sequence-level representations.

    Args:
        config: BERT configuration object.

    Returns:
        BERT model for sequence output tasks.
    """
    # Create base BERT model
    bert = CustomBertModel(config=config, add_pooling_layer=False)

    # Define inputs
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = keras.Input(shape=(None,), dtype="int32", name="token_type_ids")

    # Get BERT outputs
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

    return model


# Convenience functions for common BERT configurations
def create_bert_base_uncased() -> BertConfig:
    """Create configuration for BERT-base-uncased."""
    return BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
    )


def create_bert_large_uncased() -> BertConfig:
    """Create configuration for BERT-large-uncased."""
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
    """Create BERT configuration with RMS normalization.

    Args:
        size: Model size ('base' or 'large').
        normalization_position: Position of normalization ('pre' or 'post').

    Returns:
        BERT configuration with RMS normalization.
    """
    if size == "base":
        config = create_bert_base_uncased()
    elif size == "large":
        config = create_bert_large_uncased()
    else:
        raise ValueError(f"Unsupported size: {size}")

    config.normalization_type = "rms_norm"
    config.normalization_position = normalization_position

    return config
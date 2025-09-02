"""
BERT (Bidirectional Encoder Representations from Transformers) Implementation
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from .components import Embeddings

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Bert(keras.Model):
    """
    BERT (Bidirectional Encoder Representations from Transformers) model.

    **Architecture**:
    ```
    Input(input_ids, attention_mask, token_type_ids)
           │
           ▼
    Embeddings(Word + Position + Token Type) -> LayerNorm -> Dropout
           │
           ▼
    TransformerLayer₁ (Self-Attention -> FFN)
           │
           ▼
          ...
           │
           ▼
    TransformerLayerₙ (Self-Attention -> FFN)
           │
           ▼
    Sequence Output (shape=[batch, seq_len, hidden_size])
           │
           └─> (Optional) Pooler(Dense + Tanh on [CLS] token)
                               │
                               ▼
                         Pooled Output (shape=[batch, hidden_size])
    ```

    Args:
        vocab_size: Integer, size of the vocabulary. Defaults to 30522.
        hidden_size: Integer, dimensionality of encoder layers and pooler.
            Defaults to 768.
        num_layers: Integer, number of hidden transformer layers. Defaults to 12.
        num_heads: Integer, number of attention heads for each attention layer.
            Defaults to 12.
        intermediate_size: Integer, dimensionality of the "intermediate"
            (feed-forward) layer. Defaults to 3072.
        hidden_act: String, the non-linear activation function in the encoder.
            Defaults to "gelu".
        hidden_dropout_prob: Float, dropout probability for all fully connected
            layers in embeddings, encoder, and pooler. Defaults to 0.1.
        attention_probs_dropout_prob: Float, dropout ratio for attention
            probabilities. Defaults to 0.1.
        max_position_embeddings: Integer, maximum sequence length for positional
            embeddings. Defaults to 512.
        type_vocab_size: Integer, vocabulary size for token type IDs.
            Defaults to 2.
        initializer_range: Float, stddev of truncated normal initializer for
            all weight matrices. Defaults to 0.02.
        layer_norm_eps: Float, epsilon for normalization layers. Defaults to 1e-12.
        pad_token_id: Integer, ID of padding token. Defaults to 0.
        position_embedding_type: String, type of position embedding.
            Defaults to "absolute".
        use_cache: Boolean, whether to use caching in attention layers.
            Defaults to True.
        classifier_dropout: Optional float, dropout for final classifier head.
            Defaults to None.
        normalization_type: String, type of normalization layer.
            Defaults to "layer_norm".
        normalization_position: String, position of normalization ('pre' or 'post').
            Defaults to "post".
        attention_type: String, type of attention mechanism.
            Defaults to "multi_head_attention".
        ffn_type: String, type of feed-forward network. Defaults to "mlp".
        use_stochastic_depth: Boolean, whether to enable stochastic depth.
            Defaults to False.
        stochastic_depth_rate: Float, drop path rate for stochastic depth.
            Defaults to 0.1.
        add_pooling_layer: Boolean, whether to add a pooling layer.
            Defaults to True.
        **kwargs: Additional keyword arguments for the `keras.Model` base class.

    Input shape:
        Can be a single tensor of `input_ids` or a dictionary.
        - `input_ids`: 2D tensor with shape `(batch_size, sequence_length)`.
        - `attention_mask`: (Optional) 2D tensor, shape `(batch_size, sequence_length)`.
        - `token_type_ids`: (Optional) 2D tensor, shape `(batch_size, sequence_length)`.
        - `position_ids`: (Optional) 2D tensor, shape `(batch_size, sequence_length)`.

    Output shape:
        - If `add_pooling_layer=False`: A single tensor of shape
          `(batch_size, sequence_length, hidden_size)`.
        - If `add_pooling_layer=True`: A tuple `(sequence_output, pooled_output)`.

    Attributes:
        embeddings: The embedding layer instance.
        encoder_layers: A list of `TransformerLayer` instances.
        pooler: The pooling layer instance (if `add_pooling_layer` is True).

    Example:
        ```python
        # Create a BERT-base model configuration as a dictionary
        bert_config = create_bert_base_uncased()
        model = Bert(**bert_config, add_pooling_layer=True)

        # Use the model
        input_ids = keras.random.uniform((2, 128), 0, 30522, dtype='int32')
        sequence_output, pooled_output = model(input_ids)
        ```
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
        normalization_type: str = "layer_norm",
        normalization_position: str = "post",
        attention_type: str = "multi_head_attention",
        ffn_type: str = "mlp",
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        add_pooling_layer: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store all configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.add_pooling_layer = add_pooling_layer

        # Validate configuration
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
            raise ValueError("hidden_dropout_prob must be between 0 and 1")

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.embeddings = Embeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout_prob=self.hidden_dropout_prob,
            normalization_type=self.normalization_type,
            name="embeddings"
        )

        # Create transformer encoder layers
        self.encoder_layers: List[TransformerLayer] = []
        for i in range(self.num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                attention_type=self.attention_type,
                ffn_type=self.ffn_type,
                dropout_rate=self.hidden_dropout_prob,
                attention_dropout_rate=self.attention_probs_dropout_prob,
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=self.stochastic_depth_rate,
                activation=self.hidden_act,
                use_bias=True,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                bias_initializer="zeros",
                name=f"encoder_layer_{i}"
            )
            self.encoder_layers.append(transformer_layer)

        # Create pooler if needed
        self.pooler = None
        if self.add_pooling_layer:
            self.pooler = keras.layers.Dense(
                units=self.hidden_size,
                activation="tanh",
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name="pooler"
            )

        logger.info(
            f"Created BERT model with {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, pooling={self.add_pooling_layer}"
        )

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        return_dict: bool = False
    ) -> Union[
        keras.KerasTensor,
        Tuple[keras.KerasTensor, keras.KerasTensor],
        Dict[str, keras.KerasTensor]
    ]:
        """
        Forward pass of the BERT model.

        Args:
            inputs: Input token IDs or dictionary containing multiple inputs.
            attention_mask: Mask to avoid attention on padding tokens.
            token_type_ids: Token type IDs for distinguishing sequences.
            position_ids: Position IDs for positional embeddings.
            training: Boolean, whether the model is in training mode.
            return_dict: Boolean, whether to return outputs as a dictionary.

        Returns:
            Model outputs. The format depends on `return_dict` and `add_pooling_layer`:
            - `return_dict=False`, no pooling: `sequence_output` tensor.
            - `return_dict=False`, with pooling: `(sequence_output, pooled_output)` tuple.
            - `return_dict=True`: Dictionary with keys `last_hidden_state` and
              (if pooling) `pooler_output`.
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
            first_token_tensor = sequence_output[:, 0]
            pooled_output = self.pooler(first_token_tensor)

        # Return in requested format
        if return_dict:
            outputs = {"last_hidden_state": sequence_output}
            if pooled_output is not None:
                outputs["pooler_output"] = pooled_output
            return outputs
        else:
            if pooled_output is not None:
                return sequence_output, pooled_output
            else:
                return sequence_output

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "position_embedding_type": self.position_embedding_type,
            "use_cache": self.use_cache,
            "classifier_dropout": self.classifier_dropout,
            "normalization_type": self.normalization_type,
            "normalization_position": self.normalization_position,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "add_pooling_layer": self.add_pooling_layer,
        })
        return config

# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------


def create_bert_for_classification(
    config: Dict[str, Any],
    num_labels: int,
    classifier_dropout: Optional[float] = None
) -> keras.Model:
    """
    Create a BERT model for sequence classification tasks.

    This function builds a complete model by adding a classification head on top
    of the pooled output of a BERT model.

    Args:
        config: Dictionary containing BERT model hyperparameters.
        num_labels: Integer, the number of classification labels.
        classifier_dropout: Optional float, dropout rate for the classifier head.
            If None, it uses the dropout rate from the `config` dictionary.

    Returns:
        A complete `keras.Model` for sequence classification.

    Example:
        ```python
        bert_config = create_bert_base_uncased()
        model = create_bert_for_classification(bert_config, num_labels=2)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """
    logger.info(f"Creating BERT classification model with {num_labels} labels")

    # Create base BERT model with pooling
    bert = Bert(**config, add_pooling_layer=True, name="bert")

    # Define inputs using Keras Functional API
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
    final_dropout_rate = classifier_dropout
    if final_dropout_rate is None:
        final_dropout_rate = (
            config.get("classifier_dropout") or config.get("hidden_dropout_prob", 0.1)
        )

    if final_dropout_rate > 0.0:
        pooled_output = keras.layers.Dropout(
            final_dropout_rate,
            name="classifier_dropout"
        )(pooled_output)

    # Final classification layer
    logits = keras.layers.Dense(
        units=num_labels,
        kernel_initializer=keras.initializers.TruncatedNormal(
            stddev=config.get("initializer_range", 0.02)
        ),
        name="classifier"
    )(pooled_output)

    # Create the final model
    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=logits,
        name="bert_for_classification"
    )

    logger.info(
        f"Created BERT classification model with {model.count_params()} parameters"
    )
    return model


def create_bert_for_sequence_output(
    config: Dict[str, Any]
) -> keras.Model:
    """
    Create a BERT model for sequence-level output tasks.

    This function builds a BERT model that outputs sequence representations,
    suitable for tasks like token classification or question answering.

    Args:
        config: Dictionary containing BERT model hyperparameters.

    Returns:
        A `keras.Model` that returns sequence-level representations.

    Example:
        ```python
        bert_config = create_bert_base_uncased()
        model = create_bert_for_sequence_output(bert_config)

        # For token classification, add a classification head
        num_tags = 9  # e.g., for NER
        sequence_output = model.output
        logits = keras.layers.Dense(num_tags)(sequence_output)
        token_classifier = keras.Model(model.input, logits)
        ```
    """
    logger.info("Creating BERT model for sequence output tasks")

    # Create base BERT model without pooling
    bert = Bert(**config, add_pooling_layer=False, name="bert")

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

    logger.info(
        f"Created BERT sequence model with {model.count_params()} parameters"
    )
    return model


def create_bert_base_uncased() -> Dict[str, Any]:
    """
    Get configuration parameters for BERT-base-uncased model.

    Returns:
        A dictionary with parameters for the base model size.
    """
    return {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
    }


def create_bert_large_uncased() -> Dict[str, Any]:
    """
    Get configuration parameters for BERT-large-uncased model.

    Returns:
        A dictionary with parameters for the large model size.
    """
    return {
        "vocab_size": 30522,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "intermediate_size": 4096,
    }


def create_bert_with_rms_norm(
    size: str = "base",
    normalization_position: str = "pre"
) -> Dict[str, Any]:
    """
    Create BERT configuration with RMS normalization.

    Args:
        size: String, model size ('base' or 'large').
        normalization_position: String, position of normalization ('pre' or 'post').

    Returns:
        A dictionary of BERT configuration parameters with RMS normalization.

    Raises:
        ValueError: If an unsupported size is provided.
    """
    if size == "base":
        config = create_bert_base_uncased()
    elif size == "large":
        config = create_bert_large_uncased()
    else:
        raise ValueError(f"Unsupported size: {size}. Use 'base' or 'large'")

    config["normalization_type"] = "rms_norm"
    config["normalization_position"] = normalization_position

    logger.info(
        f"Created BERT-{size} config with RMS normalization "
        f"({normalization_position})"
    )
    return config


def create_bert_with_advanced_features(
    size: str = "base",
    normalization_type: str = "rms_norm",
    normalization_position: str = "pre",
    attention_type: str = "multi_head_attention",
    ffn_type: str = "swiglu",
    use_stochastic_depth: bool = True,
    stochastic_depth_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Create BERT configuration with advanced dl-techniques features.

    Args:
        size: String, model size ('base' or 'large').
        normalization_type: String, type of normalization layer.
        normalization_position: String, position of normalization.
        attention_type: String, type of attention mechanism.
        ffn_type: String, type of feed-forward network.
        use_stochastic_depth: Boolean, whether to use stochastic depth.
        stochastic_depth_rate: Float, drop path rate for stochastic depth.

    Returns:
        An advanced BERT configuration dictionary using dl-techniques features.
    """
    if size == "base":
        config = create_bert_base_uncased()
    elif size == "large":
        config = create_bert_large_uncased()
    else:
        raise ValueError(f"Unsupported size: {size}. Use 'base' or 'large'")

    # Apply advanced features
    config["normalization_type"] = normalization_type
    config["normalization_position"] = normalization_position
    config["attention_type"] = attention_type
    config["ffn_type"] = ffn_type
    config["use_stochastic_depth"] = use_stochastic_depth
    config["stochastic_depth_rate"] = stochastic_depth_rate

    logger.info(
        f"Created advanced BERT-{size} config with {normalization_type}, "
        f"{attention_type}, {ffn_type}"
    )
    return config

# ---------------------------------------------------------------------

import keras
from keras import initializers, layers
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .components import ModernBertEmbeddings, ModernBertEncoderLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBERT(keras.Model):
    """
    ModernBERT (A Modern Bidirectional Encoder) model implementation.

    This model refactors the original BERT architecture to include modern
    techniques like Rotary Position Embeddings (RoPE), GeGLU activations in the
    feed-forward network, and a mixture of global and local attention mechanisms
    to efficiently handle long sequences.

    **Intent**:
    Provide a flexible and powerful transformer encoder model that incorporates
    recent advancements in the field, suitable for a wide range of NLP tasks.
    It is designed to be highly configurable and serializable within the Keras
    ecosystem, following modern best practices.

    **Architecture**:
    ```
    Input (input_ids, token_type_ids)
           ↓
    ModernBertEmbeddings (Token, Type Embeddings + LayerNorm + Dropout)
           ↓
    ModernBertEncoderLayer x num_layers
      - Attention (Global or Local with RoPE)
      - FeedForward (GeGLU)
      - LayerNorm & Residuals
           ↓
    Final Layer Normalization
           ↓
    Sequence Output [batch, seq_len, hidden_size]
           ↓
    (Optional) Tanh Pooling on [CLS] token
           ↓
    Pooled Output [batch, hidden_size]
    ```

    Args:
        vocab_size: Integer, size of the vocabulary. Defaults to 50368.
        hidden_size: Integer, dimensionality of the encoder layers and the
            pooler layer. Defaults to 768.
        num_layers: Integer, number of hidden layers in the Transformer encoder.
            Defaults to 22.
        num_heads: Integer, number of attention heads for each attention layer in
            the Transformer encoder. Defaults to 12.
        intermediate_size: Integer, dimensionality of the "intermediate"
            (i.e., feed-forward) layer in the Transformer encoder.
            Defaults to 1152.
        hidden_act: String, the non-linear activation function in the encoder.
            Defaults to "gelu".
        hidden_dropout_prob: Float, the dropout probability for all fully
            connected layers in the embeddings and encoder. Defaults to 0.1.
        attention_probs_dropout_prob: Float, the dropout ratio for the attention
            probabilities. Defaults to 0.1.
        type_vocab_size: Integer, the vocabulary size of the `token_type_ids`.
            Defaults to 2.
        initializer_range: Float, the standard deviation of the truncated_normal
            initializer for initializing all weight matrices. Defaults to 0.02.
        layer_norm_eps: Float, the epsilon used by the layer normalization layers.
            Defaults to 1e-12.
        use_bias: Boolean, whether to use bias vectors in layers. Defaults to False.
        rope_theta_local: Float, theta value for local attention RoPE.
            Defaults to 10000.0.
        rope_theta_global: Float, theta value for global attention RoPE.
            Defaults to 160000.0.
        max_seq_len: Integer, maximum sequence length for RoPE. Defaults to 8192.
        global_attention_interval: Integer, interval at which to apply global
            attention. A value of 3 means every 3rd layer is global.
            Defaults to 3.
        local_attention_window_size: Integer, window size for local attention.
            Defaults to 128.
        add_pooling_layer: Boolean, whether to add a pooling layer to extract a
            fixed-size representation of the sequence. Defaults to True.
        **kwargs: Additional arguments for the `keras.Model` base class.

    Input shape:
        A dictionary containing:
        - `input_ids`: 2D tensor of shape `(batch_size, sequence_length)`.
        - `attention_mask`: (Optional) 2D tensor of shape `(batch_size, sequence_length)`.
        - `token_type_ids`: (Optional) 2D tensor of shape `(batch_size, sequence_length)`.
        Alternatively, can be a single 2D tensor representing `input_ids`.

    Output shape:
        If `return_dict=False`:
            - A tuple of `(sequence_output, pooled_output)` if `add_pooling_layer=True`.
            - `sequence_output` tensor if `add_pooling_layer=False`.
        If `return_dict=True`:
            - A dictionary with keys `"last_hidden_state"` and optionally
              `"pooler_output"`.
        `sequence_output` has shape `(batch_size, sequence_length, hidden_size)`.
        `pooled_output` has shape `(batch_size, hidden_size)`.

    Attributes:
        embeddings: `ModernBertEmbeddings` layer instance.
        encoder_layers: List of `ModernBertEncoderLayer` instances.
        final_norm: Final `LayerNormalization` layer.
        pooler: `Dense` layer for pooling if `add_pooling_layer=True`.

    Example:
        ```python
        config = { "vocab_size": 30000, "hidden_size": 768, "num_layers": 12 }
        model = ModernBERT(**config)
        input_ids = keras.ops.random.randint(0, 30000, (2, 512))
        # Get sequence and pooled outputs
        sequence, pooled = model(input_ids)
        ```
    """

    def __init__(
        self,
        vocab_size: int = 50368,
        hidden_size: int = 768,
        num_layers: int = 22,
        num_heads: int = 12,
        intermediate_size: int = 1152,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_bias: bool = False,
        rope_theta_local: float = 10000.0,
        rope_theta_global: float = 160000.0,
        max_seq_len: int = 8192,
        global_attention_interval: int = 3,
        local_attention_window_size: int = 128,
        add_pooling_layer: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Store configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.max_seq_len = max_seq_len
        self.global_attention_interval = global_attention_interval
        self.local_attention_window_size = local_attention_window_size
        self.add_pooling_layer = add_pooling_layer

        # Create sub-layers in __init__
        self.embeddings = ModernBertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout_prob=self.hidden_dropout_prob,
            use_bias=self.use_bias,
            name="embeddings",
        )

        # Group encoder layer args into a dict for cleaner instantiation
        encoder_args = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "use_bias": self.use_bias,
            "rope_theta_local": self.rope_theta_local,
            "rope_theta_global": self.rope_theta_global,
            "max_seq_len": self.max_seq_len,
            "local_attention_window_size": self.local_attention_window_size,
        }

        self.encoder_layers = []
        for i in range(self.num_layers):
            is_global = (i + 1) % self.global_attention_interval == 0
            layer = ModernBertEncoderLayer(
                is_global=is_global, name=f"encoder_layer_{i}", **encoder_args
            )
            self.encoder_layers.append(layer)

        self.final_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=self.use_bias, name="final_layer_norm"
        )

        self.pooler = None
        if self.add_pooling_layer:
            self.pooler = layers.Dense(
                units=self.hidden_size,
                activation="tanh",
                use_bias=self.use_bias,
                kernel_initializer=initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name="pooler",
            )

    def call(
        self,
        inputs,
        attention_mask=None,
        token_type_ids=None,
        training=None,
        return_dict=False,
    ):
        """
        Forward pass for the ModernBERT model.

        Args:
            inputs: Can be a dictionary with keys 'input_ids', 'attention_mask',
                'token_type_ids', or a single tensor representing `input_ids`.
            attention_mask: (Optional) Tensor of shape (batch_size, sequence_length)
                with 1s for valid tokens and 0s for padding. Used to avoid
                performing attention on padding token indices.
            token_type_ids: (Optional) Tensor of shape (batch_size, sequence_length)
                for segment embeddings (e.g., distinguishing question and answer).
            training: (Optional) Boolean, indicating if the model is in
                training mode. Passed to layers like Dropout.
            return_dict: (Optional) Boolean, if True, the model returns a
                dictionary of outputs. Otherwise, it returns a tensor or tuple.

        Returns:
            If `return_dict` is True, a dictionary containing
            `"last_hidden_state"` and optionally `"pooler_output"`.
            Otherwise, returns a tuple `(sequence_output, pooled_output)` if
            pooling is enabled, or just `sequence_output`.
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
        else:
            input_ids = inputs
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, training=training
        )
        hidden_states = embedding_output
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, training=training
            )

        sequence_output = self.final_norm(hidden_states, training=training)
        pooled_output = None
        if self.pooler is not None:
            # The pooler operates on the first token's representation ([CLS])
            first_token_tensor = sequence_output[:, 0]
            pooled_output = self.pooler(first_token_tensor, training=training)

        if return_dict:
            outputs = {"last_hidden_state": sequence_output}
            if pooled_output is not None:
                outputs["pooler_output"] = pooled_output
            return outputs
        else:
            if pooled_output is not None:
                return sequence_output, pooled_output
            return sequence_output

    def get_config(self) -> Dict[str, Any]:
        """Returns the model's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "hidden_act": self.hidden_act,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
                "type_vocab_size": self.type_vocab_size,
                "initializer_range": self.initializer_range,
                "layer_norm_eps": self.layer_norm_eps,
                "use_bias": self.use_bias,
                "rope_theta_local": self.rope_theta_local,
                "rope_theta_global": self.rope_theta_global,
                "max_seq_len": self.max_seq_len,
                "global_attention_interval": self.global_attention_interval,
                "local_attention_window_size": self.local_attention_window_size,
                "add_pooling_layer": self.add_pooling_layer,
            }
        )
        return config


# ---------------------------------------------------------------------


def create_modern_bert_base() -> Dict[str, Any]:
    """
    Creates the configuration dictionary for a ModernBERT-base model.

    This function provides a standard set of hyperparameters for a 'base' sized
    ModernBERT, which can be passed directly to the ModernBERT model's
    constructor.

    Architecture:
        - 22 layers
        - 768 hidden size
        - 12 attention heads
        - 1152 intermediate size

    Returns:
        A dictionary containing the configuration parameters for the model.
    """
    logger.info("Creating ModernBERT-base configuration dictionary")
    return {
        "vocab_size": 50368,
        "hidden_size": 768,
        "num_layers": 22,
        "num_heads": 12,
        "intermediate_size": 1152,
        "hidden_act": "gelu",
        "use_bias": False,
        "global_attention_interval": 3,
        "local_attention_window_size": 128,
        "max_seq_len": 8192,
    }


def create_modern_bert_large() -> Dict[str, Any]:
    """
    Creates the configuration dictionary for a ModernBERT-large model.

    This function provides a standard set of hyperparameters for a 'large' sized
    ModernBERT, which can be passed directly to the ModernBERT model's
    constructor.

    Architecture:
        - 28 layers
        - 1024 hidden size
        - 16 attention heads
        - 2624 intermediate size

    Returns:
        A dictionary containing the configuration parameters for the model.
    """
    logger.info("Creating ModernBERT-large configuration dictionary")
    return {
        "vocab_size": 50368,
        "hidden_size": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "intermediate_size": 2624,
        "hidden_act": "gelu",
        "use_bias": False,
        "global_attention_interval": 3,
        "local_attention_window_size": 128,
        "max_seq_len": 8192,
    }


def create_modern_bert_for_classification(
    config: Dict[str, Any],
    num_labels: int,
    classifier_dropout: Optional[float] = None,
) -> keras.Model:
    """
    Builds a Keras Model for sequence classification using ModernBERT.

    This function instantiates a ModernBERT model from the provided configuration
    and attaches a classification head on top of the pooled output. The head
    consists of an optional dropout layer and a dense layer for logits.

    Args:
        config: A dictionary of parameters to initialize the ModernBERT model.
        num_labels: The number of classes for the classification task.
        classifier_dropout: (Optional) Float, the dropout rate for the
            classification head. If None, it defaults to the `hidden_dropout_prob`
            from the main config.

    Returns:
        A `keras.Model` instance ready for compilation and training.
    """
    logger.info(f"Creating ModernBERT classification model with {num_labels} labels")
    input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = layers.Input(shape=(None,), dtype="int32", name="token_type_ids")

    modern_bert = ModernBERT(add_pooling_layer=True, name="modern_bert", **config)

    bert_outputs = modern_bert(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        return_dict=True,
    )
    pooled_output = bert_outputs["pooler_output"]

    dropout_rate = (
        classifier_dropout
        if classifier_dropout is not None
        else config.get("hidden_dropout_prob", 0.1)
    )
    if dropout_rate > 0.0:
        pooled_output = layers.Dropout(dropout_rate, name="classifier_dropout")(
            pooled_output
        )

    logits = layers.Dense(
        units=num_labels,
        kernel_initializer=initializers.TruncatedNormal(
            stddev=config.get("initializer_range", 0.02)
        ),
        name="classifier",
    )(pooled_output)

    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=logits,
        name="modern_bert_for_classification",
    )
    logger.info(
        f"Created ModernBERT classification model with {model.count_params()} parameters"
    )
    return model


def create_modern_bert_for_sequence_output(config: Dict[str, Any]) -> keras.Model:
    """
    Builds a Keras Model that returns sequence outputs from ModernBERT.

    This function instantiates a ModernBERT model from the provided configuration
    that is optimized for sequence-level tasks (e.g., token classification,
    feature extraction) by returning the last hidden state for all tokens.

    Args:
        config: A dictionary of parameters to initialize the ModernBERT model.

    Returns:
        A `keras.Model` instance that outputs the final hidden states.
    """
    logger.info("Creating ModernBERT model for sequence output")
    input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = layers.Input(shape=(None,), dtype="int32", name="token_type_ids")

    modern_bert = ModernBERT(add_pooling_layer=False, name="modern_bert", **config)

    sequence_output = modern_bert(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    )

    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=sequence_output,
        name="modern_bert_for_sequence_output",
    )
    logger.info(
        f"Created ModernBERT sequence model with {model.count_params()} parameters"
    )
    return model


# ---------------------------------------------------------------------

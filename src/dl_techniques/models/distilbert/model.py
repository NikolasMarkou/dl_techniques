"""
DistilBERT Model Implementation with Pretrained Support
========================================================

A complete implementation of the DistilBERT (Distilled BERT) architecture with
support for loading pretrained weights. DistilBERT is a smaller, faster, cheaper
and lighter version of BERT that retains 97% of BERT's language understanding
capabilities while being 40% smaller and 60% faster.

Based on: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
(Sanh et al., 2019) https://arxiv.org/abs/1910.01108

Key Architectural Differences from BERT:
    - Number of layers reduced by half (6 vs 12 for base)
    - Token type embeddings removed
    - Pooler layer removed
    - Optional sinusoidal position embeddings

Usage Examples:
--------------

.. code-block:: python

    import keras
    from dl_techniques.nlp.heads.factory import create_nlp_head
    from dl_techniques.nlp.heads.task_types import NLPTaskConfig, NLPTaskType

    # 1. Load pretrained DistilBERT model
    distilbert_encoder = DistilBERT.from_variant("base", pretrained=True)

    # 2. Load from local weights file
    distilbert_encoder = DistilBERT.from_variant("base", pretrained="path/to/weights.keras")

    # 3. Create DistilBERT with custom configuration
    distilbert_encoder = DistilBERT.from_variant("base", vocab_size=50000)

    # 4. Combine with task-specific head
    sentiment_config = NLPTaskConfig(
        name="sentiment",
        task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        num_classes=3
    )
    sentiment_head = create_nlp_head(
        task_config=sentiment_config,
        input_dim=distilbert_encoder.hidden_size
    )

    # 5. Build complete model
    inputs = {
        "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
    }
    distilbert_outputs = distilbert_encoder(inputs)
    task_outputs = sentiment_head(distilbert_outputs)
    sentiment_model = keras.Model(inputs, task_outputs)

"""

import os
import keras
from typing import Optional, Union, Any, Dict, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers import (
    FFNType,
    AttentionType,
    TransformerLayer,
    NormalizationType,
    NormalizationPositionType,
)
from dl_techniques.layers.norms import RMSNorm
from dl_techniques.layers.nlp_heads import create_nlp_head, NLPTaskConfig

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DistilBertEmbeddings(keras.layers.Layer):
    """Embeddings layer for DistilBERT.

    Unlike BERT, DistilBERT does not use token type embeddings. This layer
    combines word embeddings with position embeddings only.

    Supports both learned and sinusoidal position embeddings as per the
    original DistilBERT implementation.

    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param hidden_size: Dimensionality of the embeddings.
    :type hidden_size: int
    :param max_position_embeddings: Maximum sequence length for positional
        embeddings.
    :type max_position_embeddings: int
    :param sinusoidal_pos_embds: Whether to use sinusoidal position embeddings
        instead of learned ones.
    :type sinusoidal_pos_embds: bool
    :param initializer_range: Standard deviation for weight initialization.
    :type initializer_range: float
    :param layer_norm_eps: Epsilon for layer normalization.
    :type layer_norm_eps: float
    :param dropout_rate: Dropout probability.
    :type dropout_rate: float
    :param normalization_type: Type of normalization to use.
    :type normalization_type: str
    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 512,
        sinusoidal_pos_embds: bool = False,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        dropout_rate: float = 0.1,
        normalization_type: str = "layer_norm",
        **kwargs: Any
    ) -> None:
        """Initialize DistilBERT embeddings layer."""
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout_rate
        self.normalization_type = normalization_type

        # Word embeddings
        self.word_embeddings = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            name="word_embeddings"
        )

        # Position embeddings (learned or sinusoidal)
        if not sinusoidal_pos_embds:
            self.position_embeddings = keras.layers.Embedding(
                input_dim=max_position_embeddings,
                output_dim=hidden_size,
                embeddings_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                name="position_embeddings"
            )
        else:
            self.position_embeddings = None

        # Normalization layer
        if normalization_type == "layer_norm":
            self.layer_norm = keras.layers.LayerNormalization(
                epsilon=layer_norm_eps,
                name="layer_norm"
            )
        elif normalization_type == "rms_norm":
            self.layer_norm = RMSNorm(
                epsilon=layer_norm_eps,
                name="layer_norm"
            )
        else:
            self.layer_norm = keras.layers.LayerNormalization(
                epsilon=layer_norm_eps,
                name="layer_norm"
            )

        self.dropout = keras.layers.Dropout(rate=dropout_rate)

    def _create_sinusoidal_embeddings(
        self,
        seq_length: int
    ) -> keras.KerasTensor:
        """Create sinusoidal position embeddings.

        :param seq_length: Length of the sequence.
        :type seq_length: int
        :return: Sinusoidal position embeddings of shape (1, seq_length, hidden_size).
        :rtype: keras.KerasTensor
        """
        position = keras.ops.arange(seq_length, dtype="float32")
        position = keras.ops.expand_dims(position, axis=1)

        div_term = keras.ops.exp(
            keras.ops.arange(0, self.hidden_size, 2, dtype="float32")
            * -(keras.ops.log(10000.0) / self.hidden_size)
        )

        sin_embeddings = keras.ops.sin(position * div_term)
        cos_embeddings = keras.ops.cos(position * div_term)

        # Interleave sin and cos
        position_embeddings = keras.ops.zeros((seq_length, self.hidden_size))
        indices_sin = keras.ops.arange(0, self.hidden_size, 2)
        indices_cos = keras.ops.arange(1, self.hidden_size, 2)

        # Use scatter or concatenation approach
        sin_part = sin_embeddings
        cos_part = cos_embeddings

        # Stack and reshape to interleave
        stacked = keras.ops.stack([sin_part, cos_part], axis=-1)
        position_embeddings = keras.ops.reshape(
            stacked, (seq_length, self.hidden_size)
        )

        return keras.ops.expand_dims(position_embeddings, axis=0)

    def call(
        self,
        input_ids: keras.KerasTensor,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the embeddings layer.

        :param input_ids: Token IDs of shape (batch_size, seq_length).
        :type input_ids: keras.KerasTensor
        :param position_ids: Position IDs of shape (batch_size, seq_length).
            If None, positions are generated automatically.
        :type position_ids: Optional[keras.KerasTensor]
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Embedded representations of shape
            (batch_size, seq_length, hidden_size).
        :rtype: keras.KerasTensor
        """
        seq_length = keras.ops.shape(input_ids)[1]

        # Word embeddings
        word_embeds = self.word_embeddings(input_ids)

        # Position embeddings
        if self.sinusoidal_pos_embds:
            position_embeds = self._create_sinusoidal_embeddings(seq_length)
        else:
            if position_ids is None:
                position_ids = keras.ops.arange(seq_length)
                position_ids = keras.ops.expand_dims(position_ids, axis=0)
            position_embeds = self.position_embeddings(position_ids)

        # Combine embeddings
        embeddings = word_embeds + position_embeds

        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "max_position_embeddings": self.max_position_embeddings,
            "sinusoidal_pos_embds": self.sinusoidal_pos_embds,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "dropout_rate": self.dropout_rate,
            "normalization_type": self.normalization_type,
        })
        return config


@keras.saving.register_keras_serializable()
class DistilBERT(keras.Model):
    """DistilBERT (Distilled BERT) model.

    DistilBERT is a smaller, faster, and lighter transformer model that
    retains 97% of BERT's language understanding capabilities while being
    40% smaller and 60% faster. It achieves this through knowledge
    distillation during pre-training.

    Key differences from BERT:
        - Number of layers reduced by half
        - Token type embeddings removed
        - Optional sinusoidal position embeddings
        - No pooler layer

    The model expects inputs as a dictionary containing 'input_ids', and
    optionally 'attention_mask' and 'position_ids'. It outputs a dictionary
    containing the 'last_hidden_state' and the forwarded 'attention_mask'.

    **Architecture Overview:**

    .. code-block:: text

        Input(input_ids, attention_mask)
               │
               ▼
        Embeddings(Word + Position) -> LayerNorm -> Dropout
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
        Output Dictionary {
            "last_hidden_state": [batch, seq_len, hidden_size],
            "attention_mask": [batch, seq_len]
        }

    :param vocab_size: Size of the vocabulary. Defaults to 30522.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Defaults to 768.
    :type hidden_size: int
    :param num_layers: Number of hidden transformer layers. Defaults to 6.
    :type num_layers: int
    :param num_heads: Number of attention heads. Defaults to 12.
    :type num_heads: int
    :param intermediate_size: Dimensionality of the FFN layer. Defaults to 3072.
    :type intermediate_size: int
    :param hidden_act: Activation function in the encoder. Defaults to "gelu".
    :type hidden_act: str
    :param dropout_rate: Dropout probability for embeddings/encoder.
        Defaults to 0.1.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout for attention probabilities.
        Defaults to 0.1.
    :type attention_dropout_rate: float
    :param max_position_embeddings: Maximum sequence length. Defaults to 512.
    :type max_position_embeddings: int
    :param sinusoidal_pos_embds: Whether to use sinusoidal position embeddings.
        Defaults to False.
    :type sinusoidal_pos_embds: bool
    :param initializer_range: Stddev for weight initialization. Defaults to 0.02.
    :type initializer_range: float
    :param layer_norm_eps: Epsilon for normalization layers. Defaults to 1e-12.
    :type layer_norm_eps: float
    :param pad_token_id: ID of padding token. Defaults to 0.
    :type pad_token_id: int
    :param normalization_type: Type of normalization layer.
        Defaults to "layer_norm".
    :type normalization_type: str
    :param normalization_position: Position of normalization ('pre' or 'post').
        Defaults to "post".
    :type normalization_position: str
    :param attention_type: Type of attention mechanism.
        Defaults to "multi_head".
    :type attention_type: str
    :param ffn_type: Type of feed-forward network. Defaults to "mlp".
    :type ffn_type: str
    :param use_stochastic_depth: Whether to enable stochastic depth.
        Defaults to False.
    :type use_stochastic_depth: bool
    :param stochastic_depth_rate: Drop path rate for stochastic depth.
        Defaults to 0.1.
    :type stochastic_depth_rate: float
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :ivar embeddings: The embedding layer instance.
    :vartype embeddings: DistilBertEmbeddings
    :ivar encoder_layers: A list of `TransformerLayer` instances.
    :vartype encoder_layers: list[TransformerLayer]

    :raises ValueError: If invalid configuration parameters are provided.

    Example:
        .. code-block:: python

            # Create standard DistilBERT model
            model = DistilBERT.from_variant("base")

            # Load pretrained DistilBERT
            model = DistilBERT.from_variant("base", pretrained=True)

            # Load from local file
            model = DistilBERT.from_variant("base", pretrained="path/to/weights.keras")

            # Use the model
            inputs = {
                "input_ids": keras.random.uniform((2, 128), 0, 30522, dtype="int32"),
                "attention_mask": keras.ops.ones((2, 128), dtype="int32")
            }
            outputs = model(inputs)
            print(outputs["last_hidden_state"].shape)
            # (2, 128, 768)

    """

    # Model variant configurations following DistilBERT specifications
    MODEL_VARIANTS = {
        "base": {
            "hidden_size": 768,
            "num_layers": 6,
            "num_heads": 12,
            "intermediate_size": 3072,
            "description": "DistilBERT-Base: 66M parameters, 40% smaller and 60% faster than BERT-Base"
        },
        "small": {
            "hidden_size": 512,
            "num_layers": 4,
            "num_heads": 8,
            "intermediate_size": 2048,
            "description": "DistilBERT-Small: Lightweight variant for resource-constrained environments"
        },
        "tiny": {
            "hidden_size": 256,
            "num_layers": 2,
            "num_heads": 4,
            "intermediate_size": 1024,
            "description": "DistilBERT-Tiny: Ultra-lightweight for mobile/edge deployment"
        },
    }

    # Pretrained weights URLs (update these with actual URLs when available)
    PRETRAINED_WEIGHTS = {
        "base": {
            "uncased": "https://example.com/distilbert_base_uncased.keras",
            "cased": "https://example.com/distilbert_base_cased.keras",
            "multilingual": "https://example.com/distilbert_base_multilingual.keras",
        },
        "small": {
            "uncased": "https://example.com/distilbert_small_uncased.keras",
        },
        "tiny": {
            "uncased": "https://example.com/distilbert_tiny_uncased.keras",
        },
    }

    # Default architecture constants
    DEFAULT_VOCAB_SIZE = 30522
    DEFAULT_MAX_POSITION_EMBEDDINGS = 512
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPSILON = 1e-12
    DEFAULT_HIDDEN_ACT = "gelu"
    DEFAULT_PAD_TOKEN_ID = 0

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = DEFAULT_HIDDEN_ACT,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS,
        sinusoidal_pos_embds: bool = False,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        normalization_type: NormalizationType = "layer_norm",
        normalization_position: NormalizationPositionType = "post",
        attention_type: AttentionType = "multi_head",
        ffn_type: FFNType = "mlp",
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        **kwargs: Any
    ) -> None:
        """Initialize the DistilBERT model instance.

        :param vocab_size: Size of the vocabulary.
        :type vocab_size: int
        :param hidden_size: Dimensionality of encoder layers.
        :type hidden_size: int
        :param num_layers: Number of hidden transformer layers.
        :type num_layers: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param intermediate_size: Dimensionality of the FFN layer.
        :type intermediate_size: int
        :param hidden_act: Activation function in the encoder.
        :type hidden_act: str
        :param dropout_rate: Dropout probability for embeddings/encoder.
        :type dropout_rate: float
        :param attention_dropout_rate: Dropout for attention scores.
        :type attention_dropout_rate: float
        :param max_position_embeddings: Maximum sequence length.
        :type max_position_embeddings: int
        :param sinusoidal_pos_embds: Whether to use sinusoidal position embeddings.
        :type sinusoidal_pos_embds: bool
        :param initializer_range: Stddev for weight initialization.
        :type initializer_range: float
        :param layer_norm_eps: Epsilon for normalization layers.
        :type layer_norm_eps: float
        :param pad_token_id: ID of the padding token.
        :type pad_token_id: int
        :param normalization_type: Type of normalization layer.
        :type normalization_type: str
        :param normalization_position: Position of normalization ('pre'/'post').
        :type normalization_position: str
        :param attention_type: Type of attention mechanism.
        :type attention_type: str
        :param ffn_type: Type of feed-forward network.
        :type ffn_type: str
        :param use_stochastic_depth: Whether to enable stochastic depth.
        :type use_stochastic_depth: bool
        :param stochastic_depth_rate: Drop rate for stochastic depth.
        :type stochastic_depth_rate: float
        :param kwargs: Additional keyword arguments for the `keras.Model`.
        """
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_heads,
            dropout_rate, attention_dropout_rate
        )

        # Store all configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate

        # Build the model architecture
        self._build_architecture()

        logger.info(
            f"Created DistilBERT foundation model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, heads={self.num_heads}"
        )

    def _validate_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float
    ) -> None:
        """Validate model configuration parameters.

        :param vocab_size: Size of the vocabulary.
        :type vocab_size: int
        :param hidden_size: Dimensionality of encoder layers.
        :type hidden_size: int
        :param num_layers: Number of transformer layers.
        :type num_layers: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param dropout_rate: Dropout probability for hidden layers.
        :type dropout_rate: float
        :param attention_dropout_rate: Dropout for attention scores.
        :type attention_dropout_rate: float
        :raises ValueError: If any configuration value is invalid.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0 and 1, "
                f"got {dropout_rate}"
            )
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be between 0 and 1, "
                f"got {attention_dropout_rate}"
            )

    def _build_architecture(self) -> None:
        """Build all model components (embeddings and encoder layers)."""
        self.embeddings = DistilBertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            sinusoidal_pos_embds=self.sinusoidal_pos_embds,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            dropout_rate=self.dropout_rate,
            normalization_type=self.normalization_type,
            name="embeddings"
        )

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
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=self.stochastic_depth_rate,
                activation=self.hidden_act,
                use_bias=True,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                bias_initializer="zeros",
                name=f"transformer_layer_{i}"
            )
            self.encoder_layers.append(transformer_layer)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of the DistilBERT foundation model.

        :param inputs: Input token IDs or a dictionary containing 'input_ids'
            and other optional tensors like 'attention_mask'.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param attention_mask: Mask to avoid attention on padding tokens.
        :type attention_mask: Optional[keras.KerasTensor]
        :param position_ids: Position IDs for positional embeddings.
        :type position_ids: Optional[keras.KerasTensor]
        :param training: Indicates if the model is in training mode.
        :type training: Optional[bool]
        :return: A dictionary with the following keys:
                 - ``last_hidden_state``: The sequence of hidden states at the
                   output of the final layer. Shape:
                   `(batch, seq_len, hidden_size)`.
                 - ``attention_mask``: The original attention mask, passed
                   through for convenience in downstream models.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If dictionary input does not contain 'input_ids'.
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            attention_mask = inputs.get("attention_mask", attention_mask)
            position_ids = inputs.get("position_ids", position_ids)
        else:
            input_ids = inputs

        # Get embeddings (no token_type_ids for DistilBERT)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            training=training
        )

        # Pass through encoder layers
        hidden_states = embedding_output
        for i, encoder_layer in enumerate(self.encoder_layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_idx=i,
                training=training
            )

        return {
            "last_hidden_state": hidden_states,
            "attention_mask": attention_mask
        }

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True,
        by_name: bool = True
    ) -> None:
        """Load pretrained weights into the model.

        This method handles loading weights with smart mismatch handling,
        particularly useful when the vocabulary size or architecture differs
        slightly from the pretrained model.

        :param weights_path: Path to the weights file (.keras format).
        :type weights_path: str
        :param skip_mismatch: Whether to skip layers with mismatched shapes.
            Useful when loading weights with different vocab_size or config.
        :type skip_mismatch: bool
        :param by_name: Whether to load weights by layer name.
        :type by_name: bool
        :raises FileNotFoundError: If weights_path doesn't exist.
        :raises ValueError: If weights cannot be loaded.

        Example:
            .. code-block:: python

                model = DistilBERT.from_variant("base", vocab_size=50000)
                model.load_pretrained_weights(
                    "distilbert_base_uncased.keras",
                    skip_mismatch=True
                )
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            # Build model if not already built
            if not self.built:
                dummy_input = {
                    "input_ids": keras.random.uniform(
                        (1, 128), 0, self.vocab_size, dtype="int32"
                    ),
                    "attention_mask": keras.ops.ones((1, 128), dtype="int32")
                }
                self(dummy_input, training=False)

            logger.info(f"Loading pretrained weights from {weights_path}")

            # Load weights with appropriate settings
            self.load_weights(
                weights_path,
                skip_mismatch=skip_mismatch,
                by_name=by_name
            )

            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. "
                    "Layers with shape mismatches were skipped (e.g., embedding layer)."
                )
            else:
                logger.info("All weights loaded successfully.")

        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

    @staticmethod
    def _download_weights(
        variant: str,
        dataset: str = "uncased",
        cache_dir: Optional[str] = None
    ) -> str:
        """Download pretrained weights from URL.

        :param variant: Model variant name.
        :type variant: str
        :param dataset: Dataset/version the weights were trained on.
            Options: "uncased", "cased", "multilingual".
        :type dataset: str
        :param cache_dir: Directory to cache downloaded weights.
            If None, uses default Keras cache directory.
        :type cache_dir: Optional[str]
        :return: Path to the downloaded weights file.
        :rtype: str
        :raises ValueError: If variant or dataset is not available.

        Example:
            .. code-block:: python

                weights_path = DistilBERT._download_weights("base", "uncased")
        """
        if variant not in DistilBERT.PRETRAINED_WEIGHTS:
            raise ValueError(
                f"No pretrained weights available for variant '{variant}'. "
                f"Available variants: {list(DistilBERT.PRETRAINED_WEIGHTS.keys())}"
            )

        if dataset not in DistilBERT.PRETRAINED_WEIGHTS[variant]:
            raise ValueError(
                f"No pretrained weights available for dataset '{dataset}'. "
                f"Available datasets for {variant}: "
                f"{list(DistilBERT.PRETRAINED_WEIGHTS[variant].keys())}"
            )

        url = DistilBERT.PRETRAINED_WEIGHTS[variant][dataset]

        logger.info(f"Downloading DistilBERT-{variant} ({dataset}) weights...")

        # Download weights using Keras utility
        weights_path = keras.utils.get_file(
            fname=f"distilbert_{variant}_{dataset}.keras",
            origin=url,
            cache_dir=cache_dir,
            cache_subdir="models/distilbert"
        )

        logger.info(f"Weights downloaded to: {weights_path}")
        return weights_path

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "uncased",
        cache_dir: Optional[str] = None,
        **kwargs: Any
    ) -> "DistilBERT":
        """Create a DistilBERT model from a predefined variant.

        :param variant: The name of the variant, one of "base", "small", "tiny".
        :type variant: str
        :param pretrained: If True, loads pretrained weights from default URL.
            If string, treats it as a path to local weights file.
        :type pretrained: Union[bool, str]
        :param weights_dataset: Dataset/version for pretrained weights.
            Options: "uncased", "cased", "multilingual".
            Only used if pretrained=True.
        :type weights_dataset: str
        :param cache_dir: Directory to cache downloaded weights.
        :type cache_dir: Optional[str]
        :param kwargs: Additional arguments to override the variant's defaults.
        :type kwargs: Any
        :return: A DistilBERT model instance configured for the specified variant.
        :rtype: DistilBERT
        :raises ValueError: If the specified variant is not recognized.

        Example:
            .. code-block:: python

                # Load pretrained DistilBERT-base
                model = DistilBERT.from_variant("base", pretrained=True)

                # Load pretrained DistilBERT-base (cased)
                model = DistilBERT.from_variant("base", pretrained=True, weights_dataset="cased")

                # Load from local file
                model = DistilBERT.from_variant("base", pretrained="path/to/weights.keras")

                # Create with custom vocab size (will skip embedding weights)
                model = DistilBERT.from_variant("base", pretrained=True, vocab_size=50000)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")

        logger.info(f"Creating DistilBERT-{variant.upper()} model")
        logger.info(f"Configuration: {description}")

        # Handle pretrained weights
        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                # Load from local file path
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                # Download from URL
                try:
                    load_weights_path = cls._download_weights(
                        variant=variant,
                        dataset=weights_dataset,
                        cache_dir=cache_dir
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {str(e)}. "
                        f"Continuing with random initialization."
                    )
                    load_weights_path = None

            # Determine if we need to skip mismatches
            # Check if vocab_size differs from default
            pretrained_vocab_size = cls.DEFAULT_VOCAB_SIZE
            custom_vocab_size = kwargs.get("vocab_size", config.get("vocab_size"))

            if custom_vocab_size and custom_vocab_size != pretrained_vocab_size:
                skip_mismatch = True
                logger.info(
                    f"vocab_size ({custom_vocab_size}) differs from pretrained "
                    f"({pretrained_vocab_size}). Will skip embedding layer weights."
                )

            # Check if other architectural parameters differ
            pretrained_config_keys = ["hidden_size", "num_layers", "num_heads", "intermediate_size"]
            for key in pretrained_config_keys:
                if key in kwargs and kwargs[key] != config.get(key):
                    skip_mismatch = True
                    logger.info(
                        f"{key} differs from pretrained configuration. "
                        f"Will skip layers with shape mismatches."
                    )

        # Override defaults with kwargs
        config.update(kwargs)

        # Create model
        model = cls(**config)

        # Load pretrained weights if available
        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path,
                    skip_mismatch=skip_mismatch,
                    by_name=True
                )
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {str(e)}")
                raise

        return model

    def get_config(self) -> Dict[str, Any]:
        """Return the model's configuration for serialization.

        :return: A dictionary containing the model's configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "max_position_embeddings": self.max_position_embeddings,
            "sinusoidal_pos_embds": self.sinusoidal_pos_embds,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "normalization_type": self.normalization_type,
            "normalization_position": self.normalization_position,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DistilBERT":
        """Create a model instance from its configuration.

        :param config: A dictionary containing the model's configuration.
        :type config: Dict[str, Any]
        :return: A new DistilBERT model instance.
        :rtype: DistilBERT
        """
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the model summary with additional DistilBERT-specific information.

        :param kwargs: Additional arguments passed to `keras.Model.summary`.
        """
        super().summary(**kwargs)
        logger.info("DistilBERT Foundation Model Configuration:")
        logger.info(
            f"  - Architecture: {self.num_layers} layers, "
            f"{self.hidden_size} hidden size"
        )
        logger.info(
            f"  - Attention: {self.num_heads} heads, {self.attention_type}"
        )
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(
            f"  - Max sequence length: {self.max_position_embeddings}"
        )
        logger.info(
            f"  - Position embeddings: "
            f"{'sinusoidal' if self.sinusoidal_pos_embds else 'learned'}"
        )
        logger.info(
            f"  - Normalization: {self.normalization_type} "
            f"({self.normalization_position})"
        )
        logger.info(
            f"  - Feed-forward: {self.ffn_type}, "
            f"{self.intermediate_size} intermediate size"
        )
        if self.use_stochastic_depth:
            logger.info(
                "  - Stochastic depth enabled: "
                f"rate={self.stochastic_depth_rate}"
            )


# ---------------------------------------------------------------------
# Integration with NLP Task Heads
# ---------------------------------------------------------------------


def create_distilbert_with_head(
    distilbert_variant: str,
    task_config: NLPTaskConfig,
    pretrained: Union[bool, str] = False,
    weights_dataset: str = "uncased",
    cache_dir: Optional[str] = None,
    distilbert_config_overrides: Optional[Dict[str, Any]] = None,
    head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Factory function to create a DistilBERT model with a task-specific head.

    This function demonstrates the intended integration pattern:
    1. Instantiate a foundational `DistilBERT` model (optionally pretrained).
    2. Instantiate a task-specific head from the `dl_techniques.nlp.heads`
       factory.
    3. Combine them into a single, end-to-end `keras.Model`.

    :param distilbert_variant: The DistilBERT variant to use (e.g., "base", "small").
    :type distilbert_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task.
    :type task_config: NLPTaskConfig
    :param pretrained: If True, loads pretrained weights. If string,
        path to local weights file.
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset for pretrained weights ("uncased", "cased", etc.).
    :type weights_dataset: str
    :param cache_dir: Directory to cache downloaded weights.
    :type cache_dir: Optional[str]
    :param distilbert_config_overrides: Optional dictionary to override default
        DistilBERT configuration for the chosen variant. Defaults to None.
    :type distilbert_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Optional dictionary to override default head
        configuration. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :return: A complete `keras.Model` ready for the specified task.
    :rtype: keras.Model

    Example:
        .. code-block:: python

            from dl_techniques.layers.nlp_heads import NLPTaskType

            # Define a task
            ner_task = NLPTaskConfig(
                name="ner",
                task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
                num_classes=9
            )

            # Create the full model with pretrained DistilBERT
            ner_model = create_distilbert_with_head(
                distilbert_variant="base",
                task_config=ner_task,
                pretrained=True,
                head_config_overrides={"use_task_attention": True}
            )
            ner_model.summary()
    """
    distilbert_config_overrides = distilbert_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(
        f"Creating DistilBERT-{distilbert_variant} with a '{task_config.name}' head."
    )

    # 1. Create the foundational DistilBERT model (with optional pretrained weights)
    distilbert_encoder = DistilBERT.from_variant(
        distilbert_variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **distilbert_config_overrides
    )

    # 2. Create the task head
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=distilbert_encoder.hidden_size,
        **head_config_overrides,
    )

    # 3. Define inputs and build the end-to-end model
    # Note: DistilBERT doesn't use token_type_ids
    inputs = {
        "input_ids": keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        ),
        "attention_mask": keras.Input(
            shape=(None,), dtype="int32", name="attention_mask"
        ),
    }

    # Get hidden states from the encoder
    encoder_outputs = distilbert_encoder(inputs)

    # Pass encoder outputs to the task head
    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": encoder_outputs["attention_mask"],
    }
    task_outputs = task_head(head_inputs)

    # Create the final model
    model_name = f"distilbert_{distilbert_variant}_with_{task_config.name}_head"
    model = keras.Model(
        inputs=inputs,
        outputs=task_outputs,
        name=model_name
    )

    logger.info(
        f"Successfully created model with {model.count_params():,} parameters."
    )
    return model

# ---------------------------------------------------------------------
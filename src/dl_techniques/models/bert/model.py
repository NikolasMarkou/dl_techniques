"""
BERT Model Implementation
==================================================

A complete implementation of the BERT (Bidirectional Encoder Representations from Transformers) architecture.
This implementation provides a flexible, production-ready BERT model with modern dl-techniques features.

Based on: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
(Devlin et al., 2018) https://arxiv.org/abs/1810.04805

Theory and Architecture Overview:
---------------------------------

BERT revolutionized natural language processing by introducing bidirectional context understanding
through transformer architecture. Unlike previous models that processed text left-to-right or
right-to-left, BERT reads the entire sequence simultaneously, enabling richer contextual representations.

**Key Innovations:**

1. **Bidirectional Context**: Uses masked language modeling (MLM) to enable bidirectional encoding.
   During pre-training, ~15% of input tokens are masked, and the model learns to predict them
   using both left and right context.

2. **Deep Transformer Architecture**: Stacks multiple transformer encoder layers, each containing:
   - Multi-head self-attention mechanism
   - Position-wise feed-forward networks
   - Residual connections and layer normalization
   - Dropout for regularization

3. **Universal Sentence Representations**: Through the [CLS] token pooling mechanism, BERT
   learns sentence-level representations suitable for classification tasks.

4. **Pre-training + Fine-tuning Paradigm**: Pre-trained on large text corpora with MLM and
   Next Sentence Prediction (NSP), then fine-tuned on downstream tasks.

**Architecture Components:**

```
Input Processing:
    TokenIDs + SegmentIDs + PositionIDs
               │
               ▼
    Word Embeddings + Segment Embeddings + Position Embeddings
               │
               ▼
    LayerNorm + Dropout
               │
               ▼
Transformer Stack (N layers):
    MultiHeadAttention(bidirectional)
               │
    Add & Norm  │
               ▼
    FeedForward(GELU activation)
               │
    Add & Norm  │
               ▼
    Layer N Output
               │
               ▼
Output Processing:
    Sequence Output: [batch_size, seq_len, hidden_size]
               │
               └─→ [CLS] Token → Dense + Tanh → Pooled Output
```

**Mathematical Foundation:**

Self-Attention Mechanism:
- Query, Key, Value matrices: Q = XW_Q, K = XW_K, V = XW_V
- Attention scores: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Multi-head: Concat(head_1, ..., head_h)W_O where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)

Position Embeddings:
- Learned absolute position embeddings for each position up to max_position_embeddings
- Added to word embeddings to provide positional information

**Pre-training Objectives:**

1. **Masked Language Model (MLM)**:
   - Randomly mask 15% of input tokens
   - 80% replaced with [MASK], 10% with random token, 10% unchanged
   - Predict original token using bidirectional context
   - Loss: Cross-entropy over vocabulary for masked positions

2. **Next Sentence Prediction (NSP)**:
   - Given sentence pairs, predict if second sentence follows first
   - Binary classification using [CLS] token representation
   - Helps learn sentence-level representations

**Model Variants:**

- **BERT-Base**: 12 layers, 768 hidden, 12 attention heads, 110M parameters
- **BERT-Large**: 24 layers, 1024 hidden, 16 attention heads, 340M parameters

**Modern Extensions in this Implementation:**

- Configurable attention mechanisms (standard, window, differential)
- Advanced normalization options (LayerNorm, RMSNorm, DynamicTanh)
- Stochastic depth for improved training stability
- Flexible feed-forward networks (MLP, SwiGLU, etc.)
- Enhanced regularization techniques
- Production-ready serialization and deployment features

Key Features:
-------------
- Full BERT architecture with bidirectional transformer encoder
- Support for all standard BERT variants (Base, Large)
- Advanced dl-techniques integration (RMSNorm, SwiGLU, etc.)
- Flexible attention mechanisms and normalization strategies
- Configurable stochastic depth and regularization
- Complete serialization support for production deployment
- Comprehensive factory functions for common use cases
- Memory-efficient implementation with gradient checkpointing support

Training Strategies:
------------------
- **Pre-training**: Large-scale unsupervised learning on text corpora
- **Fine-tuning**: Task-specific supervised learning with lower learning rates
- **Feature Extraction**: Using pre-trained representations as fixed features
- **Discriminative Fine-tuning**: Different learning rates for different layers

Common Applications:
------------------
- **Text Classification**: Sentiment analysis, topic classification, spam detection
- **Named Entity Recognition**: Identifying people, places, organizations in text
- **Question Answering**: SQuAD-style reading comprehension tasks
- **Sequence Labeling**: Part-of-speech tagging, linguistic analysis
- **Sentence Similarity**: Computing semantic similarity between sentences
- **Language Understanding**: GLUE/SuperGLUE benchmark tasks

Usage Examples:
--------------
```python
# Create BERT-Base model for binary classification
config = create_bert_base_uncased()
model = create_bert_for_classification(config, num_labels=2)

# Create advanced BERT with modern features
config = create_bert_with_advanced_features(
    size="base",
    normalization_type="rms_norm",
    attention_type="differential_attention",
    ffn_type="swiglu"
)
model = BERT.from_variant("base_advanced", **config)

# Fine-tuning for custom task
bert_model = BERT.from_variant("base", add_pooling_layer=True)
inputs = keras.Input(shape=(512,), dtype="int32", name="input_ids")
outputs = bert_model(inputs)
classifier = keras.layers.Dense(num_classes)(outputs[1])  # Use pooled output
model = keras.Model(inputs, classifier)

# Multi-input format
inputs = {
    "input_ids": keras.random.uniform((2, 128), 0, 30522, dtype="int32"),
    "attention_mask": keras.ops.ones((2, 128), dtype="int32"),
    "token_type_ids": keras.ops.zeros((2, 128), dtype="int32")
}
sequence_output, pooled_output = model(inputs)
```

Implementation Notes:
-------------------
- Follows Keras 3.8+ best practices with full type hints
- Memory-efficient with optional gradient checkpointing
- Supports mixed precision training out of the box
- Compatible with TensorFlow 2.18+ backend
- Thread-safe for inference in production environments
- Comprehensive error handling and validation
- Extensive logging for debugging and monitoring
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.embedding.bert_embeddings import BertEmbeddings

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BERT(keras.Model):
    """
    BERT (Bidirectional Encoder Representations from Transformers) model.

    A modern, flexible implementation of the BERT architecture with support for various
    advanced features from dl-techniques library including different attention mechanisms,
    normalization strategies, and regularization techniques.

    **Architecture Overview:**
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
        Can be a single tensor of `input_ids` or a dictionary containing:
        - `input_ids`: 2D tensor with shape `(batch_size, sequence_length)`.
        - `attention_mask`: (Optional) 2D tensor, shape `(batch_size, sequence_length)`.
        - `token_type_ids`: (Optional) 2D tensor, shape `(batch_size, sequence_length)`.
        - `position_ids`: (Optional) 2D tensor, shape `(batch_size, sequence_length)`.

    Output shape:
        - If `add_pooling_layer=False`: A single tensor of shape
          `(batch_size, sequence_length, hidden_size)`.
        - If `add_pooling_layer=True`: A tuple `(sequence_output, pooled_output)`.
        - If `return_dict=True`: A dictionary with keys `last_hidden_state`
          and optionally `pooler_output`.

    Attributes:
        embeddings: The embedding layer instance.
        encoder_layers: A list of `TransformerLayer` instances.
        pooler: The pooling layer instance (if `add_pooling_layer` is True).

    Raises:
        ValueError: If invalid configuration parameters are provided.

    Example:
        >>> # Create standard BERT-base model
        >>> model = BERT.from_variant("base", add_pooling_layer=True)
        >>>
        >>> # Create BERT with advanced features
        >>> config = create_bert_with_advanced_features("base", normalization_type="rms_norm")
        >>> model = BERT(**config)
        >>>
        >>> # Use the model
        >>> input_ids = keras.random.uniform((2, 128), 0, 30522, dtype="int32")
        >>> sequence_output, pooled_output = model(input_ids)
    """

    # Model variant configurations following BERT paper specifications
    MODEL_VARIANTS = {
        "base": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_size": 3072,
            "description": "BERT-Base: 110M parameters, suitable for most applications"
        },
        "large": {
            "vocab_size": 30522,
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "intermediate_size": 4096,
            "description": "BERT-Large: 340M parameters, maximum performance"
        },
        "small": {
            "vocab_size": 30522,
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8,
            "intermediate_size": 2048,
            "description": "BERT-Small: Lightweight variant for resource-constrained environments"
        },
        "tiny": {
            "vocab_size": 30522,
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "intermediate_size": 1024,
            "description": "BERT-Tiny: Ultra-lightweight for mobile/edge deployment"
        },
    }

    # Architecture constants following BERT specifications
    DEFAULT_MAX_POSITION_EMBEDDINGS = 512
    DEFAULT_TYPE_VOCAB_SIZE = 2
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPSILON = 1e-12
    DEFAULT_HIDDEN_ACT = "gelu"
    DEFAULT_PAD_TOKEN_ID = 0

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

        # Validate configuration parameters
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_heads,
            hidden_dropout_prob, attention_probs_dropout_prob
        )

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

        # Initialize layer containers
        self.encoder_layers: List[TransformerLayer] = []
        self.embeddings: Optional[BertEmbeddings] = None
        self.pooler: Optional[keras.layers.Dense] = None

        # Build the model architecture
        self._build_architecture()

        # Initialize the Model base class
        super().__init__(**kwargs)

        logger.info(
            f"Created BERT model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, heads={self.num_heads}, "
            f"pooling={self.add_pooling_layer}"
        )

    def _validate_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float
    ) -> None:
        """Validate model configuration parameters.

        Args:
            vocab_size: Vocabulary size to validate
            hidden_size: Hidden dimension size to validate
            num_layers: Number of layers to validate
            num_heads: Number of attention heads to validate
            hidden_dropout_prob: Hidden dropout rate to validate
            attention_probs_dropout_prob: Attention dropout rate to validate

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= hidden_dropout_prob <= 1.0):
            raise ValueError(
                f"hidden_dropout_prob must be between 0 and 1, got {hidden_dropout_prob}"
            )
        if not (0.0 <= attention_probs_dropout_prob <= 1.0):
            raise ValueError(
                f"attention_probs_dropout_prob must be between 0 and 1, "
                f"got {attention_probs_dropout_prob}"
            )

    def _build_architecture(self) -> None:
        """Build all model components following modern Keras 3 patterns."""

        # Create embeddings layer
        self.embeddings = BertEmbeddings(
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
        self.encoder_layers = []
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
        if self.add_pooling_layer:
            self.pooler = keras.layers.Dense(
                units=self.hidden_size,
                activation="tanh",
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name="pooler"
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

        Raises:
            ValueError: If inputs are not properly formatted.
        """
        # Parse inputs
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
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
                layer_idx=i,  # For differential attention compatibility
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

    @classmethod
    def from_variant(
        cls,
        variant: str,
        add_pooling_layer: bool = True,
        **kwargs: Any
    ) -> "BERT":
        """
        Create a BERT model from a predefined variant.

        Args:
            variant: String, one of "base", "large", "small", "tiny"
            add_pooling_layer: Boolean, whether to add pooling layer
            **kwargs: Additional arguments passed to the constructor

        Returns:
            BERT model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # Create BERT-Base model
            >>> model = BERT.from_variant("base", add_pooling_layer=True)
            >>>
            >>> # Create BERT-Large with advanced features
            >>> model = BERT.from_variant("large", normalization_type="rms_norm")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)  # Remove description field

        logger.info(f"Creating BERT-{variant.upper()} model")
        logger.info(f"Configuration: {cls.MODEL_VARIANTS[variant]['description']}")

        return cls(
            add_pooling_layer=add_pooling_layer,
            **config,
            **kwargs
        )

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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BERT":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            BERT model instance
        """
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional BERT-specific information."""
        super().summary(**kwargs)

        # Print additional model information
        total_blocks = self.num_layers
        hidden_params = self.hidden_size * self.num_heads

        logger.info("BERT Model Configuration:")
        logger.info(f"  - Architecture: {self.num_layers} layers, {self.hidden_size} hidden size")
        logger.info(f"  - Attention: {self.num_heads} heads, {self.attention_type}")
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(f"  - Max sequence length: {self.max_position_embeddings}")
        logger.info(f"  - Normalization: {self.normalization_type} ({self.normalization_position})")
        logger.info(f"  - Feed-forward: {self.ffn_type}, {self.intermediate_size} intermediate size")
        logger.info(f"  - Regularization: dropout={self.hidden_dropout_prob}")
        if self.use_stochastic_depth:
            logger.info(f"  - Stochastic depth: {self.stochastic_depth_rate}")
        logger.info(f"  - Pooling layer: {self.add_pooling_layer}")
        logger.info(f"  - Total transformer blocks: {total_blocks}")

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
            If None, uses the dropout rate from the config dictionary.

    Returns:
        A complete `keras.Model` for sequence classification.

    Raises:
        ValueError: If num_labels is not positive or config is invalid.

    Example:
        >>> config = create_bert_base_uncased()
        >>> model = create_bert_for_classification(config, num_labels=2)
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    """
    if num_labels <= 0:
        raise ValueError(f"num_labels must be positive, got {num_labels}")

    logger.info(f"Creating BERT classification model with {num_labels} labels")

    # Create base BERT model with pooling
    bert = BERT(**config, add_pooling_layer=True, name="bert")

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

# ---------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------

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
        >>> config = create_bert_base_uncased()
        >>> model = create_bert_for_sequence_output(config)
        >>>
        >>> # For token classification, add a classification head
        >>> num_tags = 9  # e.g., for NER
        >>> sequence_output = model.output
        >>> logits = keras.layers.Dense(num_tags)(sequence_output)
        >>> token_classifier = keras.Model(model.input, logits)
    """
    logger.info("Creating BERT model for sequence output tasks")

    # Create base BERT model without pooling
    bert = BERT(**config, add_pooling_layer=False, name="bert")

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

# ---------------------------------------------------------------------

def create_bert(
    variant: str = "base",
    num_classes: Optional[int] = None,
    task_type: str = "classification",
    **kwargs: Any
) -> keras.Model:
    """
    Convenience function to create BERT models for common tasks.

    Args:
        variant: String, model variant ("base", "large", "small", "tiny")
        num_classes: Optional integer, number of classes for classification tasks
        task_type: String, type of task ("classification", "sequence_output")
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        BERT model instance configured for the specified task

    Raises:
        ValueError: If invalid task_type or missing num_classes for classification

    Example:
        >>> # Create classification model
        >>> model = create_bert("base", num_classes=2, task_type="classification")
        >>>
        >>> # Create sequence output model
        >>> model = create_bert("base", task_type="sequence_output")
    """
    # Get base configuration
    if variant in BERT.MODEL_VARIANTS:
        config = BERT.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Update with any additional kwargs
    config.update(kwargs)

    if task_type == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be provided for classification task")
        return create_bert_for_classification(config, num_classes)
    elif task_type == "sequence_output":
        return create_bert_for_sequence_output(config)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

# ---------------------------------------------------------------------

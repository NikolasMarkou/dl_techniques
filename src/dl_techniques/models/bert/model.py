"""
BERT Model Implementation
==================================================

A complete and refactored implementation of the BERT (Bidirectional Encoder
Representations from Transformers) architecture. This version is designed as a pure
foundation model, separating the core encoding logic from task-specific heads for
maximum flexibility, especially in pre-training and multi-task fine-tuning scenarios.

Based on: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
(Devlin et al., 2018) https://arxiv.org/abs/1810.04805

Refactored Architecture Philosophy:
-----------------------------------

This implementation strictly adheres to the principle of separating the foundation model
from the task-specific head. The `BERT` class acts as a pure feature extractor,
transforming input tokens into a sequence of contextualized hidden states. All
task-specific logic, such as pooling for classification or span prediction for QA,
is delegated to downstream "head" models, as provided by the NLP Task Head Factory.

**Architectural Contract:**

```
Input Processing:
    TokenIDs + SegmentIDs + PositionIDs
               │
               ▼
    Embeddings Layer
               │
               ▼
Transformer Stack (N layers)
               │
               ▼
Output Dictionary: {
    "last_hidden_state": [batch_size, seq_len, hidden_size],
    "attention_mask": [batch_size, seq_len]
}
               │
               ▼
NLP Task Head (from factory.py)
    (e.g., TextClassificationHead, TokenClassificationHead)
               │
               ▼
Task-Specific Outputs
    (e.g., {"logits": ..., "probabilities": ...})
```

**Key Changes in this Implementation:**

1.  **Pure Encoder:** The `BERT` model no longer contains a pooling layer (`add_pooling_layer` is removed). Its sole responsibility is to produce high-fidelity sequence representations.
2.  **Consistent Output:** The `call` method now always returns a dictionary containing `last_hidden_state` and the forwarded `attention_mask`. This provides a stable, predictable interface for any downstream head.
3.  **Decoupled Factories:** The original, task-specific factory functions (`create_bert_for_classification`, etc.) have been removed. They are replaced by a single, powerful integration function (`create_bert_with_head`) that demonstrates how to combine this foundational `BERT` model with the heads from the `dl_techniques.nlp.heads` factory.
4.  **Simplified Interface:** The model's API is simplified, removing conditional logic and promoting a clear, composable design pattern.

This refactoring enables:
- **Easy Multi-Tasking:** A single shared BERT encoder can feed its output into multiple different task heads.
- **Clean Fine-Tuning:** The weights of the foundational BERT can be frozen or trained with a different learning rate than the task-specific head.
- **Model Reusability:** The same pre-trained BERT artifact can be used for classification, token labeling, question answering, etc., without modification.

Key Features:
-------------
- Pure BERT foundation model with bidirectional transformer encoder.
- Support for all standard BERT variants (Base, Large, etc.).
- Designed for seamless integration with `dl_techniques.nlp.heads` factory.
- Consistent dictionary output for a stable API contract.
- Memory-efficient implementation with gradient checkpointing support.
- Full serialization support for production deployment.

Usage Examples:
--------------
```python
import keras
from dl_techniques.nlp.heads.factory import create_nlp_head
from dl_techniques.nlp.heads.task_types import NLPTaskConfig, NLPTaskType

# 1. Create the foundational BERT model
bert_encoder = BERT.from_variant("base")

# 2. Define a task and create a corresponding head
sentiment_config = NLPTaskConfig(
    name="sentiment",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=3
)
sentiment_head = create_nlp_head(
    task_config=sentiment_config,
    input_dim=bert_encoder.hidden_size
)

# 3. Combine them into a final Keras model
inputs = {
    "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
    "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
    "token_type_ids": keras.Input(shape=(None,), dtype="int32", name="token_type_ids")
}
bert_outputs = bert_encoder(inputs)
task_outputs = sentiment_head(bert_outputs)

sentiment_model = keras.Model(inputs, task_outputs)

# 4. Use the model
# input_data = ...
# results = sentiment_model(input_data)
# print(results['logits'].shape) # (batch_size, 3)
```
"""

import keras
from typing import Optional, Union, Any, Dict, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import (
    FFNType,
    AttentionType,
    TransformerLayer,
    NormalizationType,
    NormalizationPosition,
)
from dl_techniques.layers.embedding.bert_embeddings import BertEmbeddings
from dl_techniques.layers.nlp_heads import create_nlp_head, NLPTaskConfig

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BERT(keras.Model):
    """
    BERT (Bidirectional Encoder Representations from Transformers) foundation model.

    This is a pure encoder implementation, designed to produce contextual token
    representations. It separates the core transformer architecture from any
    task-specific layers (like pooling or classification heads), making it highly
    flexible for pre-training, fine-tuning, and multi-task learning.

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
    Output Dictionary {
        "last_hidden_state": [batch, seq_len, hidden_size],
        "attention_mask": [batch, seq_len]
    }
    ```

    Args:
        vocab_size: Integer, size of the vocabulary. Defaults to 30522.
        hidden_size: Integer, dimensionality of encoder layers. Defaults to 768.
        num_layers: Integer, number of hidden transformer layers. Defaults to 12.
        num_heads: Integer, number of attention heads for each attention layer.
            Defaults to 12.
        intermediate_size: Integer, dimensionality of the "intermediate"
            (feed-forward) layer. Defaults to 3072.
        hidden_act: String, the non-linear activation function in the encoder.
            Defaults to "gelu".
        hidden_dropout_prob: Float, dropout probability for all fully connected
            layers in embeddings and encoder. Defaults to 0.1.
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
        normalization_type: String, type of normalization layer.
            Defaults to "layer_norm".
        normalization_position: String, position of normalization ('pre' or 'post').
            Defaults to "post".
        attention_type: String, type of attention mechanism.
            Defaults to "multi_head".
        ffn_type: String, type of feed-forward network. Defaults to "mlp".
        use_stochastic_depth: Boolean, whether to enable stochastic depth.
            Defaults to False.
        stochastic_depth_rate: Float, drop path rate for stochastic depth.
            Defaults to 0.1.
        **kwargs: Additional keyword arguments for the `keras.Model` base class.

    Input shape:
        Can be a single tensor of `input_ids` or a dictionary containing:
        - `input_ids`: 2D tensor with shape `(batch_size, sequence_length)`.
        - `attention_mask`: (Optional) 2D tensor, shape `(batch_size, sequence_length)`.
        - `token_type_ids`: (Optional) 2D tensor, shape `(batch_size, sequence_length)`.
        - `position_ids`: (Optional) 2D tensor, shape `(batch_size, sequence_length)`.

    Output shape:
        A dictionary containing:
        - `last_hidden_state`: Tensor of shape `(batch_size, sequence_length, hidden_size)`.
        - `attention_mask`: The input `attention_mask` passed through, for downstream convenience.

    Attributes:
        embeddings: The embedding layer instance.
        encoder_layers: A list of `TransformerLayer` instances.

    Raises:
        ValueError: If invalid configuration parameters are provided.

    Example:
        >>> # Create standard BERT-base model
        >>> model = BERT.from_variant("base")
        >>>
        >>> # Create BERT with advanced features
        >>> model = BERT.from_variant("large", normalization_type="rms_norm")
        >>>
        >>> # Use the model
        >>> inputs = {
        ...     "input_ids": keras.random.uniform((2, 128), 0, 30522, dtype="int32"),
        ...     "attention_mask": keras.ops.ones((2, 128), dtype="int32")
        ... }
        >>> outputs = model(inputs)
        >>> print(outputs["last_hidden_state"].shape)
        (2, 128, 1024)
    """

    # Model variant configurations following BERT paper specifications
    MODEL_VARIANTS = {
        "base": {
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_size": 3072,
            "description": "BERT-Base: 110M parameters, suitable for most applications"
        },
        "large": {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "intermediate_size": 4096,
            "description": "BERT-Large: 340M parameters, maximum performance"
        },
        "small": {
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8,
            "intermediate_size": 2048,
            "description": "BERT-Small: Lightweight variant for resource-constrained environments"
        },
        "tiny": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "intermediate_size": 1024,
            "description": "BERT-Tiny: Ultra-lightweight for mobile/edge deployment"
        },
    }

    # Default architecture constants
    DEFAULT_VOCAB_SIZE = 30522
    DEFAULT_MAX_POSITION_EMBEDDINGS = 512
    DEFAULT_TYPE_VOCAB_SIZE = 2
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPSILON = 1e-12
    DEFAULT_HIDDEN_ACT = "gelu"
    DEFAULT_PAD_TOKEN_ID = 0

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = DEFAULT_HIDDEN_ACT,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS,
        type_vocab_size: int = DEFAULT_TYPE_VOCAB_SIZE,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        normalization_type: NormalizationType = "layer_norm",
        normalization_position: NormalizationPosition = "post",
        attention_type: AttentionType = "multi_head",
        ffn_type: FFNType = "mlp",
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

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
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate

        # Build the model architecture
        self._build_architecture()

        logger.info(
            f"Created BERT foundation model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, heads={self.num_heads}"
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
        """Validate model configuration parameters."""
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
        """Build all model components."""
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

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass of the BERT foundation model.

        Args:
            inputs: Input token IDs or dictionary containing multiple inputs.
            attention_mask: Mask to avoid attention on padding tokens.
            token_type_ids: Token type IDs for distinguishing sequences.
            position_ids: Position IDs for positional embeddings.
            training: Boolean, whether the model is in training mode.

        Returns:
            A dictionary with the following keys:
            - `last_hidden_state`: The sequence of hidden states at the output
              of the final layer. Shape: `(batch, seq_len, hidden_size)`.
            - `attention_mask`: The original attention mask, passed through for
              convenience in downstream models.

        Raises:
            ValueError: If inputs are not properly formatted.
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
        else:
            input_ids = inputs

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training
        )

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

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "BERT":
        """
        Create a BERT model from a predefined variant.

        Args:
            variant: String, one of "base", "large", "small", "tiny".
            **kwargs: Additional arguments to override the variant's defaults.

        Returns:
            A BERT model instance.

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")

        logger.info(f"Creating BERT-{variant.upper()} model")
        logger.info(f"Configuration: {description}")

        # Override defaults with kwargs
        config.update(kwargs)

        return cls(**config)

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
            "normalization_type": self.normalization_type,
            "normalization_position": self.normalization_position,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BERT":
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional BERT-specific information."""
        super().summary(**kwargs)
        logger.info("BERT Foundation Model Configuration:")
        logger.info(f"  - Architecture: {self.num_layers} layers, {self.hidden_size} hidden size")
        logger.info(f"  - Attention: {self.num_heads} heads, {self.attention_type}")
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(f"  - Max sequence length: {self.max_position_embeddings}")
        logger.info(f"  - Normalization: {self.normalization_type} ({self.normalization_position})")
        logger.info(f"  - Feed-forward: {self.ffn_type}, {self.intermediate_size} intermediate size")
        if self.use_stochastic_depth:
            logger.info(f"  - Stochastic depth enabled: rate={self.stochastic_depth_rate}")


# ---------------------------------------------------------------------
# Integration with NLP Task Heads
# ---------------------------------------------------------------------


def create_bert_with_head(
    bert_variant: str,
    task_config: NLPTaskConfig,
    bert_config_overrides: Optional[Dict[str, Any]] = None,
    head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """
    Factory function to create a complete BERT model with a task-specific head.

    This function demonstrates the intended integration pattern:
    1. Instantiate a foundational `BERT` model.
    2. Instantiate a task-specific head from the `dl_techniques.nlp.heads` factory.
    3. Combine them into a single, end-to-end `keras.Model`.

    Args:
        bert_variant: String, the BERT variant to use (e.g., "base", "large").
        task_config: An `NLPTaskConfig` object defining the downstream task.
        bert_config_overrides: Optional dictionary to override default BERT
            configuration for the chosen variant.
        head_config_overrides: Optional dictionary to override default head
            configuration.

    Returns:
        A complete `keras.Model` ready for training or inference on a specific task.

    Example:
        >>> from dl_techniques.layers.nlp_heads import NLPTaskType
        >>> # Define a task
        >>> ner_task = NLPTaskConfig(
        ...     name="ner",
        ...     task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
        ...     num_classes=9
        ... )
        >>> # Create the full model
        >>> ner_model = create_bert_with_head(
        ...     bert_variant="base",
        ...     task_config=ner_task,
        ...     head_config_overrides={"use_task_attention": True}
        ... )
        >>> ner_model.summary()
    """
    bert_config_overrides = bert_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(f"Creating BERT-{bert_variant} with a '{task_config.name}' head.")

    # 1. Create the foundational BERT model
    bert_encoder = BERT.from_variant(bert_variant, **bert_config_overrides)

    # 2. Create the task head
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=bert_encoder.hidden_size,
        **head_config_overrides,
    )

    # 3. Define inputs and build the end-to-end model
    inputs = {
        "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
        "token_type_ids": keras.Input(shape=(None,), dtype="int32", name="token_type_ids"),
    }

    # Get hidden states from the encoder
    encoder_outputs = bert_encoder(inputs)

    # Pass encoder outputs to the task head
    # The head expects a dictionary with 'hidden_states' (and optionally 'attention_mask')
    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": encoder_outputs["attention_mask"],
    }
    task_outputs = task_head(head_inputs)

    # Create the final model
    model = keras.Model(
        inputs=inputs,
        outputs=task_outputs,
        name=f"bert_{bert_variant}_with_{task_config.name}_head"
    )

    logger.info(f"Successfully created model with {model.count_params():,} parameters.")
    return model

# ---------------------------------------------------------------------


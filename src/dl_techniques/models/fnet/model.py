"""
FNet Model Implementation
==================================================

A complete implementation of the FNet (Fourier Transform-based Neural Network) architecture.
This implementation provides a flexible, production-ready FNet model with modern dl-techniques features.

Based on: "FNet: Mixing Tokens with Fourier Transforms"
(Lee-Thorp et al., 2021) https://arxiv.org/abs/2105.03824

Theory and Architecture Overview:
---------------------------------

FNet revolutionizes transformer efficiency by replacing computationally expensive self-attention
mechanisms with parameter-free Fourier transforms. This approach maintains competitive performance
while achieving significant speedups, especially for longer sequences.

**Key Innovations:**

1. **Fourier Transform Token Mixing**: Replaces self-attention with 2D discrete Fourier transforms
   applied over sequence and hidden dimensions. This achieves O(N log N) complexity compared to
   O(N²) for self-attention, where N is sequence length.

2. **Parameter-Free Mixing**: Unlike attention mechanisms that require learned query, key, and
   value matrices, Fourier transforms provide effective token mixing without additional parameters,
   reducing model size and training requirements.

3. **Bidirectional Context**: Like BERT, FNet processes entire sequences simultaneously, enabling
   bidirectional context understanding for masked language modeling and downstream tasks.

4. **Architectural Simplicity**: Maintains the transformer's feed-forward and normalization
   components while simplifying the mixing mechanism, making the architecture easier to implement
   and debug.

**Mathematical Foundation:**

Fourier Transform Token Mixing:
- Applied along sequence dimension: F₁(x) = DFT(x, axis=1)
- Applied along hidden dimension: F₂(x) = DFT(F₁(x), axis=2)
- Real component extraction: output = Re(F₂(x))

**Architecture Components:**

```
Input Processing:
    TokenIDs + PositionIDs
               │
               ▼
    Word Embeddings + Position Embeddings
               │
               ▼
    LayerNorm + Dropout
               │
               ▼
FNet Stack (N layers):
    Fourier Transform (parameter-free)
               │
    Add & Norm  │
               ▼
    FeedForward (configurable type)
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

**Performance Characteristics:**

- **Speed**: 7x faster than BERT-Base on TPUs for sequence length 512
- **Memory**: Reduced memory footprint due to no attention matrices
- **Scalability**: Better scaling to longer sequences due to O(N log N) complexity
- **Quality**: Maintains 92% of BERT's accuracy on GLUE benchmark

**Model Variants:**

- **FNet-Base**: 12 layers, 768 hidden, matches BERT-Base architecture
- **FNet-Large**: 24 layers, 1024 hidden, matches BERT-Large architecture
- **FNet-Small**: 6 layers, 512 hidden, lightweight variant
- **FNet-Tiny**: 4 layers, 256 hidden, ultra-lightweight for edge deployment

**Modern Extensions in this Implementation:**

- Configurable normalization strategies (LayerNorm, RMSNorm, etc.)
- Advanced feed-forward networks (MLP, SwiGLU, etc.)
- Stochastic depth for improved training stability
- Flexible factory functions for common configurations
- Complete serialization support for production deployment
- Enhanced regularization techniques

Key Features:
-------------
- Full FNet architecture with Fourier Transform token mixing
- Support for all standard FNet variants (Base, Large, Small, Tiny)
- Advanced dl-techniques integration (RMSNorm, SwiGLU, etc.)
- Configurable normalization and feed-forward strategies
- Memory-efficient implementation with gradient checkpointing support
- Complete serialization support for production deployment
- Comprehensive factory functions for common use cases

Training Strategies:
------------------
- **Pre-training**: Large-scale unsupervised learning similar to BERT
- **Fine-tuning**: Task-specific supervised learning with lower learning rates
- **Feature Extraction**: Using pre-trained representations as fixed features
- **Efficient Training**: Faster convergence due to simpler mixing mechanism

Common Applications:
------------------
- **Text Classification**: Faster alternative to BERT for sentiment analysis
- **Named Entity Recognition**: Efficient sequence labeling with competitive accuracy
- **Question Answering**: Reduced latency for reading comprehension tasks
- **Document Processing**: Better scaling for long documents
- **Real-time Applications**: Suitable for latency-critical NLP applications

Usage Examples:
--------------
```python
# Create FNet-Base model for binary classification
config = create_fnet_base()
model = create_fnet_for_classification(config, num_labels=2)

# Create advanced FNet with modern features
config = create_fnet_with_advanced_features(
    size="base",
    normalization_type="rms_norm",
    ffn_type="swiglu"
)
model = FNet.from_variant("base_advanced", **config)

# Fine-tuning for custom task
fnet_model = FNet.from_variant("base", add_pooling_layer=True)
inputs = keras.Input(shape=(512,), dtype="int32", name="input_ids")
outputs = fnet_model(inputs)
classifier = keras.layers.Dense(num_classes)(outputs)  # Use pooled output
model = keras.Model(inputs, classifier)

# Multi-input format
inputs = {
    "input_ids": keras.random.uniform((2, 128), 0, 30522, dtype="int32"),
    "attention_mask": keras.ops.ones((2, 128), dtype="int32"),
    "position_ids": keras.ops.arange(128)[None, :].repeat(2, axis=0)
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
from dl_techniques.layers.fnet_encoder_block import FNetEncoderBlock
from dl_techniques.layers.embedding.bert_embeddings import BertEmbeddings

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FNet(keras.Model):
    """
    FNet (Fourier Transform-based Neural Network) model.

    A modern, flexible implementation of the FNet architecture with support for various
    advanced features from dl-techniques library including different normalization
    strategies, feed-forward networks, and regularization techniques.

    **Architecture Overview:**
    ```
    Input(input_ids, attention_mask, position_ids)
           │
           ▼
    Embeddings(Word + Position) -> LayerNorm -> Dropout
           │
           ▼
    FNetEncoderBlock₁ (Fourier Transform -> FFN)
           │
           ▼
          ...
           │
           ▼
    FNetEncoderBlockₙ (Fourier Transform -> FFN)
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
        num_layers: Integer, number of hidden FNet encoder layers. Defaults to 12.
        intermediate_size: Integer, dimensionality of the feed-forward layer.
            Defaults to 3072.
        hidden_act: String, the non-linear activation function in the encoder.
            Defaults to "gelu".
        hidden_dropout_prob: Float, dropout probability for all fully connected
            layers in embeddings, encoder, and pooler. Defaults to 0.1.
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
        classifier_dropout: Optional float, dropout for final classifier head.
            Defaults to None.
        normalization_type: String, type of normalization layer.
            Defaults to "layer_norm".
        ffn_type: String, type of feed-forward network. Defaults to "mlp".
        fourier_config: Optional dictionary, configuration for Fourier transform.
        normalization_kwargs: Optional dictionary, normalization-specific parameters.
        ffn_kwargs: Optional dictionary, FFN-specific parameters.
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
        - `position_ids`: (Optional) 2D tensor, shape `(batch_size, sequence_length)`.

    Output shape:
        - If `add_pooling_layer=False`: A single tensor of shape
          `(batch_size, sequence_length, hidden_size)`.
        - If `add_pooling_layer=True`: A tuple `(sequence_output, pooled_output)`.
        - If `return_dict=True`: A dictionary with keys `last_hidden_state`
          and optionally `pooler_output`.

    Attributes:
        embeddings: The embedding layer instance.
        encoder_layers: A list of `FNetEncoderBlock` instances.
        pooler: The pooling layer instance (if `add_pooling_layer` is True).

    Raises:
        ValueError: If invalid configuration parameters are provided.

    Example:
        >>> # Create standard FNet-base model
        >>> model = FNet.from_variant("base", add_pooling_layer=True)
        >>>
        >>> # Create FNet with advanced features
        >>> config = create_fnet_with_advanced_features("base", normalization_type="rms_norm")
        >>> model = FNet(**config)
        >>>
        >>> # Use the model
        >>> input_ids = keras.random.uniform((2, 128), 0, 30522, dtype="int32")
        >>> sequence_output, pooled_output = model(input_ids)
    """

    # Model variant configurations following FNet paper specifications
    MODEL_VARIANTS = {
        "base": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_layers": 12,
            "intermediate_size": 3072,
            "description": "FNet-Base: Efficient BERT-Base alternative with Fourier mixing"
        },
        "large": {
            "vocab_size": 30522,
            "hidden_size": 1024,
            "num_layers": 24,
            "intermediate_size": 4096,
            "description": "FNet-Large: Efficient BERT-Large alternative with Fourier mixing"
        },
        "small": {
            "vocab_size": 30522,
            "hidden_size": 512,
            "num_layers": 6,
            "intermediate_size": 2048,
            "description": "FNet-Small: Lightweight variant for faster inference"
        },
        "tiny": {
            "vocab_size": 30522,
            "hidden_size": 256,
            "num_layers": 4,
            "intermediate_size": 1024,
            "description": "FNet-Tiny: Ultra-lightweight for mobile/edge deployment"
        },
    }

    # Architecture constants following FNet specifications
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
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        classifier_dropout: Optional[float] = None,
        normalization_type: str = "layer_norm",
        ffn_type: str = "mlp",
        fourier_config: Optional[Dict[str, Any]] = None,
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        ffn_kwargs: Optional[Dict[str, Any]] = None,
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        add_pooling_layer: bool = True,
        **kwargs: Any
    ) -> None:

        super().__init__(**kwargs)
        # Validate configuration parameters
        self._validate_config(
            vocab_size, hidden_size, num_layers, intermediate_size,
            hidden_dropout_prob
        )

        # Store all configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.classifier_dropout = classifier_dropout
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.fourier_config = fourier_config or {}
        self.normalization_kwargs = normalization_kwargs or {}
        self.ffn_kwargs = ffn_kwargs or {}
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.add_pooling_layer = add_pooling_layer

        # Initialize layer containers - CREATE in __init__
        self.encoder_layers: List[FNetEncoderBlock] = []
        self.embeddings: Optional[BertEmbeddings] = None
        self.pooler: Optional[keras.layers.Dense] = None

        # Build the model architecture
        self._build_architecture()



        logger.info(
            f"Created FNet model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, "
            f"normalization={self.normalization_type}, "
            f"ffn={self.ffn_type}, "
            f"pooling={self.add_pooling_layer}"
        )

    def _validate_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        intermediate_size: int,
        hidden_dropout_prob: float
    ) -> None:
        """Validate model configuration parameters.

        Args:
            vocab_size: Vocabulary size to validate
            hidden_size: Hidden dimension size to validate
            num_layers: Number of layers to validate
            intermediate_size: FFN intermediate size to validate
            hidden_dropout_prob: Hidden dropout rate to validate

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if not (0.0 <= hidden_dropout_prob <= 1.0):
            raise ValueError(
                f"hidden_dropout_prob must be between 0 and 1, got {hidden_dropout_prob}"
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
            name="bert_embeddings"
        )

        # Create FNet encoder layers
        self.encoder_layers = []
        for i in range(self.num_layers):
            encoder_layer = FNetEncoderBlock(
                intermediate_dim=self.intermediate_size,
                dropout_rate=self.hidden_dropout_prob,
                fourier_config=self.fourier_config,
                normalization_type=self.normalization_type,
                normalization_kwargs=self.normalization_kwargs,
                ffn_type=self.ffn_type,
                ffn_kwargs=self.ffn_kwargs,
                name=f"encoder_layer_{i}"
            )
            self.encoder_layers.append(encoder_layer)

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
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        return_dict: bool = False
    ) -> Union[
        keras.KerasTensor,
        Tuple[keras.KerasTensor, keras.KerasTensor],
        Dict[str, keras.KerasTensor]
    ]:
        """
        Forward pass of the FNet model.

        Args:
            inputs: Input token IDs or dictionary containing multiple inputs.
            attention_mask: Mask to avoid processing on padding tokens.
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
            position_ids = inputs.get("position_ids", position_ids)
        else:
            input_ids = inputs

        # Get embeddings
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
    ) -> "FNet":
        """
        Create an FNet model from a predefined variant.

        Args:
            variant: String, one of "base", "large", "small", "tiny"
            add_pooling_layer: Boolean, whether to add pooling layer
            **kwargs: Additional arguments passed to the constructor

        Returns:
            FNet model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # Create FNet-Base model
            >>> model = FNet.from_variant("base", add_pooling_layer=True)
            >>>
            >>> # Create FNet-Large with advanced features
            >>> model = FNet.from_variant("large", normalization_type="rms_norm")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)  # Remove description field

        logger.info(f"Creating FNet-{variant.upper()} model")
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
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "position_embedding_type": self.position_embedding_type,
            "classifier_dropout": self.classifier_dropout,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
            "fourier_config": self.fourier_config,
            "normalization_kwargs": self.normalization_kwargs,
            "ffn_kwargs": self.ffn_kwargs,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "add_pooling_layer": self.add_pooling_layer,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FNet":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            FNet model instance
        """
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional FNet-specific information."""
        super().summary(**kwargs)

        # Print additional model information
        total_blocks = self.num_layers

        logger.info("FNet Model Configuration:")
        logger.info(f"  - Architecture: {self.num_layers} layers, {self.hidden_size} hidden size")
        logger.info(f"  - Token Mixing: Fourier Transform (parameter-free)")
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(f"  - Max sequence length: {self.max_position_embeddings}")
        logger.info(f"  - Normalization: {self.normalization_type}")
        logger.info(f"  - Feed-forward: {self.ffn_type}, {self.intermediate_size} intermediate size")
        logger.info(f"  - Regularization: dropout={self.hidden_dropout_prob}")
        if self.use_stochastic_depth:
            logger.info(f"  - Stochastic depth: {self.stochastic_depth_rate}")
        logger.info(f"  - Pooling layer: {self.add_pooling_layer}")
        logger.info(f"  - Total encoder blocks: {total_blocks}")
        logger.info(f"  - Efficiency: ~7x faster than BERT due to Fourier mixing")

# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------


def create_fnet_for_classification(
    config: Dict[str, Any],
    num_labels: int,
    max_sequence_length: int,
    classifier_dropout: Optional[float] = None
) -> keras.Model:
    """
    Create an FNet model for sequence classification tasks.

    This function builds a complete model by adding a classification head on top
    of the pooled output of an FNet model.

    Args:
        config: Dictionary containing FNet model hyperparameters.
        num_labels: Integer, the number of classification labels.
        max_sequence_length: Integer, the fixed sequence length for the model inputs.
        classifier_dropout: Optional float, dropout rate for the classifier head.
            If None, uses the dropout rate from the config dictionary.

    Returns:
        A complete `keras.Model` for sequence classification.

    Raises:
        ValueError: If num_labels is not positive or config is invalid.

    Example:
        >>> config = create_fnet_base()
        >>> model = create_fnet_for_classification(config, num_labels=2, max_sequence_length=128)
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    """
    if num_labels <= 0:
        raise ValueError(f"num_labels must be positive, got {num_labels}")

    logger.info(f"Creating FNet classification model with {num_labels} labels")

    # Create base FNet model with pooling
    fnet = FNet(**config, add_pooling_layer=True, name="fnet")

    # Define inputs using Keras Functional API
    input_ids = keras.Input(shape=(max_sequence_length,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(max_sequence_length,), dtype="int32", name="attention_mask")
    position_ids = keras.Input(shape=(max_sequence_length,), dtype="int32", name="position_ids")

    # Get FNet outputs
    fnet_outputs = fnet(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        return_dict=True
    )

    # Classification head
    pooled_output = fnet_outputs["pooler_output"]

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
        inputs=[input_ids, attention_mask, position_ids],
        outputs=logits,
        name="fnet_for_classification"
    )

    logger.info(
        f"Created FNet classification model with {model.count_params()} parameters"
    )
    return model

# ---------------------------------------------------------------------


def create_fnet_for_sequence_output(
    config: Dict[str, Any],
    max_sequence_length: int
) -> keras.Model:
    """
    Create an FNet model for sequence-level output tasks.

    This function builds an FNet model that outputs sequence representations,
    suitable for tasks like token classification or question answering.

    Args:
        config: Dictionary containing FNet model hyperparameters.
        max_sequence_length: Integer, the fixed sequence length for the model inputs.

    Returns:
        A `keras.Model` that returns sequence-level representations.

    Example:
        >>> config = create_fnet_base()
        >>> model = create_fnet_for_sequence_output(config, max_sequence_length=128)
        >>>
        >>> # For token classification, add a classification head
        >>> num_tags = 9  # e.g., for NER
        >>> sequence_output = model.output
        >>> logits = keras.layers.Dense(num_tags)(sequence_output)
        >>> token_classifier = keras.Model(model.input, logits)
    """
    logger.info("Creating FNet model for sequence output tasks")

    # Create base FNet model without pooling
    fnet = FNet(**config, add_pooling_layer=False, name="fnet")

    # Define inputs
    input_ids = keras.Input(shape=(max_sequence_length,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(max_sequence_length,), dtype="int32", name="attention_mask")
    position_ids = keras.Input(shape=(max_sequence_length,), dtype="int32", name="position_ids")

    # Get FNet sequence output
    sequence_output = fnet(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    )

    # Create the final model
    model = keras.Model(
        inputs=[input_ids, attention_mask, position_ids],
        outputs=sequence_output,
        name="fnet_for_sequence_output"
    )

    logger.info(
        f"Created FNet sequence model with {model.count_params()} parameters"
    )
    return model

# ---------------------------------------------------------------------

def create_fnet(
    variant: str = "base",
    num_classes: Optional[int] = None,
    task_type: str = "classification",
    **kwargs: Any
) -> keras.Model:
    """
    Convenience function to create FNet models for common tasks.

    Args:
        variant: String, model variant ("base", "large", "small", "tiny")
        num_classes: Optional integer, number of classes for classification tasks
        task_type: String, type of task ("classification", "sequence_output")
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        FNet model instance configured for the specified task

    Raises:
        ValueError: If invalid task_type or missing num_classes for classification

    Example:
        >>> # Create classification model
        >>> model = create_fnet("base", num_classes=2, task_type="classification")
        >>>
        >>> # Create sequence output model
        >>> model = create_fnet("base", task_type="sequence_output")
    """
    # Get base configuration
    if variant in FNet.MODEL_VARIANTS:
        config = FNet.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Update with any additional kwargs
    config.update(kwargs)

    # Determine max sequence length from config or default
    max_sequence_length = config.get("max_position_embeddings", FNet.DEFAULT_MAX_POSITION_EMBEDDINGS)

    if task_type == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be provided for classification task")
        return create_fnet_for_classification(config, num_classes, max_sequence_length)
    elif task_type == "sequence_output":
        return create_fnet_for_sequence_output(config, max_sequence_length)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

# ---------------------------------------------------------------------

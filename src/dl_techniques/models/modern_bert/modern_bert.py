"""
ModernBERT: A High-Performance BERT Successor
==============================================

An advanced, refactored implementation of a BERT-style encoder that serves as a
powerful, efficient, and direct replacement for classic BERT architectures.
ModernBERT integrates a suite of contemporary deep learning techniques to
deliver superior performance, faster processing, and better stability, especially
for long-context applications.

This model is engineered as a pure foundation model, separating the core
encoding logic from task-specific heads. It supports pretrained weights and
maintains an API consistent with the original BERT implementation, ensuring
seamless interchangeability and ease of adoption.

-----------------------------
Key Architectural Upgrades
-----------------------------

ModernBERT distinguishes itself from its predecessors through several key
innovations inspired by recent advances in large language models:

1.  **Pre-Layer Normalization (Pre-LN):** Applies layer normalization *before*
    the attention and feed-forward blocks. This significantly improves training
    stability, allows for faster convergence, and reduces sensitivity to
    learning rate schedules, making it easier to train deep models.

2.  **GeGLU Activation Function:** Replaces the standard GELU with a Gated GELU
    (GeGLU) in the feed-forward network. GeGLU offers a more sophisticated
    gating mechanism that can improve performance and contribute to more
    stable training dynamics.

3.  **Hybrid Local & Global Attention:** Employs an efficient alternating
    attention strategy. Most layers use computationally cheaper **windowed
    (local) attention**, while periodic **global attention** layers ensure
    that long-range dependencies across the entire sequence are captured.
    This hybrid approach enables the model to process much longer sequences
    (e.g., up to 8192 tokens) far more efficiently than models relying solely
    on global attention.

4.  **Bias-Free Layers:** Removes bias parameters from most linear layers and
    normalization, reallocating the parameter budget to more impactful parts
    of the model for better efficiency.

These enhancements make ModernBERT a state-of-the-art encoder for a wide
range of NLP tasks, including retrieval-augmented generation (RAG), semantic
search, classification, and code analysis.

-------------------
Usage Examples
-------------------

.. code-block:: python

    import keras
    from dl_techniques.nlp.heads.factory import create_nlp_head
    from dl_techniques.nlp.heads.task_types import NLPTaskConfig, NLPTaskType

    # 1. Load a ModernBERT variant (e.g., from a pretrained checkpoint)
    # model = ModernBERT.from_variant("base", pretrained="path/to/weights.keras")
    model = ModernBERT.from_variant("base")

    # 2. Define a downstream task (e.g., Named Entity Recognition)
    ner_config = NLPTaskConfig(
        name="ner",
        task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
        num_classes=17
    )

    # 3. Create a complete model with a task head
    ner_model = create_modern_bert_with_head(
        bert_variant="base",
        task_config=ner_config
    )
    ner_model.summary()

    # 4. Use the model for inference
    inputs = {
        "input_ids": keras.random.uniform((2, 256), 0, 50368, dtype="int32"),
        "attention_mask": keras.ops.ones((2, 256), dtype="int32")
    }
    predictions = ner_model(inputs)
    print(predictions.shape)
    # (2, 256, 17)

"""

import os
import keras
from keras import layers, ops
from typing import Optional, Union, Any, Dict, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.nlp_heads import create_nlp_head, NLPTaskConfig
from dl_techniques.layers.embedding.modern_bert_embeddings import ModernBertEmbeddings


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBERT(keras.Model):
    """ModernBERT (A Modern Bidirectional Encoder) foundation model.

    This model refactors the original BERT architecture to include modern
    techniques such as Pre-Layer Normalization, GeGLU activations, and a
    hybrid attention mechanism combining efficient windowed attention with
    periodic global attention. It is designed for high performance and
    configurability.

    The model expects inputs as a dictionary containing 'input_ids', and
    optionally 'attention_mask' and 'token_type_ids'. It outputs a dictionary
    containing the 'last_hidden_state' and the forwarded 'attention_mask'.

    **Architecture Overview:**

    .. code-block:: text

        Input(input_ids, attention_mask, token_type_ids)
               │
               ▼
        ModernBertEmbeddings -> Dropout
               │
               ▼
        TransformerLayer₁ (Pre-LN, Windowed Attention -> GeGLU FFN)
               │
               ▼
              ... (Layers with windowed attention)
               │
               ▼
        TransformerLayerₖ (Pre-LN, Global Attention -> GeGLU FFN)
               │
               ▼
              ... (Alternating local and global attention)
               │
               ▼
        TransformerLayerₙ
               │
               ▼
        Final Layer Normalization
               │
               ▼
        Output Dictionary {
            "last_hidden_state": [batch, seq_len, hidden_size],
            "attention_mask": [batch, seq_len]
        }

    :param vocab_size: Size of the vocabulary. Defaults to 50368.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Defaults to 768.
    :type hidden_size: int
    :param num_layers: Number of hidden transformer layers. Defaults to 22.
    :type num_layers: int
    :param num_heads: Number of attention heads for each attention layer.
        Defaults to 12.
    :type num_heads: int
    :param intermediate_size: Dimensionality of the "intermediate"
        (feed-forward) layer. Defaults to 1152.
    :type intermediate_size: int
    :param hidden_act: The non-linear activation function in the FFN.
        Defaults to "gelu".
    :type hidden_act: str
    :param hidden_dropout_prob: Dropout probability for all fully connected
        layers in embeddings and encoder. Defaults to 0.1.
    :type hidden_dropout_prob: float
    :param attention_probs_dropout_prob: Dropout ratio for attention
        probabilities. Defaults to 0.1.
    :type attention_probs_dropout_prob: float
    :param type_vocab_size: Vocabulary size for token type IDs.
        Defaults to 2.
    :type type_vocab_size: int
    :param initializer_range: Stddev of truncated normal initializer for
        all weight matrices. Defaults to 0.02.
    :type initializer_range: float
    :param layer_norm_eps: Epsilon for normalization layers. Defaults to 1e-12.
    :type layer_norm_eps: float
    :param use_bias: Whether to use bias vectors in linear layers.
        Defaults to False.
    :type use_bias: bool
    :param global_attention_interval: Interval for inserting a global attention
        layer. E.g., 3 means every 3rd layer is global. Defaults to 3.
    :type global_attention_interval: int
    :param local_attention_window_size: Window size for local (windowed)
        attention layers. Defaults to 128.
    :type local_attention_window_size: int
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :raises ValueError: If invalid configuration parameters are provided.

    Example:
        .. code-block:: python

            # Create standard ModernBERT-base model
            model = ModernBERT.from_variant("base")

            # Use the model
            inputs = {
                "input_ids": keras.random.uniform((2, 256), 0, 50368, dtype="int32"),
                "attention_mask": keras.ops.ones((2, 256), dtype="int32")
            }
            outputs = model(inputs)
            print(outputs["last_hidden_state"].shape)
            # (2, 256, 768)
    """

    MODEL_VARIANTS = {
        "tiny": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "intermediate_size": 384,  # Consistent 1.5x ratio
            "use_bias": False,
            "global_attention_interval": 2,  # Global attention every 2 layers
            "local_attention_window_size": 64,
            "description": "ModernBERT-Tiny: Ultra-lightweight for mobile/edge deployment",
        },
        "base": {
            "hidden_size": 768,
            "num_layers": 22,
            "num_heads": 12,
            "intermediate_size": 1152,
            "use_bias": False,
            "global_attention_interval": 3,
            "local_attention_window_size": 128,
            "description": "ModernBERT-Base: 95M parameters, efficient base model",
        },
        "large": {
            "hidden_size": 1024,
            "num_layers": 28,
            "num_heads": 16,
            "intermediate_size": 2624,
            "use_bias": False,
            "global_attention_interval": 3,
            "local_attention_window_size": 128,
            "description": "ModernBERT-Large: 280M parameters, high-performance model",
        },
    }

    # Pretrained weights URLs (placeholders, update with actual URLs when available)
    PRETRAINED_WEIGHTS = {
        "tiny": {"uncased": "https://example.com/modern_bert_tiny_uncased.keras"},
        "base": {"uncased": "https://example.com/modern_bert_base_uncased.keras"},
        "large": {"uncased": "https://example.com/modern_bert_large_uncased.keras"},
    }

    # Default architecture constants
    DEFAULT_VOCAB_SIZE = 50368
    DEFAULT_TYPE_VOCAB_SIZE = 2
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPSILON = 1e-12
    DEFAULT_HIDDEN_ACT = "gelu"

    def __init__(
            self,
            vocab_size: int = DEFAULT_VOCAB_SIZE,
            hidden_size: int = 768,
            num_layers: int = 22,
            num_heads: int = 12,
            intermediate_size: int = 1152,
            hidden_act: str = DEFAULT_HIDDEN_ACT,
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            type_vocab_size: int = DEFAULT_TYPE_VOCAB_SIZE,
            initializer_range: float = DEFAULT_INITIALIZER_RANGE,
            layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
            use_bias: bool = False,
            global_attention_interval: int = 3,
            local_attention_window_size: int = 128,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(
            hidden_size, num_layers, num_heads,
            hidden_dropout_prob, attention_probs_dropout_prob,
            global_attention_interval
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
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        self.global_attention_interval = global_attention_interval
        self.local_attention_window_size = local_attention_window_size

        # Build the model architecture
        self._build_architecture()

        logger.info(
            f"Created ModernBERT foundation model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, heads={self.num_heads}"
        )

    def _validate_config(
            self,
            hidden_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dropout_prob: float,
            attention_probs_dropout_prob: float,
            global_attention_interval: int
    ) -> None:
        """Validate model configuration parameters."""
        if hidden_size <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("Sizes and layer/head counts must be positive.")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= hidden_dropout_prob <= 1.0):
            raise ValueError(f"hidden_dropout_prob must be between 0 and 1.")
        if not (0.0 <= attention_probs_dropout_prob <= 1.0):
            raise ValueError(f"attention_probs_dropout_prob must be between 0 and 1.")
        if global_attention_interval <= 0:
            raise ValueError("global_attention_interval must be positive.")

    def _build_architecture(self) -> None:
        """Build all model components (embeddings, encoder layers, final norm)."""
        self.embeddings = ModernBertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            dropout_rate=self.hidden_dropout_prob,
            use_bias=self.use_bias,
            name="embeddings",
        )

        self.encoder_layers: List[TransformerLayer] = []
        for i in range(self.num_layers):
            # Every k-th layer uses global attention, others use windowed.
            is_global = (i + 1) % self.global_attention_interval == 0

            attention_type = "multi_head" if is_global else "window"
            attention_args = (
                {} if is_global
                else {"window_size": self.local_attention_window_size}
            )

            layer = TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=attention_type,
                attention_args=attention_args,
                normalization_position='pre',
                ffn_type='geglu',
                ffn_args={'activation': self.hidden_act},
                dropout_rate=self.hidden_dropout_prob,
                attention_dropout_rate=self.attention_probs_dropout_prob,
                use_bias=self.use_bias,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name=f"encoder_layer_{i}",
            )
            self.encoder_layers.append(layer)

        # Final normalization layer after the transformer stack
        self.final_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            center=self.use_bias,  # Use bias for centering if use_bias=True
            name="final_layer_norm"
        )

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            attention_mask: Optional[keras.KerasTensor] = None,
            token_type_ids: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of the ModernBERT foundation model.

        :param inputs: Input token IDs or a dictionary containing 'input_ids'
            and other optional tensors like 'attention_mask'.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param attention_mask: Mask to avoid attention on padding tokens.
        :type attention_mask: Optional[keras.KerasTensor]
        :param token_type_ids: Token type IDs for distinguishing sequences.
        :type token_type_ids: Optional[keras.KerasTensor]
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
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
        else:
            input_ids = inputs

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            training=training
        )

        hidden_states = embedding_output
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )

        sequence_output = self.final_norm(hidden_states, training=training)

        return {
            "last_hidden_state": sequence_output,
            "attention_mask": attention_mask,
        }

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True,
            by_name: bool = True
    ) -> None:
        """Load pretrained weights into the model.

        :param weights_path: Path to the weights file (.keras format).
        :type weights_path: str
        :param skip_mismatch: Whether to skip layers with mismatched shapes.
        :type skip_mismatch: bool
        :param by_name: Whether to load weights by layer name.
        :type by_name: bool
        :raises FileNotFoundError: If weights_path doesn't exist.
        :raises ValueError: If weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        try:
            if not self.built:
                dummy_input = {
                    "input_ids": keras.random.uniform(
                        (1, 128), 0, self.vocab_size, dtype="int32"
                    )
                }
                self(dummy_input, training=False)
            logger.info(f"Loading pretrained weights from {weights_path}")
            self.load_weights(weights_path, skip_mismatch=skip_mismatch, by_name=by_name)
            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. Layers with shape "
                    "mismatches were skipped (e.g., embedding layer)."
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
        """Download pretrained weights from URL."""
        if variant not in ModernBERT.PRETRAINED_WEIGHTS:
            raise ValueError(f"No pretrained weights for variant '{variant}'.")
        if dataset not in ModernBERT.PRETRAINED_WEIGHTS[variant]:
            raise ValueError(f"No pretrained weights for dataset '{dataset}'.")

        url = ModernBERT.PRETRAINED_WEIGHTS[variant][dataset]
        logger.info(f"Downloading ModernBERT-{variant} ({dataset}) weights...")
        weights_path = keras.utils.get_file(
            fname=f"modern_bert_{variant}_{dataset}.keras",
            origin=url,
            cache_dir=cache_dir,
            cache_subdir="models/modern_bert"
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
    ) -> "ModernBERT":
        """Create a ModernBERT model from a predefined variant."""
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")
        logger.info(f"Creating ModernBERT-{variant.upper()} model")
        logger.info(f"Configuration: {description}")

        load_weights_path = None
        skip_mismatch = False
        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                try:
                    load_weights_path = cls._download_weights(variant, weights_dataset, cache_dir)
                except Exception as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {e}. "
                        "Continuing with random initialization."
                    )

            # Check for config mismatches that require skipping weights
            if kwargs.get("vocab_size", config.get("vocab_size")) != cls.DEFAULT_VOCAB_SIZE:
                skip_mismatch = True
                logger.info("Custom vocab_size differs from pretrained. Will skip embedding layer.")

        config.update(kwargs)
        model = cls(**config)

        if load_weights_path:
            try:
                model.load_pretrained_weights(load_weights_path, skip_mismatch=skip_mismatch)
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
                raise
        return model

    def get_config(self) -> Dict[str, Any]:
        """Return the model's configuration for serialization."""
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
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "use_bias": self.use_bias,
            "global_attention_interval": self.global_attention_interval,
            "local_attention_window_size": self.local_attention_window_size,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModernBERT":
        """Create a model instance from its configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the model summary with additional ModernBERT-specific information."""
        super().summary(**kwargs)
        logger.info("ModernBERT Foundation Model Configuration:")
        logger.info(
            f"  - Architecture: {self.num_layers} layers, {self.hidden_size} hidden size"
        )
        logger.info(
            f"  - Attention: Mixed Global/Window (Global every {self.global_attention_interval} layers)"
        )
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(f"  - Normalization: Pre-LN with final LayerNorm")
        logger.info(
            f"  - Feed-forward: GeGLU, {self.intermediate_size} intermediate size"
        )


# ---------------------------------------------------------------------

def create_modern_bert_with_head(
        bert_variant: str,
        task_config: NLPTaskConfig,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "uncased",
        cache_dir: Optional[str] = None,
        bert_config_overrides: Optional[Dict[str, Any]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Factory function to create a ModernBERT model with a task-specific head.

    :param bert_variant: The ModernBERT variant to use (e.g., "base", "large").
    :type bert_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task.
    :type task_config: NLPTaskConfig
    :param pretrained: If True, loads pretrained weights. If string,
        path to local weights file.
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset for pretrained weights ("uncased", etc.).
    :type weights_dataset: str
    :param cache_dir: Directory to cache downloaded weights.
    :type cache_dir: Optional[str]
    :param bert_config_overrides: Optional dictionary to override default BERT
        configuration for the chosen variant.
    :type bert_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Optional dictionary to override default head
        configuration.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :return: A complete `keras.Model` ready for the specified task.
    :rtype: keras.Model
    """
    bert_config_overrides = bert_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(f"Creating ModernBERT-{bert_variant} with a '{task_config.name}' head.")

    bert_encoder = ModernBERT.from_variant(
        bert_variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **bert_config_overrides
    )
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=bert_encoder.hidden_size,
        **head_config_overrides,
    )

    inputs = {
        "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
        "token_type_ids": keras.Input(shape=(None,), dtype="int32", name="token_type_ids"),
    }

    encoder_outputs = bert_encoder(inputs)

    # Some heads (like QuestionAnsweringHead) may internally use ops that
    # require the attention mask to be a float. This cast ensures compatibility.
    attention_mask_float = ops.cast(
        encoder_outputs["attention_mask"],
        dtype=encoder_outputs["last_hidden_state"].dtype
    )

    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": attention_mask_float,
    }
    task_outputs = task_head(head_inputs)

    model_name = f"modern_bert_{bert_variant}_with_{task_config.name}_head"
    model = keras.Model(inputs=inputs, outputs=task_outputs, name=model_name)

    logger.info(f"Successfully created model with {model.count_params():,} parameters.")
    return model

# ---------------------------------------------------------------------

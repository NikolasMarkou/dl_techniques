"""
FNet Model Implementation with Pretrained Support
==================================================

A complete and refactored implementation of the FNet (Fourier Transform-based
Neural Network) architecture with support for loading pretrained weights. This
version is designed as a pure foundation model, separating the core encoding
logic from task-specific heads for maximum flexibility, especially in
pre-training and multi-task fine-tuning scenarios.

Based on: "FNet: Mixing Tokens with Fourier Transforms"
(Lee-Thorp et al., 2021) https://arxiv.org/abs/2105.03824

Usage Examples:
--------------

.. code-block:: python

    import keras
    from dl_techniques.nlp.heads.factory import create_nlp_head
    from dl_techniques.nlp.heads.task_types import NLPTaskConfig, NLPTaskType

    # 1. Load pretrained FNet model
    fnet_encoder = FNet.from_variant("base", pretrained=True)

    # 2. Load from local weights file
    fnet_encoder = FNet.from_variant("large", pretrained="path/to/weights.keras")

    # 3. Create FNet with custom configuration
    fnet_encoder = FNet.from_variant("base", vocab_size=50000)

    # 4. Combine with task-specific head
    sentiment_config = NLPTaskConfig(
        name="sentiment",
        task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        num_classes=3
    )
    sentiment_head = create_nlp_head(
        task_config=sentiment_config,
        input_dim=fnet_encoder.hidden_size
    )

    # 5. Build complete model
    inputs = {
        "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
        "token_type_ids": keras.Input(shape=(None,), dtype="int32", name="token_type_ids")
    }
    fnet_outputs = fnet_encoder(inputs)
    task_outputs = sentiment_head(fnet_outputs)
    sentiment_model = keras.Model(inputs, task_outputs)

"""

import os
import keras
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.embedding.bert_embeddings import BertEmbeddings
from dl_techniques.layers.fnet_encoder_block import FNetEncoderBlock
from dl_techniques.layers.nlp_heads import NLPTaskConfig, create_nlp_head
from dl_techniques.layers.transformers import FFNType, NormalizationPositionType, NormalizationType

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FNet(keras.Model):
    """FNet (Fourier Transform-based Neural Network) model.

    This is a pure encoder implementation with pretrained weights support,
    designed to produce contextual token representations. It separates the
    core transformer-like architecture from any task-specific layers (like
    pooling or classification heads), making it highly flexible for pre-training,
    fine-tuning, and multi-task learning.

    The model expects inputs as a dictionary containing 'input_ids', and
    optionally 'attention_mask', and 'token_type_ids'. It
    outputs a dictionary containing the 'last_hidden_state' and the forwarded
    'attention_mask'.

    **Architecture Overview:**

    .. code-block:: text

        Input(input_ids, attention_mask, token_type_ids)
               │
               ▼
        Embeddings(Word + Position + Token Type) -> LayerNorm -> Dropout
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
        Output Dictionary {
            "last_hidden_state": [batch, seq_len, hidden_size],
            "attention_mask": [batch, seq_len]
        }

    :param vocab_size: Size of the vocabulary. Defaults to 30522.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Defaults to 768.
    :type hidden_size: int
    :param num_layers: Number of hidden transformer layers. Defaults to 12.
    :type num_layers: int
    :param intermediate_size: Dimensionality of the "intermediate"
        (feed-forward) layer. Defaults to 3072.
    :type intermediate_size: int
    :param hidden_dropout_prob: Dropout probability for all fully connected
        layers in embeddings and encoder. Defaults to 0.1.
    :type hidden_dropout_prob: float
    :param max_position_embeddings: Maximum sequence length for positional
        embeddings. Defaults to 512.
    :type max_position_embeddings: int
    :param type_vocab_size: Vocabulary size for token type IDs.
        Defaults to 2.
    :type type_vocab_size: int
    :param initializer_range: Stddev of truncated normal initializer for
        all weight matrices. Defaults to 0.02.
    :type initializer_range: float
    :param layer_norm_eps: Epsilon for normalization layers. Defaults to 1e-12.
    :type layer_norm_eps: float
    :param pad_token_id: ID of padding token. Defaults to 0.
    :type pad_token_id: int
    :param position_embedding_type: Type of position embedding.
        Defaults to "absolute".
    :type position_embedding_type: str
    :param normalization_type: Type of normalization layer.
        Defaults to "layer_norm".
    :type normalization_type: str
    :param normalization_position: Position of normalization ('pre' or 'post').
        Defaults to "post".
    :type normalization_position: str
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
    :vartype embeddings: dl_techniques.layers.embedding.bert_embeddings.BertEmbeddings
    :ivar encoder_layers: A list of `FNetEncoderBlock` instances.
    :vartype encoder_layers: list[dl_techniques.layers.fnet_encoder_block.FNetEncoderBlock]

    :raises ValueError: If invalid configuration parameters are provided.

    Example:
        .. code-block:: python

            # Create standard FNet-base model
            model = FNet.from_variant("base")

            # Load pretrained FNet-base
            model = FNet.from_variant("base", pretrained=True)

            # Load from local file
            model = FNet.from_variant("large", pretrained="path/to/weights.keras")

            # Use the model
            inputs = {
                "input_ids": keras.random.uniform((2, 128), 0, 30522, dtype="int32"),
                "attention_mask": keras.ops.ones((2, 128), dtype="int32")
            }
            outputs = model(inputs)
            print(outputs["last_hidden_state"].shape)
            # (2, 128, 768)

    """

    # Model variant configurations following FNet paper specifications
    MODEL_VARIANTS = {
        "large": {
            "hidden_size": 1024,
            "num_layers": 24,
            "intermediate_size": 4096,
            "description": "FNet-Large: 340M parameters, maximum performance",
        },
        "base": {
            "hidden_size": 768,
            "num_layers": 12,
            "intermediate_size": 3072,
            "description": "FNet-Base: 110M parameters, suitable for most applications",
        },
        "small": {
            "hidden_size": 512,
            "num_layers": 6,
            "intermediate_size": 2048,
            "description": "FNet-Small: Lightweight variant for resource-constrained environments",
        },
        "tiny": {
            "hidden_size": 256,
            "num_layers": 4,
            "intermediate_size": 512,
            "description": "FNet-Tiny: Ultra-lightweight for mobile/edge deployment",
        },
    }

    # Pretrained weights URLs (replace with actual URLs when available)
    PRETRAINED_WEIGHTS = {
        "base": {
            "uncased": "https://example.com/fnet_base_uncased.keras",
            "cased": "https://example.com/fnet_base_cased.keras",
        },
        "large": {
            "uncased": "https://example.com/fnet_large_uncased.keras",
            "cased": "https://example.com/fnet_large_cased.keras",
        },
        "small": {
            "uncased": "https://example.com/fnet_small_uncased.keras",
            "cased": "https://example.com/fnet_small_cased.keras",
        },
        "tiny": {
            "uncased": "https://example.com/fnet_tiny_uncased.keras",
            "cased": "https://example.com/fnet_tiny_cased.keras",
        },
    }

    # Default architecture constants
    DEFAULT_VOCAB_SIZE = 30522
    DEFAULT_MAX_POSITION_EMBEDDINGS = 512
    DEFAULT_TYPE_VOCAB_SIZE = 2
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPSILON = 1e-12
    DEFAULT_PAD_TOKEN_ID = 0

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_size: int = 768,
        num_layers: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS,
        type_vocab_size: int = DEFAULT_TYPE_VOCAB_SIZE,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        position_embedding_type: str = "absolute",
        normalization_type: NormalizationType = "layer_norm",
        normalization_position: NormalizationPositionType = "post",
        ffn_type: FFNType = "mlp",
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initializes the FNet model instance.

        :param vocab_size: Size of the vocabulary.
        :type vocab_size: int
        :param hidden_size: Dimensionality of encoder layers.
        :type hidden_size: int
        :param num_layers: Number of hidden transformer layers.
        :type num_layers: int
        :param intermediate_size: Dimensionality of the FFN layer.
        :type intermediate_size: int
        :param hidden_dropout_prob: Dropout probability for embeddings/encoder.
        :type hidden_dropout_prob: float
        :param max_position_embeddings: Maximum sequence length.
        :type max_position_embeddings: int
        :param type_vocab_size: Vocabulary size for token type IDs.
        :type type_vocab_size: int
        :param initializer_range: Stddev for weight initialization.
        :type initializer_range: float
        :param layer_norm_eps: Epsilon for normalization layers.
        :type layer_norm_eps: float
        :param pad_token_id: ID of the padding token.
        :type pad_token_id: int
        :param position_embedding_type: Type of position embedding.
        :type position_embedding_type: str
        :param normalization_type: Type of normalization layer.
        :type normalization_type: str
        :param normalization_position: Position of normalization ('pre'/'post').
        :type normalization_position: str
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
        self._validate_config(vocab_size, hidden_size, num_layers, hidden_dropout_prob)

        # Store all configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate

        # Build the model architecture
        self._build_architecture()

        logger.info(
            f"Created FNet foundation model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}"
        )

    def _validate_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        hidden_dropout_prob: float,
    ) -> None:
        """Validate model configuration parameters.

        :param vocab_size: Size of the vocabulary.
        :type vocab_size: int
        :param hidden_size: Dimensionality of encoder layers.
        :type hidden_size: int
        :param num_layers: Number of transformer layers.
        :type num_layers: int
        :param hidden_dropout_prob: Dropout probability for hidden layers.
        :type hidden_dropout_prob: float
        :raises ValueError: If any configuration value is invalid.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 <= hidden_dropout_prob <= 1.0):
            raise ValueError(
                f"hidden_dropout_prob must be between 0 and 1, "
                f"got {hidden_dropout_prob}"
            )

    def _build_architecture(self) -> None:
        """Build all model components (embeddings and encoder layers)."""
        self.embeddings = BertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout_prob=self.hidden_dropout_prob,
            normalization_type=self.normalization_type,
            name="embeddings",
        )

        self.encoder_layers: List[FNetEncoderBlock] = []
        for i in range(self.num_layers):
            encoder_layer = FNetEncoderBlock(
                intermediate_dim=self.intermediate_size,
                dropout_rate=self.hidden_dropout_prob,
                normalization_type=self.normalization_type,
                ffn_type=self.ffn_type,
                name=f"encoder_layer_{i}",
            )
            self.encoder_layers.append(encoder_layer)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of the FNet foundation model.

        :param inputs: Input token IDs or a dictionary containing 'input_ids'
            and other optional tensors like 'attention_mask'.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param attention_mask: Mask to avoid attention on padding tokens.
        :type attention_mask: Optional[keras.KerasTensor]
        :param token_type_ids: Token type IDs for distinguishing sequences.
        :type token_type_ids: Optional[keras.KerasTensor]
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
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
        else:
            input_ids = inputs

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training,
        )

        hidden_states = embedding_output
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(
                hidden_states, attention_mask=attention_mask, training=training
            )

        return {
            "last_hidden_state": hidden_states,
            "attention_mask": attention_mask,
        }

    def load_pretrained_weights(
        self, weights_path: str, skip_mismatch: bool = True, by_name: bool = True
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

                model = FNet.from_variant("base", vocab_size=50000)
                model.load_pretrained_weights(
                    "fnet_base_uncased.keras",
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
                    "attention_mask": keras.ops.ones((1, 128), dtype="int32"),
                }
                self(dummy_input, training=False)

            logger.info(f"Loading pretrained weights from {weights_path}")

            # Load weights with appropriate settings
            self.load_weights(
                weights_path, skip_mismatch=skip_mismatch, by_name=by_name
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
        variant: str, dataset: str = "uncased", cache_dir: Optional[str] = None
    ) -> str:
        """Download pretrained weights from URL.

        :param variant: Model variant name.
        :type variant: str
        :param dataset: Dataset/version the weights were trained on.
            Options: "uncased", "cased".
        :type dataset: str
        :param cache_dir: Directory to cache downloaded weights.
            If None, uses default Keras cache directory.
        :type cache_dir: Optional[str]
        :return: Path to the downloaded weights file.
        :rtype: str
        :raises ValueError: If variant or dataset is not available.

        Example:
            .. code-block:: python

                weights_path = FNet._download_weights("base", "uncased")
        """
        if variant not in FNet.PRETRAINED_WEIGHTS:
            raise ValueError(
                f"No pretrained weights available for variant '{variant}'. "
                f"Available variants: {list(FNet.PRETRAINED_WEIGHTS.keys())}"
            )

        if dataset not in FNet.PRETRAINED_WEIGHTS[variant]:
            raise ValueError(
                f"No pretrained weights available for dataset '{dataset}'. "
                f"Available datasets for {variant}: "
                f"{list(FNet.PRETRAINED_WEIGHTS[variant].keys())}"
            )

        url = FNet.PRETRAINED_WEIGHTS[variant][dataset]

        logger.info(f"Downloading FNet-{variant} ({dataset}) weights...")

        # Download weights using Keras utility
        weights_path = keras.utils.get_file(
            fname=f"fnet_{variant}_{dataset}.keras",
            origin=url,
            cache_dir=cache_dir,
            cache_subdir="models/fnet",
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
        **kwargs: Any,
    ) -> "FNet":
        """Create an FNet model from a predefined variant.

        :param variant: The name of the variant, one of "base", "large",
            "small", "tiny".
        :type variant: str
        :param pretrained: If True, loads pretrained weights from default URL.
            If string, treats it as a path to local weights file.
        :type pretrained: Union[bool, str]
        :param weights_dataset: Dataset/version for pretrained weights.
            Options: "uncased", "cased".
            Only used if pretrained=True.
        :type weights_dataset: str
        :param cache_dir: Directory to cache downloaded weights.
        :type cache_dir: Optional[str]
        :param kwargs: Additional arguments to override the variant's defaults.
        :type kwargs: Any
        :return: An FNet model instance configured for the specified variant.
        :rtype: FNet
        :raises ValueError: If the specified variant is not recognized.

        Example:
            .. code-block:: python

                # Load pretrained FNet-base
                model = FNet.from_variant("base", pretrained=True)

                # Load pretrained FNet-large (cased)
                model = FNet.from_variant("large", pretrained=True, weights_dataset="cased")

                # Load from local file
                model = FNet.from_variant("base", pretrained="path/to/weights.keras")

                # Create with custom vocab size (will skip embedding weights)
                model = FNet.from_variant("base", pretrained=True, vocab_size=50000)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")

        logger.info(f"Creating FNet-{variant.upper()} model")
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
                        variant=variant, dataset=weights_dataset, cache_dir=cache_dir
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
            pretrained_config_keys = [
                "hidden_size",
                "num_layers",
                "intermediate_size",
            ]
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
                    by_name=True,
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
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "intermediate_size": self.intermediate_size,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "max_position_embeddings": self.max_position_embeddings,
                "type_vocab_size": self.type_vocab_size,
                "initializer_range": self.initializer_range,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "position_embedding_type": self.position_embedding_type,
                "normalization_type": self.normalization_type,
                "normalization_position": self.normalization_position,
                "ffn_type": self.ffn_type,
                "use_stochastic_depth": self.use_stochastic_depth,
                "stochastic_depth_rate": self.stochastic_depth_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FNet":
        """Create a model instance from its configuration.

        :param config: A dictionary containing the model's configuration.
        :type config: Dict[str, Any]
        :return: A new FNet model instance.
        :rtype: FNet
        """
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the model summary with additional FNet-specific information.

        :param kwargs: Additional arguments passed to `keras.Model.summary`.
        """
        super().summary(**kwargs)
        logger.info("FNet Foundation Model Configuration:")
        logger.info(
            f"  - Architecture: {self.num_layers} layers, "
            f"{self.hidden_size} hidden size"
        )
        logger.info(f"  - Token Mixing: Fourier Transform (parameter-free)")
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(f"  - Max sequence length: {self.max_position_embeddings}")
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


def create_fnet_with_head(
    fnet_variant: str,
    task_config: NLPTaskConfig,
    pretrained: Union[bool, str] = False,
    weights_dataset: str = "uncased",
    cache_dir: Optional[str] = None,
    fnet_config_overrides: Optional[Dict[str, Any]] = None,
    head_config_overrides: Optional[Dict[str, Any]] = None,
    sequence_length: Optional[int] = None,
) -> keras.Model:
    """Factory function to create an FNet model with a task-specific head.

    This function demonstrates the intended integration pattern:
    1. Instantiate a foundational `FNet` model (optionally pretrained).
    2. Instantiate a task-specific head from the `dl_techniques.nlp.heads`
       factory.
    3. Combine them into a single, end-to-end `keras.Model`.

    :param fnet_variant: The FNet variant to use (e.g., "base", "large").
    :type fnet_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task.
    :type task_config: NLPTaskConfig
    :param pretrained: If True, loads pretrained weights. If string,
        path to local weights file.
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset for pretrained weights ("uncased", "cased", etc.).
    :type weights_dataset: str
    :param cache_dir: Directory to cache downloaded weights.
    :type cache_dir: Optional[str]
    :param fnet_config_overrides: Optional dictionary to override default FNet
        configuration for the chosen variant. Defaults to None.
    :type fnet_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Optional dictionary to override default head
        configuration. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param sequence_length: The fixed sequence length for the model's inputs.
        If None, the model will have dynamic sequence length, but this may
        not be compatible with FNet's Fourier Transform layer which
        requires a known length at build time. Defaults to None.
    :type sequence_length: Optional[int]
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

            # Create the full model with pretrained FNet
            ner_model = create_fnet_with_head(
                fnet_variant="base",
                task_config=ner_task,
                pretrained=True,
                sequence_length=128, # Provide a fixed length
                head_config_overrides={"use_task_attention": True}
            )
            ner_model.summary()
    """
    fnet_config_overrides = fnet_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(f"Creating FNet-{fnet_variant} with a '{task_config.name}' head.")

    # 1. Create the foundational FNet model (with optional pretrained weights)
    fnet_encoder = FNet.from_variant(
        fnet_variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **fnet_config_overrides,
    )

    # 2. Create the task head
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=fnet_encoder.hidden_size,
        **head_config_overrides,
    )

    # 3. Define inputs and build the end-to-end model
    input_shape = (sequence_length,) if sequence_length is not None else (None,)
    inputs = {
        "input_ids": keras.Input(shape=input_shape, dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(
            shape=input_shape, dtype="int32", name="attention_mask"
        ),
        "token_type_ids": keras.Input(
            shape=input_shape, dtype="int32", name="token_type_ids"
        ),
    }

    # Get hidden states from the encoder
    encoder_outputs = fnet_encoder(inputs)

    # Pass encoder outputs to the task head
    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": encoder_outputs["attention_mask"],
    }
    task_outputs = task_head(head_inputs)

    # Create the final model
    model_name = f"fnet_{fnet_variant}_with_{task_config.name}_head"
    model = keras.Model(inputs=inputs, outputs=task_outputs, name=model_name)

    logger.info(
        f"Successfully created model with {model.count_params():,} parameters."
    )
    return model

# ---------------------------------------------------------------------
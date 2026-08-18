"""
FNet, a transformer encoder whose token mixer is an unparameterized Fourier
transform instead of self-attention.

This model embodies the principle that most of what self-attention contributes
to an encoder is token mixing, not the specific content-dependent weights it
computes. Self-attention costs `O(L^2)` time and memory in sequence length and
carries four projection matrices per layer; if the downstream layers can
recover the needed interactions from any sufficiently rich mixing of positions,
then the mixer itself need not be learned at all. FNet takes that idea to its
limit and replaces the whole attention sublayer with a two-dimensional discrete
Fourier transform:

`y = Re(F_seq(F_hidden(x)))`

The transform is applied along the hidden dimension and then along the sequence
dimension, and only the real part is kept. Discarding the imaginary component
is not an approximation to be apologized for: it keeps the sublayer a real-valued
map with the same shape as its input, so the residual connection and the
feed-forward network that follow are untouched, and the imaginary part carries
no information the subsequent layers are set up to consume.

Two consequences follow, and they are the whole argument for the architecture.
The mixer has zero parameters and zero learned state, so a layer's entire
capacity sits in its feed-forward network. And because every output position is
a fixed linear combination of every input position, one layer already achieves
global receptive field -- the property attention is usually credited with -
without any pairwise score matrix. The reported cost is a few points of
accuracy against a BERT baseline in exchange for substantially faster training,
which is a favourable trade whenever the encoder is a component rather than the
product.

The consequence that matters in code is that masking works differently here,
and the difference is easy to get wrong. There is no softmax to add a `-inf`
bias to, so `attention_mask` is applied MULTIPLICATIVELY and only AFTER mixing:
a padded position's own output is zeroed, but it has already contributed to the
mixed values of every real token, because a DFT cannot be told to skip an
index. The usual "masked tokens are invisible" guarantee therefore does not
hold. The mask is still forwarded on the output so downstream heads can pool
correctly.

Everything else is a standard encoder: BERT-style embeddings, then
`num_layers` blocks of `Fourier mix -> residual -> norm -> FFN -> residual ->
norm` (post-normalization by default, with optional stochastic depth on each
branch), emitting `{"last_hidden_state", "attention_mask"}` with no pooler and
no task head, so a single encoder can back several heads at once. Four preset
variants span tiny through large.

`normalization_position='pre'` switches every block to `x = input +
branch(Norm(input))` and adds the stack-final normalization that arrangement
requires; `position_embedding_type` selects between a learned absolute table and
a fixed sinusoidal one. Both are real knobs as of 2026-08-15 — until then each
was validated, stored and serialized while the value never reached the layer that
would have honoured it, so `normalization_position='pre'` built a post-norm stack
and `position_embedding_type` was ignored entirely.

No pretrained weights are distributed with this package. `pretrained=True`
raises `NotImplementedError` rather than warning and returning a randomly
initialized model, which is a deliberate choice: the previous behaviour held a
table of unreachable weight URLs and swallowed the download failure, making an
unavailable checkpoint silently indistinguishable from a successful load. Pass
a local `.keras` path to `pretrained` instead.

References:
    - Lee-Thorp et al., 2021. FNet: Mixing Tokens with Fourier Transforms.
      (https://arxiv.org/abs/2105.03824)
    - Devlin et al., 2018. BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding.
      (https://arxiv.org/abs/1810.04805)
    - Tay et al., 2020. Efficient Transformers: A Survey.
      (https://arxiv.org/abs/2009.06732)
    - Tolstikhin et al., 2021. MLP-Mixer: An all-MLP Architecture for Vision.
      (https://arxiv.org/abs/2105.01601)
"""


import os
import keras
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.embedding.bert_embeddings import BertEmbeddings
from dl_techniques.layers.fnet_encoder_block import FNetEncoderBlock
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.heads.nlp import NLPTaskConfig, create_nlp_head
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
    :param position_embedding_type: How positional information is produced --
        ``'learned'`` (default; BERT's learned absolute table) or
        ``'sinusoidal'``. Forwarded to :class:`BertEmbeddings`, which raises on
        anything else. The legacy spelling ``'absolute'`` is normalized to
        ``'learned'``; it was this model's default while the value was never
        forwarded at all.
    :type position_embedding_type: str
    :param normalization_type: Type of normalization layer.
        Defaults to "layer_norm".
    :type normalization_type: str
    :param normalization_position: ``'post'`` (default, the original FNet
        arrangement, ``x = Norm(input + branch(input))``) or ``'pre'``
        (``x = input + branch(Norm(input))``, plus a stack-final normalization
        this model owns). Forwarded to every :class:`FNetEncoderBlock`.
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

            # Load from local file (`pretrained=True` raises NotImplementedError)
            model = FNet.from_variant("large", pretrained="path/to/weights.keras")

            # Use the model
            inputs = {
                "input_ids": keras.random.randint((2, 128), 0, 30522, dtype="int32"),
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
        position_embedding_type: str = "learned",
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
        :param position_embedding_type: ``'learned'`` (default) or
            ``'sinusoidal'``; forwarded to :class:`BertEmbeddings`. ``'absolute'``
            is accepted as the legacy spelling of ``'learned'``.
        :type position_embedding_type: str
        :param normalization_type: Type of normalization layer.
        :type normalization_type: str
        :param normalization_position: ``'post'`` (default) or ``'pre'``;
            forwarded to every :class:`FNetEncoderBlock`. ``'pre'`` also adds a
            stack-final normalization.
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
        # `'absolute'` was this model's default until 2026-08-15, while the value
        # was never forwarded to `BertEmbeddings` -- which has no such value at
        # all (`VALID_POSITION_EMBEDDING_TYPES` is `('learned', 'sinusoidal')`)
        # and defaulted to `'learned'`. Now that the value IS forwarded, the
        # legacy spelling is normalized ONCE, here, so every stored config and
        # every `get_config()` carries the single live spelling. This is a rename
        # of a value that always meant BERT's learned absolute table, not a
        # silent fallback: anything else `BertEmbeddings` does not recognize
        # still raises there.
        if position_embedding_type == "absolute":
            position_embedding_type = "learned"
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
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-071: `position_embedding_type`
        # is FORWARDED. It used to be validated, stored and serialized here while
        # `BertEmbeddings` silently used its own default ('learned'), so
        # `FNet(position_embedding_type='sinusoidal')` built a learned table. Do
        # NOT drop the argument from this call to "keep the default stable" --
        # that is the bug. See decisions.md D-071.
        self.embeddings = BertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            dropout_rate=self.hidden_dropout_prob,
            normalization_type=self.normalization_type,
            position_embedding_type=self.position_embedding_type,
            name="embeddings",
        )

        # Per-block linear drop-path schedule (0 -> stochastic_depth_rate across
        # the stack). Computed unconditionally because it is pure arithmetic; it
        # only reaches the blocks as a live rate when use_stochastic_depth is on,
        # and FNetEncoderBlock ignores the rate entirely when the flag is False.
        drop_path_rates = linear_drop_path_rates(
            num_blocks=self.num_layers, max_rate=self.stochastic_depth_rate
        )

        self.encoder_layers: List[FNetEncoderBlock] = []
        for i in range(self.num_layers):
            encoder_layer = FNetEncoderBlock(
                intermediate_dim=self.intermediate_size,
                dropout_rate=self.hidden_dropout_prob,
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=drop_path_rates[i],
                name=f"encoder_layer_{i}",
            )
            self.encoder_layers.append(encoder_layer)

        # A pre-norm stack leaves its last residual sum unnormalized, so the
        # model owns the stack-final normalization. Built only for 'pre': adding
        # it unconditionally would change every existing post-norm checkpoint's
        # weight tree.
        self.final_norm = None
        if self.normalization_position == 'pre':
            self.final_norm = create_normalization_layer(
                normalization_type=self.normalization_type,
                epsilon=self.layer_norm_eps,
                name="final_norm",
            )

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

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states, training=training)

        return {
            "last_hidden_state": hidden_states,
            "attention_mask": attention_mask,
        }

    def load_pretrained_weights(
        self, weights_path: str, skip_mismatch: bool = True
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
            if not self.built:
                dummy_input = {
                    "input_ids": keras.random.uniform(
                        (1, 128), 0, self.vocab_size, dtype="int32"
                    ),
                    "attention_mask": keras.ops.ones((1, 128), dtype="int32"),
                }
                self(dummy_input, training=False)

            logger.info(f"Loading pretrained weights from {weights_path}")

            # Keras 3 removed `by_name` from `Model.load_weights` — the
            # signature is `(filepath, skip_mismatch=False, **kwargs)` and it
            # REJECTS the unknown keyword, so this call raised
            # `ValueError: Invalid keyword arguments: {'by_name': True}` for
            # every caller. It went unnoticed because the only route here was
            # `pretrained=<path>` and the enclosing except turned the failure
            # into a warning that continued with random weights.
            report = load_weights_from_checkpoint(
                target=self,
                ckpt_path=weights_path,
                skip_prefixes=(),
                strict=not skip_mismatch,
            )
            logger.info(f"Weight transfer complete: {report}")

            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. "
                    "Layers with shape mismatches were skipped (e.g., embedding layer)."
                )
            else:
                logger.info("All weights loaded successfully.")

        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

    # `_download_weights` raises instead of falling back to random init. The
    # previous version held a `PRETRAINED_WEIGHTS` table of placeholder URLs on
    # a non-existent host; `from_variant` caught the download failure, logged a
    # warning and continued with random initialization, so `pretrained=True`
    # silently produced untrained weights. Do NOT reinstate a warn-and-return
    # branch here or in `from_variant`. No public FNet weights are distributed
    # with dl_techniques; pass a local path via
    # `pretrained="/path/to/file.keras"` or use `pretrained=False` (default).
    @staticmethod
    def _download_weights(
        variant: str, dataset: str = "uncased", cache_dir: Optional[str] = None
    ) -> str:
        """Resolve a download path for pretrained weights of ``variant``.

        Not implemented: no public FNet weights ship with ``dl_techniques``.
        Always raises, so an unavailable checkpoint is never silently
        indistinguishable from a successful load.

        :param variant: Model variant name (unused).
        :type variant: str
        :param dataset: Dataset/version identifier (unused).
        :type dataset: str
        :param cache_dir: Cache directory (unused).
        :type cache_dir: Optional[str]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained FNet weights are distributed with dl_techniques "
            f"(requested variant '{variant}', dataset '{dataset}'). Pass a local "
            f"checkpoint instead: FNet.from_variant('{variant}', "
            f"pretrained='/path/to/weights.keras')."
        )

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
        :param pretrained: If a string, a path to a local ``.keras`` weights
            file. If True, raises ``NotImplementedError`` -- no public FNet
            weights ship with ``dl_techniques``. If False (default), the model
            is randomly initialized.
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
        :raises NotImplementedError: If ``pretrained`` is True.

        Example:
            .. code-block:: python

                # Random init
                model = FNet.from_variant("base")

                # Load from local file
                model = FNet.from_variant("base", pretrained="path/to/weights.keras")

                # Create with custom vocab size (will skip embedding weights)
                model = FNet.from_variant(
                    "base", pretrained="path/to/weights.keras", vocab_size=50000
                )
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

        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                load_weights_path = cls._download_weights(
                    variant=variant, dataset=weights_dataset, cache_dir=cache_dir
                )

            pretrained_vocab_size = cls.DEFAULT_VOCAB_SIZE
            custom_vocab_size = kwargs.get("vocab_size", config.get("vocab_size"))

            if custom_vocab_size and custom_vocab_size != pretrained_vocab_size:
                skip_mismatch = True
                logger.info(
                    f"vocab_size ({custom_vocab_size}) differs from pretrained "
                    f"({pretrained_vocab_size}). Will skip embedding layer weights."
                )

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

        config.update(kwargs)
        model = cls(**config)

        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path,
                    skip_mismatch=skip_mismatch
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
        logger.info("  - Token Mixing: Fourier Transform (parameter-free)")
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

            from dl_techniques.layers.heads.nlp import NLPTaskType

            # Define a task
            ner_task = NLPTaskConfig(
                name="ner",
                task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
                num_classes=9
            )

            # Create the full model (pass a local .keras path to `pretrained`
            # to warm-start; `pretrained=True` raises NotImplementedError)
            ner_model = create_fnet_with_head(
                fnet_variant="base",
                task_config=ner_task,
                sequence_length=128, # Provide a fixed length
                head_config_overrides={"use_task_attention": True}
            )
            ner_model.summary()
    """
    fnet_config_overrides = fnet_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(f"Creating FNet-{fnet_variant} with a '{task_config.name}' head.")

    fnet_encoder = FNet.from_variant(
        fnet_variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **fnet_config_overrides,
    )

    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=fnet_encoder.hidden_size,
        **head_config_overrides,
    )

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
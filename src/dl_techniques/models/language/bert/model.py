"""
BERT implements a bidirectional transformer encoder as a foundation model, in the `BERT`
class and its `create_bert` factory. Masked language modeling lets every layer attend over
the whole input sequence, unlike an autoregressive model that must hide future tokens from
each position. By default (`include_top=False`) the class is a pure encoder: it returns
`{"last_hidden_state", "attention_mask"}` with no pooler or head. With `include_top=True`
and a `task_config` it owns ONE task head from `layers/heads/nlp`, named `task_head`, and
returns that head's output instead (a bare tensor for every task but question answering).
Every head weight lives under `task_head/`, so the encoder's weight paths are the same with
or without a head and with any task, which is what lets an encoder trained under one head
be reused, weights unchanged, under another. Attention type, FFN type,
normalization type and normalization position are constructor arguments routed through
`TransformerLayer`, so a caller can swap in a different block variant without forking the
file; the embedding layer is the shared `BertEmbeddings`, also used by `distilbert/` and
`modern_bert/`. Four preset variants span BERT-Tiny (4 layers, 256 hidden) to BERT-Large
(24 layers, 1024 hidden); see `README.md` for parameter counts. No pretrained weights ship
with this package -- `pretrained=True` raises `NotImplementedError`. Local checkpoints load
by path, skipping mismatched embedding shapes by name.

References:
    - Devlin et al., 2018. BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding.
      (https://arxiv.org/abs/1810.04805)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Xiong et al., 2020. On Layer Normalization in the Transformer
      Architecture. (https://arxiv.org/abs/2002.04745)
    - Turc et al., 2019. Well-Read Students Learn Better: On the Importance of
      Pre-training Compact Models. (https://arxiv.org/abs/1908.08962)
"""


import os
import keras
from typing import Optional, Union, Any, Dict, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.sequence_length_guard import (
    RecordedBuildShapeMixin,
    check_sequence_length,
    refuse_static_sequence_length,
)
from dl_techniques.utils.weight_transfer import (
    load_backbone_from_checkpoint,
)
from dl_techniques.layers.transformers import (
    FFNType,
    AttentionType,
    TransformerLayer,
    NormalizationType,
    NormalizationPositionType,
)
from dl_techniques.layers.activations import resolve_activation
from dl_techniques.layers.embedding.bert_embeddings import BertEmbeddings
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    build_model_task_head,
    finalize_head_outputs,
    task_config_to_dict,
    validate_task_head_knobs,
)
from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.bert.model")
class BERT(RecordedBuildShapeMixin, keras.Model):
    """BERT (Bidirectional Encoder Representations from Transformers) model.

    An encoder with pretrained weights support, designed to produce contextual
    token representations. With ``include_top=False`` (the default) it holds no
    task-specific layer at all, which keeps it usable for pre-training,
    fine-tuning and multi-task learning. With ``include_top=True`` it also owns
    one NLP task head, built from ``task_config``.

    The model expects a tensor of ``input_ids``, or a dictionary containing
    'input_ids' and optionally 'attention_mask', 'token_type_ids', and
    'position_ids'. Headless, it outputs a dictionary containing the
    'last_hidden_state' and the forwarded 'attention_mask'. With a head it
    outputs what the head returns, reduced by
    :func:`~dl_techniques.layers.heads.nlp.finalize_head_outputs`.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input dict                          │
        │  input_ids       [B, S]    (required)│
        │  attention_mask  [B, S]    (optional)│
        │  token_type_ids  [B, S]    (optional)│
        │  position_ids    [B, S]    (optional)│
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  BertEmbeddings                      │
        │    Word ⊕ Position ⊕ TokenType       │
        │    → Norm(layer_norm_eps) → Dropout  │
        │    (learned | sinusoidal positions)  │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  TransformerLayer₁ (Self-Attn → FFN) │
        └───────────────┬──────────────────────┘
                        │
                        ▼
                       ...
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  TransformerLayerₙ (Self-Attn → FFN) │
        └───────────────┬──────────────────────┘
                        │
        ┌───────────────┴──────────────────────┐
        │  TransformerLayer internals          │
        │    x ──► Attn ──► (+) ──► Norm       │
        │    └──── residual ──┘                │
        │    x ──► FFN  ──► (+) ──► Norm       │
        │    └──── residual ──┘                │
        │  'pre' puts each Norm before its     │
        │  sublayer instead of after           │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Output dict                         │
        │    "last_hidden_state"  [B, S, H]    │
        │    "attention_mask"     [B, S]       │
        │  Mask echoed; ones_like(input_ids)   │
        │  when no attention_mask is supplied  │
        └──────────────────────────────────────┘

    Variants:

    .. code-block:: text

        variant    layers   hidden   heads   intermediate
        tiny         4        256      4        1024
        small        6        512      8        2048
        base        12        768     12        3072
        large       24       1024     16        4096

    Head integration (``include_top=True``):

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  same inputs as the headless model   │
        │  (only input_ids is required)        │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  embeddings → encoder layers         │
        │  (identical weights and paths)       │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  task_head (build_model_task_head)   │
        │    {hidden_states: last_hidden_state,│
        │     attention_mask: mask cast to the │
        │     hidden-state dtype}              │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  drop 'probabilities'/'predictions'  │
        │  when 'logits' is present, then      │
        │  unwrap any 1-key dict → tensor      │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  21 of 23 tasks → bare tensor        │
        │  QA / SPAN_EXTRACTION → dict of      │
        │    {start_logits, end_logits}        │
        │  MULTIPLE_CHOICE → ValueError        │
        └──────────────────────────────────────┘

    :param vocab_size: Size of the vocabulary. Defaults to 30522.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Defaults to 768.
    :type hidden_size: int
    :param num_layers: Number of hidden transformer layers. Defaults to 12.
    :type num_layers: int
    :param num_heads: Number of attention heads for each attention layer.
        Defaults to 12.
    :type num_heads: int
    :param intermediate_size: Dimensionality of the "intermediate"
        (feed-forward) layer. Defaults to 3072.
    :type intermediate_size: int
    :param hidden_act: The non-linear activation function in the encoder.
        Defaults to ``"gelu_tanh"``, the tanh approximation used by the
        original release this class ports
        (https://github.com/google-research/bert/blob/master/modeling.py).
        Pass ``"gelu"`` for the exact/erf form instead; the two differ by up to
        4.732e-04 per activation call, so this is a forward-pass choice, not a
        training-only one. Resolution goes through
        :func:`dl_techniques.layers.activations.resolve_activation`, which
        understands the extended names on top of the Keras vocabulary.
    :type hidden_act: str
    :param hidden_dropout_rate: Dropout probability for all fully connected
        layers in embeddings and encoder. Defaults to 0.1. Upstream this field is
        ``BertConfig.hidden_dropout_prob``; it is spelled ``_rate`` here because
        every dropout rate in this repository is (D-130), and the value and
        meaning are unchanged.
    :type hidden_dropout_rate: float
    :param attention_probs_dropout_rate: Dropout ratio for attention
        probabilities. Defaults to 0.1. Upstream:
        ``BertConfig.attention_probs_dropout_prob``.
    :type attention_probs_dropout_rate: float
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
        Applies to the embedding norm and to every one of the
        ``2 * num_layers`` in-block norms. Before 2026-08-19 the in-block
        norms ignored this value and ran at the normalization factory's
        ``1e-6`` default instead; weight shapes are unchanged, so existing
        ``.keras`` files still load, but forward values move slightly
        (measured at ``hidden_size=64, num_layers=4``: max |delta| 1.9e-06,
        mean 4.0e-07). See ``decisions.md`` D-007.
    :type layer_norm_eps: float
    :param pad_token_id: ID of padding token. Defaults to 0. Nothing in
        this model reads it during the forward pass -- it is stored and
        serialized for tokenizer/config round-tripping only. No attention
        mask is derived from it: a call without an ``attention_mask``
        attends to padding tokens like real tokens, so supply
        ``attention_mask`` explicitly. Measured at ``hidden_size=32,
        num_layers=2``: two models differing only in ``pad_token_id``
        give ``max|delta| = 0.0``, while a real mask moves the output by
        0.0216. See ``decisions.md`` D-007 (plan-2026-08-23T203721-009b7ccf).
    :type pad_token_id: int
    :param position_embedding_type: ``'learned'`` (default) or
        ``'sinusoidal'``; forwarded to :class:`BertEmbeddings`, which raises on
        anything else. ``'absolute'`` is accepted as the legacy spelling of
        ``'learned'`` and normalized to it.
    :type position_embedding_type: str
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
    :param include_top: Whether the model owns a task head. Defaults to False,
        which builds the bare encoder: no head layer and no head weight exist.
    :type include_top: bool
    :param task_config: The task the head serves, as an ``NLPTaskConfig`` or a
        dict of its fields. Required when ``include_top`` is True, refused
        otherwise. Stored, and serialized, as a plain dict.
        ``MULTIPLE_CHOICE`` is refused: the head would be fed a token
        sequence, not one row per choice.
    :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
    :param head_config_overrides: Extra keyword arguments for the head
        constructor (for example ``{"pooling_type": "mean"}``). Only valid
        with ``include_top=True``. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :ivar embeddings: The embedding layer instance.
    :vartype embeddings: dl_techniques.layers.embedding.bert_embeddings.BertEmbeddings
    :ivar encoder_layers: A list of `TransformerLayer` instances.
    :vartype encoder_layers: list[dl_techniques.layers.transformer.TransformerLayer]
    :ivar task_head: The task head named ``"task_head"``, or None when
        ``include_top`` is False.
    :vartype task_head: Optional[dl_techniques.layers.heads.nlp.factory.BaseNLPHead]

    :raises ValueError: If invalid configuration parameters are provided,
        including ``include_top=True`` without a ``task_config``, and a
        ``task_config`` or ``head_config_overrides`` given with
        ``include_top=False``.

    Example:
        .. code-block:: python

            # Create standard BERT-base model
            model = BERT.from_variant("base")

            # Load from local file (`pretrained=True` raises NotImplementedError)
            model = BERT.from_variant("large", pretrained="path/to/weights.keras")

            # Use the model
            inputs = {
                "input_ids": keras.random.randint((2, 128), 0, 30522, dtype="int32"),
                "attention_mask": keras.ops.ones((2, 128), dtype="int32")
            }
            outputs = model(inputs)
            outputs["last_hidden_state"].shape
            # (2, 128, 768)

            # Same encoder, plus a sentiment head: returns bare logits (2, 3)
            from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
            clf = BERT.from_variant(
                "base",
                include_top=True,
                task_config=NLPTaskConfig(
                    name="sentiment",
                    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
                    num_classes=3,
                ),
            )
            logits = clf(inputs)

    """

    # Model variant configurations following BERT paper specifications
    MODEL_VARIANTS = {
        "large": {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "intermediate_size": 4096,
            "description": "BERT-Large: maximum performance"
        },
        "base": {
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_size": 3072,
            "description": "BERT-Base: suitable for most applications"
        },
        "small": {
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8,
            "intermediate_size": 2048,
            "description": "BERT-Small: Lightweight variant for resource-constrained environments"
        },
        # DECISION plan-2026-08-22T035419-a11304c8/D-110: intermediate_size stays 1024;
        # the public checkpoint this row mirrors uses 1024, not the 4x-hidden 512. See decisions.md.
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
    # DECISION plan-2026-08-23T091307-9a110062/D-500: stays "gelu_tanh", not "gelu";
    # the two differ by max|erf - tanh| = 4.732e-04 per call. See decisions.md.
    DEFAULT_HIDDEN_ACT = "gelu_tanh"
    DEFAULT_PAD_TOKEN_ID = 0
    _SEQUENCE_LENGTH_MESSAGE = (
        "seq_len={seq_len} exceeds max_position_embeddings={limit}. BERT's "
        "position embeddings are a learned table of {limit} rows, so there is "
        "no embedding for position {limit}. Truncate the input to {limit} "
        "tokens, or build the model with max_position_embeddings>={seq_len}."
    )

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = DEFAULT_HIDDEN_ACT,
        hidden_dropout_rate: float = 0.1,
        attention_probs_dropout_rate: float = 0.1,
        max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS,
        type_vocab_size: int = DEFAULT_TYPE_VOCAB_SIZE,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        position_embedding_type: str = "learned",
        normalization_type: NormalizationType = "layer_norm",
        normalization_position: NormalizationPositionType = "post",
        attention_type: AttentionType = "multi_head",
        ffn_type: FFNType = "mlp",
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        include_top: bool = False,
        task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Initializes the BERT model instance.

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
        :param hidden_dropout_rate: Dropout probability for embeddings/encoder.
        :type hidden_dropout_rate: float
        :param attention_probs_dropout_rate: Dropout for attention scores.
        :type attention_probs_dropout_rate: float
        :param max_position_embeddings: Maximum sequence length.
        :type max_position_embeddings: int
        :param type_vocab_size: Vocabulary size for token type IDs.
        :type type_vocab_size: int
        :param initializer_range: Stddev for weight initialization.
        :type initializer_range: float
        :param layer_norm_eps: Epsilon for normalization layers.
        :type layer_norm_eps: float
        :param pad_token_id: ID of the padding token. ADVISORY ONLY: it is
            stored and serialized but never read, and no attention mask is
            derived from it -- without an explicit ``attention_mask`` padding is
            fully attended to. See the class docstring for the full rule.
        :type pad_token_id: int
        :param position_embedding_type: ``'learned'`` or ``'sinusoidal'``;
            ``'absolute'`` is the legacy spelling of ``'learned'``.
        :type position_embedding_type: str
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
        :param include_top: Whether to build the task head.
        :type include_top: bool
        :param task_config: Task of the head; required iff ``include_top``.
        :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
        :param head_config_overrides: Head constructor overrides; only valid
            with ``include_top``.
        :type head_config_overrides: Optional[Dict[str, Any]]
        :param kwargs: Additional keyword arguments for the `keras.Model`.
        """
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_heads,
            hidden_dropout_rate, attention_probs_dropout_rate
        )
        validate_task_head_knobs(include_top, task_config, head_config_overrides)

        # Store all configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_probs_dropout_rate = attention_probs_dropout_rate
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        # DECISION plan-2026-08-23T203721-009b7ccf/D-007: pad_token_id stays unread;
        # deriving a mask from it moves every mask-less forward pass by max|delta|=0.0216. See decisions.md.
        self.pad_token_id = pad_token_id
        # DECISION plan-2026-08-17T183311-79c63e38/D-015: normalize 'absolute' to 'learned' here, once;
        # BertEmbeddings only accepts ('learned', 'sinusoidal') and would raise otherwise. See decisions.md.
        if position_embedding_type == "absolute":
            position_embedding_type = "learned"
        self.position_embedding_type = position_embedding_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.include_top = include_top
        self.task_config = (
            task_config_to_dict(task_config, vocab_size=vocab_size)
            if task_config is not None else None
        )
        self.head_config_overrides = (
            dict(head_config_overrides) if head_config_overrides is not None else None
        )

        # Build the model architecture
        self._build_architecture()

        # Created after every encoder layer and under its own fixed name, so
        # the encoder's layer names, weight paths and RNG draws are the same
        # with or without a head.
        self.task_head = None
        if self.include_top:
            self.task_head = build_model_task_head(
                self.task_config,
                input_dim=self.hidden_size,
                head_config_overrides=self.head_config_overrides,
                vocab_size=self.vocab_size,
            )

        logger.info(
            f"Created BERT foundation model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, heads={self.num_heads}"
            + (f", task head '{self.task_config['name']}'" if self.include_top else "")
        )

    def _validate_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dropout_rate: float,
        attention_probs_dropout_rate: float
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
        :param hidden_dropout_rate: Dropout probability for hidden layers.
        :type hidden_dropout_rate: float
        :param attention_probs_dropout_rate: Dropout for attention scores.
        :type attention_probs_dropout_rate: float
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
        if not (0.0 <= hidden_dropout_rate <= 1.0):
            raise ValueError(
                f"hidden_dropout_rate must be between 0 and 1, "
                f"got {hidden_dropout_rate}"
            )
        if not (0.0 <= attention_probs_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_probs_dropout_rate must be between 0 and 1, "
                f"got {attention_probs_dropout_rate}"
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
            dropout_rate=self.hidden_dropout_rate,
            normalization_type=self.normalization_type,
            position_embedding_type=self.position_embedding_type,
            name="embeddings"
        )

        # DECISION plan-2026-08-19T070627-a616f581/D-007: pass layer_norm_eps to both norm
        # args dicts; without it the block norms silently ran at the factory's 1e-6 default. See decisions.md.
        _norm_args = {'epsilon': self.layer_norm_eps}

        self.encoder_layers: List[TransformerLayer] = []
        for i in range(self.num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                normalization_type=self.normalization_type,
                attention_norm_args=dict(_norm_args),
                ffn_norm_args=dict(_norm_args),
                normalization_position=self.normalization_position,
                attention_type=self.attention_type,
                ffn_type=self.ffn_type,
                dropout_rate=self.hidden_dropout_rate,
                attention_dropout_rate=self.attention_probs_dropout_rate,
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=self.stochastic_depth_rate,
                activation=resolve_activation(self.hidden_act),
                use_bias=True,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                bias_initializer="zeros",
                name=f"encoder_layer_{i}"
            )
            self.encoder_layers.append(transformer_layer)

    def build(self, input_shape: Any) -> None:
        """Materialise the embeddings, every encoder layer and, if any, the head.

        :param input_shape: Shape of ``input_ids`` — ``(batch, seq_len)`` — or a
            mapping/sequence of shapes whose ``input_ids`` entry has that shape.
        :type input_shape: Any
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-049: build() must stay; without it
        # Keras marks the model built with zero variables and weight tying silently breaks. See decisions.md.
        if self.built:
            return

        ids_shape = input_shape
        if isinstance(ids_shape, dict):
            ids_shape = ids_shape.get("input_ids", ids_shape)
        if (
            isinstance(ids_shape, (list, tuple))
            and ids_shape
            and isinstance(ids_shape[0], (list, tuple))
        ):
            ids_shape = ids_shape[0]
        ids_shape = tuple(ids_shape)

        if len(ids_shape) != 2:
            raise ValueError(
                "BERT.build expects the shape of `input_ids`, i.e. "
                f"(batch_size, seq_length); got {ids_shape}"
            )
        # Before any variable exists: variables created inside a step-function
        # trace that then fails are left unusable (measured).
        self._refuse_over_long_sequence(ids_shape[1])

        self.embeddings.build(ids_shape)

        hidden_shape = (ids_shape[0], ids_shape[1], self.hidden_size)
        for encoder_layer in self.encoder_layers:
            encoder_layer.build(hidden_shape)

        if self.include_top:
            self.task_head.build(self._head_input_shape(ids_shape))

        super().build(input_shape)

    def _head_input_shape(self, ids_shape: tuple) -> Dict[str, tuple]:
        """Shapes of the dict ``call`` feeds the head, from the ids shape.

        :param ids_shape: Shape of ``input_ids`` — ``(batch, seq_len)``.
        :type ids_shape: tuple
        :return: ``hidden_states`` and ``attention_mask`` shapes.
        :rtype: Dict[str, tuple]
        """
        return {
            "hidden_states": (ids_shape[0], ids_shape[1], self.hidden_size),
            "attention_mask": (ids_shape[0], ids_shape[1]),
        }

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Compute the output shapes of the returned tensors.

        Keras builds a symbolic call of this model from this method, never
        from ``call``, so it must report what ``call`` returns in both modes.

        :param input_shape: Shape of ``input_ids`` — ``(batch, seq_len)``.
        :type input_shape: Any
        :return: Headless, a mapping with ``last_hidden_state`` and
            ``attention_mask``. With a head, the head's output shapes after
            the same reduction ``call`` applies: one shape, or a mapping of
            two or more.
        :rtype: Any
        """
        ids_shape = input_shape
        if isinstance(ids_shape, dict):
            ids_shape = ids_shape.get("input_ids", ids_shape)
        if (
            isinstance(ids_shape, (list, tuple))
            and ids_shape
            and isinstance(ids_shape[0], (list, tuple))
        ):
            ids_shape = ids_shape[0]
        ids_shape = tuple(ids_shape)
        if self.include_top:
            # finalize_head_outputs only drops and unwraps dict entries, so it
            # applies to a dict of shapes exactly as to a dict of tensors.
            return finalize_head_outputs(
                self.task_head.compute_output_shape(
                    self._head_input_shape(ids_shape)
                )
            )
        return {
            "last_hidden_state": (ids_shape[0], ids_shape[1], self.hidden_size),
            "attention_mask": (ids_shape[0], ids_shape[1]),
        }

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass of the BERT foundation model.

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
        :return: With ``include_top=False``, a dictionary with the keys:
                 - ``last_hidden_state``: The sequence of hidden states at the
                   output of the final layer. Shape:
                   `(batch, seq_len, hidden_size)`.
                 - ``attention_mask``: The original attention mask, passed
                   through for convenience in downstream models.
                 With ``include_top=True``, the head's output reduced by
                 ``finalize_head_outputs``: a bare tensor (logits, or
                 embeddings for a similarity head), or a dict of two or more
                 tensors (``start_logits``/``end_logits`` for QA).
        :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
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

        input_ids = self._validate_sequence_length(input_ids)

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

        # DECISION plan-2026-08-19T163559-499b6f0e/D-032: resolve the echoed mask only
        # here, at the return, never before the encoder loop. See decisions.md.
        resolved_mask = (
            attention_mask if attention_mask is not None
            else keras.ops.ones_like(input_ids)
        )

        if self.include_top:
            # Cast so a float16 hidden state never meets an int mask inside
            # the head's pooling (mixed precision).
            head_outputs = self.task_head(
                {
                    "hidden_states": hidden_states,
                    "attention_mask": keras.ops.cast(
                        resolved_mask, hidden_states.dtype
                    ),
                },
                training=training,
            )
            return finalize_head_outputs(head_outputs)

        return {
            "last_hidden_state": hidden_states,
            "attention_mask": resolved_mask,
        }

    def _validate_sequence_length(
        self, input_ids: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Reject a sequence longer than the learned position table.

        :param input_ids: The resolved token-id tensor, rank >= 2, whose LAST
            axis is the sequence.
        :type input_ids: keras.KerasTensor
        :return: ``input_ids`` itself, or, inside a TensorFlow graph traced
            with a dynamic sequence axis, ``input_ids`` gated on a runtime
            check of its length.
        :rtype: keras.KerasTensor
        :raises ValueError: if the sequence length is statically known and
            exceeds ``max_position_embeddings``.

        A ``predict``/``fit``/``evaluate`` that has seen two lengths retraces
        its step function with the sequence axis relaxed to ``None`` and then
        never re-enters Python for a new length, so a static check alone let a
        later over-long batch through. In such a graph the check runs at run
        time instead and raises ``InvalidArgumentError`` (a Python
        ``ValueError`` cannot cross a graph boundary) with the same message.
        XLA drops that assertion, and Keras's default ``jit_compile="auto"``
        selects XLA whenever a GPU is visible; there the XLA compile of the
        over-long length fails instead, with ``InvalidArgumentError`` naming
        the node ``seq_len_exceeds_max_position_embeddings_<max>``. A symbolic
        ``keras.Input`` with a dynamic axis has nothing to check and returns
        unchanged.
        """
        # DECISION plan-2026-08-23T203721-009b7ccf/D-023: without this check an
        # over-length sequence fails differently per device (finite garbage on
        # GPU, InvalidArgumentError on CPU). See decisions.md.
        return check_sequence_length(
            self,
            input_ids,
            self.max_position_embeddings,
            "max_position_embeddings",
            self._SEQUENCE_LENGTH_MESSAGE,
        )

    def _refuse_over_long_sequence(self, seq_len: Optional[int]) -> None:
        """Raise for a static ``seq_len`` above ``max_position_embeddings``.

        Called from ``build``, which parses the shape itself and does not trace
        ``call``, so it needs the refusal before any sub-layer is built.

        :param seq_len: The static sequence length, or None when unknown.
        :type seq_len: Optional[int]
        :raises ValueError: if ``seq_len`` exceeds ``max_position_embeddings``.
        """
        refuse_static_sequence_length(
            self,
            seq_len,
            self.max_position_embeddings,
            self._SEQUENCE_LENGTH_MESSAGE.format(
                seq_len=seq_len, limit=self.max_position_embeddings
            ),
        )

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True
    ) -> None:
        """Load pretrained ENCODER weights into the model.

        Only the encoder is transferred: a ``task_head`` in the checkpoint is
        never copied, and this model's own head (if any) keeps its fresh
        initialization. So a checkpoint of any BERT, headless or trained under
        any head, warm-starts a model with any other head or none. To restore
        a whole model, head included, use ``keras.models.load_model``.

        Every encoder layer must load, and every checkpoint layer must find
        its place, or this raises: a checkpoint of another hidden size, depth
        (deeper or shallower) or width is refused, never half-loaded or
        truncated. On a vocabulary change (the checkpoint's ``vocab_size``
        differs from this model's) only the variables of ``embeddings`` shaped
        by the vocabulary keep their fresh initialization; the position and
        token-type tables and the LayerNorm still load, or this raises.

        :param weights_path: Path to the weights file (.keras format).
        :type weights_path: str
        :param skip_mismatch: Allow the vocabulary exception above. ``False``
            refuses a vocabulary change too.
        :type skip_mismatch: bool
        :raises FileNotFoundError: If weights_path doesn't exist.
        :raises ValueError: If weights cannot be loaded, or any encoder layer
            outside the exception is left unloaded.

        Example:
            .. code-block:: python

                model = BERT.from_variant("base", vocab_size=50000)
                model.load_pretrained_weights(
                    "bert_base_uncased.keras",
                    skip_mismatch=True
                )
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            # build() materializes the whole tree that call() runs. The dummy
            # forward this replaced drew int32 ids from `keras.random.uniform`,
            # which Keras 3.8 refuses (float dtypes only), so every
            # `pretrained=<path>` load raised here before reaching the transfer.
            if not self.built:
                self.build((1, min(128, self.max_position_embeddings)))

            logger.info(f"Loading pretrained weights from {weights_path}")

            # Load weights with appropriate settings
            # Keras 3 removed `by_name` from `Model.load_weights` — the
            # signature is `(filepath, skip_mismatch=False, **kwargs)` and it
            # REJECTS the unknown keyword, so this call raised
            # `ValueError: Invalid keyword arguments: {'by_name': True}` for
            # every caller. It went unnoticed because the only route here was
            # `pretrained=<path>` and the enclosing except turned the failure
            # into a warning that continued with random weights.
            # DECISION plan-2026-10-11T004932-b83935e1/D-007: skip the checkpoint's task_head; do NOT
            # pass skip_prefixes=() -- a head of another task raises on shape, one of equal shape is copied silently. See decisions.md.
            # DECISION plan-2026-10-11T004932-b83935e1/D-020: complete-or-raise; do NOT go back to
            # strict=not skip_mismatch -- a backbone that does not fit then loads nothing and returns normally. See decisions.md.
            load_backbone_from_checkpoint(
                self,
                weights_path,
                head_prefixes=(TASK_HEAD_NAME,),
                vocab_layers=("embeddings",),
                tables=("word_embeddings/embeddings",),
                allow_vocab_change=skip_mismatch,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load weights from {weights_path}: {str(e)}"
            ) from e

    # No public BERT weights are distributed with dl_techniques; this raises rather
    # than falling back to a random-init model. Pass a local path via `pretrained="/path/to/file.keras"`.
    @staticmethod
    def _download_weights(
        variant: str,
        dataset: str = "uncased",
        cache_dir: Optional[str] = None
    ) -> str:
        """Pretrained-weights download stub. Always raises ``NotImplementedError``.

        To load weights, pass ``pretrained="/path/to/checkpoint.keras"`` to
        :meth:`from_variant`. To get a random-init model, omit ``pretrained``.

        :param variant: Model variant name (unused).
        :type variant: str
        :param dataset: Dataset/version identifier (unused).
        :type dataset: str
        :param cache_dir: Cache directory (unused).
        :type cache_dir: Optional[str]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No public pretrained BERT weights are distributed with "
            f"dl_techniques (requested variant '{variant}', dataset "
            f"'{dataset}'). Pass a local checkpoint instead: "
            f"BERT.from_variant('{variant}', pretrained='/path/to/weights.keras')."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "uncased",
        cache_dir: Optional[str] = None,
        **kwargs: Any
    ) -> "BERT":
        """Create a BERT model from a predefined variant.

        :param variant: The name of the variant, one of "base", "large",
            "small", "tiny".
        :type variant: str
        :param pretrained: If a string, a path to a local ``.keras`` weights
            file. If True, raises ``NotImplementedError`` — no public BERT
            weights ship with ``dl_techniques``. If False (default), the model
            is randomly initialized.
        :type pretrained: Union[bool, str]
        :param weights_dataset: unreachable, exactly like ``cache_dir``. It is
            forwarded only to ``_download_weights``, which raises
            ``NotImplementedError`` before reading it, and ``pretrained=True``
            is the only route there. Kept for signature stability with the
            house ``from_variant`` shape; no value of it changes anything.
        :type weights_dataset: str
        :param cache_dir: unreachable for the same reason; the
            ``_download_weights`` docstring already marks it ``(unused)``.
        :type cache_dir: Optional[str]
        :param kwargs: Additional arguments to override the variant's defaults,
            including ``include_top``, ``task_config`` and
            ``head_config_overrides``. A local ``pretrained`` path restores
            the encoder by layer name, every layer or raise (see
            :meth:`load_pretrained_weights`); a head absent from the
            checkpoint keeps its fresh initialization.
        :type kwargs: Any
        :return: A BERT model instance configured for the specified variant.
        :rtype: BERT
        :raises ValueError: If the specified variant is not recognized.
        :raises NotImplementedError: If ``pretrained`` is True.

        Example:
            .. code-block:: python

                # Random init
                model = BERT.from_variant("base")

                # Load from local file
                model = BERT.from_variant("base", pretrained="path/to/weights.keras")

                # A checkpoint of another vocab size: the word table stays fresh
                model = BERT.from_variant(
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

        logger.info(f"Creating BERT-{variant.upper()} model")
        logger.info(f"Configuration: {description}")

        load_weights_path = None

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                # DECISION plan_2026-05-11_9357982a/D-001: keep this except clause narrow;
                # a bare Exception once swallowed NotImplementedError and returned an untrained model. See decisions.md.
                try:
                    load_weights_path = cls._download_weights(
                        variant=variant,
                        dataset=weights_dataset,
                        cache_dir=cache_dir
                    )
                except (IOError, OSError, ValueError) as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {str(e)}. "
                        f"Continuing with random initialization."
                    )
                    load_weights_path = None

        config.update(kwargs)
        model = cls(**config)

        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path, skip_mismatch=True
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
            "hidden_dropout_rate": self.hidden_dropout_rate,
            "attention_probs_dropout_rate": self.attention_probs_dropout_rate,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "position_embedding_type": self.position_embedding_type,
            "normalization_type": self.normalization_type,
            "normalization_position": self.normalization_position,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "include_top": self.include_top,
            "task_config": (
                dict(self.task_config) if self.task_config is not None else None
            ),
            "head_config_overrides": (
                dict(self.head_config_overrides)
                if self.head_config_overrides is not None else None
            ),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BERT":
        """Create a model instance from its configuration.

        :param config: A dictionary containing the model's configuration.
        :type config: Dict[str, Any]
        :return: A new BERT model instance.
        :rtype: BERT
        """
        # `use_cache` was a serialized constructor argument that reached nothing
        # (D-015). It is dropped rather than refused because `bert/` is the most
        # reachable package in the tree and this method is `cls(**config)` --
        # without the pop, every `.keras` file written before 2026-08-18 would
        # fail to load with an unexpected-keyword `TypeError`.
        config = dict(config)
        config.pop("use_cache", None)
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the model summary with additional BERT-specific information.

        :param kwargs: Additional arguments passed to `keras.Model.summary`.
        """
        super().summary(**kwargs)
        logger.info("BERT Foundation Model Configuration:")
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
# Module-level Factory
# ---------------------------------------------------------------------


def create_bert(
    variant: str = "base",
    vocab_size: Optional[int] = None,
    pretrained: Union[bool, str] = False,
    weights_dataset: str = "uncased",
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> "BERT":
    """Convenience factory that mirrors ``create_resnet`` / ``create_tree_transformer``.

    Thin wrapper around :meth:`BERT.from_variant` exposing the most common
    construction arguments at module level. Behaves identically to calling
    ``BERT.from_variant(...)`` directly.

    :param variant: BERT variant name (``"tiny"``, ``"small"``, ``"base"``,
        ``"large"``). Defaults to ``"base"``.
    :type variant: str
    :param vocab_size: Optional vocabulary size override. If ``None`` (default),
        the variant's default vocab size is used. If provided, forwarded as
        ``vocab_size=...`` in ``kwargs``.
    :type vocab_size: Optional[int]
    :param pretrained: If ``True``, attempts to load pretrained weights — note
        that no public BERT weights are distributed by this library, so
        ``True`` will raise ``NotImplementedError``. If a string path, loads
        local weights from that path. If ``False`` (default), random init.
    :type pretrained: Union[bool, str]
    :param weights_dataset: unreachable. Forwarded to ``BERT.from_variant``,
        which forwards it to ``_download_weights``, which raises
        ``NotImplementedError`` before reading it -- and ``pretrained=True`` is
        the only route there. No value of this argument does anything. Kept for
        signature stability. Defaults to ``"uncased"``.
    :type weights_dataset: str
    :param cache_dir: unreachable for the same reason. Defaults to ``None``.
    :type cache_dir: Optional[str]
    :param kwargs: Additional keyword arguments forwarded to
        :meth:`BERT.from_variant` (e.g. ``max_position_embeddings``, or
        ``include_top=True`` with a ``task_config`` for a model with a head).
    :type kwargs: Any

    :returns: Configured ``BERT`` instance.
    :rtype: BERT

    :raises NotImplementedError: If ``pretrained=True`` (no public weights).
    :raises FileNotFoundError: If ``pretrained`` is a string path that does
        not exist.
    :raises ValueError: If ``variant`` is not a recognized BERT variant, or
        the head knobs are inconsistent (see :class:`BERT`).

    Example:
        >>> bert = create_bert("base")
        >>> bert = create_bert("tiny", vocab_size=200)
        >>> ner = create_bert(
        ...     "tiny", include_top=True,
        ...     task_config=NLPTaskConfig(
        ...         name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
        ...         num_classes=9))
    """
    if vocab_size is not None:
        kwargs["vocab_size"] = vocab_size
    return BERT.from_variant(
        variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------

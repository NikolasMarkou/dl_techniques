"""
BERT implements a bidirectional transformer encoder as a pure foundation model, in the
`BERT` class and its `create_bert` / `create_bert_with_head` factories. Masked language
modeling lets every layer attend over the whole input sequence, unlike an autoregressive
model that must hide future tokens from each position. The class stays a pure encoder: it
returns `{"last_hidden_state", "attention_mask"}` with no pooler or head, so several task
heads can share one set of weights during fine-tuning. Attention type, FFN type,
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
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.transformers import (
    FFNType,
    AttentionType,
    TransformerLayer,
    NormalizationType,
    NormalizationPositionType,
)
from dl_techniques.layers.activations import resolve_activation
from dl_techniques.layers.embedding.bert_embeddings import BertEmbeddings
from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskConfig, NLPTaskType
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.bert.model")
class BERT(keras.Model):
    """BERT (Bidirectional Encoder Representations from Transformers) model.

    This is a pure encoder implementation with pretrained weights support,
    designed to produce contextual token representations. It separates the
    core transformer architecture from any task-specific layers (like pooling
    or classification heads), making it highly flexible for pre-training,
    fine-tuning, and multi-task learning.

    The model expects inputs as a dictionary containing 'input_ids', and
    optionally 'attention_mask', 'token_type_ids', and 'position_ids'. It
    outputs a dictionary containing the 'last_hidden_state' and the forwarded
    'attention_mask'.

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

    Head integration (create_bert_with_head):

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  keras.Input × 3 (all required)      │
        │    input_ids / attention_mask /      │
        │    token_type_ids                    │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  BERT encoder (BERT.from_variant)    │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  NLP task head (create_nlp_head)     │
        │    {hidden_states, attention_mask}   │
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
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :ivar embeddings: The embedding layer instance.
    :vartype embeddings: dl_techniques.layers.embedding.bert_embeddings.BertEmbeddings
    :ivar encoder_layers: A list of `TransformerLayer` instances.
    :vartype encoder_layers: list[dl_techniques.layers.transformer.TransformerLayer]

    :raises ValueError: If invalid configuration parameters are provided.

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
            print(outputs["last_hidden_state"].shape)
            # (2, 128, 768)

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
        :param kwargs: Additional keyword arguments for the `keras.Model`.
        """
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_heads,
            hidden_dropout_rate, attention_probs_dropout_rate
        )

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
        """Materialise the embeddings and every encoder layer.

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

        self.embeddings.build(ids_shape)

        hidden_shape = (ids_shape[0], ids_shape[1], self.hidden_size)
        for encoder_layer in self.encoder_layers:
            encoder_layer.build(hidden_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape: Any) -> Dict[str, Any]:
        """Compute the output shapes of the two returned tensors.

        :param input_shape: Shape of ``input_ids`` — ``(batch, seq_len)``.
        :type input_shape: Any
        :return: Mapping with ``last_hidden_state`` and ``attention_mask``.
        :rtype: Dict[str, Any]
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
    ) -> Dict[str, keras.KerasTensor]:
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

        self._validate_sequence_length(input_ids)

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
        return {
            "last_hidden_state": hidden_states,
            "attention_mask": (
                attention_mask if attention_mask is not None
                else keras.ops.ones_like(input_ids)
            )
        }

    def _validate_sequence_length(
        self, input_ids: keras.KerasTensor
    ) -> None:
        """Reject a sequence longer than the learned position table.

        :param input_ids: The resolved token-id tensor, rank >= 2, whose LAST
            axis is the sequence.
        :type input_ids: keras.KerasTensor
        :raises ValueError: if the sequence length is statically known and
            exceeds ``max_position_embeddings``.

        Returns silently when the length is not statically known (a symbolic
        trace over ``keras.Input(shape=(None,))``); there is nothing to check
        at that point and the concrete-shape call that follows will do it.
        """
        # DECISION plan-2026-08-23T203721-009b7ccf/D-023: without this check an
        # over-length sequence fails differently per device (finite garbage on
        # GPU, InvalidArgumentError on CPU). See decisions.md.
        shape = getattr(input_ids, "shape", None)
        if shape is None or len(shape) < 2:
            return
        try:
            seq_len = int(shape[-1])
        except TypeError:
            # Dynamic sequence axis (symbolic build). Nothing to check.
            return

        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_position_embeddings="
                f"{self.max_position_embeddings}. BERT's position embeddings "
                f"are a learned table of {self.max_position_embeddings} rows, "
                "so there is no embedding for position "
                f"{self.max_position_embeddings}. Truncate the input to "
                f"{self.max_position_embeddings} tokens, or build the model "
                f"with max_position_embeddings>={seq_len}."
            )

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True
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

                model = BERT.from_variant("base", vocab_size=50000)
                model.load_pretrained_weights(
                    "bert_base_uncased.keras",
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
        :param kwargs: Additional arguments to override the variant's defaults.
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

                # Create with custom vocab size (will skip embedding weights)
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
        skip_mismatch = False

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

            pretrained_vocab_size = cls.DEFAULT_VOCAB_SIZE
            custom_vocab_size = kwargs.get("vocab_size", config.get("vocab_size"))

            if custom_vocab_size and custom_vocab_size != pretrained_vocab_size:
                skip_mismatch = True
                logger.info(
                    f"vocab_size ({custom_vocab_size}) differs from pretrained "
                    f"({pretrained_vocab_size}). Will skip embedding layer weights."
                )

            pretrained_config_keys = ["hidden_size", "num_layers", "num_heads", "intermediate_size"]
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
        :meth:`BERT.from_variant` (e.g. ``dropout_rate``, ``max_position_embeddings``).
    :type kwargs: Any

    :returns: Configured ``BERT`` instance.
    :rtype: BERT

    :raises NotImplementedError: If ``pretrained=True`` (no public weights).
    :raises FileNotFoundError: If ``pretrained`` is a string path that does
        not exist.
    :raises ValueError: If ``variant`` is not a recognized BERT variant.

    Example:
        >>> bert = create_bert("base")
        >>> bert = create_bert("tiny", vocab_size=200)
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
# Integration with NLP Task Heads
# ---------------------------------------------------------------------


def create_bert_with_head(
    bert_variant: str,
    task_config: NLPTaskConfig,
    pretrained: Union[bool, str] = False,
    weights_dataset: str = "uncased",
    cache_dir: Optional[str] = None,
    bert_config_overrides: Optional[Dict[str, Any]] = None,
    head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Factory function to create a BERT model with a task-specific head.

    This function demonstrates the intended integration pattern:
    1. Instantiate a foundational `BERT` model (optionally pretrained).
    2. Instantiate a task-specific head from the `dl_techniques.layers.heads.nlp`
       factory.
    3. Combine them into a single, end-to-end `keras.Model`.

    :param bert_variant: The BERT variant to use (e.g., "base", "large").
    :type bert_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task.
    :type task_config: NLPTaskConfig
    :param pretrained: If a string, a path to a local ``.keras`` weights file.
        If True, raises ``NotImplementedError`` -- no public BERT weights ship
        with ``dl_techniques``. If False (default), random init.
    :type pretrained: Union[bool, str]
    :param weights_dataset: unreachable. See ``create_bert``: it reaches
        ``_download_weights``, which raises ``NotImplementedError`` before
        reading it. No value does anything.
    :type weights_dataset: str
    :param cache_dir: unreachable for the same reason.
    :type cache_dir: Optional[str]
    :param bert_config_overrides: Optional dictionary to override default BERT
        configuration for the chosen variant. Defaults to None.
    :type bert_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Optional dictionary to override default head
        configuration. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :raises ValueError: if ``task_config.task_type`` is
        ``NLPTaskType.MULTIPLE_CHOICE``. That head needs a
        ``(batch, num_choices, hidden)`` input this factory cannot produce; see
        the ``D-024`` anchor in the body for the actionable alternative.
    :return: A complete ``keras.Model`` ready for the specified task.

        All three inputs are required, by name: ``input_ids``,
        ``attention_mask`` and ``token_type_ids``. The functional wrapper
        declares one ``keras.Input`` per key, so a data dict missing one
        raises ``ValueError: Missing data for input "token_type_ids"``.
        For a single-segment task pass ``np.zeros_like(input_ids)``.

        This is stricter than the encoder it wraps: :class:`BERT` itself
        treats the other two inputs as optional, and so does upstream
        HuggingFace. The difference is not fixed here, since relaxing a
        public factory's input signature is a breaking change for a
        convenience gap; see ``decisions.md`` D-009
        (plan-2026-08-23T203721-009b7ccf). Call the encoder directly for
        the two-key form.

        The head's output dict is post-processed by two rules, in order.
        First, if it contains ``logits``, the derived duplicates
        ``probabilities`` (``softmax(logits)``) and ``predictions``
        (``argmax(logits, axis=-1)``, unusable as a loss target) are
        dropped. Second, any dict left with a single key is unwrapped to
        that bare tensor, including ``{"embeddings": ...}``, not only
        ``{"logits": ...}``.

        Of the 23 task types this factory builds, 21 give a bare tensor
        and 2 give a dict (``QUESTION_ANSWERING`` and ``SPAN_EXTRACTION``,
        which keep ``{"start_logits": ..., "end_logits": ...}`` since
        those are two independent tensors; ``MULTIPLE_CHOICE`` is refused
        with a ``ValueError``, see the D-024 anchor above). The three
        ``TextSimilarityHead`` tasks (``TEXT_SIMILARITY``,
        ``PARAPHRASE_DETECTION``, ``DUPLICATE_DETECTION``) return a bare
        embeddings tensor ``(None, embedding_dim)``, not a dict and not
        logits, since this factory always feeds the head a single
        sequence and the head's pair branch is unreachable from here.

        The bare-tensor form is what makes
        ``loss="sparse_categorical_crossentropy"`` and
        ``metrics=["accuracy"]`` compile, and what makes
        ``model.predict(x).shape`` read.
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
            ner_model = create_bert_with_head(
                bert_variant="base",
                task_config=ner_task,
                head_config_overrides={"use_task_attention": True}
            )
            ner_model.summary()
    """
    bert_config_overrides = bert_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(
        f"Creating BERT-{bert_variant} with a '{task_config.name}' head."
    )

    bert_encoder = BERT.from_variant(
        bert_variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **bert_config_overrides
    )

    # DECISION plan-2026-08-23T203721-009b7ccf/D-024: raise on MULTIPLE_CHOICE rather than
    # build a wrong graph; this factory can only feed the head a rank-3 token sequence, not choices. See decisions.md.
    if task_config.task_type == NLPTaskType.MULTIPLE_CHOICE:
        raise ValueError(
            "create_bert_with_head cannot build a MULTIPLE_CHOICE model. "
            "MultipleChoiceHead scores a (batch, num_choices, hidden) tensor, "
            "but this factory feeds the head the token sequence "
            "(batch, seq_len, hidden), which yields one score per TOKEN and "
            "ignores num_classes entirely. Build the encoder and the head "
            "separately: encode each choice with BERT, stack the pooled "
            "representations into (batch, num_choices, hidden), and call "
            "create_nlp_head(task_config=...) on that."
        )

    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=bert_encoder.hidden_size,
        **head_config_overrides,
    )

    inputs = {
        "input_ids": keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        ),
        "attention_mask": keras.Input(
            shape=(None,), dtype="int32", name="attention_mask"
        ),
        "token_type_ids": keras.Input(
            shape=(None,), dtype="int32", name="token_type_ids"
        ),
    }

    encoder_outputs = bert_encoder(inputs)

    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": encoder_outputs["attention_mask"],
    }
    task_outputs = task_head(head_inputs)

    # DECISION plan-2026-08-23T203721-009b7ccf/D-018: drop probabilities/predictions when
    # logits is present, then unwrap any remaining single-key dict; a dict output made this factory uncompilable. See decisions.md.
    derived_from_logits = ("probabilities", "predictions")

    model_outputs = task_outputs
    if isinstance(model_outputs, dict):
        if "logits" in model_outputs:
            model_outputs = {
                key: value
                for key, value in model_outputs.items()
                if key not in derived_from_logits
            }
        if len(model_outputs) == 1:
            model_outputs = next(iter(model_outputs.values()))

    model_name = f"bert_{bert_variant}_with_{task_config.name}_head"
    model = keras.Model(
        inputs=inputs,
        outputs=model_outputs,
        name=model_name
    )

    logger.info(
        f"Successfully created model with {model.count_params():,} parameters."
    )
    return model

# ---------------------------------------------------------------------
"""
BERT, a bidirectional transformer encoder, packaged as a pure foundation model.

This model embodies the principle of bidirectional pre-training: a token's
representation should be conditioned on context from both directions at once,
not on a left-to-right prefix. Autoregressive language models are forced into
one direction because the training objective would otherwise let a token see
itself. BERT escapes that constraint by changing the objective rather than the
architecture -- masked language modelling corrupts a fraction of the input and
asks the model to reconstruct it, so every layer may attend over the whole
sequence without leaking the answer. The result is a representation in which
each position carries evidence from its full surroundings, which is what makes
a single pre-trained encoder transferable to tagging, span extraction and
sentence classification alike.

The encoder here is deliberately just the encoder. It emits
`{"last_hidden_state", "attention_mask"}` and owns no pooler and no
classification head, so the same weights serve several heads simultaneously
during multi-task fine-tuning, and a head can be swapped without touching the
foundation. The forwarded `attention_mask` is part of the contract: downstream
heads need to know which positions are padding, and recomputing it from the
inputs at every call site is how mask bugs get introduced.

Architecturally the stack is the standard one -- embeddings (word + learned
absolute position + token type, then normalization and dropout) followed by
`num_layers` identical transformer blocks. What is not standard is that the
block internals are supplied by factories rather than hard-coded: attention
type, FFN type, normalization type and normalization position are all
constructor arguments routed through `TransformerLayer`. The default is the
published configuration (multi-head attention, an MLP feed-forward, post-layer
normalization), but a caller can substitute a modern variant without forking
the file. The embedding layer is likewise the shared
`layers.embedding.bert_embeddings.BertEmbeddings` rather than a private copy;
`distilbert/` and `modern_bert/` build on the same layer, so a change there is
felt by three packages.

Four preset variants span the usual capacity range, from BERT-Tiny (4 layers,
256 hidden) through BERT-Large (24 layers, 1024 hidden). Parameter counts are
not restated here; `README.md` section 7 holds the single measured table and
the one-liner that re-derives it.

No pretrained weights are distributed with this package. `pretrained=True`
raises `NotImplementedError` rather than warning and returning a randomly
initialized model, which is a deliberate choice: the previous behaviour held a
table of unreachable weight URLs and swallowed the download failure, making an
unavailable checkpoint silently indistinguishable from a successful load. Local
checkpoints are loaded by path, with mismatched embedding shapes skipped by
name when `vocab_size` or the architecture differs from the checkpoint's.

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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
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

    **Architecture Overview:**

    .. code-block:: text

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
        Defaults to ``"gelu_tanh"``, the **tanh approximation** used by the
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
        Applies to the embedding norm AND to every one of the ``2 * num_layers``
        in-block norms. **Numerics change (2026-08-19, decisions.md D-007):** the
        in-block norms previously ignored this knob and ran at the normalization
        factory's ``1e-6`` default. Weight shapes are unchanged -- existing
        ``.keras`` files still load -- but forward values move slightly
        (measured at ``hidden_size=64, num_layers=4``: max |delta| 1.9e-06,
        mean 4.0e-07, i.e. ~6e-07 relative).
    :type layer_norm_eps: float
    :param pad_token_id: ID of padding token. Defaults to 0.
        **ADVISORY ONLY -- nothing in this model reads it.** It is stored and
        serialized for tokenizer/config round-tripping and for nothing else. In
        particular NO attention mask is derived from it: if you call the model
        without an ``attention_mask``, padding tokens are attended to exactly
        like real tokens. Supply ``attention_mask`` yourself. MEASURED at
        ``hidden_size=32, num_layers=2`` on a batch padded with token id 5: two
        models differing only in ``pad_token_id`` (0 vs 5) give
        ``max|delta| = 0.0``, a mask-less call equals an all-ones-mask call
        exactly, and the real ``(ids != 5)`` mask moves the output by 0.0216 --
        i.e. masking works, it is just never inferred. See ``decisions.md``
        D-007 (plan-2026-08-23T203721-009b7ccf) and
        ``tests/test_models/test_bert/test_pad_token_id_is_advisory_only.py``.
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
        # DECISION plan-2026-08-22T035419-a11304c8/D-110
        # `intermediate_size` is 1024, NOT 512. 512 was a 2x-hidden entry in a
        # table whose other three rows are all 4x-hidden, and it disagreed with
        # the released checkpoint this row names. MEASURED 2026-08-22 by fetching
        # https://huggingface.co/google/bert_uncased_L-4_H-256_A-4/raw/main/config.json
        # (Turc et al. 2019, "Well-Read Students Learn Better" -- the release
        # that publishes every (L, H) combination): hidden_size=256,
        # num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024.
        # The sibling rows were fetched in the same pass and all AGREE at 4x
        # (base 768->3072, large 1024->4096, small L-6_H-512_A-8 -> 2048), so 512
        # was the single outlier, not a deliberate ladder. Do NOT "restore" 512
        # for parameter-budget reasons: this row exists to mirror a named public
        # checkpoint, and halving its FFN makes `from_variant("tiny")` unable to
        # accept those weights. Use `from_variant("tiny", intermediate_size=512)`
        # if a smaller FFN is wanted.
        # See decisions.md D-110.
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
    # DECISION plan-2026-08-23T091307-9a110062/D-500
    # The default is the TANH APPROXIMATION, not the bare string "gelu".
    # Keras' "gelu" resolves to keras.activations.gelu with approximate=False,
    # i.e. the exact/erf form, while the original release this class ports
    # computes 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))):
    #   https://github.com/google-research/bert/blob/master/modeling.py
    # Do NOT "simplify" this back to "gelu" -- that is inference-changing
    # (max|erf - tanh| = 4.732e-04 per call, compounding over 12-24 layers),
    # not a cosmetic rename. The sibling ports distilbert/modern_bert track
    # HuggingFace, whose configs DO specify the exact form, so this must stay
    # BERT-specific and must not be propagated to them. See decisions.md D-500.
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
        # DECISION plan-2026-08-23T203721-009b7ccf/D-007
        # pad_token_id is stored and serialized but DELIBERATELY never read.
        # Do NOT "fix" this by deriving `attention_mask = input_ids !=
        # pad_token_id` in call() when no mask is supplied. Three reasons, and
        # the first is measured: that mutation moves the output of every
        # mask-less forward pass that exists today by max|delta| = 0.0216 at a
        # 2-layer config -- silently, with no error and no shape change.
        # Second, upstream HF BERT does not do it either (it defaults the mask
        # to all-ones), so wiring it would be a divergence from the reference,
        # not a fidelity fix. Third, the natural home for such a derivation is
        # the shared `BertEmbeddings`, which `fnet`, `distilbert` and
        # `modern_bert` also use -- a HIGH-blast-radius edit for a knob nobody
        # has asked to work. The footgun is documented in the class docstring
        # instead, exactly as `distilbert` documents its own under D-003
        # (plan-2026-08-10T183739-b007f435), and pinned by
        # tests/test_models/test_bert/test_pad_token_id_is_advisory_only.py.
        # See decisions.md D-007 (plan-2026-08-23T203721-009b7ccf).
        self.pad_token_id = pad_token_id
        # DECISION plan-2026-08-17T183311-79c63e38/D-015
        # `'absolute'` was this model's default while the value was never
        # forwarded to `BertEmbeddings` -- which has no such value at all
        # (`VALID_POSITION_EMBEDDING_TYPES` is `('learned', 'sinusoidal')`), so
        # wiring the shipped default up naively would have RAISED. The legacy
        # spelling is normalized ONCE, here, so every stored config and every
        # `get_config()` carries the single live spelling.
        #
        # WHAT NOT TO DO: do not drop `position_embedding_type` from the
        # `BertEmbeddings(...)` call in `_build_architecture` "to keep the
        # default stable" -- that is the defect this closes, and it is the same
        # one FNet closed as D-071. Do not reinstate `use_cache`: BERT here is a
        # bidirectional encoder with no incremental-decoding path and no KV
        # cache in the stack, so it named a mechanism that does not exist.
        # See decisions.md D-015.
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

        # DECISION plan-2026-08-19T070627-a616f581/D-007
        # `layer_norm_eps` used to reach ONLY `BertEmbeddings` (above). The
        # `TransformerLayer` loop below passed neither `attention_norm_args` nor
        # `ffn_norm_args`, so `TransformerLayer._create_normalization_layer`
        # fell through to `create_normalization_layer`'s own `epsilon=1e-6`
        # default and all `2 * num_layers` encoder norms ran at 1e-6 while the
        # embedding norm ran at BERT's own 1e-12 -- a 1e6x split INSIDE one
        # model. MEASURED pre-fix at `num_layers=2`: 4 of 4 block norms at
        # 1e-06, embeddings at 1e-12.
        # WHAT NOT TO DO: do not "fix" this by changing the factory's 1e-6
        # default -- that factory is shared by every transformer in the repo.
        # These two dicts are the per-site channel it already provides. Do not
        # write 1e-12 as a literal here either; `self.layer_norm_eps` is the
        # model's knob and a test asserts the block norms TRACK it.
        # See decisions.md D-007.
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
        # DECISION plan-2026-08-19T163559-499b6f0e/D-049
        # `BERT` had NO `build()`. Keras therefore marked it built with ZERO
        # variables, and every consumer that inspects a backbone's variables
        # BEFORE calling it saw an empty model. That is not a cosmetic contract
        # gap: `CausalLanguageModel.build` locates the embedding matrix by shape
        # over `backbone.variables`, so weight tying — ON BY DEFAULT — silently
        # fell back to an untied `Dense`, AND the save/load halves took different
        # branches, which is why the `.keras` round trip raised
        # "expected 2 variables, but received 0".
        # Do NOT replace this with a dummy forward pass: a forward pass under a
        # `StatelessScope` (which is where Keras rebuilds a model on load) does
        # not persist the variables it creates. See decisions.md D-049.
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

        # DECISION plan-2026-08-19T163559-499b6f0e/D-032
        # The echoed mask is RESOLVED HERE, at the return, and nowhere else.
        # This is the fourth and last member of the F-87 family; `fnet` and
        # `modern_bert` were repaired under D-031 and `distilbert` under D-062,
        # and `bert` -- the family's namesake -- was MISSED by both. Echoing a
        # possibly-`None` `attention_mask` makes the output STRUCTURE depend on
        # the INPUT, so `BERT.predict({"input_ids": ids})` raised
        # `ValueError: Structures don't have the same nested structure` (Keras
        # concatenates per-batch outputs and a `None` slot has no structure).
        # `model(inputs)` always worked, which is why no test caught it.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT drop the "attention_mask" key when it is None -- that is
        #     the same input-dependent output structure wearing different
        #     clothes, and downstream heads read the key unconditionally.
        #   * Do NOT resolve the mask BEFORE the encoder loop. For BERT it is
        #     an exact no-op (MEASURED max|delta| = 0.000000e+00 over the whole
        #     output, 2 layers, seq 12, against max|out| = 3.027470e+00), but
        #     the SAME edit in `modern_bert/model.py` measured 6.415714e-01,
        #     because `WindowAttention._call_grid` zero-pads a rank-2 mask up to
        #     its synthetic grid. All four siblings resolve at the RETURN only,
        #     so no shipped checkpoint's numerics move and the rule is uniform
        #     rather than per-model. See decisions.md D-032.
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
        # DECISION plan-2026-08-23T203721-009b7ccf/D-023
        # WITHOUT this check the model was silently wrong, and DIFFERENTLY
        # wrong per device. MEASURED at `max_position_embeddings=16`, seq_len
        # 17: on GPU the forward returned a finite result with no warning (an
        # out-of-range position gather reads zeros), while on CPU the same call
        # raised `InvalidArgumentError` from `Embedding.call()`. A silent wrong
        # answer on the device this repo trains on is the worst of the three
        # outcomes, so it is converted into a loud one here.
        #
        # This is the RIGHT home for it, which was re-derived rather than
        # assumed. `BERT.call` is on EVERY entry path: MEASURED, Keras 3
        # executes a Functional graph by calling each operation's `call()` with
        # real tensors, so `create_bert_with_head(...).predict(x)` reaches this
        # line with a fully static `TensorShape([2, 17])` even though the graph
        # was BUILT over `keras.Input(shape=(None,))`. Do NOT move this into
        # `create_bert_with_head`: the factory only ever sees the symbolic
        # `None` sequence axis and could not check anything, and the bare
        # `BERT(...)` path would lose the guard entirely.
        #
        # Do NOT "fix" the underlying gather by clamping or wrapping the
        # position ids -- that would keep the wrong answer and hide it.
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

    # `_download_weights` raises instead of falling back to random init. The
    # previous version held a `PRETRAINED_WEIGHTS` table of placeholder URLs
    # pointing at a non-existent host; `from_variant` caught the download
    # failure, logged a warning and returned a randomly-initialized model, so
    # `pretrained=True` silently produced untrained weights. Do NOT reinstate a
    # warn-and-return branch here or in
    # `from_variant`: the except clause there is deliberately narrow (see the
    # D-001 anchor) so this NotImplementedError propagates to the caller. No
    # public BERT weights are distributed with dl_techniques; pass a local path
    # via `pretrained="/path/to/file.keras"` or use `pretrained=False` (default).
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
        :param weights_dataset: UNREACHABLE, exactly like ``cache_dir``. It is
            forwarded only to ``_download_weights``, which raises
            ``NotImplementedError`` before reading it, and ``pretrained=True``
            is the only route there. The docstring said "Only used if
            pretrained=True", which is true and misleading: that path never
            returns. Kept for signature stability with the house
            ``from_variant`` shape, not because any value of it does anything.
        :type weights_dataset: str
        :param cache_dir: UNREACHABLE for the same reason; the
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
                # DECISION plan_2026-05-11_9357982a/D-001
                # Do NOT broaden this except clause. Catching `Exception` here
                # was an I-01 bug: it silently swallowed `NotImplementedError`
                # from `_download_weights` and returned a random-init model
                # masquerading as pretrained. Only catch concrete I/O errors
                # that legitimately indicate a missing/corrupt local mirror.
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
    :param weights_dataset: UNREACHABLE. Forwarded to ``BERT.from_variant``,
        which forwards it to ``_download_weights``, which raises
        ``NotImplementedError`` before reading it -- and ``pretrained=True`` is
        the only route there. No value of this argument does anything. Kept for
        signature stability. Defaults to ``"uncased"``.
    :type weights_dataset: str
    :param cache_dir: UNREACHABLE for the same reason. Defaults to ``None``.
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
    :param weights_dataset: UNREACHABLE. See ``create_bert``: it reaches
        ``_download_weights``, which raises ``NotImplementedError`` before
        reading it. No value does anything.
    :type weights_dataset: str
    :param cache_dir: UNREACHABLE for the same reason.
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
    :return: A complete `keras.Model` ready for the specified task.

        **All THREE inputs are required**, by name:
        ``input_ids``, ``attention_mask`` and ``token_type_ids``. The
        Functional wrapper declares one ``keras.Input`` per key, and Keras
        matches a data dict against that declaration exactly. MEASURED --
        ``predict({"input_ids": ..., "attention_mask": ...})`` raises
        ``ValueError: Missing data for input "token_type_ids". You passed a
        data dictionary with keys ['input_ids', 'attention_mask']. Expected the
        following keys: ['attention_mask', 'input_ids', 'token_type_ids']``.
        For a single-segment task pass ``np.zeros_like(input_ids)``.

        This is STRICTER than the encoder it wraps, deliberately.
        :class:`BERT` itself treats both of the other two as optional --
        MEASURED, ``BERT.from_variant("tiny")({"input_ids": ids})`` forwards
        fine and returns ``['last_hidden_state', 'attention_mask']`` -- and
        upstream HuggingFace treats ``token_type_ids`` as optional too. The
        difference is not fixed here because relaxing a public factory's input
        signature is a breaking change for a convenience gap; see
        ``decisions.md`` D-009 (plan-2026-08-23T203721-009b7ccf). Call the
        encoder directly when you want the two-key form.

        **Output contract.** The head's dict is post-processed by exactly two
        rules, in order. (1) If it contains `logits`, the derived duplicates
        `probabilities` (`softmax(logits)`) and `predictions`
        (`argmax(logits, axis=-1)`, int32, unusable as a loss target) are
        dropped. (2) **ANY** dict that is left with a single key is then
        unwrapped to that bare tensor -- this includes `{"embeddings": ...}`,
        not only `{"logits": ...}`.

        So of the 23 task types this factory BUILDS, **21 give a bare tensor
        and 2 give a dict**. (24 task types are reachable through
        `get_head_class`; `MULTIPLE_CHOICE` is refused here with a
        `ValueError` -- see the D-024 anchor above.)
        The two are `QUESTION_ANSWERING` and `SPAN_EXTRACTION`, which
        keep `{"start_logits": ..., "end_logits": ...}` because those are
        genuinely independent tensors. Worth calling out explicitly, because
        rule (2) is easy to miss: the three `TextSimilarityHead` tasks
        (`TEXT_SIMILARITY`, `PARAPHRASE_DETECTION`, `DUPLICATE_DETECTION`)
        return a **bare embeddings tensor** `(None, embedding_dim)`, NOT a
        dict and NOT logits -- this factory always feeds the head a single
        sequence, so the head's pair branch (which would emit three tensors) is
        unreachable from here.

        The bare-tensor form is what makes
        `loss="sparse_categorical_crossentropy"` and `metrics=["accuracy"]`
        compile, and what makes `model.predict(x).shape` read.
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

    # DECISION plan-2026-08-23T203721-009b7ccf/D-024
    # MULTIPLE_CHOICE builds without error here and produces a SEMANTICALLY
    # WRONG graph, which is worse than a raise. `MultipleChoiceHead` expects
    # `(batch, num_choices, hidden)` and scores each CHOICE; this factory can
    # only ever feed it the token sequence `(batch, seq_len, hidden)`, so
    # MEASURED, `logits` comes out `(None, None)` -- one score per TOKEN -- and
    # `num_classes` is ignored entirely. Step 4's bare-tensor contract made
    # that output look MORE legitimate, not less. `get_head_class` in
    # `layers/heads/nlp/factory.py` already set the precedent for this subsystem
    # ("NO SILENT FALLBACK"); refusing here matches it.
    # Do NOT relax this to a warning: a warning in a factory is a silent wrong
    # answer with a log line. The head itself is NOT deprecated -- build it
    # directly with `create_nlp_head` and feed it a rank-3 choices tensor.
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

    # DECISION plan-2026-08-23T203721-009b7ccf/D-018
    # A head's `probabilities` entry is exactly `softmax(logits)` and its
    # `predictions` entry is exactly `argmax(logits, axis=-1)`, so exposing
    # either beside `logits` makes the model's output a dict for no information
    # gain -- and a dict output is what made this factory impossible to
    # compile: MEASURED,
    # `metrics=['accuracy']`, `metrics={'logits': ...}` and the dict+list forms
    # ALL raise, `loss='sparse_categorical_crossentropy'` raises `KeyError:
    # The path: ('logits',)`, and `model.predict(x).shape` raises
    # `AttributeError: 'dict' object has no attribute 'shape'`. Only
    # `loss={'logits': ...}` with NO metrics compiled, i.e. there was no way to
    # attach a metric at all. Do NOT "fix" this by editing
    # `layers/heads/nlp/factory.py` to stop emitting `probabilities`: that file
    # is HIGH blast radius (~20 model packages) and the heads' own dict
    # contract is used directly by callers that never go through this factory.
    # Do NOT collapse unconditionally either -- `QuestionAnsweringHead` emits
    # `{'start_logits', 'end_logits'}`, two genuinely independent tensors, and
    # that dict is KEPT. The two names below are the complete derived set,
    # RE-DERIVED by an AST walk over every `call()` in
    # `layers/heads/nlp/factory.py` rather than assumed: only
    # `TextClassificationHead` / `MultipleChoiceHead` (`probabilities`) and
    # `TokenClassificationHead` (`predictions`, an int32 argmax no loss can
    # consume) put a pure function of `logits` beside it.
    #
    # CORRECTED 2026-08-24 (step 13.2). This anchor used to say
    # "`TextSimilarityHead` emits three independent tensors; those keep their
    # dicts". That is the opposite of what the code does, and it named a branch
    # this factory CANNOT REACH. `TextSimilarityHead.call` has two branches: the
    # PAIR branch (`hidden_states_1`/`hidden_states_2`) returns
    # `{'similarity_score', 'embeddings_1', 'embeddings_2'}`, and the
    # SINGLE-SEQUENCE branch returns `{'embeddings': ...}`. `head_inputs` above
    # is always the single-sequence form, so the pair branch is unreachable from
    # here and TEXT_SIMILARITY / PARAPHRASE_DETECTION / DUPLICATE_DETECTION all
    # arrive at the `len(...) == 1` unwrap and come out as a BARE TENSOR
    # `(None, embedding_dim)` -- MEASURED. The rule this code actually
    # implements, and the one to keep, is: drop the derived duplicates when
    # `logits` is present, then unwrap ANY remaining single-key dict, INCLUDING
    # `{'embeddings'}`. That is the right rule -- a one-key dict carries exactly
    # one informative tensor, and leaving it wrapped would make the model
    # non-compilable with plain `metrics=[...]` for no gain. Restricting the
    # unwrap to logits-bearing dicts was considered and REJECTED for exactly
    # that reason. Every reachable NLPTaskType's resulting contract is pinned by
    # `tests/test_models/test_bert/test_every_head_family_output_contract.py`.
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
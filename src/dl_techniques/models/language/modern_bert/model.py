"""
ModernBERT, a BERT-shaped encoder rebuilt with the training recipe that came
out of the large-language-model era.

This model embodies the principle that an encoder's ceiling is usually set by
its optimization behaviour and its context budget, not by its parameter count.
Classic BERT was trained at 512 tokens with post-layer normalization and dense
global attention everywhere, and each of those three choices independently caps
what the architecture can do: post-normalization makes deep stacks fragile to
learning rate, quadratic global attention makes long sequences unaffordable,
and a 512-token window makes whole classes of retrieval task impossible.
ModernBERT changes all three while leaving the encoder's interface identical,
so it is a drop-in replacement rather than a new family.

Pre-layer normalization is the stability change. Normalizing the input to each
sublayer rather than the residual sum leaves the skip path an unmodified
identity from input to output, so gradient magnitude no longer depends on how
many normalization layers it has passed through. That is what removes the
learning-rate warmup sensitivity that makes deep post-normalization stacks
awkward to train.

Hybrid local/global attention is the paper's cost change, and it is where the
design does something non-obvious. Rather than making every layer cheap, the
stack alternates: most layers use a 1-D sliding band spanning
`local_attention_window_size` tokens (`local_attention_window_size // 2` either
side of each query), and every `global_attention_interval`-th layer uses full
global attention. In the paper the local layers do the bulk of the work at
linear cost, while the periodic global layers stitch the windows together so
information still crosses the whole sequence -- a token pair `L` apart is
connected after one global layer, not after `L / window` local ones.

By default (`include_top=False`) the class is a pure encoder returning
`{"last_hidden_state", "attention_mask"}`. With `include_top=True` and a
`task_config` it owns ONE task head from `layers/heads/nlp`, named `task_head`,
and returns that head's output instead, exactly as `models/language/bert` does;
every head weight lives under `task_head/`, so an encoder trained under one head
is reused, weights unchanged, under another. `create_modern_bert` is the
module-level factory.

References:
    - Warner et al., 2024. Smarter, Better, Faster, Longer: A Modern
      Bidirectional Encoder. (https://arxiv.org/abs/2412.13663)
    - Devlin et al., 2018. BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding.
      (https://arxiv.org/abs/1810.04805)
    - Xiong et al., 2020. On Layer Normalization in the Transformer
      Architecture. (https://arxiv.org/abs/2002.04745)
    - Shazeer, 2020. GLU Variants Improve Transformer.
      (https://arxiv.org/abs/2002.05202)
    - Beltagy et al., 2020. Longformer: The Long-Document Transformer.
      (https://arxiv.org/abs/2004.05150)
"""


import os
import keras
from typing import Optional, Union, Any, Dict, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.sequence_length_guard import check_sequence_length
from dl_techniques.utils.weight_transfer import (
    load_backbone_from_checkpoint,
)
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    build_model_task_head,
    finalize_head_outputs,
    task_config_to_dict,
    validate_task_head_knobs,
)
from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME
from dl_techniques.layers.embedding.modern_bert_embeddings import ModernBertEmbeddings
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.modern_bert.model")
class ModernBERT(keras.Model):
    """ModernBERT (A Modern Bidirectional Encoder) foundation model.

    This model refactors the original BERT architecture to include modern
    techniques such as Pre-Layer Normalization, GeGLU activations, and a
    hybrid attention mechanism combining 1-D banded local attention with
    periodic global attention. It is designed for high performance and
    configurability.

    The model expects a tensor of 'input_ids', or a dictionary containing
    'input_ids' and optionally 'attention_mask' and 'token_type_ids'.
    Headless (``include_top=False``, the default) it outputs a dictionary
    containing the 'last_hidden_state' and the forwarded 'attention_mask'.
    With ``include_top=True`` it also owns one NLP task head, fed
    ``{"hidden_states": last_hidden_state, "attention_mask": <mask cast to the
    hidden-state dtype>}``, and outputs what the head returns, reduced by
    :func:`~dl_techniques.layers.heads.nlp.finalize_head_outputs`: a bare
    tensor, or a dict for question answering. The knobs, their validation and
    the head-swap route are those of
    :class:`~dl_techniques.models.language.bert.model.BERT`.

    Positional signal: the embedding stage carries none — it sums word and
    token-type embeddings only. Global layers get theirs from RoPE, applied
    to queries and keys inside the attention layer (``group_query`` with
    ``num_kv_heads == num_heads``, plain multi-head attention plus RoPE).
    Local layers carry no positional term of their own: they use
    ``window_band`` attention, a 1-D symmetric sliding band in which query
    ``i`` attends key ``j`` iff ``abs(i - j) <= local_attention_window_size // 2``.
    This matches upstream's ``transformers/modular_modernbert.py``, which sets
    ``sliding_window = local_attention // 2``, so
    ``local_attention_window_size=128`` means 64 tokens either side, a
    128-token span. There is no grid folding, no square padding and no
    relative position bias; position reaches the local layers only through
    the residual stream from the global layers' RoPE.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input dict                          │
        │  input_ids       [B, L]    (required)│
        │  attention_mask  [B, L]    (optional)│
        │  token_type_ids  [B, L]    (optional)│
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  ModernBertEmbeddings                │
        │    Word ⊕ TokenType → Norm → Dropout │
        │    NO positional term here           │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  TransformerLayer₁  (Pre-LN)         │
        │    band attention → GeGLU FFN        │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  ...                                 │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  TransformerLayerₖ  (Pre-LN)         │
        │    global attention + RoPE → GeGLU   │
        │    every global_attention_interval-th│
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  ...  alternating local / global     │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  TransformerLayerₙ                   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  final_layer_norm (LayerNorm)        │
        │  center=use_bias                     │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output dict                         │
        │    "last_hidden_state"  [B, L, H]    │
        │    "attention_mask"     [B, L]       │
        │  Mask echoed; ones_like(input_ids)   │
        │  when no attention_mask is supplied  │
        └──────────────────────────────────────┘

    Variants:

    .. code-block:: text

        variant   hidden  layers  heads  interm.  interval  window
        tiny        256      4      4      384       2        64
        base        768     22     12     1152       3       128
        large      1024     28     16     2624       3       128

    base has 152.7M parameters, large has 399.6M, measured at the shipped
    interval of 3. `window` is the local band's full span in tokens; the
    layer receives half of it as the half-width. The hybrid schedule
    matches the published architecture rather than minimizing memory — see
    the trade-off note on `global_attention_interval` below.

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
    :param hidden_dropout_rate: Dropout probability for all fully connected
        layers in embeddings and encoder. Defaults to 0.1. Upstream this field is
        ``ModernBertConfig.hidden_dropout_prob``; it is spelled ``_rate`` here
        because every dropout rate in this repository is (D-130), and the value
        and meaning are unchanged.
    :type hidden_dropout_rate: float
    :param attention_probs_dropout_rate: Dropout ratio for attention
        probabilities. Defaults to 0.1. Upstream:
        ``ModernBertConfig.attention_probs_dropout_prob``.
    :type attention_probs_dropout_rate: float
    :param type_vocab_size: Vocabulary size for token type IDs.
        Defaults to 2.
    :type type_vocab_size: int
    :param initializer_range: Stddev of truncated normal initializer for
        all weight matrices. Defaults to 0.02.
    :type initializer_range: float
    :param layer_norm_eps: Epsilon for normalization layers. Defaults to 1e-12.
        Applies to the embedding norm, ``final_layer_norm`` and every one of the
        ``2 * num_layers`` in-block norms (decisions.md D-007). Weight shapes
        are unchanged, so existing ``.keras`` files still load.
    :type layer_norm_eps: float
    :param use_bias: Whether to use bias vectors in linear layers.
        Defaults to False.
    :type use_bias: bool
    :param global_attention_interval: Interval for inserting a global attention
        layer. E.g., 3 means every 3rd layer is global. Defaults to 3, and all
        three shipped presets use a hybrid schedule (``tiny`` 2, ``base``/``large`` 3);
        see decisions.md D-135 for the measurement that restored 3.

        The hybrid schedule matches the published architecture rather than
        minimizing memory, and past L ~ 2048 it costs slightly more than
        all-global: the local band is a dense N x N masked attention, the
        same O(N^2) order as global, plus the mask. Peak host RSS measured on
        CPU (``from_variant("base", global_attention_interval=i)``,
        constructed and forwarded once, 3 draws per cell, min-max range):

        =====  ==========================  ==========================
        L      interval=1 (all global)     interval=3 (hybrid)
        =====  ==========================  ==========================
        1024   1.820 - 1.830 GB            1.689 - 1.739 GB
        2048   2.323 - 2.529 GB            2.303 - 2.489 GB (a tie)
        4096   4.758 - 4.952 GB            5.122 - 5.135 GB
        =====  ==========================  ==========================

        The hybrid wins at L=1024, is indistinguishable at L=2048 (the
        ranges overlap), and loses by about 4% at L=4096 — below this
        model's advertised 8192-token context, so for most of that range the
        paper's schedule is the more expensive one. It is kept anyway to
        match the published architecture. Re-measure with::

            CUDA_VISIBLE_DEVICES='' .venv/bin/python -c "
            import sys, resource, numpy as np; sys.path.insert(0, 'src')
            from dl_techniques.models.language.modern_bert.model import ModernBERT
            i, L = int(sys.argv[1]), int(sys.argv[2])
            m = ModernBERT.from_variant('base', global_attention_interval=i)
            m({'input_ids': np.zeros((1, L), 'int32'),
               'attention_mask': np.ones((1, L), 'int32')}, training=False)
            print(i, L, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
            " <interval> <L>

    :type global_attention_interval: int
    :param local_attention_window_size: The local band's full span in
        tokens, matching upstream's ``local_attention``: a token attends
        ``local_attention_window_size // 2`` tokens either side. The
        ``window_band`` layer takes that half-width, so this value is halved
        on the way in. It is not a square spatial edge length. Defaults to 128.
    :type local_attention_window_size: int
    :param max_position_embeddings: Longest sequence the global layers' RoPE
        tables are precomputed for; a longer input raises. Defaults to 8192.
    :type max_position_embeddings: int
    :param global_rope_theta: RoPE base frequency for the global layers.
        Defaults to 160000.0.
    :type global_rope_theta: float
    :param include_top: Whether the model owns a task head. Defaults to False:
        no head layer and no head weight exist.
    :type include_top: bool
    :param task_config: The head's task, as an ``NLPTaskConfig`` or a dict of
        its fields. Required when ``include_top`` is True, refused otherwise;
        stored as a plain dict. ``MULTIPLE_CHOICE`` is refused.
    :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
    :param head_config_overrides: Extra keyword arguments for the head
        constructor (e.g. ``{"pooling_type": "mean"}``). Only valid with
        ``include_top=True``. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :ivar embeddings: The embedding layer instance.
    :vartype embeddings: dl_techniques.layers.embedding.modern_bert_embeddings.ModernBertEmbeddings
    :ivar encoder_layers: The ``num_layers`` ``TransformerLayer`` instances.
    :vartype encoder_layers: List[TransformerLayer]
    :ivar final_norm: LayerNormalization applied after the stack.
    :vartype final_norm: keras.layers.LayerNormalization
    :ivar task_head: The task head named ``"task_head"``, or None when
        ``include_top`` is False.
    :vartype task_head: Optional[dl_techniques.layers.heads.nlp.factory.BaseNLPHead]

    :raises ValueError: If invalid configuration parameters are provided,
        including ``include_top=True`` without a ``task_config``, and a
        ``task_config`` or ``head_config_overrides`` given with
        ``include_top=False``.

    Input shape:
        Mapping with ``input_ids`` of shape ``(batch_size, seq_len)``, plus the
        optional ``attention_mask`` and ``token_type_ids`` at the same shape.

    Output shape:
        Headless, a mapping with ``last_hidden_state`` at
        ``(batch_size, seq_len, hidden_size)`` and ``attention_mask`` at
        ``(batch_size, seq_len)``. With a head, the head's finalized output.

    Example:
        .. code-block:: python

            # Create standard ModernBERT-base model
            model = ModernBERT.from_variant("base")

            # Use the model
            inputs = {
                "input_ids": keras.random.randint((2, 256), 0, 50368, dtype="int32"),
                "attention_mask": keras.ops.ones((2, 256), dtype="int32")
            }
            outputs = model(inputs)
            outputs["last_hidden_state"].shape
            # (2, 256, 768)

            # Same encoder, plus a sentiment head: returns bare logits (2, 3)
            from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
            clf = ModernBERT.from_variant(
                "base",
                include_top=True,
                task_config=NLPTaskConfig(
                    name="sentiment",
                    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
                    num_classes=3,
                ),
            )
            logits = clf(inputs, training=False)
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
            "description": (
                "ModernBERT-Base: 152.7M parameters (measured), hybrid 1-D "
                "local band / global attention"
            ),
        },
        "large": {
            "hidden_size": 1024,
            "num_layers": 28,
            "num_heads": 16,
            "intermediate_size": 2624,
            "use_bias": False,
            "global_attention_interval": 3,
            "local_attention_window_size": 128,
            "description": (
                "ModernBERT-Large: 399.6M parameters (measured), hybrid 1-D "
                "local band / global attention"
            ),
        },
    }

    # Default architecture constants
    DEFAULT_VOCAB_SIZE = 50368
    DEFAULT_TYPE_VOCAB_SIZE = 2
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPSILON = 1e-12
    DEFAULT_HIDDEN_ACT = "gelu"
    #: Longest sequence the global layers' RoPE tables are precomputed for.
    #: A longer input raises in ``RotaryPositionEmbedding.call``.
    DEFAULT_MAX_POSITION_EMBEDDINGS = 8192
    #: RoPE base for the global layers. Warner et al. use a large base on the
    #: global layers precisely because they must resolve 8192-token distances.
    DEFAULT_GLOBAL_ROPE_THETA = 160000.0
    _SEQUENCE_LENGTH_MESSAGE = (
        "seq_len={seq_len} exceeds max_position_embeddings={limit}. "
        "ModernBERT's global-attention layers precompute their RoPE tables for "
        "{limit} positions, so there is no rotation for position {limit}. "
        "Truncate the input to {limit} tokens, or build the model with "
        "max_position_embeddings>={seq_len}."
    )

    def __init__(
            self,
            vocab_size: int = DEFAULT_VOCAB_SIZE,
            hidden_size: int = 768,
            num_layers: int = 22,
            num_heads: int = 12,
            intermediate_size: int = 1152,
            hidden_act: str = DEFAULT_HIDDEN_ACT,
            hidden_dropout_rate: float = 0.1,
            attention_probs_dropout_rate: float = 0.1,
            type_vocab_size: int = DEFAULT_TYPE_VOCAB_SIZE,
            initializer_range: float = DEFAULT_INITIALIZER_RANGE,
            layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
            use_bias: bool = False,
            global_attention_interval: int = 3,
            local_attention_window_size: int = 128,
            max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS,
            global_rope_theta: float = DEFAULT_GLOBAL_ROPE_THETA,
            include_top: bool = False,
            task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]] = None,
            head_config_overrides: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> None:
        """Initialize the ModernBERT model instance.

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
        :param hidden_act: Activation function inside the GeGLU FFN.
        :type hidden_act: str
        :param hidden_dropout_rate: Dropout probability for embeddings/encoder.
        :type hidden_dropout_rate: float
        :param attention_probs_dropout_rate: Dropout for attention scores.
        :type attention_probs_dropout_rate: float
        :param type_vocab_size: Vocabulary size for token type IDs.
        :type type_vocab_size: int
        :param initializer_range: Stddev for weight initialization.
        :type initializer_range: float
        :param layer_norm_eps: Epsilon for every normalization layer.
        :type layer_norm_eps: float
        :param use_bias: Whether linear layers carry bias vectors.
        :type use_bias: bool
        :param global_attention_interval: Every k-th layer is global.
        :type global_attention_interval: int
        :param local_attention_window_size: The local band's full span in
            tokens (upstream's ``local_attention``); the layer receives
            ``local_attention_window_size // 2`` as its half-width. Not a
            square spatial edge length.
        :type local_attention_window_size: int
        :param max_position_embeddings: RoPE precomputation length.
        :type max_position_embeddings: int
        :param global_rope_theta: RoPE base frequency for global layers.
        :type global_rope_theta: float
        :param include_top: Whether to build the task head.
        :type include_top: bool
        :param task_config: Task of the head; required iff ``include_top``.
        :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
        :param head_config_overrides: Head constructor overrides; only valid
            with ``include_top``.
        :type head_config_overrides: Optional[Dict[str, Any]]
        :param kwargs: Additional keyword arguments for ``keras.Model``.
        :raises ValueError: If any configuration value is invalid.
        """
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(
            hidden_size, num_layers, num_heads,
            hidden_dropout_rate, attention_probs_dropout_rate,
            global_attention_interval, max_position_embeddings,
            global_rope_theta
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
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        self.global_attention_interval = global_attention_interval
        self.local_attention_window_size = local_attention_window_size
        self.max_position_embeddings = max_position_embeddings
        self.global_rope_theta = global_rope_theta
        self.include_top = include_top
        self.task_config = (
            task_config_to_dict(task_config) if task_config is not None else None
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
            )

        logger.info(
            f"Created ModernBERT foundation model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, heads={self.num_heads}"
            + (f", task head '{self.task_config['name']}'" if self.include_top else "")
        )

    def _validate_config(
            self,
            hidden_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dropout_rate: float,
            attention_probs_dropout_rate: float,
            global_attention_interval: int,
            max_position_embeddings: int,
            global_rope_theta: float
    ) -> None:
        """Validate model configuration parameters.

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
        :param global_attention_interval: Every k-th layer is global.
        :type global_attention_interval: int
        :param max_position_embeddings: RoPE precomputation length.
        :type max_position_embeddings: int
        :param global_rope_theta: RoPE base frequency.
        :type global_rope_theta: float
        :raises ValueError: If any configuration value is invalid.
        """
        if hidden_size <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("Sizes and layer/head counts must be positive.")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= hidden_dropout_rate <= 1.0):
            raise ValueError(
                f"hidden_dropout_rate must be between 0 and 1, got {hidden_dropout_rate}"
            )
        if not (0.0 <= attention_probs_dropout_rate <= 1.0):
            raise ValueError(
                "attention_probs_dropout_rate must be between 0 and 1, got "
                f"{attention_probs_dropout_rate}"
            )
        if global_attention_interval <= 0:
            raise ValueError("global_attention_interval must be positive.")
        if max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive, got "
                f"{max_position_embeddings}"
            )
        if global_rope_theta <= 0:
            raise ValueError(
                f"global_rope_theta must be positive, got {global_rope_theta}"
            )

    def _build_architecture(self) -> None:
        """Build all model components: embeddings, encoder layers, final norm.

        The attention type of layer ``i`` is decided here from
        ``global_attention_interval``; the anchors in the body record why the
        two branches are spelled the way they are and what must not be
        "simplified".
        """
        self.embeddings = ModernBertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            dropout_rate=self.hidden_dropout_rate,
            use_bias=self.use_bias,
            name="embeddings",
        )

        self.encoder_layers: List[TransformerLayer] = []
        for i in range(self.num_layers):
            # Every k-th layer uses global attention, others use the 1-D band.
            is_global = (i + 1) % self.global_attention_interval == 0
            attention_type = "group_query" if is_global else "window_band"

            attention_args = (
                {
                    "num_kv_heads": self.num_heads,
                    "max_seq_len": self.max_position_embeddings,
                    "rope_theta": self.global_rope_theta,
                }
                if is_global
                else {"window_size": self.local_attention_window_size // 2}
            )

            layer = TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=attention_type,
                attention_args=attention_args,
                attention_norm_args={'epsilon': self.layer_norm_eps},
                ffn_norm_args={'epsilon': self.layer_norm_eps},
                normalization_position='pre',
                ffn_type='geglu',
                ffn_args={'activation': self.hidden_act},
                dropout_rate=self.hidden_dropout_rate,
                attention_dropout_rate=self.attention_probs_dropout_rate,
                use_bias=self.use_bias,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name=f"encoder_layer_{i}",
            )
            self.encoder_layers.append(layer)

        # Final normalization layer after the transformer stack
        self.final_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            center=self.use_bias,  # Use bias for centering if use_bias=True
            name="final_layer_norm"
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method ModernBERT inherits ``Layer.build``, which marks the
        model built while every sub-layer is still unbuilt -- Keras warns about
        exactly that at ``layers/layer.py:393``. The shared helper traces
        ``call()`` on symbolic inputs, so what gets built cannot drift from what
        gets called: the task head is built exactly when ``include_top`` makes
        ``call()`` run it.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        :type input_shape: Any
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            attention_mask: Optional[keras.KerasTensor] = None,
            token_type_ids: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
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
        :return: With ``include_top=False``, a dictionary with the keys:
                 - ``last_hidden_state``: The sequence of hidden states at the
                   output of the final layer. Shape:
                   `(batch, seq_len, hidden_size)`.
                 - ``attention_mask``: The attention mask, passed through for
                   convenience in downstream models. When the caller omitted
                   it, an all-ones mask of the same shape as ``input_ids`` is
                   returned rather than ``None``, so the output structure is
                   independent of the input and ``predict()`` works on a
                   single-key input dict.
                 With ``include_top=True``, the head's output reduced by
                 ``finalize_head_outputs``: a bare tensor, or a dict of two or
                 more tensors (``start_logits``/``end_logits`` for QA). The
                 head's pooling sees the same resolved mask.
        :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
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

        # Only the global layers' RoPE tables bound the length; a model with no
        # global layer (num_layers < global_attention_interval) has no limit.
        if self.num_layers >= self.global_attention_interval:
            input_ids = check_sequence_length(
                self, input_ids, self.max_position_embeddings,
                "max_position_embeddings", self._SEQUENCE_LENGTH_MESSAGE,
            )

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
        resolved_mask = (
            attention_mask if attention_mask is not None
            else keras.ops.ones_like(input_ids)
        )

        if self.include_top:
            # Cast so a float16 hidden state never meets an int mask inside
            # the head's pooling (mixed precision; QA needs a float mask).
            head_outputs = self.task_head(
                {
                    "hidden_states": sequence_output,
                    "attention_mask": keras.ops.cast(
                        resolved_mask, sequence_output.dtype
                    ),
                },
                training=training,
            )
            return finalize_head_outputs(head_outputs)

        return {
            "last_hidden_state": sequence_output,
            "attention_mask": resolved_mask,
        }

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True
    ) -> None:
        """Load pretrained ENCODER weights into the model from a local checkpoint.

        Only the encoder is transferred, by layer name: a ``task_head`` in the
        checkpoint is never copied, and this model's own head (if any) keeps
        its fresh initialization. So a checkpoint of any ModernBERT, headless
        or trained under any head, warm-starts a model with any other head or
        none. To restore a whole model, head included, use
        ``keras.models.load_model``.

        Every encoder layer must load, and every checkpoint layer must find
        its place, or this raises: a checkpoint of another hidden size, depth
        (deeper or shallower) or width is refused, never half-loaded or
        truncated. On a vocabulary change (the checkpoint's ``vocab_size``
        differs from this model's) only the variables of ``embeddings`` shaped
        by the vocabulary keep their fresh initialization; the rest still
        load, or this raises.

        :param weights_path: Path to the weights file (.keras format).
        :type weights_path: str
        :param skip_mismatch: Allow the vocabulary exception above. ``False``
            refuses a vocabulary change too.
        :type skip_mismatch: bool
        :raises FileNotFoundError: If weights_path doesn't exist.
        :raises ValueError: If weights cannot be loaded, or any encoder layer
            outside the exception is left unloaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        try:
            # build() materializes the tree call() runs, as in BERT. The dummy
            # forward this replaced was fixed at length 128, which the global
            # layers' RoPE refuses when max_position_embeddings < 128.
            if not self.built:
                self.build((1, min(128, self.max_position_embeddings)))
            logger.info(f"Loading pretrained weights from {weights_path}")

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

    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = "uncased",
            cache_dir: Optional[str] = None
    ) -> str:
        """Resolve a download path for pretrained weights; always raises.

        :param variant: Model variant name (unused).
        :type variant: str
        :param dataset: Dataset/version identifier (unused).
        :type dataset: str
        :param cache_dir: Cache directory (unused).
        :type cache_dir: Optional[str]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained ModernBERT weights are distributed with "
            f"dl_techniques (requested variant '{variant}', dataset "
            f"'{dataset}'). Pass a local checkpoint instead: "
            f"ModernBERT.from_variant('{variant}', "
            f"pretrained='/path/to/weights.keras')."
        )

    @classmethod
    def from_variant(
            cls,
            variant: str,
            pretrained: Union[bool, str] = False,
            weights_dataset: str = "uncased",
            cache_dir: Optional[str] = None,
            **kwargs: Any
    ) -> "ModernBERT":
        """Create a ModernBERT model from a predefined variant.

        :param variant: One of the keys in :attr:`MODEL_VARIANTS`.
        :type variant: str
        :param pretrained: If a string, a path to a local ``.keras`` weights
            file. If True, raises ``NotImplementedError`` -- no public
            ModernBERT weights ship with ``dl_techniques``. If False (default),
            the model is randomly initialized.
        :type pretrained: Union[bool, str]
        :param weights_dataset: Dataset/version for pretrained weights.
        :type weights_dataset: str
        :param cache_dir: Directory to cache downloaded weights.
        :type cache_dir: Optional[str]
        :param kwargs: Additional arguments overriding the variant's defaults,
            including ``include_top``, ``task_config`` and
            ``head_config_overrides``. A local ``pretrained`` path restores
            the encoder only, every layer or raise (see
            :meth:`load_pretrained_weights`); a head keeps its fresh
            initialization.
        :type kwargs: Any
        :return: A configured ModernBERT instance.
        :rtype: ModernBERT
        :raises ValueError: If the variant is not recognized.
        :raises NotImplementedError: If ``pretrained`` is True.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")
        logger.info(f"Creating ModernBERT-{variant.upper()} model")
        logger.info(f"Configuration: {description}")

        load_weights_path = None
        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                load_weights_path = cls._download_weights(
                    variant, weights_dataset, cache_dir
                )

        config.update(kwargs)
        model = cls(**config)

        if load_weights_path:
            try:
                model.load_pretrained_weights(load_weights_path, skip_mismatch=True)
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
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
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "use_bias": self.use_bias,
            "global_attention_interval": self.global_attention_interval,
            "local_attention_window_size": self.local_attention_window_size,
            "max_position_embeddings": self.max_position_embeddings,
            "global_rope_theta": self.global_rope_theta,
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
    def from_config(cls, config: Dict[str, Any]) -> "ModernBERT":
        """Create a model instance from its configuration.

        :param config: A dictionary containing the model's configuration.
        :type config: Dict[str, Any]
        :return: A new ModernBERT model instance.
        :rtype: ModernBERT
        """
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the model summary with additional ModernBERT-specific information.

        :param kwargs: Additional arguments passed to ``keras.Model.summary``.
        """
        super().summary(**kwargs)
        logger.info("ModernBERT Foundation Model Configuration:")
        logger.info(
            f"  - Architecture: {self.num_layers} layers, {self.hidden_size} hidden size"
        )
        logger.info(
            f"  - Attention: Mixed Global/Window (Global every {self.global_attention_interval} layers)"
        )
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info("  - Normalization: Pre-LN with final LayerNorm")
        logger.info(
            f"  - Feed-forward: GeGLU, {self.intermediate_size} intermediate size"
        )


# ---------------------------------------------------------------------

def create_modern_bert(
        variant: str = "base",
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "uncased",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
) -> "ModernBERT":
    """Convenience factory that mirrors ``create_bert`` / ``create_resnet``.

    Thin wrapper around :meth:`ModernBERT.from_variant`; behaves identically
    to calling it directly.

    :param variant: ModernBERT variant name (``"tiny"``, ``"base"``,
        ``"large"``). Defaults to ``"base"``.
    :type variant: str
    :param pretrained: ``True`` raises ``NotImplementedError`` (no public
        weights); a string path loads a local ``.keras`` checkpoint (the
        encoder only); ``False`` (default) gives random init.
    :type pretrained: Union[bool, str]
    :param weights_dataset: unreachable: only ``pretrained=True`` reads it,
        and that raises first. Kept for signature stability.
    :type weights_dataset: str
    :param cache_dir: unreachable for the same reason.
    :type cache_dir: Optional[str]
    :param kwargs: Forwarded to :meth:`ModernBERT.from_variant` (e.g.
        ``vocab_size``, ``local_attention_window_size``, or ``include_top=True`` with a
        ``task_config`` for a model with a head).
    :type kwargs: Any
    :return: Configured ``ModernBERT`` instance.
    :rtype: ModernBERT
    :raises NotImplementedError: If ``pretrained=True``.
    :raises ValueError: If ``variant`` is unknown, or the head knobs are
        inconsistent (see :class:`ModernBERT`).

    Example:
        >>> encoder = create_modern_bert("tiny")
        >>> ner = create_modern_bert(
        ...     "tiny", include_top=True,
        ...     task_config=NLPTaskConfig(
        ...         name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
        ...         num_classes=9))
    """
    return ModernBERT.from_variant(
        variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **kwargs,
    )

# ---------------------------------------------------------------------

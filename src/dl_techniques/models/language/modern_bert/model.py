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
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskConfig
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

    The model expects inputs as a dictionary containing 'input_ids', and
    optionally 'attention_mask' and 'token_type_ids'. It outputs a dictionary
    containing the 'last_hidden_state' and the forwarded 'attention_mask'.

    **Where the positional signal comes from.** The embedding stage carries
    none -- it sums word and token-type embeddings only. Global layers get
    theirs from RoPE, applied to queries and keys inside the attention layer
    (``group_query`` with ``num_kv_heads == num_heads``, i.e. plain multi-head
    attention plus RoPE). Local layers carry **no positional term of their own**:
    they use ``window_band`` attention, a 1-D symmetric sliding band in which
    query ``i`` attends key ``j`` iff ``abs(i - j) <= local_attention_window_size // 2``.
    That is the paper's local layer, and it matches upstream exactly --
    ``transformers/modular_modernbert.py`` sets
    ``sliding_window = local_attention // 2``, so ``local_attention_window_size=128``
    means "64 tokens either side", a 128-token span. There is no grid folding,
    no square padding and no relative position bias (a 2-D tile concept the band
    layout refuses); position reaches the local layers only through the residual
    stream from the global layers' RoPE.

    Until 2026-08-25 the local layers used ``window`` attention instead, which
    folds the ``(B, L, D)`` sequence into a ``ceil(sqrt(L))`` square grid and
    attends within ``local_attention_window_size``-SQUARE blocks -- a synthetic
    adjacency text does not have, degenerate to whole-sequence attention for
    every ``L <= local_attention_window_size**2``. Do not restore it.

    **Architecture Overview:**

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

    **Variants:**

    .. code-block:: text

        variant   hidden  layers  heads  interm.  interval  window
        tiny        256      4      4      384       2        64
        base        768     22     12     1152       3       128
        large      1024     28     16     2624       3       128

        base 152.7M / large 399.6M parameters (measured 2026-08-25 at the
        shipped interval of 3; the 160.6M / 409.5M figures carried until then
        were the all-global interval=1 repair, which has more RoPE state).
        `window` is the local band's FULL span in tokens; the layer
        receives half of it as the half-width.
        interval == 1 at base/large was a REPAIR for the old square-
        block local layer, which raised ResourceExhaustedError at a
        sequence length of EIGHT (D-135). The 1-D band removed that
        cause, so the paper's interval of 3 is restored; the restore
        is pinned by measurement in the D-135 anchors below.

        The hybrid schedule is ARCHITECTURAL FIDELITY, not a memory
        optimization -- see the trade-off note on the
        `global_attention_interval` parameter below before quoting it
        as one.

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
        Applies to the embedding norm, ``final_layer_norm`` AND every one of the
        ``2 * num_layers`` in-block norms. **Numerics change (2026-08-19,
        decisions.md D-007):** the in-block norms previously ignored this knob and
        ran at the normalization factory's ``1e-6`` default. Weight shapes are
        unchanged -- existing ``.keras`` files still load -- but forward values
        move slightly.
    :type layer_norm_eps: float
    :param use_bias: Whether to use bias vectors in linear layers.
        Defaults to False.
    :type use_bias: bool
    :param global_attention_interval: Interval for inserting a global attention
        layer. E.g., 3 means every 3rd layer is global. Defaults to 3, and all
        three shipped presets now use a hybrid schedule (``tiny`` 2,
        ``base``/``large`` 3); see the D-135 anchors in ``MODEL_VARIANTS`` for
        the measurement that restored 3.

        **The hybrid schedule is architectural fidelity, NOT a memory
        optimization in this implementation, and past L ~ 2048 it costs
        slightly MORE than all-global.** The local band is a dense ``N x N``
        masked attention -- the same ``O(N^2)`` order as global, plus the mask
        -- so a "local" layer saves nothing asymptotically and pays for the
        band mask. MEASURED (this repo, CPU, host peak RSS via ``ru_maxrss``,
        ``from_variant("base", global_attention_interval=i)`` constructed AND
        forwarded once, **n = 3 draws per cell**, min-max):

        =====  ==========================  ==========================
        L      interval=1 (all global)     interval=3 (hybrid)
        =====  ==========================  ==========================
        1024   1.820 - 1.830 GB            **1.689 - 1.739 GB**
        2048   2.323 - 2.529 GB            2.303 - 2.489 GB (a tie)
        4096   **4.758 - 4.952 GB**        5.122 - 5.135 GB
        =====  ==========================  ==========================

        The hybrid wins at L=1024, is INDISTINGUISHABLE at L=2048 (the two
        ranges overlap; a single draw at that length can show either sign --
        do not quote one), and loses by about 4% at L=4096. The crossover is
        therefore below ``DEFAULT_MAX_POSITION_EMBEDDINGS = 8192``, so at most
        of this model's advertised context the paper's schedule is the more
        expensive one. It is kept anyway: matching the published architecture
        outranks a ~4% memory delta, and D-135's only reason for forcing 1 was
        that the local layer was BROKEN, which it no longer is. Re-measure with::

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
    :param local_attention_window_size: The local band's **full span in
        tokens**, exactly as upstream's ``local_attention``: a token attends
        ``local_attention_window_size // 2`` tokens either side. The
        ``window_band`` layer takes that half-width, so this value is halved on
        the way in. It is NOT a square spatial edge length. Defaults to 128.
    :type local_attention_window_size: int
    :param max_position_embeddings: Longest sequence the global layers' RoPE
        tables are precomputed for; a longer input raises. Defaults to 8192.
    :type max_position_embeddings: int
    :param global_rope_theta: RoPE base frequency for the global layers.
        Defaults to 160000.0.
    :type global_rope_theta: float
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :ivar embeddings: The embedding layer instance.
    :vartype embeddings: dl_techniques.layers.embedding.modern_bert_embeddings.ModernBertEmbeddings
    :ivar encoder_layers: The ``num_layers`` ``TransformerLayer`` instances.
    :vartype encoder_layers: List[TransformerLayer]
    :ivar final_norm: LayerNormalization applied after the stack.
    :vartype final_norm: keras.layers.LayerNormalization

    :raises ValueError: If invalid configuration parameters are provided.

    Input shape:
        Mapping with ``input_ids`` of shape ``(batch_size, seq_len)``, plus the
        optional ``attention_mask`` and ``token_type_ids`` at the same shape.

    Output shape:
        Mapping with ``last_hidden_state`` at
        ``(batch_size, seq_len, hidden_size)`` and ``attention_mask`` at
        ``(batch_size, seq_len)``.

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
        :param local_attention_window_size: The local band's FULL span in
            tokens (upstream's ``local_attention``); the layer receives
            ``local_attention_window_size // 2`` as its half-width. Not a
            square spatial edge length.
        :type local_attention_window_size: int
        :param max_position_embeddings: RoPE precomputation length.
        :type max_position_embeddings: int
        :param global_rope_theta: RoPE base frequency for global layers.
        :type global_rope_theta: float
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
        gets called.

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
                 - ``attention_mask``: The attention mask, passed through for
                   convenience in downstream models. When the caller omitted
                   it, an all-ones mask of the same shape as ``input_ids`` is
                   returned rather than ``None``, so the output structure is
                   independent of the input and ``predict()`` works on a
                   single-key input dict.
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
            "attention_mask": (
                attention_mask if attention_mask is not None
                else keras.ops.ones_like(input_ids)
            ),
        }

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True
    ) -> None:
        """Load pretrained weights into the model from a local checkpoint.

        :param weights_path: Path to the weights file (.keras format).
        :type weights_path: str
        :param skip_mismatch: Whether to skip layers with mismatched shapes.
        :type skip_mismatch: bool
        :raises FileNotFoundError: If weights_path doesn't exist.
        :raises ValueError: If weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        try:
            if not self.built:
                dummy_input = {
                    "input_ids": keras.random.randint(
                        (1, 128), 0, self.vocab_size, dtype="int32"
                    )
                }
                self(dummy_input, training=False)
            logger.info(f"Loading pretrained weights from {weights_path}")

            report = load_weights_from_checkpoint(
                target=self,
                ckpt_path=weights_path,
                skip_prefixes=(),
                strict=not skip_mismatch,
            )
            logger.info(f"Weight transfer complete: {report}")
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
        :param kwargs: Additional arguments overriding the variant's defaults.
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
        skip_mismatch = False
        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                load_weights_path = cls._download_weights(
                    variant, weights_dataset, cache_dir
                )

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

    **Head integration:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  keras.Input × 3 (ALL required)      │
        │    input_ids / attention_mask /      │
        │    token_type_ids                    │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  ModernBERT encoder (from_variant)   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  cast attention_mask → float         │
        │  (some heads, e.g. QA, need it)      │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  NLP task head (create_nlp_head)     │
        │    {hidden_states, attention_mask}   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  head output returned AS IS          │
        │  (no logits/derived-key collapse     │
        │   here, unlike create_bert_with_head)│
        └──────────────────────────────────────┘

    :param bert_variant: The ModernBERT variant to use (e.g., "base", "large").
    :type bert_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task.
    :type task_config: NLPTaskConfig
    :param pretrained: If a string, a path to a local ``.keras`` weights file.
        If True, raises ``NotImplementedError`` -- no public ModernBERT weights
        ship with ``dl_techniques``. If False (default), random init.
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
    :return: A complete `keras.Model` ready for the specified task. All three
        inputs are required by name; the Functional wrapper declares one
        ``keras.Input`` per key and Keras matches a data dict against that
        declaration exactly. For a single-segment task pass
        ``np.zeros_like(input_ids)`` as ``token_type_ids``.
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
    attention_mask_float = keras.ops.cast(
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

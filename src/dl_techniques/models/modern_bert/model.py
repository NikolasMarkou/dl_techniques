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
stack alternates: most layers use windowed local attention over
`local_attention_window_size` tokens, and every `global_attention_interval`-th
layer uses full global attention. In the paper the local layers do the bulk of
the work at linear cost, while the periodic global layers stitch the windows
together so information still crosses the whole sequence -- a token pair `L`
apart is connected after one global layer, not after `L / window` local ones.

**That cost argument does not hold for this implementation, and the direction
of the error is the opposite of an approximation.** The windowed attention
layer this package reuses is a spatial one: it folds the sequence into a
synthetic `ceil(sqrt(L))`-square grid and then pads that grid up to a multiple
of `window_size`, so every window is padded to exactly `window_size**2` token
slots. Two consequences, both mechanical:

* Whenever `L <= window_size**2` the padded grid is a single window and the
  local layer is **dense attention**, not windowed attention. At the
  `window_size=128` that `base` and `large` ship, the threshold is 16384 --
  above `DEFAULT_MAX_POSITION_EMBEDDINGS = 8192`, so **no admissible sequence
  length is ever windowed for those two variants**. `tiny` (`window_size=64`,
  threshold 4096) is the only variant where windowing engages at all, and only
  for `4097 <= L <= 8192`, where the padded grid really does partition into
  four windows (a 2x2 grid, no cross-window information flow).
* The score matrix in a local layer is `window_size**2 x window_size**2`
  *independent of `L`* -- roughly `2.7e8` entries per head per sample at
  `window_size=128`. That is ~16,384x the cost of dense attention at `L=128`
  and still ~4x dense attention at `L=8192`. The local layers are the
  expensive ones here, not the cheap ones.

So the hybrid schedule as built buys no affordability -- but the failure is
per-variant, not uniform, and both convenient summaries of it are wrong.
Measured 2026-08-18 by sweeping `L` over the admissible range against the real
grid-formation/padding/partition code:

* `base` and `large` (`window_size=128`) partition into exactly **one** window
  at every admissible `L`. Their local layers are dense attention over a
  padded 16384-slot window: padding cost and no RoPE, for exactly zero
  locality benefit.
* `tiny` (`window_size=64`) partitions into **one** window for `L <= 4096` and
  into **four** for `4097 <= L <= 8192`. In that band the windowing is real.

Do not write "windowing always degenerates" (false for `tiny` above 4096) and
do not write "windowing delivers linear cost" (false for `base` and `large` at
every admissible length). Where a local layer is dense it additionally carries
no RoPE (only `is_global` layers receive `rope_theta`) and gets its order
signal from a relative position bias over the synthetic grid instead. That bias
is weak at its initialization scale but genuinely wired: injecting `N(0, 1)`
into the table moves the output by `4.82e-03` (one window) and `3.68e-01`
(multi-window) against a zeroed table, measured 2026-08-18 in float64. This is
documented rather than fixed because no 1-D sliding-window attention layer
exists in `layers/attention/`; writing one is the real fix, and it is an open
decision. Pinned by
`tests/test_models/test_modern_bert/test_shipped_window_size.py`.

Positional information lives entirely in the attention layers, as in the
paper: the embedding stage sums word and token-type embeddings and adds no
positional term at all. The global layers therefore apply rotary position
embeddings to their queries and keys, which is what makes them position-aware;
without that they would be exactly permutation-equivariant, and a
permutation-equivariant encoder is not a language model at all.

GeGLU replaces the plain GELU feed-forward. Gating gives the FFN a
multiplicative interaction its linear-then-activate predecessor lacks, which
buys quality at equal parameter count. Bias terms are removed from most linear
and normalization layers: they contribute little at this scale, and the freed
budget is spent on width instead.

The class is a pure foundation model. It emits `{"last_hidden_state",
"attention_mask"}`, owns no pooler and no task head, and keeps BERT's API so
the two are interchangeable at call sites. Its embedding stage is its own
`layers.embedding.modern_bert_embeddings.ModernBertEmbeddings` rather than the
`BertEmbeddings` that `bert/` and `distilbert/` share, because the token-type
table BERT carries is not part of this design. Three preset variants span tiny,
base (150M) and large (280M), each pinning its own
`global_attention_interval` and `local_attention_window_size`.

No pretrained weights are distributed with this package. `pretrained=True`
raises `NotImplementedError` rather than warning and returning a randomly
initialized model, which is a deliberate choice: the previous behaviour held a
table of unreachable weight URLs and swallowed the download failure, making an
unavailable checkpoint silently indistinguishable from a successful load. Pass
a local `.keras` path to `pretrained` instead.

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
from keras import layers, ops
from typing import Optional, Union, Any, Dict, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskConfig
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

    **Where the positional signal comes from.** The embedding stage carries
    none — it sums word and token-type embeddings only. Global layers get
    theirs from RoPE, applied to queries and keys inside the attention layer
    (``group_query`` with ``num_kv_heads == num_heads``, i.e. plain multi-head
    attention plus RoPE). Local layers get theirs from a learnable relative
    position bias over a **synthetic 2-D grid**: ``window`` attention reshapes
    the ``(B, L, D)`` sequence into a ``ceil(sqrt(L))`` square grid and attends
    within ``local_attention_window_size``-square blocks of it, so a local
    layer's neighbourhood is a set of strided runs of tokens rather than a
    contiguous 1-D window, and for every ``L <= local_attention_window_size**2``
    (16384 at the default) it degenerates to attention over the whole padded
    grid. Per variant, measured: ``base`` and ``large`` (window 128) never
    partition at any admissible length, while ``tiny`` (window 64) partitions
    into four windows for ``4097 <= L <= 8192`` and only there.
    That is a real deviation from the paper's 1-D sliding window; it is
    documented rather than fixed because no 1-D sliding-window attention layer
    exists in ``layers/attention/``.

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
        TransformerLayerₖ (Pre-LN, Global Attention + RoPE -> GeGLU FFN)
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
        attention layers. This is the edge length of a **square spatial**
        window over the synthetic grid described above, not a 1-D token
        window. Defaults to 128.
    :type local_attention_window_size: int
    :param max_position_embeddings: Longest sequence the global layers' RoPE
        tables are precomputed for; a longer input raises. Defaults to 8192.
    :type max_position_embeddings: int
    :param global_rope_theta: RoPE base frequency for the global layers.
        Defaults to 160000.0.
    :type global_rope_theta: float
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :raises ValueError: If invalid configuration parameters are provided.

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
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
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
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(
            hidden_size, num_layers, num_heads,
            hidden_dropout_prob, attention_probs_dropout_prob,
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
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
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
            hidden_dropout_prob: float,
            attention_probs_dropout_prob: float,
            global_attention_interval: int,
            max_position_embeddings: int,
            global_rope_theta: float
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
            raise ValueError(
                f"hidden_dropout_prob must be between 0 and 1, got {hidden_dropout_prob}"
            )
        if not (0.0 <= attention_probs_dropout_prob <= 1.0):
            raise ValueError(
                "attention_probs_dropout_prob must be between 0 and 1, got "
                f"{attention_probs_dropout_prob}"
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

            # DECISION plan-2026-08-14T233721-d4f9beb2/D-007
            # Global layers are 'group_query' with `num_kv_heads == num_heads`
            # (arithmetically plain MHA) because that is the ONLY registry entry
            # that reaches plain self-attention AND carries RoPE.
            #
            # WHAT NOT TO DO: do NOT "simplify" this back to
            # `attention_type="multi_head"` with `attention_args={"use_rope":
            # True}`. `MultiHeadAttention` declares no RoPE parameter, and
            # `create_attention_layer` USED TO FILTER kwargs to the registry's
            # declared names and DROP the rest SILENTLY — so that spelling
            # constructed, logged, tested and serialized cleanly while doing
            # nothing. HISTORICAL as of 2026-08-17
            # (plan-2026-08-17T183311-79c63e38/D-011): the factory now RAISES on
            # an undeclared key, so the shortcut fails loudly instead. The
            # instruction is unchanged — `'group_query'` is still the only entry
            # that is plain self-attention AND carries RoPE. That was the defect:
            # with no positional term anywhere in
            # the embeddings either, every global layer was exactly
            # permutation-equivariant. Measured 2026-08-15 on CPU with
            # `global_attention_interval=1`: `max|P f(x) - f(P x)| = 1.19e-07`
            # (float32 noise) before, `4.06e-01` after. See decisions.md D-007.
            #
            # SUPERSEDE-NOTE 2026-08-18
            # (plan-2026-08-18T140459-7991552f/D-019): a proposal to widen this
            # ruling to EVERY layer -- delete the local branch so the whole
            # stack is 'group_query' and RoPE reaches all of it ("Option B") --
            # was evaluated, MEASURED and CANCELLED. The ruling above is
            # unchanged and still applies to the global layers only; the
            # measurement that killed the widening is in the D-019 anchor
            # below.
            #
            # Local layers stay 'window', which is a SPATIAL layer: it reshapes
            # a `(B, L, D)` text sequence into a synthetic `ceil(sqrt(L))^2` grid
            # and attends inside `window_size`-square blocks of it, so its
            # neighbourhood is not a 1-D token window and, whenever
            # `L <= window_size^2`, is the whole (padded) sequence. Its order
            # information is the Swin-convention relative position bias over
            # that synthetic grid. This is documented, not fixed: no 1-D
            # sliding-window attention exists in `layers/attention/`.
            #
            # DECISION plan-2026-08-17T183311-79c63e38/D-027
            # The local branch is kept as 'window' even though its COST claim is
            # not merely approximate but INVERTED. Windows are padded to
            # `window_size**2` slots, so a local layer's score matrix is
            # `window_size**2 x window_size**2` INDEPENDENT of L: ~2.7e8 entries
            # per head per sample at the shipped `window_size=128`, i.e. ~16,384x
            # dense attention at L=128 and ~4x at L=8192. With
            # DEFAULT_MAX_POSITION_EMBEDDINGS = 8192 < 128**2, base and large can
            # never window an admissible length at all.
            #
            # WHAT NOT TO DO, and why:
            #   * Do NOT "fix" this by lowering `local_attention_window_size`
            #     until windowing engages. Below `L = window_size**2` the layer is
            #     dense over a padded grid; above it, the neighbourhood is a set
            #     of STRIDED token runs over a synthetic square grid, not the
            #     paper's contiguous 1-D window. A smaller window buys a
            #     different wrong adjacency, not the paper's.
            #   * Do NOT write a 1-D sliding-window path inline here. It is new
            #     architecture and belongs in `layers/attention/`, behind the
            #     factory, with its own tests — not in a model's builder.
            #   * Do NOT re-describe the local layers as the cheap ones. They are
            #     the expensive ones in this implementation.
            # Implement-or-delete is an OPEN decision: see decisions.md D-027.
            # Pinned by tests/test_models/test_modern_bert/test_shipped_window_size.py.
            #
            # SUPERSEDE-NOTE 2026-08-18
            # (plan-2026-08-18T140459-7991552f/D-019): the "delete" half of the
            # open decision above was taken up, measured, and CLOSED as
            # CANCELLED. Everything above still stands as written -- including
            # "base and large can never window an admissible length at all",
            # which the sweep confirmed -- except that it must NOT be read as
            # "windowing degenerates everywhere". See the D-019 anchor
            # immediately below for the per-variant numbers.
            #
            # DECISION plan-2026-08-18T140459-7991552f/D-019
            # Emitting 'window' for `base`/`large` is a MEASURED no-op. Swept
            # 2026-08-18 over the real grid/pad/partition code, L in 8..8192
            # against DEFAULT_MAX_POSITION_EMBEDDINGS = 8192:
            #     window_size=128 (base, large) -> 1 window at EVERY admissible L
            #     window_size=64  (tiny)        -> 1 window for L <= 4096,
            #                                      4 windows for 4097..8192
            # So base and large pay the 16384-slot padding on two thirds of
            # their layers, and forgo RoPE there, for exactly zero locality
            # benefit -- while tiny above 4096 tokens really does partition
            # into a 2x2 grid with no cross-window information flow.
            #
            # WHAT NOT TO DO, and why:
            #   * Do NOT delete `local_attention_window_size` or
            #     `global_attention_interval`. That was "Option B" and it was
            #     CANCELLED on 2026-08-18 BECAUSE OF the numbers above: its
            #     premise was "the window path degenerates at every shipped
            #     variant size", and that is false for `tiny` above 4096. At
            #     `tiny` the knob buys a real 4x attention saving, so deleting
            #     it removes a capability rather than cleaning up a no-op.
            #   * Do NOT quietly stop emitting 'window' for base/large here
            #     either. It is the right fix, but it is a PER-VARIANT
            #     CONFIGURATION change with a weight-tree consequence -- a
            #     local layer is a 5-tensor fused-QKV subtree (qkv, proj,
            #     relative_position_bias_table) while a global one is 4
            #     projections plus 2 RoPE caches, so EVERY parameter name under
            #     a swapped layer changes, not just the bias table. It is
            #     deferred to a follow-up plan with the measurement attached.
            #   * Do NOT re-derive any of this by reading. Two readings failed
            #     here already: "the local layers have no positional term at
            #     all" is FALSE (injecting `N(0,1)` into the relative-position
            #     bias table moves the output 4.82e-03 one-window / 3.68e-01
            #     multi-window against a zeroed table, float64), and the
            #     "degenerate everywhere" premise was equally a reading.
            # Pinned by tests/test_models/test_modern_bert/test_shipped_window_size.py
            # (TestShippedWindowSizeIsDenseAttention). See decisions.md D-019.
            attention_type = "group_query" if is_global else "window"
            attention_args = (
                {
                    "num_kv_heads": self.num_heads,
                    "max_seq_len": self.max_position_embeddings,
                    "rope_theta": self.global_rope_theta,
                }
                if is_global
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
            skip_mismatch: bool = True
    ) -> None:
        """Load pretrained weights into the model.

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
                # DECISION plan-2026-08-17T183311-79c63e38/D-044
                # `keras.random.randint`, NOT `keras.random.uniform(..., dtype=
                # "int32")`. Keras 3 rejects an integer dtype on `uniform`
                # ("requires a floating point dtype"), so this build raised on
                # EVERY unbuilt model and the whole `pretrained="<path>"` route
                # was unreachable -- the README documented it as the supported
                # alternative to the (raising) `pretrained=True`. Same defect
                # and same fix as `distilbert/model.py` (D-024 there).
                dummy_input = {
                    "input_ids": keras.random.randint(
                        (1, 128), 0, self.vocab_size, dtype="int32"
                    )
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
                    "Weights loaded with skip_mismatch=True. Layers with shape "
                    "mismatches were skipped (e.g., embedding layer)."
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
    # branch here or in `from_variant`. No public ModernBERT weights are
    # distributed with dl_techniques; pass a local path via
    # `pretrained="/path/to/file.keras"` or use `pretrained=False` (default).
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
            "max_position_embeddings": self.max_position_embeddings,
            "global_rope_theta": self.global_rope_theta,
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

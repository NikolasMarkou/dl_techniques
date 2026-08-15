"""
Decoder-only language model that substitutes :class:`WaveFieldAttention` for
dot-product multi-head attention inside a GPT-2-style pre-norm stack, with a
weight-tied LM head and a `field_size` hyperparameter that jointly with
`max_seq_len` decides whether the stack is actually causal.

The mechanism being explored is a replacement for pairwise attention's quadratic
mixing. Softmax attention computes an `N x N` interaction matrix, which is both its
expressive strength and its cost. Wave-field attention instead routes information
through a shared medium: each token is mapped to an absolute position on a 1-D field
grid of `field_size` cells and *deposits* its value there, weighted by the magnitude
of its key, using a bilinear split across the two cells its real-valued position
falls between. The field is then convolved with a per-head damped-wave kernel
`k(t) = exp(-alpha * t) * cos(omega * t + phi)` for `t >= 0`, evaluated by FFT; a
learnable coupling matrix mixes heads at each grid position; and every token
*gathers* back from the convolved field at its own position. Tokens never see each
other directly -- they see what the medium carries -- so cost is `O(N*D +
G log G * H * D_h)` rather than `O(N^2 * D)`. The damping sets an effective
interaction range and the oscillation a preferred phase relationship, which is the
sense in which the kernel is a learned, structured, infinitely-long convolution
rather than a fixed local window. Two multiplicative gates then restore some
content-dependence that a pure convolution lacks: `sigmoid(Q / sqrt(d_h))` modulates
the gathered field per token, and a projection of the block input gates the output.

The model wrapped around that layer is deliberately conventional, so the attention
is the only variable under test: learned token and positional embeddings, a
LayerNorm and dropout on the embedding output, `depth` pre-norm blocks each running
attention and a `4D` GELU FFN over residual connections, a final LayerNorm, and a
head that by default reuses the transposed token embedding matrix rather than
learning its own `V x D` projection. The block is defined locally as
`WaveFieldDecoderBlock` rather than assembled from `TransformerLayer` or the
attention factory for one concrete reason: those expect a `(B, N, N)` attention mask,
while `WaveFieldAttention` takes a `(B, N)` padding mask (it has no pairwise matrix
to mask), and bending either side to fit would have meant changing shared code for a
single consumer.

Two build-time details are not optional. Sub-layers are built EAGERLY in
`_build_architecture` and again explicitly in the block's `build`, because
`WaveFieldAttention` creates weights through an initializer that calls
`keras.random.normal`; under Keras 3's symbolic tracing of a subclassed
`keras.Model`, nested build can be skipped, and the failure mode is not an error but
an initializer that re-fires on every forward pass with no backing variable. And the
embedding pipeline is ordered add -> norm -> dropout: `PositionalEmbedding`'s own
dropout is disabled and `embed_dropout` is kept as a separate post-norm layer,
because the layer's internal order would give add -> dropout -> norm, which under an
identical dropout mask differs by up to ~38% of signal RMS at this model's default
dropout rate. That is a behaviour change, not a cleanup (D-006).

Causality (MEASURED, NOT GUARANTEED — read this before decoding autoregressively):
    This module builds NO explicit causal mask. Whatever token-level causality
    the stack has comes entirely from :class:`WaveFieldAttention`'s
    left-aligned damped wave kernel, and that layer explicitly refuses to
    guarantee it: the kernel is causal on the FIELD GRID only (output at grid
    cell `g` depends only on cells `<= g`), while the bilinear scatter/gather
    spans two grid cells, so a later token can deposit into a cell an earlier
    token gathers from. See the "Causality" section of
    ``src/dl_techniques/layers/attention/wave_field_attention.py`` — "no
    sufficient condition on ``field_size`` / ``max_seq_len`` is offered here",
    "Do NOT rely on this layer for autoregressive decoding".

    Whether a leak occurs is a property of the exact
    ``(field_size, max_seq_len)`` PAIR — through the field stride
    ``(field_size - 1) / (max_seq_len - 1)`` — and the ratio
    ``field_size / max_seq_len`` is only a lossy summary of it. Measured
    end-to-end on this model (``max_seq_len=32``, ``embed_dim=64``,
    ``depth=2``, seeded, random init, one token substituted; the reported
    value is the worst absolute logit change over ALL earlier positions and
    ALL perturbed positions, against logits of magnitude ~1.1)::

        ratio  field_size  stride   worst leak (CPU / GPU)
        0.50    16         0.4839   5.46e-04 / 5.50e-04    LEAKS
        0.75    24         0.7419   3.77e-04 / 4.05e-04    LEAKS
        1.00    32         1.0000   6.71e-08 / below 1e-5  clean
        1.50    48         1.5161   4.96e-05 / 1.24e-04    LEAKS
        2.00    64         2.0323   5.96e-08 / below 1e-5  clean   <- DEFAULT
        4.00   128         4.0968   8.94e-08 / below 1e-5  clean

    A "clean" row is NOT an exact zero. The clean residue is float32 noise at
    logits of magnitude ~1.1, and it is process-history dependent: the same
    config and seed was measured at 0.0 in one process ordering and 1.565e-07
    in another, ratio 2.0 gives 0.0 at seeds 1234/7 but 6.109e-07 at seed 99,
    and values up to 1.185e-06 have been observed on GPU with no code change.
    What is pinned — and all that should be relied on — is the ORDER OF
    MAGNITUDE: a clean row stays below 1e-5, a leaky row stays above 1e-5, and
    the two are separated by ~40x in the measurements above.

    ``field_size`` defaults to ``2 * max_seq_len`` (ratio 2.0, D-002), which
    measured clean at every configuration tested and whose stride
    ``(2M - 1) / (M - 1) > 2`` keeps consecutive tokens more than one grid
    cell apart. That is evidence, not a proof, and it is NOT a ratio
    threshold: ratio 1.50 leaks while ratio 1.00 does not, so the property is
    not monotone in the ratio, and any change to ``field_size`` or
    ``max_seq_len`` must be re-measured rather than reasoned about. The pin is
    ``TestWaveFieldLLMCausalityRatioSweep`` in
    ``tests/test_models/test_wave_field/test_model.py``.

Other deliberate choices:

- No pretrained weights are distributed with this package. `pretrained=True` reaches
  `_download_weights`, which raises `NotImplementedError` naming the local-path
  alternative, rather than warning and returning a randomly initialized model — the
  behaviour it replaced handed a caller who asked for pretrained weights an untrained
  model with no exception (D-005). The surrounding `except` clause is narrowed to
  concrete I/O errors for the same reason; broadening it to `Exception` would swallow
  that `NotImplementedError` and reinstate the bug. A local checkpoint is loaded by
  passing `pretrained="/path/to/file.keras"`, with `skip_mismatch=True` so a differing
  vocabulary does not block the rest of the weights — through
  `utils.weight_transfer.load_weights_or_raise`, which refuses a load that changes
  none of the model's variables. `skip_mismatch=True` makes a checkpoint that matches
  NOTHING restore nothing and return normally, which read as success here until
  2026-08-15.
- The class default `vocab_size` is 50261 — tiktoken `gpt2`'s 50257 plus 4 special
  tokens — matching the training script's default so direct instantiation cannot
  silently disagree with the trainer (D-005).
- `call` returns a dict `{"logits", "last_hidden_state"}` rather than a bare tensor,
  so the shared causal-LM loss and data wrappers that key on `"logits"` work unchanged.

References:
    - Radford et al., 2019. Language Models are Unsupervised Multitask Learners.
      (GPT-2 reference architecture; OpenAI technical report, no arXiv id)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Gu et al., 2022. Efficiently Modeling Long Sequences with Structured State
      Spaces. (https://arxiv.org/abs/2111.00396)
    - Poli et al., 2023. Hyena Hierarchy: Towards Larger Convolutional Language
      Models. (https://arxiv.org/abs/2302.10866)
    - Xiong et al., 2020. On Layer Normalization in the Transformer Architecture.
      (https://arxiv.org/abs/2002.04745)
    - Press & Wolf, 2017. Using the Output Embedding to Improve Language Models.
      (https://arxiv.org/abs/1608.05859)
    - Cooley & Tukey, 1965. An Algorithm for the Machine Calculation of Complex
      Fourier Series. Mathematics of Computation 19(90).
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_or_raise
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.attention.wave_field_attention import (
    WaveFieldAttention,
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WaveFieldDecoderBlock(keras.layers.Layer):
    """Pre-norm transformer decoder block with :class:`WaveFieldAttention`.

    Sub-layers (pre-norm GPT-2 style):

    1. ``attn_norm`` -> :class:`WaveFieldAttention` -> residual
    2. ``ffn_norm``  -> Dense(4D, gelu) -> Dense(D) -> Dropout -> residual

    No causal mask is built here: the ONLY mask this block forwards to
    attention is the optional padding mask ``(B, N)``. Token-level causality
    is therefore whatever :class:`WaveFieldAttention` happens to provide,
    which is a MEASURED property of the ``(field_size, max_seq_len)`` pair and
    is not guaranteed — see the "Causality" section of this module's docstring
    for the measured table.

    :param embed_dim: Hidden dim (must be divisible by ``num_heads``).
    :param num_heads: Number of attention heads.
    :param ffn_intermediate_size: FFN hidden width (default ``4 * embed_dim``).
    :param max_seq_len: Maximum sequence length (used by attention to map
        token indices to field cells).
    :param field_size: Wave field grid resolution.
    :param dropout_rate: Dropout on FFN output and embedding pipeline.
    :param attention_dropout_rate: Dropout on attention output.
    :param layer_norm_eps: LayerNorm epsilon.
    :param initializer_range: Stddev for TruncatedNormal weight init.
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        field_size: int,
        ffn_intermediate_size: Optional[int] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be positive, got "
                f"embed_dim={embed_dim}, num_heads={num_heads}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if max_seq_len <= 0 or field_size <= 0:
            raise ValueError(
                f"max_seq_len and field_size must be positive, got "
                f"max_seq_len={max_seq_len}, field_size={field_size}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.field_size = field_size
        self.ffn_intermediate_size = (
            ffn_intermediate_size
            if ffn_intermediate_size is not None
            else 4 * embed_dim
        )
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        kernel_init = keras.initializers.TruncatedNormal(
            stddev=initializer_range,
        )

        self.attn_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="attn_norm",
        )
        self.attention = WaveFieldAttention(
            dim=embed_dim,
            num_heads=num_heads,
            field_size=field_size,
            max_seq_len=max_seq_len,
            dropout_rate=attention_dropout_rate,
            kernel_initializer=kernel_init,
            name="attention",
        )

        self.ffn_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="ffn_norm",
        )
        self.ffn_dense_1 = keras.layers.Dense(
            self.ffn_intermediate_size,
            activation="gelu",
            kernel_initializer=kernel_init,
            name="ffn_dense_1",
        )
        self.ffn_dense_2 = keras.layers.Dense(
            embed_dim,
            kernel_initializer=kernel_init,
            name="ffn_dense_2",
        )
        self.ffn_dropout = keras.layers.Dropout(
            dropout_rate, name="ffn_dropout",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer so a ``.keras`` reload restores
        weights onto already-built sub-layers (H5).

        Explicit build of WaveFieldAttention is especially required: when the
        block is invoked, Keras 3's ``__call__`` wrapper triggers build for the
        block but does not always reach nested sub-layer build paths before the
        inner call is traced -- so ``add_weight`` inside the attention layer can
        fail with "'NoneType' object has no attribute 'assign'". Building it
        explicitly here pins variable creation to the block's build phase.

        Args:
            input_shape: Shape of the block input ``(B, seq, embed_dim)``.
        """
        # Attention block: pre-norm -> WaveFieldAttention.
        self.attn_norm.build(input_shape)
        self.attention.build(input_shape)

        # FFN block: pre-norm -> dense_1 -> dense_2 -> dropout.
        self.ffn_norm.build(input_shape)
        self.ffn_dense_1.build(input_shape)
        ffn_hidden_shape = tuple(input_shape[:-1]) + (self.ffn_intermediate_size,)
        self.ffn_dense_2.build(ffn_hidden_shape)
        self.ffn_dropout.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        # Block 1: pre-norm + WaveFieldAttention + residual.
        h = self.attn_norm(inputs)
        h = self.attention(
            h, attention_mask=attention_mask, training=training,
        )
        x = inputs + h

        # Block 2: pre-norm + FFN + residual.
        h = self.ffn_norm(x)
        h = self.ffn_dense_1(h)
        h = self.ffn_dense_2(h)
        h = self.ffn_dropout(h, training=training)
        return x + h

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "field_size": self.field_size,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WaveFieldLLM(keras.Model):
    """Decoder-only language model with WaveFieldAttention blocks.

    Mirrors the public surface of :class:`GPT2` so it slots into the same
    training pipeline. The two notable differences are:

    1. Attention is :class:`WaveFieldAttention` (FFT damped-wave field). No
       explicit causal mask is constructed anywhere in this module, and
       token-level causality is a MEASURED property of the
       ``(field_size, max_seq_len)`` pair rather than a guarantee — see the
       "Causality" section of this module's docstring.
    2. A new hyperparameter ``field_size`` (defaults to ``2 * max_seq_len``,
       see ``DECISION plan_2026-05-07_1519e34f/D-002``).

    Output is a dict ``{"logits", "last_hidden_state"}`` so that
    :class:`MaskedCausalLMLoss` and the standard CLM data-wrapper that keys
    on ``"logits"`` work unchanged.

    :param vocab_size: Vocabulary size. Default 50261 (Tiktoken ``gpt2``
        + 4 special tokens — see DECISION ``D-005``).
    :param embed_dim: Hidden dim. Default 768.
    :param depth: Number of decoder blocks. Default 12.
    :param num_heads: Number of attention heads. Default 12.
    :param max_seq_len: Maximum sequence length. Default 1024.
    :param field_size: Wave field grid resolution. ``None`` -> ``2 * max_seq_len``
        (see DECISION ``D-002``). This value and ``max_seq_len`` jointly decide
        whether the stack leaks future tokens; do not change either without
        re-running the ratio sweep named in the module docstring's "Causality"
        section.
    :param dropout_rate: Dropout for embedding and FFN paths. Default 0.0.
    :param attention_dropout_rate: Dropout on attention output. Default 0.0.
    :param initializer_range: Stddev for TruncatedNormal weight init.
        Default 0.02.
    :param layer_norm_eps: LayerNorm epsilon. Default 1e-5.
    :param tie_word_embeddings: Reuse transposed token embedding as LM head
        (DECISION ``D-003``). Default True.
    :param kwargs: Forwarded to ``keras.Model``.
    """

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "xl": {
            "embed_dim": 1600,
            "depth": 48,
            "num_heads": 25,
            "max_seq_len": 1024,
            "field_size": 2048,
            "description": "WaveFieldLLM XL: ~1.5B parameter class",
        },
        "large": {
            "embed_dim": 1280,
            "depth": 36,
            "num_heads": 20,
            "max_seq_len": 1024,
            "field_size": 2048,
            "description": "WaveFieldLLM Large: ~774M parameter class",
        },
        "medium": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "max_seq_len": 1024,
            "field_size": 2048,
            "description": "WaveFieldLLM Medium: ~355M parameter class",
        },
        "small": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "max_seq_len": 1024,
            "field_size": 2048,
            "description": "WaveFieldLLM Small: ~124M parameter class",
        },
        "tiny": {
            "embed_dim": 256,
            "depth": 4,
            "num_heads": 4,
            "max_seq_len": 512,
            "field_size": 1024,
            "description": "WaveFieldLLM Tiny: lightweight for testing",
        },
    }

    # DECISION plan_2026-05-07_1519e34f/D-005 — class default vocab matches
    # train script default (tiktoken `gpt2` 50257 base + 4 special) so no
    # silent vocab mismatch when a user instantiates the class directly.
    DEFAULT_VOCAB_SIZE = 50261
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPS = 1e-5

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 1024,
        field_size: Optional[int] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPS,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # DECISION plan_2026-05-07_1519e34f/D-002 — default field_size to
        # 2 * max_seq_len: sub-cell bilinear precision at modest FFT cost.
        if field_size is None:
            field_size = 2 * max_seq_len

        self._validate_config(
            vocab_size, embed_dim, depth, num_heads,
            field_size, max_seq_len,
            dropout_rate, attention_dropout_rate,
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.field_size = field_size
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.tie_word_embeddings = tie_word_embeddings

        self._build_architecture()

        logger.info(
            f"Created WaveFieldLLM: depth={self.depth}, "
            f"embed_dim={self.embed_dim}, heads={self.num_heads}, "
            f"max_seq_len={self.max_seq_len}, field_size={self.field_size}, "
            f"tie_word_embeddings={self.tie_word_embeddings}"
        )

    @staticmethod
    def _validate_config(
        vocab_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        field_size: int,
        max_seq_len: int,
        dropout_rate: float,
        attention_dropout_rate: float,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if field_size <= 1:
            raise ValueError(
                f"field_size must be > 1, got {field_size}"
            )
        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be positive, got {max_seq_len}"
            )
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {dropout_rate}"
            )
        if not 0.0 <= attention_dropout_rate <= 1.0:
            raise ValueError(
                f"attention_dropout_rate must be between 0 and 1, "
                f"got {attention_dropout_rate}"
            )

    def _build_architecture(self) -> None:
        kernel_init = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range,
        )

        self.token_embeddings = keras.layers.Embedding(
            self.vocab_size,
            self.embed_dim,
            embeddings_initializer=kernel_init,
            name="token_embeddings",
        )
        # DECISION plan-2026-08-13T091555-230c101d/D-006
        # `PositionalEmbedding` owns the slice + broadcast-add, so the manual
        # `ops.arange` / `token_emb + pos_emb` pair is gone. Its own dropout is
        # deliberately DISABLED (`dropout_rate=0.0`) and `embed_dropout` is kept
        # as a separate layer applied AFTER `embed_norm`: this model's order is
        # add -> norm -> dropout, while `PositionalEmbedding.call` would give
        # add -> dropout -> norm. Under an IDENTICAL dropout mask those two
        # orders differ by max |delta| 0.395 on unit-variance activations at the
        # model's own default dropout_rate=0.1 (~38% of signal RMS), so folding
        # `embed_dropout` into this call is a behaviour change, not a cleanup.
        # Do NOT "simplify" by passing dropout_rate=self.dropout_rate here and
        # deleting `embed_dropout`. See decisions.md D-006.
        self.position_embeddings = create_embedding_layer(
            'positional_learned',
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout_rate=0.0,
            scale=self.initializer_range,
            name="position_embeddings",
        )
        self.embed_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="embed_norm",
        )
        self.embed_dropout = keras.layers.Dropout(
            self.dropout_rate, name="embed_dropout",
        )

        self.blocks = [
            WaveFieldDecoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                max_seq_len=self.max_seq_len,
                field_size=self.field_size,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                layer_norm_eps=self.layer_norm_eps,
                initializer_range=self.initializer_range,
                name=f"block_{i}",
            )
            for i in range(self.depth)
        ]

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="final_norm",
        )

        if not self.tie_word_embeddings:
            self.lm_head = keras.layers.Dense(
                self.vocab_size,
                use_bias=False,
                kernel_initializer=kernel_init,
                name="lm_head",
            )
        else:
            self.lm_head = None

        # Eagerly build the embedding tables and decoder blocks. The
        # WaveFieldAttention layer constructs its weights via
        # `IdentityPlusNoise` which calls `keras.random.normal` at build
        # time. Under Keras 3's symbolic call tracing for `keras.Model`
        # subclasses, nested-layer build can be skipped, leading to the
        # initializer firing on every forward pass with no backing
        # variable. Eagerly building here pins variable creation to model
        # construction time, before any tracing occurs.
        block_input_shape: Tuple[Optional[int], ...] = (
            None, self.max_seq_len, self.embed_dim,
        )
        self.token_embeddings.build((None, self.max_seq_len))
        self.position_embeddings.build(block_input_shape)
        self.embed_norm.build(block_input_shape)
        for block in self.blocks:
            block.build(block_input_shape)
        self.final_norm.build(block_input_shape)
        if self.lm_head is not None:
            self.lm_head.build(block_input_shape)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain 'input_ids' key"
                )
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        # `position_embeddings` slices its table to the incoming seq_len and
        # adds it to the token embeddings (see D-006 for why its own dropout
        # is off and `embed_dropout` stays a separate post-norm layer).
        x = self.position_embeddings(
            self.token_embeddings(input_ids), training=training,
        )

        x = self.embed_norm(x)
        x = self.embed_dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x, attention_mask=attention_mask, training=training,
            )

        x = self.final_norm(x)

        if self.tie_word_embeddings:
            embedding_weights = self.token_embeddings.embeddings
            logits = ops.matmul(x, ops.transpose(embedding_weights))
        else:
            logits = self.lm_head(x)

        return {
            "logits": logits,
            "last_hidden_state": x,
        }

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        return {
            "logits": (*input_shape, self.vocab_size),
            "last_hidden_state": (*input_shape, self.embed_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "field_size": self.field_size,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
        })
        return config

    # DECISION plan-2026-08-13T091555-230c101d/D-005
    # _download_weights raises NotImplementedError instead of falling back to
    # random-init. The prior behaviour (logger.warning + return model) handed a
    # caller who asked for pretrained=True an untrained model with no
    # exception. Do NOT reinstate a warn-and-return branch here or in
    # from_variant. No public WaveFieldLLM weights are distributed with
    # dl_techniques; users must pass a local path via
    # pretrained="/path/to/file.keras" or pretrained=False (default).
    @staticmethod
    def _download_weights(
        variant: str,
        cache_dir: Optional[str] = None,
    ) -> str:
        """Resolve a local path for pretrained weights of ``variant``.

        Not implemented: no public WaveFieldLLM weights are distributed with
        ``dl_techniques``. Always raises ``NotImplementedError``. This method
        exists to mirror the BERT / GPT-2 factory recipe and to provide an
        explicit failure mode in place of a silent random-init fallback.

        :param variant: Variant name (unused).
        :type variant: str
        :param cache_dir: Cache directory (unused).
        :type cache_dir: Optional[str]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Pretrained WaveFieldLLM weights are not distributed with "
            "dl_techniques. Pass pretrained=<local_path> to load a local "
            "checkpoint, or pretrained=False to random-init."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "WaveFieldLLM":
        """Instantiate from a named variant in :data:`MODEL_VARIANTS`.

        :param variant: Variant name: ``'tiny'``, ``'small'``, ``'medium'``,
            ``'large'``, ``'xl'``.
        :param pretrained: If ``True``, raises ``NotImplementedError`` (no
            public WaveFieldLLM weights are distributed by this library). If a
            string path is supplied, the model is built (with a dummy forward
            pass) and weights are loaded with ``skip_mismatch=True``. If
            ``False`` (default), returns a random-initialized model.
        :param kwargs: Override any variant parameter.
        :raises ValueError: If the variant name is not recognized.
        :raises NotImplementedError: If ``pretrained=True``.
        :raises FileNotFoundError: If ``pretrained`` is a path that does not
            exist.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
        config.update(kwargs)

        model = cls(**config)

        if pretrained:
            import os
            weights_path = pretrained if isinstance(pretrained, str) else None
            if weights_path is not None:
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(
                        f"Weights file not found: {weights_path}"
                    )
                if not model.built:
                    import numpy as np
                    dummy = np.random.randint(
                        0, model.vocab_size, (1, 32),
                    ).astype("int32")
                    model(dummy, training=False)
                # DECISION plan-2026-08-14T233721-d4f9beb2/D-070: do NOT go back
                # to a bare `model.load_weights(path, skip_mismatch=True)`. A
                # non-matching checkpoint restores NOTHING and returns normally,
                # so the old log line reported a successful load of an untrained
                # model. See decisions.md D-070.
                load_weights_or_raise(model, weights_path, skip_mismatch=True)
            else:
                # DECISION plan-2026-08-13T091555-230c101d/D-005
                # Do NOT broaden this except clause to `Exception`: that would
                # swallow the NotImplementedError from _download_weights and
                # return a random-init model masquerading as pretrained, which
                # is the exact bug this branch replaced. Only concrete I/O
                # errors (a missing/corrupt local mirror) are caught.
                try:
                    resolved_path = cls._download_weights(variant)
                except (IOError, OSError, ValueError) as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {e}. "
                        f"Continuing with random initialization."
                    )
                    resolved_path = None
                if resolved_path is not None:
                    if not model.built:
                        import numpy as np
                        dummy = np.random.randint(
                            0, model.vocab_size, (1, 32),
                        ).astype("int32")
                        model(dummy, training=False)
                    # DECISION plan-2026-08-14T233721-d4f9beb2/D-070: same guard on
                    # the downloaded-mirror path.
                    load_weights_or_raise(model, resolved_path, skip_mismatch=True)

        return model

# ---------------------------------------------------------------------
# Module-level Factory
# ---------------------------------------------------------------------


def create_wave_field_llm(
    variant: str = "small",
    vocab_size: Optional[int] = None,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> "WaveFieldLLM":
    """Convenience factory that mirrors ``create_bert`` / ``create_gpt2``.

    Thin wrapper around :meth:`WaveFieldLLM.from_variant` exposing the most
    common construction arguments at module level. Behaves identically to
    calling ``WaveFieldLLM.from_variant(...)`` directly.

    :param variant: Variant name (``"tiny"``, ``"small"``, ``"medium"``,
        ``"large"``, ``"xl"``). Defaults to ``"small"``.
    :type variant: str
    :param vocab_size: Optional vocabulary size override. If ``None``
        (default), the variant's own vocabulary size is used. If provided, it
        is forwarded as ``vocab_size=...`` in ``kwargs``.
    :type vocab_size: Optional[int]
    :param pretrained: If ``True``, raises ``NotImplementedError`` — no public
        WaveFieldLLM weights are distributed by this library. If a string path,
        loads local weights from that path. If ``False`` (default), random
        init.
    :type pretrained: Union[bool, str]
    :param kwargs: Additional keyword arguments forwarded to
        :meth:`WaveFieldLLM.from_variant` (e.g. ``dropout_rate``,
        ``tie_word_embeddings``).
    :type kwargs: Any

    :returns: Configured ``WaveFieldLLM`` instance.
    :rtype: WaveFieldLLM

    :raises NotImplementedError: If ``pretrained=True`` (no public weights).
    :raises FileNotFoundError: If ``pretrained`` is a string path that does
        not exist.
    :raises ValueError: If ``variant`` is not a recognized variant.

    Example:
        >>> model = create_wave_field_llm("small")
        >>> model = create_wave_field_llm("tiny", vocab_size=200)
    """
    if vocab_size is not None:
        kwargs["vocab_size"] = vocab_size
    return WaveFieldLLM.from_variant(
        variant,
        pretrained=pretrained,
        **kwargs,
    )

# ---------------------------------------------------------------------

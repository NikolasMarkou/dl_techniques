"""The class-conditional latent Diffusion Transformer (DiT), built by the
``DiT`` class.

The model patchifies a latent, adds a frozen 2-D sin-cos position table,
builds one conditioning vector from the timestep and class label, runs
``depth`` adaLN-Zero transformer blocks, reads out through a zero-initialized
final layer, and unpatchifies back to the latent grid. Every block and the
final layer start as an exact identity, so a 28-block stack trains from a
well-behaved starting point; the model output is exactly ``0.0`` at
initialization, which is expected, not a defect.

This port is channels-last throughout: input is ``(B, H, W, C)`` and output
is ``(B, H, W, out_channels)``, unlike the upstream PyTorch implementation's
``(N, C, H, W)``. See :func:`unpatchify_tokens` for the axis derivation this
layout change requires.

References:
    - Peebles, W. and Xie, S. "Scalable Diffusion Models with Transformers."
      arXiv:2212.09748, 2022. https://arxiv.org/abs/2212.09748
    - Ho, J., Jain, A. and Abbeel, P. "Denoising Diffusion Probabilistic
      Models." arXiv:2006.11239, 2020. https://arxiv.org/abs/2006.11239
    - He, K. et al. "Masked Autoencoders Are Scalable Vision Learners."
      arXiv:2111.06377, 2021. https://arxiv.org/abs/2111.06377 (origin of
      the fixed 2-D sin-cos positional table this model uses).
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import numpy as np

from dl_techniques.layers.embedding.class_label_embedding import ClassLabelEmbedding
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.sincos_pos_embed_2d import get_2d_sincos_pos_embed
from dl_techniques.layers.embedding.timestep_embedding import TimestepEmbedding
from dl_techniques.models.vision_language.dit.blocks import DiTBlock, DiTFinalLayer
from dl_techniques.models.vision_language.dit.config import (
    DIT_VARIANTS,
    get_variant_config,
    normalize_variant_name,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

#: The three positional inputs :meth:`DiT.call` takes, in order. Named so the
#: model, its error messages and the tests agree on the tuple layout instead of
#: each spelling out indices.
MODEL_INPUT_NAMES: Tuple[str, ...] = ("x", "t", "y")

#: Number of leading output channels classifier-free guidance is applied
#: to: three, not ``in_channels``. See :meth:`DiT.forward_with_cfg`.
CFG_GUIDED_CHANNELS: int = 3

#: Standard deviation of the label-table initializer, from upstream's
#: ``nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)``.
LABEL_TABLE_INIT_STDDEV: float = 0.02


def flattened_linear_xavier(
    fan_in: int, fan_out: int
) -> keras.initializers.Initializer:
    """Xavier-uniform for a conv kernel treated as a flattened ``Linear``.

    Pure and allocation-free, returning a fresh initializer object every
    call. Callers pass statically known integers; nothing here reads a layer
    or a tensor.

    # DECISION plan-2026-09-02T170923-1285ed83/D-013: fan_out is computed as if
    # the kernel were reshaped to (D, p*p*C_in), matching upstream — a plain Keras
    # glorot_* default computes fan_out as p*p*D instead, a 4x difference at p=2. See decisions.md.

    :param fan_in: ``patch_h * patch_w * in_channels``.
    :type fan_in: int
    :param fan_out: The projection width, ``hidden_size``.
    :type fan_out: int
    :return: A fresh ``RandomUniform`` over ``[-limit, +limit]`` with
        ``limit = sqrt(6 / (fan_in + fan_out))``.
    :rtype: keras.initializers.Initializer
    :raises ValueError: If either fan is not positive.
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError(
            f"both fans must be positive, got fan_in={fan_in}, fan_out={fan_out}"
        )
    limit = math.sqrt(6.0 / float(fan_in + fan_out))
    return keras.initializers.RandomUniform(minval=-limit, maxval=limit)


def unpatchify_tokens(
    tokens: Any,
    grid_height: int,
    grid_width: int,
    patch_size: int,
    channels: int,
) -> Any:
    """Fold per-token patch payloads back into a channels-last image.

    Interface contract: pure, backend-agnostic, no weights. ``tokens`` is
    ``(B, grid_height * grid_width, patch_size ** 2 * channels)`` with the token
    axis in row-major order over the grid and the payload axis in row-major
    order over ``(patch_row, patch_col, channel)``. The result is
    ``(B, grid_height * patch_size, grid_width * patch_size, channels)``.
    Non-square grids are supported here even though :class:`DiT` itself only
    builds square ones -- that is what lets a test detect a transposed
    interleave, which is invisible on a square grid.

    The derivation, not a transcription of upstream's einsum
    ``'nhwpqc->nchpwq'`` (which targets NCHW and is not the answer here):
    reshaping ``tokens`` to ``(B, h, w, p, q, c)`` gives an element indexed
    ``[b, i, j, pi, pj, ci]`` where ``i, j`` select the patch and ``pi, pj``
    the pixel inside it. Its destination pixel is::

        row    = i * p + pi
        column = j * p + pj

    A single ``reshape`` produces exactly that pairing when the two source axes
    of each destination axis are adjacent and in that order, i.e. when the axis
    order is ``(b, i, pi, j, pj, ci)``. That is the permutation
    ``(0, 1, 3, 2, 4, 5)`` -- it swaps ``w`` past ``p`` and nothing else.

    .. code-block:: text

        tokens [B, h*w, p*p*c]
             │ reshape
             ▼
        [B, h, w, p, q, c]        axes: (b, i, j, pi, pj, ci)
             │ transpose (0, 1, 3, 2, 4, 5)
             ▼
        [B, h, p, w, q, c]        axes: (b, i, pi, j, pj, ci)
             │ reshape           ⊕ row = i*p + pi, col = j*p + pj
             ▼
        image [B, h*p, w*p, c]

    The transposed alternative ``(0, 2, 4, 1, 3, 5)`` produces a tensor of the
    identical shape on a square grid and passes every shape assertion, so the
    guard for this function is a delta-impulse at an asymmetric patch
    coordinate whose destination index is computed independently.

    :param tokens: ``(B, T, patch_size ** 2 * channels)`` token payloads.
    :type tokens: Any
    :param grid_height: Number of patch rows ``h``.
    :type grid_height: int
    :param grid_width: Number of patch columns ``w``.
    :type grid_width: int
    :param patch_size: Patch side ``p``.
    :type patch_size: int
    :param channels: Output channel count ``c``.
    :type channels: int
    :return: ``(B, grid_height * patch_size, grid_width * patch_size, channels)``.
    :rtype: Any
    :raises ValueError: If any dimension is not a positive integer.
    """
    for name, value in (
        ("grid_height", grid_height),
        ("grid_width", grid_width),
        ("patch_size", patch_size),
        ("channels", channels),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")

    batch = keras.ops.shape(tokens)[0]
    h, w, p, c = grid_height, grid_width, patch_size, channels

    x = keras.ops.reshape(tokens, (batch, h, w, p, p, c))
    x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return keras.ops.reshape(x, (batch, h * p, w * p, c))


def _unpack_triple(value: Any, owner: str, kind: str) -> Tuple[Any, Any, Any]:
    """Split an ``(x, t, y)`` input (or shape) triple into its three parts.

    :param value: A list or tuple of exactly three elements.
    :type value: Any
    :param owner: Class name, used in the error message.
    :type owner: str
    :param kind: ``"inputs"`` or ``"input_shape"``, used in the error message.
    :type kind: str
    :return: The three elements, in :data:`MODEL_INPUT_NAMES` order.
    :rtype: Tuple[Any, Any, Any]
    :raises ValueError: If ``value`` is not a triple.
    """
    if isinstance(value, dict):
        missing = [k for k in MODEL_INPUT_NAMES if k not in value]
        if missing:
            raise ValueError(
                f"{owner} {kind} dict is missing {missing}; required keys are "
                f"{list(MODEL_INPUT_NAMES)}"
            )
        return tuple(value[k] for k in MODEL_INPUT_NAMES)

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(
            f"{owner} takes {kind} as the triple {list(MODEL_INPUT_NAMES)} -- a "
            f"list/tuple of exactly three, or a dict with those keys. Got "
            f"{type(value).__name__} of length "
            f"{len(value) if isinstance(value, (list, tuple)) else 'n/a'}."
        )
    return value[0], value[1], value[2]


@register_dl_technique("dl_techniques.models.dit.model")
class DiT(keras.Model):
    """Class-conditional latent Diffusion Transformer, channels-last.

    Maps ``(x, t, y)`` -- a noised latent, its diffusion timestep and its class
    label -- to a same-resolution prediction with ``out_channels`` channels,
    where ``out_channels = 2 * in_channels`` when ``learn_sigma`` is set. The
    second half of the channel axis is then a variance-interpolation logit, NOT
    a second epsilon prediction.

    .. code-block:: text

        x [B, H, W, C]      t [B]              y [B] int labels
             │                │                     │
             ▼                ▼                     ▼
        ┌──────────────┐  ┌──────────────┐    ┌──────────────────┐
        │PatchEmbedding│  │TimestepEmbed │    │ClassLabelEmbed   │
        │  2D  Conv2D  │  │ sin-cos(256) │    │ table rows =     │
        │  p x p, /p   │  │ →Dense→SiLU  │    │ num_classes + 1  │
        │  use_bias    │  │ →Dense       │    │ (null row = the  │
        └──────────────┘  └──────────────┘    │  last index)     │
             │ [B, T, D]       │ [B, D]       └──────────────────┘
             ▼                 │                     │ [B, D]
        ┌──────────────┐       └──────────► ⊕ ◄──────┘
        │ + pos_embed  │                    │
        │ frozen 2-D   │                    ▼  c [B, D]
        │ sin-cos      │             (conditioning vector,
        │ [1, T, D]    │              shared by every block)
        └──────────────┘                    │
             │ [B, T, D]                    │
             ▼                              │
        ┌────────────────────────────┐      │
        │ DiTBlock  x depth          │◄─────┤
        │   adaLN-Zero (6-way)       │      │
        │   MultiHeadAttention       │      │
        │   GELU-tanh MLP            │      │
        └────────────────────────────┘      │
             │ [B, T, D]                    │
             ▼                              │
        ┌────────────────────────────┐      │
        │ DiTFinalLayer (2-way adaLN │◄─────┘
        │   + zero-init Dense)       │
        └────────────────────────────┘
             │ [B, T, p*p*out_channels]
             ▼
        ┌────────────────────────────┐
        │ unpatchify_tokens          │
        └────────────────────────────┘
             │
             ▼
        out [B, H, W, out_channels]

    The twelve named variants, transcribed in
    :data:`~dl_techniques.models.vision_language.dit.config.DIT_VARIANTS` and
    reachable through :meth:`from_variant`:

    .. code-block:: text

        variant     depth   hidden_size   num_heads   patch_size
        ────────    ─────   ───────────   ─────────   ──────────
        DiT-XL/2       28          1152          16            2
        DiT-XL/4       28          1152          16            4
        DiT-XL/8       28          1152          16            8
        DiT-L/2        24          1024          16            2
        DiT-L/4        24          1024          16            4
        DiT-L/8        24          1024          16            8
        DiT-B/2        12           768          12            2
        DiT-B/4        12           768          12            4
        DiT-B/8        12           768          12            8
        DiT-S/2        12           384           6            2
        DiT-S/4        12           384           6            4
        DiT-S/8        12           384           6            8

    Note the token count ``T = (input_size / patch_size) ** 2`` is set by the
    patch size, not by the scale, so ``DiT-XL/2`` and ``DiT-XL/8`` have almost
    the same parameter count and wildly different attention cost.

    :param input_size: Side of the square latent grid (``H == W``). This is
        the latent resolution, not the pixel resolution.
    :type input_size: int
    :param patch_size: Patch side ``p``. Must divide ``input_size``.
    :type patch_size: int
    :param in_channels: Latent channel count ``C`` of the input.
    :type in_channels: int
    :param hidden_size: Transformer width ``D``. Must be divisible by
        ``num_heads``.
    :type hidden_size: int
    :param depth: Number of :class:`DiTBlock` layers.
    :type depth: int
    :param num_heads: Attention heads per block.
    :type num_heads: int
    :param mlp_ratio: Block FFN hidden width as a multiple of ``hidden_size``.
    :type mlp_ratio: float
    :param class_dropout_rate: Probability of replacing a label with the null
        row during training. ``0.0`` means the table has exactly
        ``num_classes`` rows, there is no null row, and classifier-free
        guidance is unavailable.
    :type class_dropout_rate: float
    :param num_classes: Number of real class labels.
    :type num_classes: int
    :param learn_sigma: If ``True``, emit ``2 * in_channels`` channels.
    :type learn_sigma: bool
    :param norm_epsilon: Epsilon of every ``LayerNormalization`` in the stack.
    :type norm_epsilon: float
    :param dropout_rate: Dropout inside the block attention and MLP.
    :type dropout_rate: float
    :param use_bias: Whether the patch projection, attention and MLP carry
        biases. Upstream is ``True`` throughout.
    :type use_bias: bool
    :param frequency_embedding_size: Width of the timestep sinusoidal ladder
        before its MLP. Upstream is ``256``.
    :type frequency_embedding_size: int
    :param label_seed: Optional seed for the label-dropout RNG.
    :type label_seed: Optional[int]
    :param kwargs: Forwarded to :class:`keras.Model`.
    :type kwargs: Any
    :raises ValueError: For a non-positive dimension, a ``patch_size`` that does
        not divide ``input_size``, a ``hidden_size`` not divisible by
        ``num_heads``, or an out-of-range ``class_dropout_rate``.

    Example:
        >>> model = DiT.from_variant("DiT-S/2", input_size=8, num_classes=10)
        >>> out = model([x, t, y])  # doctest: +SKIP
    """

    #: House-contract alias for the variant registry (``models/CLAUDE.md``
    #: § House Model Module Shape). Bound to the same object as
    #: :data:`~dl_techniques.models.vision_language.dit.config.DIT_VARIANTS`,
    #: not a copy, so the two never drift out of sync.
    MODEL_VARIANTS: Dict[str, Dict[str, int]] = DIT_VARIANTS

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_rate: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        frequency_embedding_size: int = 256,
        label_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for name, value in (
            ("input_size", input_size),
            ("patch_size", patch_size),
            ("in_channels", in_channels),
            ("hidden_size", hidden_size),
            ("depth", depth),
            ("num_heads", num_heads),
            ("num_classes", num_classes),
            ("frequency_embedding_size", frequency_embedding_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        if input_size % patch_size != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by patch_size "
                f"({patch_size}); a ragged or non-square token grid is out of "
                "scope for this port, exactly as it is upstream, which asserts "
                "h == w == sqrt(T) when it unpatchifies"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio!r}")
        if not 0.0 <= float(class_dropout_rate) < 1.0:
            raise ValueError(
                f"class_dropout_rate must lie in [0.0, 1.0), got "
                f"{class_dropout_rate!r}"
            )
        if norm_epsilon <= 0.0:
            raise ValueError(f"norm_epsilon must be positive, got {norm_epsilon!r}")

        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.class_dropout_rate = float(class_dropout_rate)
        self.num_classes = int(num_classes)
        self.learn_sigma = bool(learn_sigma)
        self.norm_epsilon = float(norm_epsilon)
        self.dropout_rate = float(dropout_rate)
        self.use_bias = bool(use_bias)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.label_seed = label_seed

        # Derived geometry. `out_channels` doubles under `learn_sigma`; the
        # extra half is a variance-interpolation logit.
        self.out_channels = (
            self.in_channels * 2 if self.learn_sigma else self.in_channels
        )
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Every sub-layer is created with an explicit name so weight paths
        # stay stable across a .keras round trip.
        self.x_embedder = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.hidden_size,
            kernel_initializer=flattened_linear_xavier(
                fan_in=self.patch_size * self.patch_size * self.in_channels,
                fan_out=self.hidden_size,
            ),
            bias_initializer="zeros",
            use_bias=self.use_bias,
            flatten=True,
            name="x_embedder",
        )
        self.t_embedder = TimestepEmbedding(
            hidden_size=self.hidden_size,
            frequency_embedding_size=self.frequency_embedding_size,
            name="t_embedder",
        )
        self.y_embedder = ClassLabelEmbedding(
            num_classes=self.num_classes,
            hidden_size=self.hidden_size,
            dropout_rate=self.class_dropout_rate,
            # Override required: the house default is "uniform", upstream is
            # normal(std=0.02); a wrong table init has no shape/count/round-trip symptom.
            embeddings_initializer=keras.initializers.RandomNormal(
                stddev=LABEL_TABLE_INIT_STDDEV
            ),
            seed=self.label_seed,
            name="y_embedder",
        )
        self.blocks = [
            DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                norm_epsilon=self.norm_epsilon,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                name=f"block_{i}",
            )
            for i in range(self.depth)
        ]
        self.final_layer = DiTFinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            norm_epsilon=self.norm_epsilon,
            use_bias=self.use_bias,
            name="final_layer",
        )

        #: Created in `build()` as a non-trainable weight, never here.
        self.pos_embed = None

        logger.info(
            "Initialized DiT: input=%dx%dx%d patch=%d hidden=%d depth=%d "
            "heads=%d tokens=%d out_channels=%d",
            self.input_size,
            self.input_size,
            self.in_channels,
            self.patch_size,
            self.hidden_size,
            self.depth,
            self.num_heads,
            self.num_patches,
            self.out_channels,
        )

    def _input_shapes(
        self, input_shape: Any
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Normalize a build/compute input spec to three shape tuples.

        :param input_shape: ``[x_shape, t_shape, y_shape]`` or the dict form.
        :type input_shape: Any
        :return: The three shapes as tuples.
        :rtype: Tuple[Tuple[Optional[int], ...], ...]
        :raises ValueError: If the spec is not a triple, or ``x_shape`` is not
            rank 4.
        """
        x_shape, t_shape, y_shape = _unpack_triple(
            input_shape, type(self).__name__, "input_shape"
        )
        shapes = tuple(tuple(s) for s in (x_shape, t_shape, y_shape))
        if len(shapes[0]) != 4:
            raise ValueError(
                f"x_shape must be rank 4 (B, H, W, C) -- this port is "
                f"channels-LAST, unlike upstream's (N, C, H, W). Got "
                f"{shapes[0]!r}"
            )
        return shapes

    def build(self, input_shape: Any) -> None:
        """Materialize the positional table and exactly the tree ``call`` runs.

        :param input_shape: ``[x_shape, t_shape, y_shape]``.
        :type input_shape: Any
        :raises ValueError: If the spec is malformed.
        """
        if self.built:
            return

        x_shape, t_shape, y_shape = self._input_shapes(input_shape)
        batch = x_shape[0]
        token_shape = (batch, self.num_patches, self.hidden_size)
        c_shape = (batch, self.hidden_size)

        # The frozen 2-D sin-cos table, computed with NumPy and installed through a
        # Constant initializer — never a plain tensor attribute (does not survive a
        # .keras round trip) and never add_weight(zeros)+.assign() (StatelessScope discards it).
        # DECISION plan-2026-09-02T170923-1285ed83/D-028: table follows the variable dtype
        # policy but never narrows below float32 — do not replace with a bare
        # `dtype=self.variable_dtype`, which would drop precision under pure float16. See decisions.md.
        table_dtype = "float64" if self.variable_dtype == "float64" else "float32"
        table = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.num_patches, self.hidden_size),
            initializer=keras.initializers.Constant(
                np.asarray(table, dtype=table_dtype).reshape(
                    1, self.num_patches, self.hidden_size
                )
            ),
            trainable=False,
            dtype=table_dtype,
        )

        self.x_embedder.build(x_shape)
        self.t_embedder.build(t_shape)
        self.y_embedder.build(y_shape)
        for block in self.blocks:
            block.build([token_shape, c_shape])
        self.final_layer.build([token_shape, c_shape])

        super().build(input_shape)
        logger.debug(
            "Built DiT '%s': tokens=%d hidden=%d depth=%d",
            self.name,
            self.num_patches,
            self.hidden_size,
            self.depth,
        )

    def unpatchify(self, tokens: Any) -> Any:
        """``(B, T, p*p*out_channels)`` -> ``(B, H, W, out_channels)``.

        A thin bind of :func:`unpatchify_tokens` to this model's stored
        geometry. Exposed as a method so a test can drive the interleave
        directly without a forward pass.

        :param tokens: Per-token patch payloads.
        :type tokens: Any
        :return: ``(B, input_size, input_size, out_channels)``.
        :rtype: Any
        """
        return unpatchify_tokens(
            tokens,
            grid_height=self.grid_size,
            grid_width=self.grid_size,
            patch_size=self.patch_size,
            channels=self.out_channels,
        )

    def call(
        self,
        inputs: Union[List[Any], Tuple[Any, ...], Dict[str, Any]],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the full forward pass.

        :param inputs: The triple ``[x, t, y]`` -- ``x`` is ``(B, H, W, C)``
            channels-last, ``t`` is ``(B,)`` diffusion timesteps and ``y`` is
            ``(B,)`` integer class labels. A dict keyed by
            :data:`MODEL_INPUT_NAMES` is also accepted.
        :type inputs: Union[List[Any], Tuple[Any, ...], Dict[str, Any]]
        :param training: Keras training flag. It gates the label dropout that
            makes classifier-free guidance possible, so a training forward pass
            and an inference one differ by more than dropout noise.
        :type training: Optional[bool]
        :return: ``(B, H, W, out_channels)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``inputs`` is not a triple.
        """
        x, t, y = _unpack_triple(inputs, type(self).__name__, "inputs")

        tokens = self.x_embedder(x, training=training)
        tokens = tokens + keras.ops.cast(self.pos_embed, tokens.dtype)

        c = self.t_embedder(t, training=training) + self.y_embedder(
            y, training=training
        )

        for block in self.blocks:
            tokens = block([tokens, c], training=training)

        tokens = self.final_layer([tokens, c], training=training)
        return self.unpatchify(tokens)

    def forward_with_cfg(
        self,
        x: Any,
        t: Any,
        y: Any,
        cfg_scale: float,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass with the batched classifier-free-guidance trick.

        The caller stacks a conditional half and an unconditional half into one
        batch: ``y[:B//2]`` are real labels and ``y[B//2:]`` are the null-row
        index ``num_classes``. This method takes the first half of ``x``,
        duplicates it so both halves see identical latents, runs one forward
        pass, mixes the two epsilon halves and returns a full-batch tensor whose
        two halves are identical in the guided channels.

        .. code-block:: text

            x [B, H, W, C]                 t [B]        y [B] = [cond | null]
             │
             ├─ half = x[:B/2]
             ▼
            combined = concat([half, half])   [B, H, W, C]
             │
             ▼  self(...)
            model_out [B, H, W, out_channels]
             │
             ├── eps  = model_out[..., :3]     ◄── three channels, see below
             └── rest = model_out[..., 3:]
                   │
            cond_eps, uncond_eps = split(eps, 2, axis=0)
            half_eps = uncond_eps ⊕ cfg_scale * (cond_eps - uncond_eps)
            eps = concat([half_eps, half_eps], axis=0)
             │
             ▼
            concat([eps, rest], axis=-1)    [B, H, W, out_channels]

        Guidance is unavailable when ``class_dropout_rate == 0``: there is no
        null row for the unconditional half to index, and the label lookup will
        fail rather than silently condition on a real class.

        :param x: ``(B, H, W, C)``. Only the first half is read.
        :type x: Any
        :param t: ``(B,)`` diffusion timesteps, full batch.
        :type t: Any
        :param y: ``(B,)`` labels, full batch: conditional then null.
        :type y: Any
        :param cfg_scale: Guidance strength ``s``. ``s = 1`` reproduces the
            conditional prediction; ``s = 0`` the unconditional one.
        :type cfg_scale: float
        :param training: Forwarded to :meth:`call`. Leave ``None``/``False``
            for sampling -- label dropout must not fire.
        :type training: Optional[bool]
        :return: ``(B, H, W, out_channels)``.
        :rtype: keras.KerasTensor
        """
        batch = keras.ops.shape(x)[0]
        half = x[: batch // 2]
        combined = keras.ops.concatenate([half, half], axis=0)
        model_out = self([combined, t, y], training=training)

        # DECISION plan-2026-09-02T170923-1285ed83/D-014: guides exactly three channels,
        # not `self.in_channels` — upstream does the same for exact reproducibility.
        # WHAT NOT TO DO: the obvious `model_out[..., :self.in_channels]` fix
        # changes published cfg_scale results. See decisions.md.
        eps = model_out[..., :CFG_GUIDED_CHANNELS]
        rest = model_out[..., CFG_GUIDED_CHANNELS:]

        eps_batch = keras.ops.shape(eps)[0]
        cond_eps = eps[: eps_batch // 2]
        uncond_eps = eps[eps_batch // 2:]
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = keras.ops.concatenate([half_eps, half_eps], axis=0)

        return keras.ops.concatenate([eps, rest], axis=-1)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Return ``(B, input_size, input_size, out_channels)`` from the config.

        :param input_shape: ``[x_shape, t_shape, y_shape]``.
        :type input_shape: Any
        :return: The output shape.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If the spec is malformed.
        """
        x_shape, _, _ = self._input_shapes(input_shape)
        return (x_shape[0], self.input_size, self.input_size, self.out_channels)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: A JSON-serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "input_size": self.input_size,
                "patch_size": self.patch_size,
                "in_channels": self.in_channels,
                "hidden_size": self.hidden_size,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "class_dropout_rate": self.class_dropout_rate,
                "num_classes": self.num_classes,
                "learn_sigma": self.learn_sigma,
                "norm_epsilon": self.norm_epsilon,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "frequency_embedding_size": self.frequency_embedding_size,
                "label_seed": self.label_seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DiT":
        """Rebuild from :meth:`get_config`.

        :param config: A configuration dictionary.
        :type config: Dict[str, Any]
        :return: The reconstructed model.
        :rtype: DiT
        """
        return cls(**config)

    @classmethod
    def _download_weights(cls, variant: str) -> str:
        """Pretrained-weights stub. Always raises.

        :param variant: The requested variant name.
        :type variant: str
        :return: Never returns.
        :rtype: str
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained DiT weights ship with dl_techniques (variant "
            f"'{variant}'). This is an architecture-faithful port, not a "
            f"weight-compatible one -- the final layer's 2-way modulation is "
            f"scale-first here, so an upstream checkpoint would not load "
            f"correctly even if one were bundled. Use "
            f"DiT.from_variant('{variant}', pretrained=False) and then "
            f"model.load_weights('/path/to/weights.keras')."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "DiT":
        """Build one of the twelve named DiT configurations.

        The variant fixes ``(depth, hidden_size, patch_size, num_heads)``.
        Everything else -- ``input_size``, ``in_channels``, ``num_classes``,
        ``class_dropout_rate``, ``learn_sigma`` -- keeps its constructor
        default unless overridden through ``kwargs``.

        :param variant: Any spelling accepted by
            :func:`~dl_techniques.models.vision_language.dit.config.normalize_variant_name`,
            e.g. ``"DiT-S/2"``, ``"S/2"`` or ``"dit_s_2"``.
        :type variant: str
        :param pretrained: ``False`` (default) for random init. A string is
            treated as a local weights path and loaded after the model is
            built. ``True`` raises ``NotImplementedError``.
        :type pretrained: Union[bool, str]
        :param kwargs: Overrides applied on top of the variant row.
        :type kwargs: Any
        :return: The configured model.
        :rtype: DiT
        :raises ValueError: If ``variant`` is not a registered name.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        """
        canonical = normalize_variant_name(variant)

        if pretrained is True:
            # Raised before anything is allocated, so a caller who asked for
            # trained weights never receives a randomly initialized model.
            cls._download_weights(canonical)

        config = get_variant_config(canonical)
        config.update(kwargs)

        logger.info(
            "Creating %s: depth=%d hidden_size=%d patch_size=%d num_heads=%d",
            canonical,
            config["depth"],
            config["hidden_size"],
            config["patch_size"],
            config["num_heads"],
        )
        model = cls(**config)

        if isinstance(pretrained, str):
            model.load_weights(pretrained)
            logger.info("Loaded DiT weights from %s", pretrained)
        return model


def create_dit(
    variant: str = "DiT-S/2",
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> DiT:
    """Module-level thin factory over :meth:`DiT.from_variant`.

    Contains no logic of its own: it exists because the repo's model packages
    all expose a ``create_<name>`` entry point, and a second implementation of
    the variant lookup would be a second thing to keep in lockstep.

    :param variant: One of the twelve keys of
        :data:`~dl_techniques.models.vision_language.dit.config.DIT_VARIANTS`,
        in any accepted spelling.
    :type variant: str
    :param pretrained: ``False`` for random init, a local weights path to load,
        or ``True`` to raise ``NotImplementedError``.
    :type pretrained: Union[bool, str]
    :param kwargs: Overrides forwarded to the constructor.
    :type kwargs: Any
    :return: The configured model.
    :rtype: DiT
    :raises ValueError: If ``variant`` is not a registered name.
    :raises NotImplementedError: If ``pretrained`` is ``True``.

    Example:
        >>> model = create_dit("DiT-S/2", input_size=8, num_classes=10)
    """
    return DiT.from_variant(variant, pretrained=pretrained, **kwargs)

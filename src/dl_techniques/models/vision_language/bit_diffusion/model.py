"""
DiTXA, the cross-attention diffusion transformer of the bidirectional bridge.

:class:`DiTXA` predicts the bridge score of a noisy tensor ``x_t`` given the
opposite endpoint ``x_cond``, a time and a prompt-kind label. Text-to-image and
image-to-text are one model: a per-sample ``direction`` tensor, not a Python
branch, picks the conditioning embedder, so one traced graph serves batches that
mix both directions. Size presets come from :meth:`DiTXA.from_variant` and
:func:`create_ditxa`; ``pretrained`` takes a local ``.keras`` path, and ``True``
raises. Two upstream details are reproduced rather than repaired: the
conditioning vector averages two timestep embeddings, and classifier-free
guidance computes ``cond + s * (cond - uncond)``.

References:
    - Peebles & Xie (2023). *Scalable Diffusion Models with Transformers*.
      arXiv:2212.09748. The DiT backbone, adaLN-Zero conditioning and the
      patchify/unpatchify pair this model extends.
    - He et al. (2022). *Masked Autoencoders Are Scalable Vision Learners*.
      arXiv:2111.06377. Source of ``get_2d_sincos_pos_embed`` (via
      ``facebookresearch/mae``), used here for the single fixed positional table.
    - Ho & Salimans (2022). *Classifier-Free Diffusion Guidance*.
      arXiv:2207.12598. The dropout row of the class-label table exists for the
      unconditional branch; the guidance formula itself lands in ``sde.py``.
    - Upstream ``dit.py`` (``DiTWithCrossAttention``), staged verbatim under the
      plan's ``reference/`` directory, the line-by-line source of this port.
"""

import math
import keras
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------------

from dl_techniques.layers.embedding.class_label_embedding import ClassLabelEmbedding
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.sincos_pos_embed_2d import get_2d_sincos_pos_embed
from dl_techniques.layers.embedding.timestep_embedding import TimestepEmbedding
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.layers.transformers.sd3_adaln import modulate
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import (
    load_weights_from_checkpoint,
    pretrained_path_or_raise,
)

from .blocks import DiTXABlock
from .config import PROMPT_NUM_CLASSES

# ---------------------------------------------------------------------------

#: Keys :meth:`DiTXA.call` requires of its input dictionary.
REQUIRED_INPUT_KEYS: Tuple[str, ...] = ("x_t", "t", "y", "x_cond", "direction")

#: The one optional key. Absent means "every sample attends to its conditioning".
OPTIONAL_INPUT_KEYS: Tuple[str, ...] = ("cond_mask",)

#: Upstream's ``self.time_scale``. ``t`` lives in ``[0, 1]`` and the sinusoidal
#: basis expects the DDPM-era ``[0, 1000]``, so the model, not the embedder,
#: multiplies.
DEFAULT_TIME_SCALE: float = 1000.0

#: Number of chunks the final layer's adaLN emits: ``shift`` and ``scale``, no gate.
NUM_FINAL_ADALN_CHUNKS: int = 2

# ---------------------------------------------------------------------------

# DECISION plan-2026-09-02T094601-77d4a04e/D-016: not "glorot_uniform" on the conv.
# Keras takes fan_out = p*p*D for a conv kernel; upstream uses fan_out = D. See decisions.md.
def flattened_linear_xavier(
    fan_in: int, fan_out: int
) -> keras.initializers.Initializer:
    """Return Xavier-uniform for a conv kernel treated as a flattened ``Linear``.

    Keras' ``glorot_uniform`` on a ``(p, p, C, D)`` kernel uses
    ``fan_out = p * p * D``; upstream reshapes the kernel to a ``Linear`` first,
    so ``fan_out = D``. A fresh initializer is returned on every call, because a
    shared instance draws identically every time and the model has three patch
    embedders.

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

# ---------------------------------------------------------------------------

def _as_batch_vector(value: Any) -> Any:
    """Drop a trailing singleton axis so ``(B, 1)`` and ``(B,)`` behave alike.

    :param value: A tensor of rank 1 or 2.
    :return: The same tensor with shape ``(B,)``.
    """
    if len(value.shape) > 1 and value.shape[-1] == 1:
        return keras.ops.squeeze(value, axis=-1)
    return value

# ---------------------------------------------------------------------------

@register_dl_technique(package="dl_techniques.models.bit_diffusion.model")
class DiTXAFinalLayer(keras.layers.Layer):
    """Turn final tokens into a channels-last bridge tensor.

    The projection is zero in kernel and bias, so a freshly built model predicts
    exactly zero (adaLN-Zero); with the zero gates of every block, the whole
    stack is the zero map at step 0, which keeps a deep stack trainable. The
    unpatchify writes the width as ``w * p``; upstream writes ``h * p`` for both
    axes (``dit.py:239``), which only works because it asserts ``H == W``.

    .. code-block:: text

        x [B, N, hidden]            c [B, hidden]
                 │                             │
                 ▼                             ▼
        norm_final (non-affine)              SiLU
                 │                             │
                 │                             ▼
                 │                  ┌─────────────────────┐
                 │                  │ adaln_dense         │  zero-init
                 │                  │ Dense(2 * hidden)   │
                 │                  └─────────────────────┘
                 │                             │  shift, scale
                 ▼                             │
             modulate ◄────────────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ linear              │  zero-init
        │ Dense(p * p * C)    │
        └─────────────────────┘
                 │  [B, N, p * p * C]
                 ▼
        reshape [B, h, w, p, p, C]
                 │
                 ▼
        transpose (0, 1, 3, 2, 4, 5)
                 │
                 ▼
        reshape [B, h * p, w * p, C]

    :param hidden_size: Model width. Must be positive.
    :type hidden_size: int
    :param patch_size: Patch side ``p``; the projection emits ``p * p *
        out_channels`` numbers per token. Must be positive.
    :type patch_size: int
    :param out_channels: Channel count ``C`` of the reconstructed bridge tensor;
        :class:`DiTXA` passes its ``in_channels``. Must be positive.
    :type out_channels: int
    :param grid_height: Patch-grid rows ``h``, ``H // patch_size``. Must be
        positive.
    :type grid_height: int
    :param grid_width: Patch-grid columns ``w``, ``W // patch_size``. Must be
        positive.
    :type grid_width: int
    :param norm_epsilon: Epsilon of the non-affine ``LayerNormalization``;
        bare Keras defaults to ``1e-3``. Not validated.
    :type norm_epsilon: float
    :param **kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    Input shape:
        ``[x, c]``: ``(B, N, hidden_size)`` and ``(B, hidden_size)``, with
        ``N == grid_height * grid_width``.

    Output shape:
        ``(B, grid_height * patch_size, grid_width * patch_size, out_channels)``.

    Example:
        >>> import keras
        >>> head = DiTXAFinalLayer(64, 2, 4, 4, 4)
        >>> x = keras.random.normal((2, 16, 64))
        >>> c = keras.random.normal((2, 64))
        >>> head([x, c]).shape
        (2, 8, 8, 4)
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        grid_height: int,
        grid_width: int,
        norm_epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """Validate the sizes and create the sub-layers.

        Arguments are documented on the class. Sizes are converted with
        ``int()``, so a float is truncated before the check.

        :raises ValueError: If any size argument is not positive.
        """
        super().__init__(**kwargs)

        for name, value in (
            ("hidden_size", hidden_size),
            ("patch_size", patch_size),
            ("out_channels", out_channels),
            ("grid_height", grid_height),
            ("grid_width", grid_width),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.norm_epsilon = float(norm_epsilon)

        self.norm_final = keras.layers.LayerNormalization(
            epsilon=self.norm_epsilon, center=False, scale=False, name="norm_final"
        )
        # Four separate Zeros() instances, one per initializer slot; an
        # initializer instance is never shared across layers.
        self.adaln_dense = keras.layers.Dense(
            NUM_FINAL_ADALN_CHUNKS * self.hidden_size,
            use_bias=True,
            kernel_initializer=keras.initializers.Zeros(),
            bias_initializer=keras.initializers.Zeros(),
            name="adaln_modulation",
        )
        self.linear = keras.layers.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            use_bias=True,
            kernel_initializer=keras.initializers.Zeros(),
            bias_initializer=keras.initializers.Zeros(),
            name="linear",
        )

    def build(self, input_shape: Any) -> None:
        """Build the norm and both Dense layers. Returns at once if already built.

        :param input_shape: ``[x_shape, c_shape]``.
        :type input_shape: Any
        :raises ValueError: If ``input_shape`` is not a pair of shapes.
        """
        if self.built:
            return
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "DiTXAFinalLayer expects [x_shape, c_shape]; got "
                f"{input_shape!r}"
            )
        x_shape, c_shape = tuple(input_shape[0]), tuple(input_shape[1])

        self.norm_final.build(x_shape)
        self.adaln_dense.build(c_shape)
        self.linear.build(x_shape)

        super().build(input_shape)

    def unpatchify(self, tokens: Any) -> Any:
        """Reshape ``(B, N, p*p*C)`` tokens into ``(B, h*p, w*p, C)``.

        Uses the configured grid, so ``N`` must equal
        ``grid_height * grid_width``.

        :param tokens: Per-token payloads.
        :return: The channels-last bridge tensor.
        """
        batch = keras.ops.shape(tokens)[0]
        p = self.patch_size
        h, w, c = self.grid_height, self.grid_width, self.out_channels
        x = keras.ops.reshape(tokens, (batch, h, w, p, p, c))
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        return keras.ops.reshape(x, (batch, h * p, w * p, c))

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Modulate, project and unpatchify.

        :param inputs: ``[x, c]``.
        :type inputs: List[keras.KerasTensor]
        :param training: Forwarded to the Dense sub-layers, which ignore it.
        :type training: Optional[bool]
        :return: ``(B, h*p, w*p, out_channels)``.
        :rtype: keras.KerasTensor
        """
        x, c = inputs[0], inputs[1]
        shift, scale = keras.ops.split(
            self.adaln_dense(keras.ops.silu(c), training=training),
            NUM_FINAL_ADALN_CHUNKS,
            axis=-1,
        )
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x, training=training)
        return self.unpatchify(x)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Return the reconstructed bridge shape.

        :param input_shape: ``[x_shape, c_shape]``.
        :type input_shape: Any
        :return: ``(B, h*p, w*p, out_channels)``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch = tuple(input_shape[0])[0]
        return (
            batch,
            self.grid_height * self.patch_size,
            self.grid_width * self.patch_size,
            self.out_channels,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: A JSON-serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "patch_size": self.patch_size,
                "out_channels": self.out_channels,
                "grid_height": self.grid_height,
                "grid_width": self.grid_width,
                "norm_epsilon": self.norm_epsilon,
            }
        )
        return config

# ---------------------------------------------------------------------------

@register_dl_technique(package="dl_techniques.models.bit_diffusion.model")
class DiTXA(keras.Model):
    """Predict the bridge score for a batch that may mix both directions.

    ``direction`` is a per-sample tensor, not a Python branch: both conditioning
    embedders are built and run on every sample and ``keras.ops.where`` keeps
    one. That costs one extra patch-embedding convolution per pass and buys a
    single traced graph, stock ``fit()`` and mixed-direction batches. Two
    upstream details are reproduced rather than repaired (see
    ``PORT_NOTES.md``): ``c`` averages the two timestep embeddings instead of
    plain DiT's ``t + y``, and ``forward_cond_scale`` multiplies the raw
    ``x_cond`` pixels in the forward direction only. Inputs must be
    ``input_size`` square; nothing checks it, and a non-square input with the
    same patch count is reshaped silently.

    Main path:

    .. code-block:: text

        x_t [B, H, W, C]
                 │
                 ▼
        ┌─────────────────────┐
        │ x_embedder          │  kernel = stride = p
        └─────────────────────┘
                 │  [B, N, hidden]
                 ▼
        + pos_embed (fixed 2D sin-cos)
                 │
                 ▼
        ┌─────────────────────┐
        │ DiTXABlock x depth  │◄── c, cond_tokens
        └─────────────────────┘
                 │  [B, N, hidden]
                 ▼
        ┌─────────────────────┐
        │ final_layer         │◄── c
        └─────────────────────┘
                 │
                 ▼
        output [B, H, W, C]

    With ``drop_path_rate > 0`` each block's change to the tokens, not its
    output, passes through ``StochasticDepth``.

    Conditioning vector ``c``:

    .. code-block:: text

        t [B]                     direction [B]
                 │                           │
                 ▼                           ▼
        * time_scale              t_cond: 1 reverse, 0 forward
                 │                           │  * time_scale
                 ▼                           ▼
        ┌─────────────────────┐   ┌─────────────────────┐
        │ t_embedder          │   │ cond_t_embedder     │
        └─────────────────────┘   └─────────────────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    average of the two
                               │
                               ▼
                    + y_embedder(y)
                               │
                               ▼
                    c [B, hidden]

    Conditioning tokens:

    .. code-block:: text

        x_cond [B, H, W, C]
                 │
                 ├───────────────────────────┐
                 ▼                           │
        * forward_cond_scale                 │
                 │                           │
                 ▼                           ▼
        ┌─────────────────────┐   ┌─────────────────────┐
        │ cond_embedder_      │   │ cond_embedder_      │
        │ forward             │   │ reverse             │
        └─────────────────────┘   └─────────────────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                where(reverse, reverse, forward)
                               │  [B, N, hidden]
                               ▼
                + pos_embed (same table as x)
                               │
                               ▼
                * cond_mask (if given)
                               │
                               ▼
                cond_tokens [B, N, hidden]

    Variants (``MODEL_VARIANTS``):

    .. code-block:: text

        variant  input  patch  channels  hidden  depth  heads
        tiny     8      2      4         64      2      4
        S        32     2      4         384     12     6
        B        32     2      4         768     12     12
        L        32     2      4         1024    24     16
        XL       32     2      4         1152    28     16

    :param input_size: Side of the square bridge tensor; the fixed positional
        table is built on a square grid. Must be positive.
    :type input_size: int
    :param patch_size: Patch side. Must be positive and divide ``input_size``.
    :type patch_size: int
    :param in_channels: Bridge channel count, also ``out_channels``: upstream
        fixes ``learn_sigma=False`` for this architecture. Must be positive.
    :type in_channels: int
    :param hidden_size: Model width. Must be positive and divisible by
        ``num_heads``.
    :type hidden_size: int
    :param depth: Number of :class:`~...blocks.DiTXABlock` layers. Must be
        positive.
    :type depth: int
    :param num_heads: Attention heads in both attention sub-layers of a block.
        Must be positive.
    :type num_heads: int
    :param mlp_ratio: Block MLP expansion factor; :class:`DiTXABlock` rejects a
        non-positive value.
    :type mlp_ratio: float
    :param num_classes: Prompt-kind classes. Default
        :data:`~...config.PROMPT_NUM_CLASSES`.
    :type num_classes: int
    :param class_dropout_rate: Label-dropout probability of ``y_embedder``. Any
        positive value adds an extra unconditional row to the label table.
    :type class_dropout_rate: float
    :param forward_cond_scale: Multiplier on the raw ``x_cond`` pixels, in the
        forward direction only, before patch embedding. Opaque upstream
        hyperparameter; default ``1.0``.
    :type forward_cond_scale: float
    :param time_scale: Multiplier applied to both ``t`` and ``t_cond`` before
        embedding. Upstream's ``1000``.
    :type time_scale: float
    :param drop_path_rate: Maximum stochastic-depth rate, in ``[0, 1)``; rates
        ramp linearly from ``0`` at block 0. ``0.0`` (the default, and
        upstream's only behaviour) creates no ``StochasticDepth`` sub-layer.
    :type drop_path_rate: float
    :param dropout_rate: Attention and MLP dropout inside every block.
    :type dropout_rate: float
    :param norm_epsilon: Epsilon of every ``LayerNormalization`` in the model.
    :type norm_epsilon: float
    :param qk_norm_epsilon: Epsilon of every per-head Q/K ``RMSNorm``.
    :type qk_norm_epsilon: float
    :param use_bias: Whether the block projections carry biases. Upstream is
        ``qkv_bias=True``.
    :type use_bias: bool
    :param frequency_embedding_size: Width of both timestep embedders'
        sinusoidal basis, decoupled from ``hidden_size``.
    :type frequency_embedding_size: int
    :param label_seed: Seed of the label embedder's own dropout RNG.
    :type label_seed: Optional[int]
    :param **kwargs: Standard ``keras.Model`` keyword arguments.

    :ivar pos_embed: The single fixed ``(1, num_patches, hidden_size)`` sin-cos
        table, non-trainable and always float32, added to ``x`` and to the
        conditioning tokens. ``None`` until built.
    :ivar x_embedder: ``PatchEmbedding2D`` of ``x_t``.
    :ivar cond_embedder_forward: ``PatchEmbedding2D`` of ``x_cond``, forward.
    :ivar cond_embedder_reverse: ``PatchEmbedding2D`` of ``x_cond``, reverse.
    :ivar t_embedder: Timestep embedder of ``t``.
    :ivar cond_t_embedder: Independent timestep embedder of ``t_cond``.
    :ivar y_embedder: The class-label table with its dropout row.
    :ivar blocks: ``depth`` ``DiTXABlock`` layers.
    :ivar drop_paths: One ``StochasticDepth`` per block, ``None`` where the
        rate is 0.
    :ivar final_layer: :class:`DiTXAFinalLayer`.

    Input shape:
        A dictionary with ``x_t`` ``(B, H, W, C)``, ``t`` ``(B,)`` or
        ``(B, 1)`` in ``[0, 1]``, ``y`` ``(B,)`` integer labels, ``x_cond``
        ``(B, H, W, C)``, ``direction`` ``(B,)`` (above 0.5 means reverse) and
        the optional ``cond_mask`` ``(B,)`` (``1`` attend, ``0`` zero the
        stream; other values scale it).

    Output shape:
        ``(B, H, W, C)``.

    Example:
        .. code-block:: python

            import keras
            from dl_techniques.models.vision_language.bit_diffusion import DiTXA

            model = DiTXA.from_variant("tiny")
            batch = {
                "x_t": keras.random.normal((2, 8, 8, 4)),
                "t": keras.ops.convert_to_tensor([0.2, 0.8]),
                "y": keras.ops.convert_to_tensor([0, 2]),
                "x_cond": keras.random.normal((2, 8, 8, 4)),
                "direction": keras.ops.convert_to_tensor([0.0, 1.0]),
            }
            model(batch).shape  # (2, 8, 8, 4)
    """

    #: hidden / depth / heads follow upstream ``DiTXA_{S,B,L,XL}/2``, all patch 2
    #: (D-003). ``tiny`` is a seconds-fast model on the ``tiny`` bridge preset's
    #: ``(8, 8, 4)`` geometry.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "input_size": 8,
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 64,
            "depth": 2,
            "num_heads": 4,
            "description": "DiTXA-tiny: the `tiny` bridge preset, for tests",
        },
        "S": {
            "input_size": 32,
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 384,
            "depth": 12,
            "num_heads": 6,
            "description": "DiTXA-S/2: smallest shipped research variant",
        },
        "B": {
            "input_size": 32,
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 768,
            "depth": 12,
            "num_heads": 12,
            "description": "DiTXA-B/2",
        },
        "L": {
            "input_size": 32,
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 1024,
            "depth": 24,
            "num_heads": 16,
            "description": "DiTXA-L/2: one of the two variants upstream trains",
        },
        "XL": {
            "input_size": 32,
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 1152,
            "depth": 28,
            "num_heads": 16,
            "description": "DiTXA-XL/2: the largest variant upstream trains",
        },
    }

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = PROMPT_NUM_CLASSES,
        class_dropout_rate: float = 0.1,
        forward_cond_scale: float = 1.0,
        time_scale: float = DEFAULT_TIME_SCALE,
        drop_path_rate: float = 0.0,
        dropout_rate: float = 0.0,
        norm_epsilon: float = 1e-6,
        qk_norm_epsilon: float = 1e-6,
        use_bias: bool = True,
        frequency_embedding_size: int = 256,
        label_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the sizes and create every sub-layer.

        Arguments are documented on the class. Sizes are converted with
        ``int()``, so a float is truncated before the check.

        :raises ValueError: If a size argument is not positive, ``input_size``
            is not divisible by ``patch_size``, ``hidden_size`` is not divisible
            by ``num_heads``, ``drop_path_rate`` is outside ``[0, 1)``, or a
            block rejects ``mlp_ratio``.
        """
        super().__init__(**kwargs)

        for name, value in (
            ("input_size", input_size),
            ("patch_size", patch_size),
            ("in_channels", in_channels),
            ("hidden_size", hidden_size),
            ("depth", depth),
            ("num_heads", num_heads),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if int(input_size) % int(patch_size) != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by patch_size "
                f"({patch_size})"
            )
        if int(hidden_size) % int(num_heads) != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if not 0.0 <= float(drop_path_rate) < 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}"
            )

        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        # `learn_sigma` is asserted False upstream for this architecture
        # (dit.py:398, 405), so the head predicts a score, not a score plus a
        # variance, and out_channels never doubles.
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.num_classes = int(num_classes)
        self.class_dropout_rate = float(class_dropout_rate)
        self.forward_cond_scale = float(forward_cond_scale)
        self.time_scale = float(time_scale)
        self.drop_path_rate = float(drop_path_rate)
        self.dropout_rate = float(dropout_rate)
        self.norm_epsilon = float(norm_epsilon)
        self.qk_norm_epsilon = float(qk_norm_epsilon)
        self.use_bias = bool(use_bias)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.label_seed = label_seed

        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        patch_fan_in = self.patch_size * self.patch_size * self.in_channels

        # Every flattened_linear_xavier(...) call below returns a fresh
        # instance; the three patch embedders must not share one.
        self.x_embedder = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.hidden_size,
            kernel_initializer=flattened_linear_xavier(
                patch_fan_in, self.hidden_size
            ),
            bias_initializer=keras.initializers.Zeros(),
            flatten=True,
            name="x_embedder",
        )
        self.cond_embedder_forward = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.hidden_size,
            kernel_initializer=flattened_linear_xavier(
                patch_fan_in, self.hidden_size
            ),
            bias_initializer=keras.initializers.Zeros(),
            flatten=True,
            name="cond_embedder_forward",
        )
        self.cond_embedder_reverse = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.hidden_size,
            kernel_initializer=flattened_linear_xavier(
                patch_fan_in, self.hidden_size
            ),
            bias_initializer=keras.initializers.Zeros(),
            flatten=True,
            name="cond_embedder_reverse",
        )
        self.t_embedder = TimestepEmbedding(
            hidden_size=self.hidden_size,
            frequency_embedding_size=self.frequency_embedding_size,
            name="t_embedder",
        )
        self.cond_t_embedder = TimestepEmbedding(
            hidden_size=self.hidden_size,
            frequency_embedding_size=self.frequency_embedding_size,
            name="cond_t_embedder",
        )
        self.y_embedder = ClassLabelEmbedding(
            num_classes=self.num_classes,
            hidden_size=self.hidden_size,
            dropout_rate=self.class_dropout_rate,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
            seed=self.label_seed,
            name="y_embedder",
        )

        self.blocks = [
            DiTXABlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                norm_epsilon=self.norm_epsilon,
                qk_norm_epsilon=self.qk_norm_epsilon,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                name=f"block_{i}",
            )
            for i in range(self.depth)
        ]

        # DECISION plan-2026-09-02T094601-77d4a04e/D-017: drop the block's delta, not its output.
        # DiTXABlock adds its own residuals; dropping the output zeros everything. See decisions.md.
        rates = linear_drop_path_rates(self.depth, self.drop_path_rate)
        self.drop_path_rates = list(rates)
        self.drop_paths = [
            StochasticDepth(drop_path_rate=rate, name=f"drop_path_{i}")
            if rate > 0.0
            else None
            for i, rate in enumerate(rates)
        ]

        self.final_layer = DiTXAFinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            grid_height=self.grid_size,
            grid_width=self.grid_size,
            norm_epsilon=self.norm_epsilon,
            name="final_layer",
        )

        self.pos_embed = None

    @staticmethod
    def _input_shapes(input_shape: Any) -> Dict[str, Tuple[Optional[int], ...]]:
        """Normalize ``build``/``compute_output_shape`` input to a shape dict.

        :param input_shape: A dict of shapes keyed like the model's inputs.
        :type input_shape: Any
        :return: The same mapping with tuple shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        :raises ValueError: If it is not a mapping, or a required key is absent.
        """
        if not isinstance(input_shape, dict):
            raise ValueError(
                "DiTXA takes a dictionary of inputs; got shape spec "
                f"{type(input_shape).__name__}. Required keys: "
                f"{list(REQUIRED_INPUT_KEYS)}"
            )
        missing = [k for k in REQUIRED_INPUT_KEYS if k not in input_shape]
        if missing:
            raise ValueError(
                f"missing required input(s) {missing}; got "
                f"{sorted(input_shape)}. Optional: {list(OPTIONAL_INPUT_KEYS)}"
            )
        return {k: tuple(v) for k, v in input_shape.items()}

    def build(self, input_shape: Any) -> None:
        """Create the positional table and build every sub-layer ``call`` runs.

        Both conditioning embedders are built, because both run on every pass
        (D-005). Returns at once if already built.

        :param input_shape: A dict of input shapes with every required key.
        :type input_shape: Any
        :raises ValueError: If ``input_shape`` is not a dict or lacks a required
            key.
        """
        if self.built:
            return

        shapes = self._input_shapes(input_shape)
        x_shape = shapes["x_t"]
        cond_shape = shapes["x_cond"]
        batch = x_shape[0]
        t_shape = shapes["t"]
        y_shape = shapes["y"]
        token_shape = (batch, self.num_patches, self.hidden_size)
        c_shape = (batch, self.hidden_size)

        # Installed through a Constant initializer: a plain tensor attribute skips the
        # .keras round trip, and an assign inside build() is discarded by StatelessScope.
        table = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.num_patches, self.hidden_size),
            initializer=keras.initializers.Constant(
                np.asarray(table, dtype="float32").reshape(
                    1, self.num_patches, self.hidden_size
                )
            ),
            trainable=False,
            dtype="float32",
        )

        self.x_embedder.build(x_shape)
        self.cond_embedder_forward.build(cond_shape)
        self.cond_embedder_reverse.build(cond_shape)
        self.t_embedder.build(t_shape)
        self.cond_t_embedder.build(t_shape)
        self.y_embedder.build(y_shape)
        for block in self.blocks:
            block.build([token_shape, c_shape, token_shape])
        for drop_path in self.drop_paths:
            if drop_path is not None:
                drop_path.build(token_shape)
        self.final_layer.build([token_shape, c_shape])

        super().build(input_shape)
        logger.debug(
            "Built DiTXA '%s': hidden=%d depth=%d heads=%d patches=%d",
            self.name,
            self.hidden_size,
            self.depth,
            self.num_heads,
            self.num_patches,
        )

    def _is_reverse(self, direction: Any) -> Any:
        """Turn a ``(B,)`` direction flag into a ``(B, 1, 1)`` boolean selector.

        :param direction: ``0`` forward, ``1`` reverse, float or integer; any
            value above 0.5 counts as reverse.
        :return: Boolean tensor broadcastable against ``(B, N, D)``.
        """
        flag = keras.ops.cast(_as_batch_vector(direction), "float32") > 0.5
        return keras.ops.reshape(flag, (-1, 1, 1))

    def _embed_conditioning(
        self,
        x_cond: Any,
        direction: Any,
        cond_mask: Optional[Any] = None,
        training: Optional[bool] = None,
    ) -> Any:
        """Build the conditioning token stream; a masked sample's stream is exactly zero.

        Both conditioning embedders run and ``direction`` selects. The mask is
        applied once, after ``pos_embed`` is added, before any block, as
        upstream: masking first would leave the positional signal in a masked
        sample.

        :param x_cond: ``(B, H, W, C)`` conditioning bridge tensor.
        :param direction: ``(B,)`` per-sample direction flag.
        :param cond_mask: Optional ``(B,)`` mask; ``None`` means all-ones. Values
            other than 0 and 1 scale the stream.
        :param training: Forwarded to the patch embedders.
        :return: ``(B, num_patches, hidden_size)`` conditioning tokens.
        """
        is_reverse = self._is_reverse(direction)

        # DECISION plan-2026-09-02T094601-77d4a04e/D-005: scale only the forward branch.
        # Two embedders, selected after embedding, keep the scale off reverse. See decisions.md.
        forward_tokens = self.cond_embedder_forward(
            x_cond * self.forward_cond_scale, training=training
        )
        reverse_tokens = self.cond_embedder_reverse(x_cond, training=training)
        cond_tokens = keras.ops.where(is_reverse, reverse_tokens, forward_tokens)

        cond_tokens = cond_tokens + keras.ops.cast(self.pos_embed, cond_tokens.dtype)

        # DECISION plan-2026-09-02T094601-77d4a04e/D-034: mask after the positional add.
        # Masking first leaves the positional signal in a masked sample. See decisions.md.
        if cond_mask is not None:
            mask = keras.ops.cast(
                _as_batch_vector(cond_mask), cond_tokens.dtype
            )
            cond_tokens = cond_tokens * keras.ops.reshape(mask, (-1, 1, 1))
        return cond_tokens

    def call(
        self,
        inputs: Dict[str, Any],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Predict the bridge score for a batch that may mix both directions.

        :param inputs: Dict with ``x_t``, ``t``, ``y``, ``x_cond``,
            ``direction`` and optionally ``cond_mask``; see the class.
        :type inputs: Dict[str, Any]
        :param training: Threaded to the label dropout, the block dropouts and
            stochastic depth.
        :type training: Optional[bool]
        :return: ``(B, H, W, out_channels)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``inputs`` is not a dict or a required key is
            missing.
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                "DiTXA takes a dictionary of inputs; required keys: "
                f"{list(REQUIRED_INPUT_KEYS)}"
            )
        missing = [k for k in REQUIRED_INPUT_KEYS if k not in inputs]
        if missing:
            raise ValueError(
                f"missing required input(s) {missing}; got {sorted(inputs)}"
            )

        x = inputs["x_t"]
        t = _as_batch_vector(inputs["t"])
        y = inputs["y"]
        x_cond = inputs["x_cond"]
        direction = inputs["direction"]
        cond_mask = inputs.get("cond_mask")

        is_reverse_flat = keras.ops.reshape(self._is_reverse(direction), (-1,))
        # Reverse conditions on the endpoint (the image, t = 1); forward on the
        # start point (the text, t = 0).
        t_cond = keras.ops.where(
            is_reverse_flat, keras.ops.ones_like(t), keras.ops.zeros_like(t)
        )

        tokens = self.x_embedder(x, training=training)
        tokens = tokens + keras.ops.cast(self.pos_embed, tokens.dtype)

        t_emb = self.t_embedder(t * self.time_scale, training=training)
        t_cond_emb = self.cond_t_embedder(
            t_cond * self.time_scale, training=training
        )
        y_emb = self.y_embedder(y, training=training)
        # DECISION plan-2026-09-02T094601-77d4a04e/D-035: average two separate embeddings.
        # Dropping /2 or reusing t_embedder raises nothing but changes adaLN. See decisions.md.
        c = (t_emb + t_cond_emb) / 2.0 + y_emb

        cond_tokens = self._embed_conditioning(
            x_cond, direction, cond_mask=cond_mask, training=training
        )

        for block, drop_path in zip(self.blocks, self.drop_paths):
            block_out = block([tokens, c, cond_tokens], training=training)
            if drop_path is None:
                tokens = block_out
            else:
                tokens = tokens + drop_path(block_out - tokens, training=training)

        return self.final_layer([tokens, c], training=training)

    def forward_with_cfg(
        self,
        inputs: Dict[str, Any],
        cfg_scale: float,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Return the classifier-free-guided prediction, in upstream's algebra.

        Runs two passes, one with the conditioning stream intact and one with
        ``cond_mask`` all zero, so every conditioning token is exactly zero, and
        returns::

            cond + cfg_scale * (cond - uncond)

        This is the textbook ``uncond + s * (cond - uncond)`` evaluated at
        ``s + 1``: one unit more guidance at every ``s``. The reference results
        and published ``cfg_scale`` values use this form, so it is ported as
        written. At ``s = 0`` it returns ``cond``. Any ``cond_mask`` in
        ``inputs`` is ignored; the conditional pass is fully unmasked. The label
        ``y`` is passed unchanged to both passes.

        :param inputs: The same dict :meth:`call` takes.
        :type inputs: Dict[str, Any]
        :param cfg_scale: Guidance strength ``s``. Not validated.
        :type cfg_scale: float
        :param training: Forwarded to both passes.
        :type training: Optional[bool]
        :return: ``(B, H, W, out_channels)``.
        :rtype: keras.KerasTensor
        """
        cond_inputs = {k: v for k, v in inputs.items() if k != "cond_mask"}
        # The all-zero mask comes from `t`, so no static batch size is needed.
        uncond_inputs = dict(cond_inputs)
        uncond_inputs["cond_mask"] = keras.ops.zeros_like(
            _as_batch_vector(inputs["t"])
        )

        cond = self(cond_inputs, training=training)
        uncond = self(uncond_inputs, training=training)
        # DECISION plan-2026-09-02T094601-77d4a04e/D-018: keep this non-textbook form.
        # Reference checkpoints and published cfg_scale values were tuned on it. See decisions.md.
        return cond + cfg_scale * (cond - uncond)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Return ``(B, input_size, input_size, out_channels)`` without building.

        :param input_shape: A dict of input shapes with every required key.
        :type input_shape: Any
        :return: The output bridge shape.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not a dict or lacks a required
            key.
        """
        shapes = self._input_shapes(input_shape)
        batch = shapes["x_t"][0]
        return (batch, self.input_size, self.input_size, self.out_channels)

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
                "num_classes": self.num_classes,
                "class_dropout_rate": self.class_dropout_rate,
                "forward_cond_scale": self.forward_cond_scale,
                "time_scale": self.time_scale,
                "drop_path_rate": self.drop_path_rate,
                "dropout_rate": self.dropout_rate,
                "norm_epsilon": self.norm_epsilon,
                "qk_norm_epsilon": self.qk_norm_epsilon,
                "use_bias": self.use_bias,
                "frequency_embedding_size": self.frequency_embedding_size,
                "label_seed": self.label_seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DiTXA":
        """Rebuild from :meth:`get_config`.

        :param config: A configuration dictionary.
        :type config: Dict[str, Any]
        :return: The reconstructed model.
        :rtype: DiTXA
        """
        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "DiTXA":
        """Build one of the named variants.

        :param variant: One of ``"tiny"``, ``"S"``, ``"B"``, ``"L"``, ``"XL"``.
        :type variant: str
        :param pretrained: ``False`` (default) or any other falsy value: random
            initialization. A ``str``: the path of a local ``.keras``
            checkpoint of a ``DiTXA``. The model is built from its own
            ``input_size`` and ``in_channels``, then every layer is copied from
            the checkpoint by name; a layer that does not load (another width or
            depth), or a checkpoint layer with no home here, raises before any
            weight is copied. The fixed positional table is not a layer and is
            not copied: it is a function of ``hidden_size`` and the patch grid,
            which the model computes itself. ``True`` raises
            ``NotImplementedError``: no weights are distributed with this
            package.
        :type pretrained: Union[bool, str]
        :param **kwargs: Overrides applied on top of the variant's entry.
        :type **kwargs: Any
        :return: The configured model; built when ``pretrained`` is a path.
        :rtype: DiTXA
        :raises ValueError: If ``variant`` is not a known key, a merged
            argument is invalid, or a ``pretrained`` checkpoint does not fit the
            model.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        :raises FileNotFoundError: If a ``pretrained`` path does not exist.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        # Raised before anything is allocated, so a caller who asked for
        # trained weights never receives a randomly initialized model.
        weights_path = pretrained_path_or_raise(pretrained, "DiTXA")

        config = dict(cls.MODEL_VARIANTS[variant])
        description = config.pop("description", "")
        config.update(kwargs)

        logger.info("Creating DiTXA-%s: %s", variant, description)
        model = cls(**config)

        if weights_path is not None:
            # DECISION plan-2026-10-14T062616-81938e81/D-042: build, then strict by-name copy.
            # model.load_weights skips mismatches silently after a build. See decisions.md.
            side, channels = model.input_size, model.in_channels
            model.build(
                {
                    "x_t": (None, side, side, channels),
                    "t": (None,),
                    "y": (None,),
                    "x_cond": (None, side, side, channels),
                    "direction": (None,),
                }
            )
            load_weights_from_checkpoint(
                model,
                weights_path,
                skip_prefixes=(),
                strict=True,
                allow_unloaded=(),
            )
            logger.info("Loaded DiTXA weights from %s", weights_path)
        return model

# ---------------------------------------------------------------------------

def create_ditxa(
    variant: str = "S",
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> DiTXA:
    """Create a :class:`DiTXA` through :meth:`DiTXA.from_variant`.

    :param variant: One of ``"tiny"``, ``"S"``, ``"B"``, ``"L"``, ``"XL"``.
    :type variant: str
    :param pretrained: ``False`` for random init, the path of a local
        ``.keras`` ``DiTXA`` checkpoint to load into the built model (every
        layer loads or the call raises), or ``True`` to raise
        ``NotImplementedError``. See :meth:`DiTXA.from_variant`.
    :type pretrained: Union[bool, str]
    :param **kwargs: Overrides forwarded to the constructor.
    :type **kwargs: Any
    :return: The configured model.
    :rtype: DiTXA
    :raises ValueError: If ``variant`` is not a known key, a merged argument
        is invalid, or a ``pretrained`` checkpoint does not fit the model.
    :raises NotImplementedError: If ``pretrained`` is ``True``.
    :raises FileNotFoundError: If a ``pretrained`` path does not exist.

    Example:
        >>> model = create_ditxa("tiny")
    """
    return DiTXA.from_variant(variant, pretrained=pretrained, **kwargs)

# ---------------------------------------------------------------------------

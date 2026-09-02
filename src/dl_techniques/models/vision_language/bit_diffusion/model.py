"""``DiTXA`` -- the bidirectional bridge's cross-attention diffusion transformer.

One network runs the bridge in both directions. Text->image and image->text are
not two models sharing weights; they are one model told, per sample, which way
time runs. That flag selects three things at once upstream -- the conditioning
timestep, which conditioning patch embedder reads ``x_cond``, and whether the
raw conditioning pixels are rescaled -- and upstream selects them with a Python
``bool`` because a PyTorch ``forward`` may branch on one.

A Keras ``call()`` may not. So ``direction`` here is a **per-sample tensor**:
both conditioning embedders are built and both run on every sample, and
``keras.ops.where`` picks the answer (D-005). That buys a single traced graph,
stock ``fit()``, and mixed-direction batches; it costs one extra patch-embedding
convolution per forward pass. The cost was measured before it was accepted, and
the per-row locality of the selection was measured too: flipping one sample's
flag moves that sample's output and moves every other row by exactly ``0.0``.

.. code-block:: text

    x_t (B,H,W,C)      t (B,)   y (B,)     x_cond (B,H,W,C)   direction (B,)
        |                |        |              |               |
        v                v        v              |               | 0 = forward
    PatchEmbedding2D  t_embedder  |              |               | 1 = reverse
    (kernel=stride=p) (x1000)     |              |               |
        |                |        |     +--------+--------+      |
        v                |        |     |                 |      |
    + pos_embed          |        |  x_cond * fwd_scale  x_cond  |
    (fixed 2D sin-cos,   |        |     |                 |      |
     ONE table, shared)  |        |     v                 v      |
        |                |        |  cond_embedder_   cond_embedder_
        |                |        |     forward          reverse
        |                |        |     |                 |      |
        |                |        |     +----> where(dir) <------+
        |                |        |              |
        |         t_cond = where(dir, 1, 0)      v
        |                |        |          + pos_embed   (the SAME table)
        |          cond_t_embedder|              |
        |            (x1000)      |              v
        |                |        |          * cond_mask[:, None, None]
        +--> c = (t + t_cond) / 2 + y           |   ONCE, AFTER pos_embed:
                         |                      |   a masked sample's stream is
                         |                      |   the EXACT zero tensor
                         v                      v
                    +----------------------------------+
                    |  depth x DiTXABlock(x, c, cond)  |
                    |  12-way adaLN, msa -> xa -> mlp  |
                    +----------------------------------+
                                     |
                                     v
                      DiTXAFinalLayer: 2-way adaLN,
                      Dense(p*p*out_channels), unpatchify
                                     |
                                     v
                            output (B, H, W, C)

Two upstream details are reproduced rather than repaired, and both are recorded
in ``PORT_NOTES.md``:

* ``c = (t + t_cond) / 2 + y`` -- the average, not ``t + y`` as in plain DiT.
* ``forward_cond_scale`` multiplies the raw ``x_cond`` pixels in the **forward
  direction only**. Upstream's comment justifies it as ``sqrt(4096) = 64``,
  which reconciles with neither the per-token ``token_scale = sqrt(64) = 8``
  that actually builds ``x_cond``'s bridge tensor nor anything else derivable
  from the ``sd`` preset. The CODE is ported; the comment's arithmetic is not
  treated as a derived constant, and the default stays ``1.0``.

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
      plan's ``reference/`` directory -- the line-by-line source of this port.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import numpy as np

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.layers.embedding.class_label_embedding import ClassLabelEmbedding
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.transformers.sd3_adaln import modulate
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

from .blocks import DiTXABlock, DiTXATimestepEmbedder, get_2d_sincos_pos_embed
from .config import PROMPT_NUM_CLASSES

# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

#: Keys :meth:`DiTXA.call` requires of its input dictionary.
REQUIRED_INPUT_KEYS: Tuple[str, ...] = ("x_t", "t", "y", "x_cond", "direction")

#: The one optional key. Absent means "every sample attends to its conditioning".
OPTIONAL_INPUT_KEYS: Tuple[str, ...] = ("cond_mask",)

#: Upstream's ``self.time_scale``. ``t`` lives in ``[0, 1]`` and the sinusoidal
#: basis expects the DDPM-era ``[0, 1000]`` range, so the model -- not the
#: embedder -- multiplies.
DEFAULT_TIME_SCALE: float = 1000.0

#: Number of chunks the final layer's adaLN emits: ``shift`` and ``scale``, no gate.
NUM_FINAL_ADALN_CHUNKS: int = 2


def flattened_linear_xavier(
    fan_in: int, fan_out: int
) -> keras.initializers.Initializer:
    """Xavier-uniform for a conv kernel treated as a **flattened** ``Linear``.

    # DECISION plan-2026-09-02T094601-77d4a04e/D-016
    WHAT NOT TO DO: do not pass ``"glorot_uniform"`` to :class:`PatchEmbedding2D`
    and call it equivalent. Keras computes a convolution's fans over the full
    kernel shape ``(p, p, C_in, D)``, so ``fan_out = p * p * D``; upstream
    reshapes the kernel to ``(D, p * p * C_in)`` first (``dit.py:204-207``), so
    its ``fan_out`` is ``D`` with **no** ``p * p`` factor. At ``p = 2`` that is a
    4x difference in ``fan_out`` and a real difference in the sampled range,
    with no shape symptom and no error. The exact upstream limit is written here
    from statically known integers instead. See decisions.md D-016.

    A **fresh** initializer object is returned on every call: a shared
    :class:`~keras.initializers.Initializer` instance draws bit-identically
    forever, and this model has three patch embedders.

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


def _as_batch_vector(value: Any) -> Any:
    """Drop a trailing singleton axis so ``(B, 1)`` and ``(B,)`` behave alike.

    :param value: A tensor of rank 1 or 2.
    :return: The same tensor with shape ``(B,)``.
    """
    if len(value.shape) > 1 and value.shape[-1] == 1:
        return keras.ops.squeeze(value, axis=-1)
    return value


@register_dl_technique(package="dl_techniques.models.bit_diffusion.model")
class DiTXAFinalLayer(keras.layers.Layer):
    """2-way adaLN, a zero-init projection, and the channels-last unpatchify.

    The projection is zero in weight **and** bias, so a freshly built model
    predicts exactly zero. That is deliberate (adaLN-Zero): combined with the
    zero-init gates in every block it makes the whole stack the zero map at
    step 0, which is what makes a 28-block stack trainable.

    .. code-block:: text

        c (B, hidden)                     x (B, N, hidden)
          |                                 |
        SiLU -> Dense(2*hidden, zeros)      v
          |                            LayerNorm(eps, non-affine)
          +--> split 2 -> shift, scale      |
                          \\_______ modulate ___/
                                    |
                          Dense(p*p*out_channels, zeros)
                                    |
                    reshape (B, h, w, p, p, c)
                    transpose (0, 1, 3, 2, 4, 5)
                    reshape (B, h*p, w*p, c)      <- w*p, not h*p
                                    |
                                    v
                            (B, H, W, out_channels)

    The width dimension is written ``w * p`` on purpose. Upstream writes
    ``h * p`` for both (``dit.py:239``), which is correct only because it also
    asserts ``H == W``; copying the shortcut would make the first non-square
    preset a silent transposition rather than an error.

    :param hidden_size: Model width.
    :type hidden_size: int
    :param patch_size: Patch side; the projection emits ``patch_size ** 2 *
        out_channels`` numbers per token.
    :type patch_size: int
    :param out_channels: Channel count of the reconstructed bridge tensor. Equal
        to ``in_channels`` here, because DiTXA asserts ``learn_sigma=False``.
    :type out_channels: int
    :param grid_height: Patch-grid rows, ``H // patch_size``.
    :type grid_height: int
    :param grid_width: Patch-grid columns, ``W // patch_size``.
    :type grid_width: int
    :param norm_epsilon: Epsilon of the non-affine ``LayerNormalization``.
        Explicit: bare Keras defaults to ``1e-3``.
    :type norm_epsilon: float
    :param kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    :raises ValueError: If any size argument is not positive.

    Input shape:
        ``[x, c]`` -- ``(B, N, hidden_size)`` and ``(B, hidden_size)``, with
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
        # Four SEPARATE Zeros() instances, one per initializer slot. See D-016's
        # sibling rule: an Initializer instance is never shared across layers.
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
        """Build the norm and both Dense layers.

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
        """``(B, N, p*p*C)`` -> ``(B, h*p, w*p, C)``.

        Exposed as a method so a test can drive the interleave directly.

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
        :param training: Forwarded to the Dense sub-layers.
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


@register_dl_technique(package="dl_techniques.models.bit_diffusion.model")
class DiTXA(keras.Model):
    """Bidirectional bridge diffusion transformer with cross-attention.

    See the module docstring for the forward-path diagram and for the two
    upstream oddities this port reproduces deliberately.

    :param input_size: Side of the square bridge tensor. Upstream asserts
        ``H == W``; so does this port, and the fixed positional table is built
        on a square grid.
    :type input_size: int
    :param patch_size: Patch side. Must divide ``input_size``.
    :type patch_size: int
    :param in_channels: Bridge channel count. ``learn_sigma`` is fixed at
        ``False`` upstream for this architecture, so ``out_channels`` equals it.
    :type in_channels: int
    :param hidden_size: Model width. Must be divisible by ``num_heads``.
    :type hidden_size: int
    :param depth: Number of :class:`~...blocks.DiTXABlock` layers.
    :type depth: int
    :param num_heads: Attention heads in both attention sub-layers of a block.
    :type num_heads: int
    :param mlp_ratio: Block MLP expansion factor.
    :type mlp_ratio: float
    :param num_classes: Prompt-kind classes. Default
        :data:`~...config.PROMPT_NUM_CLASSES`.
    :type num_classes: int
    :param class_dropout_rate: CFG label-dropout probability. Any positive value
        adds the extra unconditional row to the label table, which
        ``forward_with_cfg`` needs.
    :type class_dropout_rate: float
    :param forward_cond_scale: Multiplier on the **raw** ``x_cond`` pixels, in
        the **forward direction only**, before patch embedding. Opaque
        hyperparameter; default ``1.0``.
    :type forward_cond_scale: float
    :param time_scale: Multiplier applied to both ``t`` and ``t_cond`` before
        embedding. Upstream's ``1000``.
    :type time_scale: float
    :param drop_path_rate: Maximum stochastic-depth rate; rates ramp linearly
        from ``0`` at block 0. ``0.0`` (the default, and upstream's only
        behaviour) creates no ``StochasticDepth`` sub-layer at all.
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
        sinusoidal basis. Decoupled from ``hidden_size`` on purpose.
    :type frequency_embedding_size: int
    :param label_seed: Seed of the label embedder's own dropout RNG.
    :type label_seed: Optional[int]
    :param kwargs: Standard ``keras.Model`` keyword arguments.

    :raises ValueError: If ``input_size`` is not divisible by ``patch_size``, if
        ``hidden_size`` is not divisible by ``num_heads``, or if any size
        argument is not positive.

    Input shape:
        A dictionary with ``x_t`` ``(B, H, W, C)``, ``t`` ``(B,)`` in ``[0, 1]``,
        ``y`` ``(B,)`` integer labels, ``x_cond`` ``(B, H, W, C)``,
        ``direction`` ``(B,)`` (``0`` forward, ``1`` reverse) and the optional
        ``cond_mask`` ``(B,)`` (``1`` attend, ``0`` zero the stream).

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

    Attributes:
        pos_embed: The single fixed ``(1, num_patches, hidden_size)`` sin-cos
            table, non-trainable, added to ``x`` AND to the conditioning tokens.
        x_embedder, cond_embedder_forward, cond_embedder_reverse: The three
            ``PatchEmbedding2D`` projections.
        t_embedder, cond_t_embedder: Two independent timestep embedders.
        y_embedder: The class-label table with its CFG dropout row.
        blocks: ``depth`` ``DiTXABlock`` layers.
        final_layer: :class:`DiTXAFinalLayer`.
    """

    #: hidden / depth / heads follow ``DiTXA_{S,B,L,XL}/2`` (``dit.py:695-745``);
    #: every shipped variant is patch size 2 (D-003). ``tiny`` exists so a test
    #: can round-trip a real model in seconds, and rides the ``tiny`` bridge
    #: preset's ``(8, 8, 4)`` geometry rather than the ``sd`` preset's.
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

        # -- CREATE sub-layers -------------------------------------------
        # Every `flattened_linear_xavier(...)` call below returns a FRESH
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
        self.t_embedder = DiTXATimestepEmbedder(
            hidden_size=self.hidden_size,
            frequency_embedding_size=self.frequency_embedding_size,
            name="t_embedder",
        )
        self.cond_t_embedder = DiTXATimestepEmbedder(
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

        # DECISION plan-2026-09-02T094601-77d4a04e/D-017
        # WHAT NOT TO DO: do not wrap the block's OUTPUT in StochasticDepth.
        # `DiTXABlock` performs its own three residual adds internally, so
        # `drop_path(block(x))` would drop the residual stream itself and zero
        # the whole activation for that sample -- not the block's contribution.
        # The delta form below drops exactly the contribution, and at rate 0
        # (upstream's only behaviour, and this model's default) NO layer is
        # created at all, so `call()` returns the block output untouched rather
        # than `x + (out - x)`. See decisions.md D-017.
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

    # -- shape plumbing ---------------------------------------------------

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
        """Materialize the positional table and every sub-layer ``call`` runs.

        Both conditioning embedders are built unconditionally: ``direction`` is
        a per-sample tensor, so both of them run on every forward pass (D-005),
        and a ``build()`` that materialized only one would produce a weight tree
        that does not match the traced graph.

        :param input_shape: A dict of input shapes.
        :type input_shape: Any
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

        # The fixed 2D sin-cos table, computed with NumPy and installed through a
        # Constant initializer. NEVER a plain tensor attribute (it would not
        # survive a `.keras` round trip) and NEVER `add_weight(...).assign(...)`
        # inside build() (StatelessScope discards the assign and leaves the table
        # all zeros in every real model, with no shape symptom).
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

    # -- forward path -----------------------------------------------------

    def _is_reverse(self, direction: Any) -> Any:
        """``(B,)`` direction flag -> a ``(B, 1, 1)`` boolean selector.

        :param direction: ``0`` forward, ``1`` reverse. Float or integer.
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
        """Build the conditioning token stream, masked to exact zeros.

        Both conditioning embedders run; ``direction`` selects. The mask is
        applied **after** ``pos_embed`` is added, once, upstream of every block
        (``dit.py:533-542``) -- masking before it would leave a masked sample
        carrying the positional signal instead of the exact zero tensor, with no
        shape symptom and a plausible-looking output.

        Exposed as a method so a test can assert the zeros directly rather than
        inferring them from the model's output.

        :param x_cond: ``(B, H, W, C)`` conditioning bridge tensor.
        :param direction: ``(B,)`` per-sample direction flag.
        :param cond_mask: Optional ``(B,)`` mask; ``None`` means all-ones.
        :param training: Forwarded to the patch embedders.
        :return: ``(B, num_patches, hidden_size)`` conditioning tokens.
        """
        is_reverse = self._is_reverse(direction)

        # The forward branch scales the RAW pixels; the reverse branch does not.
        # Selecting between two already-embedded streams keeps that asymmetry
        # structural: there is no code path on which the reverse embedder sees a
        # scaled input.
        forward_tokens = self.cond_embedder_forward(
            x_cond * self.forward_cond_scale, training=training
        )
        reverse_tokens = self.cond_embedder_reverse(x_cond, training=training)
        cond_tokens = keras.ops.where(is_reverse, reverse_tokens, forward_tokens)

        cond_tokens = cond_tokens + keras.ops.cast(self.pos_embed, cond_tokens.dtype)

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
            ``direction`` and optionally ``cond_mask``.
        :type inputs: Dict[str, Any]
        :param training: Threaded to the label dropout, the block dropouts and
            stochastic depth.
        :type training: Optional[bool]
        :return: ``(B, H, W, out_channels)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If a required key is missing.
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
        # The AVERAGE of the two timestep embeddings, not their sum, and not
        # `t + y` as in plain DiT (dit.py:531).
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
        """Classifier-free guidance, in upstream's **non-standard** algebra.

        Two forward passes: one with the conditioning stream intact, one with
        ``cond_mask`` all-false so every conditioning token is the exact zero
        tensor (the branch :meth:`_embed_conditioning` guards). Then::

            cond + cfg_scale * (cond - uncond)

        **This is NOT the DiT paper's formula.** The textbook form is
        ``uncond + s * (cond - uncond)``; upstream (``dit.py:584``) starts from
        ``cond``, which is the textbook formula evaluated at ``s + 1``. The two
        differ by exactly one unit of guidance at every ``s``, so a "corrected"
        implementation reproduces none of the reference results at any published
        ``cfg_scale`` -- and produces perfectly plausible images while doing it.
        Ported as-written, deliberately; recorded in ``PORT_NOTES.md`` and
        pinned by ``test_the_cfg_formula_is_the_nonstandard_one.py``.

        Note the consequence at ``s = 0``: this formula returns ``cond``
        unchanged, so guidance is off. The sampler still gates on
        ``cfg_scale > 0`` before calling here (``BridgeSDE._evaluate_score``),
        because the gate saves the second forward pass, not because the value
        would differ.

        Any ``cond_mask`` present in ``inputs`` is ignored: the conditional pass
        is defined as the fully-unmasked one.

        :param inputs: The same dict :meth:`call` takes.
        :type inputs: Dict[str, Any]
        :param cfg_scale: Guidance strength ``s``.
        :type cfg_scale: float
        :param training: Forwarded to both passes.
        :type training: Optional[bool]
        :return: ``(B, H, W, out_channels)``.
        :rtype: keras.KerasTensor
        """
        cond_inputs = {k: v for k, v in inputs.items() if k != "cond_mask"}
        # The all-false mask is derived from `t`, which is `(B,)` by contract, so
        # no static batch size is needed and this stays graph-safe.
        uncond_inputs = dict(cond_inputs)
        uncond_inputs["cond_mask"] = keras.ops.zeros_like(
            _as_batch_vector(inputs["t"])
        )

        cond = self(cond_inputs, training=training)
        uncond = self(uncond_inputs, training=training)
        # DECISION plan-2026-09-02T094601-77d4a04e/D-018
        # Do NOT change this to the textbook `uncond + cfg_scale * (cond - uncond)`.
        # It is not a typo upstream and it is not a bug: it is the formula the
        # reference checkpoints and every published `cfg_scale` value were tuned
        # against, and it equals the textbook form at `cfg_scale + 1`. Nothing in
        # a shape, dtype, finiteness or round-trip test can tell the two apart.
        # See decisions.md D-018.
        return cond + cfg_scale * (cond - uncond)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Return ``(B, H, W, out_channels)`` without building anything.

        :param input_shape: A dict of input shapes.
        :type input_shape: Any
        :return: The output bridge shape.
        :rtype: Tuple[Optional[int], ...]
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
    def _download_weights(cls, variant: str) -> str:
        """Pretrained-weights stub. Always raises.

        :param variant: The requested variant name.
        :type variant: str
        :return: Never returns.
        :rtype: str
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained DiTXA weights ship with dl_techniques (variant "
            f"'{variant}'). This is an architecture-faithful port, not a "
            f"weight-compatible one. Use "
            f"DiTXA.from_variant('{variant}', pretrained=False) and then "
            f"model.load_weights('/path/to/weights.keras')."
        )

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
        :param pretrained: ``False`` (default) for random init. A string is
            treated as a local ``.keras`` weights path and loaded after the
            model is built. ``True`` raises ``NotImplementedError``.
        :type pretrained: Union[bool, str]
        :param kwargs: Overrides applied on top of the variant's entry.
        :type kwargs: Any
        :return: The configured model.
        :rtype: DiTXA
        :raises ValueError: If ``variant`` is not a known key.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        if pretrained is True:
            # Raised BEFORE anything is allocated, so a caller who asked for
            # trained weights never receives a randomly initialized model.
            cls._download_weights(variant)

        config = dict(cls.MODEL_VARIANTS[variant])
        description = config.pop("description", "")
        config.update(kwargs)

        logger.info("Creating DiTXA-%s: %s", variant, description)
        model = cls(**config)

        if isinstance(pretrained, str):
            model.load_weights(pretrained)
            logger.info("Loaded DiTXA weights from %s", pretrained)
        return model


def create_ditxa(
    variant: str = "S",
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> DiTXA:
    """Module-level thin factory over :meth:`DiTXA.from_variant`.

    :param variant: One of ``"tiny"``, ``"S"``, ``"B"``, ``"L"``, ``"XL"``.
    :type variant: str
    :param pretrained: ``False`` for random init, a local ``.keras`` path to
        load, or ``True`` to raise ``NotImplementedError``.
    :type pretrained: Union[bool, str]
    :param kwargs: Overrides forwarded to the constructor.
    :type kwargs: Any
    :return: The configured model.
    :rtype: DiTXA
    :raises ValueError: If ``variant`` is not a known key.
    :raises NotImplementedError: If ``pretrained`` is ``True``.

    Example:
        >>> model = create_ditxa("tiny")
    """
    return DiTXA.from_variant(variant, pretrained=pretrained, **kwargs)

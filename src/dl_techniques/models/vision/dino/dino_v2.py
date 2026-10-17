"""
DINOv2 vision transformer with register tokens, LayerScale, and an iBOT mask input.

DINOv2 keeps DINOv1's label-free self-distillation and adds three things to
hold up at scale on curated data. A patch-level objective from iBOT joins the
image-level DINO one: some patch embeddings are replaced by a learnable mask
token before the trunk runs, and the student must predict the teacher's
output for exactly those positions, pushing the representation to carry
local detail the [CLS] objective alone never asks for. A KoLeo regularizer
spreads features within a batch by penalizing nearest-neighbour crowding.
Register tokens, extra learnable tokens carrying no positional signal and
discarded at the output, give the network somewhere to park global
image-wide summaries that would otherwise show up as high-norm artifacts in
ordinary patch tokens.

This file is architecture only. The DINO, iBOT and KoLeo losses are
`DINOLoss`, `iBOTPatchLoss` and `KoLeoLoss` in `dl_techniques.losses.dino_loss`;
the teacher EMA is in `dl_techniques.models.vision.dino.training`.
Sinkhorn-Knopp centering is not implemented anywhere in this repository
(`DINOLoss` offers EMA centering only).

The backbone takes a 2-element list `[images, masks]`, where `masks` is a
boolean `(batch, num_patches)` tensor marking the iBOT positions; pass an
all-`False` mask to disable masking. Its output is always the same 5-key
dictionary (`x_norm_clstoken`, `x_norm_regtokens`, `x_norm_patchtokens`,
`x_prenorm`, `masks`) regardless of `training`.
`get_intermediate_layers(images, n)` returns the patch tokens of the last
`n` blocks, each through the final norm (the reference method of the same
name, `norm=True`), without the mask input. The positional table is
applied before register tokens are concatenated, so registers carry no
positional signal at all, matching their definition rather than an off-by-R
bug. By default the table is sized `num_patches + 1` for `image_size`; with
`pos_embed_grid_size=M` it is the reference's square `1 + M * M` table,
resampled to the input's patch grid as the reference does it
(`DINOv2PositionEmbedding`: torch's bicubic kernel, `interpolate_offset`,
no anti-aliasing). LayerScale gains are unconstrained, as in the reference.
`patch_size=None` resolves to 14 for every v2 variant.

`pretrained=True` loads the official weights of `small`, `base` and
`large` (`facebook/dinov2-{small,base,large}`, the checkpoints Depth
Anything starts from) from a local download, strictly
(`official_weights.py`): every official tensor is used once with a matching
shape and every backbone weight is filled, or the load raises. `tiny` and
`giant` have no wired official weights and raise `NotImplementedError`.

References:
    - Oquab et al., 2023. DINOv2: Learning Robust Visual Features without
      Supervision. (https://arxiv.org/abs/2304.07193)
    - Caron et al., 2021. Emerging Properties in Self-Supervised Vision
      Transformers. (https://arxiv.org/abs/2104.14294)
    - Zhou et al., 2021. iBOT: Image BERT Pre-Training with Online Tokenizer.
      (https://arxiv.org/abs/2111.07832)
    - Darcet et al., 2023. Vision Transformers Need Registers.
      (https://arxiv.org/abs/2309.16588)
    - Sablayrolles et al., 2018. Spreading Vectors for Similarity Search (the
      KoLeo regularizer). (https://arxiv.org/abs/1806.03198)
    - Touvron et al., 2021. Going Deeper with Image Transformers (LayerScale).
      (https://arxiv.org/abs/2103.17239)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
    - Shazeer, 2020. GLU Variants Improve Transformer (the giant variant's SwiGLU
      FFN). (https://arxiv.org/abs/2002.05202)
"""

import math
import keras
import numpy as np
from keras import layers, initializers
from typing import Optional, Union, Dict, Any, Tuple, Literal, List

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.layers.embedding.mask_token import MaskTokenApply
from dl_techniques.layers.embedding.register_tokens import RegisterTokens
from dl_techniques.layers.attention import create_attention_layer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.layers.regularization.layer_scale import LayerScale
from dl_techniques.models.vision.dino.common import reject_input_shape
from dl_techniques.models.vision.dino.official_weights import (
    OFFICIAL_DINOV2_POS_GRID,
    official_dinov2_weights_path,
    load_official_dinov2_file,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.model_build import refuse_unknown_model_kwargs

# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

# DINOv2's patch size when the caller does not specify one. `MODEL_VARIANTS`
# here carries no per-variant `patch_size`, so `create_dino_v2(patch_size=None)`
# resolves to this for EVERY variant. Oquab et al. 2023 use ViT-L/14 and ViT-g/14.
_DEFAULT_PATCH_SIZE = 14

# The cubic convolution constant of torch's `upsample_bicubic2d`
# (`aten/src/ATen/native/UpSample.h`, `get_cubic_upsample_coefficients`).
_TORCH_BICUBIC_A = -0.75


def torch_bicubic_resize_matrix(
        in_size: int,
        out_size: int,
        scale_factor: Optional[float] = None,
) -> np.ndarray:
    """The ``(out_size, in_size)`` matrix of torch's 1-D bicubic resampling.

    Interface contract (callers: `DINOv2PositionEmbedding`, once per axis):
    ``matrix @ v`` equals ``torch.nn.functional.interpolate(v, mode="bicubic",
    align_corners=False, antialias=False)`` along one axis, given
    ``scale_factor`` (the reference's offset form, where torch maps output
    index ``o`` to source ``(o + 0.5) / scale_factor - 0.5``) or, when it is
    ``None``, the output size alone (source ``(o + 0.5) * in / out - 0.5``).
    Taps use the cubic convolution kernel with ``A = -0.75`` and read
    clamped indices at the borders, as torch does. Pure numpy, float64;
    raises ``ValueError`` for a non-positive size or scale.

    :param in_size: Length of the source axis.
    :type in_size: int
    :param out_size: Length of the resampled axis.
    :type out_size: int
    :param scale_factor: torch's ``scale_factor`` for this axis, or ``None``.
    :type scale_factor: Optional[float]
    :return: The resampling matrix, rows summing to 1.
    :rtype: np.ndarray
    """
    if in_size < 1 or out_size < 1 or (scale_factor is not None and scale_factor <= 0):
        raise ValueError(
            f"sizes must be >= 1 and scale_factor > 0; got {in_size}, {out_size}, {scale_factor}")
    a = _TORCH_BICUBIC_A
    step = 1.0 / scale_factor if scale_factor is not None else in_size / out_size

    def near(t: float) -> float:  # |t| <= 1
        return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0

    def far(t: float) -> float:  # 1 < |t| < 2
        return ((a * t - 5.0 * a) * t + 8.0 * a) * t - 4.0 * a

    matrix = np.zeros((out_size, in_size), dtype=np.float64)
    for o in range(out_size):
        source = step * (o + 0.5) - 0.5
        base = math.floor(source)
        t = source - base
        for k, weight in enumerate((far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t))):
            matrix[o, min(max(base - 1 + k, 0), in_size - 1)] += weight
    return matrix


def _require_official_structure(
        config: Dict[str, Any],
        patch_size: Union[int, Tuple[int, int]],
        num_register_tokens: int,
        init_values: Optional[float],
) -> None:
    """Refuse a ``pretrained=True`` backbone whose structure the checkpoint lacks.

    Interface contract (callers: both ``from_variant`` methods): raises
    ``ValueError`` naming the first argument that differs from the official
    DINOv2 ViT-S/B/L/14 structure; returns ``None`` otherwise. Reads the merged
    variant + caller ``config`` and the three explicit arguments only.

    :param config: The variant's entry updated with the caller's kwargs.
    :type config: Dict[str, Any]
    :param patch_size: The requested patch size.
    :type patch_size: Union[int, Tuple[int, int]]
    :param num_register_tokens: The requested register-token count.
    :type num_register_tokens: int
    :param init_values: The LayerScale init (``None`` builds no LayerScale).
    :type init_values: Optional[float]
    """
    official = {
        'patch_size': (_DEFAULT_PATCH_SIZE, _DEFAULT_PATCH_SIZE),
        'num_register_tokens': 0,
        'pos_embed_grid_size': OFFICIAL_DINOV2_POS_GRID,
        'ffn_type': 'mlp',
        'attention_type': 'multi_head',
        'normalization_type': 'layer_norm',
        'qkv_bias': True,
        'ffn_bias': True,
        'in_chans': 3,
        'include_top': True,
        'interpolate_antialias': False,
    }
    given = {
        'patch_size': tuple(patch_size) if isinstance(patch_size, (tuple, list))
        else (patch_size, patch_size),
        'num_register_tokens': num_register_tokens,
    }
    for key, value in official.items():
        actual = given.get(key, config.get(key, value))
        if actual != value:
            raise ValueError(
                f"pretrained=True loads the official DINOv2 checkpoint, which has "
                f"{key}={value!r}; got {actual!r}")
    if init_values is None:
        raise ValueError(
            "pretrained=True needs LayerScale (the official blocks have it); "
            "init_values must not be None")


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.dino.dino_v2")
class DINOv2PositionEmbedding(keras.layers.Layer):
    """The reference DINOv2 position table, resampled to the input's patch grid.

    Owns one weight ``pos_embed`` of shape ``(1, 1 + M * M, D)``: a [CLS]
    row and a square ``M x M`` grid, the layout of every official DINOv2
    checkpoint (``M = 37``, from 518 px at patch 14). For an input grid
    ``(h, w)`` it adds the [CLS] row to the [CLS] token and the grid,
    resampled to ``(h, w)``, to the patch tokens, exactly as the reference's
    ``interpolate_pos_encoding``: the table is used as is when
    ``h == w == M``, and otherwise every axis is resampled with torch's
    bicubic kernel at ``scale_factor = (n + interpolate_offset) / M``
    (``interpolate_offset = 0`` resamples to the output size instead, the
    reference's other branch). The resampling is a fixed matrix per axis
    (`torch_bicubic_resize_matrix`), so the gradient reaches the stored grid.

    .. code-block:: text

        pos_embed (1, 1 + M*M, D) ──► [CLS] row ─────────────────────┐
                    └─► grid (M, M, D) ─► W_h · grid · W_wᵀ (h, w, D) ┤
        tokens (B, 1 + h*w, D) ──────────────────────────────────────(+)──► out

    :param grid_size: ``M``, the side of the stored square grid.
    :type grid_size: int
    :param output_grid: ``(h, w)``, the input's patch grid.
    :type output_grid: Tuple[int, int]
    :param interpolate_offset: The reference's offset; ``0`` means "resample
        to the output size".
    :type interpolate_offset: float
    :param initializer: Initializer of the table; ``None`` is the
        reference's ``trunc_normal_(std=0.02)``.
    :type initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        ``(batch, 1 + h * w, D)``: [CLS] then the patch tokens, row-major.

    Output shape:
        Same as the input.
    """

    def __init__(
            self,
            grid_size: int,
            output_grid: Tuple[int, int],
            interpolate_offset: float = 0.1,
            initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        output_grid = tuple(int(n) for n in output_grid)
        if int(grid_size) < 1 or len(output_grid) != 2 or min(output_grid) < 1:
            raise ValueError(
                f"grid_size must be >= 1 and output_grid two sizes >= 1; got {grid_size}, {output_grid}")
        if interpolate_offset < 0:
            raise ValueError(f"interpolate_offset must be >= 0, got {interpolate_offset}")
        self.grid_size = int(grid_size)
        self.output_grid = output_grid
        self.interpolate_offset = float(interpolate_offset)
        # A fresh instance per layer: a shared default instance is one RNG stream
        # for every table (tests/test_the_shared_initializer_population_cannot_grow.py).
        self.initializer = (
            initializers.TruncatedNormal(stddev=0.02) if initializer is None
            else keras.initializers.get(initializer))
        m = self.grid_size
        self.is_identity = output_grid == (m, m)
        self._resample = [
            torch_bicubic_resize_matrix(
                m, n, (n + self.interpolate_offset) / m if self.interpolate_offset else None
            ).astype("float32")
            for n in output_grid
        ]
        self.pos_embed = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the ``(1, 1 + M * M, D)`` table."""
        h, w = self.output_grid
        if input_shape[1] is not None and input_shape[1] != 1 + h * w:
            raise ValueError(
                f"expected 1 + {h} * {w} tokens for output_grid {self.output_grid}, "
                f"got input shape {input_shape}")
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, 1 + self.grid_size * self.grid_size, input_shape[-1]),
            initializer=self.initializer,
            trainable=True,
        )
        super().build(input_shape)

    def resampled_table(self) -> keras.KerasTensor:
        """The ``(1, 1 + h * w, D)`` table this layer adds, in float32."""
        table = keras.ops.convert_to_tensor(self.pos_embed)
        if self.is_identity:
            return table
        m, (h, w) = self.grid_size, self.output_grid
        dim = table.shape[-1]
        grid = keras.ops.reshape(table[:, 1:], (m, m * dim))
        grid = keras.ops.matmul(self._resample[0], grid)                      # (h, M*D)
        grid = keras.ops.reshape(grid, (h, m, dim))
        grid = keras.ops.einsum("wj,hjd->hwd", self._resample[1], grid)       # (h, w, D)
        return keras.ops.concatenate(
            [table[:, :1], keras.ops.reshape(grid, (1, h * w, dim))], axis=1)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Add the resampled table to the tokens."""
        return inputs + keras.ops.cast(self.resampled_table(), inputs.dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Same as the input."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Every constructor argument."""
        config = super().get_config()
        config.update({
            "grid_size": self.grid_size,
            "output_grid": list(self.output_grid),
            "interpolate_offset": self.interpolate_offset,
            "initializer": keras.initializers.serialize(self.initializer),
        })
        return config


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.dino.dino_v2")
class DINOv2Block(keras.layers.Layer):
    """Pre-norm transformer block with LayerScale gains on both branches.

    Architecture:

    .. code-block:: text

        Input x [B, N, D] ────────────────────────────┐
              │                                       │
              ▼                                       │
        LayerNorm → Attention → LayerScale → DropPath │
              │                                       │
              ▼                                       │
             (+) ◄────────────────────────────────────┘
              │
              ▼
        LayerNorm → FFN → LayerScale → DropPath ──┐
              │                                   │
              ▼                                   │
             (+) ◄────────────────────────────────┘
              │
              ▼
        Output [B, N, D]

    Attention, FFN and normalization are each selected by string through the
    shared factories. `drop_path` is one `StochasticDepth` instance called
    twice (once per branch): the layer draws a fresh mask per call and holds
    no seed state or variables, so the two branches still get independent
    masks (measured: 40 successive calls produced 0 identical mask pairs out
    of 780). See decisions.md D-014.

    :param dim: Embedding dimension. Must be positive and divisible by `num_heads`.
    :type dim: int
    :param num_heads: Number of attention heads. Must be positive.
    :type num_heads: int
    :param mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
    :type mlp_ratio: float
    :param attention_type: Attention mechanism, a key of `ATTENTION_REGISTRY`
        (`'multi_head_attention'` is not one; use `'multi_head'`).
    :type attention_type: str
    :param ffn_type: FFN type, e.g. ``"mlp"``, ``"swiglu"``.
    :type ffn_type: str
    :param normalization_type: Normalization type, e.g. ``"layer_norm"``, ``"rms_norm"``.
    :type normalization_type: str
    :param qkv_bias: Whether to use bias in the QKV projection.
    :type qkv_bias: bool
    :param proj_bias: Has no effect. `build` maps `qkv_bias` onto the
        attention layer's `use_bias` and never consults `proj_bias`; no
        attention type in `ATTENTION_REGISTRY` separates the output
        projection's bias from the QKV one. Kept for config compatibility.
    :type proj_bias: bool
    :param ffn_bias: Whether to use bias in FFN layers.
    :type ffn_bias: bool
    :param stochastic_depth_rate: Stochastic depth drop probability.
    :type stochastic_depth_rate: float
    :param init_values: LayerScale initialization value; ``None`` disables scaling.
    :type init_values: Optional[float]
    :param attention_dropout_rate: Dropout rate for attention.
    :type attention_dropout_rate: float
    :param ffn_dropout_rate: Dropout rate for FFN.
    :type ffn_dropout_rate: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        3D tensor ``(batch_size, sequence_length, embedding_dim)``.

    Output shape:
        3D tensor ``(batch_size, sequence_length, embedding_dim)``, same
        shape as input due to residual connections.

    Example:
        .. code-block:: python

            # Standard DINOv2 block
            block = DINOv2Block(
                dim=768, num_heads=12, mlp_ratio=4.0,
                stochastic_depth_rate=0.1, init_values=1e-5
            )

            # With SwiGLU FFN (for giant model)
            block = DINOv2Block(
                dim=1536, num_heads=24, ffn_type='swiglu', stochastic_depth_rate=0.3
            )
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            attention_type: str = 'multi_head',
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            qkv_bias: bool = True,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            stochastic_depth_rate: float = 0.0,
            init_values: Optional[float] = None,
            attention_dropout_rate: float = 0.0,
            ffn_dropout_rate: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        # Store all configuration
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.ffn_bias = ffn_bias
        self.stochastic_depth_rate = stochastic_depth_rate
        self.init_values = init_values
        self.attention_dropout_rate = attention_dropout_rate
        self.ffn_dropout_rate = ffn_dropout_rate

        # Create sub-layers in __init__ following Modern Keras 3 patterns

        # Normalization layers
        self.norm1 = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm1"
        )
        self.norm2 = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm2"
        )

        # Attention layer - map parameters appropriately
        attention_args = {
            'num_heads': self.num_heads,
            'dropout_rate': self.attention_dropout_rate
        }

        if self.attention_type == 'multi_head':
            attention_args['dim'] = self.dim
            attention_args['use_bias'] = self.qkv_bias
        else:
            attention_args['dim'] = self.dim

        self.attention = create_attention_layer(
            attention_type=self.attention_type,
            name="attention",
            **attention_args
        )

        # FFN layer
        hidden_dim = int(self.dim * self.mlp_ratio)
        ffn_args = {
            'output_dim': self.dim,
            'dropout_rate': self.ffn_dropout_rate,
            'use_bias': self.ffn_bias
        }

        if self.ffn_type in ['mlp', 'glu', 'geglu']:
            ffn_args['hidden_dim'] = hidden_dim
        elif self.ffn_type in ['swiglu']:
            ffn_args['ffn_expansion_factor'] = self.mlp_ratio

        self.ffn = create_ffn_layer(
            ffn_type=self.ffn_type,
            name="ffn",
            **ffn_args
        )

        # LayerScale gains for the two residual branches
        # DECISION plan-2026-10-14T062616-81938e81/D-134
        # Do NOT constrain these gains to be non-negative (LayerScale's default):
        # the reference's `gamma` is a free parameter, and about half the
        # entries of every official checkpoint are negative (small: 4441 of
        # 9216, min -10.2). A `non_neg` constraint clips them all to 0 on the
        # first optimizer step, which deletes the pretrained residual branches.
        if self.init_values is not None:
            self.ls1 = LayerScale(
                multiplier_type='CHANNEL',
                initializer=initializers.Constant(self.init_values),
                constraint=None,
                name="ls1"
            )
            self.ls2 = LayerScale(
                multiplier_type='CHANNEL',
                initializer=initializers.Constant(self.init_values),
                constraint=None,
                name="ls2"
            )
        else:
            self.ls1 = None
            self.ls2 = None

        # Stochastic depth (optional)
        # DECISION plan-2026-08-01T105809-dc0c402e/D-014: share one StochasticDepth instance between the attention and FFN branches.
        # Measured equivalent to two instances (0/780 identical mask pairs over 40 calls); splitting only adds a serialized sub-layer. See decisions.md.
        if self.stochastic_depth_rate > 0.0:
            self.drop_path = StochasticDepth(self.stochastic_depth_rate, name="drop_path")
        else:
            self.drop_path = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the transformer block with all sub-components.

        Explicitly builds all sub-layers for proper serialization.
        Following the modern Keras 3 pattern from the refined guide.
        """
        logger.debug(f"Building DINOv2Block with input_shape: {input_shape}")

        # Build normalization layers
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        # Build attention layer
        self.attention.build(input_shape)

        # Build FFN layer
        self.ffn.build(input_shape)

        # Build LayerScale layers if used
        if self.ls1 is not None:
            self.ls1.build(input_shape)
            self.ls2.build(input_shape)

        # Build stochastic depth if used
        if self.drop_path is not None:
            self.drop_path.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the transformer block."""
        # Pre-norm attention block
        x = self.norm1(inputs, training=training)
        x = self.attention(x, training=training)

        if self.ls1 is not None:
            x = self.ls1(x, training=training)

        if self.drop_path is not None:
            x = self.drop_path(x, training=training)

        x = inputs + x

        # Pre-norm FFN block
        y = self.norm2(x, training=training)
        y = self.ffn(y, training=training)

        if self.ls2 is not None:
            y = self.ls2(y, training=training)

        if self.drop_path is not None:
            y = self.drop_path(y, training=training)

        x = x + y

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the layer."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "normalization_type": self.normalization_type,
            "qkv_bias": self.qkv_bias,
            "proj_bias": self.proj_bias,
            "ffn_bias": self.ffn_bias,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "init_values": self.init_values,
            "attention_dropout_rate": self.attention_dropout_rate,
            "ffn_dropout_rate": self.ffn_dropout_rate,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.dino.dino_v2")
class DINOv2VisionTransformer(keras.Model):
    """
    DINOv2 Vision Transformer backbone implementation following Modern Keras 3 patterns.

    This implementation provides a complete DINOv2 ViT backbone using the functional API
    pattern from the refined guide, similar to the ConvNeXt V1 paradigm. It creates
    the entire model architecture in a _build_model method using keras.Input and
    functional connections.

    Architecture:

    .. code-block:: text

    Input Image (B, H, W, C)
         ↓
    PatchEmbed → (B, N, D)
         ↓
    [CLS] + [REG]₀₋ᴿ + Patches + PosEmbed → (B, 1+R+N, D)
         ↓
    TransformerBlock₁ → ... → TransformerBlockₗ
         ↓
    LayerNorm → Split: [CLS] | [REG] | [Patches]
    ```

    Provides the core Vision Transformer backbone used in DINOv2,
    supporting both self-supervised pre-training and downstream fine-tuning
    with configurable architecture and modern training enhancements.

    Args:
        image_size: Input image size (int or tuple). Must be positive.
        patch_size: Patch size for embedding (int or tuple). Must divide image_size evenly.
        in_chans: Number of input channels. Typically 1 or 3.
        embed_dim: Embedding dimension. Must be positive and divisible by num_heads.
        depth: Number of transformer blocks. Must be positive.
        num_heads: Number of attention heads. Must be positive.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim. Must be positive.
        qkv_bias: Enable bias for QKV projections.
        proj_bias: Has no effect. Forwarded to every `DINOv2Block` and read by
            none of them; see `DINOv2Block`'s Args. The output projection's bias
            follows `qkv_bias`.
        ffn_bias: Enable bias for FFN layers.
        stochastic_depth_rate: Maximum stochastic depth rate.
        drop_path_uniform: Use uniform drop rate across blocks.
        init_values: LayerScale initialization value.
        attention_type: Type of attention mechanism.
        ffn_type: Type of feed-forward network.
        normalization_type: Type of normalization.
        num_register_tokens: Number of register tokens to use.
        interpolate_antialias: Must be False when ``pos_embed_grid_size`` is
            set: the anti-aliased resampling is not implemented (the official
            checkpoints use none). Unread otherwise.
        interpolate_offset: The reference's offset for resampling the
            position table (``scale_factor = (n + offset) / M``; ``0``
            resamples to the output size). Read only when
            ``pos_embed_grid_size`` is set.
        pos_embed_grid_size: ``None`` sizes the position table for
            ``image_size`` (``num_patches + 1`` rows, added as is). An int
            ``M`` stores the reference's square ``1 + M * M`` table and
            resamples it to the input's patch grid
            (`DINOv2PositionEmbedding`); the official checkpoints use 37.
        include_top: Whether to include the final normalization layer.
        input_shape: Input shape. If None, computed from image_size and in_chans.
        **kwargs: Additional keyword arguments for the Model base class.

    Inputs:
        A 2-element list ``[images, masks]``:
            - images: 4D tensor `(batch_size, height, width, channels)`.
            - masks: 2D boolean tensor `(batch_size, num_patches)` (iBOT mask;
              `True` marks patches replaced by the learnable mask token).

    Output:
        Always a dictionary (output structure is config-fixed, not
        training-dependent) with keys:
            - 'x_norm_clstoken': CLS token features (B, D)
            - 'x_norm_regtokens': Register token features (B, R, D)
            - 'x_norm_patchtokens': Patch token features (B, N, D)
            - 'x_prenorm': Pre-normalization features (B, 1+R+N, D)
            - 'masks': Input masks (echoed through an identity op)

    Example:
        ```python
        # Standard DINOv2-Base
        backbone = DINOv2VisionTransformer(
            embed_dim=768,
            depth=12,
            num_heads=12,
            stochastic_depth_rate=0.1
        )

        # With register tokens
        backbone = DINOv2VisionTransformer(
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_register_tokens=4
        )
        ```
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3,
            'mlp_ratio': 4.0,
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'mlp_ratio': 4.0,
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4.0,
        },
        'giant': {
            'embed_dim': 1536,
            'depth': 40,
            'num_heads': 24,
            'mlp_ratio': 4.0,
            'ffn_type': 'swiglu',  # Giant uses SwiGLU by default
        }
    }

    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 14,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            stochastic_depth_rate: float = 0.0,
            drop_path_uniform: bool = False,
            init_values: Optional[float] = None,
            attention_type: str = 'multi_head',
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            num_register_tokens: int = 0,
            interpolate_antialias: bool = False,
            interpolate_offset: float = 0.1,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            pos_embed_grid_size: Optional[int] = None,
            **kwargs
    ) -> None:
        refuse_unknown_model_kwargs(kwargs, type(self).__name__)
        if pos_embed_grid_size is not None and interpolate_antialias:
            raise ValueError(
                "interpolate_antialias=True is not implemented: the position "
                "table is resampled without anti-aliasing, as the official "
                "DINOv2 checkpoints are")
        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if in_chans not in [1, 3]:
            logger.warning(f"Unusual number of input channels: {in_chans}. DINOv2 typically uses 3 channels.")

        # Store configuration
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.ffn_bias = ffn_bias
        self.stochastic_depth_rate = stochastic_depth_rate
        self.drop_path_uniform = drop_path_uniform
        self.init_values = init_values
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.include_top = include_top
        self.pos_embed_grid_size = None if pos_embed_grid_size is None else int(pos_embed_grid_size)

        # Computed attributes
        self.num_features = embed_dim
        self.num_tokens = 1  # CLS token
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        # Set input shape
        # DECISION plan-2026-08-19T163559-499b6f0e/D-115: store the raw `input_shape` argument, never the derived `(*image_size, in_chans)`.
        # `get_config` serializes this field, so a caller override survives a round trip and `None` re-derives identically on reload. See decisions.md.
        self._input_shape_arg = input_shape
        if input_shape is None:
            input_shape = (*self.image_size, self.in_chans)

        # Validate patch size alignment
        if self.image_size[0] % self.patch_size[0] != 0 or self.image_size[1] % self.patch_size[1] != 0:
            raise ValueError(f"image_size {self.image_size} must be divisible by patch_size {self.patch_size}")

        # Initialize layer lists for tracking
        self.transformer_blocks = []
        self.patch_embed = None
        self.pos_embed = None
        self.norm = None

        # DECISION plan_2026-06-15_e2759fbc/D-009: the iBOT mask token is a genuine learnable weight via MaskTokenApply, never a Dense-on-ones hack.
        # A Dense(use_bias=False, kernel_initializer='zeros') applied to ones produced a constant-zero vector at init, degenerating to zeroing masked patches. See decisions.md.
        self.mask_token_layer = MaskTokenApply(name='mask_token')

        # DECISION plan_2026-06-15_e2759fbc/D-002: CLS token is a real learnable token via ClassTokenPrepend, never a Dense-on-ones projection hack.
        # ClassTokenPrepend is safe to assign pre-super (lazy build). See decisions.md.
        self.cls_token_layer = ClassTokenPrepend(name="cls_token")

        # DECISION plan_2026-06-15_e2759fbc/D-003: register tokens are hoisted to __init__ behind a `num_register_tokens>0` guard.
        # See decisions.md.
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-064: RegisterTokens owns a real (1, R, D) add_weight, never a Dense on ones((1, R, 1)).
        # That form gives R bit-identical copies of one vector, D parameters where R*D are needed. See decisions.md.
        self.register_token_layer = None
        if self.num_register_tokens > 0:
            self.register_token_layer = RegisterTokens(
                num_tokens=self.num_register_tokens,
                embed_dim=self.embed_dim,
                initializer=initializers.TruncatedNormal(stddev=1e-6),
                name='register_tokens'
            )

        # Create inputs and build model using functional API
        inputs = keras.Input(shape=input_shape, name="input_images")

        # DECISION plan_2026-06-15_e2759fbc/D-009: the model contract is 2-input [images, masks], never a 3rd `is_training` Input.
        # Once the output structure became always the 5-key dict, nothing in the graph reads `is_training`. See decisions.md.
        masks_input = keras.Input(shape=(self.num_patches,), dtype="bool", name="input_masks")

        # Build the model
        outputs = self._build_model(inputs, masks_input)

        # DECISION plan_2026-06-15_e2759fbc/D-007: honor a caller-supplied name via kwargs.pop, never a hardcoded name= alongside **kwargs.
        # The wrapper also passes name='dinov2_backbone' into **kwargs; a hardcoded name here raised a duplicate-keyword TypeError. See decisions.md.
        name = kwargs.pop('name', f'dinov2_vit_{embed_dim}d_{depth}l')

        # Initialize the Model
        super().__init__(
            inputs=[inputs, masks_input],
            outputs=outputs,
            name=name,
            **kwargs
        )

        logger.info(
            f"Created DINOv2 ViT backbone: {embed_dim}d x {depth}l x {num_heads}h, "
            f"patches {self.num_patches}, register_tokens {num_register_tokens}"
        )

    def _build_model(
            self,
            inputs: keras.KerasTensor,
            masks: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Build the complete DINOv2 Vision Transformer architecture.

        Args:
            inputs: Input image tensor
            masks: Input mask tensor for iBOT objective

        Returns:
            The 5-key output dict (always — output structure is config-fixed).
        """
        # Build patch embedding
        x = self._build_patch_embedding(inputs)

        # Build token preparation with masks
        x = self._build_token_preparation(x, inputs, masks)

        # Build transformer blocks
        x = self._build_transformer_blocks(x)

        # Build final processing
        outputs = self._build_final_processing(x, masks)

        return outputs

    def _build_patch_embedding(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build patch embedding layer."""
        # Create patch embedding using factory
        self.patch_embed = create_embedding_layer(
            'patch_2d',
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name='patch_embed'
        )

        x = self.patch_embed(inputs)
        logger.debug(f"After patch embedding: {x.shape}")
        return x

    def _build_token_preparation(
            self,
            patch_embeddings: keras.KerasTensor,
            original_inputs: keras.KerasTensor,
            masks: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Build token preparation with CLS, register tokens, and positional embeddings."""
        # B1: apply iBOT masking via MaskTokenApply, which owns the learnable (1,1,D)
        # mask-token weight in its build() and performs the elementwise select internally.
        # DECISION plan_2026-06-15_e2759fbc/D-009: mask replacement is a real learnable token via MaskTokenApply, never a Dense-on-ones zero vector.
        # `where(mask, mask_token, patch_emb)` broadcasts the token over the batch. See decisions.md.
        x = self.mask_token_layer([patch_embeddings, masks])

        # B2: prepend a real learnable CLS token (B,N,D)->(B,N+1,D) via ClassTokenPrepend,
        # after masking, before pos-embed.
        # DECISION plan_2026-06-15_e2759fbc/D-002: use ClassTokenPrepend here too, never a Dense-on-ones projection hack.
        # See decisions.md.
        x = self.cls_token_layer(x)

        # B4/B7: positional embeddings. Without a grid size the weight is sized
        # num_patches + num_tokens (CLS) for this image size. With one, it is the
        # reference's square table, resampled to this patch grid.
        if self.pos_embed_grid_size is None:
            pos_embed_seq_len = self.num_patches + self.num_tokens
            self.pos_embed = create_embedding_layer(
                'positional_learned',
                max_seq_len=pos_embed_seq_len,
                dim=self.embed_dim,
                name='pos_embed'
            )
        else:
            # DECISION plan-2026-10-14T062616-81938e81/D-135
            # Do NOT resample with keras.ops.image.resize('bicubic'): it is TF's
            # Keys kernel (A = -0.5) at the output-size scale, while the reference
            # uses torch's kernel (A = -0.75) at scale (n + 0.1) / M, so a loaded
            # official table would land on other values at every grid but 37.
            self.pos_embed = DINOv2PositionEmbedding(
                grid_size=self.pos_embed_grid_size,
                output_grid=(self.image_size[0] // self.patch_size[0],
                             self.image_size[1] // self.patch_size[1]),
                interpolate_offset=self.interpolate_offset,
                name='pos_embed'
            )
        # DECISION plan_2026-06-15_e2759fbc/D-004: call as `x = self.pos_embed(x)`, never a nested Lambda or `_get_interpolated_pos_embed`.
        # A nested Lambda-in-Lambda is illegal layer creation in trace, and variable-resolution interpolation is out of scope. See decisions.md.
        x = self.pos_embed(x)

        # B3: insert register tokens after CLS via Concatenate (not a Lambda). Cold path on
        # the smoke (num_register_tokens=0); 'large'/'giant' auto-enable 4 registers.
        # DECISION plan_2026-06-15_e2759fbc/D-003: build register tokens via the hoisted layer and insert with Concatenate, never a Lambda.
        # See decisions.md.
        # DECISION plan_2026-06-15_e2759fbc/D-009: insert register tokens after positional embedding is applied, never before.
        # Registers are defined position-free (Darcet et al., 2023); moving insertion earlier would wrongly give them a positional signal. See decisions.md.
        if self.num_register_tokens > 0:
            # RegisterTokens reads only the batch size of `x` and emits the
            # (B, R, D) bank, so no Lambda broadcast is needed any more.
            reg_tokens = self.register_token_layer(x)
            cls = x[:, :1]
            rest = x[:, 1:]
            x = layers.Concatenate(axis=1, name='add_register_tokens')([cls, reg_tokens, rest])

        return x

    def _build_transformer_blocks(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build transformer blocks with stochastic depth."""
        # Calculate drop path rates
        if self.drop_path_uniform:
            dpr = [self.stochastic_depth_rate] * self.depth
        else:
            dpr = linear_drop_path_rates(self.depth, self.stochastic_depth_rate)

        # Create transformer blocks
        for i in range(self.depth):
            block = DINOv2Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attention_type=self.attention_type,
                ffn_type=self.ffn_type,
                normalization_type=self.normalization_type,
                qkv_bias=self.qkv_bias,
                proj_bias=self.proj_bias,
                ffn_bias=self.ffn_bias,
                stochastic_depth_rate=dpr[i],
                init_values=self.init_values,
                name=f"block_{i}"
            )
            x = block(x)
            self.transformer_blocks.append(block)

        return x

    def _build_final_processing(
            self,
            x: keras.KerasTensor,
            masks: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Build final normalization and output processing."""
        x_prenorm = x  # Store pre-normalization features

        # Final normalization
        if self.include_top:
            self.norm = create_normalization_layer(
                normalization_type=self.normalization_type,
                name="norm"
            )
            x_norm = self.norm(x)
        else:
            x_norm = x

        # DECISION plan_2026-06-15_e2759fbc/D-005: always return the 5-key dict, never branch on `is_training` via keras.ops.cond.
        # A training-dependent output structure is rejected by ops.cond; each key is produced by its own shape-inferable Lambda instead of one dict-returning Lambda. See decisions.md.
        num_reg = self.num_register_tokens

        cls_token = layers.Lambda(
            lambda t: t[:, 0], name='slice_cls_token'
        )(x_norm)

        if num_reg > 0:
            reg_tokens = layers.Lambda(
                lambda t: t[:, 1:1 + num_reg], name='slice_reg_tokens'
            )(x_norm)
            patch_tokens = layers.Lambda(
                lambda t: t[:, 1 + num_reg:], name='slice_patch_tokens'
            )(x_norm)
        else:
            reg_tokens = layers.Lambda(
                lambda t: t[:, 0:0], name='slice_reg_tokens'
            )(x_norm)
            patch_tokens = layers.Lambda(
                lambda t: t[:, 1:], name='slice_patch_tokens'
            )(x_norm)

        # DECISION plan_2026-06-15_e2759fbc/D-008: route `masks` through an identity Lambda before echoing it as an output, never the bare input tensor.
        # A bare input-as-output makes it both an input and output of this Functional model, which raises a "part of a cycle" error once the wrapper nests it. See decisions.md.
        masks_out = layers.Lambda(lambda t: t, name='masks_passthrough')(masks)

        outputs = {
            "x_norm_clstoken": cls_token,
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": x_prenorm,
            "masks": masks_out,
        }

        return outputs

    @classmethod
    def from_variant(
            cls,
            variant: Literal['tiny', 'small', 'base', 'large', 'giant'],
            image_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 14,
            num_register_tokens: int = 0,
            init_values: Optional[float] = 1e-5,
            stochastic_depth_rate: float = 0.0,
            input_shape: Optional[Tuple[int, ...]] = None,
            pretrained: bool = False,
            **kwargs
    ) -> "DINOv2VisionTransformer":
        """
        Create DINOv2 Vision Transformer from predefined variant.

        Args:
            variant: Size variant ('tiny', 'small', 'base', 'large', 'giant').
            image_size: Input image size; a multiple of ``patch_size``.
            patch_size: Patch size for patch embedding.
            num_register_tokens: Number of register tokens to use.
            init_values: LayerScale initialization value.
            stochastic_depth_rate: Maximum stochastic depth rate.
            input_shape: Input shape. If None, computed from image_size.
            pretrained: ``True`` loads the official weights of ``small``,
                ``base`` or ``large`` from their local download
                (`official_weights.py`), strictly, after building the
                official structure: ``pos_embed_grid_size`` defaults to 37
                and the table is resampled to ``image_size``.
            **kwargs: Additional arguments for the model.

        Returns:
            DINOv2VisionTransformer instance.

        Raises:
            ValueError: If variant is not recognized, or if ``pretrained``
                is combined with a structure the official checkpoint does
                not have.
            NotImplementedError: If ``pretrained`` is True for a variant
                without wired official weights (``tiny``, ``giant``).
            FileNotFoundError: If ``pretrained`` is True and the download
                is absent (checked before anything is built).
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(cls.MODEL_VARIANTS.keys())}")

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        if pretrained:
            # GUIDE section 1: resolve the weights first (raise before building),
            # build the official structure, then load strictly.
            official_dinov2_weights_path(variant)
            config.setdefault('pos_embed_grid_size', OFFICIAL_DINOV2_POS_GRID)
            _require_official_structure(
                config, patch_size=patch_size, num_register_tokens=num_register_tokens,
                init_values=init_values)

        logger.info(f"Creating DINOv2-{variant.upper()} Vision Transformer")
        logger.info(f"Configuration: {config}")

        model = cls(
            image_size=image_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            init_values=init_values,
            stochastic_depth_rate=stochastic_depth_rate,
            input_shape=input_shape,
            **config
        )
        if pretrained:
            load_official_dinov2_file(model, variant)
        return model

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        # DECISION plan-2026-08-19T163559-499b6f0e/D-082: call `super().get_config()` first, never a literal dict.
        # Without it, `name` and `trainable` reload at their defaults, so a frozen model comes back unfrozen. See decisions.md.
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'in_chans': self.in_chans,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'proj_bias': self.proj_bias,
            'ffn_bias': self.ffn_bias,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'drop_path_uniform': self.drop_path_uniform,
            'init_values': self.init_values,
            'attention_type': self.attention_type,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'num_register_tokens': self.num_register_tokens,
            'interpolate_antialias': self.interpolate_antialias,
            'interpolate_offset': self.interpolate_offset,
            'include_top': self.include_top,
            'input_shape': self._input_shape_arg,
            'pos_embed_grid_size': self.pos_embed_grid_size,
        })
        return config

    def get_intermediate_layers(
            self,
            inputs: keras.KerasTensor,
            n: int = 1,
            training: Optional[bool] = None,
    ) -> List[keras.KerasTensor]:
        """The patch tokens of the last ``n`` blocks, each through the final norm.

        Interface contract (caller: Depth Anything's DPT head, which reads
        ``n = 4``): the reference ``get_intermediate_layers(x, n, norm=True)``
        with an int ``n`` and no mask. Returns ``n`` tensors
        ``(batch, num_patches, embed_dim)``, shallowest first, with the [CLS]
        and register tokens dropped (and no final norm when
        ``include_top=False``, which builds none). Runs the backbone's own
        layers, so it shares every weight with ``__call__``. Raises
        ``ValueError`` for ``n`` outside ``[1, depth]``.

        Args:
            inputs: Images ``(batch, height, width, in_chans)`` at ``image_size``.
            n: How many of the last blocks to return.
            training: Forwarded to every block.

        Returns:
            The ``n`` normalized patch-token tensors.
        """
        if not 1 <= int(n) <= self.depth:
            raise ValueError(f"n must be in [1, {self.depth}], got {n}")
        x = self.patch_embed(inputs)
        x = self.cls_token_layer(x)
        x = self.pos_embed(x)
        if self.register_token_layer is not None:
            x = keras.ops.concatenate(
                [x[:, :1], self.register_token_layer(x), x[:, 1:]], axis=1)
        outputs = []
        first = self.depth - int(n)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, training=training)
            if i >= first:
                outputs.append(x)
        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        start = 1 + self.num_register_tokens
        return [x[:, start:] for x in outputs]

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        logger.info(f"DINOv2 ViT configuration:")
        logger.info(f"  - Input size: {self.image_size}")
        logger.info(f"  - Patch size: {self.patch_size}")
        logger.info(f"  - Embed dim: {self.embed_dim}")
        logger.info(f"  - Depth: {self.depth}")
        logger.info(f"  - Num heads: {self.num_heads}")
        logger.info(f"  - MLP ratio: {self.mlp_ratio}")
        logger.info(f"  - Num patches: {self.num_patches}")
        logger.info(f"  - Register tokens: {self.num_register_tokens}")
        logger.info(f"  - Stochastic depth rate: {self.stochastic_depth_rate}")
        logger.info(f"  - Init values: {self.init_values}")

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.dino.dino_v2")
class DINOv2(keras.Model):
    """
    Complete DINOv2 Model with classification head following modern Keras 3 patterns.

    This model provides a high-level interface for DINOv2 with automatic
    input/output handling and model variant support. Uses the functional API
    pattern for consistent architecture building.

    Architecture:

    .. code-block:: text

    Input (B, H, W, C)
         ↓
    DINOv2VisionTransformer → Features (B, D)
         ↓
    Dense Classifier (optional) → Predictions (B, num_classes)
    ```

    Provides a complete model interface that can be used for both
    pre-training and fine-tuning, with proper functional API implementation
    following modern Keras 3 best practices.

    Args:
        image_size: Input image size (int or tuple).
        patch_size: Patch size for patch embedding (int or tuple).
        num_classes: Number of output classes.
        include_top: Whether to include classification head.
        input_shape: Input shape. If None, computed from image_size.
        **backbone_kwargs: Additional arguments passed to backbone.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        - If `include_top=True`: 2D tensor with shape: `(batch_size, num_classes)`
        - If `include_top=False`: 2D tensor with shape: `(batch_size, embed_dim)`

    Example:
        ```python
        # Pre-training model (no classification head)
        model = DINOv2(
            image_size=224,
            patch_size=14,
            include_top=False,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # Fine-tuning model with classification head
        model = DINOv2(
            image_size=224,
            patch_size=14,
            num_classes=1000,
            include_top=True,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # From variant
        model = DINOv2.from_variant('base', num_classes=100)
        ```
    """

    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 14,
            num_classes: int = 1000,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **backbone_kwargs
    ) -> None:
        # Validate inputs
        if num_classes <= 0 and include_top:
            raise ValueError(f"num_classes must be positive when include_top=True, got {num_classes}")

        # DECISION plan-2026-08-19T163559-499b6f0e/D-082: pop `name`/`trainable`/`dtype` out of `**backbone_kwargs` before anything else.
        # Forwarding them to the backbone raised a duplicate-keyword TypeError on `from_config`'s `cls(**config)` path. See decisions.md.
        base_kwargs = {key: backbone_kwargs.pop(key)
                       for key in ("name", "trainable", "dtype")
                       if key in backbone_kwargs}

        # Store configuration
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.num_classes = num_classes
        self.include_top = include_top
        self.backbone_kwargs = backbone_kwargs

        # Set input shape
        # DECISION plan-2026-08-19T163559-499b6f0e/D-115: keep the raw `input_shape` argument for `get_config`, same rule as the backbone.
        # Without this key the composite silently reloaded at the 3-channel default. See decisions.md.
        self._input_shape_arg = input_shape
        if input_shape is None:
            in_chans = backbone_kwargs.get('in_chans', 3)
            input_shape = (*self.image_size, in_chans)

        # Initialize layer tracking
        self.backbone = None
        self.classifier = None

        # Create inputs
        # DECISION plan_2026-06-15_e2759fbc/D-008: prefix wrapper `keras.Input` names with "dinov2_", never reuse the backbone's own names.
        # Sharing names aliases the wrapper's symbolic inputs with the nested backbone's at `super().__init__`, raising a graph-cycle error. See decisions.md.
        # DECISION plan_2026-06-15_e2759fbc/D-009: the wrapper contract is 2-input [images, masks], never a 3rd `dinov2_is_training` Input.
        # See decisions.md.
        inputs = keras.Input(shape=input_shape, name="dinov2_input_images")

        # For inference, we typically don't need masks, so provide default
        masks = keras.Input(shape=(None,), dtype="bool", name="dinov2_input_masks")

        # Build the model
        outputs = self._build_model(inputs, masks)

        # Initialize the Model
        base_kwargs.setdefault("name", "dinov2_model")
        super().__init__(
            inputs=[inputs, masks],
            outputs=outputs,
            **base_kwargs
        )

        logger.info(f"Created DINOv2 complete model with include_top={include_top}")
        if include_top:
            logger.info(f"Classification head for {num_classes} classes")

    def _build_model(
            self,
            inputs: keras.KerasTensor,
            masks: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Build the complete DINOv2 model architecture.

        Args:
            inputs: Input tensor
            masks: Mask tensor

        Returns:
            Output tensor
        """
        # Create backbone
        self.backbone = DINOv2VisionTransformer(
            image_size=self.image_size,
            patch_size=self.patch_size,
            name='dinov2_backbone',
            **self.backbone_kwargs
        )

        # DECISION plan_2026-06-15_e2759fbc/D-006: subscript the backbone's 5-key dict output directly, never wrap it in a Lambda + `keras.ops.cond`.
        # A cond on `is_training` previously returned the whole dict in the inference branch instead of the CLS tensor. See decisions.md.
        backbone_output = self.backbone([inputs, masks])
        features = backbone_output["x_norm_clstoken"]

        # Create classifier if needed
        if self.include_top and self.num_classes > 0:
            self.classifier = layers.Dense(
                self.num_classes,
                kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                name='classifier'
            )
            outputs = self.classifier(features)
        else:
            outputs = features

        return outputs

    @classmethod
    def from_variant(
            cls,
            variant: Literal['tiny', 'small', 'base', 'large', 'giant'],
            image_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 14,
            num_classes: int = 1000,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            pretrained: bool = False,
            **kwargs
    ) -> "DINOv2":
        """
        Create DINOv2 model from predefined variant.

        Args:
            variant: Size variant ('tiny', 'small', 'base', 'large', 'giant').
            image_size: Input image size.
            patch_size: Patch size for patch embedding.
            num_classes: Number of output classes.
            include_top: Whether to include classification head.
            input_shape: Input shape.
            pretrained: ``True`` loads the official backbone weights of
                ``small``, ``base`` or ``large`` strictly, as
                `DINOv2VisionTransformer.from_variant` does; the classifier
                (which the official checkpoint does not have) stays at its
                initialization.
            **kwargs: Additional arguments for the model.

        Returns:
            DINOv2 instance.

        Raises:
            ValueError: If variant is not recognized, or if ``pretrained``
                is combined with a non-official backbone structure.
            NotImplementedError: If ``pretrained`` is True for ``tiny`` or ``giant``.
            FileNotFoundError: If ``pretrained`` is True and the download is absent.
        """
        if variant not in DINOv2VisionTransformer.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(DINOv2VisionTransformer.MODEL_VARIANTS.keys())}")

        config = DINOv2VisionTransformer.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        if pretrained:
            official_dinov2_weights_path(variant)
            config.setdefault('pos_embed_grid_size', OFFICIAL_DINOV2_POS_GRID)
            # The checkpoint holds the gains; any init builds the LayerScale it fills.
            config.setdefault('init_values', 1.0)
            _require_official_structure(
                dict(config, include_top=True), patch_size=patch_size,
                num_register_tokens=config.get('num_register_tokens', 0),
                init_values=config.get('init_values'))

        logger.info(f"Creating DINOv2-{variant.upper()} complete model")

        model = cls(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            include_top=include_top,
            input_shape=input_shape,
            **config
        )
        if pretrained:
            load_official_dinov2_file(model.backbone, variant)
        return model

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        # DECISION plan-2026-08-19T163559-499b6f0e/D-082: call `super().get_config()` first, never a literal dict.
        # Without it, `name` and `trainable` reload at their defaults, so a frozen model comes back unfrozen. See decisions.md.
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'include_top': self.include_top,
            'input_shape': self._input_shape_arg,
            **self.backbone_kwargs
        })
        return config

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        logger.info(f"DINOv2 Model configuration:")
        logger.info(f"  - Input size: {self.image_size}")
        logger.info(f"  - Patch size: {self.patch_size}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Backbone config: {self.backbone_kwargs}")

# ---------------------------------------------------------------------

def create_dino_v2(
        variant: Literal['tiny', 'small', 'base', 'large', 'giant'] = 'base',
        *,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_classes: int = 1000,
        include_top: bool = True,
        num_register_tokens: Optional[int] = None,
        init_values: Optional[float] = 1e-5,
        stochastic_depth_rate: float = 0.0,
        ffn_type: Optional[str] = None,
        pretrained: bool = False,
        **kwargs
) -> DINOv2:
    """
    Factory function to create DINOv2 model variants with sensible defaults.

    Signature note (converged surface): ``create_dino_v1``, ``create_dino_v2`` and
    ``create_dino_v3`` share ``(variant, *, image_size, patch_size, num_classes,
    include_top, **kwargs)``. The redundant ``input_shape`` spelling was removed —
    the input shape is always derived as ``(*image_size, in_chans)``. Passing
    ``input_shape=`` raises ``TypeError`` rather than silently disagreeing with
    ``image_size``.

    Variant-defers precedence rule, shared by all three factories: a parameter
    passed as ``None`` defers to the variant's own ``MODEL_VARIANTS`` entry (or, if
    the entry says nothing, to this version's default). An EXPLICIT non-``None``
    value ALWAYS wins over the variant's. Three parameters use it here:

    - ``patch_size``: ``DINOv2VisionTransformer.MODEL_VARIANTS`` defines no
      per-variant patch size, so ``None`` resolves to 14 for every variant.
    - ``ffn_type``: the ``giant`` entry sets ``'swiglu'``; every other variant
      resolves to ``'mlp'``. Passing ``ffn_type='mlp'`` explicitly on ``giant``
      now genuinely gives you MLP — before this rule, ``'mlp'`` was the default
      AND the promotion trigger, so an explicit ``'mlp'`` on ``giant`` was
      silently upgraded to SwiGLU and the caller could not opt out.
    - ``num_register_tokens``: ``None`` gives 4 on ``large``/``giant`` and 0
      elsewhere; an explicit ``0`` on ``giant`` now genuinely means zero.

    Recommended configurations:
    - tiny/small/base/large: 'mlp' FFN, 0 or 4 register tokens
    - giant: 'swiglu' FFN, 4 register tokens, higher stochastic_depth_rate

    Args:
        variant: Size variant ('tiny', 'small', 'base', 'large', 'giant').
        image_size: Integer or ``(height, width)``, input image size.
        patch_size: Patch size for patch embedding. ``None`` defers to the variant
            (v2 has no per-variant patch size, so ``None`` -> 14).
        num_classes: Number of output classes.
        include_top: Whether to include classification head.
        num_register_tokens: Number of register tokens. ``None`` defers to the
            variant (4 for 'large'/'giant', 0 otherwise).
        init_values: LayerScale initialization value.
        stochastic_depth_rate: Maximum stochastic depth rate.
        ffn_type: Type of FFN. ``None`` defers to the variant ('swiglu' for
            'giant', 'mlp' otherwise).
        pretrained: ``True`` loads the official backbone weights of
            ``small``, ``base`` or ``large`` from their local download
            (`DINOv2.from_variant`); ``tiny`` and ``giant`` raise
            ``NotImplementedError``.
        **kwargs: Additional arguments for the model.

    Returns:
        DINOv2 instance.

    Raises:
        TypeError: If ``input_shape`` is passed — use ``image_size`` instead.

    Example:
        ```python
        # Standard DINOv2-Base for ImageNet
        model = create_dino_v2('base', num_classes=1000)

        # DINOv2-Giant — SwiGLU and 4 register tokens come from the variant
        model = create_dino_v2('giant', stochastic_depth_rate=0.3)

        # Pre-training model (no classification head)
        model = create_dino_v2('base', include_top=False)

        # CIFAR-10 model
        model = create_dino_v2(
            'small',
            num_classes=10,
            image_size=32,
            patch_size=16,
        )
        ```
    """
    reject_input_shape(kwargs, "create_dino_v2")

    if patch_size is None:
        patch_size = _DEFAULT_PATCH_SIZE

    # DECISION plan-2026-08-01T105809-dc0c402e/D-017: `None` means "defer to the variant"; an explicit value always wins.
    # A `ffn_type='mlp'` default couldn't distinguish "caller said mlp" from "caller said nothing", so `giant` silently overrode an explicit `'mlp'`. See decisions.md.
    variant_config = DINOv2VisionTransformer.MODEL_VARIANTS.get(variant, {})
    if ffn_type is None:
        ffn_type = variant_config.get('ffn_type', 'mlp')
    if num_register_tokens is None:
        # The official checkpoints have no registers; a pretrained 'large' gets 0.
        num_register_tokens = 4 if variant in ('large', 'giant') and not pretrained else 0

    logger.info(f"Creating DINOv2-{variant.upper()} model with:")
    logger.info(f"  - FFN type: {ffn_type}")
    logger.info(f"  - Register tokens: {num_register_tokens}")
    logger.info(f"  - Stochastic depth rate: {stochastic_depth_rate}")
    logger.info(f"  - Init values: {init_values}")

    return DINOv2.from_variant(
        variant,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        include_top=include_top,
        num_register_tokens=num_register_tokens,
        init_values=init_values,
        stochastic_depth_rate=stochastic_depth_rate,
        ffn_type=ffn_type,
        pretrained=pretrained,
        **kwargs
    )
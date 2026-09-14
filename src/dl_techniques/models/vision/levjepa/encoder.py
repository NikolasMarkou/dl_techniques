"""
Vision transformer encoder for LeVJEPA.

``LeVJEPAEncoder`` turns a video clip or a still image into a token sequence
with the CLS token at index 0. Patch embedding dispatches to
:class:`PatchEmbed3D` when ``num_frames > 1`` and :class:`PatchEmbedding2D`
when it is 1, so the temporal axis enters as a tubelet rather than as extra
channels. Position information arrives one of two ways, never both: a frozen 3D
sincos table added to the tokens, or :class:`VideoRoPE3D` rotating ``q`` and
``k`` inside every block. Attention can be gated by a block-causal mask that is
bidirectional within a frame and causal across frames.

The reference's ``out_layers`` multi-feature output is not ported; only the
final sequence is returned. There is no positional-embedding interpolation: the
sincos table is built once for the configured ``input_shape`` and
``num_frames``, and an input whose rank does not match raises. ``use_rope=True``
builds no ``pos_embed`` weight at all.

References:
    - LeVJEPA PyTorch reference, ``module.py::VisionTransformer``.
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words.
      (https://arxiv.org/abs/2010.11929)
    - Tong et al., 2022. VideoMAE. (https://arxiv.org/abs/2203.12602)
"""

import keras
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.embedding.patch_embed_3d import PatchEmbed3D
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.sincos_pos_embed_3d import get_3d_sincos_pos_embed
from dl_techniques.models.vision.levjepa.blocks import LeVJEPABlock, REFERENCE_INIT_STD
from dl_techniques.models.vision.levjepa.masking import (
    build_block_causal_mask,
    random_token_drop,
)

# ---------------------------------------------------------------------

AttnMode = Literal["full", "block_causal"]

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.levjepa.encoder")
class LeVJEPAEncoder(keras.Model):
    """Encode a video clip or image into a CLS-first token sequence.

    The encoder patch-embeds the input, adds position information, optionally
    drops a fraction of patch tokens at train time, prepends a learnable CLS
    token, optionally builds a block-causal attention mask, runs the
    :class:`~dl_techniques.models.vision.levjepa.blocks.LeVJEPABlock` stack, and
    normalizes. Each block receives the patch grid dimensions, the surviving
    ``token_ids`` and the mask, so rotation and masking stay correct after
    tokens are dropped.

    Architecture:

    .. code-block:: text

          input [B, T, H, W, C] or [B, H, W, C]
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
            num_frames > 1                num_frames == 1
        ┌─────────────────────┐   ┌─────────────────────┐
        │ PatchEmbed3D        │   │ PatchEmbedding2D    │
        └──────────┬──────────┘   └──────────┬──────────┘
                   └────────────┬────────────┘
                                ▼
                   x [B, N, D]   N = T'*H'*W'
                                ▼
                ┌───────────────────────────────┐
                │ + frozen 3D sincos table      │
                │  (use_rope=False only)        │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ random_token_drop             │
                │  (training and rate > 0)      │
                └───────────────┬───────────────┘
                                ▼
                   x [B, N_kept, D], token_ids
                                ▼
                ┌───────────────────────────────┐
                │ prepend cls_token             │
                │  + cls pos row (not use_rope) │
                └───────────────┬───────────────┘
                                ▼
                       [B, 1 + N_kept, D]
                                ▼
                ┌───────────────────────────────┐
                │ build_block_causal_mask       │
                │  ('block_causal' only)        │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ LeVJEPABlock x depth          │
                │  t/h/w, token_ids, attn_mask  │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ LayerNorm eps 1e-6            │
                └───────────────┬───────────────┘
                                ▼
               [B, 1 + N_kept, D], CLS at index 0

    Positional modes (mutually exclusive):

    .. code-block:: text

        use_rope    pos_embed weight   rotation inside blocks
        ────────    ────────────────   ──────────────────────
        False       frozen sincos      none
        True        None               VideoRoPE3D on q and k

    Positional table slicing:

    .. code-block:: text

        pos_embed [1, 1 + num_patches, D]
             │
             ├─ [:, :1, :]  ──►  added to cls_token
             └─ [:, 1:, :]  ──►  added to patch tokens

    :param input_shape: Spatial input shape ``(height, width, channels)``. The
        spatial dimensions must be divisible by ``patch_size``. Defaults to
        ``(224, 224, 3)``.
    :type input_shape: Tuple[int, int, int]
    :param num_frames: Frames per clip. ``1`` (default) routes through
        :class:`PatchEmbedding2D`; any value above 1 routes through
        :class:`PatchEmbed3D` and must be divisible by ``tubelet_size``.
    :type num_frames: int
    :param patch_size: Spatial patch size. Defaults to ``16``.
    :type patch_size: int
    :param tubelet_size: Temporal patch size, read only when
        ``num_frames > 1``. Defaults to ``2``.
    :type tubelet_size: int
    :param embed_dim: Token embedding dimension. Must be positive. Defaults to
        ``192``.
    :type embed_dim: int
    :param depth: Number of :class:`LeVJEPABlock` layers. Must be positive.
        Defaults to ``12``.
    :type depth: int
    :param num_heads: Attention heads per block. Must divide ``embed_dim``.
        Defaults to ``3``.
    :type num_heads: int
    :param mlp_ratio: MLP hidden-dimension multiplier. Defaults to ``4.0``.
    :type mlp_ratio: float
    :param qkv_bias: Give every block's QKV projection a bias. Defaults to
        ``True``.
    :type qkv_bias: bool
    :param use_rope: Rotate ``q`` and ``k`` with :class:`VideoRoPE3D` instead of
        adding a frozen sincos table. ``True`` leaves ``pos_embed`` as ``None``;
        ``False`` (default) builds the table and every block runs without
        rotation. One flag selects both halves, so the two mechanisms cannot be
        combined.
    :type use_rope: bool
    :param rope_theta: Rotary base frequency, forwarded to every block when
        ``use_rope=True``. Defaults to ``10000.0``.
    :type rope_theta: float
    :param attn_mode: ``'full'`` (default) for unmasked attention, or
        ``'block_causal'`` for the bidirectional-within-frame and
        causal-across-frame mask from :func:`build_block_causal_mask`.
    :type attn_mode: AttnMode
    :param token_dropout_rate: Fraction of patch tokens dropped during training
        only, via :func:`random_token_drop`. ``0.0`` (default) is a no-op.
    :type token_dropout_rate: float
    :param dropout_rate: Dropout forwarded to every block's output and MLP
        dropout. Defaults to ``0.0``.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout forwarded to every block's
        post-softmax attention dropout. Defaults to ``0.0``.
    :type attention_dropout_rate: float
    :param init_std: Base truncated-normal std for every kernel and for the CLS
        token. Defaults to ``0.02``.
    :type init_std: float
    :param uniform_power: Forwarded to :func:`get_3d_sincos_pos_embed` on the
        video path. The image path does not read it. Defaults to ``False``.
    :type uniform_power: bool
    :param name: Model name. ``None`` gives ``"levjepa_encoder"``.
    :type name: Optional[str]
    :param kwargs: Additional ``Model`` base-class arguments.

    :ivar cls_token: Learnable ``(1, 1, embed_dim)`` weight, created in
        ``build()``.
    :ivar pos_embed: Frozen ``(1, 1 + num_patches, embed_dim)`` non-trainable
        weight, or ``None`` when ``use_rope=True``.
    :ivar blocks: The ``depth``-long list of :class:`LeVJEPABlock` instances.
    :ivar norm: Final ``LayerNormalization(epsilon=1e-6)``.

    Input shape:
        ``(batch, T, height, width, channels)`` when ``num_frames > 1``, else
        ``(batch, height, width, channels)``.

    Output shape:
        ``(batch, 1 + num_patches_kept, embed_dim)``, CLS token at index 0.

    :raises ValueError: From ``__init__``, if ``attn_mode`` is neither
        ``'full'`` nor ``'block_causal'``, if any dimension or dropout
        parameter is out of range, if the spatial dimensions are not divisible
        by ``patch_size``, or if ``num_frames`` is not divisible by
        ``tubelet_size`` on the video path.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.models.vision.levjepa.encoder import LeVJEPAEncoder

        # Image mode
        enc = LeVJEPAEncoder(
            input_shape=(64, 64, 3), num_frames=1, patch_size=16,
            embed_dim=192, depth=2, num_heads=3,
        )
        enc(keras.random.normal((2, 64, 64, 3))).shape  # (2, 17, 192)

        # Video mode, RoPE, block-causal
        enc = LeVJEPAEncoder(
            input_shape=(32, 32, 3), num_frames=4, tubelet_size=2,
            patch_size=16, embed_dim=192, depth=2, num_heads=3,
            use_rope=True, attn_mode="block_causal",
        )
        enc(keras.random.normal((2, 4, 32, 32, 3))).shape  # (2, 9, 192)
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_frames: int = 1,
        patch_size: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        attn_mode: AttnMode = "full",
        token_dropout_rate: float = 0.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        init_std: float = REFERENCE_INIT_STD,
        uniform_power: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create every sub-layer.

        Arguments are documented on the class.

        :raises ValueError: If the configuration is invalid.
        """
        if name is None:
            name = "levjepa_encoder"
        super().__init__(name=name, **kwargs)

        img_h, img_w, img_c = input_shape
        if img_h <= 0 or img_w <= 0 or img_c <= 0:
            raise ValueError(f"All input_shape dimensions must be positive, got {input_shape}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if img_h % patch_size != 0 or img_w % patch_size != 0:
            raise ValueError(
                f"input_shape spatial dims {input_shape[:2]} must be divisible "
                f"by patch_size ({patch_size})"
            )
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if attn_mode not in ("full", "block_causal"):
            raise ValueError(
                f"attn_mode must be one of 'full', 'block_causal', got {attn_mode!r}"
            )
        if not (0.0 <= token_dropout_rate < 1.0):
            raise ValueError(f"token_dropout_rate must be in [0, 1), got {token_dropout_rate}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be in [0, 1], got {attention_dropout_rate}"
            )

        # DECISION D-013: use_rope is the only positional toggle; do not add a
        # separate pos_embed argument. See decisions.md.
        self.input_shape_config = tuple(input_shape)
        self.num_frames = int(num_frames)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.qkv_bias = bool(qkv_bias)
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.attn_mode = str(attn_mode)
        self.token_dropout_rate = float(token_dropout_rate)
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.init_std = float(init_std)
        self.uniform_power = bool(uniform_power)

        self.is_video = self.num_frames > 1
        if self.is_video and self.num_frames % self.tubelet_size != 0:
            raise ValueError(
                f"num_frames ({self.num_frames}) must be divisible by "
                f"tubelet_size ({self.tubelet_size}) for the video path"
            )

        self.h_patches = img_h // self.patch_size
        self.w_patches = img_w // self.patch_size
        self.t_patches = (self.num_frames // self.tubelet_size) if self.is_video else 1
        self.tokens_per_frame = self.h_patches * self.w_patches
        self.num_patches = self.t_patches * self.tokens_per_frame

        base_kernel_init = {
            "class_name": "TruncatedNormal",
            "config": {"stddev": self.init_std},
        }

        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=self.patch_size,
                tubelet_size=self.tubelet_size,
                embed_dim=self.embed_dim,
                kernel_initializer=keras.initializers.get(base_kernel_init),
                flatten=True,
                name="patch_embed",
            )
        else:
            self.patch_embed = PatchEmbedding2D(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                kernel_initializer=keras.initializers.get(base_kernel_init),
                flatten=True,
                name="patch_embed",
            )

        self.blocks: List[LeVJEPABlock] = []
        for i in range(self.depth):
            block = LeVJEPABlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                use_rope=self.use_rope,
                rope_theta=self.rope_theta,
                num_prefix_tokens=1,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                layer_id=i + 1,
                init_std=self.init_std,
                name=f"block_{i}",
            )
            self.blocks.append(block)

        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")

        # Created in build(): cls_token always, pos_embed only when not use_rope.
        self.cls_token = None
        self.pos_embed = None

        logger.info(
            f"Initialized LeVJEPAEncoder with embed_dim={self.embed_dim}, "
            f"depth={self.depth}, num_heads={self.num_heads}, "
            f"num_frames={self.num_frames}, is_video={self.is_video}, "
            f"use_rope={self.use_rope}, attn_mode={self.attn_mode}, "
            f"num_patches={self.num_patches}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the CLS token and optional sincos table, then build sub-layers.

        :param input_shape: Input shape, 5D for video or 4D for image.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape``'s rank does not match
            ``is_video``.
        """
        if self.built:
            return

        expected_rank = 5 if self.is_video else 4
        if len(input_shape) != expected_rank:
            raise ValueError(
                f"Expected {expected_rank}D input "
                f"({'batch, T, height, width, channels' if self.is_video else 'batch, height, width, channels'}), "
                f"got {len(input_shape)}D input with shape {input_shape}"
            )

        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=self.init_std),
            trainable=True,
        )

        if not self.use_rope:
            # The table reaches the weight through a Constant initializer, never
            # add_weight(zeros) plus .assign(), which StatelessScope discards.
            if self.is_video:
                table = get_3d_sincos_pos_embed(
                    embed_dim=self.embed_dim,
                    grid_size=self.h_patches,
                    grid_depth=self.t_patches,
                    cls_token=True,
                    uniform_power=self.uniform_power,
                )
            else:
                from dl_techniques.layers.embedding.sincos_pos_embed_2d import (
                    get_2d_sincos_pos_embed,
                )

                table = get_2d_sincos_pos_embed(
                    embed_dim=self.embed_dim,
                    grid_size=self.h_patches,
                    cls_token=True,
                    extra_tokens=1,
                )
            # Both builders return (N, embed_dim) with the CLS row included. The
            # leading batch axis is what makes the [:, 1:, :] slicing in call()
            # address tokens rather than features.
            table = table[None, ...]
            self.pos_embed = self.add_weight(
                name="pos_embed",
                shape=table.shape,
                initializer=keras.initializers.Constant(table),
                trainable=False,
            )

        self.patch_embed.build(input_shape)

        block_input_shape = (None, self.num_patches + 1, self.embed_dim)
        for block in self.blocks:
            block.build(block_input_shape)

        self.norm.build(block_input_shape)

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Run the encoder forward pass.

        :param inputs: ``(batch, T, height, width, channels)`` video or
            ``(batch, height, width, channels)`` image, matching ``is_video``.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag. Token dropping happens
            only when ``training`` is truthy and ``token_dropout_rate > 0``.
        :type training: Optional[bool]
        :return: ``(batch, 1 + num_patches_kept, embed_dim)``, CLS at index 0.
        :rtype: keras.KerasTensor
        """
        x = self.patch_embed(inputs, training=training)

        if self.pos_embed is not None:
            patch_pos_embed = self.pos_embed[:, 1:, :]
            x = x + patch_pos_embed

        x, token_ids = random_token_drop(x, self.token_dropout_rate, training=training)

        batch_size = keras.ops.shape(x)[0]
        cls_token = keras.ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        if self.pos_embed is not None:
            cls_pos_embed = self.pos_embed[:, :1, :]
            cls_token = cls_token + cls_pos_embed
        x = keras.ops.concatenate([cls_token, x], axis=1)

        attn_mask = None
        if self.attn_mode == "block_causal":
            # token_ids already carries the batch axis when tokens were dropped.
            attn_mask = build_block_causal_mask(
                num_frames=self.t_patches,
                tokens_per_frame=self.tokens_per_frame,
                token_ids=token_ids,
                num_prefix_tokens=1,
                batch_size=None if token_ids is not None else batch_size,
            )

        for block in self.blocks:
            x = block(
                x,
                num_frames=self.t_patches,
                height_patches=self.h_patches,
                width_patches=self.w_patches,
                token_ids=token_ids,
                attn_mask=attn_mask,
                training=training,
            )

        return self.norm(x, training=training)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        Token dropping makes the kept-token count a runtime quantity, so this
        reports the upper bound, with no dropping.

        :param input_shape: Input shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, 1 + num_patches, embed_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        return (batch_size, self.num_patches + 1, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the model for serialization.

        :return: Dictionary holding every ``__init__`` parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape_config,
                "num_frames": self.num_frames,
                "patch_size": self.patch_size,
                "tubelet_size": self.tubelet_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "use_rope": self.use_rope,
                "rope_theta": self.rope_theta,
                "attn_mode": self.attn_mode,
                "token_dropout_rate": self.token_dropout_rate,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "init_std": self.init_std,
                "uniform_power": self.uniform_power,
            }
        )
        return config

# ---------------------------------------------------------------------

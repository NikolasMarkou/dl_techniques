"""
``LeVJEPAEncoder``: the LeVJEPA Vision Transformer encoder.

Ports the LeVJEPA PyTorch reference's ``VisionTransformer.forward`` (minus its
multi-output-features ``out_layers`` branch, which LeVJEPA's own training
consumer never needs -- see the ``DECISION`` note in ``__init__``):

.. code-block:: python

    def forward(self, x):
        # x: (B, C, T, H, W) or (B, C, H, W)  [this port is channels-last]
        pos_embed = self.pos_embed  # frozen sincos, or None when use_rope
        x = self.patch_embed(x)
        if pos_embed is not None: x = x + patch_pos_embed
        x, token_ids = random_token_drop(x, self.token_drop_rate, training)
        cls_token = self.cls_token.expand(B, -1, -1)
        if pos_embed is not None: cls_token = cls_token + cls_pos_embed
        x = torch.cat([cls_token, x], dim=1)
        attn_mask = build_block_causal_mask(...) if attn_mode == "block_causal" else None
        for blk in self.blocks:
            x = blk(x, T=T, H_patches=H_patches, W_patches=W_patches,
                     token_ids=token_ids, attn_mask=attn_mask)
        x = self.norm(x)
        return x

Dispatches patch embedding to :class:`~dl_techniques.layers.embedding.patch_embed_3d.PatchEmbed3D`
for video (``num_frames > 1``) or
:class:`~dl_techniques.layers.embedding.patch_embedding.PatchEmbedding2D` for a
still image (``num_frames == 1``), per plan.md Assumption A1.

Architecture:
    .. code-block:: text

        Input: (B, T, H, W, C) video  or  (B, H, W, C) image
                            │
              PatchEmbed3D  │  PatchEmbedding2D
              (num_frames>1)│  (num_frames==1)
                            ▼
                    x: (B, N, D)   N = T'*H'*W'
                            │
            use_rope=False  │  use_rope=True
            + frozen 3D     │  (no additive pos_embed;
              sincos table  │   VideoRoPE3D rotates inside
                            │   each LeVJEPABlock instead)
                            ▼
              random_token_drop(x, token_drop_rate, training)
              -> (x, token_ids)   [identity when drop_rate<=0 or not training]
                            │
                            ▼
              prepend cls_token (+ cls pos_embed row, if not use_rope)
                            │
                            ▼
              build_block_causal_mask(...) if attn_mode == "block_causal"
              else None (full/unmasked attention)
                            │
                            ▼
              LeVJEPABlock × depth  (each forwarded T/H'/W'/token_ids/attn_mask)
                            │
                            ▼
              final LayerNorm(eps=1e-6)
                            │
                            ▼
              (B, 1 + N_kept, D), CLS at index 0

Foundational Mathematics:
    Identical in kind to a standard ViT encoder (Dosovitskiy et al., 2020),
    extended with a temporal axis (tubelet embedding, per Tong et al. 2022's
    VideoMAE), an optional 3-axis rotary position embedding in place of an
    additive one, and an optional block-causal (bidirectional-within-frame,
    causal-across-frame) attention mask for autoregressive-over-time
    pretraining.

References:
    - LeVJEPA PyTorch reference, ``module.py::VisionTransformer`` (pasted
      transcript; no public arXiv id in this plan's context).
    - Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words". arXiv:2010.11929.
    - Tong, Z., et al. (2022). "VideoMAE". arXiv:2203.12602.
"""

import keras
from keras import ops
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
    """LeVJEPA's shared ViT encoder: video or image in, CLS-first sequence out.

    Dispatches to a tubelet (:class:`PatchEmbed3D`) or 2D
    (:class:`PatchEmbedding2D`) patch embedding depending on ``num_frames``,
    prepends a learnable CLS token, adds a frozen 3D sincos positional table
    OR rotates q/k with :class:`~dl_techniques.layers.embedding.video_rope.VideoRoPE3D`
    inside each block (mutually exclusive -- see the ``use_rope`` parameter),
    optionally drops a fraction of patch tokens at train time, and optionally
    gates attention with a block-causal mask before running the
    :class:`~dl_techniques.models.vision.levjepa.blocks.LeVJEPABlock` stack.

    **Scope simplifications from the reference** (both deliberate, not
    gaps -- see the ``DECISION`` notes in ``__init__``):

    * No ``out_layers`` multi-feature-map output: LeVJEPA only ever consumes
      the final CLS token, so only the last layer's normalized sequence is
      returned.
    * No dynamic positional-embedding interpolation: the frozen sincos table
      is built once for the configured ``input_shape``/``num_frames``, and a
      call with a mismatched spatial/temporal size raises rather than
      resampling the table.

    :param input_shape: Spatial input shape ``(height, width, channels)``.
        Must be divisible by ``patch_size``. Defaults to ``(224, 224, 3)``.
    :type input_shape: Tuple[int, int, int]
    :param num_frames: Number of frames per clip. ``1`` (default) routes
        through :class:`PatchEmbedding2D` (the still-image path, plan.md
        Assumption A1); any value ``> 1`` routes through :class:`PatchEmbed3D`
        and must be divisible by ``tubelet_size``.
    :type num_frames: int
    :param patch_size: Spatial patch size. Defaults to ``16``.
    :type patch_size: int
    :param tubelet_size: Temporal patch size, used only when
        ``num_frames > 1``. Defaults to ``2``.
    :type tubelet_size: int
    :param embed_dim: Token embedding dimension. Must be positive.
    :type embed_dim: int
    :param depth: Number of :class:`LeVJEPABlock` layers. Must be positive.
    :type depth: int
    :param num_heads: Attention heads per block. Must divide ``embed_dim``.
    :type num_heads: int
    :param mlp_ratio: MLP hidden-dimension multiplier. Defaults to ``4.0``.
    :type mlp_ratio: float
    :param qkv_bias: Whether every block's QKV projection has a bias.
        Defaults to ``True``.
    :type qkv_bias: bool
    :param use_rope: Whether to use :class:`VideoRoPE3D` rotation instead of
        an additive frozen sincos positional table. ``True`` builds NO
        ``pos_embed`` weight at all (``self.pos_embed is None``); ``False``
        (default) builds the frozen sincos table and every block runs
        without RoPE. There is no separate toggle to request both: the
        reference's own ``VisionTransformer.__init__`` has exactly one
        ``use_rope`` flag and no independent ``pos_embed`` argument to
        conflict with it, so the two mechanisms are mutually exclusive BY
        CONSTRUCTION rather than by a runtime check -- see the ``DECISION``
        note in ``__init__`` resolving plan.md Success Criterion 6 against
        this fact.
    :type use_rope: bool
    :param rope_theta: Rotary base frequency, forwarded to every block's
        ``VideoRoPE3D`` when ``use_rope=True``. Defaults to ``10000.0``.
    :type rope_theta: float
    :param attn_mode: ``'full'`` (default) for unmasked attention, or
        ``'block_causal'`` for the bidirectional-within-frame /
        causal-across-frame mask (:func:`build_block_causal_mask`).
    :type attn_mode: AttnMode
    :param token_drop_rate: Fraction of patch tokens dropped at train time
        only (:func:`random_token_drop`). ``0.0`` (default) is a true no-op.
    :type token_drop_rate: float
    :param dropout_rate: Dropout rate forwarded to every block's output/MLP
        dropout. Defaults to ``0.0``.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout rate forwarded to every block's
        post-softmax attention dropout. Defaults to ``0.0``.
    :type attention_dropout_rate: float
    :param init_std: Base truncated-normal std for every kernel and for the
        CLS token. Defaults to ``0.02``.
    :type init_std: float
    :param uniform_power: Forwarded to :func:`get_3d_sincos_pos_embed` /
        the 2D image path's band split. Defaults to ``False``.
    :type uniform_power: bool
    :param name: Model name; auto-generated when ``None``.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

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

    :raises ValueError: If ``attn_mode`` is not ``'full'``/``'block_causal'``,
        if any dimension/dropout parameter is invalid, or if the spatial
        dimensions are not divisible by ``patch_size`` (video: also frames by
        ``tubelet_size``). Raised from ``__init__``.

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
        token_drop_rate: float = 0.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        init_std: float = REFERENCE_INIT_STD,
        uniform_power: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create every sub-layer.

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
        if not (0.0 <= token_drop_rate < 1.0):
            raise ValueError(f"token_drop_rate must be in [0, 1), got {token_drop_rate}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be in [0, 1], got {attention_dropout_rate}"
            )

        # DECISION plan-2026-09-03-2a714a91/D-013
        # Resolving plan.md Success Criterion 6 ("LeVJEPAEncoder(...,
        # use_rope=True, pos_embed=<non-None>) raises ValueError") against the
        # ACTUAL reference constructor, which has exactly one `use_rope: bool`
        # flag and NO separate `pos_embed` argument
        # (`if self.use_rope: self.pos_embed = None else: self.pos_embed =
        # <frozen sincos weight>`). There is therefore no way for a caller of
        # THIS port to pass both `use_rope=True` and a non-None `pos_embed` --
        # the public API the reference specifies simply does not expose that
        # combination, so the criterion is satisfied BY CONSTRUCTION, not by a
        # runtime raise. Inventing a separate `pos_embed=` constructor
        # parameter the reference does not have, purely to give the raise
        # somewhere to fire, would ship a knob nobody asked for and that no
        # other part of this plan uses. Resolution taken: `use_rope: bool` is
        # the ONLY toggle (matching the reference exactly), and
        # `test_encoder.py` pins the mutual exclusion as
        # `use_rope=True -> encoder.pos_embed is None` after build, which is
        # the observable form of "the two mechanisms cannot coexist" that
        # actually applies to this constructor's real surface. See
        # decisions.md D-013.
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
        self.token_drop_rate = float(token_drop_rate)
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
        """Create the CLS token, the optional sincos table, and build every sub-layer.

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
            # Pure NumPy table -> Constant initializer -> add_weight, computed
            # ONCE here. NEVER add_weight(zeros) + .assign(): StatelessScope
            # discards the assign and the table stays all zeros (see the
            # sincos_pos_embed_3d.py / _2d.py module docstrings).
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
            # Both builders return a 2D (N, embed_dim) table (N includes the
            # prepended CLS row). Add the leading batch axis here so the
            # weight broadcasts over any batch size at `x + patch_pos_embed`
            # / `cls_token + cls_pos_embed` below -- a bare (N, D) weight
            # indexed with the same `[:, 1:, :]` slicing used at call time
            # would slice the WRONG axis (index 2 into a rank-2 tensor).
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
            ``(batch, height, width, channels)`` image, matching
            ``is_video``.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag. Token dropping is a
            no-op unless ``training`` is exactly truthy AND
            ``token_drop_rate > 0``.
        :type training: Optional[bool]
        :return: ``(batch, 1 + num_patches_kept, embed_dim)``, CLS at index 0.
        :rtype: keras.KerasTensor
        """
        x = self.patch_embed(inputs, training=training)

        if self.pos_embed is not None:
            patch_pos_embed = self.pos_embed[:, 1:, :]
            x = x + patch_pos_embed

        x, token_ids = random_token_drop(x, self.token_drop_rate, training=training)

        batch_size = ops.shape(x)[0]
        cls_token = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        if self.pos_embed is not None:
            cls_pos_embed = self.pos_embed[:, :1, :]
            cls_token = cls_token + cls_pos_embed
        x = ops.concatenate([cls_token, x], axis=1)

        attn_mask = None
        if self.attn_mode == "block_causal":
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

        Token dropping makes the exact kept-token count a runtime quantity;
        this reports the UPPER BOUND (no dropping).

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
                "token_drop_rate": self.token_drop_rate,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "init_std": self.init_std,
                "uniform_power": self.uniform_power,
            }
        )
        return config


# ---------------------------------------------------------------------

"""
SD3 MMDiT dual-stream diffusion transformer model (Keras 3 port).

This module assembles the Stable Diffusion 3 MMDiT ``SD3Transformer2DModel`` from
the already-built, separately-tested step-1..5 primitives into a single
``keras.Model``:

- :class:`~dl_techniques.layers.embedding.patch_embedding.PatchEmbedding2D`
  -- Conv2D patchify of the ``(B, H, W, in_channels)`` latent into
  ``(B, N, embedding_size)`` patch tokens.
- a FIXED (non-trainable) 2D sin-cos positional embedding cropped to the actual
  patch grid (D-006, see below) -- added to the patch tokens.
- ``context_embedder`` (Dense) -- projects ``encoder_hidden_states
  (B, L, joint_attention_dim)`` to ``(B, L, embedding_size)``.
- a combined timestep + pooled-text conditioning
  (:class:`~dl_techniques.layers.embedding.scalar_sinusoidal_embedding.ScalarSinusoidalEmbedding`
  for the scalar timestep, plus a 2-layer SiLU MLP for the pooled projection)
  summed into a single ``(B, embedding_size)`` vector fed to every block + head.
- a stack of :class:`~dl_techniques.models.sd3_mmdit.blocks.MMDiTBlock` blocks
  (the last is ``context_pre_only`` and drops the text stream; selected indices
  use the dual-attention path).
- :class:`~dl_techniques.models.sd3_mmdit.blocks.MMDiTFinalLayer` -- AdaLN +
  Dense projecting tokens to ``out_channels * patch_size**2``, then unpatchify
  back to ``(B, H, W, out_channels)``.

**Intent**

Given a packed latent ``(B, H, W, in_channels)``, text sequence features
``encoder_hidden_states (B, L, joint_attention_dim)``, a pooled text vector
``pooled_projections (B, pooled_projection_dim)`` and a scalar diffusion
``timestep (B,)`` in ``[0, 1000]``, predict a rectified-flow velocity field of
shape ``(B, H, W, out_channels)`` (= the latent shape, since
``in_channels == out_channels``).

**Positional embedding (DECISION D-006).** SD3's ``PatchEmbed`` precomputes a
2D sin-cos positional grid of side ``pos_embed_max_size`` and CROP-centers it to
the actual patch grid at forward time (``cropped_pos_embed``). This port mirrors
that exactly: a non-trainable weight of shape
``(pos_embed_max_size**2, embedding_size)`` is materialized once at ``build()``
(numpy 2D sincos), then center-cropped to ``(h*w, embedding_size)`` at call
time and added to the patch tokens. This was chosen over reusing
``PositionEmbeddingSine2D`` (whose NCHW output needs an error-prone reshape;
pre-mortem risk 5 / assumption-2 fallback). See decisions.md D-006.

PyTorch reference (faithfully ported)::

    # SD3Transformer2DModel.forward(hidden_states, encoder_hidden_states,
    #                               pooled_projections, timestep)
    hidden = self.pos_embed(hidden_states)          # patchify + cropped pos embed
    temb   = self.time_text_embed(timestep, pooled_projections)  # (B, dim)
    enc    = self.context_embedder(encoder_hidden_states)
    for block in self.transformer_blocks:
        enc, hidden = block(hidden, enc, temb)      # last block: enc discarded
    hidden = self.norm_out(hidden, temb)            # MMDiTFinalLayer
    output = self.unpatchify(hidden)                # (B, H, W, out_channels)
"""

import keras
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.scalar_sinusoidal_embedding import (
    ScalarSinusoidalEmbedding,
)
from dl_techniques.models.sd3_mmdit.blocks import MMDiTBlock, MMDiTFinalLayer
from dl_techniques.models.sd3_mmdit.config import (
    SD3MMDiTConfig,
    get_sd3_config,
)

# ---------------------------------------------------------------------
# 2D sin-cos positional embedding helpers (numpy; computed once at build)
# ---------------------------------------------------------------------


def _build_2d_sincos_pos_embed(
    grid_size: int,
    dim: int,
    base_size: int = 16,
) -> np.ndarray:
    """Build a 2D sin-cos positional embedding grid (numpy, float32).

    Mirrors PyTorch ``get_2d_sincos_pos_embed``: a 2D meshgrid of (h, w)
    coordinates is built, each axis is embedded with a 1D sin-cos basis of
    ``dim // 2`` dimensions, and the two are concatenated along the feature
    axis to give a ``dim``-wide embedding per grid cell. The ``base_size``
    rescales coordinates (SD3 uses ``sample_size // patch_size`` ~ the training
    grid) so the absolute scale of the positional signal is grid-resolution
    independent.

    :param grid_size: Side length of the square grid (produces ``grid_size**2``
        positions).
    :type grid_size: int
    :param dim: Total embedding width (must be divisible by 4 so each axis gets
        an even ``dim // 2``).
    :type dim: int
    :param base_size: Coordinate-normalization base. Defaults to ``16``.
    :type base_size: int
    :return: Array of shape ``(grid_size**2, dim)``.
    :rtype: np.ndarray
    """
    if dim % 4 != 0:
        raise ValueError(
            f"embedding_size ({dim}) must be divisible by 4 for the 2D sin-cos "
            f"positional embedding (each spatial axis takes dim//2, which must "
            f"itself be even for the sin/cos split)."
        )

    # Coordinate grids, normalized by base_size (SD3 convention).
    grid_h = np.arange(grid_size, dtype="float32") / (grid_size / base_size)
    grid_w = np.arange(grid_size, dtype="float32") / (grid_size / base_size)
    # PyTorch uses meshgrid(w, h) then stack -> grid[0]=w-broadcast, grid[1]=h.
    grid_w_m, grid_h_m = np.meshgrid(grid_w, grid_h)  # each (grid, grid)
    # Row-major flatten (matches the patch-token order from PatchEmbedding2D,
    # which reshapes (h_patches, w_patches) row-major).
    pos_h = grid_h_m.reshape(-1)  # (grid_size**2,)
    pos_w = grid_w_m.reshape(-1)

    emb_h = _1d_sincos(dim // 2, pos_h)  # (N, dim//2)
    emb_w = _1d_sincos(dim // 2, pos_w)  # (N, dim//2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (N, dim)
    return emb.astype("float32")


def _1d_sincos(dim: int, pos: np.ndarray) -> np.ndarray:
    """1D sin-cos embedding of a coordinate vector.

    :param dim: Embedding width for this axis (must be even).
    :type dim: int
    :param pos: 1D array of coordinates, shape ``(M,)``.
    :type pos: np.ndarray
    :return: Array of shape ``(M, dim)`` (``[sin | cos]`` concatenation).
    :rtype: np.ndarray
    """
    if dim % 2 != 0:
        raise ValueError(f"per-axis dim must be even, got {dim}")
    omega = np.arange(dim // 2, dtype="float64") / (dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)  # (dim//2,)
    out = pos.reshape(-1).astype("float64")[:, None] * omega[None, :]  # (M, dim//2)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, dim)
    return emb


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class SD3MMDiT(keras.Model):
    """Stable Diffusion 3 MMDiT dual-stream velocity predictor.

    Consumes a DICT of inputs and returns a rectified-flow velocity field with
    the same spatial shape as the latent.

    Call inputs (a single ``dict`` -- keeps the multi-input model serializable):

    - ``"latent"``:               ``(B, H, W, in_channels)`` noisy latent.
    - ``"encoder_hidden_states"``: ``(B, L, joint_attention_dim)`` text features.
    - ``"pooled_projections"``:    ``(B, pooled_projection_dim)`` pooled text vec.
    - ``"timestep"``:             ``(B,)`` diffusion time in ``[0, 1000]``.

    Output: ``(B, H, W, out_channels)`` velocity.

    :param config: The :class:`SD3MMDiTConfig` describing the model.
    :type config: SD3MMDiTConfig
    :param kwargs: Additional ``keras.Model`` arguments.

    :raises TypeError: If ``config`` is not an :class:`SD3MMDiTConfig`.
    """

    def __init__(
        self,
        config: SD3MMDiTConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(config, SD3MMDiTConfig):
            raise TypeError(
                f"config must be an SD3MMDiTConfig, got {type(config)}"
            )

        self.config = config
        dim = config.embedding_size
        self.dim = dim
        self.patch_size = config.patch_size
        self.out_channels = config.out_channels
        self.depth = config.depth
        self.pos_embed_max_size = config.pos_embed_max_size
        # Base size for the sin-cos coordinate normalization (SD3: training grid).
        self._pos_base_size = config.sample_size // config.patch_size

        # --- patchify ---------------------------------------------------
        self.pos_embed = PatchEmbedding2D(
            patch_size=config.patch_size,
            embed_dim=dim,
            name="pos_embed",
        )

        # --- text feature projection -----------------------------------
        self.context_embedder = keras.layers.Dense(dim, name="context_embedder")

        # --- combined timestep + pooled-text conditioning --------------
        self.time_embed = ScalarSinusoidalEmbedding(
            dim=dim, input_range=(0.0, 1000.0), name="time_embed"
        )
        # PixArtAlphaTextProjection: Dense -> SiLU -> Dense.
        self.pooled_proj_in = keras.layers.Dense(dim, name="pooled_proj_in")
        self.pooled_proj_out = keras.layers.Dense(dim, name="pooled_proj_out")

        # --- transformer block stack -----------------------------------
        dual_layers = set(config.dual_attention_layers)
        self.transformer_blocks = [
            MMDiTBlock(
                dim=dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                context_pre_only=(i == self.depth - 1),
                use_dual_attention=(i in dual_layers),
                qk_norm=config.qk_norm,
                eps=config.eps,
                name=f"block_{i}",
            )
            for i in range(self.depth)
        ]

        # --- final velocity head ---------------------------------------
        self.final_layer = MMDiTFinalLayer(
            dim=dim,
            out_channels=self.out_channels * self.patch_size * self.patch_size,
            eps=config.eps,
            name="final_layer",
        )

        # Materialized in build(): the fixed 2D sin-cos positional grid.
        self.pos_embed_table = None

        logger.debug(
            f"Initialized SD3MMDiT(dim={dim}, num_heads={config.num_heads}, "
            f"depth={self.depth}, patch_size={self.patch_size}, "
            f"in_channels={config.in_channels}, out_channels={self.out_channels}, "
            f"joint_attention_dim={config.joint_attention_dim}, "
            f"pooled_projection_dim={config.pooled_projection_dim}, "
            f"pos_embed_max_size={self.pos_embed_max_size}, "
            f"dual_attention_layers={config.dual_attention_layers})"
        )

    def build(self, input_shape: Dict[str, Tuple[Optional[int], ...]]) -> None:
        """Build every sub-layer and the fixed positional-embedding weight.

        :param input_shape: Dict of per-key input shapes (keys ``"latent"``,
            ``"encoder_hidden_states"``, ``"pooled_projections"``,
            ``"timestep"``).
        :type input_shape: Dict[str, Tuple[Optional[int], ...]]
        """
        latent_shape = tuple(input_shape["latent"])
        enc_shape = tuple(input_shape["encoder_hidden_states"])
        pooled_shape = tuple(input_shape["pooled_projections"])

        # --- patchify ---------------------------------------------------
        self.pos_embed.build(latent_shape)
        token_shape = (latent_shape[0], None, self.dim)  # (B, N, dim)

        # --- text projection -------------------------------------------
        self.context_embedder.build(enc_shape)
        enc_token_shape = tuple(enc_shape[:-1]) + (self.dim,)  # (B, L, dim)

        # --- conditioning ----------------------------------------------
        # ScalarSinusoidalEmbedding consumes (B, 1) (squeezed to (B,)) -> (B, dim).
        self.time_embed.build((latent_shape[0], 1))
        self.pooled_proj_in.build(pooled_shape)
        self.pooled_proj_out.build((pooled_shape[0], self.dim))
        cond_shape = (latent_shape[0], self.dim)  # (B, dim)

        # --- block stack -----------------------------------------------
        for blk in self.transformer_blocks:
            blk.build([token_shape, enc_token_shape, cond_shape])

        # --- final head -------------------------------------------------
        self.final_layer.build([token_shape, cond_shape])

        # --- fixed 2D sin-cos positional grid (D-006) ------------------
        # DECISION plan_2026-06-12_dfce0712/D-006: store the 2D sin-cos
        # positional grid as a NON-TRAINABLE weight (numpy-computed here) of
        # shape (pos_embed_max_size**2, dim) and CROP-center it to the actual
        # patch grid at call time -- mirroring SD3's PatchEmbed.cropped_pos_embed.
        # Do NOT reuse PositionEmbeddingSine2D: its NCHW output needs an
        # error-prone reshape/transpose to (B, N, dim) (pre-mortem risk 5 /
        # assumption-2 fallback). The non-trainable add_weight form (like
        # ScalarSinusoidalEmbedding's freq) is serialization-safe.
        # See decisions.md D-006.
        table = _build_2d_sincos_pos_embed(
            grid_size=self.pos_embed_max_size,
            dim=self.dim,
            base_size=self._pos_base_size,
        )  # (max**2, dim)
        self.pos_embed_table = self.add_weight(
            name="pos_embed_table",
            shape=(self.pos_embed_max_size * self.pos_embed_max_size, self.dim),
            initializer=keras.initializers.Constant(table),
            trainable=False,
            dtype="float32",
        )

        super().build(input_shape)

    def _cropped_pos_embed(self, h: int, w: int) -> keras.KerasTensor:
        """Center-crop the (max, max, dim) positional grid to (1, h*w, dim).

        Mirrors SD3 ``PatchEmbed.cropped_pos_embed``: ``top = (max - h) // 2``,
        ``left = (max - w) // 2``; slice the reshaped ``(max, max, dim)`` grid to
        ``(h, w, dim)`` then flatten row-major to ``(1, h*w, dim)``.

        :param h: Patch-grid height (``H // patch_size``).
        :type h: int
        :param w: Patch-grid width (``W // patch_size``).
        :type w: int
        :return: ``(1, h*w, dim)`` cropped positional embedding.
        :rtype: keras.KerasTensor
        """
        max_size = self.pos_embed_max_size
        grid = keras.ops.reshape(
            self.pos_embed_table, (max_size, max_size, self.dim)
        )
        top = (max_size - h) // 2
        left = (max_size - w) // 2
        grid = grid[top : top + h, left : left + w, :]  # (h, w, dim)
        grid = keras.ops.reshape(grid, (1, h * w, self.dim))
        return grid

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the dual-stream MMDiT forward.

        :param inputs: Dict with keys ``"latent"``, ``"encoder_hidden_states"``,
            ``"pooled_projections"``, ``"timestep"`` (see class docstring).
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Forwarded to sub-layers.
        :type training: Optional[bool]
        :return: Velocity ``(B, H, W, out_channels)``.
        :rtype: keras.KerasTensor
        """
        latent = inputs["latent"]
        encoder_hidden_states = inputs["encoder_hidden_states"]
        pooled_projections = inputs["pooled_projections"]
        timestep = inputs["timestep"]

        # Static spatial shape (known at trace time; latent has fixed H, W).
        latent_shape = latent.shape
        H = latent_shape[1]
        W = latent_shape[2]
        p = self.patch_size
        h = H // p
        w = W // p

        # --- patchify + cropped positional embedding -------------------
        hidden = self.pos_embed(latent, training=training)  # (B, N, dim)
        pos = self._cropped_pos_embed(h, w)  # (1, N, dim)
        hidden = hidden + keras.ops.cast(pos, hidden.dtype)

        # --- text feature projection -----------------------------------
        enc = self.context_embedder(encoder_hidden_states, training=training)

        # --- combined conditioning -------------------------------------
        # Feed timestep as (B, 1): ScalarSinusoidalEmbedding squeezes a trailing
        # size-1 axis to (B,). Passing the raw (B,) tensor is ambiguous when
        # B == 1 (the layer would squeeze the batch axis to a scalar). Do NOT
        # pass `timestep` directly here -- always expand to (B, 1) first.
        timestep = keras.ops.reshape(timestep, (-1, 1))
        t_cond = self.time_embed(timestep, training=training)  # (B, dim)
        pooled = self.pooled_proj_in(pooled_projections, training=training)
        pooled = keras.activations.silu(pooled)
        pooled = self.pooled_proj_out(pooled, training=training)  # (B, dim)
        cond = t_cond + pooled  # (B, dim)

        # --- transformer block stack -----------------------------------
        for i, blk in enumerate(self.transformer_blocks):
            if i < self.depth - 1:
                hidden, enc = blk([hidden, enc, cond], training=training)
            else:
                # Final block is context_pre_only: returns the image stream only.
                hidden = blk([hidden, enc, cond], training=training)

        # --- final velocity head ---------------------------------------
        hidden = self.final_layer(
            [hidden, cond], training=training
        )  # (B, N, out_channels*p*p)

        # --- unpatchify back to (B, H, W, out_channels) ----------------
        c = self.out_channels
        B = keras.ops.shape(hidden)[0]
        # (B, h, w, p, p, c)
        hidden = keras.ops.reshape(hidden, (B, h, w, p, p, c))
        # "b h w p q c -> b h p w q c" then merge -> (B, h*p, w*q, c)
        hidden = keras.ops.transpose(hidden, (0, 1, 3, 2, 4, 5))
        out = keras.ops.reshape(hidden, (B, h * p, w * p, c))
        return out

    def compute_output_shape(
        self, input_shape: Dict[str, Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Return the latent shape with the last dim set to ``out_channels``.

        Works before ``build()`` (uses only stored config + the latent shape).

        :param input_shape: Dict of per-key input shapes (uses ``"latent"``).
        :type input_shape: Dict[str, Tuple[Optional[int], ...]]
        :return: ``(B, H, W, out_channels)``.
        :rtype: Tuple[Optional[int], ...]
        """
        latent_shape = tuple(input_shape["latent"])
        return latent_shape[:-1] + (self.out_channels,)

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config (the ``SD3MMDiTConfig`` as a dict)."""
        config = super().get_config()
        config["config"] = self.config.to_dict()
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SD3MMDiT":
        """Reconstruct from :meth:`get_config` output.

        The config is a plain (frozen) dataclass, reconstructed via
        :meth:`SD3MMDiTConfig.from_dict` (NOT ``deserialize_keras_object``).

        :param config: The serialized config dict.
        :type config: Dict[str, Any]
        :return: A reconstructed :class:`SD3MMDiT`.
        :rtype: SD3MMDiT
        """
        config = dict(config)
        config["config"] = SD3MMDiTConfig.from_dict(config["config"])
        return cls(**config)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_sd3_mmdit(
    variant: str = "tiny",
    **overrides: Any,
) -> SD3MMDiT:
    """Build an :class:`SD3MMDiT` from a named preset.

    Retrieves the ``(config, ae)`` pair for ``variant`` via
    :func:`get_sd3_config` (which runs all config invariants), applies any field
    ``overrides`` (re-validated by ``SD3MMDiTConfig.__post_init__``), and returns
    the constructed model. The paired ``AutoEncoderParams`` is not needed by the
    transformer and is discarded here.

    :param variant: One of the config presets (``"tiny"`` or ``"full"``).
    :type variant: str
    :param overrides: Field overrides applied to the preset ``SD3MMDiTConfig``
        (e.g. ``depth=2``). Re-validated on construction.
    :type overrides: Any
    :return: The constructed (un-built) transformer model.
    :rtype: SD3MMDiT
    """
    import dataclasses

    config, _ = get_sd3_config(variant)
    if overrides:
        config = dataclasses.replace(config, **overrides)

    logger.info(
        "Creating SD3MMDiT variant='%s' (embedding_size=%d, depth=%d)",
        variant,
        config.embedding_size,
        config.depth,
    )
    return SD3MMDiT(config=config)

# ---------------------------------------------------------------------

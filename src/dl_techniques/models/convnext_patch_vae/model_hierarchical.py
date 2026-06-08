"""Hierarchical (2-level) resolution-agnostic ConvNeXt Patch-Ladder-VAE.

This module houses the novel machinery for the pool-derived two-level
hierarchical variant of ``ConvNeXtPatchVAE``. It currently provides the
learned top-down conditional prior :class:`_L2ConditionalPrior`; the
coarse-head, the :class:`HierarchicalConvNeXtPatchVAE` model, its config,
presets, and factory are added in later plan steps.

Design provenance: the ``_L2ConditionalPrior`` structure is recovered
verbatim from the deleted hierarchical model at commit ``fdb84888`` and
adapted to the *pool-derived* layout (one fine encoder; the coarse latent
``z2`` is an ``AvgPool2D`` of the fine encoder's last hidden features).
See ``plans/plan_2026-06-08_e3917bd5/`` (findings, plan §Steps 1,
decisions D-001/D-002).
"""

from __future__ import annotations

import copy
import keras
from keras import ops
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------

from dl_techniques.layers.convnext_v2_block import ConvNextV2Block
from dl_techniques.layers.sampling import create_sampling_layer
from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.utils.logger import logger

from .config import HierarchicalConvNeXtPatchVAEConfig
from .decoder import ConvNeXtPatchDecoder
from .encoder import ConvNeXtPatchEncoder

# ------------------------------------------------------------------

# Keys produced by ``keras.Model.get_config()`` that are forwardable
# straight to ``keras.Model.__init__``. Mirrors model.py's filter so
# :meth:`from_config` drops unknown super-class keys before kwargs
# forwarding (defensive — matches the single-scale sibling).
_KERAS_BASE_KEYS = {"name", "trainable", "dtype"}

# ------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class _L2ConditionalPrior(keras.layers.Layer):
    """Learned top-down conditional prior ``p(z1 | z2)`` for the pool-derived
    Patch-Ladder-VAE.

    # DECISION plan_2026-06-08_e3917bd5/D-001: POOL-DERIVED architecture.
    Unlike the deleted fdb84888 two-encoder layout (a separate coarse
    encoder produced ``z2``), here ``z2`` is an ``AvgPool2D`` summary of the
    single fine encoder's last hidden features. This prior therefore maps
    the COARSE latent ``z2 (B, Hp/2, Wp/2, D2)`` UP to fine-grid Gaussian
    prior params ``(mu_p, log_var_p)`` both ``(B, Hp, Wp, D1)``. Do NOT
    reintroduce a second bottom-up encoder or any grid-dependent op
    (``Dense`` / ``GlobalAveragePooling`` / fixed reshape): every cross-scale
    op here MUST stay a parameter-free upsample or a 1x1 / depthwise conv so
    resolution-agnosticism is preserved. See decisions.md D-001.

    Architecture::

        z2 (B, Hp/2, Wp/2, D2)
          -> UpSampling2D(up_factor, "nearest") -> (B, Hp, Wp, D2)
          -> Conv2D(embed_dim, 1)               -> (B, Hp, Wp, embed_dim)
          -> depth x ConvNextV2Block (external residual)
          -> LayerNormalization
          -> [mu_head      (Conv2D 1x1, zeros-init)] -> mu_p
             [log_var_head (Conv2D 1x1, zeros-init)] -> log_var_p

    Both heads are zero-initialized so that at step 0, regardless of ``z2``,
    the prior emits ``mu_p = 0`` and ``log_var_p = 0`` (i.e. ``p(z1|z2) =
    N(0, I)``). This is the VDVAE / NVAE / Ladder-VAE delta recipe and makes
    the step-0 conditional KL numerically identical to the standard KL
    against ``N(0,I)`` (SC3).

    Args:
        latent_dim_fine: Width of each output head's channel axis (D1).
        embed_dim: Internal ConvNeXt block width.
        depth: Number of ``ConvNextV2Block`` layers stacked.
        kernel_size: Depthwise kernel size inside each block.
        up_factor: Nearest-neighbor upsample factor mapping the coarse grid
            ``(Hp/2, Wp/2)`` to the fine grid ``(Hp, Wp)`` (default 2).
        dropout_rate: Per-block dropout rate (forwarded to each block).
        spatial_dropout_rate: Per-block spatial dropout rate.
        kernel_regularizer: Optional regularizer for the conv kernels
            (deep-copied per sub-layer to avoid weight sharing).

    Input shape:
        4D tensor ``(B, Hp/2, Wp/2, D2)`` — the coarse latent.

    Output shape:
        Tuple ``(mu_p, log_var_p)``, each ``(B, Hp, Wp, latent_dim_fine)``.
    """

    def __init__(
        self,
        latent_dim_fine: int,
        embed_dim: int,
        depth: int,
        kernel_size: int = 3,
        up_factor: int = 2,
        dropout_rate: float = 0.0,
        spatial_dropout_rate: float = 0.0,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if up_factor < 2:
            raise ValueError(f"up_factor must be >= 2, got {up_factor}")
        if latent_dim_fine < 1 or embed_dim <= 0 or depth < 1:
            raise ValueError(
                "latent_dim_fine and embed_dim must be positive and depth "
                f"must be >= 1; got latent_dim_fine={latent_dim_fine}, "
                f"embed_dim={embed_dim}, depth={depth}"
            )
        if kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be positive, got {kernel_size}"
            )

        # Store config (for get_config round-trip).
        self.latent_dim_fine = latent_dim_fine
        self.embed_dim = embed_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.kernel_regularizer = kernel_regularizer

        # Sub-layers created in __init__ (Keras 3 Golden Rule). The
        # kernel_regularizer is deep-copied per sub-layer to avoid weight
        # sharing (mirrors encoder.py).
        self.upsample = keras.layers.UpSampling2D(
            size=(up_factor, up_factor),
            interpolation="nearest",
            name="prior_up",
        )
        self.proj_in = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=1,
            padding="valid",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="prior_proj_in",
        )
        self.blocks = [
            ConvNextV2Block(
                kernel_size=kernel_size,
                filters=embed_dim,
                kernel_regularizer=copy.deepcopy(kernel_regularizer),
                dropout_rate=dropout_rate,
                spatial_dropout_rate=spatial_dropout_rate,
                name=f"prior_block_{i}",
            )
            for i in range(depth)
        ]
        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="prior_norm"
        )
        # DECISION plan_2026-06-08_e3917bd5/D-002: VDVAE-delta zero-init.
        # Both prior heads use zeros kernel AND zeros bias so that at step 0,
        # for ANY z2, mu_p = 0 and log_var_p = 0 -> p(z1|z2) = N(0, I). This
        # makes mu1 = mu_p + delta = 0 + delta exactly the standard encoder
        # mu at init, so the existing encoder mu_head can be REUSED as the
        # VDVAE delta (no dedicated delta conv) and the step-0 conditional KL
        # is bit-exact to the N(0,I) KL (SC3). Do NOT switch these to a
        # default/Glorot init: a nonzero prior at step 0 breaks the VDVAE
        # delta semantics and the step-0 identity. See decisions.md D-002.
        self.mu_head = keras.layers.Conv2D(
            filters=latent_dim_fine,
            kernel_size=1,
            padding="valid",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="prior_mu_head",
        )
        self.log_var_head = keras.layers.Conv2D(
            filters=latent_dim_fine,
            kernel_size=1,
            padding="valid",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="prior_log_var_head",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build each child in computational order."""
        if len(input_shape) != 4:
            raise ValueError(
                f"_L2ConditionalPrior expects 4D input "
                f"(B, Hp/2, Wp/2, D2); got {input_shape}"
            )
        B, Hpc, Wpc, _ = input_shape

        self.upsample.build(input_shape)
        # After upsample: (B, Hpc*up, Wpc*up, D2) — coarse channel width D2
        # is preserved by the nearest-neighbor upsample.
        Hp = None if Hpc is None else Hpc * self.up_factor
        Wp = None if Wpc is None else Wpc * self.up_factor
        post_upsample_shape = self.upsample.compute_output_shape(input_shape)

        self.proj_in.build(post_upsample_shape)
        # proj_in -> embed_dim channels; blocks preserve embed_dim.
        block_in_shape = (B, Hp, Wp, self.embed_dim)
        for blk in self.blocks:
            blk.build(block_in_shape)
        self.norm.build(block_in_shape)
        # Heads map embed_dim -> latent_dim_fine.
        self.mu_head.build(block_in_shape)
        self.log_var_head.build(block_in_shape)

        super().build(input_shape)

    def call(
        self,
        z2: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Map the coarse latent ``z2`` to fine-grid prior params.

        Args:
            z2: ``(B, Hp/2, Wp/2, D2)`` coarse latent.
            training: Standard Keras training flag.

        Returns:
            Tuple ``(mu_p, log_var_p)``, both ``(B, Hp, Wp, latent_dim_fine)``.
        """
        x = self.upsample(z2)
        x = self.proj_in(x)
        for blk in self.blocks:
            # External residual (mirrors encoder.py:247): the block emits the
            # residual delta; we add it back here.
            residual = x
            x = blk(x, training=training)
            x = residual + x
        x = self.norm(x, training=training)
        mu_p = self.mu_head(x)
        log_var_p = self.log_var_head(x)
        return mu_p, log_var_p

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return ``(mu_shape, log_var_shape)`` — works before build."""
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape, got {input_shape}"
            )
        B, Hpc, Wpc, _ = input_shape
        Hp = None if Hpc is None else Hpc * self.up_factor
        Wp = None if Wpc is None else Wpc * self.up_factor
        head_shape = (B, Hp, Wp, self.latent_dim_fine)
        return head_shape, head_shape

    def get_config(self) -> Dict[str, Any]:
        """Return constructor kwargs for serialization."""
        config = super().get_config()
        config.update(
            {
                "latent_dim_fine": self.latent_dim_fine,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "kernel_size": self.kernel_size,
                "up_factor": self.up_factor,
                "dropout_rate": self.dropout_rate,
                "spatial_dropout_rate": self.spatial_dropout_rate,
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "_L2ConditionalPrior":
        """Reconstruct, deserializing any regularizer."""
        config = dict(config)
        reg = config.get("kernel_regularizer")
        if isinstance(reg, dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(reg)
        return cls(**config)


@keras.saving.register_keras_serializable()
class _CoarseLatentHead(keras.layers.Layer):
    """Pool-derived coarse Gaussian latent head ``q(z2 | h)``.

    # DECISION plan_2026-06-08_e3917bd5/D-001: POOL-DERIVED coarse latent.
    The coarse latent ``z2`` is NOT produced by a separate bottom-up coarse
    encoder (the deleted fdb84888 layout). Instead it is a parameter-free
    ``AvgPool2D`` summary of the single fine encoder's last hidden features
    ``h (B, Hp, Wp, embed_dim)``, followed by two 1x1 conv heads for the
    Gaussian ``(mu2, log_var2)``. Do NOT replace the pool with a strided/
    ``Dense``/``GlobalAveragePooling`` reduction: only a parameter-free pool
    plus 1x1 convs keeps the head resolution-agnostic. See decisions.md
    D-001.

    Architecture::

        h (B, Hp, Wp, embed_dim)
          -> AvgPool2D(pool_factor)              -> (B, Hp/p, Wp/p, embed_dim)
          -> [mu_head      (Conv2D 1x1)]         -> mu2
             [log_var_head (Conv2D 1x1, zeros)]  -> log_var2

    The ``log_var_head`` is zero-initialized so that at step 0 the coarse
    posterior log-variance is exactly 0 (mirrors the encoder's
    ``log_var_head`` recipe, reducing the step-0 coarse KL).

    Args:
        latent_dim_coarse: Per-patch coarse latent width (D2).
        pool_factor: Average-pool factor mapping the fine grid ``(Hp, Wp)``
            to the coarse grid ``(Hp/pool, Wp/pool)`` (default 2).
        kernel_regularizer: Optional regularizer for the conv kernels
            (deep-copied per head to avoid weight sharing).

    Input shape:
        4D tensor ``(B, Hp, Wp, embed_dim)`` — the fine encoder's last
        hidden features.

    Output shape:
        Tuple ``(mu2, log_var2)``, each
        ``(B, Hp/pool, Wp/pool, latent_dim_coarse)``.
    """

    def __init__(
        self,
        latent_dim_coarse: int,
        pool_factor: int = 2,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if latent_dim_coarse < 1:
            raise ValueError(
                f"latent_dim_coarse must be >= 1, got {latent_dim_coarse}"
            )
        if pool_factor < 2:
            raise ValueError(f"pool_factor must be >= 2, got {pool_factor}")

        # Store config (for get_config round-trip).
        self.latent_dim_coarse = latent_dim_coarse
        self.pool_factor = pool_factor
        self.kernel_regularizer = kernel_regularizer

        # Sub-layers created in __init__ (Keras 3 Golden Rule). The pool is
        # parameter-free and resolution-agnostic. The kernel_regularizer is
        # deep-copied per head to avoid weight sharing (guarding None).
        self.pool = keras.layers.AveragePooling2D(
            pool_size=pool_factor,
            strides=pool_factor,
            padding="valid",
            name="coarse_pool",
        )
        self.mu_head = keras.layers.Conv2D(
            filters=latent_dim_coarse,
            kernel_size=1,
            padding="valid",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="coarse_mu_head",
        )
        self.log_var_head = keras.layers.Conv2D(
            filters=latent_dim_coarse,
            kernel_size=1,
            padding="valid",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="coarse_log_var_head",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build each child in computational order."""
        if len(input_shape) != 4:
            raise ValueError(
                f"_CoarseLatentHead expects 4D input "
                f"(B, Hp, Wp, embed_dim); got {input_shape}"
            )
        self.pool.build(input_shape)
        # After pool: (B, Hp/pool, Wp/pool, embed_dim) — channel width is
        # preserved by the average pool.
        pooled_shape = self.pool.compute_output_shape(input_shape)
        self.mu_head.build(pooled_shape)
        self.log_var_head.build(pooled_shape)

        super().build(input_shape)

    def call(
        self,
        h: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Derive the coarse Gaussian params from the fine hidden features.

        Args:
            h: ``(B, Hp, Wp, embed_dim)`` — fine encoder last hidden features.
            training: Standard Keras training flag (pool/conv are
                training-agnostic; accepted for signature uniformity).

        Returns:
            Tuple ``(mu2, log_var2)``, both
            ``(B, Hp/pool, Wp/pool, latent_dim_coarse)``.
        """
        p = self.pool(h)
        mu2 = self.mu_head(p)
        log_var2 = self.log_var_head(p)
        return mu2, log_var2

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return ``(mu2_shape, log_var2_shape)`` — works before build."""
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape, got {input_shape}"
            )
        B, H, W, _ = input_shape
        Hc = None if H is None else H // self.pool_factor
        Wc = None if W is None else W // self.pool_factor
        head_shape = (B, Hc, Wc, self.latent_dim_coarse)
        return head_shape, head_shape

    def get_config(self) -> Dict[str, Any]:
        """Return constructor kwargs for serialization."""
        config = super().get_config()
        config.update(
            {
                "latent_dim_coarse": self.latent_dim_coarse,
                "pool_factor": self.pool_factor,
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "_CoarseLatentHead":
        """Reconstruct, deserializing any regularizer."""
        config = dict(config)
        reg = config.get("kernel_regularizer")
        if isinstance(reg, dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(reg)
        return cls(**config)


# ----------------------------------------------------------------------
# Named variant presets (hierarchical surface — kept self-contained)
# ----------------------------------------------------------------------
#: Named variant presets for :class:`HierarchicalConvNeXtPatchVAE` — config
#: overrides only. All other fields inherit
#: :class:`HierarchicalConvNeXtPatchVAEConfig` defaults. Mirrors the
#: single-scale ``ConvNeXtPatchVAE.PRESETS`` convention but adds the
#: hierarchical fields (``coarse_latent_dim``/D2, ``prior_depth``/M).
HIERARCHICAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny":  {"embed_dim": 64,  "encoder_depth": 2, "decoder_depth": 2,
              "latent_dim": 8,  "coarse_latent_dim": 8,  "prior_depth": 1},
    "base":  {"embed_dim": 128, "encoder_depth": 4, "decoder_depth": 4,
              "latent_dim": 16, "coarse_latent_dim": 16, "prior_depth": 2},
    "large": {"embed_dim": 192, "encoder_depth": 6, "decoder_depth": 6,
              "latent_dim": 32, "coarse_latent_dim": 32, "prior_depth": 2},
}


@keras.saving.register_keras_serializable()
class HierarchicalConvNeXtPatchVAE(keras.Model):
    """Two-level (fine ``z1`` / coarse ``z2``) hierarchical ConvNeXt patch VAE.

    # DECISION plan_2026-06-08_e3917bd5/D-004: SIBLING class, NOT a subclass
    of :class:`ConvNeXtPatchVAE`. This is a standalone ``keras.Model`` in a
    separate module with its own ``@register_keras_serializable`` identity so
    the two ``.keras`` serialization formats stay isolated. Do NOT make this a
    subclass of ``ConvNeXtPatchVAE`` (or its config a subclass of
    ``ConvNeXtPatchVAEConfig``): subclassing entangles the serialized formats
    and makes ``from_dict`` fragile when fields diverge. The shared machinery
    is reused by COPYING the recovered fdb84888 methods, not by inheritance.
    See decisions.md D-004.

    Architecture (POOL-DERIVED, D-001):

    - One fine :class:`ConvNeXtPatchEncoder` (gaussian) emits the fine VDVAE
      delta ``mu`` and bottom-up ``log_var1`` plus (via ``return_features``)
      the pre-mu-head hidden features ``h``.
    - :class:`_CoarseLatentHead` pools ``h`` -> coarse Gaussian ``(mu2,
      log_var2)`` on the coarse grid ``(Hp/pool, Wp/pool)``.
    - :class:`_L2ConditionalPrior` maps a coarse sample ``z2`` UP to fine-grid
      Gaussian prior params ``(mu_p, log_var_p)``.
    - VDVAE delta: ``mu1 = mu_p + delta`` (D-002). At step 0 the zero-init
      prior gives ``mu_p=0``, so ``mu1 = delta`` and the conditional fine KL is
      bit-exact to the ``N(0,I)`` KL (SC3).
    - Free-bits on the COARSE KL only (D-006).

    Both latents are Gaussian (D-003); there is no vMF path and no
    ``jit_compile=False`` override (``keras.random.normal`` is XLA-safe).

    Args:
        config: :class:`HierarchicalConvNeXtPatchVAEConfig`. If ``None``,
            defaults are used.
        **kwargs: Passthrough to :class:`keras.Model`.

    Input shape:
        4D tensor ``(B, H, W, C)`` with ``H``/``W`` multiples of
        ``patch_size`` and ``Hp = H // patch_size`` divisible by
        ``pool_factor``.

    Output shape:
        Dict with legacy aliases (``reconstruction``, ``z``, ``mu``,
        ``log_var`` — pointing at the FINE level) plus explicit two-level keys
        (``z1``, ``z2``, ``mu1``, ``mu2``, ``log_var1``, ``log_var2``).
    """

    #: Named variant presets — see module-level :data:`HIERARCHICAL_PRESETS`.
    PRESETS: Dict[str, Dict[str, Any]] = HIERARCHICAL_PRESETS

    def __init__(
        self,
        config: Optional[HierarchicalConvNeXtPatchVAEConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            config = HierarchicalConvNeXtPatchVAEConfig()
        self.config = config
        cfg = config

        # Materialize the kernel regularizer (if any) once for all sub-layers.
        kreg: Optional[keras.regularizers.Regularizer] = None
        if cfg.kernel_regularizer_config is not None:
            kreg = keras.regularizers.deserialize(cfg.kernel_regularizer_config)

        # --- Sub-modules (created in __init__, Keras 3 Golden Rule) ---
        # Fine encoder (gaussian). Emits (delta_mu, log_var1[, h]). `delta_mu`
        # is the VDVAE residual (D-002), reused as the fine posterior mean
        # offset; `h` is the pre-mu-head feature tap (return_features=True).
        self.encoder = ConvNeXtPatchEncoder(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.encoder_depth,
            kernel_size=cfg.kernel_size,
            latent_dim=cfg.latent_dim,
            dropout_rate=cfg.dropout_rate,
            spatial_dropout_rate=cfg.spatial_dropout_rate,
            kernel_regularizer=kreg,
            sampling_type="gaussian",
            name="encoder",
        )
        # Pool-derived coarse Gaussian head q(z2|h) (D-001).
        self.coarse_head = _CoarseLatentHead(
            latent_dim_coarse=cfg.coarse_latent_dim,
            pool_factor=cfg.pool_factor,
            kernel_regularizer=kreg,
            name="coarse_head",
        )
        # Learned top-down conditional prior p(z1|z2) (D-001/D-002).
        self.prior = _L2ConditionalPrior(
            latent_dim_fine=cfg.latent_dim,
            embed_dim=cfg.effective_prior_embed_dim,
            depth=cfg.prior_depth,
            kernel_size=3,
            up_factor=cfg.pool_factor,
            dropout_rate=cfg.dropout_rate,
            spatial_dropout_rate=cfg.spatial_dropout_rate,
            kernel_regularizer=kreg,
            name="l2_prior",
        )
        # I3 invariant: ONE stateless Gaussian Sampling named "sampling",
        # reused for BOTH z1 and z2 (the reparameterization trick is
        # parameter-free, so a single shared layer is correct).
        self.sampling = create_sampling_layer("gaussian", name="sampling")
        self.decoder = ConvNeXtPatchDecoder(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.decoder_depth,
            kernel_size=cfg.kernel_size,
            img_channels=cfg.img_channels,
            dropout_rate=cfg.dropout_rate,
            spatial_dropout_rate=cfg.spatial_dropout_rate,
            kernel_regularizer=kreg,
            name="decoder",
        )
        # SIGReg at BOTH levels (A7), N-scaled per grid at the call site.
        self.sigreg_l1 = SIGRegLayer(
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
            name="sigreg_l1",
        )
        self.sigreg_l2 = SIGRegLayer(
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
            name="sigreg_l2",
        )

        # Cached scalar weights pulled from config (used in call/train_step).
        self._beta_kl_l1 = float(cfg.beta_kl_l1)
        self._beta_kl_l2 = float(cfg.beta_kl_l2)
        self._lambda_sigreg_l1 = float(cfg.lambda_sigreg_l1)
        self._lambda_sigreg_l2 = float(cfg.lambda_sigreg_l2)
        self._free_bits = float(cfg.free_bits)
        self._gamma_clip = cfg.gamma_clip

        # --- Loss component trackers (per-component Mean) ---
        # DECISION plan_2026-05-25_fb57d478/D-001 (mirrored): explicit aggregate
        # `loss` tracker named "loss". Keras 3.8 does not auto-create
        # self.loss_tracker until compile(loss=...); our train_step bypasses
        # compiled loss (losses come from add_loss). Without this, the
        # history['loss'] contract pins at 0.0. Mandatory.
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_l1_loss_tracker = keras.metrics.Mean(name="kl_l1_loss")
        self.kl_l2_loss_tracker = keras.metrics.Mean(name="kl_l2_loss")
        self.sigreg_l1_loss_tracker = keras.metrics.Mean(name="sigreg_l1_loss")
        self.sigreg_l2_loss_tracker = keras.metrics.Mean(name="sigreg_l2_loss")
        # Weighted variants (beta * kl, lambda * sigreg) so the actual
        # optimizer contribution per component is visible alongside raw values.
        self.kl_l1_weighted_tracker = keras.metrics.Mean(name="kl_l1_weighted")
        self.kl_l2_weighted_tracker = keras.metrics.Mean(name="kl_l2_weighted")
        self.sigreg_l1_weighted_tracker = keras.metrics.Mean(
            name="sigreg_l1_weighted"
        )
        self.sigreg_l2_weighted_tracker = keras.metrics.Mean(
            name="sigreg_l2_weighted"
        )

        # Edge-case advisory: SIGReg statistic on too-few coarse patches.
        if cfg.coarse_patches_per_side ** 2 < cfg.sigreg_knots:
            logger.warning(
                "HierarchicalConvNeXtPatchVAE: coarse num_patches "
                "(%d) < sigreg_knots (%d). Coarse SIGReg statistic will "
                "still produce a valid scalar but with high variance.",
                cfg.coarse_patches_per_side ** 2,
                cfg.sigreg_knots,
            )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build each child against propagated shapes."""
        if len(input_shape) != 4:
            raise ValueError(
                f"HierarchicalConvNeXtPatchVAE expects 4D input "
                f"(B, H, W, C), got {input_shape}"
            )
        # Encoder on the image. Emits (mu, log_var) and the hidden feature tap
        # h shaped (B, Hp, Wp, embed_dim).
        self.encoder.build(input_shape)
        mu1_shape, log_var1_shape = self.encoder.compute_output_shape(input_shape)
        B, Hp, Wp, _ = mu1_shape
        h_shape = (B, Hp, Wp, self.config.embed_dim)

        # Coarse head on the hidden features.
        self.coarse_head.build(h_shape)
        mu2_shape, _ = self.coarse_head.compute_output_shape(h_shape)

        # Prior on a coarse sample -> fine-grid prior params.
        self.prior.build(mu2_shape)

        # Sampling on (mu, log_var) pairs (fine and coarse share one layer;
        # build against the fine pair shape — both validate identically).
        self.sampling.build([mu1_shape, log_var1_shape])

        # Decoder on the fine latent grid (B, Hp, Wp, D1).
        self.decoder.build(mu1_shape)

        # SIGReg layers on the (B, N, D) reshaped latent views.
        self.sigreg_l1.build((B, Hp * Wp, self.config.latent_dim))
        Hc = None if Hp is None else Hp // self.config.pool_factor
        Wc = None if Wp is None else Wp // self.config.pool_factor
        n2 = None if (Hc is None or Wc is None) else Hc * Wc
        self.sigreg_l2.build((B, n2, self.config.coarse_latent_dim))

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Expose per-component trackers alongside Keras' default metrics.

        Dedup by ``id`` so a metric registered both by Keras (via super)
        and by our custom logic appears once. Mirrors the single-scale
        sibling.
        """
        base = list(super().metrics)
        extras = [
            self.loss_tracker,
            self.recon_loss_tracker,
            self.kl_l1_loss_tracker,
            self.kl_l2_loss_tracker,
            self.sigreg_l1_loss_tracker,
            self.sigreg_l2_loss_tracker,
            self.kl_l1_weighted_tracker,
            self.kl_l2_weighted_tracker,
            self.sigreg_l1_weighted_tracker,
            self.sigreg_l2_weighted_tracker,
        ]
        seen, out = set(), []
        for m in base + extras:
            if id(m) not in seen:
                out.append(m)
                seen.add(id(m))
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """End-to-end two-level forward with ``add_loss`` of five components.

        Args:
            inputs: ``(B, H, W, C)``.
            training: Standard Keras training flag.

        Returns:
            Dict with legacy aliases (``reconstruction``, ``z``, ``mu``,
            ``log_var`` — FINE level) plus explicit two-level keys.
        """
        # 1. Fine encoder. delta_mu is the VDVAE delta (encoder mu_head output);
        #    log_var1 is the fine bottom-up log_var; h is the pre-head tap.
        delta_mu, log_var1, h = self.encoder(
            inputs, training=training, return_features=True
        )
        # 2. Pool-derived coarse Gaussian params.
        mu2, log_var2 = self.coarse_head(h, training=training)
        # 3. Coarse sample.
        z2 = self.sampling([mu2, log_var2], training=training)
        # 4. Top-down conditional prior on the FINE grid (B, Hp, Wp, D1).
        mu_p, log_var_p = self.prior(z2, training=training)
        # DECISION plan_2026-06-08_e3917bd5/D-002: VDVAE residual posterior mean.
        # 5. mu1 = mu_p + delta_mu, reusing the encoder mu_head output as the
        # VDVAE delta (A3). At step 0 the zero-init prior gives mu_p = 0, so
        # mu1 = delta_mu is exactly the standard encoder mu and the conditional
        # fine KL is bit-exact to the N(0,I) KL (SC3). Do NOT add a dedicated
        # delta conv: the encoder mu_head already supplies the identical tensor
        # (use-before-reuse). See decisions.md D-002.
        mu1 = mu_p + delta_mu
        # 6. Fine sample.
        z1 = self.sampling([mu1, log_var1], training=training)
        # 7. Decode the fine latent grid.
        logits = self.decoder(z1, training=training)

        # 8. Recon (per-sample-then-batch-mean scalar; resolution-invariant).
        recon_loss = self._compute_recon(inputs, logits)
        # 9. Fine conditional KL: KL(q(z1|x)=N(mu1,lv1) || p=N(mu_p,lv_p)),
        #    per-patch sum over D1, mean over (B, Hp, Wp).
        kl_l1 = self._compute_kl_l2_conditional(mu1, log_var1, mu_p, log_var_p)
        # 10. Coarse KL vs N(0,I) with free-bits floor (D-006).
        kl_l2 = self._compute_kl_free_bits(mu2, log_var2)
        # 11/12. SIGReg at both levels, N-scaled per grid (A7).
        sigreg_l1 = self._compute_sigreg(z1, self.sigreg_l1)
        sigreg_l2 = self._compute_sigreg(z2, self.sigreg_l2)

        # 13. Five add_loss calls.
        self.add_loss(recon_loss)
        self.add_loss(self._beta_kl_l1 * kl_l1)
        self.add_loss(self._beta_kl_l2 * kl_l2)
        self.add_loss(self._lambda_sigreg_l1 * sigreg_l1)
        self.add_loss(self._lambda_sigreg_l2 * sigreg_l2)

        # 14. Tracker updates (raw + weighted).
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_l1_loss_tracker.update_state(kl_l1)
        self.kl_l2_loss_tracker.update_state(kl_l2)
        self.sigreg_l1_loss_tracker.update_state(sigreg_l1)
        self.sigreg_l2_loss_tracker.update_state(sigreg_l2)
        self.kl_l1_weighted_tracker.update_state(self._beta_kl_l1 * kl_l1)
        self.kl_l2_weighted_tracker.update_state(self._beta_kl_l2 * kl_l2)
        self.sigreg_l1_weighted_tracker.update_state(
            self._lambda_sigreg_l1 * sigreg_l1
        )
        self.sigreg_l2_weighted_tracker.update_state(
            self._lambda_sigreg_l2 * sigreg_l2
        )

        # 15. Pixel-space reconstruction.
        if self.config.recon_loss_type == "bce":
            recon = ops.sigmoid(logits)
        else:
            recon = logits

        # 16. Aliases point at the FINE level for callback compatibility.
        return {
            "reconstruction": recon,
            "z": z1, "mu": mu1, "log_var": log_var1,
            "z1": z1, "z2": z2,
            "mu1": mu1, "mu2": mu2,
            "log_var1": log_var1, "log_var2": log_var2,
        }

    # ------------------------------------------------------------------
    # Loss component helpers (float32 internally — mixed-precision safe)
    # ------------------------------------------------------------------
    def _compute_recon(
        self, x: keras.KerasTensor, logits: keras.KerasTensor
    ) -> keras.KerasTensor:
        x_f = ops.cast(x, "float32")
        l_f = ops.cast(logits, "float32")
        if self.config.recon_loss_type == "mse":
            return ops.mean(ops.square(x_f - l_f))
        # BCE with logits — numerically stable formulation.
        bce = (
            ops.maximum(l_f, 0.0)
            - l_f * x_f
            + ops.log1p(ops.exp(-ops.abs(l_f)))
        )
        return ops.mean(bce)

    def _compute_kl_l2_conditional(
        self,
        mu_q: keras.KerasTensor,
        log_var_q: keras.KerasTensor,
        mu_p: keras.KerasTensor,
        log_var_p: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Closed-form KL(q(z1|x) || p(z1|z2)) for diagonal Gaussians.

        Recovered verbatim from fdb84888 (adapted naming: here q is the FINE
        posterior and p the learned conditional prior)::

            KL(q || p) = 0.5 * sum_d [
                lv_p - lv_q
                + (exp(lv_q) + (mu_q - mu_p)^2) * exp(-lv_p)
                - 1
            ]

        Averaged over ``(B, Hp, Wp)`` so the magnitude is resolution-invariant.
        Both ``log_var_q`` and ``log_var_p`` are clipped to ``[-10, +10]`` for
        float32 stability (A4).

        Sanity check (SC3): with ``mu_p=0, log_var_p=0`` the formula reduces to
        ``-0.5 * (1 + lv_q - mu_q^2 - exp(lv_q))``, i.e. the standard KL against
        ``N(0,I)`` — verified bit-exact at step 0 with zero-init prior heads.
        """
        mu_q_f = ops.cast(mu_q, "float32")
        mu_p_f = ops.cast(mu_p, "float32")
        lv_q_f = ops.clip(ops.cast(log_var_q, "float32"), -10.0, 10.0)
        lv_p_f = ops.clip(ops.cast(log_var_p, "float32"), -10.0, 10.0)
        diff_sq = ops.square(mu_q_f - mu_p_f)
        kl_per_patch = 0.5 * ops.sum(
            lv_p_f - lv_q_f
            + (ops.exp(lv_q_f) + diff_sq) * ops.exp(-lv_p_f)
            - 1.0,
            axis=-1,
        )
        return ops.mean(kl_per_patch)

    def _compute_kl_free_bits(
        self, mu: keras.KerasTensor, log_var: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Coarse KL vs ``N(0,I)`` with a per-patch free-bits floor.

        Standard diagonal-Gaussian KL per patch (sum over D2), then the
        free-bits floor is applied per patch before the grid mean (D-006).
        """
        mu_f = ops.cast(mu, "float32")
        lv_f = ops.clip(ops.cast(log_var, "float32"), -10.0, 10.0)
        kl_per_patch = -0.5 * ops.sum(
            1.0 + lv_f - ops.square(mu_f) - ops.exp(lv_f), axis=-1
        )  # (B, Hp/p, Wp/p)
        # DECISION plan_2026-06-08_e3917bd5/D-006: free-bits collapse gate on
        # the COARSE KL only, via the graph-safe `ops.maximum(KL_per_patch,
        # free_bits)` (no Python branch on a tensor). This keeps z2 in use; do
        # NOT apply free-bits to the fine (conditional) KL — the learned prior
        # already gives z1 capacity, and the collapse risk is on the
        # unconditional coarse latent. Per-patch-mean preserves
        # resolution-invariance. See decisions.md D-006.
        gated = ops.maximum(kl_per_patch, self._free_bits)
        return ops.mean(gated)

    def _compute_sigreg(
        self, z: keras.KerasTensor, sigreg_layer: SIGRegLayer
    ) -> keras.KerasTensor:
        """SIGReg on the per-image patch distribution, multiplied by N.

        Same pattern as ``model.py:_compute_sigreg`` but parametrized by the
        SIGReg layer so a single helper serves both levels. The N-scale uses
        the grid's own ``Hp*Wp`` so per-level pressure is O(N).
        """
        z_f = ops.cast(z, "float32")
        shape = ops.shape(z_f)
        B, Hp, Wp, D = shape[0], shape[1], shape[2], shape[3]
        z_patches = ops.reshape(z_f, (B, Hp * Wp, D))
        return sigreg_layer(z_patches) * ops.cast(Hp * Wp, "float32")

    # ------------------------------------------------------------------
    # Public encode / decode / sample API
    # ------------------------------------------------------------------
    def encode(
        self, x: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Deterministically encode pixels into fine ``(mu1, log_var1)``.

        Deterministic (``training=False``, ``z2 = mu2`` — no sampling) so the
        result is stable for the SC5 save/load round-trip:

            delta_mu, log_var1, h = encoder(x)
            mu2, _ = coarse_head(h);  z2_det = mu2
            mu_p, _ = prior(z2_det);  mu1 = mu_p + delta_mu

        Returns:
            Tuple ``(mu1, log_var1)``, each ``(B, Hp, Wp, latent_dim)``.
        """
        delta_mu, log_var1, h = self.encoder(
            x, training=False, return_features=True
        )
        mu2, _ = self.coarse_head(h, training=False)
        # Deterministic coarse code = posterior mean (no reparameterization).
        mu_p, _ = self.prior(mu2, training=False)
        mu1 = mu_p + delta_mu
        return mu1, log_var1

    def decode(self, z1: keras.KerasTensor) -> keras.KerasTensor:
        """Decode a fine per-patch latent grid back to pixels.

        Applies the appropriate output activation based on
        ``recon_loss_type`` (sigmoid for BCE, identity for MSE).
        """
        logits = self.decoder(z1, training=False)
        if self.config.recon_loss_type == "bce":
            return ops.sigmoid(logits)
        return logits

    def sample(
        self,
        num_samples: int,
        hp: Optional[int] = None,
        wp: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> keras.KerasTensor:
        """Coherent two-level prior sampling (adapted from fdb84888:866).

        Pipeline::

            z2 ~ N(0, I)                       # coarse grid (hp/p, wp/p)
            mu_p, lv_p = self.prior(z2)        # fine-grid conditional prior
            z1 = mu_p + exp(0.5*clip(lv_p)) * eps,  eps ~ N(0, I)
            return self.decode(z1)

        Drawing ``z1`` from ``p(z1|z2)`` (not independent ``N(0,I)``) is what
        makes pure-prior sampling coherent: the two latents come from the joint
        generative distribution.

        Args:
            num_samples: Number of images to generate.
            hp: Fine patch grid height. Defaults to ``config.patches_per_side``.
            wp: Fine patch grid width. Defaults to ``config.patches_per_side``.
            seed: Optional RNG seed.

        Returns:
            ``(num_samples, hp * patch_size, wp * patch_size, img_channels)``.
        """
        cfg = self.config
        hp = cfg.patches_per_side if hp is None else int(hp)
        wp = cfg.patches_per_side if wp is None else int(wp)
        if hp <= 0 or wp <= 0:
            raise ValueError(
                f"hp and wp must be positive, got hp={hp}, wp={wp}"
            )
        pf = cfg.pool_factor
        if hp % pf != 0 or wp % pf != 0:
            raise ValueError(
                f"hp ({hp}) and wp ({wp}) must each be divisible by "
                f"pool_factor ({pf}) so the coarse grid is integer."
            )
        hpc, wpc = hp // pf, wp // pf
        z2 = keras.random.normal(
            shape=(num_samples, hpc, wpc, cfg.coarse_latent_dim),
            seed=seed,
        )
        mu_p, lv_p = self.prior(z2, training=False)
        eps = keras.random.normal(
            shape=ops.shape(mu_p),
            seed=None if seed is None else seed + 1,
        )
        z1 = mu_p + ops.exp(0.5 * ops.clip(lv_p, -10.0, 10.0)) * eps
        return self.decode(z1)

    def sample_from(
        self,
        x: keras.KerasTensor,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> keras.KerasTensor:
        """Coherent sampling around a real anchor ``x``.

        Reparameterizes the FINE posterior at temperature ``t``:
        ``z1 = mu1 + t * exp(0.5 * log_var1) * eps``.

        - ``temperature=0.0`` -> deterministic ``decode(encode(x)[0])`` (decode
          the posterior mean ``mu1``).
        - ``temperature=1.0`` -> matches the VAE posterior scale.
        - ``temperature>1.0`` -> more diverse variations.

        Args:
            x: Real anchor image batch ``(B, H, W, C)``.
            temperature: Reparameterization noise scale.
            seed: Optional RNG seed.

        Returns:
            ``(B, H, W, C)`` coherent reconstruction / variation of ``x``.
        """
        mu1, log_var1 = self.encode(x)
        t = float(temperature)
        eps = keras.random.normal(ops.shape(mu1), seed=seed) * t
        z1 = mu1 + ops.exp(0.5 * ops.clip(log_var1, -10.0, 10.0)) * eps
        return self.decode(z1)

    # ------------------------------------------------------------------
    # Custom train_step / test_step (mirrors model.py verbatim)
    # ------------------------------------------------------------------
    def train_step(self, data: Any) -> Dict[str, Any]:
        """One training step: forward, sum add_loss outputs, apply grads."""
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            _ = self(x, training=True)
            losses = self.losses
            if losses:
                loss = ops.cast(losses[0], "float32")
                for extra in losses[1:]:
                    loss = loss + ops.cast(extra, "float32")
            else:
                loss = ops.convert_to_tensor(0.0, dtype="float32")
        grads = tape.gradient(loss, self.trainable_variables)
        if self._gamma_clip is not None:
            c = float(self._gamma_clip)
            grads = [
                None if g is None else ops.clip(g, -c, c) for g in grads
            ]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Any) -> Dict[str, Any]:
        """One eval step: forward only, no gradients."""
        x = data[0] if isinstance(data, tuple) else data
        _ = self(x, training=False)
        losses = self.losses
        if losses:
            loss = ops.cast(losses[0], "float32")
            for extra in losses[1:]:
                loss = loss + ops.cast(extra, "float32")
        else:
            loss = ops.convert_to_tensor(0.0, dtype="float32")
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    # ------------------------------------------------------------------
    # Named variants
    # ------------------------------------------------------------------
    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: bool = False,
        **overrides: Any,
    ) -> "HierarchicalConvNeXtPatchVAE":
        """Build a named variant from :attr:`PRESETS`.

        Args:
            variant: One of :attr:`PRESETS` keys (``"tiny"``, ``"base"``,
                ``"large"``).
            pretrained: If ``True``, raises :class:`NotImplementedError` — no
                public checkpoints exist.
            **overrides: Forwarded to
                :class:`HierarchicalConvNeXtPatchVAEConfig`, taking precedence
                over the preset values.

        Returns:
            Unbuilt :class:`HierarchicalConvNeXtPatchVAE`.

        Raises:
            ValueError: ``variant`` not in :attr:`PRESETS`.
            NotImplementedError: ``pretrained=True``.
        """
        if variant not in cls.PRESETS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: "
                f"{sorted(cls.PRESETS)}"
            )
        cfg_kwargs = {**cls.PRESETS[variant], **overrides}
        cfg = HierarchicalConvNeXtPatchVAEConfig(**cfg_kwargs)
        model = cls(config=cfg)
        if pretrained:
            try:
                weights_path = cls._download_weights(variant)
                model.load_weights(weights_path)
            except (IOError, OSError, ValueError) as e:
                logger.error(
                    "Pretrained weight load failed for variant '%s': %s",
                    variant, e,
                )
                raise
        return model

    @classmethod
    def _download_weights(cls, variant: str) -> str:
        """Resolve a pretrained-weights path — none are published.

        Loud failure beats silent random-init (repo-wide convention).

        Raises:
            NotImplementedError: Always — no public checkpoints exist.
        """
        raise NotImplementedError(
            f"No pretrained weights are published for hierarchical "
            f"convnext_patch_vae variant '{variant}'."
        )

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------
    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Return the dict-of-shapes matching :meth:`call` output."""
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape, got {input_shape}"
            )
        mu1_shape, log_var1_shape = self.encoder.compute_output_shape(input_shape)
        B, Hp, Wp, _ = mu1_shape
        h_shape = (B, Hp, Wp, self.config.embed_dim)
        mu2_shape, log_var2_shape = self.coarse_head.compute_output_shape(h_shape)
        # z1/z2 share mu1/mu2 shapes (Sampling preserves shape).
        recon_shape = self.decoder.compute_output_shape(mu1_shape)
        return {
            "reconstruction": recon_shape,
            "z": mu1_shape, "mu": mu1_shape, "log_var": log_var1_shape,
            "z1": mu1_shape, "z2": mu2_shape,
            "mu1": mu1_shape, "mu2": mu2_shape,
            "log_var1": log_var1_shape, "log_var2": log_var2_shape,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], custom_objects=None
    ) -> "HierarchicalConvNeXtPatchVAE":
        config = dict(config)
        cfg_dict = config.pop("config", None)
        cfg = (
            HierarchicalConvNeXtPatchVAEConfig.from_dict(cfg_dict)
            if cfg_dict is not None
            else HierarchicalConvNeXtPatchVAEConfig()
        )
        # Defensive narrowing: forward only the known-safe keras.Model
        # super-keys, dropping any future serialized field.
        extra = {k: v for k, v in config.items() if k in _KERAS_BASE_KEYS}
        return cls(config=cfg, **extra)


# ----------------------------------------------------------------------
# Module-level factory
# ----------------------------------------------------------------------
def create_hierarchical_convnext_patch_vae(
    variant: str = "base",
    *,
    pretrained: bool = False,
    **overrides: Any,
) -> HierarchicalConvNeXtPatchVAE:
    """Create a :class:`HierarchicalConvNeXtPatchVAE` from a named variant.

    Thin module-level delegate to
    :meth:`HierarchicalConvNeXtPatchVAE.from_variant`. Mirrors
    ``create_convnext_patch_vae``.

    Args:
        variant: One of :attr:`HierarchicalConvNeXtPatchVAE.PRESETS` keys
            (``"tiny"``, ``"base"``, ``"large"``).
        pretrained: If ``True``, raises :class:`NotImplementedError`.
        **overrides: Forwarded to :class:`HierarchicalConvNeXtPatchVAEConfig`.

    Returns:
        Unbuilt :class:`HierarchicalConvNeXtPatchVAE`.

    Raises:
        ValueError: ``variant`` not in PRESETS.
        NotImplementedError: ``pretrained=True``.
    """
    return HierarchicalConvNeXtPatchVAE.from_variant(
        variant, pretrained=pretrained, **overrides,
    )

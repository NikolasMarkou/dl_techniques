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
from typing import Any, Dict, Optional, Tuple

# ------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------

from dl_techniques.layers.convnext_v2_block import ConvNextV2Block
from dl_techniques.utils.logger import logger

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

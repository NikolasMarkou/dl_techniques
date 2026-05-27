"""Two-level hierarchical :class:`HierarchicalConvNeXtPatchVAE`.

Sibling of :class:`ConvNeXtPatchVAE` — a coarse global L1 + fine L2 design
recommended in ``analyses/analysis_2026-05-26_05ccde10/summary.md`` Section 6.

Pipeline:

::

    x : (B, H, W, C)
        |               |
        |               |
        v               v
    L1 encoder       L2 encoder
    (patch=p1)       (patch=p2)
        |               |
    (mu_l1, lv_l1)   (mu_l2, lv_l2)
        |               |
    Sampling         Sampling
        |               |
    z_l1             z_l2
        |               |
        +-----> _L2ConditionedDecoder(z_l2, z_l1) -----> reconstruction
                  ^
                  tile-broadcast z_l1 over the L2 grid (UpSampling2D(tile_factor)),
                  concatenate with the L2 proj_in features, project back to
                  embed_dim_l2 via 1x1 Conv2D, then run the existing
                  ConvNextV2Block stack + LN + Conv2DTranspose head.

Losses (all assembled via ``add_loss`` in ``call()``):

    recon
  + beta_kl_l1   * KL(z_l1)
  + beta_kl_l2   * KL(z_l2)
  + lambda_l1    * (sigreg(z_l1) * N_l1)
  + lambda_l2    * (sigreg(z_l2) * N_l2)

L1 has NO pixel-space reconstruction head — it is a conditioning latent only.

Anchored decisions (see ``plans/plan_2026-05-27_dee954c6/decisions.md``):

- D-001: explicit ``self.loss_tracker`` mirrors the single-scale D-001 of
  ``plan_2026-05-25_fb57d478`` — required for Keras 3.8 + ``add_loss`` +
  custom ``train_step``.
- D-002: SIGReg at BOTH scales is multiplied by ``N = Hp * Wp`` at the
  call site (the layer is N-agnostic; without ``* N`` SIGReg pressure
  collapses 16x per resolution doubling).
- D-003: L2 decoder-only conditioning via nearest-neighbor tile-broadcast.
  L2 encoder remains a pure bottom-up extractor (`encode(x)` stays
  self-contained). Nearest-neighbor (not bilinear) so the conditioning
  signal is locally constant within each L1 patch footprint — bilinear
  would blur structural information across patch boundaries.
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
from dl_techniques.layers.sampling import Sampling
from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.utils.logger import logger

from .config import HierarchicalConvNeXtPatchVAEConfig
from .encoder import ConvNeXtPatchEncoder

# ------------------------------------------------------------------

_KERAS_BASE_KEYS = {"name", "trainable", "dtype"}


@keras.saving.register_keras_serializable()
class _L2ConditionedDecoder(keras.layers.Layer):
    """L2 decoder that consumes ``z_l2`` and a tile-broadcast ``z_l1``.

    Structurally mirrors :class:`ConvNeXtPatchDecoder` but inserts a
    conditioning concat after ``proj_in``:

        z_l2 -> proj_in -> features (B, Hp2, Wp2, embed_dim_l2)
        z_l1 -> UpSampling2D(tile_factor, "nearest") -> (B, Hp2, Wp2, latent_l1)
        concat -> (B, Hp2, Wp2, embed_dim_l2 + latent_l1)
        cond_proj (1x1 Conv2D) -> (B, Hp2, Wp2, embed_dim_l2)
        N x ConvNextV2Block (external residual)
        LayerNormalization
        Conv2DTranspose(img_channels, kernel=patch_size_l2, stride=patch_size_l2)
        -> raw logits (B, Hp2 * patch_size_l2, Wp2 * patch_size_l2, img_channels)

    Args:
        patch_size_l2: L2 stride.
        embed_dim_l2: ConvNeXt block width.
        depth: Block count.
        kernel_size: Depthwise kernel.
        img_channels: Output channels.
        latent_dim_l1: Width of the L1 conditioning latent.
        tile_factor: UpSampling2D factor (``patch_size_l1 // patch_size_l2``).
        dropout_rate, spatial_dropout_rate, kernel_regularizer: forwarded.
    """

    def __init__(
        self,
        patch_size_l2: int,
        embed_dim_l2: int,
        depth: int,
        kernel_size: int,
        img_channels: int,
        latent_dim_l1: int,
        tile_factor: int,
        dropout_rate: float = 0.0,
        spatial_dropout_rate: float = 0.0,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if patch_size_l2 <= 0 or embed_dim_l2 <= 0 or depth < 1:
            raise ValueError(
                "patch_size_l2, embed_dim_l2 must be positive and depth >= 1"
            )
        if tile_factor < 2:
            raise ValueError(
                f"tile_factor must be >= 2, got {tile_factor}"
            )
        self.patch_size_l2 = patch_size_l2
        self.embed_dim_l2 = embed_dim_l2
        self.depth = depth
        self.kernel_size = kernel_size
        self.img_channels = img_channels
        self.latent_dim_l1 = latent_dim_l1
        self.tile_factor = tile_factor
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.kernel_regularizer = kernel_regularizer

        # z_l2 -> embed_dim_l2
        self.proj_in = keras.layers.Conv2D(
            filters=embed_dim_l2,
            kernel_size=1,
            padding="valid",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="proj_in",
        )
        # DECISION plan_2026-05-27_dee954c6/D-003: nearest-neighbor
        # tile-broadcast keeps conditioning piecewise-constant inside
        # each L1 patch footprint. Bilinear would blur structural
        # information across L1 patch boundaries.
        self.upsample = keras.layers.UpSampling2D(
            size=tile_factor,
            interpolation="nearest",
            name="cond_upsample",
        )
        self.concat = keras.layers.Concatenate(axis=-1, name="cond_concat")
        # (embed_dim_l2 + latent_dim_l1) -> embed_dim_l2
        self.cond_proj = keras.layers.Conv2D(
            filters=embed_dim_l2,
            kernel_size=1,
            padding="valid",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="cond_proj",
        )
        self.blocks = [
            ConvNextV2Block(
                kernel_size=kernel_size,
                filters=embed_dim_l2,
                kernel_regularizer=copy.deepcopy(kernel_regularizer),
                dropout_rate=dropout_rate,
                spatial_dropout_rate=spatial_dropout_rate,
                name=f"block_{i}",
            )
            for i in range(depth)
        ]
        self.pre_head_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="pre_head_norm"
        )
        self.head = keras.layers.Conv2DTranspose(
            filters=img_channels,
            kernel_size=patch_size_l2,
            strides=patch_size_l2,
            padding="valid",
            activation=None,
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="head",
        )

    def build(self, input_shape: Any) -> None:
        # Input is a list/tuple: [z_l2_shape, z_l1_shape].
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "_L2ConditionedDecoder expects build([z_l2_shape, z_l1_shape]); "
                f"got {input_shape}"
            )
        z_l2_shape, z_l1_shape = input_shape[0], input_shape[1]
        if len(z_l2_shape) != 4 or len(z_l1_shape) != 4:
            raise ValueError(
                f"Both inputs must be 4D, got z_l2={z_l2_shape}, z_l1={z_l1_shape}"
            )
        B, Hp2, Wp2, _ = z_l2_shape
        self.proj_in.build(z_l2_shape)
        # After upsample: (B, Hp1*tile, Wp1*tile, latent_dim_l1) which == (B, Hp2, Wp2, latent_dim_l1).
        upsampled_shape = (z_l1_shape[0], Hp2, Wp2, self.latent_dim_l1)
        self.upsample.build(z_l1_shape)
        # Concat outputs (B, Hp2, Wp2, embed_dim_l2 + latent_dim_l1).
        post_proj_shape = (B, Hp2, Wp2, self.embed_dim_l2)
        concat_out_shape = (B, Hp2, Wp2, self.embed_dim_l2 + self.latent_dim_l1)
        self.concat.build([post_proj_shape, upsampled_shape])
        self.cond_proj.build(concat_out_shape)
        block_in_shape = (B, Hp2, Wp2, self.embed_dim_l2)
        for blk in self.blocks:
            blk.build(block_in_shape)
        self.pre_head_norm.build(block_in_shape)
        self.head.build(block_in_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        z_l2, z_l1 = inputs[0], inputs[1]
        feat = self.proj_in(z_l2)
        cond = self.upsample(z_l1)
        merged = self.concat([feat, cond])
        x = self.cond_proj(merged)
        for blk in self.blocks:
            residual = x
            x = blk(x, training=training)
            x = residual + x
        x = self.pre_head_norm(x, training=training)
        x = self.head(x)
        return x

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        z_l2_shape = input_shape[0]
        B, Hp2, Wp2, _ = z_l2_shape
        H = None if Hp2 is None else Hp2 * self.patch_size_l2
        W = None if Wp2 is None else Wp2 * self.patch_size_l2
        return (B, H, W, self.img_channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "patch_size_l2": self.patch_size_l2,
                "embed_dim_l2": self.embed_dim_l2,
                "depth": self.depth,
                "kernel_size": self.kernel_size,
                "img_channels": self.img_channels,
                "latent_dim_l1": self.latent_dim_l1,
                "tile_factor": self.tile_factor,
                "dropout_rate": self.dropout_rate,
                "spatial_dropout_rate": self.spatial_dropout_rate,
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "_L2ConditionedDecoder":
        config = dict(config)
        reg = config.get("kernel_regularizer")
        if isinstance(reg, dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(reg)
        return cls(**config)


# ------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class _L2ConditionalPrior(keras.layers.Layer):
    """Learned conditional prior ``p(z_l2 | z_l1)`` for hierarchical VAE.

    Consumes the L1 latent ``z_l1`` and emits ``(mu_p, log_var_p)``
    parameterizing a diagonal Gaussian over each L2 patch position.
    Architecture mirrors the encoder/decoder block idiom:

        z_l1 (B, Hp1, Wp1, latent_l1)
          -> UpSampling2D(tile_factor, "nearest") -> (B, Hp2, Wp2, latent_l1)
          -> Conv2D(embed_dim, 1)                -> (B, Hp2, Wp2, embed_dim)
          -> N x ConvNextV2Block (external residual)
          -> LayerNormalization
          -> [mu_head (Conv2D 1x1, zeros-init)   ] -> mu_p
             [log_var_head (Conv2D 1x1, zeros-init)] -> log_var_p

    Both heads are zero-initialized so that at step 0, regardless of
    z_l1, the prior emits ``mu_p = 0`` and ``log_var_p = 0`` (i.e.
    ``N(0, I)``). This matches the legacy implicit prior exactly and
    enables checkpoint reuse from the old hierarchical model.

    Args:
        tile_factor: Upsample factor (``patch_size_l1 // patch_size_l2``).
        latent_dim_l1: Width of the input ``z_l1`` channel axis.
        latent_dim_l2: Width of each output head's channel axis.
        embed_dim: Internal ConvNeXt block width.
        depth: Number of ``ConvNextV2Block`` layers stacked.
        kernel_size: Depthwise kernel size inside each block.
        dropout_rate, spatial_dropout_rate, kernel_regularizer:
            Forwarded to each block.
    """

    def __init__(
        self,
        tile_factor: int,
        latent_dim_l1: int,
        latent_dim_l2: int,
        embed_dim: int,
        depth: int,
        kernel_size: int,
        dropout_rate: float = 0.0,
        spatial_dropout_rate: float = 0.0,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if tile_factor < 2:
            raise ValueError(
                f"tile_factor must be >= 2, got {tile_factor}"
            )
        if latent_dim_l1 < 1 or latent_dim_l2 < 1 or embed_dim <= 0 or depth < 1:
            raise ValueError(
                "latent_dim_l1, latent_dim_l2, embed_dim must be positive "
                "and depth must be >= 1"
            )
        self.tile_factor = tile_factor
        self.latent_dim_l1 = latent_dim_l1
        self.latent_dim_l2 = latent_dim_l2
        self.embed_dim = embed_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.kernel_regularizer = kernel_regularizer

        # D-005 (plan_2026-05-27_c3184aea): this UpSampling2D is OWNED by
        # this layer; the decoder has its own. Two zero-parameter instances
        # keep build()/get_config() symmetric and serialization simple.
        self.upsample = keras.layers.UpSampling2D(
            size=tile_factor,
            interpolation="nearest",
            name="prior_upsample",
        )
        self.proj_in = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=1,
            padding="valid",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="proj_in",
        )
        self.blocks = [
            ConvNextV2Block(
                kernel_size=kernel_size,
                filters=embed_dim,
                kernel_regularizer=copy.deepcopy(kernel_regularizer),
                dropout_rate=dropout_rate,
                spatial_dropout_rate=spatial_dropout_rate,
                name=f"block_{i}",
            )
            for i in range(depth)
        ]
        self.pre_head_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="pre_head_norm"
        )
        # DECISION plan_2026-05-27_c3184aea/D-001: both heads zero-init so
        # the prior emits exactly N(0, I) at step 0. This (a) makes the
        # at-step-0 conditional KL numerically identical to the legacy KL
        # against N(0,I), and (b) lets checkpoints trained with the old
        # prior land in this architecture and continue training from the
        # exact same operating point. Standard NVAE / Ladder-VAE recipe.
        self.mu_head = keras.layers.Conv2D(
            filters=latent_dim_l2,
            kernel_size=1,
            padding="valid",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="mu_head",
        )
        self.log_var_head = keras.layers.Conv2D(
            filters=latent_dim_l2,
            kernel_size=1,
            padding="valid",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            name="log_var_head",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(
                f"_L2ConditionalPrior expects 4D input "
                f"(B, Hp1, Wp1, latent_l1); got {input_shape}"
            )
        B, Hp1, Wp1, _ = input_shape
        self.upsample.build(input_shape)
        # After upsample: (B, Hp1*tf, Wp1*tf, latent_l1).
        Hp2 = None if Hp1 is None else Hp1 * self.tile_factor
        Wp2 = None if Wp1 is None else Wp1 * self.tile_factor
        post_upsample_shape = (B, Hp2, Wp2, self.latent_dim_l1)
        self.proj_in.build(post_upsample_shape)
        block_in_shape = (B, Hp2, Wp2, self.embed_dim)
        for blk in self.blocks:
            blk.build(block_in_shape)
        self.pre_head_norm.build(block_in_shape)
        self.mu_head.build(block_in_shape)
        self.log_var_head.build(block_in_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        x = self.upsample(inputs)
        x = self.proj_in(x)
        for blk in self.blocks:
            residual = x
            x = blk(x, training=training)
            x = residual + x
        x = self.pre_head_norm(x, training=training)
        mu_p = self.mu_head(x)
        log_var_p = self.log_var_head(x)
        return mu_p, log_var_p

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        B, Hp1, Wp1, _ = input_shape
        Hp2 = None if Hp1 is None else Hp1 * self.tile_factor
        Wp2 = None if Wp1 is None else Wp1 * self.tile_factor
        head_shape = (B, Hp2, Wp2, self.latent_dim_l2)
        return head_shape, head_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "tile_factor": self.tile_factor,
                "latent_dim_l1": self.latent_dim_l1,
                "latent_dim_l2": self.latent_dim_l2,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "kernel_size": self.kernel_size,
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
        config = dict(config)
        reg = config.get("kernel_regularizer")
        if isinstance(reg, dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(reg)
        return cls(**config)


# ------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HierarchicalConvNeXtPatchVAE(keras.Model):
    """Two-level hierarchical ConvNeXt patch VAE (L1 coarse + L2 fine).

    See module docstring for the architecture and Section 6 of
    ``analyses/analysis_2026-05-26_05ccde10/summary.md`` for the design
    rationale.
    """

    PRESETS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "embed_dim_l1": 64, "embed_dim_l2": 64,
            "encoder_depth_l1": 2, "decoder_depth_l1": 2,
            "encoder_depth_l2": 2, "decoder_depth_l2": 2,
            "latent_dim_l1": 32, "latent_dim_l2": 8,
        },
        "base": {
            "embed_dim_l1": 128, "embed_dim_l2": 128,
            "encoder_depth_l1": 4, "decoder_depth_l1": 4,
            "encoder_depth_l2": 4, "decoder_depth_l2": 4,
            "latent_dim_l1": 64, "latent_dim_l2": 16,
        },
        "large": {
            "embed_dim_l1": 192, "embed_dim_l2": 192,
            "encoder_depth_l1": 6, "decoder_depth_l1": 6,
            "encoder_depth_l2": 6, "decoder_depth_l2": 6,
            "latent_dim_l1": 96, "latent_dim_l2": 32,
        },
    }

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

        kreg: Optional[keras.regularizers.Regularizer] = None
        if cfg.kernel_regularizer_config is not None:
            kreg = keras.regularizers.deserialize(cfg.kernel_regularizer_config)

        # --- L1 encoder (coarse, no pixel recon) ---
        self.encoder_l1 = ConvNeXtPatchEncoder(
            patch_size=cfg.patch_size_l1,
            embed_dim=cfg.embed_dim_l1,
            depth=cfg.encoder_depth_l1,
            kernel_size=cfg.kernel_size,
            latent_dim=cfg.latent_dim_l1,
            dropout_rate=cfg.dropout_rate,
            spatial_dropout_rate=cfg.spatial_dropout_rate,
            kernel_regularizer=kreg,
            name="encoder_l1",
        )
        self.sampling_l1 = Sampling(name="sampling_l1")

        # --- L2 encoder (fine, bottom-up only — no conditioning at iter-1) ---
        self.encoder_l2 = ConvNeXtPatchEncoder(
            patch_size=cfg.patch_size_l2,
            embed_dim=cfg.embed_dim_l2,
            depth=cfg.encoder_depth_l2,
            kernel_size=cfg.kernel_size,
            latent_dim=cfg.latent_dim_l2,
            dropout_rate=cfg.dropout_rate,
            spatial_dropout_rate=cfg.spatial_dropout_rate,
            kernel_regularizer=kreg,
            name="encoder_l2",
        )
        self.sampling_l2 = Sampling(name="sampling_l2")

        # --- L2 decoder, conditioned on z_l1 ---
        self.decoder_l2 = _L2ConditionedDecoder(
            patch_size_l2=cfg.patch_size_l2,
            embed_dim_l2=cfg.embed_dim_l2,
            depth=cfg.decoder_depth_l2,
            kernel_size=cfg.kernel_size,
            img_channels=cfg.img_channels,
            latent_dim_l1=cfg.latent_dim_l1,
            tile_factor=cfg.tile_factor,
            dropout_rate=cfg.dropout_rate,
            spatial_dropout_rate=cfg.spatial_dropout_rate,
            kernel_regularizer=kreg,
            name="decoder_l2",
        )

        # --- SIGReg layers (one per scale, both N-agnostic; * N at call site) ---
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

        # Cached weights — mutated by BetaAnnealingCallback during training.
        self._beta_kl_l1 = float(cfg.beta_kl_l1)
        self._beta_kl_l2 = float(cfg.beta_kl_l2)
        self._lambda_sigreg_l1 = float(cfg.lambda_sigreg_l1)
        self._lambda_sigreg_l2 = float(cfg.lambda_sigreg_l2)
        self._gamma_clip = cfg.gamma_clip

        # --- Trackers (11 total) ---
        # DECISION plan_2026-05-27_dee954c6/D-001: explicit aggregate
        # `loss` tracker — required for the Keras 3.8 + add_loss +
        # custom train_step contract. Mirrors single-scale D-001
        # (plan_2026-05-25_fb57d478).
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_l1_loss_tracker = keras.metrics.Mean(name="kl_L1_loss")
        self.kl_l2_loss_tracker = keras.metrics.Mean(name="kl_L2_loss")
        self.sigreg_l1_loss_tracker = keras.metrics.Mean(name="sigreg_L1_loss")
        self.sigreg_l2_loss_tracker = keras.metrics.Mean(name="sigreg_L2_loss")
        self.kl_l1_weighted_tracker = keras.metrics.Mean(name="kl_L1_weighted")
        self.kl_l2_weighted_tracker = keras.metrics.Mean(name="kl_L2_weighted")
        self.sigreg_l1_weighted_tracker = keras.metrics.Mean(name="sigreg_L1_weighted")
        self.sigreg_l2_weighted_tracker = keras.metrics.Mean(name="sigreg_L2_weighted")

        # Edge-case advisory: SIGReg statistic on too-few patches.
        if cfg.num_patches_l1 < cfg.sigreg_knots:
            logger.warning(
                "HierarchicalConvNeXtPatchVAE: num_patches_l1 (%d) < "
                "sigreg_knots (%d). L1 SIGReg statistic remains valid but "
                "with high variance.",
                cfg.num_patches_l1, cfg.sigreg_knots,
            )

    # ------------------------------------------------------------------
    @property
    def metrics(self) -> List[keras.metrics.Metric]:
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
        # Encode both scales independently.
        mu_l1, log_var_l1 = self.encoder_l1(inputs, training=training)
        mu_l2, log_var_l2 = self.encoder_l2(inputs, training=training)
        z_l1 = self.sampling_l1([mu_l1, log_var_l1], training=training)
        z_l2 = self.sampling_l2([mu_l2, log_var_l2], training=training)

        logits = self.decoder_l2([z_l2, z_l1], training=training)

        # Losses (all float32, mixed-precision-safe).
        recon_loss = self._compute_recon(inputs, logits)
        kl_l1_loss = self._compute_kl(mu_l1, log_var_l1)
        kl_l2_loss = self._compute_kl(mu_l2, log_var_l2)
        sigreg_l1_loss = self._compute_sigreg(z_l1, self.sigreg_l1)
        sigreg_l2_loss = self._compute_sigreg(z_l2, self.sigreg_l2)

        self.add_loss(recon_loss)
        self.add_loss(self._beta_kl_l1 * kl_l1_loss)
        self.add_loss(self._beta_kl_l2 * kl_l2_loss)
        self.add_loss(self._lambda_sigreg_l1 * sigreg_l1_loss)
        self.add_loss(self._lambda_sigreg_l2 * sigreg_l2_loss)

        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_l1_loss_tracker.update_state(kl_l1_loss)
        self.kl_l2_loss_tracker.update_state(kl_l2_loss)
        self.sigreg_l1_loss_tracker.update_state(sigreg_l1_loss)
        self.sigreg_l2_loss_tracker.update_state(sigreg_l2_loss)
        self.kl_l1_weighted_tracker.update_state(self._beta_kl_l1 * kl_l1_loss)
        self.kl_l2_weighted_tracker.update_state(self._beta_kl_l2 * kl_l2_loss)
        self.sigreg_l1_weighted_tracker.update_state(
            self._lambda_sigreg_l1 * sigreg_l1_loss
        )
        self.sigreg_l2_weighted_tracker.update_state(
            self._lambda_sigreg_l2 * sigreg_l2_loss
        )

        if self.config.recon_loss_type == "bce":
            recon = ops.sigmoid(logits)
        else:
            recon = logits

        # `mu` alias -> mu_l2 keeps existing callbacks (ReconViz,
        # LatentSpaceCallback, LatentInterpolationCallback) working
        # without modification — D-005.
        return {
            "reconstruction": recon,
            "z_l1": z_l1, "mu_l1": mu_l1, "log_var_l1": log_var_l1,
            "z_l2": z_l2, "mu_l2": mu_l2, "log_var_l2": log_var_l2,
            "z": z_l2, "mu": mu_l2, "log_var": log_var_l2,
        }

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    def _compute_recon(
        self, x: keras.KerasTensor, logits: keras.KerasTensor
    ) -> keras.KerasTensor:
        x_f = ops.cast(x, "float32")
        l_f = ops.cast(logits, "float32")
        if self.config.recon_loss_type == "mse":
            return ops.mean(ops.square(x_f - l_f))
        bce = (
            ops.maximum(l_f, 0.0)
            - l_f * x_f
            + ops.log1p(ops.exp(-ops.abs(l_f)))
        )
        return ops.mean(bce)

    def _compute_kl(
        self, mu: keras.KerasTensor, log_var: keras.KerasTensor
    ) -> keras.KerasTensor:
        mu_f = ops.cast(mu, "float32")
        # Mirror single-scale H18 fix: hard clip log_var in [-10, +10].
        lv_f = ops.clip(ops.cast(log_var, "float32"), -10.0, 10.0)
        kl_per_patch = -0.5 * ops.sum(
            1.0 + lv_f - ops.square(mu_f) - ops.exp(lv_f), axis=-1
        )
        return ops.mean(kl_per_patch)

    def _compute_sigreg(
        self, z: keras.KerasTensor, sigreg_layer: SIGRegLayer
    ) -> keras.KerasTensor:
        """SIGReg on per-image patch distribution, multiplied by N.

        Same pattern as single-scale ``model.py:_compute_sigreg``.
        """
        z_f = ops.cast(z, "float32")
        shape = ops.shape(z_f)
        B, Hp, Wp, D = shape[0], shape[1], shape[2], shape[3]
        z_patches = ops.reshape(z_f, (B, Hp * Wp, D))
        # DECISION plan_2026-05-27_dee954c6/D-002: multiply by N=Hp*Wp
        # at the call site (the SIGRegLayer itself is N-agnostic). Without
        # this, effective SIGReg pressure collapses with the grid.
        return sigreg_layer(z_patches) * ops.cast(Hp * Wp, "float32")

    # ------------------------------------------------------------------
    # Public encode / decode / sample API
    # ------------------------------------------------------------------
    def encode(
        self, x: keras.KerasTensor
    ) -> Tuple[
        keras.KerasTensor, keras.KerasTensor,
        keras.KerasTensor, keras.KerasTensor,
    ]:
        """Encode pixels into ``(mu_l1, log_var_l1, mu_l2, log_var_l2)``."""
        mu_l1, log_var_l1 = self.encoder_l1(x, training=False)
        mu_l2, log_var_l2 = self.encoder_l2(x, training=False)
        return mu_l1, log_var_l1, mu_l2, log_var_l2

    def decode(
        self, z_l1: keras.KerasTensor, z_l2: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Decode ``(z_l1, z_l2)`` to pixels."""
        logits = self.decoder_l2([z_l2, z_l1], training=False)
        if self.config.recon_loss_type == "bce":
            return ops.sigmoid(logits)
        return logits

    def sample_from(
        self,
        x: keras.KerasTensor,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> keras.KerasTensor:
        """One-line coherent sampling: jitter the posterior of real anchor ``x``.

        Pure-prior sampling (:meth:`sample`) gives incoherent images
        because the L2 decoder was trained on `(z_l1, z_l2)` pairs
        extracted from the *same* image — the two latents are NOT
        independent in the data distribution. Sampling them
        independently from the prior produces global/local mismatch.

        This method keeps them correlated by reparameterizing both from
        the encoder's posterior on a real image. ``temperature=0`` gives
        deterministic reconstruction; ``temperature=1`` matches the VAE
        prior scale; higher values produce more diverse variations.

        Args:
            x: Real anchor image batch ``(B, H, W, C)``.
            temperature: Noise scale on the reparameterization. ``0.0``
                yields ``decode(mu_l1, mu_l2)`` exactly.
            seed: Optional RNG seed.

        Returns:
            Coherent reconstructions / variations of ``x``, shape
            ``(B, H, W, C)``.
        """
        mu_l1, lv_l1, mu_l2, lv_l2 = self.encode(x)
        t = float(temperature)
        eps_l1 = keras.random.normal(ops.shape(mu_l1), seed=seed) * t
        eps_l2 = keras.random.normal(
            ops.shape(mu_l2),
            seed=None if seed is None else seed + 1,
        ) * t
        z_l1 = mu_l1 + ops.exp(0.5 * ops.clip(lv_l1, -10.0, 10.0)) * eps_l1
        z_l2 = mu_l2 + ops.exp(0.5 * ops.clip(lv_l2, -10.0, 10.0)) * eps_l2
        return self.decode(z_l1, z_l2)

    def sample(
        self,
        num_samples: int,
        hp1: Optional[int] = None,
        wp1: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> keras.KerasTensor:
        """Pure-prior sampling — WARNING: produces incoherent images.

        The L2 decoder was trained on correlated `(z_l1, z_l2)` pairs.
        Sampling both from independent N(0, I) priors violates that
        training distribution. Use :meth:`sample_from` for coherent
        variations of a real anchor, or fit an aggregate-posterior /
        learnable prior to use pure-prior sampling properly.

        Kept for plumbing tests and pure-prior baselines only.
        """
        cfg = self.config
        hp1 = cfg.patches_per_side_l1 if hp1 is None else int(hp1)
        wp1 = cfg.patches_per_side_l1 if wp1 is None else int(wp1)
        if hp1 <= 0 or wp1 <= 0:
            raise ValueError(f"hp1 and wp1 must be positive, got hp1={hp1}, wp1={wp1}")
        tf_ = cfg.tile_factor
        hp2, wp2 = hp1 * tf_, wp1 * tf_
        eps_l1 = keras.random.normal(
            shape=(num_samples, hp1, wp1, cfg.latent_dim_l1), seed=seed,
        )
        eps_l2 = keras.random.normal(
            shape=(num_samples, hp2, wp2, cfg.latent_dim_l2), seed=seed,
        )
        return self.decode(eps_l1, eps_l2)

    # ------------------------------------------------------------------
    # Custom train_step
    # ------------------------------------------------------------------
    def train_step(self, data: Any) -> Dict[str, Any]:
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
            grads = [None if g is None else ops.clip(g, -c, c) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Any) -> Dict[str, Any]:
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
        if variant not in cls.PRESETS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {sorted(cls.PRESETS)}"
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
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")
        mu_l1_shape, log_var_l1_shape = self.encoder_l1.compute_output_shape(input_shape)
        mu_l2_shape, log_var_l2_shape = self.encoder_l2.compute_output_shape(input_shape)
        recon_shape = self.decoder_l2.compute_output_shape(
            [mu_l2_shape, mu_l1_shape]
        )
        return {
            "reconstruction": recon_shape,
            "z_l1": mu_l1_shape, "mu_l1": mu_l1_shape, "log_var_l1": log_var_l1_shape,
            "z_l2": mu_l2_shape, "mu_l2": mu_l2_shape, "log_var_l2": log_var_l2_shape,
            "z": mu_l2_shape, "mu": mu_l2_shape, "log_var": log_var_l2_shape,
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
    :meth:`HierarchicalConvNeXtPatchVAE.from_variant`.
    """
    return HierarchicalConvNeXtPatchVAE.from_variant(
        variant, pretrained=pretrained, **overrides,
    )

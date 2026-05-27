"""Top-level :class:`CliffordNetPatchVAEV2` model — hierarchical Clifford VAE.

Sibling of :class:`ConvNeXtPatchVAEV2`. Composes the hierarchical
:class:`CliffordNetPatchEncoderV2` + :class:`Sampling` +
:class:`CliffordNetPatchDecoderV2` + SIGReg + optional LPIPS + optional
classification head + optional :class:`CliffordSegmentationHead`, with a
custom ``train_step`` that bypasses ``compile(loss=...)`` and uses
``add_loss`` for every loss component.

Same input contract as v2: either a plain image tensor OR a dict::

    {
        "image": (B,H,W,C) float,
        "label_cls": (B,) int32   (optional; required when cls head active)
        "label_seg": (B,H,W) int32 (optional; required when seg head active)
    }

Anchored decisions (see ``plans/plan_2026-05-27_75849a91/decisions.md``):

- D-001: sibling model package; v2 model.py is untouched.
- D-002: CliffordNetBlock owns internal residual — encoder/decoder
  loops do NOT add an outer skip.
- D-003: CliffordSegmentationHead with explicit ``upsample_factor``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import tensorflow as tf
from keras import ops

from dl_techniques.layers.sampling import Sampling
from dl_techniques.losses.lpips_loss import LPIPSLoss
from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.utils.logger import logger

from .config import CliffordNetPatchVAEV2Config, PRESETS
from .decoder import CliffordNetPatchDecoderV2
from .encoder import CliffordNetPatchEncoderV2
from .heads import AttentionPoolClassifierHead, CliffordSegmentationHead
from .mae_mask import upsample_mask_to_pixels


_KERAS_BASE_KEYS = {"name", "trainable", "dtype"}


@keras.saving.register_keras_serializable(package="dl_techniques")
class CliffordNetPatchVAEV2(keras.Model):
    """Hierarchical CliffordNet patch-level VAE.

    Args:
        config: :class:`CliffordNetPatchVAEV2Config`. If ``None``, defaults
            are used.
        **kwargs: Passthrough to :class:`keras.Model`.

    Input shape:
        Tensor ``(B, H, W, C)`` OR dict
        ``{"image":(B,H,W,C), "label_cls":(B,)?, "label_seg":(B,H,W)?}``.

    Output shape:
        Dict with at least::

            {
                "reconstruction": (B,H,W,C),
                "z": (B,Hb,Wb,latent_dim),
                "mu": (B,Hb,Wb,latent_dim),
                "log_var": (B,Hb,Wb,latent_dim),
            }

        plus ``logits_cls`` (``(B,num_classes_cls)``) when cls head is
        active, and ``logits_seg`` (``(B,H,W,num_classes_seg)``) when seg
        head is active.
    """

    PRESETS: Dict[str, Dict[str, Any]] = PRESETS

    def __init__(
        self,
        config: Optional[CliffordNetPatchVAEV2Config] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            config = CliffordNetPatchVAEV2Config()
        self.config = config
        cfg = config

        kreg: Optional[keras.regularizers.Regularizer] = None
        if cfg.kernel_regularizer_config is not None:
            kreg = keras.regularizers.deserialize(cfg.kernel_regularizer_config)

        # --- VAE backbone ---
        self.encoder = CliffordNetPatchEncoderV2(
            patch_size=cfg.patch_size,
            stage_dims=cfg.stage_dims,
            stage_depths=cfg.stage_depths,
            stage_shifts=cfg.stage_shifts,
            downsample_kind=cfg.downsample_kind,
            downsample_kernel_size=cfg.downsample_kernel_size,
            cli_mode=cfg.cli_mode,
            ctx_mode=cfg.ctx_mode,
            use_global_context_in_last_stage=cfg.use_global_context_in_last_stage,
            layer_scale_init=cfg.layer_scale_init,
            block_drop_path_rate=cfg.block_drop_path_rate,
            latent_dim=cfg.latent_dim,
            kernel_regularizer=kreg,
            mae_mask_ratio=cfg.mae_mask_ratio,
            name="encoder",
        )
        self.sampling = Sampling(name="sampling")
        self.decoder = CliffordNetPatchDecoderV2(
            patch_size=cfg.patch_size,
            stage_dims=cfg.stage_dims,
            stage_depths=cfg.stage_depths,
            stage_shifts=cfg.stage_shifts,
            latent_dim=cfg.latent_dim,
            img_channels=cfg.img_channels,
            cli_mode=cfg.cli_mode,
            ctx_mode=cfg.ctx_mode,
            use_global_context_in_last_stage=cfg.use_global_context_in_last_stage,
            layer_scale_init=cfg.layer_scale_init,
            block_drop_path_rate=cfg.block_drop_path_rate,
            kernel_regularizer=kreg,
            name="decoder",
        )
        self.sigreg = SIGRegLayer(
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
            name="sigreg",
        )

        # --- Optional heads ---
        self.cls_head: Optional[AttentionPoolClassifierHead] = None
        if cfg.use_classification_head:
            self.cls_head = AttentionPoolClassifierHead(
                embed_dim=cfg.embed_dim,  # == stage_dims[-1]
                num_classes=cfg.num_classes_cls,
                num_heads=cfg.cls_head_num_heads,
                dropout_rate=cfg.cls_head_dropout,
                name="cls_head",
            )

        self.seg_head: Optional[CliffordSegmentationHead] = None
        if cfg.use_segmentation_head:
            self.seg_head = CliffordSegmentationHead(
                embed_dim=cfg.embed_dim,
                num_classes=cfg.num_classes_seg,
                upsample_factor=cfg.total_upsample_factor,
                dropout_rate=cfg.seg_head_dropout,
                name="seg_head",
            )

        # --- LPIPS perceptual loss (lazy; held by the model). ---
        self._lpips: Optional[LPIPSLoss] = None
        if cfg.lambda_lpips > 0.0:
            self._lpips = LPIPSLoss(
                layer_weights=cfg.lpips_layer_weights,
                input_range=tuple(cfg.lpips_input_range),
                name="lpips",
            )

        # Cached scalar weights (mutable; allows annealing callbacks).
        self._beta_kl = float(cfg.beta_kl)
        self._lambda_sigreg = float(cfg.lambda_sigreg)
        self._lambda_mae = float(cfg.lambda_mae)
        self._lambda_lpips = float(cfg.lambda_lpips)
        self._lambda_cls = float(cfg.lambda_cls)
        self._lambda_seg = float(cfg.lambda_seg)
        self._gamma_clip = cfg.gamma_clip

        # --- Trackers ---
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.sigreg_loss_tracker = keras.metrics.Mean(name="sigreg_loss")
        self.kl_loss_weighted_tracker = keras.metrics.Mean(
            name="kl_loss_weighted"
        )
        self.sigreg_loss_weighted_tracker = keras.metrics.Mean(
            name="sigreg_loss_weighted"
        )
        # MAE / LPIPS / cls / seg trackers exist regardless of toggles.
        self.mae_loss_tracker = keras.metrics.Mean(name="mae_loss")
        self.lpips_loss_tracker = keras.metrics.Mean(name="lpips_loss")
        self.cls_loss_tracker = keras.metrics.Mean(name="cls_loss")
        self.seg_loss_tracker = keras.metrics.Mean(name="seg_loss")

        # Advisory: SIGReg is computed on the bottleneck cells, not on the
        # patch grid. With aggressive downsampling the cell count can drop
        # below sigreg_knots.
        if cfg.num_bottleneck_cells < cfg.sigreg_knots:
            logger.warning(
                "CliffordNetPatchVAEV2: num_bottleneck_cells (%d) < "
                "sigreg_knots (%d); SIGReg statistic will have high variance.",
                cfg.num_bottleneck_cells,
                cfg.sigreg_knots,
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        base = list(super().metrics)
        extras = [
            self.loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.sigreg_loss_tracker,
            self.kl_loss_weighted_tracker,
            self.sigreg_loss_weighted_tracker,
            self.mae_loss_tracker,
            self.lpips_loss_tracker,
            self.cls_loss_tracker,
            self.seg_loss_tracker,
        ]
        seen, out = set(), []
        for m in base + extras:
            if id(m) not in seen:
                out.append(m)
                seen.add(id(m))
        return out

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack_inputs(
        inputs: Union[
            keras.KerasTensor, Dict[str, keras.KerasTensor]
        ],
    ) -> Tuple[
        keras.KerasTensor,
        Optional[keras.KerasTensor],
        Optional[keras.KerasTensor],
    ]:
        if isinstance(inputs, dict):
            image = inputs["image"]
            return image, inputs.get("label_cls"), inputs.get("label_seg")
        return inputs, None, None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def call(
        self,
        inputs: Union[
            keras.KerasTensor, Dict[str, keras.KerasTensor]
        ],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        image, label_cls, label_seg = self._unpack_inputs(inputs)

        need_pre_bn = (
            self.cls_head is not None
            or self.seg_head is not None
            or self._lambda_mae > 0.0
            or self.config.mae_mask_ratio > 0.0
        )
        if need_pre_bn:
            mu, log_var, pre_bottleneck, mask = self.encoder(
                image, training=training, output_pre_bottleneck=True
            )
        else:
            mu, log_var = self.encoder(image, training=training)
            pre_bottleneck = None
            mask = None

        z = self.sampling([mu, log_var], training=training)
        logits = self.decoder(z, training=training)

        # ---- Reconstruction loss (mask-weighted if MAE active) ----
        recon_loss, mae_loss = self._compute_recon(image, logits, mask)
        kl_loss = self._compute_kl(mu, log_var)
        sigreg_loss = self._compute_sigreg(z)

        # Pixel-space reconstruction (output activation per recon_loss_type).
        if self.config.recon_loss_type == "bce":
            recon = ops.sigmoid(logits)
        else:
            recon = logits

        # ---- LPIPS perceptual ----
        lpips_loss_val = ops.convert_to_tensor(0.0, dtype="float32")
        if self._lpips is not None and self._lambda_lpips > 0.0:
            lpips_loss_val = ops.mean(self._lpips(image, recon))
            self.add_loss(self._lambda_lpips * lpips_loss_val)

        # ---- VAE losses ----
        self.add_loss(recon_loss)
        if self._lambda_mae > 0.0 and mask is not None:
            self.add_loss(self._lambda_mae * mae_loss)
        self.add_loss(self._beta_kl * kl_loss)
        self.add_loss(self._lambda_sigreg * sigreg_loss)

        # ---- Heads ----
        outputs: Dict[str, keras.KerasTensor] = {
            "reconstruction": recon,
            "z": z,
            "mu": mu,
            "log_var": log_var,
        }

        cls_loss_val = ops.convert_to_tensor(0.0, dtype="float32")
        if self.cls_head is not None:
            assert pre_bottleneck is not None
            logits_cls = self.cls_head(pre_bottleneck, training=training)
            outputs["logits_cls"] = logits_cls
            if label_cls is not None:
                cls_loss_val = self._compute_sparse_ce(label_cls, logits_cls)
                self.add_loss(self._lambda_cls * cls_loss_val)

        seg_loss_val = ops.convert_to_tensor(0.0, dtype="float32")
        if self.seg_head is not None:
            assert pre_bottleneck is not None
            logits_seg = self.seg_head(pre_bottleneck, training=training)
            outputs["logits_seg"] = logits_seg
            if label_seg is not None:
                seg_loss_val = self._compute_sparse_ce(label_seg, logits_seg)
                self.add_loss(self._lambda_seg * seg_loss_val)

        # ---- Tracker updates ----
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.sigreg_loss_tracker.update_state(sigreg_loss)
        self.kl_loss_weighted_tracker.update_state(self._beta_kl * kl_loss)
        self.sigreg_loss_weighted_tracker.update_state(
            self._lambda_sigreg * sigreg_loss
        )
        self.mae_loss_tracker.update_state(mae_loss)
        self.lpips_loss_tracker.update_state(lpips_loss_val)
        self.cls_loss_tracker.update_state(cls_loss_val)
        self.seg_loss_tracker.update_state(seg_loss_val)

        return outputs

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    def _compute_recon(
        self,
        x: keras.KerasTensor,
        logits: keras.KerasTensor,
        mask: Optional[keras.KerasTensor],
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        x_f = ops.cast(x, "float32")
        l_f = ops.cast(logits, "float32")
        if self.config.recon_loss_type == "mse":
            err = ops.square(x_f - l_f)
        else:
            err = (
                ops.maximum(l_f, 0.0)
                - l_f * x_f
                + ops.log1p(ops.exp(-ops.abs(l_f)))
            )

        if mask is None:
            return ops.mean(err), ops.convert_to_tensor(0.0, dtype="float32")

        # Mask lives at the PATCH grid (Hp,Wp). Upsample to pixels.
        pixel_mask = upsample_mask_to_pixels(mask, self.config.patch_size)
        pixel_mask = ops.cast(pixel_mask, "float32")
        denom_vis = ops.maximum(
            ops.sum(1.0 - pixel_mask),
            ops.convert_to_tensor(1.0, dtype="float32"),
        )
        denom_mask = ops.maximum(
            ops.sum(pixel_mask),
            ops.convert_to_tensor(1.0, dtype="float32"),
        )
        err_per_pos = ops.mean(err, axis=-1, keepdims=True)
        recon_vis = ops.sum(err_per_pos * (1.0 - pixel_mask)) / denom_vis
        recon_masked = ops.sum(err_per_pos * pixel_mask) / denom_mask
        return recon_vis, recon_masked

    def _compute_kl(
        self,
        mu: keras.KerasTensor,
        log_var: keras.KerasTensor,
    ) -> keras.KerasTensor:
        mu_f = ops.cast(mu, "float32")
        lv_f = ops.clip(ops.cast(log_var, "float32"), -10.0, 10.0)
        kl_per_cell = -0.5 * ops.sum(
            1.0 + lv_f - ops.square(mu_f) - ops.exp(lv_f),
            axis=-1,
        )
        return ops.mean(kl_per_cell)

    def _compute_sigreg(self, z: keras.KerasTensor) -> keras.KerasTensor:
        z_f = ops.cast(z, "float32")
        shape = ops.shape(z_f)
        b, hb, wb, d = shape[0], shape[1], shape[2], shape[3]
        z_cells = ops.reshape(z_f, (b, hb * wb, d))
        # SIGReg ×N at call site — LESSON from V1.
        return self.sigreg(z_cells) * ops.cast(hb * wb, "float32")

    @staticmethod
    def _compute_sparse_ce(
        labels: keras.KerasTensor,
        logits: keras.KerasTensor,
    ) -> keras.KerasTensor:
        labels_i = ops.cast(labels, "int32")
        logits_f = ops.cast(logits, "float32")
        return ops.mean(
            keras.losses.sparse_categorical_crossentropy(
                labels_i, logits_f, from_logits=True
            )
        )

    # ------------------------------------------------------------------
    # Public encode / decode / sample
    # ------------------------------------------------------------------
    def encode(
        self, x: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        return self.encoder(x, training=False)

    def decode(self, z: keras.KerasTensor) -> keras.KerasTensor:
        logits = self.decoder(z, training=False)
        if self.config.recon_loss_type == "bce":
            return ops.sigmoid(logits)
        return logits

    def sample_from(
        self,
        x: keras.KerasTensor,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> keras.KerasTensor:
        mu, log_var = self.encode(x)
        t = float(temperature)
        eps = keras.random.normal(ops.shape(mu), seed=seed) * t
        z = mu + ops.exp(0.5 * ops.clip(log_var, -10.0, 10.0)) * eps
        return self.decode(z)

    def sample(
        self,
        num_samples: int,
        hb: Optional[int] = None,
        wb: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> keras.KerasTensor:
        cfg = self.config
        hb = cfg.bottleneck_per_side if hb is None else int(hb)
        wb = cfg.bottleneck_per_side if wb is None else int(wb)
        if hb <= 0 or wb <= 0:
            raise ValueError(
                f"hb and wb must be positive, got hb={hb}, wb={wb}"
            )
        eps = keras.random.normal(
            shape=(num_samples, hb, wb, cfg.latent_dim), seed=seed,
        )
        return self.decode(eps)

    # ------------------------------------------------------------------
    # train_step / test_step
    # ------------------------------------------------------------------
    def train_step(self, data: Any) -> Dict[str, Any]:
        x = self._extract_call_input(data)
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
        x = self._extract_call_input(data)
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

    @staticmethod
    def _extract_call_input(data: Any):
        if isinstance(data, dict):
            return data
        if isinstance(data, tuple):
            head = data[0]
            if (
                len(data) >= 2
                and isinstance(data[1], dict)
                and isinstance(head, (tf.Tensor, keras.KerasTensor))
            ):
                merged = {"image": head}
                merged.update(
                    {k: v for k, v in data[1].items() if k.startswith("label_")}
                )
                return merged
            return head
        return data

    # ------------------------------------------------------------------
    # Variants
    # ------------------------------------------------------------------
    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: bool = False,
        **overrides: Any,
    ) -> "CliffordNetPatchVAEV2":
        if variant not in cls.PRESETS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: "
                f"{sorted(cls.PRESETS)}"
            )
        cfg_kwargs = {**cls.PRESETS[variant], **overrides}
        cfg = CliffordNetPatchVAEV2Config(**cfg_kwargs)
        model = cls(config=cfg)
        if pretrained:
            raise NotImplementedError(
                f"No pretrained weights are published for "
                f"cliffordnet_patch_vae_v2 variant '{variant}'."
            )
        return model

    # ------------------------------------------------------------------
    # compute_output_shape (dict)
    # ------------------------------------------------------------------
    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")
        mu_shape, log_var_shape = self.encoder.compute_output_shape(input_shape)
        z_shape = mu_shape
        recon_shape = self.decoder.compute_output_shape(z_shape)
        out: Dict[str, Tuple[Optional[int], ...]] = {
            "reconstruction": recon_shape,
            "z": z_shape,
            "mu": mu_shape,
            "log_var": log_var_shape,
        }
        if self.cls_head is not None:
            out["logits_cls"] = (input_shape[0], self.config.num_classes_cls)
        if self.seg_head is not None:
            out["logits_seg"] = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                self.config.num_classes_seg,
            )
        return out

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
    ) -> "CliffordNetPatchVAEV2":
        config = dict(config)
        cfg_dict = config.pop("config", None)
        cfg = (
            CliffordNetPatchVAEV2Config.from_dict(cfg_dict)
            if cfg_dict is not None
            else CliffordNetPatchVAEV2Config()
        )
        extra = {k: v for k, v in config.items() if k in _KERAS_BASE_KEYS}
        return cls(config=cfg, **extra)


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------


def create_cliffordnet_patch_vae_v2(
    variant: str = "base",
    *,
    pretrained: bool = False,
    **overrides: Any,
) -> CliffordNetPatchVAEV2:
    """Create a :class:`CliffordNetPatchVAEV2` from a named variant.

    Args:
        variant: One of :attr:`CliffordNetPatchVAEV2.PRESETS` keys
            (``"tiny"``, ``"base"``, ``"large"``, ``"xl"``).
        pretrained: Raises :class:`NotImplementedError` — no public
            checkpoints.
        **overrides: Forwarded to :class:`CliffordNetPatchVAEV2Config`.

    Returns:
        Unbuilt :class:`CliffordNetPatchVAEV2`.
    """
    return CliffordNetPatchVAEV2.from_variant(
        variant, pretrained=pretrained, **overrides
    )

"""Top-level :class:`ConvNeXtPatchVAE` model.

Composes :class:`ConvNeXtPatchEncoder` + :class:`Sampling` +
:class:`ConvNeXtPatchDecoder` + :class:`SIGRegLayer`, with a custom
``train_step`` that bypasses ``compile(loss=...)`` and uses ``add_loss``
for the three loss terms (recon + beta_kl * KL_per_patch +
lambda_sigreg * SIGReg).

Anchored decisions (see ``plans/plan_2026-05-25_fb57d478/decisions.md``):

- D-001: explicit ``self.loss_tracker`` via ``keras.metrics.Mean(name="loss")``
  to satisfy the Keras-3.8 `history.history['loss']` contract — anchored
  at the construction site in ``__init__`` (search for
  ``# DECISION plan_2026-05-25_fb57d478/D-001``).
- D-002: SIGReg binding on ``ops.reshape(z, (B, Hp*Wp, latent_dim))``
  (post-reparam, per-image patch distribution). Anchored at the reshape
  site in ``_compute_sigreg`` (search for
  ``# DECISION plan_2026-05-25_fb57d478/D-002``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import keras
import tensorflow as tf
from keras import ops

from dl_techniques.layers.sampling import Sampling
from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.utils.logger import logger

from .config import ConvNeXtPatchVAEConfig
from .decoder import ConvNeXtPatchDecoder
from .encoder import ConvNeXtPatchEncoder


@keras.saving.register_keras_serializable()
class ConvNeXtPatchVAE(keras.Model):
    """ConvNeXt patch-level VAE with SIGReg anti-patch-collapse.

    :param config: :class:`ConvNeXtPatchVAEConfig`. If ``None``, defaults.
    :param kwargs: passthrough to :class:`keras.Model`.
    """

    def __init__(
        self,
        config: Optional[ConvNeXtPatchVAEConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            config = ConvNeXtPatchVAEConfig()
        self.config = config
        cfg = config

        # Materialize the kernel regularizer (if any) once for all sub-layers.
        kreg: Optional[keras.regularizers.Regularizer] = None
        if cfg.kernel_regularizer_config is not None:
            kreg = keras.regularizers.deserialize(cfg.kernel_regularizer_config)

        # --- Sub-modules ---
        self.encoder = ConvNeXtPatchEncoder(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.encoder_depth,
            kernel_size=cfg.kernel_size,
            latent_dim=cfg.latent_dim,
            dropout_rate=cfg.dropout_rate,
            spatial_dropout_rate=cfg.spatial_dropout_rate,
            kernel_regularizer=kreg,
            name="encoder",
        )
        self.sampling = Sampling(name="sampling")
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
        self.sigreg = SIGRegLayer(
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
            name="sigreg",
        )

        # Cached weights pulled from config — used in train_step + call().
        self._beta_kl = float(cfg.beta_kl)
        self._lambda_sigreg = float(cfg.lambda_sigreg)
        self._gamma_clip = cfg.gamma_clip

        # --- Loss component trackers (per-component Mean) ---
        # DECISION plan_2026-05-25_fb57d478/D-001: explicit aggregate `loss`
        # tracker. Keras 3.8 does not auto-create `self.loss_tracker` until
        # `compile(loss=...)` is called; our `train_step` bypasses compiled
        # loss entirely (losses come from `add_loss`). Without this explicit
        # Mean, `history.history['loss']` is pinned at 0.0 — defeats
        # EarlyStopping / ModelCheckpoint(monitor='loss') / training_curves.
        # Mirrors video_jepa plan_2026-05-24_ca745a6c/D-005. See
        # plans/plan_2026-05-25_fb57d478/decisions.md D-001.
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        # NOTE (D-003): `sigreg_loss` tracks the RAW SIGReg statistic
        # (pre-`lambda_sigreg` multiplication) so ablation comparisons
        # across `lambda_sigreg` settings remain on a single scale. The
        # weighted contribution still flows into the aggregate `loss`
        # tracker via `add_loss`. Required by test_sigreg_off_branch.
        self.sigreg_loss_tracker = keras.metrics.Mean(name="sigreg_loss")

        # Edge-case advisory: SIGReg statistic on too-few patches.
        if cfg.num_patches < cfg.sigreg_knots:
            logger.warning(
                "ConvNeXtPatchVAE: num_patches (Hp*Wp = %d) < sigreg_knots "
                "(%d). SIGReg statistic will still produce a valid scalar "
                "but with high variance. Consider larger img_size or "
                "smaller patch_size.",
                cfg.num_patches,
                cfg.sigreg_knots,
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Expose per-component trackers alongside Keras' default metrics.

        Dedup by ``id`` so a metric registered both by Keras (via super)
        and by our custom logic appears once. Mirrors the video_jepa
        pattern.
        """
        base = list(super().metrics)
        extras = [
            self.loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.sigreg_loss_tracker,
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
        """End-to-end forward with `add_loss` of the three components.

        :param inputs: ``(B, H, W, C)``.
        :param training: Standard Keras training flag.
        :return: Dict with keys ``reconstruction``, ``z``, ``mu``,
            ``log_var``. ``reconstruction`` is the pixel-space output
            after the appropriate activation (sigmoid for BCE, identity
            for MSE).
        """
        mu, log_var = self.encoder(inputs, training=training)
        z = self.sampling([mu, log_var], training=training)
        logits = self.decoder(z, training=training)

        # Recon — both branches return per-sample-then-batch-mean scalar
        # so loss magnitude is independent of resolution.
        recon_loss = self._compute_recon(inputs, logits)
        # KL averaged over (B, Hp, Wp). Per-patch mean = resolution-invariant.
        kl_loss = self._compute_kl(mu, log_var)
        # SIGReg on (B, Hp*Wp, latent_dim) view of z (D-002).
        sigreg_loss = self._compute_sigreg(z)

        self.add_loss(recon_loss)
        self.add_loss(self._beta_kl * kl_loss)
        self.add_loss(self._lambda_sigreg * sigreg_loss)

        # Per-component trackers update on every forward (train + eval).
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # Track the RAW SIGReg statistic (see D-003 in __init__).
        self.sigreg_loss_tracker.update_state(sigreg_loss)

        # Pixel-space reconstruction in [0, 1] for BCE branch, raw for MSE.
        if self.config.recon_loss_type == "bce":
            recon = ops.sigmoid(logits)
        else:
            recon = logits

        return {
            "reconstruction": recon,
            "z": z,
            "mu": mu,
            "log_var": log_var,
        }

    # ------------------------------------------------------------------
    # Loss component helpers (float32 internally — mixed-precision safe)
    # ------------------------------------------------------------------
    def _compute_recon(
        self,
        x: keras.KerasTensor,
        logits: keras.KerasTensor,
    ) -> keras.KerasTensor:
        x_f = ops.cast(x, "float32")
        l_f = ops.cast(logits, "float32")
        if self.config.recon_loss_type == "mse":
            # Mean over all axes -> scalar.
            return ops.mean(ops.square(x_f - l_f))
        # BCE with logits — numerically stable formulation.
        # bce = max(l,0) - l*x + log(1+exp(-|l|))
        bce = (
            ops.maximum(l_f, 0.0)
            - l_f * x_f
            + ops.log1p(ops.exp(-ops.abs(l_f)))
        )
        return ops.mean(bce)

    def _compute_kl(
        self,
        mu: keras.KerasTensor,
        log_var: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Per-patch KL: average over (B, Hp, Wp), sum over latent_dim.

        Per-patch averaging makes the loss magnitude resolution-invariant
        — doubling Hp*Wp does not double the loss.
        """
        mu_f = ops.cast(mu, "float32")
        lv_f = ops.cast(log_var, "float32")
        # KL per patch position summed over latent_dim:
        #   kl = -0.5 * sum_d (1 + log_var - mu^2 - exp(log_var))
        kl_per_patch = -0.5 * ops.sum(
            1.0 + lv_f - ops.square(mu_f) - ops.exp(lv_f),
            axis=-1,
        )  # (B, Hp, Wp)
        return ops.mean(kl_per_patch)

    def _compute_sigreg(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """SIGReg on the per-image patch distribution.

        D-002: regularize the same quantity KL targets — the post-reparam
        latent — viewed as a per-image distribution over patches
        ``(B, Hp*Wp, latent_dim)``. N-axis = patch axis; SIGReg averages
        over N internally and we batch-mean implicitly via the layer's
        scalar output.
        """
        z_f = ops.cast(z, "float32")
        shape = ops.shape(z_f)
        B, Hp, Wp, D = shape[0], shape[1], shape[2], shape[3]
        # DECISION plan_2026-05-25_fb57d478/D-002:
        # Bind SIGReg on (B, Hp*Wp, latent_dim) of post-reparam z. Option A
        # (per-image patch distribution) applied to Option C (post-reparam).
        # See plans/plan_2026-05-25_fb57d478/decisions.md D-002 for the
        # rejected alternatives (per-position N=batch; pre-reparam encoder
        # grid). Reshape into the (..., N, D) shape SIGRegLayer expects.
        z_patches = ops.reshape(z_f, (B, Hp * Wp, D))
        return self.sigreg(z_patches)

    # ------------------------------------------------------------------
    # Public encode / decode / sample API
    # ------------------------------------------------------------------
    def encode(
        self, x: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Encode pixels into per-patch ``(mu, log_var)``.

        :param x: ``(B, H, W, C)`` with ``H % patch_size == 0``.
        :return: Tuple ``(mu, log_var)``, each ``(B, Hp, Wp, latent_dim)``.
        """
        return self.encoder(x, training=False)

    def decode(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Decode a per-patch latent grid back to pixels.

        Applies the appropriate output activation based on
        ``recon_loss_type``.
        """
        logits = self.decoder(z, training=False)
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
        """Sample images from the unit-Gaussian prior at any patch grid.

        :param num_samples: Number of images to generate.
        :param hp: Patch grid height. Defaults to ``config.patches_per_side``.
        :param wp: Patch grid width. Defaults to ``config.patches_per_side``.
        :param seed: Optional seed for reproducibility.
        :return: ``(num_samples, hp*patch_size, wp*patch_size, img_channels)``.
        """
        cfg = self.config
        hp = cfg.patches_per_side if hp is None else int(hp)
        wp = cfg.patches_per_side if wp is None else int(wp)
        if hp <= 0 or wp <= 0:
            raise ValueError(
                f"hp and wp must be positive, got hp={hp}, wp={wp}"
            )
        eps = keras.random.normal(
            shape=(num_samples, hp, wp, cfg.latent_dim),
            seed=seed,
        )
        return self.decode(eps)

    # ------------------------------------------------------------------
    # Custom train_step (mirrors video_jepa pattern)
    # ------------------------------------------------------------------
    def train_step(self, data: Any) -> Dict[str, Any]:
        """One training step: forward, sum add_loss outputs, apply grads.

        ``data`` may be a single tensor (VAE — no labels) or an
        ``(inputs, _)`` tuple (autoencoder pipelines often pass
        ``x, x``). The label is unused — losses come from ``add_loss``
        inside :meth:`call`.
        """
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
        # Optional symmetric gradient clip per cfg.gamma_clip (mirrors
        # vae/model.py:696 idiom; disabled when None).
        if self._gamma_clip is not None:
            c = float(self._gamma_clip)
            grads = [
                None if g is None else ops.clip(g, -c, c) for g in grads
            ]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # D-001 contract: update Keras default loss_tracker so
        # `history.history['loss']` reflects the true aggregate loss.
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
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], custom_objects=None
    ) -> "ConvNeXtPatchVAE":
        cfg_dict = config.pop("config", None)
        cfg = (
            ConvNeXtPatchVAEConfig.from_dict(cfg_dict)
            if cfg_dict is not None
            else ConvNeXtPatchVAEConfig()
        )
        return cls(config=cfg, **config)

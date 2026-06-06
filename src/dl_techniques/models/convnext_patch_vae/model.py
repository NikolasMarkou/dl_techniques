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

import keras
from keras import ops
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------

from dl_techniques.layers.sampling import (
    Sampling,
    create_sampling_layer,
    vmf_kl_divergence,
)
from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.utils.logger import logger

from .config import ConvNeXtPatchVAEConfig
from .decoder import ConvNeXtPatchDecoder
from .encoder import ConvNeXtPatchEncoder

# ------------------------------------------------------------------

# Keys produced by ``keras.Model.get_config()`` that are forwardable
# straight to ``keras.Model.__init__``. Used by :meth:`from_config` to
# drop unknown super-class keys before kwargs forwarding (defensive —
# matches the video_jepa pattern, plan_ca745a6c).
_KERAS_BASE_KEYS = {"name", "trainable", "dtype"}


@keras.saving.register_keras_serializable()
class ConvNeXtPatchVAE(keras.Model):
    """ConvNeXt patch-level VAE with SIGReg anti-patch-collapse.

    Args:
        config: :class:`ConvNeXtPatchVAEConfig`. If ``None``, defaults
            are used.
        **kwargs: Passthrough to :class:`keras.Model`.

    Input shape:
        4D tensor with shape ``(B, H, W, C)`` where ``H`` and ``W`` are
        multiples of ``config.patch_size``.

    Output shape:
        Dict with keys ``reconstruction`` ``(B, H, W, C)``, and ``z`` /
        ``mu`` / ``log_var`` each ``(B, Hp, Wp, latent_dim)``.
    """

    #: Named variant presets — config overrides only. All other fields
    #: inherit :class:`ConvNeXtPatchVAEConfig` defaults. Mirrors the
    #: ``models/{bert, resnet, tree_transformer, cliffordnet, vit, ...}``
    #: ``PRESETS`` convention.
    PRESETS: Dict[str, Dict[str, Any]] = {
        "tiny":  {"embed_dim": 64,  "encoder_depth": 2, "decoder_depth": 2, "latent_dim": 8},
        "base":  {"embed_dim": 128, "encoder_depth": 4, "decoder_depth": 4, "latent_dim": 16},
        "large": {"embed_dim": 192, "encoder_depth": 6, "decoder_depth": 6, "latent_dim": 32},
    }

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
            sampling_type=cfg.sampling_type,
            name="encoder",
        )
        # DECISION plan_2026-06-06_38aa045e/D-001: dispatch the sampler via the
        # registry factory, keeping the layer NAME "sampling" in BOTH modes
        # (I3 — name-based sub-model extraction stays valid). For gaussian this
        # returns Sampling(name="sampling"), behaviorally identical to the bare
        # construction it replaces; for vmf it returns VMFSampling. Do NOT
        # rename the sampler per-mode. See decisions.md D-001.
        self.sampling = create_sampling_layer(
            cfg.sampling_type, name="sampling"
        )
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
        # Weighted variants (beta_kl * kl, lambda_sigreg * sigreg) so the
        # actual optimizer contribution is visible alongside the raw values.
        self.kl_loss_weighted_tracker = keras.metrics.Mean(name="kl_loss_weighted")
        self.sigreg_loss_weighted_tracker = keras.metrics.Mean(name="sigreg_loss_weighted")

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
    # Compile (vmf XLA opt-out)
    # ------------------------------------------------------------------
    def compile(self, *args: Any, **kwargs: Any) -> None:
        # DECISION plan_2026-06-06_38aa045e/D-001: vMF's keras.random.beta ->
        # StatelessRandomGammaV3 has NO XLA-GPU kernel (TF 2.18); force-disable
        # XLA for vmf on EVERY compile path (direct .compile(), load_model()
        # recompile) so GPU save/load + fit don't crash. This MUST travel with
        # the model so a reloaded vmf model stays jit_compile=False (mirrors
        # vae/model.py:314-337). gaussian keeps the caller's jit_compile.
        # See decisions.md D-001.
        if self.config.sampling_type == "vmf":
            kwargs["jit_compile"] = False
        return super().compile(*args, **kwargs)

    def compile_from_config(self, config: Any) -> "ConvNeXtPatchVAE":
        # Overriding compile() makes Keras route load_model()'s recompile
        # through compile_from_config (else it warns + skips recompile, leaving
        # a reloaded vmf model with stale jit_compile="auto" that XLA-crashes on
        # a later GPU .fit()). Funnel through our compile() so the vmf
        # jit_compile=False opt-out survives reload.
        config = keras.saving.deserialize_keras_object(config)
        self.compile(**config)
        return self

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
            self.kl_loss_weighted_tracker,
            self.sigreg_loss_weighted_tracker,
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

        Args:
            inputs: ``(B, H, W, C)``.
            training: Standard Keras training flag.

        Returns:
            Dict with keys ``reconstruction``, ``z``, ``mu``,
            ``log_var``. ``reconstruction`` is the pixel-space output
            after the appropriate activation (sigmoid for BCE, identity
            for MSE).
        """
        # For gaussian, `second` is the per-patch log_var (B,Hp,Wp,latent_dim);
        # for vmf it is the per-patch concentration kappa (B,Hp,Wp,1).
        mu, second = self.encoder(inputs, training=training)
        if self.config.sampling_type == "vmf":
            # DECISION plan_2026-06-06_38aa045e/D-001: apply vMF PER PATCH —
            # reshape the 4D spatial latent to (B*Hp*Wp, D), run VMFSampling on
            # the flat per-patch rows, then reshape z back to (B,Hp,Wp,D). This
            # preserves the resolution-agnostic 4D latent (I2). Do NOT global-
            # flatten to one S^{Hp*Wp*D-1} sphere (rejected — it bakes Hp*Wp
            # into the latent dim and breaks resolution-agnosticism). D is the
            # static config int; the patch count uses (-1, D) for the dynamic
            # dim. See decisions.md D-001.
            D = self.config.latent_dim
            mu_flat = ops.reshape(mu, (-1, D))
            kappa_flat = ops.reshape(second, (-1, 1))
            z_flat = self.sampling([mu_flat, kappa_flat], training=training)
            shp = ops.shape(mu)
            z = ops.reshape(z_flat, (shp[0], shp[1], shp[2], D))
        else:
            z = self.sampling([mu, second], training=training)
        logits = self.decoder(z, training=training)

        # Recon — both branches return per-sample-then-batch-mean scalar
        # so loss magnitude is independent of resolution.
        recon_loss = self._compute_recon(inputs, logits)
        # KL averaged over (B, Hp, Wp). Per-patch mean = resolution-invariant.
        kl_loss = self._compute_kl(mu, second)
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
        # Weighted contributions — actual optimizer signal per component.
        self.kl_loss_weighted_tracker.update_state(self._beta_kl * kl_loss)
        self.sigreg_loss_weighted_tracker.update_state(self._lambda_sigreg * sigreg_loss)

        # Pixel-space reconstruction in [0, 1] for BCE branch, raw for MSE.
        if self.config.recon_loss_type == "bce":
            recon = ops.sigmoid(logits)
        else:
            recon = logits

        # For vmf the "log_var" slot carries the per-patch concentration kappa
        # (B,Hp,Wp,1), NOT a log-variance (E4). The key name is held constant
        # for output-contract stability across modes.
        return {
            "reconstruction": recon,
            "z": z,
            "mu": mu,
            "log_var": second,
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
        second: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Per-patch KL: average over (B, Hp, Wp), sum over latent_dim.

        Per-patch averaging makes the loss magnitude resolution-invariant
        — doubling Hp*Wp does not double the loss.

        For gaussian, ``second`` is the per-patch log_var
        (B,Hp,Wp,latent_dim). For vmf it is the per-patch concentration kappa
        (B,Hp,Wp,1) and the KL is the closed-form vMF->uniform-sphere
        divergence averaged over all B*Hp*Wp patches.
        """
        if self.config.sampling_type == "vmf":
            # DECISION plan_2026-06-06_38aa045e/D-001: per-patch analytic vMF KL.
            # vmf_kl_divergence depends only on (kappa, dim); flatten every
            # patch's scalar kappa to (B*Hp*Wp, 1) and mean over all patches.
            # Reuse the verified sampling.py helper — do NOT reimplement the
            # Bessel KL here. See decisions.md D-001.
            kappa_flat = ops.reshape(ops.cast(second, "float32"), (-1, 1))
            return ops.mean(
                vmf_kl_divergence(kappa_flat, dim=self.config.latent_dim)
            )

        log_var = second
        mu_f = ops.cast(mu, "float32")
        lv_f = ops.clip(ops.cast(log_var, "float32"), -10.0, 10.0)
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
        # DECISION plan_2026-05-27_1a9e3221/D-001: multiply by N=Hp*Wp so
        # SIGReg penalty is O(N) — matching KL's resolution-invariant design.
        # Without this, effective SIGReg pressure collapses 16× per resolution
        # doubling (1024× weaker at 256x256 vs CIFAR, H21).
        return self.sigreg(z_patches) * ops.cast(Hp * Wp, "float32")

    # ------------------------------------------------------------------
    # Public encode / decode / sample API
    # ------------------------------------------------------------------
    def encode(
        self, x: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Encode pixels into per-patch ``(mu, log_var)``.

        Args:
            x: ``(B, H, W, C)`` with ``H % patch_size == 0``.

        Returns:
            Tuple ``(mu, log_var)``, each shaped
            ``(B, Hp, Wp, latent_dim)``.
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

    def sample_from(
        self,
        x: keras.KerasTensor,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> keras.KerasTensor:
        """Coherent sampling around a real anchor ``x`` (one-line API).

        Reparameterizes from the encoder's posterior at temperature
        ``t``: ``z = mu + t * exp(0.5 * log_var) * eps``.

        - ``temperature=0.0`` -> deterministic reconstruction (``decode(mu)``).
        - ``temperature=1.0`` -> matches the VAE prior scale.
        - ``temperature>1.0`` -> more diverse variations.

        Provides a one-line sampling API on the single-scale model.

        Args:
            x: Real anchor image batch ``(B, H, W, C)``.
            temperature: Reparameterization noise scale.
            seed: Optional RNG seed.

        Returns:
            ``(B, H, W, C)`` coherent reconstruction / variation of ``x``.
        """
        mu, second = self.encode(x)
        t = float(temperature)
        if self.config.sampling_type == "vmf":
            # vMF posterior is on the unit sphere — there is no additive
            # Gaussian reparameterization. Run the per-patch vMF sampler at the
            # encoded concentration (temperature is not a Gaussian scale here;
            # t=0 collapses to the deterministic mean direction decode(mu_hat)).
            D = self.config.latent_dim
            shp = ops.shape(mu)
            if t == 0.0:
                # Deterministic: decode the unit-normalized mean direction.
                norm = ops.sqrt(
                    ops.sum(ops.square(mu), axis=-1, keepdims=True)
                )
                z = mu / ops.maximum(norm, 1e-12)
            else:
                mu_flat = ops.reshape(mu, (-1, D))
                kappa_flat = ops.reshape(second, (-1, 1))
                z_flat = self.sampling([mu_flat, kappa_flat], training=False)
                z = ops.reshape(z_flat, (shp[0], shp[1], shp[2], D))
            return self.decode(z)
        log_var = second
        eps = keras.random.normal(ops.shape(mu), seed=seed) * t
        z = mu + ops.exp(0.5 * ops.clip(log_var, -10.0, 10.0)) * eps
        return self.decode(z)

    def sample(
        self,
        num_samples: int,
        hp: Optional[int] = None,
        wp: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> keras.KerasTensor:
        """Sample images from the unit-Gaussian prior at any patch grid.

        Args:
            num_samples: Number of images to generate.
            hp: Patch grid height. Defaults to
                ``config.patches_per_side``.
            wp: Patch grid width. Defaults to
                ``config.patches_per_side``.
            seed: Optional seed for reproducibility.

        Returns:
            Tensor of shape
            ``(num_samples, hp * patch_size, wp * patch_size, img_channels)``.
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
        if cfg.sampling_type == "vmf":
            # The vMF prior (kappa=0) IS the uniform distribution on the unit
            # sphere S^{D-1}. Marsaglia: L2-normalize the per-patch Gaussian
            # draw onto the unit sphere (per patch, over the last axis) — NOT
            # an N(0,I) draw, which is the wrong prior for the spherical latent.
            norm = ops.sqrt(ops.sum(ops.square(eps), axis=-1, keepdims=True))
            eps = eps / ops.maximum(norm, 1e-12)
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
    # Named variants
    # ------------------------------------------------------------------
    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: bool = False,
        **overrides: Any,
    ) -> "ConvNeXtPatchVAE":
        """Build a named variant from :attr:`PRESETS`.

        Args:
            variant: One of :attr:`PRESETS` keys (``"tiny"``, ``"base"``,
                ``"large"``).
            pretrained: If ``True``, attempt to load published weights via
                :meth:`_download_weights`. No public checkpoints exist —
                this raises :class:`NotImplementedError` (see D-001).
            **overrides: Forwarded to :class:`ConvNeXtPatchVAEConfig`,
                taking precedence over the preset values.

        Returns:
            Unbuilt :class:`ConvNeXtPatchVAE`. Caller must invoke the
            model on a dummy batch (or call ``.build(...)``) before
            ``save`` / ``load_weights``.

        Raises:
            ValueError: ``variant`` not in :attr:`PRESETS`.
            NotImplementedError: ``pretrained=True`` (no public weights).
        """
        if variant not in cls.PRESETS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: "
                f"{sorted(cls.PRESETS)}"
            )
        cfg_kwargs = {**cls.PRESETS[variant], **overrides}
        cfg = ConvNeXtPatchVAEConfig(**cfg_kwargs)
        model = cls(config=cfg)
        if pretrained:
            try:
                weights_path = cls._download_weights(variant)
                model.load_weights(weights_path)
            except (IOError, OSError, ValueError) as e:
                logger.error(
                    "Pretrained weight load failed for variant '%s': %s",
                    variant,
                    e,
                )
                raise
        return model

    @classmethod
    def _download_weights(cls, variant: str) -> str:
        """Resolve a pretrained-weights path for ``variant``.

        No public checkpoints are published for this VAE. Per
        ``D-001`` (and the repo-wide convention shared by
        ``models/{bert, resnet, tree_transformer, cliffordnet, vit, ...}``),
        loud failure beats silent random-init.

        Raises:
            NotImplementedError: Always — no public checkpoints exist.
        """
        # DECISION plan_2026-05-25_8faec5b6/D-001: no public checkpoints;
        # loud failure beats silent random-init. Mirrors bert/bert.py
        # (plan_9357982a/D-001), tree_transformer (plan_3c3ed037),
        # resnet, vit, cliffordnet. See
        # plans/plan_2026-05-25_8faec5b6/decisions.md D-001.
        raise NotImplementedError(
            f"No pretrained weights are published for convnext_patch_vae "
            f"variant '{variant}'."
        )

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------
    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Return the dict-of-shapes matching :meth:`call` output.

        Delegates to the encoder/decoder shape inference (guide §3.4
        Pattern 4) so the per-patch grid math stays in one place.

        Args:
            input_shape: ``(B, H, W, C)``. ``H`` and ``W`` may be ``None``.

        Returns:
            Dict with keys ``reconstruction``, ``z``, ``mu``, ``log_var``
            and shape tuples matching the runtime ``call`` output.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape, got {input_shape}"
            )
        mu_shape, log_var_shape = self.encoder.compute_output_shape(input_shape)
        # z has the same shape as mu (Sampling preserves shape).
        z_shape = mu_shape
        recon_shape = self.decoder.compute_output_shape(z_shape)
        return {
            "reconstruction": recon_shape,
            "z": z_shape,
            "mu": mu_shape,
            "log_var": log_var_shape,
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
    ) -> "ConvNeXtPatchVAE":
        config = dict(config)
        cfg_dict = config.pop("config", None)
        cfg = (
            ConvNeXtPatchVAEConfig.from_dict(cfg_dict)
            if cfg_dict is not None
            else ConvNeXtPatchVAEConfig()
        )
        # Defensive narrowing: forward only the keras.Model super-keys
        # we know are safe — drops any future-added serialized field
        # that would otherwise leak into the ctor as an unknown kwarg.
        extra = {k: v for k, v in config.items() if k in _KERAS_BASE_KEYS}
        return cls(config=cfg, **extra)


# ----------------------------------------------------------------------
# Module-level factory
# ----------------------------------------------------------------------
# DECISION plan_2026-05-25_8faec5b6/D-002: bare module-level factory
# mirrors the surface of models/{bert, resnet, tree_transformer,
# cliffordnet, vit, depth_anything, prism, lewm, gpt2}. Trades a one-line
# parallel entry-point at the cost of mild API duplication, in exchange
# for parity with the rest of the repo. See
# plans/plan_2026-05-25_8faec5b6/decisions.md D-002.
def create_convnext_patch_vae(
    variant: str = "base",
    *,
    pretrained: bool = False,
    **overrides: Any,
) -> ConvNeXtPatchVAE:
    """Create a :class:`ConvNeXtPatchVAE` from a named variant.

    Thin module-level delegate to :meth:`ConvNeXtPatchVAE.from_variant`.

    Args:
        variant: One of :attr:`ConvNeXtPatchVAE.PRESETS` keys
            (``"tiny"``, ``"base"``, ``"large"``).
        pretrained: If ``True``, attempt to load published weights.
            Currently raises :class:`NotImplementedError` (no public
            checkpoints — see D-001).
        **overrides: Forwarded to :class:`ConvNeXtPatchVAEConfig`,
            taking precedence over the preset values.

    Returns:
        Unbuilt :class:`ConvNeXtPatchVAE`.

    Raises:
        ValueError: ``variant`` not in :attr:`ConvNeXtPatchVAE.PRESETS`.
        NotImplementedError: ``pretrained=True``.
    """
    return ConvNeXtPatchVAE.from_variant(
        variant, pretrained=pretrained, **overrides
    )

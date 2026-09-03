"""Top-level Video-JEPA-Clifford model.

`VideoJEPA` composes `VideoJEPACliffordEncoder`, `VideoJEPAPredictor`, and a
middle-placement `SIGRegLayer`. It predicts future frame embeddings rather
than pixels, following the JEPA framing: no pixel decoder is needed because
loss and evaluation both operate in the encoder's own latent space. A frozen,
EMA-updated target encoder supplies the prediction targets, which avoids the
near-time-invariant collapse a live target produces; SIGReg additionally
guards against representation rank collapse.

`call` takes `{"pixels": (B, T, H, W, C)}`, encodes it to `z: (B, T, H_p, W_p,
D)`, runs the predictor, and adds the next-frame MSE, an optional
tube-masked-prediction loss, and the SIGReg loss via `add_loss`, returning
the raw prediction. Streaming inference uses `stream_reset` and `stream_step`
to run the same predictor over a rolling `K`-frame buffer, amortized O(1) per
call once the buffer is full.

References:
    - Assran et al., 2023. Self-Supervised Learning from Images with a
      Joint-Embedding Predictive Architecture (I-JEPA). CVPR 2023.
      (https://arxiv.org/abs/2301.08243)
    - Bardes et al., 2024. Revisiting Feature Prediction for Learning Visual
      Representations from Video (V-JEPA). (https://arxiv.org/abs/2404.08471)
    - LeCun, 2022. A Path Towards Autonomous Machine Intelligence.
    - Hestenes and Sobczyk, 1984. Clifford Algebra to Geometric Calculus.
    - Grill et al., 2020. Bootstrap Your Own Latent (BYOL). NeurIPS 2020.
      (https://arxiv.org/abs/2006.07733)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import keras
import tensorflow as tf
from keras import ops

from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.utils.logger import logger

from .config import VideoJEPAConfig
from .encoder import VideoJEPACliffordEncoder
from .masking import TubeMaskGenerator
from .predictor import VideoJEPAPredictor
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.video_jepa.model")
class VideoJEPA(keras.Model):
    """Video-JEPA-Clifford top-level model, pixels-only.

    Architecture (training forward):

    .. code-block:: text

        pixels [B, T, H, W, C]
        │
        ┌─────▼─────┐         ┌───────────────┐
        │ encoder     │         │ target_encoder │  EMA-updated, frozen
        └─────┬─────┘         └───────┬───────┘
              ▼                        ▼
        z [B, T, H_p, W_p, D]    z_target (loss target only)
              │
              ▼
        ┌─────────────┐
        │ predictor     │  factorized spatial + causal-temporal
        └─────┬───────┘
              ▼
        pred [B, T, H_p, W_p, D]
              │
              ├──► pred_head_h(pred[:, :-h]) vs z_target[:, h:]   MSE per horizon
              ├──► mask-prediction loss (optional, tube-masked)
              └──► SIGReg(pred.reshape(B*T, N, D))
              ▼
        pred [B, T, H_p, W_p, D]  (returned)

    :param config: :class:`VideoJEPAConfig`. Uses the default config when `None`.
    :type config: Optional[VideoJEPAConfig]
    :param kwargs: Forwarded to :class:`keras.Model`.
    """

    def __init__(
        self,
        config: Optional[VideoJEPAConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            config = VideoJEPAConfig()
        self.config = config
        cfg = config

        # --- Sub-modules ---
        self.encoder = VideoJEPACliffordEncoder(
            embed_dim=cfg.embed_dim,
            patch_size=cfg.patch_size,
            img_size=cfg.img_size,
            img_channels=cfg.img_channels,
            depth=cfg.encoder_clifford_depth,
            shifts=tuple(cfg.encoder_shifts),
            dropout_rate=cfg.dropout_rate,
            name="encoder",
        )
        # DECISION plan_2026-05-23_15151c75/D-001: a live target encoder let
        # encoder/predictor co-adapt into a near-time-invariant map (trained
        # model 84-300x worse than identity at every horizon); EMA decouples them. See decisions.md.
        self.target_encoder = VideoJEPACliffordEncoder(
            embed_dim=cfg.embed_dim,
            patch_size=cfg.patch_size,
            img_size=cfg.img_size,
            img_channels=cfg.img_channels,
            depth=cfg.encoder_clifford_depth,
            shifts=tuple(cfg.encoder_shifts),
            dropout_rate=cfg.dropout_rate,
            name="target_encoder",
        )
        # Freeze target — no gradient flows through this branch. Set here
        # in __init__ (NOT after first forward) so the layer is marked
        # non-trainable before any build / variable tracking happens; this
        # keeps its weights out of ``self.trainable_variables`` reliably.
        self.target_encoder.trainable = False
        self.predictor = VideoJEPAPredictor(
            embed_dim=cfg.embed_dim,
            num_frames_max=max(cfg.num_frames, cfg.history_size_k),
            patches_per_side=cfg.patches_per_side,
            depth=cfg.predictor_depth,
            num_heads=cfg.predictor_num_heads,
            dim_head=cfg.predictor_dim_head,
            mlp_dim=cfg.predictor_mlp_dim,
            shifts=tuple(cfg.predictor_shifts),
            dropout_rate=cfg.dropout_rate,
            name="predictor",
        )
        self.sigreg = SIGRegLayer(
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
            name="sigreg",
        )
        self._sigreg_weight = cfg.sigreg_weight

        # Mask generator is stateless and always instantiated, so save/load
        # round-trips the same weight topology regardless of the enabled flag.
        self.mask_gen = TubeMaskGenerator(
            mask_ratio=cfg.mask_ratio,
            patches_per_side=cfg.patches_per_side,
            name="tube_mask_gen",
        )
        # Zero-init per MAE convention; unused when mask_prediction_enabled=False.
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(cfg.embed_dim,),
            initializer="zeros",
            trainable=True,
        )

        # One pointwise Dense (no bias) per prediction horizon. Pointwise, so
        # a head cannot break causality.
        self.pred_heads: List[keras.layers.Dense] = [
            keras.layers.Dense(
                cfg.embed_dim,
                use_bias=False,
                name=f"pred_head_h{h}",
            )
            for h in cfg.predict_horizons
        ]

        # next_frame_loss_tracker keeps its name for CSV back-compat; it now
        # logs the combined (mean over horizons) loss.
        self.next_frame_loss_tracker = keras.metrics.Mean(
            name="next_frame_loss"
        )
        self.per_horizon_trackers: List[keras.metrics.Mean] = [
            keras.metrics.Mean(name=f"next_frame_loss_h{h}")
            for h in cfg.predict_horizons
        ]
        self.mask_loss_tracker = keras.metrics.Mean(name="mask_loss")
        self.sigreg_loss_tracker = keras.metrics.Mean(name="sigreg_loss")
        # DECISION plan_2026-05-24_ca745a6c/D-005: explicit aggregate loss
        # tracker -- Keras does not auto-create one until compile(loss=...),
        # and train_step bypasses compiled loss entirely (add_loss only). See decisions.md.
        self.loss_tracker = keras.metrics.Mean(name="loss")

        # Streaming buffer: not a weight, reset per sequence.
        self._stream_buf: Optional[Any] = None

        # DECISION plan_2026-05-23_15151c75/D-001: _ema_step is a non-trainable
        # weight so cosine-schedule progress survives reload; _ema_total_steps
        # defaults to 1.0 so cosine math stays safe if the trainer never sets it. See decisions.md.
        self._ema_step = self.add_weight(
            name="ema_step",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )
        self._ema_total_steps: float = 1.0
        # Logged every train_step under "ema_m" so cosine schedules are
        # visible in CSVLogger / history.
        self.ema_m_tracker = keras.metrics.Mean(name="ema_m")
        # Weight-space L2 divergence ratio between target and online encoders
        # (BYOL/MoCo convention); see _compute_ema_divergence and D-001 below.
        self.ema_divergence_tracker = keras.metrics.Mean(name="ema_divergence")

        # Multi-horizon prediction without a strong EMA target is a documented
        # head-collapse failure mode: heads converge to the same value.
        if len(cfg.predict_horizons) >= 2 and cfg.ema_momentum < 0.5:
            logger.warning(
                "VideoJEPA: multi-horizon (len(predict_horizons)=%d) with "
                "weak EMA target (ema_momentum=%.4f < 0.5) is the documented "
                "head-collapse regime — heads converge to the same value. "
                "Use ema_momentum >= 0.996 (default) or single horizon. "
                "See src/train/video_jepa/README.md 'Known issues'.",
                len(cfg.predict_horizons),
                cfg.ema_momentum,
            )

        # Force-build both encoders eagerly with a dummy batch so the lazy-build
        # dance does not happen inside call() under TF graph tracing. from_config
        # re-runs this, but load_model then overwrites weights from disk anyway.
        import numpy as _np
        dummy = _np.zeros(
            (1, cfg.img_size, cfg.img_size, cfg.img_channels),
            dtype=_np.float32,
        )
        _ = self.encoder(dummy, training=False)
        _ = self.target_encoder(dummy, training=False)
        self.target_encoder.set_weights(self.encoder.get_weights())

    @property
    def metrics(self) -> list:
        """Per-loss trackers, so `fit()` logs each loss alongside `loss`.

        :return: Deduplicated list of tracked metrics.
        :rtype: list
        """
        base = list(super().metrics)
        extras = [
            self.loss_tracker,
            self.next_frame_loss_tracker,
            *self.per_horizon_trackers,
            self.mask_loss_tracker,
            self.sigreg_loss_tracker,
            self.ema_m_tracker,
            self.ema_divergence_tracker,
        ]
        # Dedupe while preserving order.
        seen = set()
        out = []
        for m in base + extras:
            if id(m) not in seen:
                out.append(m)
                seen.add(id(m))
        return out

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def encode_frames(
        self, pixels: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Encode a pixel tensor through the online encoder.

        :param pixels: Input tensor, shape `(B, T, H, W, C)`.
        :param training: Forwarded to the encoder's BatchNorm/dropout sub-layers.
        :return: Encoded tensor, shape `(B, T, H_p, W_p, D)`.
        :rtype: keras.KerasTensor
        """
        cfg = self.config
        shape = ops.shape(pixels)
        B, T = shape[0], shape[1]
        H, W, C = cfg.img_size, cfg.img_size, cfg.img_channels
        flat = ops.reshape(pixels, (B * T, H, W, C))
        feat = self.encoder(flat, training=training)
        Hp = cfg.patches_per_side
        return ops.reshape(feat, (B, T, Hp, Hp, cfg.embed_dim))

    def encode_frames_target(self, pixels: keras.KerasTensor) -> keras.KerasTensor:
        """Encode pixels through the frozen EMA target encoder.

        Always runs with `training=False`; the caller wraps the result in
        `ops.stop_gradient`.

        :param pixels: Input tensor, shape `(B, T, H, W, C)`.
        :return: Encoded tensor, shape `(B, T, H_p, W_p, D)`.
        :rtype: keras.KerasTensor
        """
        cfg = self.config
        shape = ops.shape(pixels)
        B, T = shape[0], shape[1]
        H, W, C = cfg.img_size, cfg.img_size, cfg.img_channels
        flat = ops.reshape(pixels, (B * T, H, W, C))
        feat = self.target_encoder(flat, training=False)
        Hp = cfg.patches_per_side
        return ops.reshape(feat, (B, T, Hp, Hp, cfg.embed_dim))

    def set_ema_total_steps(self, n: int) -> None:
        """Set the total step count for the cosine EMA schedule.

        Harmless for `ema_schedule="none"`. The trainer calls this once
        before `fit()` so the schedule covers the whole run.

        :param n: Total training steps.
        :type n: int
        """
        self._ema_total_steps = float(max(int(n), 1))

    def sync_target_to_online(self) -> None:
        """Copy `encoder.weights` into `target_encoder.weights` bitwise.

        The constructor runs this once after a dummy build; also useful for
        tests that want to re-sync after manual weight edits.
        """
        self.target_encoder.set_weights(self.encoder.get_weights())

    def _current_momentum(self):
        """Return the EMA momentum for the current step, as a scalar tensor.

        `"none"` returns the constant `cfg.ema_momentum`. `"cosine"` ramps
        from `m0` to `1.0` across `_ema_total_steps` via a half-cosine,
        clamped to `[m0, 1.0]`. Kept as ops so `train_step` traces cleanly.

        :return: Scalar float32 tensor.
        :rtype: Any
        """
        cfg = self.config
        m0 = ops.convert_to_tensor(float(cfg.ema_momentum), dtype="float32")
        if cfg.ema_schedule == "none":
            return m0
        step = ops.cast(self._ema_step, "float32")
        total = ops.convert_to_tensor(
            max(float(self._ema_total_steps), 1.0), dtype="float32",
        )
        progress = ops.minimum(step / total, 1.0)
        pi = ops.convert_to_tensor(math.pi, dtype="float32")
        one = ops.convert_to_tensor(1.0, dtype="float32")
        m = m0 + (one - m0) * (one - ops.cos(pi * progress)) / 2.0
        m = ops.minimum(ops.maximum(m, m0), one)
        return m

    # DECISION plan_2026-05-24_aebd4cbb/D-001: weight-space L2 ratio (BYOL/MoCo
    # convention), not per-layer cosine or feature-space drift on a probe batch.
    # Sustained >1.0 signals online/target collapse. See decisions.md.
    def _compute_ema_divergence(self):
        """Weight-space L2 divergence ratio between target and online.

        Computes `sqrt(sum((t_w - e_w)^2)) / (sqrt(sum(e_w^2)) + 1e-12)`
        across all paired weights, cast to float32 so mixed-precision runs
        still produce a stable value.

        :return: Scalar float32 tensor.
        :rtype: Any
        """
        diff_sq_sum = ops.convert_to_tensor(0.0, dtype="float32")
        e_sq_sum = ops.convert_to_tensor(0.0, dtype="float32")
        for t_w, e_w in zip(self.target_encoder.weights, self.encoder.weights):
            t_f = ops.cast(t_w, "float32")
            e_f = ops.cast(e_w, "float32")
            diff_sq_sum = diff_sq_sum + ops.sum(ops.square(t_f - e_f))
            e_sq_sum = e_sq_sum + ops.sum(ops.square(e_f))
        eps = ops.convert_to_tensor(1e-12, dtype="float32")
        return ops.sqrt(diff_sq_sum) / (ops.sqrt(e_sq_sum) + eps)

    def _ema_update(self) -> None:
        """Apply one EMA step: ``t <- m * t + (1 - m) * e`` per weight."""
        m = self._current_momentum()
        one_minus_m = 1.0 - m
        for t_w, e_w in zip(self.target_encoder.weights, self.encoder.weights):
            t_w.assign(m * t_w + one_minus_m * e_w)
        self._ema_step.assign(self._ema_step + 1.0)
        self.ema_m_tracker.update_state(m)
        self.ema_divergence_tracker.update_state(self._compute_ema_divergence())

    # ------------------------------------------------------------------
    # Explicit build
    # ------------------------------------------------------------------
    @staticmethod
    def _require_pixels(mapping: Any) -> Any:
        """Return ``mapping["pixels"]``, or raise the model's own contract error.

        Interface contract (call sites: :meth:`build` and :meth:`call`). Shared
        because both take the SAME dict-shaped argument -- one a nest of shapes,
        one a nest of tensors -- and both must fail with the same named
        ``ValueError``. ``build`` runs FIRST for an explicit
        ``model.build(...)`` and for ``.keras`` deserialization, so a ``build``
        that indexed the dict directly would convert this ``ValueError`` into a
        bare ``KeyError`` and defeat ``test_rejects_missing_pixels_key``
        (MEASURED: it did, 2026-08-23).

        :param mapping: The dict passed to ``build`` or ``call``.
        :return: The value under ``"pixels"``.
        :raises ValueError: If ``mapping`` is not a dict, or has no ``"pixels"``.
        """
        # DECISION plan-2026-08-23T091307-9a110062/D-426: do not inline this
        # into call() and index the dict directly in build() -- build() runs
        # first, so a direct index turns this into a bare KeyError. See decisions.md.
        if not isinstance(mapping, dict):
            raise ValueError(
                "VideoJEPA expects inputs as a dict with key 'pixels'. "
                f"Got type={type(mapping)}."
            )
        if "pixels" not in mapping:
            raise ValueError(
                "VideoJEPA inputs dict must contain key 'pixels'. "
                f"Got keys: {list(mapping.keys())}"
            )
        return mapping["pixels"]

    # DECISION plan-2026-08-23T091307-9a110062/D-425: walks sub-layers by hand
    # instead of tracing call(), since add_loss() raises when call() runs
    # directly on KerasTensor placeholders. See decisions.md.
    def build(self, input_shape: Dict[str, Any]) -> None:
        """Materialize every weight-bearing sub-layer.

        Touches `encoder`, `target_encoder`, `predictor`, every `pred_heads`
        entry, and `sigreg`. `mask_gen` owns no weights, and `mask_token` is
        allocated in `__init__`. `pred_heads[i]` builds on the full `pred`
        rather than a causal slice, so a head is never left unbuilt for a
        probe clip shorter than its horizon; the masking branch is
        training-only and introduces no weights. The batch axis is fixed at
        `1` since no weight shape depends on it.

        A hand walk can drift from `call` silently, so
        `test_the_explicit_build_materializes_the_model.py` pins that this
        build materializes the same population a real call does.

        :param input_shape: Dict with key `pixels`, shape `(B, T, H, W, C)`.
        """
        if self.built:
            return
        cfg = self.config
        pixels_shape = tuple(self._require_pixels(input_shape)[1:])
        pixels = keras.KerasTensor((1,) + pixels_shape)

        z_online = self.encode_frames(pixels)
        self.encode_frames_target(pixels)

        pred = self.predictor(z_online)
        for head in self.pred_heads:
            head(pred)

        t_probe = pixels_shape[0]
        hp = cfg.patches_per_side
        self.sigreg(
            ops.reshape(z_online, (t_probe, hp * hp, cfg.embed_dim))
        )

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the training forward pass.

        :param inputs: Dict with key `pixels`, shape `(B, T, H, W, C)`.
        :param training: Forwarded to the encoders and predictor.
        :return: `pred`, shape `(B, T, H_p, W_p, D)`.
        :rtype: keras.KerasTensor
        """
        pixels = self._require_pixels(inputs)

        cfg = self.config
        z_online = self.encode_frames(pixels, training=training)
        z = z_online

        # Target features for the regression losses; gradient is stopped so
        # the optimizer never sees target_encoder directly, only via EMA.
        z_target = ops.stop_gradient(self.encode_frames_target(pixels))

        # Optionally substitute mask_token at masked positions.
        # The tube mask is spatial (B, H_p, W_p); broadcasting over T keeps
        # it time-invariant ⇒ causality preserved (I9).
        #
        # When mask_prediction_enabled is False, we fall back *exactly* to
        # iter-1 semantics: no mask generation, no L2, no token substitution.
        B = ops.shape(pixels)[0]
        T_dyn = ops.shape(pixels)[1]
        # DECISION plan_2026-05-24_ca745a6c/D-001: tube-mask substitution is a
        # TRAINING augmentation only. `TubeMaskGenerator` calls unseeded
        # `keras.random.uniform` and is non-deterministic across calls; running
        # it at inference would make `model(x, training=False)` self-non-
        # deterministic and break the trainer reload-check + downstream
        # consumers. EMA target encoder never sees masked tokens by design.
        # DECISION plan_2026-05-24_ca745a6c/D-003: identity check `training is True`
        # (not `bool(training)`) keeps this gate graph-safe under @tf.function.
        # `bool(<symbolic tensor>)` raises OperatorNotAllowedInGraphError; `is True`
        # constant-folds at trace time. Tensor-valued `training` short-circuits to
        # False (inference behavior). Callers wanting training-time masking must
        # pass Python True — which is what `keras.Model.fit` does. See iter-2 F5.
        masking_on = (
            (training is True)
            and cfg.mask_prediction_enabled
            and self.mask_gen.num_masked > 0
        )
        if masking_on:
            mask_spatial = self.mask_gen(B, training=training)
            # Broadcast to 5D and stay T-invariant.
            M = ops.reshape(
                mask_spatial,
                (B, 1, cfg.patches_per_side, cfg.patches_per_side, 1),
            )
            M = ops.cast(M, z.dtype)
            token = ops.reshape(self.mask_token, (1, 1, 1, 1, cfg.embed_dim))
            token = ops.cast(token, z.dtype)
            z_masked = (1.0 - M) * z + M * token
        else:
            M = None
            z_masked = z

        pred = self.predictor(z_masked, training=training)

        # DECISION plan_2026-05-23_0b664700/D-001: per-horizon Dense heads on
        # the shared predictor, same lambda per horizon, combined metric is the
        # mean of per-horizon losses -- decouples magnitude from N. See decisions.md.
        # DECISION plan-2026-08-18T140459-7991552f/D-041: every frame count below
        # comes from this batch's T, never cfg.num_frames -- using the config
        # value produced a NaN loss at T <= h and a 2.33x rescale otherwise. See decisions.md.
        t_shape = getattr(pixels, "shape", None)
        t_static = t_shape[1] if t_shape is not None and len(t_shape) > 1 else None
        # A fully dynamic time axis cannot be branched on at trace time; the
        # configured window is the only available answer there.
        num_frames_batch = int(t_static) if t_static is not None else int(cfg.num_frames)

        if num_frames_batch >= 2:
            unmasked_per_row = (
                cfg.num_patches - self.mask_gen.num_masked
                if masking_on else cfg.num_patches
            )
            per_horizon_losses = []
            for h_idx, h in enumerate(cfg.predict_horizons):
                if h >= num_frames_batch:
                    # No causal pair exists at this horizon for this clip; skip
                    # rather than emit a NaN or a silent 0.0.
                    continue
                pred_ctx = pred[:, :-h]
                pred_ctx = self.pred_heads[h_idx](pred_ctx)
                # Target is the EMA encoder's output, not a live target, so
                # identity is not the optimal solution.
                target_ctx = z_target[:, h:]
                sq = ops.square(pred_ctx - target_ctx)
                if masking_on:
                    w = (1.0 - M)
                    denom = float(
                        max(
                            1,
                            unmasked_per_row
                            * (num_frames_batch - h)
                            * cfg.embed_dim,
                        )
                    )
                    h_loss = ops.sum(sq * w) / (
                        float(ops.shape(pred_ctx)[0]) * denom
                    )
                else:
                    h_loss = ops.mean(sq)
                self.add_loss(cfg.lambda_next_frame * h_loss)
                self.per_horizon_trackers[h_idx].update_state(h_loss)
                per_horizon_losses.append(h_loss)
            # per_horizon_losses can be empty if every horizon was skipped for
            # a very short clip, in which case there is no L1 to report.
            if per_horizon_losses:
                combined = per_horizon_losses[0]
                for hl in per_horizon_losses[1:]:
                    combined = combined + hl
                combined = combined / float(len(per_horizon_losses))
                self.next_frame_loss_tracker.update_state(combined)

        # Mask-prediction loss: MSE between predictor output and the EMA
        # target at masked positions, across all T frames (the tube is
        # time-invariant, so masked slots are symmetric targets across T).
        if masking_on:
            sq_full = ops.square(pred - z_target)
            num_masked_per_clip = (
                self.mask_gen.num_masked * num_frames_batch * cfg.embed_dim
            )
            denom = float(max(1, num_masked_per_clip))
            mask_loss = ops.sum(sq_full * M) / (
                float(ops.shape(pred)[0]) * denom
            )
            self.add_loss(cfg.lambda_mask * mask_loss)
            self.mask_loss_tracker.update_state(mask_loss)

        # DECISION plan_2026-05-23_15151c75/D-002: SIGReg runs on z_online (the
        # encoder output), not on pred, regularizing the representation directly under the JEPA framing. See decisions.md.
        Hp = cfg.patches_per_side
        N = Hp * Hp
        z_online_reshaped = ops.reshape(
            z_online, (B * T_dyn, N, cfg.embed_dim)
        )
        sigreg_loss = self.sigreg(z_online_reshaped)
        self.add_loss(self._sigreg_weight * sigreg_loss)
        self.sigreg_loss_tracker.update_state(sigreg_loss)

        return pred

    def train_step(self, data: Any) -> Dict[str, Any]:
        """Run one training step, then EMA-update the frozen target encoder.

        :param data: `(inputs, _)` tuple; the label is unused since losses
            come from `add_loss` inside :meth:`call`.
        :return: Dict of metric name to current value.
        :rtype: Dict[str, Any]
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
            # DECISION plan-2026-08-19T163559-499b6f0e/D-089: scale_loss must run
            # inside the tape -- under mixed_float16 the LossScaleOptimizer divides
            # every gradient by the loss scale regardless, so skipping this divides the whole update. See decisions.md.
            scaled_loss = self.optimizer.scale_loss(loss)
        # trainable_variables excludes target_encoder (trainable=False in __init__).
        grads = tape.gradient(scaled_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # EMA update runs after the optimizer step, so target tracks the
        # post-update encoder weights (V-JEPA/BYOL convention).
        self._ema_update()
        # DECISION plan_2026-05-24_ca745a6c/D-005: update loss_tracker explicitly
        # since it is normally only auto-updated by compiled_loss, which this
        # model bypasses in favor of add_loss. See decisions.md.
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def stream_reset(self, B: int = 1) -> None:
        """Reset the internal rolling buffer.

        :param B: Unused; kept for API symmetry. The buffer is created
            lazily on the first :meth:`stream_step` call.
        :type B: int
        """
        del B
        self._stream_buf = None

    def stream_step(
        self,
        frame: keras.KerasTensor,
        horizon: Optional[int] = None,
    ) -> keras.KerasTensor:
        """Advance the stream by one frame and return its patch forecast.

        Keeps the last `K` encoded frame grids in `_stream_buf: (B, t, H_p,
        W_p, D)` with `t <= K`. The predictor accepts arbitrary `T <=
        num_frames_max`, so it runs on a growing buffer until `t == K`, then truncates.

        :param frame: `(B, H, W, C)` single-frame pixel tensor.
        :type frame: keras.KerasTensor
        :param horizon: Which configured prediction horizon `h` to emit.
            `None` selects `min(config.predict_horizons)`, the
            shortest-range forecast (frame `t+1` under the default `(1,)`).
        :type horizon: Optional[int]
        :return: `(B, H_p, W_p, D)` prediction of the encoder embedding `h`
            frames after the one just pushed.
        :rtype: keras.KerasTensor
        :raises ValueError: If `horizon` is not in `config.predict_horizons`.
        """
        cfg = self.config
        K = cfg.history_size_k

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-043: the streaming output
        # must go through pred_heads[h_idx], the same quantity training
        # supervises -- not the raw predictor output. See decisions.md.
        if horizon is None:
            horizon = min(cfg.predict_horizons)
        if horizon not in cfg.predict_horizons:
            raise ValueError(
                f"horizon={horizon} has no trained prediction head; "
                f"config.predict_horizons is {cfg.predict_horizons!r}."
            )
        h_idx = cfg.predict_horizons.index(horizon)

        # Encode single frame: (B, H, W, C) → (B, H_p, W_p, D).
        enc = self.encoder(frame, training=False)
        enc_5d = enc[:, None, ...]  # (B, 1, H_p, W_p, D)

        # Append to buffer (grow, then truncate to K).
        if self._stream_buf is None:
            self._stream_buf = enc_5d
        else:
            self._stream_buf = ops.concatenate(
                [self._stream_buf, enc_5d], axis=1
            )
            if int(self._stream_buf.shape[1]) > K:
                self._stream_buf = self._stream_buf[:, -K:]

        pred = self.predictor(
            self._stream_buf, training=False
        )  # (B, t, H_p, W_p, D)

        # Heads are pointwise, so projecting the last slice equals slicing
        # the projection, at a fraction of the cost.
        return self.pred_heads[h_idx](pred[:, -1])

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], custom_objects=None
    ) -> "VideoJEPA":
        """Create a model from a configuration dictionary.

        :param config: Configuration dictionary.
        :param custom_objects: Unused; accepted for signature compatibility.
        :return: A new model instance.
        :rtype: VideoJEPA
        """
        cfg_dict = config.pop("config", None)
        cfg = (
            VideoJEPAConfig.from_dict(cfg_dict) if cfg_dict is not None
            else VideoJEPAConfig()
        )
        return cls(config=cfg, **config)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_video_jepa(
    config: Optional[VideoJEPAConfig] = None,
    **overrides: Any,
) -> VideoJEPA:
    """Create a Video-JEPA-Clifford model.

    There is no `MODEL_VARIANTS` table: this port ships one
    `VideoJEPAConfig` and is retuned field by field rather than by selecting
    a named scale.

    :param config: A `VideoJEPAConfig`; `None` uses the package defaults.
    :type config: Optional[VideoJEPAConfig]
    :param overrides: Individual `VideoJEPAConfig` field overrides applied on
        top of `config`. Keys that are not config fields are forwarded to
        `keras.Model` instead, e.g. `name`.
    :return: A configured `VideoJEPA` instance.
    :rtype: VideoJEPA
    :raises ValueError: If the resulting config fails `VideoJEPAConfig`
        validation, e.g. `patch_size` not dividing `img_size`.

    :Example:

    >>> model = create_video_jepa(img_size=32, patch_size=8, num_frames=2)
    >>> pred = model({"pixels": pixels})
    """
    base = config if config is not None else VideoJEPAConfig()
    fields = set(VideoJEPAConfig.__dataclass_fields__)
    cfg_overrides = {k: v for k, v in overrides.items() if k in fields}
    model_kwargs = {k: v for k, v in overrides.items() if k not in fields}

    if cfg_overrides:
        merged = base.to_dict()
        merged.update(cfg_overrides)
        base = VideoJEPAConfig.from_dict(merged)

    return VideoJEPA(config=base, **model_kwargs)

# ---------------------------------------------------------------------

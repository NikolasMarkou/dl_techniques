"""Top-level Video-JEPA-Clifford model.

Composes:

- :class:`VideoJEPACliffordEncoder` (per-frame hybrid encoder, D-001).
- :class:`VideoJEPAPredictor` (factorized spatial + causal-temporal, D-002,
  iter-3 pixels-only, D-013).
- :class:`SIGRegLayer` (middle placement on (B*T, N, D), D-005).

Training forward (``call``)
---------------------------
Inputs: ``{"pixels": (B, T, H, W, C)}``.

1. Encode frames: ``pixels → z: (B, T, H_p, W_p, D)``.
2. Predict: ``pred: (B, T, H_p, W_p, D) = predictor(z)``.
3. Losses via ``add_loss`` (training forward uses ``loss=None`` compile):

   * MSE: ``mean((pred[:, :-1] - z[:, 1:]) ** 2)`` (skipped if ``T < 2``).
   * Mask-prediction L2 (iter-2, if ``mask_prediction_enabled``).
   * SIGReg: ``sigreg_weight * sigreg(pred.reshape(B*T, N, D))``.

Returns ``pred``.

Streaming inference (D-007)
---------------------------
- :meth:`stream_reset(B)` — set internal embedding buffer to ``None``.
- :meth:`stream_step(frame)` — encode one frame, append to the rolling
  ``K``-length buffer, run the predictor on the current buffer (up to
  ``K`` frames), and emit the last frame's patch-prediction
  ``(B, H_p, W_p, D)``. O(1) amortized per call once the buffer is full.

The streaming path reuses the predictor on a growing (then rolling)
``(B, t, H_p, W_p, D)`` tensor with ``t ≤ K``; predictor accepts arbitrary
``T`` ≤ ``num_frames_max``.
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


@keras.saving.register_keras_serializable()
class VideoJEPA(keras.Model):
    """Video-JEPA-Clifford top-level model (pixels-only, iter-3).

    :param config: :class:`VideoJEPAConfig`. If ``None``, default config is used.
    :param kwargs: passthrough to :class:`keras.Model`.
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
            dropout=cfg.dropout,
            name="encoder",
        )
        # DECISION plan_2026-05-23_15151c75/D-001 — EMA target encoder
        # reverses video_jepa iter-1 D-001 (live target). Identity-baseline
        # pathology (multi-horizon eval: trained model 84-300x WORSE than
        # identity at every horizon) required a momentum-decoupled target.
        # Live-target + SIGReg co-adapted the encoder/predictor into a
        # near-time-invariant feature map; SIGReg prevents rank-collapse,
        # not time-invariance. Same architecture as ``self.encoder``; weights
        # synced once at first forward, then updated via EMA in ``train_step``.
        self.target_encoder = VideoJEPACliffordEncoder(
            embed_dim=cfg.embed_dim,
            patch_size=cfg.patch_size,
            img_size=cfg.img_size,
            img_channels=cfg.img_channels,
            depth=cfg.encoder_clifford_depth,
            shifts=tuple(cfg.encoder_shifts),
            dropout=cfg.dropout,
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
            dropout=cfg.dropout,
            name="predictor",
        )
        self.sigreg = SIGRegLayer(
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
            name="sigreg",
        )
        self._sigreg_weight = cfg.sigreg_weight

        # --- Iter-2: V-JEPA tube-masked latent prediction (D-008..D-012) ---
        # Mask generator is stateless — always instantiated so that
        # save/load round-trips the layer regardless of the enabled flag.
        self.mask_gen = TubeMaskGenerator(
            mask_ratio=cfg.mask_ratio,
            patches_per_side=cfg.patches_per_side,
            name="tube_mask_gen",
        )
        # Learned mask token. Allocated unconditionally so save/load
        # round-trips the same weight topology whether or not masking is
        # enabled. Zero-init per MAE convention — no effect on the
        # `mask_prediction_enabled=False` fallback path (never consumed).
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(cfg.embed_dim,),
            initializer="zeros",
            trainable=True,
        )

        # --- Multi-horizon prediction heads (plan_2026-05-23_0b664700/D-001)
        # One linear pointwise Dense (no bias) per horizon. Kept as a list
        # attribute so Keras 3 auto-tracks sublayers (proven path; same
        # idiom as predictor blocks). Heads operate per-token on the
        # predictor output and project the embedding back into z-space —
        # they cannot break causality because they are pointwise.
        self.pred_heads: List[keras.layers.Dense] = [
            keras.layers.Dense(
                cfg.embed_dim,
                use_bias=False,
                name=f"pred_head_h{h}",
            )
            for h in cfg.predict_horizons
        ]

        # --- Iter-2: per-loss Mean trackers for logging (Step 5) ---
        # These surface next_frame_loss and mask_loss as separate columns
        # in CSVLogger / history. They are running means over the epoch.
        # `next_frame_loss_tracker` retained with the same name for CSV
        # back-compat — it now logs the *combined* (mean over horizons) loss.
        self.next_frame_loss_tracker = keras.metrics.Mean(
            name="next_frame_loss"
        )
        # Per-horizon trackers, one per entry in cfg.predict_horizons.
        self.per_horizon_trackers: List[keras.metrics.Mean] = [
            keras.metrics.Mean(name=f"next_frame_loss_h{h}")
            for h in cfg.predict_horizons
        ]
        self.mask_loss_tracker = keras.metrics.Mean(name="mask_loss")
        self.sigreg_loss_tracker = keras.metrics.Mean(name="sigreg_loss")
        # DECISION plan_2026-05-24_ca745a6c/D-005: explicit aggregate `loss`
        # tracker. Keras 3.8 does not auto-create `self.loss_tracker` until
        # `compile(loss=...)` is called; our `train_step` bypasses compiled
        # loss entirely (losses come from `add_loss`). Without this explicit
        # Mean, `history.history['loss']` is pinned at 0.0 — defeats
        # EarlyStopping / ModelCheckpoint(monitor='loss') / training_curves.
        # See iter-3 F10. Name="loss" matches the conventional Keras key.
        self.loss_tracker = keras.metrics.Mean(name="loss")

        # --- Streaming buffer (not a weight; reset per sequence) ---
        self._stream_buf: Optional[Any] = None

        # --- EMA target encoder state (plan_2026-05-23_15151c75/D-001) ---
        # ``_ema_step`` is a non-trainable scalar weight so cosine-schedule
        # progress survives ``.keras`` checkpoint reload. ``_ema_total_steps``
        # is a plain Python attribute (defaults to 1.0 to make cosine math
        # safe even if the trainer forgets ``set_ema_total_steps``); reset
        # on each fresh trainer invocation.
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
        # Weight-space L2 ratio divergence between target and online
        # encoders (BYOL/MoCo convention). See _compute_ema_divergence
        # for the formula and DECISION plan_2026-05-24_aebd4cbb/D-001
        # for the rationale (Option A over per-layer cosine / feature
        # drift). Updated inside ``_ema_update`` after the EMA assign.
        self.ema_divergence_tracker = keras.metrics.Mean(name="ema_divergence")

        # Advisory: multi-horizon prediction without a strong EMA target
        # is the documented "multi-horizon head collapse" failure mode
        # (see src/train/video_jepa/README.md "Known issues / caveats").
        # Per-horizon Dense(D, no bias) heads on a shared causal predictor
        # do NOT break time-invariance symmetry on their own; without a
        # momentum-decoupled target encoder the heads converge to the same
        # numerical value. Threshold 0.5 chosen as the boundary between
        # "near-live" and "meaningfully decoupled" targets; the trainer's
        # default is 0.996.
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

        # --- Eager build + initial weight sync for both encoders ---
        # Force-build encoder + target_encoder up front with a dummy zero
        # batch so the lazy-build dance doesn't have to happen inside
        # ``call`` (which runs under TF graph tracing during train_step).
        # After ``set_weights(get_weights())``, target == encoder bitwise.
        # Reload path: ``from_config`` re-runs this, but ``load_model``
        # immediately overwrites all weights from disk, so the dummy
        # build's effect is transient and the reloaded target weights
        # are authoritative.
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
        """Per-loss trackers so fit() logs ``next_frame_loss`` + ``mask_loss``
        (+ ``sigreg_loss``) alongside the aggregated ``loss`` column."""
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
    def encode_frames(self, pixels: keras.KerasTensor) -> keras.KerasTensor:
        """Encode a pixel tensor ``(B, T, H, W, C) → (B, T, H_p, W_p, D)``
        through the **online** encoder.

        Mirrors :meth:`LeWM.encode_pixels` but keeps the 4D patch grid
        instead of pooling.
        """
        cfg = self.config
        shape = ops.shape(pixels)
        B, T = shape[0], shape[1]
        H, W, C = cfg.img_size, cfg.img_size, cfg.img_channels
        flat = ops.reshape(pixels, (B * T, H, W, C))
        feat = self.encoder(flat)  # (B*T, H_p, W_p, D)
        Hp = cfg.patches_per_side
        return ops.reshape(feat, (B, T, Hp, Hp, cfg.embed_dim))

    def encode_frames_target(self, pixels: keras.KerasTensor) -> keras.KerasTensor:
        """Encode pixels through the **EMA target** encoder (no gradient).

        Always runs with ``training=False`` (target encoder is frozen);
        caller is expected to wrap the result in ``ops.stop_gradient``.
        """
        cfg = self.config
        shape = ops.shape(pixels)
        B, T = shape[0], shape[1]
        H, W, C = cfg.img_size, cfg.img_size, cfg.img_channels
        flat = ops.reshape(pixels, (B * T, H, W, C))
        feat = self.target_encoder(flat, training=False)
        Hp = cfg.patches_per_side
        return ops.reshape(feat, (B, T, Hp, Hp, cfg.embed_dim))

    # ------------------------------------------------------------------
    # EMA target encoder helpers (plan_2026-05-23_15151c75/D-001)
    # ------------------------------------------------------------------
    def set_ema_total_steps(self, n: int) -> None:
        """Set the total step count used by the cosine EMA schedule.

        Harmless for ``ema_schedule="none"``. Trainer calls this once
        before ``fit()`` so the schedule covers the whole run.
        """
        self._ema_total_steps = float(max(int(n), 1))

    def sync_target_to_online(self) -> None:
        """Bitwise copy of ``encoder.weights`` into ``target_encoder.weights``.

        Public helper; the constructor runs it once after a dummy build.
        Useful for tests that want to re-sync after manual weight edits.
        """
        self.target_encoder.set_weights(self.encoder.get_weights())

    def _current_momentum(self):
        """Return the EMA momentum for the current step as a scalar tensor.

        ``"none"`` → constant ``cfg.ema_momentum``.
        ``"cosine"`` → ramps from ``m0`` to ``1.0`` across
        ``_ema_total_steps`` via a half-cosine:
        ``m(t) = m0 + (1 - m0) * (1 - cos(pi * t / T)) / 2``.
        Always clamped to ``[m0, 1.0]``. Returned as a scalar tensor
        so the math stays inside the TF graph during ``train_step``.
        """
        cfg = self.config
        m0 = ops.convert_to_tensor(float(cfg.ema_momentum), dtype="float32")
        if cfg.ema_schedule == "none":
            return m0
        # cosine — keep entirely in ops so train_step traces cleanly.
        step = ops.cast(self._ema_step, "float32")
        total = ops.convert_to_tensor(
            max(float(self._ema_total_steps), 1.0), dtype="float32",
        )
        progress = ops.minimum(step / total, 1.0)
        pi = ops.convert_to_tensor(math.pi, dtype="float32")
        one = ops.convert_to_tensor(1.0, dtype="float32")
        m = m0 + (one - m0) * (one - ops.cos(pi * progress)) / 2.0
        # Clamp to [m0, 1.0].
        m = ops.minimum(ops.maximum(m, m0), one)
        return m

    # DECISION plan_2026-05-24_aebd4cbb/D-001:
    # Weight-space L2 ratio (Option A — BYOL/MoCo convention) chosen at the
    # cost of per-layer visibility (Option B: per-layer cosine) and
    # feature-space semantic drift on a fixed probe batch (Option C).
    # Single scalar, in-graph, O(P) over the weights already iterated in
    # ``_ema_update``. Cold-start property: bitwise weight sync in
    # ``__init__`` → divergence ≈ 0 (within fp32 noise). Asymptotic range
    # in published BYOL/MoCo runs is 0.01–0.3; sustained >1.0 indicates
    # online/target collapse and is the actionable signal this metric
    # exists to surface. ``+ 1e-12`` epsilon guards against the degenerate
    # case where ``encoder.weights`` are all zero (cold init pre-build).
    # See plans/plan_2026-05-24_aebd4cbb/decisions.md D-001.
    def _compute_ema_divergence(self):
        """Weight-space L2 divergence ratio between target and online.

        Computes ``sqrt(sum((t_w - e_w)^2)) / (sqrt(sum(e_w^2)) + 1e-12)``
        across all paired (target_encoder, encoder) weights as a single
        float32 scalar. Cast to float32 before the sum so mixed-precision
        runs (encoder dtype = float16) still produce a numerically stable
        divergence value matching the existing tracker idiom.

        :return: scalar float32 tensor.
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
        # Post-EMA-assign divergence snapshot (D-001).
        self.ema_divergence_tracker.update_state(self._compute_ema_divergence())

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Training forward pass.

        :param inputs: dict with key ``pixels`` (B, T, H, W, C).
        :param training: forwarded.
        :return: ``pred`` of shape (B, T, H_p, W_p, D).
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                "VideoJEPA expects inputs as a dict with key 'pixels'. "
                f"Got type={type(inputs)}."
            )
        if "pixels" not in inputs:
            raise ValueError(
                "VideoJEPA inputs dict must contain key 'pixels'. "
                f"Got keys: {list(inputs.keys())}"
            )
        pixels = inputs["pixels"]

        cfg = self.config
        # Online (gradient-carrying) encoder — feeds the predictor and
        # SIGReg. ``z`` retained as a local alias for downstream code that
        # still expects the iter-1 name (predictor-input substitution).
        z_online = self.encode_frames(pixels)               # (B, T, H_p, W_p, D)
        z = z_online

        # --- EMA target encoder (plan_2026-05-23_15151c75/D-001) ---
        # Target features for the regression losses — gradient is stopped
        # so the optimizer never sees target_encoder; EMA owns it.
        z_target = ops.stop_gradient(self.encode_frames_target(pixels))

        # --- Iter-2: optionally substitute mask_token at masked positions ---
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
            mask_spatial = self.mask_gen(B, training=training)  # (B, H_p, W_p)
            # Broadcast to 5D: (B, 1, H_p, W_p, 1). Stays T-invariant.
            M = ops.reshape(
                mask_spatial,
                (B, 1, cfg.patches_per_side, cfg.patches_per_side, 1),
            )
            M = ops.cast(M, z.dtype)
            # mask_token broadcast: (D,) -> (1, 1, 1, 1, D).
            token = ops.reshape(self.mask_token, (1, 1, 1, 1, cfg.embed_dim))
            token = ops.cast(token, z.dtype)
            z_masked = (1.0 - M) * z + M * token
        else:
            M = None
            z_masked = z

        pred = self.predictor(z_masked, training=training)  # (B,T,H_p,W_p,D)

        # --- L1: multi-horizon prediction loss
        # (plan_2026-05-23_0b664700/D-001; supersedes single-horizon t+1) ---
        # For each h in cfg.predict_horizons:
        #   pred_h = pred_head_h(pred[:, :-h])          # (B, T-h, H_p, W_p, D)
        #   target = z[:, h:]                            # (B, T-h, H_p, W_p, D)
        #   MSE on unmasked positions; weighted by lambda_next_frame
        # The shared causal predictor is unchanged across horizons; per-horizon
        # heads provide an independent linear sub-objective per h, addressing
        # the degenerate-identity pathology of single-horizon t+1 at 30fps.
        # Skip entirely if num_frames < 2 (edge case: single-frame window).
        # DECISION plan_2026-05-23_0b664700/D-001: per-horizon Dense heads on
        # the shared predictor + same lambda per horizon + combined metric =
        # mean of per-horizon losses (decouples reported magnitude from N).
        if cfg.num_frames >= 2:
            unmasked_per_row = (
                cfg.num_patches - self.mask_gen.num_masked
                if masking_on else cfg.num_patches
            )
            per_horizon_losses = []
            for h_idx, h in enumerate(cfg.predict_horizons):
                pred_ctx = pred[:, :-h]                  # (B, T-h, H_p, W_p, D)
                pred_ctx = self.pred_heads[h_idx](pred_ctx)
                # Target from EMA encoder (plan_2026-05-23_15151c75/D-001):
                # was ``z[:, h:]`` (live target) — switched so identity is
                # no longer the optimal solution.
                target_ctx = z_target[:, h:]             # (B, T-h, H_p, W_p, D)
                sq = ops.square(pred_ctx - target_ctx)
                if masking_on:
                    w = (1.0 - M)  # broadcasts (B,1,H_p,W_p,1) -> (B,T-h,...)
                    denom = float(
                        max(1, unmasked_per_row * (cfg.num_frames - h) * cfg.embed_dim)
                    )
                    h_loss = ops.sum(sq * w) / (
                        float(ops.shape(pred_ctx)[0]) * denom
                    )
                else:
                    h_loss = ops.mean(sq)
                # Same lambda per horizon (see D-001 Rejected Alternative #2).
                self.add_loss(cfg.lambda_next_frame * h_loss)
                self.per_horizon_trackers[h_idx].update_state(h_loss)
                per_horizon_losses.append(h_loss)
            # Combined tracker = mean of per-horizon losses; decouples reported
            # magnitude from N so dashboards stay comparable across runs.
            combined = per_horizon_losses[0]
            for hl in per_horizon_losses[1:]:
                combined = combined + hl
            combined = combined / float(len(per_horizon_losses))
            self.next_frame_loss_tracker.update_state(combined)

        # --- L2: mask-prediction loss (iter-2, D-008..D-012) ---
        # MSE between predictor output and *same-encoder* target at masked
        # positions, across ALL T frames (no causal slice — the tube is
        # time-invariant so masked slots across T are symmetric targets).
        if masking_on:
            # Target from EMA encoder (plan_2026-05-23_15151c75/D-001):
            # was ``z`` (live target) — switched so masked-position
            # prediction has a moving but non-co-adapting target.
            sq_full = ops.square(pred - z_target)  # (B, T, H_p, W_p, D)
            num_masked_per_clip = (
                self.mask_gen.num_masked * cfg.num_frames * cfg.embed_dim
            )
            denom = float(max(1, num_masked_per_clip))
            mask_loss = ops.sum(sq_full * M) / (
                float(ops.shape(pred)[0]) * denom
            )
            self.add_loss(cfg.lambda_mask * mask_loss)
            self.mask_loss_tracker.update_state(mask_loss)

        # --- L3: SIGReg on (B*T, N, D) ---
        # DECISION plan_2026-05-23_15151c75/D-002 — SIGReg input switched
        # from ``pred`` (predictor output) to ``z_online`` (encoder output).
        # Spec line 5: SIGReg stays on the online encoder only; the target
        # encoder is not regularized (it carries no gradient). Conceptually
        # this regularizes the representation directly under the JEPA
        # framing instead of the prediction.
        Hp = cfg.patches_per_side
        N = Hp * Hp
        z_online_reshaped = ops.reshape(
            z_online, (B * T_dyn, N, cfg.embed_dim)
        )
        sigreg_loss = self.sigreg(z_online_reshaped)
        self.add_loss(self._sigreg_weight * sigreg_loss)
        self.sigreg_loss_tracker.update_state(sigreg_loss)

        return pred

    # ------------------------------------------------------------------
    # Custom train_step (plan_2026-05-23_15151c75/D-001)
    # ------------------------------------------------------------------
    def train_step(self, data: Any) -> Dict[str, Any]:
        """One training step: forward, backward on encoder+predictor+heads,
        then EMA-update the frozen target encoder.

        ``data`` is the (inputs, _) tuple emitted by both the synthetic
        and BDD100K dataset pipelines. The label is unused (losses come
        from ``add_loss`` inside :meth:`call`).
        """
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            _ = self(x, training=True)
            # ``self.losses`` collects every ``add_loss`` from this
            # forward + any regularization losses. Sum to a scalar.
            losses = self.losses
            if losses:
                loss = ops.cast(losses[0], "float32")
                for extra in losses[1:]:
                    loss = loss + ops.cast(extra, "float32")
            else:
                loss = ops.convert_to_tensor(0.0, dtype="float32")
        # trainable_variables excludes target_encoder weights because
        # ``target_encoder.trainable = False`` in __init__.
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # EMA update AFTER optimizer step — target now tracks the
        # **post-update** encoder weights (V-JEPA / BYOL convention).
        self._ema_update()
        # DECISION plan_2026-05-24_ca745a6c/D-005: update the Keras default
        # loss_tracker so `history.history['loss']`, CSVLogger, and
        # ModelCheckpoint(monitor='loss') observe the true aggregate loss
        # (sum of add_loss + regularizers). `super().metrics` exposes
        # loss_tracker but it is only auto-updated by `self.compiled_loss(...)`,
        # which we bypass in favor of add_loss. Without this update, the
        # `loss` column is pinned at 0.0 even though gradients use the real
        # loss above — defeats EarlyStopping / training_curves/loss.png.
        # See iter-3 F10.
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    # ------------------------------------------------------------------
    # Streaming inference (D-007)
    # ------------------------------------------------------------------
    def stream_reset(self, B: int = 1) -> None:
        """Reset the internal rolling-buffer (``_stream_buf := None``).

        :param B: Placeholder for future stateful buffers — currently unused
            because the buffer is created lazily on the first
            :meth:`stream_step` call. Kept for API symmetry with LeWM.
        """
        del B  # unused
        self._stream_buf = None

    def stream_step(
        self,
        frame: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Advance the stream by one frame and return its patch prediction.

        :param frame: ``(B, H, W, C)`` single-frame pixel tensor.
        :return: ``(B, H_p, W_p, D)`` patch-prediction for the latest frame.

        Rolling buffer (D-007): keeps the last ``K`` encoded frame grids in
        ``_stream_buf: (B, t, H_p, W_p, D)`` with ``t ≤ K``. The predictor
        accepts arbitrary ``T`` (≤ ``num_frames_max``), so we can call it
        on a growing buffer until ``t == K``, then truncate.
        """
        cfg = self.config
        K = cfg.history_size_k

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

        return pred[:, -1]  # (B, H_p, W_p, D)

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
    ) -> "VideoJEPA":
        cfg_dict = config.pop("config", None)
        cfg = (
            VideoJEPAConfig.from_dict(cfg_dict) if cfg_dict is not None
            else VideoJEPAConfig()
        )
        return cls(config=cfg, **config)

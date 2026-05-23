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

from typing import Any, Dict, List, Optional

import keras
from keras import ops

from dl_techniques.regularizers.sigreg import SIGRegLayer

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

        # --- Streaming buffer (not a weight; reset per sequence) ---
        self._stream_buf: Optional[Any] = None

    @property
    def metrics(self) -> list:
        """Per-loss trackers so fit() logs ``next_frame_loss`` + ``mask_loss``
        (+ ``sigreg_loss``) alongside the aggregated ``loss`` column."""
        base = list(super().metrics)
        extras = [
            self.next_frame_loss_tracker,
            *self.per_horizon_trackers,
            self.mask_loss_tracker,
            self.sigreg_loss_tracker,
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
        """Encode a pixel tensor ``(B, T, H, W, C) → (B, T, H_p, W_p, D)``.

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
        z = self.encode_frames(pixels)                      # (B, T, H_p, W_p, D)

        # --- Iter-2: optionally substitute mask_token at masked positions ---
        # The tube mask is spatial (B, H_p, W_p); broadcasting over T keeps
        # it time-invariant ⇒ causality preserved (I9).
        #
        # When mask_prediction_enabled is False, we fall back *exactly* to
        # iter-1 semantics: no mask generation, no L2, no token substitution.
        B = ops.shape(pixels)[0]
        T_dyn = ops.shape(pixels)[1]
        masking_on = cfg.mask_prediction_enabled and self.mask_gen.num_masked > 0
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
                target_ctx = z[:, h:]                    # (B, T-h, H_p, W_p, D)
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
            sq_full = ops.square(pred - z)  # (B, T, H_p, W_p, D)
            num_masked_per_clip = (
                self.mask_gen.num_masked * cfg.num_frames * cfg.embed_dim
            )
            denom = float(max(1, num_masked_per_clip))
            mask_loss = ops.sum(sq_full * M) / (
                float(ops.shape(pred)[0]) * denom
            )
            self.add_loss(cfg.lambda_mask * mask_loss)
            self.mask_loss_tracker.update_state(mask_loss)

        # --- L3: SIGReg on (B*T, N, D) (D-005, unchanged) ---
        Hp = cfg.patches_per_side
        N = Hp * Hp
        pred_reshaped = ops.reshape(
            pred, (B * T_dyn, N, cfg.embed_dim)
        )
        sigreg_loss = self.sigreg(pred_reshaped)
        self.add_loss(self._sigreg_weight * sigreg_loss)
        self.sigreg_loss_tracker.update_state(sigreg_loss)

        return pred

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

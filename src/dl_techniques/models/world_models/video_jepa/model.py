"""
Top-level Video-JEPA-Clifford model.

Defines :class:`VideoJEPA`, which composes ``VideoJEPACliffordEncoder``, a frozen EMA
copy of it, ``VideoJEPAPredictor`` and a middle-placement ``SIGRegLayer``, plus the
:func:`create_video_jepa` factory. The prediction targets are future frame embeddings
rather than pixels, so there is no pixel decoder: loss and evaluation both live in the
encoder's own latent space. Targets come from the EMA target encoder instead of the
live one, which keeps encoder and predictor from co-adapting into a
near-time-invariant map, and SIGReg guards against rank collapse in the
representation. ``call`` takes ``{"pixels": (B, T, H, W, C)}``, returns the raw
prediction, and contributes the per-horizon next-frame MSE, an optional tube-masked
term and the SIGReg loss through ``add_loss``, so ``compile`` needs no loss;
``train_step`` sums them and then EMA-updates the target. ``stream_reset`` and
``stream_step`` run the same predictor over a rolling ``K``-frame buffer, at a cost
per frame that does not grow with the length of the stream.

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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.utils.logger import logger

from .config import VideoJEPAConfig
from .encoder import VideoJEPACliffordEncoder
from .masking import TubeMaskGenerator
from .predictor import VideoJEPAPredictor
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.video_jepa.model")
class VideoJEPA(keras.Model):
    """Predict future frame embeddings from pixels, with an EMA target encoder.

    ``call`` returns the predictor output; every loss term is contributed with
    ``add_loss``, so the model is compiled without a loss and ``train_step`` sums
    ``self.losses`` itself before EMA-updating the target encoder.

    Architecture (training forward):

    .. code-block:: text

        pixels [B, T, H, W, C]
                 │
                 ├─────────────────────────┐
                 ▼                         ▼
        ┌───────────────────┐   ┌──────────────────────┐
        │ encoder           │   │ target_encoder       │  EMA, frozen
        └───────────────────┘   └──────────────────────┘
                 │                         │
                 │ z [B, T, H_p, W_p, D]   │ stop_gradient
                 ├──► sigreg ──► add_loss  │
                 ▼                         │
        ┌───────────────────┐              │
        │ mask substitution │  (training)  │
        └───────────────────┘              │
                 │                         │
                 ▼                         │
        ┌───────────────────┐              │
        │ predictor         │              │
        └───────────────────┘              │
                 │ pred                    │
                 ├────────► losses ◄───────┘
                 ▼
        pred [B, T, H_p, W_p, D]  (returned)

    Losses:

    .. code-block:: text

        term             compared                       weight
        next frame       pred_head_h(pred[:, :-h])      lambda_next_frame
                         against z_target[:, h:]
        mask (training)  pred against z_target,         lambda_mask
                         masked slots, all T
        sigreg           z reshaped [B*T, N, D]         sigreg_weight

    A horizon with h >= T is skipped; with masking on, the two MSE terms split
    the slots between them.

    Tube mask:

    .. code-block:: text

        mask_gen(B)  [B, H_p, W_p]
                 │
                 ▼
        reshape [B, 1, H_p, W_p, 1]   broadcast over T
                 │
                 ▼
        z <- (1 - M) * z + M * mask_token

    The mask is time-invariant, and applies only when training is exactly True.

    Streaming:

    .. code-block:: text

        frame [B, H, W, C]
                 │
                 ▼
        ┌───────────────────┐
        │ encoder           │  training=False
        └───────────────────┘
                 │  [B, 1, H_p, W_p, D]
                 ▼
        buffer <- concat, keep last K
                 │  [B, t <= K, H_p, W_p, D]
                 ▼
        ┌───────────────────┐
        │ predictor         │
        └───────────────────┘
                 │
                 ▼
        pred_head_h(pred[:, -1]) ──► [B, H_p, W_p, D]

    :param config: :class:`VideoJEPAConfig`. Uses the default config when `None`.
    :type config: Optional[VideoJEPAConfig]
    :param **kwargs: Forwarded to :class:`keras.Model`.

    :ivar encoder: The online encoder, the only encoder that receives gradients.
    :vartype encoder: VideoJEPACliffordEncoder
    :ivar target_encoder: Frozen EMA copy that produces the regression targets.
    :vartype target_encoder: VideoJEPACliffordEncoder
    :ivar predictor: Factorized spatial and causal-temporal predictor.
    :vartype predictor: VideoJEPAPredictor
    :ivar pred_heads: One bias-free pointwise Dense per entry of `predict_horizons`.
    :vartype pred_heads: List[keras.layers.Dense]
    :ivar sigreg: Rank-collapse regularizer, applied to the encoder output.
    :vartype sigreg: SIGRegLayer
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
        # DECISION plan_2026-05-23_15151c75/D-001: targets come from an EMA copy; a live
        # target encoder co-adapts into a near-time-invariant map. See decisions.md.
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
        # Frozen here, before any build or variable tracking, so these weights stay
        # out of `self.trainable_variables`.
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

        # Pointwise, so a head cannot break causality.
        self.pred_heads: List[keras.layers.Dense] = [
            keras.layers.Dense(
                cfg.embed_dim,
                use_bias=False,
                name=f"pred_head_h{h}",
            )
            for h in cfg.predict_horizons
        ]

        # Logs the mean over horizons under the name "next_frame_loss".
        self.next_frame_loss_tracker = keras.metrics.Mean(
            name="next_frame_loss"
        )
        self.per_horizon_trackers: List[keras.metrics.Mean] = [
            keras.metrics.Mean(name=f"next_frame_loss_h{h}")
            for h in cfg.predict_horizons
        ]
        self.mask_loss_tracker = keras.metrics.Mean(name="mask_loss")
        self.sigreg_loss_tracker = keras.metrics.Mean(name="sigreg_loss")
        # DECISION plan_2026-05-24_ca745a6c/D-005: track the aggregate loss here; Keras
        # creates one only for compile(loss=...), which this model bypasses. See decisions.md.
        self.loss_tracker = keras.metrics.Mean(name="loss")

        # Streaming buffer: not a weight, reset per sequence.
        self._stream_buf: Optional[Any] = None

        # DECISION plan_2026-05-23_15151c75/D-001: _ema_step is a non-trainable weight so
        # schedule progress survives reload; _ema_total_steps defaults to 1.0. See decisions.md.
        self._ema_step = self.add_weight(
            name="ema_step",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )
        self._ema_total_steps: float = 1.0
        # Logged as "ema_m" so cosine schedules show up in CSVLogger and history.
        self.ema_m_tracker = keras.metrics.Mean(name="ema_m")
        # Weight-space L2 ratio between target and online encoders (BYOL/MoCo).
        self.ema_divergence_tracker = keras.metrics.Mean(name="ema_divergence")

        # Multi-horizon with a weak EMA target is a known head-collapse regime.
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

        # Both encoders are built eagerly so no lazy build happens inside call()
        # under graph tracing, and the target starts as a copy of the online weights.
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

        Frames are flattened into the batch axis, so the encoder sees 4D input and the
        result is reshaped back to five dimensions.

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

    # DECISION plan_2026-05-24_aebd4cbb/D-001: weight-space L2 ratio (BYOL/MoCo), not
    # per-layer cosine; sustained above 1.0 signals collapse. See decisions.md.
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

        Shared by :meth:`build` and :meth:`call`, which take the same dict-shaped
        argument, one a nest of shapes and one a nest of tensors, and which must fail
        the same way.

        :param mapping: The dict passed to ``build`` or ``call``.
        :return: The value under ``"pixels"``.
        :raises ValueError: If ``mapping`` is not a dict, or has no ``"pixels"``.
        """
        # DECISION plan-2026-08-23T091307-9a110062/D-426: build() must call this too;
        # indexing the dict there turns this ValueError into a KeyError. See decisions.md.
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

    # DECISION plan-2026-08-23T091307-9a110062/D-425: walk the sub-layers by hand;
    # tracing call() raises, because add_loss() rejects KerasTensors. See decisions.md.
    def build(self, input_shape: Dict[str, Any]) -> None:
        """Materialize every weight-bearing sub-layer.

        Touches `encoder`, `target_encoder`, `predictor`, every `pred_heads`
        entry, and `sigreg`. `mask_gen` owns no weights, and `mask_token` is
        allocated in `__init__`. `pred_heads[i]` builds on the full `pred`
        rather than a causal slice, so a head is never left unbuilt for a
        probe clip shorter than its horizon; the masking branch is
        training-only and introduces no weights. The batch axis is fixed at
        `1` since no weight shape depends on it.

        :param input_shape: Dict with key `pixels`, shape `(B, T, H, W, C)`.
        :raises ValueError: If `input_shape` is not a dict with a `pixels` entry.
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

        Contributes the per-horizon next-frame MSE, the optional mask term and the
        SIGReg loss with ``add_loss``, and updates the matching trackers. The mask
        branch runs only when ``training`` is the Python value ``True``.

        :param inputs: Dict with key `pixels`, shape `(B, T, H, W, C)`.
        :param training: Forwarded to the encoders and predictor.
        :return: `pred`, shape `(B, T, H_p, W_p, D)`.
        :rtype: keras.KerasTensor
        :raises ValueError: If `inputs` is not a dict with a `pixels` entry.
        """
        pixels = self._require_pixels(inputs)

        cfg = self.config
        z_online = self.encode_frames(pixels, training=training)
        z = z_online

        # Gradient stops here, so the optimizer reaches target_encoder only via EMA.
        z_target = ops.stop_gradient(self.encode_frames_target(pixels))

        B = ops.shape(pixels)[0]
        T_dyn = ops.shape(pixels)[1]
        # DECISION plan_2026-05-24_ca745a6c/D-001: tube masking is training-only; the
        # generator is unseeded, so inference would stop being deterministic. See decisions.md.
        # DECISION plan_2026-05-24_ca745a6c/D-003: test `training is True`, not
        # `bool(training)`, which raises on a symbolic tensor under tf.function. See decisions.md.
        masking_on = (
            (training is True)
            and cfg.mask_prediction_enabled
            and self.mask_gen.num_masked > 0
        )
        if masking_on:
            mask_spatial = self.mask_gen(B, training=training)
            # Broadcasting over T keeps the mask time-invariant, so causality holds.
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

        # DECISION plan_2026-05-23_0b664700/D-001: one Dense head per horizon on the
        # shared predictor; the combined metric is the mean over horizons. See decisions.md.
        # DECISION plan-2026-08-18T140459-7991552f/D-041: frame counts come from this
        # batch's T; cfg.num_frames gave a NaN at T <= h and a 2.33x rescale. See decisions.md.
        t_shape = getattr(pixels, "shape", None)
        t_static = t_shape[1] if t_shape is not None and len(t_shape) > 1 else None
        # With a fully dynamic time axis there is nothing to branch on at trace time.
        num_frames_batch = int(t_static) if t_static is not None else int(cfg.num_frames)

        if num_frames_batch >= 2:
            unmasked_per_row = (
                cfg.num_patches - self.mask_gen.num_masked
                if masking_on else cfg.num_patches
            )
            per_horizon_losses = []
            for h_idx, h in enumerate(cfg.predict_horizons):
                if h >= num_frames_batch:
                    # No causal pair exists here, so skip rather than emit NaN or 0.0.
                    continue
                pred_ctx = pred[:, :-h]
                pred_ctx = self.pred_heads[h_idx](pred_ctx)
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
            # Every horizon can be skipped on a very short clip, leaving nothing to log.
            if per_horizon_losses:
                combined = per_horizon_losses[0]
                for hl in per_horizon_losses[1:]:
                    combined = combined + hl
                combined = combined / float(len(per_horizon_losses))
                self.next_frame_loss_tracker.update_state(combined)

        # The tube is time-invariant, so masked slots are symmetric targets across T.
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

        # DECISION plan_2026-05-23_15151c75/D-002: SIGReg runs on z_online, not pred, so
        # the representation itself is regularized. See decisions.md.
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
            # DECISION plan-2026-08-19T163559-499b6f0e/D-089: scale_loss runs inside the
            # tape; the LossScaleOptimizer divides gradients regardless. See decisions.md.
            scaled_loss = self.optimizer.scale_loss(loss)
        # trainable_variables excludes target_encoder (trainable=False in __init__).
        grads = tape.gradient(scaled_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # After the optimizer step, so the target tracks post-update weights.
        self._ema_update()
        # DECISION plan_2026-05-24_ca745a6c/D-005: update loss_tracker by hand; only
        # compiled_loss does it automatically, and this model uses add_loss. See decisions.md.
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
        Everything here runs with `training=False`.

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

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-043: return pred_heads[h_idx] of the
        # prediction, the quantity training supervises, not the raw output. See decisions.md.
        if horizon is None:
            horizon = min(cfg.predict_horizons)
        if horizon not in cfg.predict_horizons:
            raise ValueError(
                f"horizon={horizon} has no trained prediction head; "
                f"config.predict_horizons is {cfg.predict_horizons!r}."
            )
        h_idx = cfg.predict_horizons.index(horizon)

        enc = self.encoder(frame, training=False)
        enc_5d = enc[:, None, ...]

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
        )

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

    There is no variant table: this port ships one `VideoJEPAConfig` and is
    retuned field by field rather than by selecting a named scale.

    :param config: A `VideoJEPAConfig`; `None` uses the package defaults.
    :type config: Optional[VideoJEPAConfig]
    :param **overrides: Individual `VideoJEPAConfig` field overrides applied on
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

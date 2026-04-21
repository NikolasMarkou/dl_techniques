"""Top-level Video-JEPA-Clifford model.

Composes:

- :class:`VideoJEPACliffordEncoder` (per-frame hybrid encoder, D-001).
- :class:`TelemetryEmbedder` (per-frame telemetry → cond_dim, D-004/D-006).
- :class:`VideoJEPAPredictor` (factorized spatial + causal-temporal, D-002).
- :class:`SIGRegLayer` (middle placement on (B*T, N, D), D-005).

Training forward (``call``)
---------------------------
Inputs: ``{"pixels": (B, T, H, W, C), "telemetry": (B, T, k)}``.

1. Encode frames: ``pixels → z: (B, T, H_p, W_p, D)``.
2. Embed telemetry: ``telemetry → c: (B, T, D)``.
3. Predict: ``pred: (B, T, H_p, W_p, D) = predictor([z, c])``.
4. Losses via ``add_loss`` (training forward uses ``loss=None`` compile):

   * MSE: ``mean((pred[:, :-1] - z[:, 1:]) ** 2)`` (skipped if ``T < 2``).
   * SIGReg: ``sigreg_weight * sigreg(pred.reshape(B*T, N, D))``.

Returns ``pred``.

Streaming inference (D-007)
---------------------------
- :meth:`stream_reset(B)` — set internal embedding buffer to ``None``.
- :meth:`stream_step(frame, telemetry_frame)` — encode one frame, append
  to the rolling ``K``-length buffer, run the predictor on the current
  buffer (up to ``K`` frames), and emit the last frame's patch-prediction
  ``(B, H_p, W_p, D)``. O(1) amortized per call once the buffer is full.

The streaming path reuses the predictor on a growing (then rolling)
``(B, t, H_p, W_p, D)`` tensor with ``t ≤ K``; predictor accepts arbitrary
``T`` ≤ ``num_frames_max``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import keras
from keras import ops

from dl_techniques.regularizers.sigreg import SIGRegLayer

from .config import VideoJEPAConfig
from .encoder import VideoJEPACliffordEncoder
from .predictor import VideoJEPAPredictor
from .telemetry_embedder import TelemetryEmbedder


@keras.saving.register_keras_serializable()
class VideoJEPA(keras.Model):
    """Video-JEPA-Clifford top-level model.

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
        self.telemetry_embedder = TelemetryEmbedder(
            cond_dim=cfg.cond_dim,
            telemetry_dim=cfg.telemetry_dim,
            name="telemetry_embedder",
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

        # --- Streaming buffer (not a weight; reset per sequence) ---
        self._stream_buf: Optional[Any] = None

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

        :param inputs: dict with keys ``pixels`` (B, T, H, W, C) and
            ``telemetry`` (B, T, k).
        :param training: forwarded.
        :return: ``pred`` of shape (B, T, H_p, W_p, D).
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                "VideoJEPA expects inputs as a dict with keys 'pixels' and "
                f"'telemetry'. Got type={type(inputs)}."
            )
        pixels = inputs["pixels"]
        telemetry = inputs["telemetry"]

        z = self.encode_frames(pixels)                      # (B, T, H_p, W_p, D)
        c = self.telemetry_embedder(telemetry, training=training)  # (B, T, D)
        pred = self.predictor([z, c], training=training)    # (B, T, H_p, W_p, D)

        # --- MSE next-frame patch loss (D-003) ---
        # Skip if T < 2 (edge case: single-frame window).
        T = ops.shape(pixels)[1]
        # We cannot cleanly branch on a dynamic T inside tf.function; rely on
        # the fact that slicing pred[:, :-1] on T=1 yields an empty tensor —
        # ops.mean on empty is NaN, which we must avoid. Use a static Python
        # check on the config's num_frames, and only add the loss if T >= 2.
        if self.config.num_frames >= 2:
            pred_ctx = pred[:, :-1]           # (B, T-1, H_p, W_p, D)
            target_ctx = z[:, 1:]             # (B, T-1, H_p, W_p, D)
            pred_loss = ops.mean(ops.square(pred_ctx - target_ctx))
            self.add_loss(pred_loss)

        # --- SIGReg on (B*T, N, D) (D-005) ---
        cfg = self.config
        Hp = cfg.patches_per_side
        N = Hp * Hp
        B = ops.shape(pred)[0]
        pred_reshaped = ops.reshape(
            pred, (B * T, N, cfg.embed_dim)
        )
        sigreg_loss = self.sigreg(pred_reshaped)
        self.add_loss(self._sigreg_weight * sigreg_loss)

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
        telemetry_frame: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Advance the stream by one frame and return its patch prediction.

        :param frame: ``(B, H, W, C)`` single-frame pixel tensor.
        :param telemetry_frame: ``(B, k)`` single-frame telemetry vector.
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

        # Embed telemetry window — reuse the same number of frames as buffer.
        t = int(self._stream_buf.shape[1])
        # We only have `telemetry_frame` for the current frame; pad by
        # replicating for history (caller is expected to maintain their
        # own telemetry buffer in realistic use — for the in-library smoke
        # test we replicate).
        tel_1 = ops.expand_dims(telemetry_frame, axis=1)   # (B, 1, k)
        tel_window = ops.broadcast_to(
            tel_1, (ops.shape(tel_1)[0], t, cfg.telemetry_dim)
        )
        c = self.telemetry_embedder(tel_window, training=False)

        pred = self.predictor(
            [self._stream_buf, c], training=False
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

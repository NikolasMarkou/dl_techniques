"""
LeWM — Learning the World with Minimal Supervision, Keras 3 port.

Top-level model packaging encoder (ViT), projector, action-embedder,
autoregressive predictor, pred-projector, and SIGReg regularizer.

**Forward contract** (`call`):

- inputs: dict with keys
    * ``pixels``: float tensor, shape ``(B, T, H, W, C)`` — history + 1 future frame.
    * ``action``: float tensor, shape ``(B, T-1, action_dim)`` — actions taken
      between successive frames. The final action is internally padded with zeros
      so the action time axis matches the pixel time axis.
- returns: predicted embedding tensor of shape ``(B, T, embed_dim)``.

**Losses** are added via `self.add_loss()` inside `call` (so `model.fit`
with `loss=None` trains correctly):

- MSE prediction loss between ``pred_emb[:, :-1]`` and ``target_emb[:, 1:]``.
- SIGReg weighted by ``config.sigreg_weight``.

**Inference** — call `rollout(pixels_history, action_sequence)` for
autoregressive rollout. See method docstring for shape details.

See `/tmp/lewm_source/jepa.py` for the PyTorch reference; see decisions.md
entries D-001 (live target encoder, no EMA) and D-002 (MLPProjector uses
LayerNorm, matching upstream default).
"""

import keras
from keras import ops
from typing import Any, Dict, Optional

from dl_techniques.models.vit.model import ViT
from dl_techniques.models.lewm.config import LeWMConfig
from dl_techniques.models.lewm.embedder import ActionEmbedder
from dl_techniques.models.lewm.projector import MLPProjector
from dl_techniques.models.lewm.predictor import ARPredictor
from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class LeWM(keras.Model):
    """LeWM — JEPA-style action-conditioned world model.

    :param config: LeWMConfig dataclass (or None to use defaults).
    :param kwargs: passthrough to `keras.Model`.
    """

    def __init__(
        self,
        config: Optional[LeWMConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config if config is not None else LeWMConfig()

        cfg = self.config

        # Vision encoder: ViT-tiny (192d, patch=14, img=224), CLS-pooled feature.
        self.encoder = ViT(
            input_shape=cfg.input_image_shape,
            num_classes=1,  # ignored when include_top=False
            scale=cfg.encoder_scale,
            patch_size=cfg.patch_size,
            include_top=False,
            pooling="cls",
            name="encoder",
        )

        # Projector after encoder: (B, D_enc) -> (B, D) where D = embed_dim.
        # For tiny ViT D_enc = 192, embed_dim = 192, so this is an internal
        # LayerNorm+GELU refinement rather than a dim change.
        self.projector = MLPProjector(
            input_dim=cfg.embed_dim,
            hidden_dim=cfg.projector_hidden_dim,
            output_dim=cfg.embed_dim,
            name="projector",
        )

        self.action_encoder = ActionEmbedder(
            action_dim=cfg.action_dim,
            smoothed_dim=cfg.smoothed_dim,
            emb_dim=cfg.embed_dim,
            mlp_scale=cfg.mlp_scale,
            name="action_encoder",
        )

        self.predictor = ARPredictor(
            num_frames=cfg.num_frames,
            depth=cfg.depth,
            num_heads=cfg.heads,
            dim_head=cfg.dim_head,
            mlp_dim=cfg.mlp_dim,
            input_dim=cfg.embed_dim,
            hidden_dim=cfg.embed_dim,
            output_dim=cfg.embed_dim,
            dropout=cfg.dropout,
            emb_dropout=cfg.emb_dropout,
            name="predictor",
        )

        self.pred_proj = MLPProjector(
            input_dim=cfg.embed_dim,
            hidden_dim=cfg.projector_hidden_dim,
            output_dim=cfg.embed_dim,
            name="pred_proj",
        )

        self.sigreg = SIGRegLayer(
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
            name="sigreg",
        )

        self._sigreg_weight = cfg.sigreg_weight

    # ------------------------------------------------------------------
    # Core forward helpers
    # ------------------------------------------------------------------

    def encode_pixels(self, pixels: keras.KerasTensor) -> keras.KerasTensor:
        """Encode pixel batch `(B, T, H, W, C)` -> embedding `(B, T, D)`."""
        shape = ops.shape(pixels)
        B, T = shape[0], shape[1]
        H, W, C = self.config.img_size, self.config.img_size, self.config.img_channels
        flat = ops.reshape(pixels, (B * T, H, W, C))
        feat = self.encoder(flat)               # (B*T, D_enc)
        proj = self.projector(feat)             # (B*T, D)
        emb = ops.reshape(proj, (B, T, self.config.embed_dim))
        return emb

    def encode_actions(self, action: keras.KerasTensor) -> keras.KerasTensor:
        """Embed action batch `(B, T_a, A)` -> `(B, T_a, D)`."""
        return self.action_encoder(action)

    def predict_next(
        self,
        emb: keras.KerasTensor,
        act_emb: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Predict embeddings via ARPredictor + pred_proj."""
        pred = self.predictor([emb, act_emb], training=training)      # (B, T, D)
        B = ops.shape(pred)[0]
        T = ops.shape(pred)[1]
        D = self.config.embed_dim
        flat = ops.reshape(pred, (B * T, D))
        flat = self.pred_proj(flat, training=training)
        return ops.reshape(flat, (B, T, D))

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
            ``action`` (B, T-1, A).
        :param training: passed to submodules.
        :return: ``pred_emb`` of shape (B, T, D).
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                f"LeWM expects `inputs` to be a dict with 'pixels' and 'action' "
                f"keys. Got type={type(inputs)}."
            )
        pixels = inputs["pixels"]
        action = inputs["action"]

        # DECISION D-001: target encoder is live (no EMA, no stop_gradient).
        # Upstream LeWM uses the same encoder for both context and target.
        # Gradient flows through both paths. See decisions.md.
        emb = self.encode_pixels(pixels)                 # (B, T, D)

        # Pad action along time axis with zeros so act_emb has T timesteps.
        # Upstream pads before action_encoder (the "append zero action" trick).
        pad_shape = (ops.shape(action)[0], 1, self.config.action_dim)
        zero_pad = ops.zeros(pad_shape, dtype=action.dtype)
        action_padded = ops.concatenate([action, zero_pad], axis=1)      # (B, T, A)
        act_emb = self.encode_actions(action_padded)                     # (B, T, D)

        pred_emb = self.predict_next(emb, act_emb, training=training)    # (B, T, D)

        # Self-supervised MSE between pred[:, :-1] and target[:, 1:].
        pred_ctx = pred_emb[:, :-1]     # (B, T-1, D)
        target_ctx = emb[:, 1:]         # (B, T-1, D)
        pred_loss = ops.mean(ops.square(pred_ctx - target_ctx))
        self.add_loss(pred_loss)

        # SIGReg on the projected embeddings, shape (T, B, D) to match upstream.
        emb_tbd = ops.transpose(emb, (1, 0, 2))
        sigreg_loss = self.sigreg(emb_tbd)
        self.add_loss(self._sigreg_weight * sigreg_loss)

        return pred_emb

    # ------------------------------------------------------------------
    # Inference: autoregressive rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        pixels_history: keras.KerasTensor,
        action_sequence: keras.KerasTensor,
    ) -> Dict[str, keras.KerasTensor]:
        """Autoregressive rollout from a history of pixel observations.

        :param pixels_history: `(B, S, H_hist, H, W, C)` — history_size frames
            replicated across S action-plan samples.
        :param action_sequence: `(B, S, T, action_dim)` — full action sequence
            (history + future). H_hist = self.config.history_size.
        :return: dict with
            * ``predicted_emb``: `(B, S, T, D)` — predicted embeddings for
              the full rollout horizon (history frames use encoder output
              directly, future frames use predictor output).

        Implementation mirrors upstream `/tmp/lewm_source/jepa.py:rollout`.
        """
        cfg = self.config
        HS = cfg.history_size
        D = cfg.embed_dim

        B = ops.shape(action_sequence)[0]
        S = ops.shape(action_sequence)[1]
        T = ops.shape(action_sequence)[2]

        # Split actions into initial-history actions + future actions.
        # act_0: (B, S, HS, A); act_future: (B, S, T-HS, A).
        act_0 = action_sequence[:, :, :HS, :]
        act_future = action_sequence[:, :, HS:, :]

        # Encode history pixels — collapse (B, S) and use first sample plane
        # replicated over S to avoid re-encoding (S copies of same frames).
        # pixels_history: (B, S, HS, H, W, C). We take s=0 then tile.
        pixels_0 = pixels_history[:, 0]     # (B, HS, H, W, C)
        emb_0 = self.encode_pixels(pixels_0)  # (B, HS, D)

        # Broadcast over S: (B, S, HS, D) -> flatten (B*S, HS, D).
        emb = ops.broadcast_to(
            ops.expand_dims(emb_0, axis=1), (B, S, HS, D)
        )
        emb = ops.reshape(emb, (B * S, HS, D))

        act = ops.reshape(act_0, (B * S, HS, cfg.action_dim))
        act_future_flat = ops.reshape(act_future, (B * S, T - HS, cfg.action_dim))

        n_steps = T - HS
        for t in range(int(n_steps)):
            act_emb = self.encode_actions(act)
            # Truncate to last HS steps for the predictor's fixed window.
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb_step = self.predict_next(emb_trunc, act_trunc, training=False)
            pred_last = pred_emb_step[:, -1:]  # (B*S, 1, D)
            emb = ops.concatenate([emb, pred_last], axis=1)

            next_act = act_future_flat[:, t:t+1, :]
            act = ops.concatenate([act, next_act], axis=1)

        # Final step — one more prediction using the fully-assembled action seq.
        act_emb = self.encode_actions(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb_step = self.predict_next(emb_trunc, act_trunc, training=False)
        pred_last = pred_emb_step[:, -1:]
        emb = ops.concatenate([emb, pred_last], axis=1)

        # Reshape back (B*S, T_full, D) -> (B, S, T_full, D) and truncate to T.
        T_full = ops.shape(emb)[1]
        pred_rollout = ops.reshape(emb, (B, S, T_full, D))
        # The upstream rollout produces HS + n_steps + 1 steps; we keep all.
        return {"predicted_emb": pred_rollout}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"config": self.config.to_dict()})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects=None) -> "LeWM":
        cfg_dict = config.pop("config", None)
        cfg = LeWMConfig.from_dict(cfg_dict) if cfg_dict is not None else LeWMConfig()
        return cls(config=cfg, **config)

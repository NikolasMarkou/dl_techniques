"""``LeWM``, an action-conditioned world model that predicts in embedding space, plus the ``create_lewm`` factory.

It encodes each frame with a ViT, then predicts the next frame's embedding
from past embeddings and the action taken, using an AdaLN-zero conditional
transformer. There is no separate pixel decoder and no EMA target encoder:
the same live encoder produces both the prediction target and the context,
and a SIGReg term keeps the embedding space from collapsing instead.

Call takes a dict with ``pixels`` (B, T, H, W, C: history plus one future
frame) and ``action`` (B, T-1, action_dim), and returns predicted
embeddings of shape (B, T, embed_dim). The MSE prediction loss and the
weighted SIGReg loss are both added internally via ``add_loss``, so
``model.fit`` trains correctly with ``loss=None``. Use ``rollout`` for
autoregressive inference.

References:
    - Sobal et al., 2024. Learning the World with Minimal Supervision (LeWM).
    - Assran et al., 2023. Self-Supervised Learning from Images with a Joint-
      Embedding Predictive Architecture (I-JEPA). CVPR 2023.
      (https://arxiv.org/abs/2301.08243)
    - LeCun, 2022. A Path Towards Autonomous Machine Intelligence.
      (OpenReview: BZ5a1r-kVsf)
    - Skean et al., 2025. SIGReg / hyperspherical-energy anti-collapse
      regularization, as implemented in ``dl_techniques.layers.sigreg``.
"""

import keras
from keras import ops
from typing import Any, Dict, Optional

from dl_techniques.models.vision.vit.model import ViT
from dl_techniques.models.vision.lewm.config import LeWMConfig
from dl_techniques.models.vision.lewm.embedder import ActionEmbedder
from dl_techniques.models.vision.lewm.projector import MLPProjector
from dl_techniques.models.vision.lewm.predictor import ARPredictor
from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.lewm.model")
class LeWM(keras.Model):
    """JEPA-style action-conditioned world model.

    Architecture:

    .. code-block:: text

        pixels  [B, T, H, W, C]        action  [B, T-1, A]
              |                             |
              v                        zero-pad to T
        ViT encoder (per frame)             |
              |                             v
        MLPProjector                  ActionEmbedder
              |  emb [B, T, D]              |  act_emb [B, T, D]
              +-------------+---------------+
                            v
                      ARPredictor
                            |
                      MLPProjector (pred_proj)
                            |
                            v
                 pred_emb  [B, T, D]
                            |
                +-----------+-----------+
                v                       v
          MSE(pred[:-1], emb[1:])   SIGReg(emb)
                added via add_loss, weighted and summed

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

        # CLS-pooled feature; num_classes is ignored since include_top=False.
        self.encoder = ViT(
            input_shape=cfg.input_image_shape,
            num_classes=1,
            scale=cfg.encoder_scale,
            patch_size=cfg.patch_size,
            include_top=False,
            pooling="cls",
            name="encoder",
        )

        # Maps the encoder feature to embed_dim; for the default scale the two dims already match.
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
            dropout_rate=cfg.dropout_rate,
            emb_dropout_rate=cfg.emb_dropout_rate,
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

        # Track the weighted MSE and SIGReg terms separately so the CSV log shows both, not just their sum.
        self.pred_loss_tracker = keras.metrics.Mean(name="pred_loss")
        self.sigreg_loss_tracker = keras.metrics.Mean(name="sigreg_loss")

    # ------------------------------------------------------------------
    # Core forward helpers
    # ------------------------------------------------------------------

    def encode_pixels(
        self,
        pixels: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Encode pixel batch `(B, T, H, W, C)` -> embedding `(B, T, D)`.

        :param training: forwarded to the encoder and projector so the
            train/eval mode is explicit (consistent with `predict_next`).
        """
        shape = ops.shape(pixels)
        B, T = shape[0], shape[1]
        H, W, C = self.config.img_size, self.config.img_size, self.config.img_channels
        flat = ops.reshape(pixels, (B * T, H, W, C))
        feat = self.encoder(flat, training=training)
        proj = self.projector(feat, training=training)
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
        pred = self.predictor([emb, act_emb], training=training)
        B = ops.shape(pred)[0]
        T = ops.shape(pred)[1]
        D = self.config.embed_dim
        flat = ops.reshape(pred, (B * T, D))
        flat = self.pred_proj(flat, training=training)
        return ops.reshape(flat, (B, T, D))

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    @staticmethod
    def _require_pixels_and_action(mapping: Any) -> Any:
        """Return ``(mapping["pixels"], mapping["action"])``, or raise.

        Shared between :meth:`build` and :meth:`call`, which both take the
        same dict-shaped argument and must fail the same way.

        :param mapping: The dict passed to ``build`` or ``call``.
        :return: The values under ``"pixels"`` and ``"action"``.
        :raises ValueError: If ``mapping`` is not a dict.
        """
        # DECISION plan-2026-08-23T091307-9a110062/D-426: keep this shared; do not inline into call and index the dict directly in build.
        # build runs first, so a direct index there would turn this ValueError into a bare TypeError (regression measured on video_jepa). See decisions.md.
        if not isinstance(mapping, dict):
            raise ValueError(
                f"LeWM expects `inputs` to be a dict with 'pixels' and 'action' "
                f"keys. Got type={type(mapping)}."
            )
        return mapping["pixels"], mapping["action"]

    def build(self, input_shape: Dict[str, Any]) -> None:
        """Materialize every weight-bearing sub-layer.

        Repeats ``call``'s forward shapes through the same
        ``encode_pixels`` / ``encode_actions`` / ``predict_next`` helpers,
        skipping only the loss tail (``add_loss`` and the trackers), which
        owns no weights. The batch axis is fixed at 1 since no weight shape
        depends on it.

        :param input_shape: dict with keys ``pixels`` (B, T, H, W, C) and
            ``action`` (B, T-1, A).
        """
        # DECISION plan-2026-08-23T091307-9a110062/D-424: build re-derives shapes instead of tracing call, since add_loss inside call raises on traced KerasTensor placeholders.
        # test_the_explicit_build_materializes_the_model.py pins that build and call materialize the same 188-layer population, so a drift fails loudly. See decisions.md.
        if self.built:
            return
        cfg = self.config
        pixels_shape, action_shape = self._require_pixels_and_action(input_shape)
        pixels = keras.KerasTensor((1,) + tuple(pixels_shape[1:]))
        action = keras.KerasTensor((1,) + tuple(action_shape[1:]))

        emb = self.encode_pixels(pixels)

        zero_pad = ops.zeros((1, 1, cfg.action_dim), dtype=action.dtype)
        action_padded = ops.concatenate([action, zero_pad], axis=1)
        act_emb = self.encode_actions(action_padded)

        self.predict_next(emb, act_emb)
        self.sigreg(ops.transpose(emb, (1, 0, 2)))

        super().build(input_shape)

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
        pixels, action = self._require_pixels_and_action(inputs)

        # The same live encoder produces both context and target embeddings; gradient flows through both.
        emb = self.encode_pixels(pixels, training=training)

        # Zero-pad the action sequence to T timesteps before embedding it.
        pad_shape = (ops.shape(action)[0], 1, self.config.action_dim)
        zero_pad = ops.zeros(pad_shape, dtype=action.dtype)
        action_padded = ops.concatenate([action, zero_pad], axis=1)
        act_emb = self.encode_actions(action_padded)

        pred_emb = self.predict_next(emb, act_emb, training=training)

        pred_ctx = pred_emb[:, :-1]
        target_ctx = emb[:, 1:]
        pred_loss = ops.mean(ops.square(pred_ctx - target_ctx))
        self.add_loss(pred_loss)

        # Transpose to (T, B, D) to match the SIGReg layer's expected axis order.
        emb_tbd = ops.transpose(emb, (1, 0, 2))
        sigreg_loss = self.sigreg(emb_tbd)
        weighted_sigreg = self._sigreg_weight * sigreg_loss
        self.add_loss(weighted_sigreg)

        self.pred_loss_tracker.update_state(pred_loss)
        self.sigreg_loss_tracker.update_state(weighted_sigreg)

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

        :param pixels_history: `(B, S, HS, H, W, C)` — ``HS = history_size``
            frames. Only the ``s = 0`` plane is encoded. ``S`` must equal 1;
            to roll out distinct per-sample histories, tile them externally
            or call ``rollout`` once per history.
        :param action_sequence: `(B, S, T, action_dim)` — full action sequence
            of horizon ``T`` (history + future), with ``T >= history_size``.
        :return: dict with ``predicted_emb`` of shape `(B, S, T + 1, D)`. The
            rollout keeps every step it produces: the first ``HS`` entries
            along the time axis are encoder-derived embeddings of the
            observed history, and the remaining ``T + 1 - HS`` entries are
            predictor-derived. Score only the predictor-derived tail against
            ground truth.
        """
        cfg = self.config
        HS = cfg.history_size
        D = cfg.embed_dim

        B = ops.shape(action_sequence)[0]
        S = ops.shape(action_sequence)[1]
        T = ops.shape(action_sequence)[2]

        # rollout runs an eager Python loop, so T is concrete here; reject a horizon shorter than the history window.
        if int(T) < HS:
            raise ValueError(
                f"rollout: action_sequence horizon T={int(T)} must be >= "
                f"history_size={HS}."
            )

        # Distinct per-S histories would otherwise be silently dropped, since only pixels_history[:, 0] is encoded.
        if int(S) != 1:
            raise ValueError(
                f"rollout: S must equal 1 (got S={int(S)}). Only "
                f"pixels_history[:, 0] is encoded; passing distinct per-S "
                f"histories would be silently dropped. Tile externally or "
                f"call rollout once per history."
            )

        act_0 = action_sequence[:, :, :HS, :]
        act_future = action_sequence[:, :, HS:, :]

        # Encode the shared history once, then broadcast it over S rather than re-encoding S copies.
        pixels_0 = pixels_history[:, 0]
        emb_0 = self.encode_pixels(pixels_0, training=False)

        emb = ops.broadcast_to(
            ops.expand_dims(emb_0, axis=1), (B, S, HS, D)
        )
        emb = ops.reshape(emb, (B * S, HS, D))

        act = ops.reshape(act_0, (B * S, HS, cfg.action_dim))
        act_future_flat = ops.reshape(act_future, (B * S, T - HS, cfg.action_dim))

        n_steps = T - HS
        for t in range(int(n_steps)):
            act_emb = self.encode_actions(act)
            # Truncate to the predictor's fixed HS-step window.
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb_step = self.predict_next(emb_trunc, act_trunc, training=False)
            pred_last = pred_emb_step[:, -1:]
            emb = ops.concatenate([emb, pred_last], axis=1)

            next_act = act_future_flat[:, t:t+1, :]
            act = ops.concatenate([act, next_act], axis=1)

        # One more prediction step using the fully-assembled action sequence.
        act_emb = self.encode_actions(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb_step = self.predict_next(emb_trunc, act_trunc, training=False)
        pred_last = pred_emb_step[:, -1:]
        emb = ops.concatenate([emb, pred_last], axis=1)

        # T_full = HS + n_steps + 1 = T + 1: every produced step is kept, none truncated.
        T_full = ops.shape(emb)[1]
        pred_rollout = ops.reshape(emb, (B, S, T_full, D))
        return {"predicted_emb": pred_rollout}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def metrics(self):
        """Return the framework's own trackers plus the per-component loss trackers.

        :return: list of metrics, so Keras resets all of them per epoch and
            the CSV log carries ``pred_loss`` / ``sigreg_loss`` next to ``loss``.
        """
        return [*super().metrics, self.pred_loss_tracker, self.sigreg_loss_tracker]

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


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_lewm(
    config: Optional[LeWMConfig] = None,
    **overrides: Any,
) -> LeWM:
    """Create a LeWM action-conditioned world model.

    There is no ``MODEL_VARIANTS`` table and none was invented: LeWM ships one
    configuration (``LeWMConfig``, mirroring the upstream YAML defaults) and is
    retuned field by field rather than by picking a named scale, so there is no
    published scale family to enumerate. Encoder size is the one scale knob and
    it lives in ``LeWMConfig.encoder_scale``, forwarded to ``ViT``.

    :param config: A ``LeWMConfig``; ``None`` uses the upstream defaults.
    :param overrides: Individual ``LeWMConfig`` field overrides applied on top
        of ``config``. Any key that is not a config field is forwarded to
        ``keras.Model`` instead (e.g. ``name``).
    :returns: A configured ``LeWM`` instance.
    :raises ValueError: If the resulting config is inconsistent (e.g.
        ``num_frames`` too small to cover ``history_size + num_preds``).

    Example::

        model = create_lewm(img_size=64, patch_size=16, depth=1, history_size=2)
        out = model({"pixels": pixels, "action": actions})
    """
    base = config if config is not None else LeWMConfig()
    fields = set(LeWMConfig.__dataclass_fields__)
    cfg_overrides = {k: v for k, v in overrides.items() if k in fields}
    model_kwargs = {k: v for k, v in overrides.items() if k not in fields}

    if cfg_overrides:
        merged = base.to_dict()
        merged.update(cfg_overrides)
        # num_frames is a serialized field with a 0 sentinel: if the caller
        # resized the horizon without restating it, re-derive it rather than
        # carrying the old value forward into a ValueError.
        if "num_frames" not in cfg_overrides:
            merged["num_frames"] = 0
        base = LeWMConfig.from_dict(merged)

    return LeWM(config=base, **model_kwargs)

# ---------------------------------------------------------------------

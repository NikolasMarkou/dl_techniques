"""``LeVJEPATrainingModel``: the LeVJEPA multiview training wrapper.

Ports the LeVJEPA PyTorch reference's ``multiview_forward``, minus the two
``rearrange`` calls that convert between channels-first and channels-last
layouts: this repo's video tensors are already channels-last (``B, T, H, W,
C``) at every call site, so those two lines are dropped rather than ported.

Runs the shared encoder over a global view and every local view, projects
the CLS tokens with :class:`LeVJEPAProjector`, and adds two loss terms
(prediction and SIGReg) via ``self.add_loss(...)`` inside :meth:`call`, so
the model trains under a stock ``model.compile(loss=None)`` plus
``model.fit()``, with no custom ``train_step``.

References:
    - LeVJEPA PyTorch reference, ``main.py::multiview_forward`` (pasted
      transcript; no public arXiv id in this plan's context).
"""

import keras
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.models.vision.levjepa.encoder import LeVJEPAEncoder
from dl_techniques.models.vision.levjepa.projector import LeVJEPAProjector
from dl_techniques.regularizers.sigreg import SIGRegLayer

# ---------------------------------------------------------------------

# DECISION plan-2026-09-03T113223-2a714a91/D-017: `to_float_normalized` is a
# plain cast to float32, not ImageNet mean/std normalization -- this port's data
# sources already emit float32 in [0, 1]. Add mean/std at the dataset boundary
# if a future domain needs it. See decisions.md.


@register_dl_technique("dl_techniques.models.levjepa.training")
class LeVJEPATrainingModel(keras.Model):
    """LeVJEPA multiview self-supervised training wrapper.

    Owns one shared :class:`LeVJEPAEncoder` (run on both the global and the
    local views), one :class:`LeVJEPAProjector`, and one
    :class:`~dl_techniques.regularizers.sigreg.SIGRegLayer`
    (``normalize_by_n=True``, per D-003 -- this training model is the one
    consumer of that flag). Both loss terms are added via ``self.add_loss(...)``
    inside :meth:`call`, so the model trains under a stock
    ``model.compile(loss=None)`` + ``model.fit()``.

    Architecture:

    .. code-block:: text

        {"global_frame": [B,T,H,W,C], "local_frames": [B,V,T,H,W,C]}
            |                                    |
        encoder(global_frame)          encoder(reshape local_frames -> [B*V,T,H,W,C])
        -> [B, 1+N, D]                  -> [B*V, 1+N, D]
            |                                    |
        CLS at index 0                  CLS at index 0, reshape -> [B, V, D]
            +----------------- concat(axis=1) ---+
                                |
                            [B, 1+V, D]
                                |
                            projector
                                |
                    embeddings [B, 1+V, D]
                       /                    \\
        global_emb = embeddings[:, :1]   sigreg(transpose -> [1+V, B, D])
                       \\                    /
        pred_loss = mean((global_emb - embeddings)**2)   sigreg_loss
                       \\                    /
                loss = pred_loss + sigreg_weight * sigreg_loss
                        (both added via self.add_loss)

    :param encoder: A (built or unbuilt) :class:`LeVJEPAEncoder` in video mode
        (``num_frames > 1``), run on both the global and every local view.
        Passing an already-constructed encoder (e.g. from
        :func:`~dl_techniques.models.vision.levjepa.model.create_levjepa` or
        :func:`~dl_techniques.models.vision.levjepa.model.from_variant`)
        keeps encoder construction (variant tables, RoPE-vs-sincos, attention
        mode) entirely in ``model.py`` rather than duplicated here.
    :type encoder: LeVJEPAEncoder
    :param projector_hidden_dim: :class:`LeVJEPAProjector`'s hidden width.
        Defaults to ``2048``, matching the reference.
    :type projector_hidden_dim: int
    :param projector_output_dim: :class:`LeVJEPAProjector`'s output width.
        ``None`` (default) uses the encoder's ``embed_dim``.
    :type projector_output_dim: Optional[int]
    :param sigreg_knots: :class:`SIGRegLayer` integration-knot count. Defaults
        to ``17``, matching upstream.
    :type sigreg_knots: int
    :param sigreg_num_proj: :class:`SIGRegLayer` projection count. Defaults to
        ``1024``, matching upstream.
    :type sigreg_num_proj: int
    :param sigreg_weight: The paper's :math:`\\lambda` weighting the SIGReg
        term. Defaults to ``0.02``.
    :type sigreg_weight: float
    :param name: Model name; auto-generated when ``None``.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

    :ivar encoder: The shared :class:`LeVJEPAEncoder`.
    :ivar projector: The :class:`LeVJEPAProjector` head.
    :ivar sigreg: The :class:`SIGRegLayer` regularizer, ``normalize_by_n=True``.
    :ivar pred_loss_tracker: ``keras.metrics.Mean`` over the unweighted
        prediction loss, created in :meth:`build`.
    :ivar sigreg_loss_tracker: ``keras.metrics.Mean`` over the weighted SIGReg
        term, created in :meth:`build`.

    Input shape:
        A dict with keys ``"global_frame"`` ``(B, T, H, W, C)`` and
        ``"local_frames"`` ``(B, V, T, H, W, C)``.

    Output shape:
        ``(B, 1 + V, D)`` -- the projected embeddings of every view, global
        first.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.models.vision.levjepa.model import create_levjepa
        from dl_techniques.models.vision.levjepa.training import LeVJEPATrainingModel

        encoder = create_levjepa(
            variant="vit_tiny", input_shape=(32, 32, 3), num_frames=4,
        )
        model = LeVJEPATrainingModel(encoder=encoder, sigreg_num_proj=64)
        batch = {
            "global_frame": keras.random.normal((2, 4, 32, 32, 3)),
            "local_frames": keras.random.normal((2, 2, 4, 32, 32, 3)),
        }
        embeddings = model(batch)  # (2, 3, 192)
    """

    def __init__(
        self,
        encoder: LeVJEPAEncoder,
        projector_hidden_dim: int = 2048,
        projector_output_dim: Optional[int] = None,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        sigreg_weight: float = 0.02,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create every sub-layer.

        :raises ValueError: If ``encoder`` is not in video mode, or
            ``sigreg_weight`` is negative.
        """
        if name is None:
            name = "levjepa_training"
        super().__init__(name=name, **kwargs)

        if not isinstance(encoder, LeVJEPAEncoder):
            raise ValueError(
                f"encoder must be a LeVJEPAEncoder instance, got {type(encoder)}"
            )
        if not encoder.is_video:
            raise ValueError(
                "LeVJEPATrainingModel's multiview forward requires a "
                "video-mode encoder (num_frames > 1); got an image-mode "
                f"encoder (num_frames={encoder.num_frames})."
            )
        if sigreg_weight < 0.0:
            raise ValueError(f"sigreg_weight must be >= 0, got {sigreg_weight}")

        self.encoder = encoder
        self.projector = LeVJEPAProjector(
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
            name="projector",
        )
        # normalize_by_n=True: reproduces the reference's shipped-config SIGReg
        # magnitude for this (n_views, batch, dim) usage -- see D-003.
        self.sigreg = SIGRegLayer(
            knots=sigreg_knots,
            num_proj=sigreg_num_proj,
            normalize_by_n=True,
            name="sigreg",
        )
        self.sigreg_weight = float(sigreg_weight)

        # Created in build(): the EMA shadow weights (one non-trainable
        # variable per encoder weight, after the encoder is built so shapes
        # are known -- see the update_ema_shadow docstring for why this
        # cannot be deferred to first-call) and the two loss trackers.
        self._ema_shadow_weights: list = []
        self._ema_shadow_seeded: bool = False
        self.pred_loss_tracker: Optional[keras.metrics.Mean] = None
        self.sigreg_loss_tracker: Optional[keras.metrics.Mean] = None

        logger.info(
            f"Initialized LeVJEPATrainingModel with encoder embed_dim="
            f"{encoder.embed_dim}, sigreg_weight={self.sigreg_weight}"
        )

    def build(self, input_shape: Dict[str, Any]) -> None:
        """Build the encoder, projector, SIGReg, EMA shadow, and trackers.

        :param input_shape: Dict with keys ``"global_frame"``
            ``(B, T, H, W, C)`` and ``"local_frames"`` ``(B, V, T, H, W, C)``.
        :type input_shape: Dict[str, Any]
        :raises ValueError: If ``input_shape`` is not a dict with both keys.
        """
        if self.built:
            return
        if not isinstance(input_shape, dict) or "global_frame" not in input_shape or "local_frames" not in input_shape:
            raise ValueError(
                "LeVJEPATrainingModel expects `inputs`/`input_shape` to be a "
                "dict with 'global_frame' and 'local_frames' keys. Got "
                f"{input_shape!r}."
            )

        global_shape = tuple(input_shape["global_frame"])
        if not self.encoder.built:
            self.encoder.build(global_shape)

        embed_dim = self.encoder.embed_dim
        self.projector.build((None, None, embed_dim))
        self.sigreg.build((None, None, embed_dim))

        # EMA shadow: one non-trainable variable per encoder weight, created
        # here (not lazily inside update_ema_shadow) since Keras 3's
        # StatelessScope forbids adding new state after built=True.
        #
        # DECISION plan-2026-09-03T113223-2a714a91/D-020: seeded with plain zeros,
        # not the encoder's live weights -- `ops.convert_to_numpy` raises inside
        # `build()`'s tracing context. The shadow is copy-seeded instead on
        # `update_ema_shadow`'s first call, a genuinely eager context. See decisions.md.
        self._ema_shadow_weights = [
            self.add_weight(
                name=f"ema_shadow_{i}",
                shape=w.shape,
                dtype=w.dtype,
                initializer="zeros",
                trainable=False,
            )
            for i, w in enumerate(self.encoder.weights)
        ]
        self._ema_shadow_seeded = False

        self.pred_loss_tracker = keras.metrics.Mean(name="pred_loss")
        self.sigreg_loss_tracker = keras.metrics.Mean(name="sigreg_loss")

        super().build(input_shape)

    def call(self, inputs: Dict[str, Any], training: Optional[bool] = None) -> Any:
        """Run the multiview forward pass and register both loss terms.

        :param inputs: Dict with ``"global_frame"`` ``(B, T, H, W, C)`` and
            ``"local_frames"`` ``(B, V, T, H, W, C)``.
        :type inputs: Dict[str, Any]
        :param training: Standard Keras training flag.
        :type training: Optional[bool]
        :return: ``embeddings``, shape ``(B, 1 + V, D)``.
        :rtype: keras.KerasTensor
        """
        global_frame = keras.ops.cast(inputs["global_frame"], "float32")
        local_frames = keras.ops.cast(inputs["local_frames"], "float32")

        batch_size = keras.ops.shape(global_frame)[0]
        num_local = local_frames.shape[1]
        t, h, w, c = (
            local_frames.shape[2],
            local_frames.shape[3],
            local_frames.shape[4],
            local_frames.shape[5],
        )

        global_tokens = self.encoder(global_frame, training=training)  # (B, 1+N, D)
        global_cls = global_tokens[:, :1, :]  # (B, 1, D)

        local_flat = keras.ops.reshape(local_frames, (batch_size * num_local, t, h, w, c))
        local_tokens = self.encoder(local_flat, training=training)  # (B*V, 1+N, D)
        local_cls_flat = local_tokens[:, 0, :]  # (B*V, D)
        local_cls = keras.ops.reshape(
            local_cls_flat, (batch_size, num_local, self.encoder.embed_dim)
        )  # (B, V, D)

        views_in = keras.ops.concatenate([global_cls, local_cls], axis=1)  # (B, 1+V, D)
        embeddings = self.projector(views_in, training=training)  # (B, 1+V, D)
        global_emb = embeddings[:, :1, :]  # (B, 1, D)

        # global_emb broadcasts against embeddings' (1+V) axis; the global-vs-
        # itself term (index 0) is exactly 0 and is kept in the mean, faithful
        # to the reference (it does not slice the global column out).
        pred_loss = keras.ops.mean(keras.ops.square(global_emb - embeddings))

        embeddings_vbd = keras.ops.transpose(embeddings, (1, 0, 2))  # (1+V, B, D)
        sigreg_loss = self.sigreg(embeddings_vbd, training=training)
        weighted_sigreg = self.sigreg_weight * sigreg_loss

        self.add_loss(pred_loss)
        self.add_loss(weighted_sigreg)

        self.pred_loss_tracker.update_state(pred_loss)
        self.sigreg_loss_tracker.update_state(weighted_sigreg)

        return embeddings

    def update_ema_shadow(self, decay: float) -> None:
        """Update the EMA shadow copy of the encoder's weights.

        Duck-typed contract for :class:`~dl_techniques.callbacks.ema_shadow_callback.EMAShadowCallback`
        (Step 5). Never called from inside a ``GradientTape``: the shadow
        weights receive no gradient, only a plain ``.assign()``.

        :param decay: The EMA decay for this update, ``shadow = decay *
            shadow + (1 - decay) * live``.
        :type decay: float
        """
        if not self._ema_shadow_seeded:
            # First call: a plain copy (decay=0 equivalent), not a blend
            # against the zero-initialized shadow (D-020) -- otherwise the
            # shadow would start at `(1 - decay) * live`, a scaled-down
            # version of the live weights, instead of the live weights
            # themselves.
            for shadow_w, live_w in zip(self._ema_shadow_weights, self.encoder.weights):
                shadow_w.assign(keras.ops.cast(live_w, shadow_w.dtype))
            self._ema_shadow_seeded = True
            return
        for shadow_w, live_w in zip(self._ema_shadow_weights, self.encoder.weights):
            shadow_w.assign(decay * shadow_w + (1.0 - decay) * keras.ops.cast(live_w, shadow_w.dtype))

    @property
    def metrics(self):
        """Expose the per-component loss trackers alongside the framework's
        own trackers, mirroring ``lewm/model.py`` -- so the CSV log carries
        ``pred_loss`` / ``sigreg_loss`` next to ``loss``."""
        trackers = []
        if self.pred_loss_tracker is not None:
            trackers.append(self.pred_loss_tracker)
        if self.sigreg_loss_tracker is not None:
            trackers.append(self.sigreg_loss_tracker)
        return [*super().metrics, *trackers]

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        :return: Dictionary holding every ``__init__`` parameter.
        :rtype: Dict[str, Any]

        Note:
            DECISION plan-2026-09-03T113223-2a714a91/D-018: the config carries the
            encoder's own full config via `keras.saving.serialize_keras_object`,
            not duplicated constructor arguments -- `projector`/`sigreg` are plain
            Layers Keras already reconstructs through build-then-load-weights. See decisions.md.
        """
        config = super().get_config()
        config.update(
            {
                "encoder": keras.saving.serialize_keras_object(self.encoder),
                "projector_hidden_dim": self.projector.hidden_dim,
                "projector_output_dim": self.projector.output_dim,
                "sigreg_knots": self.sigreg.knots,
                "sigreg_num_proj": self.sigreg.num_proj,
                "sigreg_weight": self.sigreg_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects=None) -> "LeVJEPATrainingModel":
        """Reconstruct from a config produced by :meth:`get_config`.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :param custom_objects: Optional custom objects for deserialization.
        :return: A new :class:`LeVJEPATrainingModel` instance.
        :rtype: LeVJEPATrainingModel
        """
        config = dict(config)
        encoder_config = config.pop("encoder")
        encoder = keras.saving.deserialize_keras_object(
            encoder_config, custom_objects=custom_objects
        )
        return cls(encoder=encoder, **config)


# ---------------------------------------------------------------------

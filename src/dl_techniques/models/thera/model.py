# DECISION plan_2026-06-11_f662207d/D-009
# The Thera model is a PURE RESIDUAL-FIELD predictor. Several choices are
# deliberate and must NOT be "simplified" / "completed":
#
#  1. The model outputs the RAW heat-field residual `(B, Hq, Wq, out_dim)`. The
#     MEAN/VAR denormalization and the `+ source_nearest` residual add live in
#     the TRAINER (run_train.py / step 11), NOT here. Do NOT add denorm or the
#     nearest-neighbour add inside `call`: the trainer owns the data statistics
#     and the upsampled-source term, and baking them in would double-apply them.
#  2. `from_variant` enumerates SIX real architectural configs
#     (backbone {edsr-baseline, rdn} x tail {air, plus, pro}). This is genuine
#     taxonomy (INV-8), not a cosmetic alias table: each pair is a distinct
#     network. Do NOT collapse it to a single config with flags.
#  3. backbone + tail are serialized via `keras.saving.serialize_keras_object`
#     inside `get_config` (and rebuilt via `deserialize_keras_object` in
#     `from_config`). Do NOT store only their string keys: a caller may pass a
#     custom backbone/tail instance to the ctor that no key would reconstruct.
#  4. backbone, tail, hypernetwork are FLAT attributes (no nested lists). Nested
#     layer lists silently fail to restore their weights through `.keras` reload
#     (LESSONS iter-1) -- the per-weight round-trip in the test is the oracle.
#
# See decisions.md D-009.
"""THERA model: backbone -> tail -> hypernetwork heat-field decoder.

This module assembles the THERA (Aliasing-Free Arbitrary-Scale Super-Resolution
with Neural Heat Fields) model from the step 4-7 components into a single
:class:`keras.Model`, plus a :func:`build_thera` factory and the six-config
:meth:`Thera.from_variant` taxonomy.

Pipeline (reference ``model/thera.py`` ``apply`` = ``apply_encoder`` then
``apply_decoder``)::

    encoding = tail(backbone(source))                 # apply_encoder
    field    = hypernetwork.decode(encoding, coords, t)  # apply_decoder

The ``backbone`` is an EDSR-baseline or RDN feature extractor (spatial-shape
preserving, no upsampling); the ``tail`` optionally refines those features
(identity / ConvNeXt / SwinIR); the :class:`TheraHypernetwork` turns the
encoding, at each query coordinate, into a per-pixel neural heat field and
evaluates it. Arbitrary-scale super-resolution falls out of the continuous
``coords`` grid: the SAME trained model decodes any query resolution.

The model output is the RAW residual field ``(B, Hq, Wq, out_dim)`` (see the
D-009 anchor): the MEAN/VAR denormalization and the ``+ source_nearest``
residual add are performed by the TRAINER, not the model.

Reference:
    Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
    Neural Heat Fields" (original JAX/Flax ``model/thera.py``).
"""

import keras
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.thera_heat_field import DEFAULT_K_INIT
from dl_techniques.models.thera.edsr_backbone import EDSRBackbone
from dl_techniques.models.thera.rdn_backbone import RDNBackbone
from dl_techniques.models.thera.tails import build_thera_tail
from dl_techniques.models.thera.hypernetwork import TheraHypernetwork

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

# THERA frequency-disk scale for the heat-field components init (reference).
DEFAULT_COMPONENTS_INIT_SCALE: float = 16.0

# THERA hidden width: 32 for the tiny "air" size, 512 otherwise.
_AIR_HIDDEN_DIM: int = 32
_DEFAULT_HIDDEN_DIM: int = 512

_VALID_BACKBONES: Tuple[str, ...] = ("edsr-baseline", "rdn")
_VALID_SIZES: Tuple[str, ...] = ("air", "plus", "pro")

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Thera(keras.Model):
    """THERA arbitrary-scale super-resolution model (pure residual-field predictor).

    Assembles a feature ``backbone`` (EDSR-baseline or RDN), an optional feature
    ``tail`` (air / plus / pro), and a :class:`TheraHypernetwork` neural-heat-field
    decoder. The model is called on a 3-tuple ``(source, coords, t)`` and returns
    the raw heat-field residual at the query coordinates.

    Output is the RAW residual field (see the module D-009 anchor): the trainer
    performs MEAN/VAR denormalization and adds the nearest-neighbour upsampled
    source -- the model does NOT.

    Args:
        hidden_dim: Heat-field hidden width ``N`` (frequency-component count).
            ``32`` for the THERA "air" size, ``512`` otherwise. Must be positive.
        out_dim: Output channel count (e.g. 3 for an RGB residual). Defaults to 3.
        backbone: A built-or-buildable feature-backbone LAYER instance
            (:class:`EDSRBackbone` or :class:`RDNBackbone`), shape-preserving
            ``(B, H, W, C_in) -> (B, H, W, C_feat)``.
        tail: A feature-refiner tail LAYER instance (air / plus / pro) consuming
            and (spatially) preserving the backbone features.
        k_init: Initial heat-conductivity scalar ``k`` forwarded to the
            hypernetwork. Defaults to the THERA reference
            ``sqrt(log 4) / (2*pi^2)`` when ``None``.
        components_init_scale: Frequency-disk scale forwarded to the heat-field
            ``components`` init. Defaults to ``16.0`` when ``None``.
        **kwargs: Forwarded to :class:`keras.Model`.

    Input:
        A 3-tuple ``(source, coords, t)``:
            - ``source``: ``(B, Hs, Ws, C_in)`` low-resolution image.
            - ``coords``: ``(B, Hq, Wq, 2)`` query coordinates (THERA pixel-center
              convention, channel order ``[h, w]``).
            - ``t``: heat-diffusion time, broadcastable to ``(B, 1)``.

    Output:
        ``(B, Hq, Wq, out_dim)`` raw residual field.

    Example:
        >>> model = build_thera(out_dim=3, backbone="edsr-baseline", size="air")
        >>> source = keras.random.normal((2, 16, 16, 3))
        >>> import numpy as np
        >>> from dl_techniques.layers.grid_sample import make_grid
        >>> coords = keras.ops.broadcast_to(
        ...     keras.ops.convert_to_tensor(make_grid(24))[None], (2, 24, 24, 2))
        >>> t = keras.ops.ones((2, 1))
        >>> out = model((source, coords, t))   # (2, 24, 24, 3)
    """

    # Six real architectural configs: (backbone, size). INV-8 taxonomy (D-009).
    MODEL_VARIANTS: Dict[str, Tuple[str, str]] = {
        "edsr-air": ("edsr-baseline", "air"),
        "edsr-plus": ("edsr-baseline", "plus"),
        "edsr-pro": ("edsr-baseline", "pro"),
        "rdn-air": ("rdn", "air"),
        "rdn-plus": ("rdn", "plus"),
        "rdn-pro": ("rdn", "pro"),
    }

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        backbone: keras.layers.Layer,
        tail: keras.layers.Layer,
        k_init: Optional[float] = None,
        components_init_scale: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if backbone is None or tail is None:
            raise ValueError("backbone and tail must be provided layer instances")

        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        # Preserve None-vs-explicit so get_config round-trips the caller's intent.
        self.k_init = DEFAULT_K_INIT if k_init is None else float(k_init)
        self.components_init_scale = (
            DEFAULT_COMPONENTS_INIT_SCALE
            if components_init_scale is None
            else float(components_init_scale)
        )

        # FLAT sublayer attributes (D-009: no nested lists -> reliable reload).
        self.backbone = backbone
        self.tail = tail
        self.hypernetwork = TheraHypernetwork(
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            k_init=self.k_init,
            components_init_scale=self.components_init_scale,
            name="hypernetwork",
        )

    def build(self, input_shape: Any) -> None:
        # Keras-3 four-strike build-ordering (CRITICAL for the SC-8 .keras
        # round-trip): explicitly build every sublayer with the correctly
        # propagated shape BEFORE super().build(), so a reload restores all
        # weights and no unbuilt-sublayer warning fires (LESSONS.md).
        #
        # input_shape is the 3-input list [source_shape, coords_shape, t_shape].
        source_shape, coords_shape, t_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        # backbone -> feature map. Propagate its shape via compute_output_shape.
        if not self.backbone.built:
            self.backbone.build(source_shape)
        feat_shape = self.backbone.compute_output_shape(source_shape)

        # tail -> encoding. All three tails expose compute_output_shape.
        if not self.tail.built:
            self.tail.build(feat_shape)
        encoding_shape = self.tail.compute_output_shape(feat_shape)

        # hypernetwork consumes [encoding, coords, t]; it normalizes a multi-input
        # build shape to the encoding shape internally.
        if not self.hypernetwork.built:
            self.hypernetwork.build([encoding_shape, coords_shape, t_shape])

        super().build(input_shape)

    # -----------------------------------------------------------------

    def apply_encoder(
        self,
        source: Any,
        training: Optional[bool] = None,
    ) -> Any:
        """THERA ``apply_encoder``: ``tail(backbone(source))`` -> encoding.

        Args:
            source: Low-resolution input image ``(B, Hs, Ws, C_in)``.
            training: Forwarded to the backbone and tail.

        Returns:
            Encoding feature map ``(B, Hs, Ws, C_feat)``.
        """
        feats = self.backbone(source, training=training)
        return self.tail(feats, training=training)

    def apply_decoder(
        self,
        encoding: Any,
        coords: Any,
        t: Any,
        return_jac: bool = False,
        training: Optional[bool] = None,
    ) -> Any:
        """THERA ``apply_decoder``: evaluate the heat field at the query coords.

        Args:
            encoding: Backbone+tail encoding ``(B, Hs, Ws, C_feat)``.
            coords: Query coordinates ``(B, Hq, Wq, 2)`` (pixel-center, ``[h, w]``).
            t: Heat-diffusion time, broadcastable to ``(B, 1)``.
            return_jac: Accepted for signature parity with the THERA reference;
                the coordinate-Jacobian wiring lives in step 9 (the Jacobian-TV
                regularizer), NOT here. Currently ignored.
            training: Forwarded to the hypernetwork.

        Returns:
            Raw residual field ``(B, Hq, Wq, out_dim)``.
        """
        # return_jac is intentionally a pass-through no-op here (step 9 owns the
        # nested-tape Jacobian; the model stays a plain forward predictor).
        return self.hypernetwork.decode(encoding, coords, t, training=training)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Forward pass: ``inputs = (source, coords, t)`` -> residual field.

        Args:
            inputs: 3-tuple ``(source, coords, t)``.
            training: Forwarded to the backbone, tail, and hypernetwork.

        Returns:
            Raw residual field ``(B, Hq, Wq, out_dim)``.
        """
        source, coords, t = inputs
        encoding = self.apply_encoder(source, training=training)
        return self.apply_decoder(encoding, coords, t, training=training)

    # -----------------------------------------------------------------

    @classmethod
    def from_variant(cls, variant: str, **overrides: Any) -> "Thera":
        """Build one of the six THERA configs by name.

        The six configs are the real ``backbone x tail`` taxonomy (D-009):
        ``{edsr,rdn}-{air,plus,pro}``.

        Args:
            variant: One of ``MODEL_VARIANTS`` keys
                (``edsr-air``, ``edsr-plus``, ``edsr-pro``, ``rdn-air``,
                ``rdn-plus``, ``rdn-pro``).
            **overrides: Forwarded to :func:`build_thera` (e.g. ``out_dim``,
                ``k_init``, ``components_init_scale``).

        Returns:
            A constructed :class:`Thera` model.

        Raises:
            ValueError: If ``variant`` is not a known config name.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown THERA variant '{variant}'; expected one of "
                f"{sorted(cls.MODEL_VARIANTS)}"
            )
        backbone_key, size = cls.MODEL_VARIANTS[variant]
        logger.info(f"Thera.from_variant('{variant}') -> backbone={backbone_key}, size={size}")
        return build_thera(backbone=backbone_key, size=size, **overrides)

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "k_init": self.k_init,
            "components_init_scale": self.components_init_scale,
            # Serialize the actual backbone/tail instances (D-009): a caller may
            # have passed a custom layer no string key could reconstruct.
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "tail": keras.saving.serialize_keras_object(self.tail),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Thera":
        config = dict(config)
        config["backbone"] = keras.saving.deserialize_keras_object(
            config["backbone"]
        )
        config["tail"] = keras.saving.deserialize_keras_object(config["tail"])
        return cls(**config)


# ---------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------


def build_thera(
    out_dim: int = 3,
    backbone: str = "edsr-baseline",
    size: str = "pro",
    k_init: Optional[float] = None,
    components_init_scale: Optional[float] = None,
) -> Thera:
    """Build a THERA model from a backbone key + size key (THERA ``build_thera``).

    Args:
        out_dim: Output channel count (3 for RGB residual). Defaults to 3.
        backbone: ``"edsr-baseline"`` (EDSR feature extractor) or ``"rdn"``
            (Residual Dense Network). Defaults to ``"edsr-baseline"``.
        size: ``"air"`` (identity tail, ``hidden_dim=32``), ``"plus"`` (ConvNeXt
            tail), or ``"pro"`` (SwinIR tail). ``hidden_dim`` is ``512`` for
            ``plus``/``pro``. Defaults to ``"pro"``.
        k_init: Heat-conductivity init forwarded to the model. ``None`` -> THERA
            reference default.
        components_init_scale: Frequency-disk scale forwarded to the model.
            ``None`` -> ``16.0``.

    Returns:
        A constructed :class:`Thera` model.

    Raises:
        ValueError: If ``backbone`` or ``size`` is not a known key.
    """
    if backbone not in _VALID_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'; expected one of {list(_VALID_BACKBONES)}"
        )
    if size not in _VALID_SIZES:
        raise ValueError(
            f"Unknown size '{size}'; expected one of {list(_VALID_SIZES)}"
        )

    hidden_dim = _AIR_HIDDEN_DIM if size == "air" else _DEFAULT_HIDDEN_DIM

    if backbone == "edsr-baseline":
        backbone_layer: keras.layers.Layer = EDSRBackbone(
            num_feats=64, num_blocks=16, name="backbone_edsr"
        )
    else:  # "rdn"
        backbone_layer = RDNBackbone(name="backbone_rdn")

    tail_layer = build_thera_tail(size)

    return Thera(
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        backbone=backbone_layer,
        tail=tail_layer,
        k_init=k_init,
        components_init_scale=components_init_scale,
    )

# ---------------------------------------------------------------------

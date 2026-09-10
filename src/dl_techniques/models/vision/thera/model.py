"""THERA model: backbone, tail, hypernetwork heat-field decoder.

``Thera`` assembles a feature backbone, a refiner tail, and
``TheraHypernetwork`` into one ``keras.Model``, alongside a ``build_thera``
factory and the six-config ``Thera.from_variant`` taxonomy (``{edsr-baseline,
rdn}`` backbone times ``{air, plus, pro}`` tail). The backbone extracts
features at the input resolution and the tail refines them, then the
hypernetwork evaluates a per-pixel neural heat field at whatever query
coordinates it is handed, so one trained model decodes any target resolution.
The model is called on a 3-tuple ``(source, coords, t)``. Mean and variance
denormalization and the nearest-neighbour source add happen in the trainer,
not here.

References:
    - Becker et al. Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
      Neural Heat Fields.
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.thera_heat_field import DEFAULT_K_INIT
from dl_techniques.models.vision.thera.edsr_backbone import EDSRBackbone
from dl_techniques.models.vision.thera.rdn_backbone import RDNBackbone
from dl_techniques.models.vision.thera.tails import build_thera_tail
from dl_techniques.models.vision.thera.hypernetwork import TheraHypernetwork
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# THERA frequency-disk scale for the heat-field components init (reference).
DEFAULT_COMPONENTS_INIT_SCALE: float = 16.0

# THERA hidden width: 32 for the tiny "air" size, 512 otherwise.
_AIR_HIDDEN_DIM: int = 32
_DEFAULT_HIDDEN_DIM: int = 512

_VALID_BACKBONES: Tuple[str, ...] = ("edsr-baseline", "rdn")
_VALID_SIZES: Tuple[str, ...] = ("air", "plus", "pro")

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.thera.model")
class Thera(keras.Model):
    """THERA arbitrary-scale super-resolution model.

    Assembles a feature `backbone` (EDSR-baseline or RDN), a feature `tail`
    (air is an identity, plus and pro refine), and a
    :class:`TheraHypernetwork` decoder. The model is called on a 3-tuple
    `(source, coords, t)` and returns the raw heat-field residual at the query
    coordinates. The query height and width come from `coords`, so the same
    weights decode any output resolution.

    Architecture:

    .. code-block:: text

          source [B, Hs, Ws, C_in]    coords [B, Hq, Wq, 2]   t [B, 1]
                      │                         │                 │
                      ▼                         │                 │
            ┌───────────────────┐               │                 │
            │     backbone      │               │                 │
            └───────────────────┘               │                 │
                      │ [B, Hs, Ws, C_feat]     │                 │
                      ▼                         │                 │
            ┌───────────────────┐               │                 │
            │       tail        │               │                 │
            └───────────────────┘               │                 │
                      │ encoding                │                 │
                      │                         │                 │
                      └────────────────────┬────┴─────────────────┘
                                           ▼
                               ┌───────────────────────┐
                               │  hypernetwork.decode  │
                               └───────────────────────┘
                                           │
                            ┌──────────────┴──────────────┐
                            ▼                             ▼
                     residual field                   jacobian
                  [B, Hq, Wq, out_dim]            [.., out_dim, 2]
                                                  (return_jac only)

    The backbone is shape-preserving, so the encoding stays at source resolution.

    Variants:

    .. code-block:: text

        variant      backbone        tail   hidden_dim
        edsr-air     edsr-baseline   air            32
        edsr-plus    edsr-baseline   plus          512
        edsr-pro     edsr-baseline   pro           512
        rdn-air      rdn             air            32
        rdn-plus     rdn             plus          512
        rdn-pro      rdn             pro           512

    `hidden_dim` follows the size key and is applied by `build_thera`.

    :param hidden_dim: Heat-field hidden width, the frequency-component count.
        `32` for the `air` size, `512` otherwise. Must be positive.
    :type hidden_dim: int
    :param out_dim: Output channel count, e.g. 3 for an RGB residual.
    :type out_dim: int
    :param backbone: A built-or-buildable feature-backbone layer instance
        (:class:`EDSRBackbone` or :class:`RDNBackbone`), shape-preserving.
    :type backbone: keras.layers.Layer
    :param tail: A feature-refiner tail layer instance (air / plus / pro). A
        manually built `plus` tail for a backbone that does not emit 64
        channels needs `build_thera_tail('plus', in_channels=<backbone_out>)`,
        or its `build()` raises `ValueError`.
    :type tail: keras.layers.Layer
    :param k_init: Initial heat-conductivity scalar forwarded to the
        hypernetwork. Uses the THERA reference default when `None`.
    :type k_init: Optional[float]
    :param components_init_scale: Frequency-disk scale forwarded to the
        heat-field `components` init. Defaults to `16.0` when `None`.
    :type components_init_scale: Optional[float]
    :param kwargs: Forwarded to :class:`keras.Model`.
    :raises ValueError: If `hidden_dim` or `out_dim` is not positive, or if
        either `backbone` or `tail` is `None`.

    Input:
        A 3-tuple ``(source, coords, t)``:
            - ``source``: ``(B, Hs, Ws, C_in)`` low-resolution image.
            - ``coords``: ``(B, Hq, Wq, 2)`` query coordinates (pixel-center
              convention, channel order ``[h, w]``).
            - ``t``: heat-diffusion time, broadcastable to ``(B, 1)``.

    Output:
        ``(B, Hq, Wq, out_dim)`` raw residual field.

    Example:
        >>> model = build_thera(out_dim=3, backbone="edsr-baseline", size="air")
        >>> source = keras.random.normal((2, 16, 16, 3))
        >>> import numpy as np
        >>> from dl_techniques.layers.spatial_layer import coordinate_grid
        >>> coords = keras.ops.broadcast_to(
        ...     keras.ops.convert_to_tensor(coordinate_grid(24))[None], (2, 24, 24, 2))
        >>> t = keras.ops.ones((2, 1))
        >>> out = model((source, coords, t))
    """

    # Each value is a (backbone key, size key) pair for `build_thera`.
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
        """Initialize the model from prebuilt backbone and tail instances."""
        super().__init__(**kwargs)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if backbone is None or tail is None:
            raise ValueError("backbone and tail must be provided layer instances")

        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        # `None` resolves to the reference default here, so `get_config` always
        # emits a concrete value.
        self.k_init = DEFAULT_K_INIT if k_init is None else float(k_init)
        self.components_init_scale = (
            DEFAULT_COMPONENTS_INIT_SCALE
            if components_init_scale is None
            else float(components_init_scale)
        )

        # Flat attributes, not nested lists, so a .keras reload finds them (D-009).
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
        """Build every sublayer explicitly before `super().build()`.

        :param input_shape: The 3-input list `[source_shape, coords_shape, t_shape]`.
        """
        source_shape, coords_shape, t_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        # The feature shape comes from the backbone itself, not an assumption
        # about channel counts.
        if not self.backbone.built:
            self.backbone.build(source_shape)
        feat_shape = self.backbone.compute_output_shape(source_shape)

        # All three tails expose compute_output_shape.
        if not self.tail.built:
            self.tail.build(feat_shape)
        encoding_shape = self.tail.compute_output_shape(feat_shape)

        # The hypernetwork takes [encoding, coords, t] and reduces that to the
        # encoding shape internally.
        if not self.hypernetwork.built:
            self.hypernetwork.build([encoding_shape, coords_shape, t_shape])

        super().build(input_shape)

    # -----------------------------------------------------------------

    def apply_encoder(
        self,
        source: Any,
        training: Optional[bool] = None,
    ) -> Any:
        """Run `tail(backbone(source))` to produce the encoding.

        :param source: Low-resolution input image, shape `(B, Hs, Ws, C_in)`.
        :param training: Forwarded to the backbone and tail.
        :return: Encoding feature map, shape `(B, Hs, Ws, C_feat)`.
        :rtype: Any
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
        """Evaluate the heat field at the query coordinates.

        :param encoding: Backbone+tail encoding, shape `(B, Hs, Ws, C_feat)`.
        :param coords: Query coordinates, shape `(B, Hq, Wq, 2)`, pixel-center
            convention, channel order `[h, w]`.
        :param t: Heat-diffusion time, broadcastable to `(B, 1)`.
        :param return_jac: When `True`, also return the exact per-pixel
            spatial Jacobian `d(field)/d(rel_coords)` at `t=0`.
        :param training: Forwarded to the hypernetwork.
        :return: Raw residual field `(B, Hq, Wq, out_dim)` when `return_jac`
            is `False`; otherwise `(out, jac)` with `jac` shape
            `(B, Hq, Wq, out_dim, 2)`.
        :rtype: Any
        """
        if return_jac:
            return self.hypernetwork.decode_with_jac(
                encoding, coords, t, training=training
            )
        return self.hypernetwork.decode(encoding, coords, t, training=training)

    def call(
        self,
        inputs: Any,
        training: Optional[bool] = None,
        return_jac: bool = False,
    ) -> Any:
        """Forward pass: `inputs = (source, coords, t)` to residual field.

        :param inputs: 3-tuple `(source, coords, t)`.
        :param training: Forwarded to the backbone, tail, and hypernetwork.
        :param return_jac: When `True`, also return the exact spatial Jacobian
            `d(field)/d(rel_coords)` at `t=0`. Defaults to `False`.
        :return: Raw residual field `(B, Hq, Wq, out_dim)` when `return_jac`
            is `False`; otherwise `(out, jac)`.
        :rtype: Any
        """
        source, coords, t = inputs
        encoding = self.apply_encoder(source, training=training)
        return self.apply_decoder(
            encoding, coords, t, return_jac=return_jac, training=training
        )

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Output shape of `call((source, coords, t))`.

        The query height and width are set by `coords`, not by the
        low-resolution `source`: decoding runs at the resolution of the query
        grid.

        :param input_shape: 3-element list `[source_shape, coords_shape,
            t_shape]`, each a shape tuple. `coords_shape` is `(B, Hq, Wq, 2)`.
        :return: `(B, Hq, Wq, out_dim)`.
        :rtype: Tuple[Optional[int], ...]
        """
        coords_shape = input_shape[1]
        return (coords_shape[0], coords_shape[1], coords_shape[2], self.out_dim)

    # -----------------------------------------------------------------

    @classmethod
    def from_variant(cls, variant: str, **overrides: Any) -> "Thera":
        """Build one of the six THERA configs by name.

        :param variant: One of `MODEL_VARIANTS` keys (`edsr-air`, `edsr-plus`,
            `edsr-pro`, `rdn-air`, `rdn-plus`, `rdn-pro`).
        :type variant: str
        :param overrides: Forwarded to :func:`build_thera`, e.g. `out_dim`,
            `k_init`, `components_init_scale`.
        :return: A constructed :class:`Thera` model.
        :rtype: Thera
        :raises ValueError: If `variant` is not a known config name.
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
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary. Serializes the actual `backbone`
            and `tail` instances, not string keys, since a caller may have
            passed a custom layer no key could reconstruct.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "k_init": self.k_init,
            "components_init_scale": self.components_init_scale,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "tail": keras.saving.serialize_keras_object(self.tail),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Thera":
        """Create a model from a configuration dictionary.

        :param config: Configuration dictionary.
        :return: A new model instance.
        :rtype: Thera
        """
        config = dict(config)
        config["backbone"] = keras.saving.deserialize_keras_object(
            config["backbone"]
        )
        config["tail"] = keras.saving.deserialize_keras_object(config["tail"])
        return cls(**config)


# ---------------------------------------------------------------------


def build_thera(
    out_dim: int = 3,
    backbone: str = "edsr-baseline",
    size: str = "pro",
    k_init: Optional[float] = None,
    components_init_scale: Optional[float] = None,
) -> Thera:
    """Build a THERA model from a backbone key and a size key.

    The tail is built against the backbone's own output channel count, so the
    two always agree.

    :param out_dim: Output channel count, 3 for an RGB residual.
    :type out_dim: int
    :param backbone: `"edsr-baseline"` (EDSR feature extractor) or `"rdn"`
        (Residual Dense Network).
    :type backbone: str
    :param size: `"air"` (identity tail, `hidden_dim=32`), `"plus"` (ConvNeXt
        tail), or `"pro"` (SwinIR tail). `hidden_dim` is `512` for `plus`/`pro`.
    :type size: str
    :param k_init: Heat-conductivity init forwarded to the model. Uses the
        THERA reference default when `None`.
    :type k_init: Optional[float]
    :param components_init_scale: Frequency-disk scale forwarded to the
        model. Defaults to `16.0` when `None`.
    :type components_init_scale: Optional[float]
    :return: A constructed :class:`Thera` model.
    :rtype: Thera
    :raises ValueError: If `backbone` or `size` is not a known key.
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
    else:
        backbone_layer = RDNBackbone(name="backbone_rdn")

    feat_ch = backbone_layer.compute_output_shape((None, None, None, 3))[-1]
    tail_layer = build_thera_tail(size, in_channels=feat_ch)

    return Thera(
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        backbone=backbone_layer,
        tail=tail_layer,
        k_init=k_init,
        components_init_scale=components_init_scale,
    )

# ---------------------------------------------------------------------


def create_thera(
    variant: str = "edsr-pro",
    **overrides: Any,
) -> Thera:
    """Create a THERA model from one of the six named configs.

    An alias over :meth:`Thera.from_variant`. :func:`build_thera` is the
    equivalent entry point taking `backbone` and `size` separately.

    :param variant: One of `Thera.MODEL_VARIANTS` (`edsr-air`, `edsr-plus`,
        `edsr-pro`, `rdn-air`, `rdn-plus`, `rdn-pro`).
    :type variant: str
    :param overrides: Forwarded to :func:`build_thera`, e.g. `out_dim`,
        `k_init`, `components_init_scale`.
    :return: A constructed :class:`Thera` model.
    :rtype: Thera
    :raises ValueError: If `variant` is not a known config name.

    Example:
        >>> model = create_thera("edsr-air", out_dim=3)
    """
    return Thera.from_variant(variant, **overrides)

# ---------------------------------------------------------------------
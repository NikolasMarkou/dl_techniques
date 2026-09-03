"""SAM 2 FPN neck (:class:`SAM2FpnNeck`) and image encoder
(:class:`SAM2ImageEncoder`).

The neck turns the Hiera trunk's four ascending-stage feature levels into
four ``d_model``-wide levels plus one sine positional encoding per level:
one lateral ``1x1`` convolution per level, then a top-down addition over
the two coarsest levels. The encoder wraps trunk and neck and applies the
``scalp`` level drop, reading ``vision_features`` as the last of the
already-scalped levels.

Two index conventions are easy to invert without a shape error surfacing:
the trunk emits levels finest-first while ``backbone_channel_list`` is
widest-first, so lateral convolution ``convs[n - i]`` pairs with
``inputs[i]``; and ``scalp`` drops from the four built levels before
``vision_features`` is read, not after.

References:
    - Ravi et al., 2024. SAM 2: Segment Anything in Images and Videos.
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.embedding.positional_embedding_sine_2d import (
    PositionEmbeddingSine2D,
)
from dl_techniques.models.vision_language.sam.sam2.hiera import Hiera
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Interpolation names :class:`keras.layers.UpSampling2D` accepts here. The
#: single home for this rule -- ``__init__`` raises against it and
#: ``from_config`` normalizes into it.
_SUPPORTED_INTERP_MODELS = frozenset({"nearest", "bilinear"})

#: Spellings accepted by the pre-Keras top-down step that
#: :class:`keras.layers.UpSampling2D` does not accept. Applied by
#: ``from_config`` only, after ``.lower().strip()``.
_LEGACY_INTERP_ALIASES = {"nn": "nearest"}

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam2.neck")
class SAM2FpnNeck(keras.layers.Layer):
    """Feature-pyramid neck of the SAM 2 image encoder.

    Consumes the trunk's per-stage feature levels in **ascending stage order**
    (``inputs[0]`` finest and narrowest, ``inputs[-1]`` coarsest and widest),
    applies one ``1x1`` lateral convolution per level, fuses the two coarsest
    levels top-down, and computes a fixed sine positional encoding per fused
    level.

    **Index orientation.** ``backbone_channel_list`` is in DESCENDING channel
    order (widest first) because that is the order the trunk's own
    ``channel_list`` uses; the forward pass receives levels in the opposite
    order. Lateral convolution ``convs[n - i]`` is therefore the one applied to
    ``inputs[i]``, so ``inputs[i]`` must carry
    ``backbone_channel_list[n - i]`` channels. ``build`` asserts exactly that,
    so an accidentally reversed input list raises rather than silently training
    a transposed pyramid.

    **Top-down gating.** Only the levels named in ``fpn_top_down_levels``
    receive the upsampled coarser level; every other level is lateral-only.
    With the shipped ``(2, 3)`` and a four-level trunk this means level 3 is
    lateral-only in practice (it is visited first, so there is nothing to add
    yet) and only ``3 -> 2`` ever carries a top-down signal. Levels 0 and 1 --
    the two high-resolution levels the mask decoder consumes as skips -- get no
    cross-level fusion at all.

    **Positional-encoding width.** ``num_pos_feats`` is the parameter of
    :class:`PositionEmbeddingSine2D`, which emits ``2 * num_pos_feats``
    channels; the default ``d_model // 2`` therefore yields a ``d_model``-wide
    encoding, which is what the memory attention adds to its
    ``d_model``-wide input. Read the OUTPUT width, not this parameter, when
    comparing against the reference configuration. The resolved
    ``num_pos_feats`` must itself be EVEN, so a defaulted ``d_model`` must be a
    multiple of 4, not merely even. ``from_config`` is deliberately more
    lenient than ``__init__`` and rounds a non-conforming stored value UP with
    a warning rather than making an archive unloadable.

    :param d_model: Output channel width of every level.
    :type d_model: int
    :param backbone_channel_list: Trunk channel widths in DESCENDING order
        (widest/coarsest first).
    :type backbone_channel_list: Sequence[int]
    :param fpn_top_down_levels: Level indices (in ascending-stage indexing)
        that receive a top-down addition. ``None`` means every level.
    :type fpn_top_down_levels: Optional[Sequence[int]]
    :param fpn_interp_model: Interpolation passed to the owned
        :class:`keras.layers.UpSampling2D` sub-layer for the 2x top-down step.
        One of ``'nearest'`` or ``'bilinear'``; anything else raises. The
        legacy alias ``'nn'`` is accepted by ``from_config`` only, for
        archives written before this restriction.
    :type fpn_interp_model: str
    :param num_pos_feats: Half the positional-encoding output width.
        ``None`` defers to ``d_model // 2``.
    :type num_pos_feats: Optional[int]
    :param pos_enc_temperature: Sine positional-encoding temperature.
    :type pos_enc_temperature: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``d_model`` is not positive or not even, if
        ``d_model`` is not a multiple of 4 while ``num_pos_feats`` is
        defaulted, if the resolved ``num_pos_feats`` is not positive and even,
        if ``backbone_channel_list`` is empty or holds a non-positive width,
        if ``fpn_interp_model`` is outside ``{'nearest', 'bilinear'}``, or
        if a ``fpn_top_down_levels`` entry is out of range.

    Example:
        >>> import numpy as np
        >>> neck = SAM2FpnNeck(d_model=32, backbone_channel_list=(128, 64))
        >>> levels, positions = neck([
        ...     np.zeros((1, 8, 8, 64), dtype="float32"),
        ...     np.zeros((1, 4, 4, 128), dtype="float32"),
        ... ])
        >>> [tuple(level.shape) for level in levels]
        [(1, 8, 8, 32), (1, 4, 4, 32)]
    """

    def __init__(
            self,
            d_model: int = 256,
            backbone_channel_list: Sequence[int] = (1152, 576, 288, 144),
            fpn_top_down_levels: Optional[Sequence[int]] = (2, 3),
            fpn_interp_model: str = "nearest",
            num_pos_feats: Optional[int] = None,
            pos_enc_temperature: float = 10000.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even so the sine positional encoding can "
                f"split it between the two spatial axes, got {d_model}"
            )
        # DECISION plan-2026-08-28T181715-3870472c/D-007: the defaulted path
        # needs `% 4`, not `% 2` -- d_model=10 passed the old check but produced num_pos_feats=5, which PositionEmbeddingSine2D cannot split evenly. See decisions.md.
        if num_pos_feats is None and d_model % 4 != 0:
            raise ValueError(
                f"d_model ({d_model}) must be a positive multiple of 4 when "
                f"num_pos_feats is defaulted, not merely even: the sine "
                f"encoding receives num_pos_feats = d_model // 2 = "
                f"{d_model // 2}, and that value must ITSELF be even because "
                f"PositionEmbeddingSine2D splits it between its sine and "
                f"cosine halves. Use d_model = {((d_model + 3) // 4) * 4}."
            )
        if len(backbone_channel_list) == 0:
            raise ValueError("backbone_channel_list must not be empty")
        if any(int(c) <= 0 for c in backbone_channel_list):
            raise ValueError(
                f"backbone_channel_list entries must be positive, got "
                f"{tuple(backbone_channel_list)}"
            )
        if fpn_interp_model not in _SUPPORTED_INTERP_MODELS:
            raise ValueError(
                f"fpn_interp_model must be one of "
                f"{sorted(_SUPPORTED_INTERP_MODELS)}, got "
                f"{fpn_interp_model!r}. The legacy alias 'nn' (and any "
                f"uppercase or whitespace-padded spelling) is accepted by "
                f"from_config only, so archives written before this "
                f"restriction still load; new code must pass the exact "
                f"keras.layers.UpSampling2D interpolation name."
            )

        # Store ALL configuration parameters.
        self.d_model = int(d_model)
        self.backbone_channel_list = tuple(
            int(c) for c in backbone_channel_list)
        self.fpn_interp_model = fpn_interp_model
        self.pos_enc_temperature = float(pos_enc_temperature)

        num_levels = len(self.backbone_channel_list)
        if fpn_top_down_levels is None:
            self.fpn_top_down_levels: Optional[Tuple[int, ...]] = None
            self._top_down_levels = tuple(range(num_levels))
        else:
            levels = tuple(int(level) for level in fpn_top_down_levels)
            for level in levels:
                if not 0 <= level < num_levels:
                    raise ValueError(
                        f"fpn_top_down_levels entry {level} is out of range "
                        f"for {num_levels} levels"
                    )
            self.fpn_top_down_levels = levels
            self._top_down_levels = levels

        # DECISION plan-2026-08-04T044628-4c240b4c/D-013: this stays d_model //
        # 2, not d_model -- PositionEmbeddingSine2D emits 2*num_pos_feats channels, so the invariant is the OUTPUT width == d_model. See decisions.md.
        self.num_pos_feats = (
            self.d_model // 2 if num_pos_feats is None else int(num_pos_feats)
        )
        if self.num_pos_feats <= 0:
            raise ValueError(
                f"num_pos_feats must be positive, got {self.num_pos_feats}")
        if self.num_pos_feats % 2 != 0:
            raise ValueError(
                f"num_pos_feats ({self.num_pos_feats}) must be even because "
                f"PositionEmbeddingSine2D splits it between its sine and "
                f"cosine halves; it was "
                f"{'derived from d_model as d_model // 2' if num_pos_feats is None else 'passed explicitly'} "
                f"with d_model = {self.d_model}. Use "
                f"num_pos_feats = {self.num_pos_feats + 1}."
            )

        # Sub-layers -- created unconditionally, built explicitly in build().
        #
        # convs[j] is built for backbone_channel_list[j] and is applied to
        # inputs[n - j]. See the class docstring.
        self.convs = [
            keras.layers.Conv2D(
                filters=self.d_model,
                kernel_size=1,
                strides=1,
                padding="valid",
                name=f"lateral_conv_{index}",
            )
            for index in range(num_levels)
        ]
        #: Stateless (weightless) and therefore safe to share across levels.
        self.position_encoding = PositionEmbeddingSine2D(
            num_pos_feats=self.num_pos_feats,
            temperature=self.pos_enc_temperature,
            normalize=True,
            name="position_encoding",
        )
        # DECISION plan-2026-08-31T095434-b4829a10/D-002: build this
        # unconditionally, once, even when _top_down_levels is empty -- never move construction into call() or expand to a per-level list. See decisions.md D-002/D-003.
        self.upsample = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation=self.fpn_interp_model,
            name="top_down_upsample",
        )

    @property
    def num_levels(self) -> int:
        """Number of pyramid levels this neck produces.

        :return: ``len(backbone_channel_list)``.
        :rtype: int
        """
        return len(self.backbone_channel_list)

    @property
    def pos_enc_channels(self) -> int:
        """Channel width of each returned positional encoding.

        :return: ``2 * num_pos_feats``.
        :rtype: int
        """
        return 2 * self.num_pos_feats

    def build(self, input_shape: Sequence[Tuple[Optional[int], ...]]) -> None:
        """Build one lateral convolution per level.

        :param input_shape: One ``(batch, height, width, channels)`` shape per
            trunk level, in ascending stage order.
        :type input_shape: Sequence[Tuple[Optional[int], ...]]
        :raises ValueError: If the number of levels or any level's channel
            width disagrees with ``backbone_channel_list``.
        """
        if self.built:
            return

        shapes = [tuple(shape) for shape in input_shape]
        n = self.num_levels - 1
        if len(shapes) != self.num_levels:
            raise ValueError(
                f"SAM2FpnNeck expects {self.num_levels} trunk levels to match "
                f"backbone_channel_list, got {len(shapes)}"
            )

        for index, shape in enumerate(shapes):
            if len(shape) != 4:
                raise ValueError(
                    f"level {index} must be a rank-4 channels-last shape, got "
                    f"{shape}"
                )
            expected = self.backbone_channel_list[n - index]
            if shape[-1] is not None and int(shape[-1]) != expected:
                raise ValueError(
                    f"level {index} carries {shape[-1]} channels but "
                    f"backbone_channel_list[{n - index}] is {expected}; the "
                    f"trunk returns levels in ASCENDING stage order while "
                    f"backbone_channel_list is DESCENDING -- check the "
                    f"orientation of one of the two lists"
                )
            self.convs[n - index].build(shape)

        self.position_encoding.build(
            (shapes[0][0], shapes[0][1], shapes[0][2], self.d_model))

        # ONE representative build is correct here, and a per-level list would
        # be invented complexity: `UpSampling2D` creates no weights and its
        # build is shape-agnostic, so the same object serves every resolution
        # it is applied at. The shape is the FUSED one -- the upsample
        # consumes `previous`, which is post-lateral-convolution and therefore
        # `d_model`-wide, never a raw trunk width. Same precedent as
        # `position_encoding` directly above.
        self.upsample.build(
            (shapes[0][0], shapes[0][1], shapes[0][2], self.d_model))

        logger.debug(
            "SAM2FpnNeck built: %d levels, d_model %d, top-down at %s",
            self.num_levels, self.d_model, self._top_down_levels,
        )
        super().build(input_shape)

    def _sine_position_encoding(self, features: Any) -> Any:
        """Compute the channels-last sine positional encoding of a level.

        ``PositionEmbeddingSine2D`` emits ``(batch, channels, height, width)``
        in float32 at every dtype policy, so this transposes back to
        channels-last and casts to the feature dtype. Both steps are required;
        neither is cosmetic.

        :param features: Fused level, ``(batch, height, width, d_model)``.
        :type features: Any
        :return: ``(batch, height, width, 2 * num_pos_feats)``.
        :rtype: Any
        """
        position = self.position_encoding(features)
        position = ops.transpose(position, (0, 2, 3, 1))
        return ops.cast(position, features.dtype)

    def call(
            self,
            inputs: Sequence[Any],
            training: Optional[bool] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Fuse the trunk levels and encode their positions.

        :param inputs: One feature map per trunk level, ascending stage order.
        :type inputs: Sequence[Any]
        :param training: Keras training flag. Unused -- the neck holds no
            training-dependent behaviour.
        :type training: Optional[bool]
        :return: ``(levels, positions)``, both in ascending stage order.
        :rtype: Tuple[List[Any], List[Any]]
        """
        xs = list(inputs)
        n = len(xs) - 1

        levels: List[Any] = [None] * len(xs)
        positions: List[Any] = [None] * len(xs)

        previous: Optional[Any] = None
        # Coarse -> fine. `i` indexes the ASCENDING trunk list, so the loop
        # starts at the coarsest level and the lateral convolution index is
        # mirrored.
        for i in range(n, -1, -1):
            lateral = self.convs[n - i](xs[i])
            if i in self._top_down_levels and previous is not None:
                # Parameter-free 2x step, run through the owned
                # `keras.layers.UpSampling2D` sub-layer created in
                # `__init__` and built in `build`. Nothing is constructed
                # here.
                previous = lateral + self.upsample(previous)
            else:
                previous = lateral
            levels[i] = previous
            positions[i] = self._sine_position_encoding(previous)

        return levels, positions

    def compute_output_shape(
            self, input_shape: Sequence[Tuple[Optional[int], ...]]
    ) -> Tuple[
        List[Tuple[Optional[int], ...]], List[Tuple[Optional[int], ...]]
    ]:
        """Return the per-level feature and positional-encoding shapes.

        :param input_shape: One shape per trunk level, ascending stage order.
        :type input_shape: Sequence[Tuple[Optional[int], ...]]
        :return: ``(level_shapes, position_shapes)``.
        :rtype: Tuple[List[Tuple[Optional[int], ...]],
            List[Tuple[Optional[int], ...]]]
        """
        level_shapes = [
            (shape[0], shape[1], shape[2], self.d_model)
            for shape in (tuple(s) for s in input_shape)
        ]
        position_shapes = [
            (shape[0], shape[1], shape[2], self.pos_enc_channels)
            for shape in level_shapes
        ]
        return level_shapes, position_shapes

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "backbone_channel_list": self.backbone_channel_list,
            "fpn_top_down_levels": self.fpn_top_down_levels,
            "fpn_interp_model": self.fpn_interp_model,
            "num_pos_feats": self.num_pos_feats,
            "pos_enc_temperature": self.pos_enc_temperature,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SAM2FpnNeck":
        """Rebuild from a config, tolerating two superseded stored values.

        Guide section 6.3 migration path, in two clauses.

        **``fpn_interp_model``.** The 2x top-down step used to run through a
        helper that lowercased, stripped and accepted the alias ``'nn'``; the
        owned :class:`keras.layers.UpSampling2D` accepts none of that, and the
        constructor now raises on it. A stored value is therefore normalized
        with ``.lower().strip()`` and the legacy alias remapped, with a
        warning. Numerics are UNCHANGED by that substitution: ``'nn'`` and
        ``'nearest'`` reached the identical
        ``UpSampling2D(interpolation='nearest')`` before the change, and the
        sub-layer is weightless, so the rebuilt weight tree is identical too.

        **``num_pos_feats``.** The constructor's evenness rule on
        the resolved ``num_pos_feats`` is a NEW rejection of a value the old
        ``d_model % 2`` check accepted, so an archive written before it must
        still load. A non-conforming value is rounded UP here with a warning,
        never raised on. When the stored ``num_pos_feats`` is exactly
        ``d_model // 2`` -- the default -- ``d_model`` is widened to the next
        multiple of 4 alongside it, preserving the invariant
        ``pos_enc_channels == d_model``. Rounding up rather than down preserves
        capacity, and the width change cannot break anything that previously
        worked: such a neck's position encoder could never complete a forward
        pass, so no model carrying one was ever trainable or servable.

        :param config: Serialized configuration.
        :type config: Dict[str, Any]
        :return: The reconstructed neck.
        :rtype: SAM2FpnNeck
        """
        config = dict(config)
        interp = config.get("fpn_interp_model")
        if isinstance(interp, str):
            normalized = interp.lower().strip()
            normalized = _LEGACY_INTERP_ALIASES.get(normalized, normalized)
            if normalized != interp:
                logger.warning(
                    "SAM2FpnNeck config carries fpn_interp_model=%r, a legacy "
                    "spelling the constructor no longer accepts; remapping it "
                    "to %r. Numerics are UNCHANGED -- both reached the same "
                    "keras.layers.UpSampling2D(interpolation='%s') step, and "
                    "that sub-layer is weightless, so the rebuilt weight tree "
                    "is identical.",
                    interp, normalized, normalized,
                )
                config["fpn_interp_model"] = normalized

        d_model = config.get("d_model")
        num_pos_feats = config.get("num_pos_feats")
        if (isinstance(num_pos_feats, int) and num_pos_feats > 0
                and num_pos_feats % 2 != 0):
            substitute = num_pos_feats + 1
            widened = (isinstance(d_model, int)
                       and num_pos_feats == d_model // 2)
            logger.warning(
                "SAM2FpnNeck config carries num_pos_feats=%d (d_model=%s), "
                "which is odd; this archive predates the evenness requirement "
                "and its position encoder could never run a forward pass. "
                "Substituting num_pos_feats=%d%s. The position-encoding output "
                "width changes from %d to %d, so stored weights for this "
                "layer will not match.",
                num_pos_feats, d_model, substitute,
                (" and d_model=%d" % (2 * substitute)) if widened else "",
                2 * num_pos_feats, 2 * substitute,
            )
            config["num_pos_feats"] = substitute
            if widened:
                config["d_model"] = 2 * substitute
        return cls(**config)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam2.neck")
class SAM2ImageEncoder(keras.layers.Layer):
    """Hiera trunk plus FPN neck, with the ``scalp`` level drop.

    The order of the last two operations is the whole point of this class and
    is read from the reference implementation, not inferred:

    .. code-block:: text

        levels, positions = neck(trunk(images))
        levels, positions = levels[:-scalp], positions[:-scalp]
        vision_features   = levels[-1]

    The drop happens FIRST, so ``vision_features`` is the coarsest RETAINED
    level, not the coarsest level the neck built. At ``image_size=1024`` with
    four trunk stages the retained strides are 4 / 8 / 16 and
    ``vision_features`` is stride 16 -- a ``64 x 64`` grid, which is exactly
    the ``feat_sizes`` the memory attention's rotary tables are built for.
    Taking ``vision_features`` before the drop would give stride 32 and
    contradict it, with no shape error at this boundary.

    ``backbone_fpn[0]`` and ``backbone_fpn[1]`` (strides 4 and 8) are the
    high-resolution skips the mask decoder fuses into its upscaling path.

    :param trunk: The Hiera trunk, or its serialized configuration.
    :type trunk: Union[Hiera, Dict[str, Any]]
    :param neck: The FPN neck, or its serialized configuration.
    :type neck: Union[SAM2FpnNeck, Dict[str, Any]]
    :param scalp: Number of coarsest levels to discard.
    :type scalp: int
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``scalp`` is negative or would discard every level,
        or if the trunk and neck disagree on the number of levels or on their
        channel widths.

    Example:
        >>> import numpy as np
        >>> encoder = SAM2ImageEncoder.from_variant("tiny")
        >>> out = encoder(np.zeros((1, 64, 64, 3), dtype="float32"))
        >>> len(out["backbone_fpn"]), tuple(out["vision_features"].shape)
        (3, (1, 4, 4, 32))
    """

    #: Geometry per variant. The trunk half is read from
    #: ``Hiera.MODEL_VARIANTS`` at construction time -- it is NOT restated
    #: here, because a geometry restated in two homes is a latent defect.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "d_model": 32,
            "fpn_top_down_levels": (2, 3),
            "scalp": 1,
        },
        "hiera_l": {
            "d_model": 256,
            "fpn_top_down_levels": (2, 3),
            "scalp": 1,
        },
    }

    def __init__(
            self,
            trunk: Union[Hiera, Dict[str, Any]],
            neck: Union[SAM2FpnNeck, Dict[str, Any]],
            scalp: int = 1,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.trunk = (
            trunk if isinstance(trunk, keras.layers.Layer)
            else keras.saving.deserialize_keras_object(trunk)
        )
        self.neck = (
            neck if isinstance(neck, keras.layers.Layer)
            else keras.saving.deserialize_keras_object(neck)
        )
        self.scalp = int(scalp)

        num_levels = self.neck.num_levels
        if self.scalp < 0:
            raise ValueError(f"scalp must not be negative, got {self.scalp}")
        if self.scalp >= num_levels:
            raise ValueError(
                f"scalp={self.scalp} would discard every one of the "
                f"{num_levels} neck levels"
            )
        if len(self.trunk.channel_list) != num_levels:
            raise ValueError(
                f"the trunk produces {len(self.trunk.channel_list)} levels but "
                f"the neck is configured for {num_levels}"
            )
        if tuple(self.trunk.channel_list) != self.neck.backbone_channel_list:
            raise ValueError(
                f"the trunk's channel_list {tuple(self.trunk.channel_list)} "
                f"does not match the neck's backbone_channel_list "
                f"{self.neck.backbone_channel_list}; both are in DESCENDING "
                f"stage order"
            )

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "SAM2ImageEncoder":
        """Construct trunk, neck and encoder from the variant tables.

        :param variant: Variant key shared with ``Hiera.MODEL_VARIANTS``.
        :type variant: str
        :param kwargs: Explicit overrides for the encoder/neck geometry; any
            value given here wins over the variant table.
        :type kwargs: Any
        :return: The configured image encoder.
        :rtype: SAM2ImageEncoder
        :raises ValueError: If ``variant`` is not a known key.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown SAM2ImageEncoder variant '{variant}'. Available: "
                f"{sorted(cls.MODEL_VARIANTS)}"
            )
        config = dict(cls.MODEL_VARIANTS[variant])
        config.update(kwargs)
        scalp = config.pop("scalp")

        trunk = Hiera.from_variant(variant)
        neck = SAM2FpnNeck(
            backbone_channel_list=tuple(trunk.channel_list), **config)
        logger.info("Creating SAM2ImageEncoder variant '%s'", variant)
        return cls(trunk=trunk, neck=neck, scalp=scalp)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the trunk and the neck.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        self.trunk.build(tuple(input_shape))
        self.neck.build(self.trunk.compute_output_shape(tuple(input_shape)))
        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Encode one batch of images.

        :param inputs: ``(batch, height, width, channels)``.
        :type inputs: Any
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: ``{'vision_features', 'vision_pos_enc', 'backbone_fpn'}``.
            ``backbone_fpn`` and ``vision_pos_enc`` hold the RETAINED levels in
            ascending stage order; ``vision_features`` is the coarsest of them.
        :rtype: Dict[str, Any]
        """
        levels, positions = self.neck(
            self.trunk(inputs, training=training), training=training)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-014: drop from the coarse
        # end (levels[:-scalp]) before reading vision_features, never levels[scalp:] -- both keep the count at three but the wrong drop hands downstream the wrong skips with no shape error. See decisions.md.
        if self.scalp > 0:
            levels = levels[: -self.scalp]
            positions = positions[: -self.scalp]

        return {
            "vision_features": levels[-1],
            "vision_pos_enc": positions,
            "backbone_fpn": levels,
        }

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Any]:
        """Return the retained level shapes, derived from stored config.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Same keys as :meth:`call`.
        :rtype: Dict[str, Any]
        """
        trunk_shapes = self.trunk.compute_output_shape(tuple(input_shape))
        level_shapes, position_shapes = self.neck.compute_output_shape(
            trunk_shapes)
        if self.scalp > 0:
            level_shapes = level_shapes[: -self.scalp]
            position_shapes = position_shapes[: -self.scalp]
        return {
            "vision_features": level_shapes[-1],
            "vision_pos_enc": position_shapes,
            "backbone_fpn": level_shapes,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "trunk": keras.saving.serialize_keras_object(self.trunk),
            "neck": keras.saving.serialize_keras_object(self.neck),
            "scalp": self.scalp,
        })
        return config

# ---------------------------------------------------------------------

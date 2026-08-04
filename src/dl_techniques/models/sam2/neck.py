"""SAM 2 FPN neck and image encoder.

The neck turns the Hiera trunk's four ascending-stage feature levels into four
``d_model``-wide levels plus one sine positional encoding per level. The image
encoder wraps trunk + neck and applies the ``scalp`` level drop.

Two mechanisms in this file are silent when ported wrong -- they produce a
model that builds, forward-passes and trains, with no shape error anywhere:

1. **The lateral-convolution index orientation.** The trunk returns levels in
   ASCENDING stage order (finest first) while ``backbone_channel_list`` is in
   DESCENDING order (widest first). The lateral convolutions are therefore
   indexed ``convs[n - i]`` against ``xs[i]``. Reversing either list alone
   would still line the channel counts up.
2. **The ``scalp`` drop happens BEFORE ``vision_features`` is taken.** The neck
   builds four levels; the encoder returns three and reads the last of the
   already-scalped list. Dropping the other end keeps the count at three.

Both are guarded behaviourally in ``tests/test_models/test_sam2/test_neck.py``.
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.upsample import upsample
from dl_techniques.layers.embedding.positional_embedding_sine_2d import (
    PositionEmbeddingSine2D,
)
from dl_techniques.models.sam2.hiera import Hiera

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
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
    comparing against the reference configuration.

    :param d_model: Output channel width of every level.
    :type d_model: int
    :param backbone_channel_list: Trunk channel widths in DESCENDING order
        (widest/coarsest first).
    :type backbone_channel_list: Sequence[int]
    :param fpn_top_down_levels: Level indices (in ascending-stage indexing)
        that receive a top-down addition. ``None`` means every level.
    :type fpn_top_down_levels: Optional[Sequence[int]]
    :param fpn_interp_model: Interpolation passed to
        :func:`dl_techniques.layers.upsample.upsample` for the 2x top-down
        step.
    :type fpn_interp_model: str
    :param num_pos_feats: Half the positional-encoding output width.
        ``None`` defers to ``d_model // 2``.
    :type num_pos_feats: Optional[int]
    :param pos_enc_temperature: Sine positional-encoding temperature.
    :type pos_enc_temperature: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``d_model`` is not positive or not even, if
        ``backbone_channel_list`` is empty or holds a non-positive width, or if
        a ``fpn_top_down_levels`` entry is out of range.

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
        if len(backbone_channel_list) == 0:
            raise ValueError("backbone_channel_list must not be empty")
        if any(int(c) <= 0 for c in backbone_channel_list):
            raise ValueError(
                f"backbone_channel_list entries must be positive, got "
                f"{tuple(backbone_channel_list)}"
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

        # DECISION plan-2026-08-04T044628-4c240b4c/D-013
        # Do NOT "fix" this to `d_model` to match the reference config's
        # literal `num_pos_feats: 256`. That literal belongs to a class which
        # halves it internally; this repo's `PositionEmbeddingSine2D` does not,
        # and emits `2 * num_pos_feats` channels. Passing 256 here yields a
        # 512-wide encoding that cannot be added to the 256-wide features the
        # memory attention consumes. The invariant is the OUTPUT width:
        # `pos_enc_channels == d_model`. See decisions.md D-013.
        self.num_pos_feats = (
            self.d_model // 2 if num_pos_feats is None else int(num_pos_feats)
        )
        if self.num_pos_feats <= 0:
            raise ValueError(
                f"num_pos_feats must be positive, got {self.num_pos_feats}")

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
                # Parameter-free nearest 2x, reused from layers/upsample.py.
                # UpSampling2D is weightless, so constructing it here adds no
                # variables and nothing to track.
                previous = lateral + upsample(previous, self.fpn_interp_model)
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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
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

        # DECISION plan-2026-08-04T044628-4c240b4c/D-014
        # The drop is applied BEFORE `vision_features` is read. Do NOT
        # reorder these two lines, and do NOT drop from the finest end
        # (`levels[self.scalp:]`) -- that keeps the level COUNT at three
        # while silently handing the mask decoder the wrong skips and the
        # memory attention a stride-32 grid its rotary tables are not built
        # for. Neither error produces a shape error here. See decisions.md
        # D-014 and test_neck.py::TestScalpIdentity.
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

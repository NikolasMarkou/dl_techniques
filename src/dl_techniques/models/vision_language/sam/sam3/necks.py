"""Sam3DualViTDetNeck, the ViTDet-style SimpleFPN neck for SAM 3.

Turns the single feature map from :class:`Sam3ViTDetBackbone` into a
four-scale pyramid by resampling that one map to four resolutions (transposed
convs to upsample, max-pool to downsample, identity at scale 1.0), rather
than reading from a multi-block trunk pyramid. "Dual" means two structurally
identical but independently-weighted copies of the four-branch conv stack
read the same trunk feature: one feeds the detector, one feeds the tracker.
Each branch adds its own fixed 2D sine positional encoding, computed on that
branch's own grid.

The branch convs carry no normalization. The sine encoding omits the
reference's half-pixel center offset (a constant angular shift, largest at
the coarsest level), which is inert today and binding only if released SAM 3
weights are ever loaded.

References:
    - Li et al., 2022. Exploring Plain Vision Transformer Backbones for
      Object Detection.
    - Carion et al., 2020. End-to-End Object Detection with Transformers.
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.embedding.positional_embedding_sine_2d import (
    PositionEmbeddingSine2D,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# module-private helpers
# ---------------------------------------------------------------------

SUPPORTED_SCALES: Tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)


def _build_scale_stack(
        dim: int, d_model: int, scale: float, prefix: str
) -> List[keras.layers.Layer]:
    """Build one scale branch: the resampler, then the shared 1x1 / 3x3 pair.

    :param dim: Trunk channel width feeding this branch.
    :type dim: int
    :param d_model: Common output width of every branch.
    :type d_model: int
    :param scale: Resolution multiplier; one of :data:`SUPPORTED_SCALES`.
    :type scale: float
    :param prefix: Name prefix for every layer in the branch.
    :type prefix: str
    :return: The branch's layers, in application order.
    :rtype: List[keras.layers.Layer]
    :raises ValueError: If ``scale`` is not supported.
    """
    layers: List[keras.layers.Layer] = []
    if scale == 4.0:
        layers.append(keras.layers.Conv2DTranspose(
            dim // 2, kernel_size=2, strides=2, padding="valid",
            name=f"{prefix}_dconv_2x2_0",
        ))
        layers.append(keras.layers.Activation("gelu", name=f"{prefix}_gelu"))
        layers.append(keras.layers.Conv2DTranspose(
            dim // 4, kernel_size=2, strides=2, padding="valid",
            name=f"{prefix}_dconv_2x2_1",
        ))
    elif scale == 2.0:
        layers.append(keras.layers.Conv2DTranspose(
            dim // 2, kernel_size=2, strides=2, padding="valid",
            name=f"{prefix}_dconv_2x2",
        ))
    elif scale == 1.0:
        pass
    elif scale == 0.5:
        layers.append(keras.layers.MaxPooling2D(
            pool_size=2, strides=2, padding="valid",
            name=f"{prefix}_maxpool_2x2",
        ))
    else:
        raise ValueError(
            f"scale_factor={scale} is not supported; supported scales are "
            f"{SUPPORTED_SCALES}"
        )

    # DECISION plan-2026-08-04T044628-4c240b4c/D-095: use_bias=True on both
    # convs, no normalization anywhere in this stack.
    # The SAM 3.0 dual neck has none; adding one would change the embedding scale the whole detector and tracker read. See decisions.md.
    layers.append(keras.layers.Conv2D(
        d_model, kernel_size=1, use_bias=True, name=f"{prefix}_conv_1x1",
    ))
    layers.append(keras.layers.Conv2D(
        d_model, kernel_size=3, padding="same", use_bias=True,
        name=f"{prefix}_conv_3x3",
    ))
    return layers


def _encode_position(
        pe_layer: keras.layers.Layer, feature: Any, d_model: int
) -> Any:
    """Compute the sine positional encoding for one branch output.

    :param pe_layer: The :class:`PositionEmbeddingSine2D` instance.
    :type pe_layer: keras.layers.Layer
    :param feature: Channels-last branch output ``(batch, h, w, d_model)``.
    :type feature: Any
    :param d_model: Expected encoding width.
    :type d_model: int
    :return: Channels-last encoding ``(batch, h, w, d_model)`` in ``feature``'s
        dtype.
    :rtype: Any
    :raises ValueError: If the encoding's width or spatial extent disagrees with
        ``feature`` after the transpose.
    """
    # DECISION plan-2026-08-04T044628-4c240b4c/D-096: transpose and validate
    # the encoding's shape here, not in __init__.
    # PositionEmbeddingSine2D returns channels-first 2*num_pos_feats channels; on a square grid a forgotten transpose is shape-compatible but wrong. See decisions.md.
    pos = ops.transpose(pe_layer(feature), (0, 2, 3, 1))
    expected = tuple(feature.shape[1:])
    got = tuple(pos.shape[1:])
    if got != expected:
        raise ValueError(
            f"positional encoding shape {got} must match the feature's "
            f"{expected} (width {d_model}); a channels-first encoding that was "
            f"not transposed reaches this check"
        )
    return ops.cast(pos, feature.dtype)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam3.necks")
class Sam3DualViTDetNeck(keras.layers.Layer):
    """SAM 3's dual SimpleFPN neck over a single channels-last trunk map.

    **Architecture Overview:**

    .. code-block:: text

                     trunk map (B, g, g, dim)
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
        sam3 conv stack               sam2 conv stack   (independent weights)
              │                            │
        ┌─────┼─────┬─────┐          ┌─────┼─────┬─────┐
        4x   2x    1x   0.5x         4x   2x    1x   0.5x
        │     │     │     │          │     │     │     │
        └──── 1x1 -> 3x3 ────┘       └──── 1x1 -> 3x3 ────┘
              │                            │
        + per-scale sine PE          + per-scale sine PE

    Known divergence from the reference: this neck's shared
    :class:`PositionEmbeddingSine2D` normalizes coordinates to pixel centers,
    ``(k - 0.5) / H * 2*pi``, while the reference normalizes to pixel edges,
    ``k / H * 2*pi``. The gap is a constant ``pi / H`` angular shift, accepted
    because the layer is shared with SAM 2 and other consumers. Measured at
    the four shipped grids (``H = 288/144/72/36``): max absolute difference
    ``0.010908 / 0.021815 / 0.043619 / 0.087156``, largest at the coarsest
    level. ``sam3_pos`` feeds the detector's cross-attention key embedding
    directly, so this is a real input reparametrization, binding on any
    future load of released SAM 3 weights.

    :param dim: Trunk channel width (the neck's input width).
    :type dim: int
    :param d_model: Common output width of every scale of every branch.
        Must be a positive multiple of **4**, not merely even. The sine
        encoding receives ``num_pos_feats = d_model // 2`` features per axis,
        and :class:`PositionEmbeddingSine2D` splits that width again between
        its sine and cosine halves -- so ``d_model // 2`` must ITSELF be even.
        ``d_model = 10`` is even, yields ``num_pos_feats = 5``, and builds a
        position encoder that can never run a forward pass.
    :type d_model: int
    :param scale_factors: Resolution multipliers, in output order (finest
        first at the settled configuration).
    :type scale_factors: Sequence[float]
    :param add_sam2_neck: Whether to build the second, independently-weighted
        conv stack. ``True`` at the settled configuration.
    :type add_sam2_neck: bool
    :param pe_temperature: Temperature of the sine positional encoding.
    :type pe_temperature: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``dim`` is not divisible by 4, if ``d_model`` is not
        a positive multiple of 4, if ``scale_factors`` is empty, or if it names
        an unsupported scale.

    Example:
        >>> import numpy as np
        >>> neck = Sam3DualViTDetNeck(dim=16, d_model=8)
        >>> out = neck(np.zeros((1, 8, 8, 16), dtype="float32"))
        >>> [f.shape[1] for f in out["sam3_features"]]
        [32, 16, 8, 4]
    """

    FEATURE_KEYS: Tuple[str, ...] = (
        "sam3_features", "sam3_pos", "sam2_features", "sam2_pos",
    )

    def __init__(
            self,
            dim: int = 1024,
            d_model: int = 256,
            scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
            add_sam2_neck: bool = True,
            pe_temperature: float = 10000.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim <= 0 or dim % 4 != 0:
            raise ValueError(
                f"dim ({dim}) must be a positive multiple of 4 -- the 4.0 scale "
                f"branch narrows it to dim // 4"
            )
        # DECISION plan-2026-08-28T181715-3870472c/D-006: constraint is
        # d_model % 4, not d_model % 2.
        # d_model // 2 becomes PositionEmbeddingSine2D's num_pos_feats, which must itself be even; d_model=10 passed the old check and built an encoder that could never forward. See decisions.md.
        if d_model <= 0 or d_model % 4 != 0:
            raise ValueError(
                f"d_model ({d_model}) must be a positive multiple of 4, not "
                f"merely even: the sine encoding receives num_pos_feats = "
                f"d_model // 2 = {d_model // 2} features per axis, and that "
                f"value must ITSELF be even because PositionEmbeddingSine2D "
                f"splits it between its sine and cosine halves. Use "
                f"d_model = {((d_model + 3) // 4) * 4 if d_model > 0 else 4}."
            )
        scale_factors = tuple(float(s) for s in scale_factors)
        if not scale_factors:
            raise ValueError("scale_factors must name at least one scale")
        for scale in scale_factors:
            if scale not in SUPPORTED_SCALES:
                raise ValueError(
                    f"scale_factor={scale} is not supported; supported scales "
                    f"are {SUPPORTED_SCALES}"
                )

        # Store ALL configuration parameters.
        self.dim = int(dim)
        self.d_model = int(d_model)
        self.scale_factors = scale_factors
        self.add_sam2_neck = bool(add_sam2_neck)
        self.pe_temperature = float(pe_temperature)

        # Sub-layers -- created UNCONDITIONALLY, built explicitly in build().
        #
        # DECISION plan-2026-08-04T044628-4c240b4c/D-134: keep this layer's
        # pixel-center normalization; do not edit it to match the reference's
        # pixel-edge convention.
        # It is shared with SAM 2, which already depends on the current formula (D-042). See decisions.md.
        self.position_encoding = PositionEmbeddingSine2D(
            num_pos_feats=self.d_model // 2,
            temperature=self.pe_temperature,
            name="position_encoding",
        )
        # DECISION plan-2026-08-04T044628-4c240b4c/D-097: build sam2_convs
        # and sam3_convs with two separate calls, never a shared stack.
        # A shared-stack port matches the reference at init and diverges only after training, so a value-level test would miss it; guard by weight-independence instead. Corrected by D-134: the two stacks are NOT bit-identical at init here (unlike the reference's deepcopy). See decisions.md.
        #
        # DECISION plan-2026-08-04T044628-4c240b4c/D-098: store both stacks
        # flat, never as a nested per-scale list.
        # A nested List[List[Layer]] tracks params/weights correctly but silently restores fresh kernels on a .keras round trip (measured delta 5.95 vs 0.0 flat). See decisions.md.
        sam3_branches = [
            _build_scale_stack(self.dim, self.d_model, scale, f"sam3_{index}")
            for index, scale in enumerate(self.scale_factors)
        ]
        self._branch_sizes: Tuple[int, ...] = tuple(
            len(branch) for branch in sam3_branches
        )
        self.sam3_convs: List[keras.layers.Layer] = [
            layer for branch in sam3_branches for layer in branch
        ]
        self.sam2_convs: List[keras.layers.Layer] = [
            layer
            for index, scale in enumerate(self.scale_factors)
            for layer in _build_scale_stack(
                self.dim, self.d_model, scale, f"sam2_{index}"
            )
        ] if self.add_sam2_neck else []

    def branches(
            self, stack: List[keras.layers.Layer]
    ) -> List[List[keras.layers.Layer]]:
        """Re-slice a flat stack into its per-scale branches.

        The flat storage is a serialization requirement (D-098), so this is the
        accessor every consumer -- including the tests -- uses to reach one
        scale's layers.

        :param stack: :attr:`sam3_convs` or :attr:`sam2_convs`.
        :type stack: List[keras.layers.Layer]
        :return: One list of layers per scale, empty when ``stack`` is empty.
        :rtype: List[List[keras.layers.Layer]]
        """
        branches: List[List[keras.layers.Layer]] = []
        offset = 0
        for size in self._branch_sizes if stack else ():
            branches.append(stack[offset:offset + size])
            offset += size
        return branches

    # -----------------------------------------------------------------
    # shape arithmetic
    # -----------------------------------------------------------------

    def _scaled_extent(
            self, extent: Optional[int], scale: float
    ) -> Optional[int]:
        """Apply one scale factor to one spatial extent.

        :param extent: Trunk grid extent along an axis, or ``None``.
        :type extent: Optional[int]
        :param scale: Resolution multiplier.
        :type scale: float
        :return: The branch's extent along that axis.
        :rtype: Optional[int]
        """
        if extent is None:
            return None
        return int(extent * scale) if scale >= 1.0 else int(extent) // 2

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the sine encoding and every layer of both conv stacks.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not a rank-4 channels-last tensor of
            width ``dim``, or if a non-unit scale cannot be applied exactly to
            its spatial extent.
        """
        if self.built:
            return
        input_shape = tuple(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"Sam3DualViTDetNeck expects a rank-4 channels-last trunk map, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] != self.dim:
            raise ValueError(
                f"trunk map width {input_shape[-1]} must equal the configured "
                f"dim ({self.dim})"
            )
        for axis, extent in enumerate(input_shape[1:3]):
            if extent is not None and extent % 2 != 0 and 0.5 in self.scale_factors:
                raise ValueError(
                    f"trunk grid axis {axis} has odd extent {extent}, which the "
                    f"0.5 scale's 2x2 max-pool cannot halve exactly"
                )

        for stack in (self.sam3_convs, self.sam2_convs):
            for branch in self.branches(stack):
                shape = tuple(input_shape)
                for layer in branch:
                    layer.build(shape)
                    shape = tuple(layer.compute_output_shape(shape))
        for scale in self.scale_factors:
            self.position_encoding.build((
                input_shape[0],
                self._scaled_extent(input_shape[1], scale),
                self._scaled_extent(input_shape[2], scale),
                self.d_model,
            ))
        logger.debug(
            "Sam3DualViTDetNeck built: dim=%d d_model=%d scales=%s "
            "dual=%s ladder=%s",
            self.dim, self.d_model, self.scale_factors, self.add_sam2_neck,
            [self._scaled_extent(input_shape[1], s)
             for s in self.scale_factors],
        )
        super().build(input_shape)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def _run_stack(
            self, stack: List[keras.layers.Layer], inputs: Any
    ) -> Tuple[List[Any], List[Any]]:
        """Run one whole conv stack over the trunk map.

        Every layer in a branch is deterministic -- convolutions, one activation
        and one max-pool -- so none of them takes a ``training`` flag.

        :param stack: A FLAT stack; re-sliced per scale by :meth:`branches`.
        :type stack: List[keras.layers.Layer]
        :param inputs: The trunk map, channels-last.
        :type inputs: Any
        :return: ``(features, positional_encodings)``, one entry per scale.
        :rtype: Tuple[List[Any], List[Any]]
        """
        features: List[Any] = []
        encodings: List[Any] = []
        for branch in self.branches(stack):
            x = inputs
            for layer in branch:
                x = layer(x)
            features.append(x)
            # Per-scale, per-branch: computed on THIS output's own grid.
            encodings.append(
                _encode_position(self.position_encoding, x, self.d_model)
            )
        return features, encodings

    def call(
            self, inputs: Any, training: Optional[bool] = None
    ) -> Dict[str, List[Any]]:
        """Resample the trunk map to every scale, twice, with per-scale encodings.

        :param inputs: The trunk's single feature map,
            ``(batch, height, width, dim)``.
        :type inputs: Any
        :param training: Unused; every sub-layer here is deterministic. Present
            for the Keras call contract.
        :type training: Optional[bool]
        :return: ``sam3_features`` / ``sam3_pos`` and, when ``add_sam2_neck``,
            ``sam2_features`` / ``sam2_pos``; the sam2 lists are EMPTY when the
            second stack is disabled, so the key set never varies.
        :rtype: Dict[str, List[Any]]
        """
        sam3_features, sam3_pos = self._run_stack(self.sam3_convs, inputs)
        sam2_features, sam2_pos = self._run_stack(self.sam2_convs, inputs)
        return {
            "sam3_features": sam3_features,
            "sam3_pos": sam3_pos,
            "sam2_features": sam2_features,
            "sam2_pos": sam2_pos,
        }

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, List[Tuple[Optional[int], ...]]]:
        """Return the per-key output shapes, derived from stored config only.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: One shape list per output key.
        :rtype: Dict[str, List[Tuple[Optional[int], ...]]]
        """
        input_shape = tuple(input_shape)
        shapes = [
            (input_shape[0],
             self._scaled_extent(input_shape[1], scale),
             self._scaled_extent(input_shape[2], scale),
             self.d_model)
            for scale in self.scale_factors
        ]
        dual = list(shapes) if self.add_sam2_neck else []
        return {
            "sam3_features": list(shapes),
            "sam3_pos": list(shapes),
            "sam2_features": dual,
            "sam2_pos": list(dual),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "d_model": self.d_model,
            "scale_factors": self.scale_factors,
            "add_sam2_neck": self.add_sam2_neck,
            "pe_temperature": self.pe_temperature,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Sam3DualViTDetNeck":
        """Rebuild from a config, tolerating a pre-``% 4`` stored ``d_model``.

        Guide section 6.3 migration path. The constructor's multiple-of-4 rule
        is a NEW rejection of a value the old ``% 2`` check accepted, so an
        archive written before it must still load. An odd ``d_model // 2`` is
        rounded UP here (``d_model`` to the next multiple of 4) with a warning,
        never raised on. Rounding up rather than down preserves capacity, and
        the width change cannot break anything that previously worked: such a
        neck's position encoder could never complete a forward pass, so no
        model carrying one was ever trainable or servable.

        :param config: Serialized configuration.
        :type config: Dict[str, Any]
        :return: The reconstructed neck.
        :rtype: Sam3DualViTDetNeck
        """
        config = dict(config)
        d_model = config.get("d_model")
        if isinstance(d_model, int) and d_model > 0 and d_model % 4 != 0:
            substitute = ((d_model + 3) // 4) * 4
            logger.warning(
                "Sam3DualViTDetNeck config carries d_model=%d, whose sine "
                "width num_pos_feats=%d is odd; this archive predates the "
                "multiple-of-4 requirement and its position encoder could "
                "never run a forward pass. Substituting d_model=%d "
                "(num_pos_feats=%d). The neck output width changes from %d to "
                "%d, so stored weights for this layer will not match.",
                d_model, d_model // 2, substitute, substitute // 2,
                d_model, substitute,
            )
            config["d_model"] = substitute
        return cls(**config)

# ---------------------------------------------------------------------

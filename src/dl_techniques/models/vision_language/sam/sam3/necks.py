"""
SAM 3 Dual SimpleFPN Neck: one trunk map, four scales, two weight sets.
=======================================================================

:class:`Sam3DualViTDetNeck` is the ViTDet-style SimpleFPN that turns the ONE
feature map emitted by :class:`Sam3ViTDetBackbone` into the multi-scale pyramid
the detector and the tracker consume.

Based on:
---------
- Li, Y., Mao, H., Girshick, R., & He, K. (2022). ViTDet / SimpleFPN.
- Carion, N. et al. (2020). DETR -- the sine positional encoding reused here.

Key Features:
------------
- ONE trunk map resampled to four resolutions; no multi-block trunk pyramid.
- "Dual" = two structurally identical, INDEPENDENTLY-WEIGHTED copies of the
  four-branch conv stack reading the SAME trunk feature, one feeding the SAM 3
  detector and one the SAM-2-style tracker. One backbone, two neck weight sets,
  never two backbones.
- A fixed 2D sine positional encoding per branch, on that branch's OWN grid.

Architecture Overview:
---------------------
1. **Resample** the single ``(batch, grid, grid, dim)`` trunk map: scale ``4.0``
   = ``ConvT(dim -> dim/2, k=2, s=2) -> GELU -> ConvT(dim/2 -> dim/4, k=2,
   s=2)``; ``2.0`` = ``ConvT(dim -> dim/2, k=2, s=2)``; ``1.0`` = identity;
   ``0.5`` = ``MaxPool(k=2, s=2)``.
2. On EVERY branch: ``Conv(1x1 -> d_model, bias) -> Conv(3x3, pad=1 ->
   d_model, bias)``.
3. Add that branch's own sine encoding. A ``72x72`` trunk gives ``288 / 144 /
   72 / 36``.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM3.necks import Sam3DualViTDetNeck
neck = Sam3DualViTDetNeck(dim=1024, d_model=256,
                          scale_factors=(4.0, 2.0, 1.0, 0.5))
```

Measured caveats:
----------------
- The branch convs carry **no normalization of any kind**. The SAM 3.1 three-way
  neck adds an optional norm there; this is the SAM 3.0 dual neck and has none.
- The sine encoding omits the reference's half-pixel centre offset, a constant
  angular shift of ``pi / H`` MEASURED at ``0.010908 / 0.021815 / 0.043630 /
  0.087266`` radians for ``H = 288 / 144 / 72 / 36`` -- largest at the coarsest
  level, and BINDING on any future transfer of released SAM 3 weights (D-134,
  carrying D-042 forward).
- Per-scale encodings are not shared: the normalized coordinate pitch differs at
  every resolution, so one encoding resampled across scales is a value defect
  with no shape symptom.
- The transpose check is on the OUTPUT, not on a constructor argument: the
  reused sine layer returns channels-FIRST and emits ``2 * num_pos_feats``
  channels, and on a square grid whose side happens to equal ``d_model`` a
  forgotten transpose broadcasts silently instead of raising.
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

    # DECISION plan-2026-08-04T044628-4c240b4c/D-095
    # `use_bias=True` on BOTH convs and NO normalization between or after them.
    # Do NOT add a norm here "to match the FPN elsewhere in the repo": the SAM
    # 3.0 dual neck has none, and the SAM 3.1 tri-neck's optional `neck_norm`
    # also flips these biases off when it is enabled. Adding a norm here would
    # change the embedding scale the whole detector and tracker read, which is
    # the exact class of silent divergence iteration 2 measured for SAM 2's
    # unnormalized neck. See decisions.md D-095.
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
    # DECISION plan-2026-08-04T044628-4c240b4c/D-096
    # `PositionEmbeddingSine2D` holds TWO conventions that both differ from this
    # module's: it emits `2 * num_pos_feats` channels (so `d_model // 2` is the
    # argument that yields a `d_model`-wide encoding), and it returns
    # channels-FIRST `(B, C, H, W)`. Hence the explicit transpose and cast here,
    # and hence the check below is on the RESULT rather than on the constructor
    # argument: on a square grid whose side equals `d_model` a forgotten
    # transpose is shape-compatible and adds a silently wrong tensor. Do NOT
    # move this validation into `__init__` and do NOT delete the transpose.
    # See decisions.md D-096.
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


@keras.saving.register_keras_serializable()
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

    **Known divergence from the reference -- the sine positional encoding.**
    This neck reuses the repo-wide :class:`PositionEmbeddingSine2D`, which
    normalizes coordinates to pixel CENTRES, ``(k - 0.5) / H * 2*pi``, while the
    reference normalizes to pixel EDGES, ``k / H * 2*pi``
    (``sam3/model/position_encoding.py:102-116`` at the pinned SHA -- no
    offset). The divergence is a constant angular shift of ``pi / H`` and it is
    ACCEPTED, not fixed: the layer is shared with SAM 2 and other consumers and
    changing it here would move them. MEASURED in float64 at the four SHIPPED
    grids, with ``num_pos_feats = d_model // 2 = 128`` and an encoding amplitude
    of 1: max absolute difference **0.010908 / 0.021815 / 0.043619 / 0.087156**
    at ``H = 288 / 144 / 72 / 36`` -- i.e. exactly ``pi / H``, largest on the
    COARSEST level. ``sam3_pos`` is LIVE on the detector's main path (it becomes
    ``memory_pos``, the image cross-attention key embedding), so this is a real
    input reparametrization, not a spare output. It is the same deviation
    iteration 1 accepted for SAM 2 (D-042); D-134 carries it forward and states
    the consequence for any future checkpoint load. The magnitude is PINNED by
    ``tests/test_models/test_sam3/test_necks.py::TestReferencePeDivergence``, so
    a change in it is loud rather than silent.

    :param dim: Trunk channel width (the neck's input width).
    :type dim: int
    :param d_model: Common output width of every scale of every branch.
        Must be even, because the sine encoding is built from ``d_model // 2``
        features per axis.
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
        positive and even, if ``scale_factors`` is empty, or if it names an
        unsupported scale.

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
        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError(
                f"d_model ({d_model}) must be positive and even; the sine "
                f"encoding uses d_model // 2 features per axis"
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
        # DECISION plan-2026-08-04T044628-4c240b4c/D-134
        # This shared layer normalizes to pixel CENTRES, `(k - 0.5) / H`, where
        # the reference normalizes to pixel EDGES, `k / H`. Do NOT "fix" it by
        # editing `layers/embedding/positional_embedding_sine_2d.py`: that layer
        # is byte-frozen for this plan and SAM 2 already depends on its current
        # formula (D-042). Do NOT fork or wrap it here either -- a wrapper would
        # hide a documented constant behind an indirection. The divergence is a
        # constant `pi / H` angular shift, MEASURED at 0.010908 / 0.021815 /
        # 0.043619 / 0.087156 for the four shipped grids H = 288/144/72/36, and
        # it is ACCEPTED for a fresh-init port and BINDING on any future load of
        # released SAM 3 weights. See decisions.md D-134 (and D-042).
        self.position_encoding = PositionEmbeddingSine2D(
            num_pos_feats=self.d_model // 2,
            temperature=self.pe_temperature,
            name="position_encoding",
        )
        # DECISION plan-2026-08-04T044628-4c240b4c/D-097
        # These two stacks are built by two SEPARATE calls, so every weight in
        # `sam2_convs` is independent of its `sam3_convs` twin. Do NOT
        # "de-duplicate" this by reusing one stack for both outputs: a
        # shared-stack port has the same shapes AND the same forward values as
        # the reference on a fresh model and diverges only after training, so
        # there is no value-level symptom a fresh-model test could see. The
        # guards are therefore a trainable-weight COUNT and a weight-
        # independence probe, never an output comparison.
        #
        # CORRECTED (D-134, was D-097): the two stacks of THIS port are NOT
        # numerically identical at initialization. The reference clones with
        # `sam2_convs = deepcopy(self.convs)`, so ITS two stacks are bit-
        # identical at step 0; this port calls `_build_scale_stack` twice, so
        # each draws its own weights. Measured by the existing
        # `test_at_initialization_the_two_necks_already_differ` (min delta
        # strictly > 0). Nothing reachable depends on it -- both shipped
        # variants set `add_sam2_neck=False` -- but do not repeat the claim.
        # See decisions.md D-097 and D-134.
        #
        # DECISION plan-2026-08-04T044628-4c240b4c/D-098
        # Both stacks are stored FLAT -- one list of layers per neck, with the
        # per-scale branch boundaries kept separately in `_branch_sizes`. The
        # obvious spelling is a list of per-scale lists, and it is WRONG here:
        # a nested `List[List[Layer]]` attribute is tracked well enough that
        # `count_params()`, `weights`, `trainable_weights` and the forward pass
        # are all correct, but the `.keras` save/load round trip SILENTLY
        # RESTORES FRESHLY-INITIALIZED KERNELS. MEASURED on this layer and
        # reproduced on a 12-line stand-alone `Dense` layer: nested gives a
        # round-trip output delta of 5.95 with matching weight COUNT and
        # matching weight PATHS, flat gives exactly 0.0. Do NOT re-nest these
        # lists for readability -- there is no exception and no shape symptom,
        # only wrong weights. Pinned by
        # `test_necks.py::TestSerialization::test_full_keras_roundtrip_preserves_outputs`
        # and by the framework-level regression test beside it. See D-098.
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

# ---------------------------------------------------------------------

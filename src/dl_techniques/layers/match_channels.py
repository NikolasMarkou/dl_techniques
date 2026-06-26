"""
Parameter-free channel-count matching layer.

This module implements `MatchChannels`, a WEIGHTLESS Keras 3 layer that coerces an
input tensor's channel count (last axis, NHWC) to a fixed `target_channels` using
one of three parameter-free operations selected at build time from the static
channel delta:

- **Zero-pad** (when `in_channels < target_channels`): append `target - in` zero
  channels along the last axis.
- **Slice** (when `in_channels > target_channels`): keep `target` channels along
  the last axis. The `slice_side` keyword selects which end is kept: `'head'`
  (default, current behavior) keeps the LEADING channels (`inputs[..., :target]`);
  `'tail'` keeps the TRAILING channels (`inputs[..., -target:]`).
- **Passthrough** (when `in_channels == target_channels`): return the input
  unchanged.

Motivation
----------

The bias-free ConvUNeXt denoiser matches channel counts between levels with 1x1
convolutions (`Conv2D(kernel_size=1, use_bias=False)`). This layer is the
parameter-free replacement used by the `--zero-pad-channels` variant: zero-pad on
channel INCREASE (encoder / bottleneck), slice on channel DECREASE (decoder).

Bias-free / homogeneity property
--------------------------------

Both implemented operations are LINEAR and degree-1 (positively) HOMOGENEOUS:

- Zero-padding is concatenation with a constant-zero tensor:
  `pad(alpha * x) = [alpha * x, 0] = alpha * [x, 0] = alpha * pad(x)`.
- Slicing is a coordinate projection:
  `slice(alpha * x) = alpha * slice(x)`.

Neither adds learnable parameters and neither introduces an additive bias /
offset. Therefore the layer preserves the bias-free, scale-homogeneous invariant
of the denoiser (`f(alpha * x) = alpha * f(x)`): inserting it leaves whatever
homogeneity the surrounding network has intact, and it is invisible to a
`use_bias` / `LayerNormalization.center` / GRN-`beta` bias-free audit because it
holds no weights at all.

Precedent: `layers/pixel_shuffle.py` uses `keras.ops.pad` for channel padding;
this layer follows the same backend-agnostic `keras.ops` style.
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MatchChannels(keras.layers.Layer):
    """Match an input tensor's channel count to a target, parameter-free.

    The layer matches the last-axis (NHWC channel) dimension of the input to a
    fixed ``target_channels`` using a parameter-free operation chosen at build
    time from the static channel delta: zero-pad if the input has FEWER channels,
    slice if it has MORE (keeping either end via ``slice_side``), passthrough if
    equal.

    It holds NO weights and adds NO bias / offset. Both zero-padding and slicing
    are linear and degree-1 homogeneous (``f(alpha * x) = alpha * f(x)``), so the
    layer preserves the bias-free, scale-homogeneous invariant of a denoiser into
    which it is inserted. A tail slice is equally a coordinate projection, so it is
    just as weightless and degree-1 homogeneous as a head slice.

    Args:
        target_channels: Positive integer. The desired number of output channels
            (size of the last axis). Must be ``> 0``.
        slice_side: Which end to keep when the input has MORE channels than
            ``target_channels``. ``'head'`` (default) keeps the LEADING channels
            (``inputs[..., :target]``, the original behavior); ``'tail'`` keeps the
            TRAILING channels (``inputs[..., -target:]``). Ignored on the zero-pad
            and passthrough branches. Must be ``'head'`` or ``'tail'``.
        **kwargs: Standard ``keras.layers.Layer`` keyword arguments (e.g.
            ``name``, ``dtype``).

    Input shape:
        Rank-4 tensor ``(batch, height, width, channels)`` (NHWC). Batch and
        spatial dimensions may be dynamic; the channel dimension must be known at
        build time.

    Output shape:
        ``(batch, height, width, target_channels)`` — identical to the input
        except the last axis is exactly ``target_channels``.

    Raises:
        ValueError: If ``target_channels <= 0`` or ``slice_side`` is not one of
            ``'head'`` / ``'tail'``.

    Example:
        >>> import numpy as np, keras
        >>> from dl_techniques.layers.match_channels import MatchChannels
        >>> x = np.random.randn(2, 8, 8, 4).astype("float32")
        >>> MatchChannels(8)(x).shape  # zero-pad 4 -> 8
        (2, 8, 8, 8)
        >>> MatchChannels(2)(x).shape  # slice 4 -> 2
        (2, 8, 8, 2)
    """

    def __init__(
        self,
        target_channels: int,
        slice_side: str = "head",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if target_channels <= 0:
            raise ValueError(
                f"target_channels must be a positive integer, got {target_channels}"
            )
        if slice_side not in ("head", "tail"):
            raise ValueError(
                f"slice_side must be 'head' or 'tail', got {slice_side!r}"
            )

        self.target_channels = int(target_channels)
        self.slice_side = slice_side
        # Recorded in build() from the concrete input shape; the channel delta is
        # static (build-time), so no per-call shape inference of the delta is needed.
        self._in_channels: Optional[int] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Record the static input channel count. Creates NO weights.

        Args:
            input_shape: Shape tuple of the input tensor; the last entry is the
                input channel count and must be known (not ``None``).
        """
        self._in_channels = int(input_shape[-1])
        # Weightless by design: the channel delta is a static integer, so the
        # pad / slice / passthrough decision needs no trainable state.
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Match channels via zero-pad, slice, or passthrough.

        Args:
            inputs: Rank-4 NHWC input tensor.

        Returns:
            Tensor with the last axis resized to ``target_channels``.
        """
        if self._in_channels == self.target_channels:
            # Passthrough: channels already match (no-op).
            return inputs

        if self._in_channels < self.target_channels:
            # Zero-pad the channel axis. delta is a static int; pad amounts are
            # static while batch/spatial dims stay dynamic (keras.ops.pad handles
            # dynamic leading dims). Concatenation-with-zeros is degree-1
            # homogeneous and bias-free.
            delta = self.target_channels - self._in_channels
            return ops.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, delta]])

        # in_channels > target_channels: slice. A coordinate projection is degree-1
        # homogeneous and bias-free regardless of which end is kept.
        # DECISION plan_2026-06-26_0ec1a304/D-002: the 'tail' branch is a real,
        # registered, serialization-safe primitive (not a Lambda closure) because
        # the bfconvunext --extra-zero-output-channels output must keep the LAST
        # `output_channels` channels. Do NOT replace this with a Lambda slice (does
        # not round-trip across processes) nor add a separate SliceLastChannels
        # layer (duplicates channel-slice logic MatchChannels already owns).
        if self.slice_side == "tail":
            return inputs[..., -self.target_channels :]
        return inputs[..., : self.target_channels]

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Replace the last dimension with ``target_channels``.

        Args:
            input_shape: Input shape tuple.

        Returns:
            Output shape tuple identical to the input except the last axis is
            ``target_channels``.
        """
        return (*input_shape[:-1], self.target_channels)

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        Returns:
            Config dict including ``target_channels`` and ``slice_side``.
            ``_in_channels`` is re-derived in ``build`` from the input shape, so it
            is not stored.
        """
        config = super().get_config()
        config.update(
            {
                "target_channels": self.target_channels,
                "slice_side": self.slice_side,
            }
        )
        return config

# ---------------------------------------------------------------------

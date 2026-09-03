"""
``MatchChannels`` coerces an input tensor's last-axis (NHWC) channel count to
a fixed ``target_channels``, using whichever of three parameter-free
operations the static channel delta calls for: zero-pad when the input has
fewer channels, slice when it has more (keeping either the leading or
trailing channels, via ``slice_side``), or a passthrough when they already
match.

The bias-free ConvUNeXt denoiser normally matches channel counts between
levels with a 1x1 convolution; this layer is the parameter-free replacement
used by its ``--zero-pad-channels`` variant. Both zero-padding and slicing
are linear and degree-1 homogeneous (``f(alpha * x) = alpha * f(x)``), and
neither adds a weight or a bias, so inserting this layer preserves whatever
bias-free, scale-homogeneous invariant the surrounding network has.
"""

import keras
from typing import Any, Dict, Optional, Tuple
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.match_channels")
class MatchChannels(keras.layers.Layer):
    """Match an input tensor's channel count to a target, parameter-free.

    The layer matches the last-axis (NHWC channel) dimension of the input to a
    fixed ``target_channels`` using a parameter-free operation chosen at build
    time from the static channel delta: zero-pad if the input has fewer channels,
    slice if it has more (keeping either end via ``slice_side``), passthrough if
    equal.

    It holds no weights and adds no bias or offset. Both zero-padding and slicing
    are linear and degree-1 homogeneous (``f(alpha * x) = alpha * f(x)``), so the
    layer preserves the bias-free, scale-homogeneous invariant of a denoiser into
    which it is inserted. A tail slice is equally a coordinate projection, so it is
    just as weightless and degree-1 homogeneous as a head slice.

    Architecture:

    .. code-block:: text

        ┌───────────────────────────────────────┐
        │  Input [B, H, W, C_in]                │
        └──────────────────┬────────────────────┘
                           ▼
                ┌───────────────────────────┐
                │  Compare C_in vs target   │
                │  (static, at build time)  │
                └────────────┬──────────────┘
                             │
              ┌──────────────┼───────────────┐
              ▼              ▼               ▼
        C_in < target     C_in == target   C_in > target
        ┌─────────────┐   ┌─────────────┐  ┌───────────────────────┐
        │  Zero-pad   │   │ Passthrough │  │  Slice                │
        │  channels   │   │  (no-op)    │  │  slice_side='head':   │
        │  [..., 0:Δ] │   │             │  │   inputs[...,:target] │
        │  appended   │   │             │  │  slice_side='tail':   │
        │             │   │             │  │   inputs[...,-target:]│
        └──────┬──────┘   └──────┬──────┘  └──────────┬────────────┘
               │                 │                    │
               └─────────────────┼────────────────────┘
                                 ▼
                ┌───────────────────────────────┐
                │  Output [B, H, W, target_C]   │
                └───────────────────────────────┘

    :param target_channels: Desired number of output channels (size of the
        last axis). Must be positive.
    :type target_channels: int
    :param slice_side: Which end to keep when the input has more channels than
        ``target_channels``. ``'head'`` (default) keeps the leading channels
        (``inputs[..., :target]``); ``'tail'`` keeps the trailing channels
        (``inputs[..., -target:]``). Ignored on the zero-pad and passthrough
        branches.
    :type slice_side: str
    :param kwargs: Additional keyword arguments for the Layer base class.
    :raises ValueError: If ``target_channels <= 0`` or ``slice_side`` is not
        one of ``'head'`` / ``'tail'``.

    Input shape:
        Rank-4 tensor ``(batch, height, width, channels)`` (NHWC). Batch and
        spatial dimensions may be dynamic; the channel dimension must be known at
        build time.

    Output shape:
        ``(batch, height, width, target_channels)`` — identical to the input
        except the last axis is exactly ``target_channels``.

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
        # Recorded in build() from the concrete input shape.
        self._in_channels: Optional[int] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Record the static input channel count. Creates no weights.

        :param input_shape: Shape tuple of the input tensor; the last entry
            is the input channel count and must be known (not ``None``).
        """
        self._in_channels = int(input_shape[-1])
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Match channels via zero-pad, slice, or passthrough.

        :param inputs: Rank-4 NHWC input tensor.
        :return: Tensor with the last axis resized to ``target_channels``.
        """
        if self._in_channels == self.target_channels:
            return inputs

        if self._in_channels < self.target_channels:
            delta = self.target_channels - self._in_channels
            return keras.ops.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, delta]])

        # DECISION plan_2026-06-26_0ec1a304/D-002: 'tail' stays a real primitive,
        # not a Lambda -- a Lambda slice does not round-trip .keras across processes. See decisions.md.
        if self.slice_side == "tail":
            return inputs[..., -self.target_channels :]
        return inputs[..., : self.target_channels]

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Replace the last dimension with ``target_channels``.

        :param input_shape: Input shape tuple.
        :return: Output shape tuple identical to the input except the last
            axis is ``target_channels``.
        """
        return (*input_shape[:-1], self.target_channels)

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        :return: Config dict including ``target_channels`` and ``slice_side``.
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

"""
Pool features globally by summing over spatial dimensions.

Performs global sum pooling on spatial feature maps of arbitrary rank, reducing
an input tensor to a per-channel descriptor by summing over the selected spatial
axes: y_c = sum_{s} x_{s,c}. Unlike average pooling (which normalizes by area)
or max pooling (which identifies peak response), sum pooling measures the total
magnitude of feature activation, making it suited for tasks where total feature
quantity matters such as object counting and density estimation.

Assumes the Keras default "channels_last" data format, i.e. inputs of shape
(batch, *spatial, channels). The spatial rank is inferred from the input shape
at build time.

References:
    - Lempitsky, V. and Zisserman, A. "Learning To Count Objects in Images".
      https://www.robots.ox.ac.uk/~vgg/publications/2010/Lempitsky10/
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, Union, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.global_sum_pool")
class GlobalSumPooling(keras.layers.Layer):
    """
    Global sum pooling operation over configurable spatial axes.

    Sums over the selected spatial axes of a (batch, *spatial, channels) tensor.
    By default every spatial axis is summed, reducing the input to a
    (batch, channels) channel descriptor: y_c = sum_s x_{s,c}. Preserves the
    total activation magnitude rather than averaging or taking the maximum,
    making it useful for object counting and density estimation where the
    integral of a learned density map corresponds to a count.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input [batch, *spatial, C]              │
        │  rank 3: [batch, W, C]                   │
        │  rank 4: [batch, H, W, C]                │
        │  rank 5: [batch, D, H, W, C]             │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Sum over selected spatial axes          │
        │  y_c = Σ_s x_{s,c}                       │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Output [batch, *kept_spatial, C]        │
        │  (summed axes dropped, or 1 if keepdims) │
        └──────────────────────────────────────────┘

    :param axes: Spatial axis or axes to sum over. Negative values are counted
        from the end, so -2 is the last spatial axis. The batch axis (0) and the
        channel axis (-1) cannot be summed. If None, all spatial axes are
        summed. Defaults to None.
    :type axes: Optional[Union[int, Sequence[int]]]
    :param keepdims: Whether to keep the summed dimensions as size 1.
        Defaults to False.
    :type keepdims: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            axes: Optional[Union[int, Sequence[int]]] = None,
            keepdims: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(axes, int):
            axes = (axes,)
        elif axes is not None:
            axes = tuple(int(axis) for axis in axes)
            if not axes:
                raise ValueError("axes must not be empty")
            if len(set(axes)) != len(axes):
                raise ValueError(f"axes must not contain duplicates, got {axes}")

        # Configuration as provided by the user, kept verbatim for serialization
        self.axes = axes
        self.keepdims = keepdims

        # Positive, sorted axes resolved against the input rank at build time
        self._sum_axes: Optional[Tuple[int, ...]] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Infer the spatial rank and resolve the axes to sum over.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self._sum_axes = self._resolve_axes(len(input_shape))
        super().build(input_shape)

    def _resolve_axes(self, input_rank: int) -> Tuple[int, ...]:
        """Normalize the configured axes against a concrete input rank.

        :param input_rank: Rank of the input tensor, including batch and channels.
        :type input_rank: int
        :return: Sorted tuple of positive spatial axis indices.
        :rtype: Tuple[int, ...]
        """
        spatial_rank = input_rank - 2

        if spatial_rank < 1:
            raise ValueError(
                f"Expected an input of rank >= 3 with shape "
                f"(batch, *spatial, channels), got rank {input_rank}"
            )

        # Default: every spatial axis
        if self.axes is None:
            return tuple(range(1, spatial_rank + 1))

        resolved = []
        for axis in self.axes:
            positive_axis = axis + input_rank if axis < 0 else axis

            if not 1 <= positive_axis <= spatial_rank:
                raise ValueError(
                    f"axis {axis} is not a spatial axis of a rank-{input_rank} "
                    f"input; valid spatial axes are 1..{spatial_rank} "
                    f"(or -2..{-input_rank + 1})"
                )

            resolved.append(positive_axis)

        if len(set(resolved)) != len(resolved):
            raise ValueError(
                f"axes {self.axes} resolve to duplicate axes {tuple(resolved)} "
                f"for a rank-{input_rank} input"
            )

        return tuple(sorted(resolved))

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation.

        :param inputs: Input tensor of shape (batch, *spatial, channels).
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode (unused).
        :type training: Optional[bool]
        :return: Tensor with the selected spatial dimensions summed out.
        :rtype: keras.KerasTensor
        """
        return ops.sum(inputs, axis=self._sum_axes, keepdims=self.keepdims)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        sum_axes = self._sum_axes or self._resolve_axes(len(input_shape))

        if self.keepdims:
            return tuple(
                1 if index in sum_axes else dim
                for index, dim in enumerate(input_shape)
            )

        return tuple(
            dim for index, dim in enumerate(input_shape)
            if index not in sum_axes
        )

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axes": list(self.axes) if self.axes is not None else None,
            "keepdims": self.keepdims,
        })
        return config

# ---------------------------------------------------------------------

"""Densify a sparse depth map with Gaussian-window splatting.

``SparseDepthSplat`` takes a full-resolution, grid-aligned sparse depth map
``(B, H, W, 1)`` and a validity mask of the same shape, and returns a dense
depth map plus a confidence map. It convolves the masked depth values and the
mask itself with the same fixed Gaussian kernel, then divides one by the other,

    dense = conv(depth * mask) / conv(mask)

which is Shepard's method with a Gaussian window. Since the input is already
grid-aligned there is no point list to rasterize, so no scatter or segment-sum
operation appears anywhere in the layer; the smoothing comes from two
``GaussianFilter`` instances rather than from fresh kernel math.

The caller supplies a dense map, not a ragged point list. Both Gaussian filters
are non-trainable. Where the accumulated weight is at or below ``epsilon``, so
no sparse sample falls inside the kernel's receptive field, the depth output is
forced to exactly ``0.0`` rather than left as a division artifact. The
accumulated weight doubles as the confidence map: the kernel sums to 1 and the
mask holds 0 or 1, so that weight already lies in ``[0, 1]``.
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.signal_processing.gaussian_filter import GaussianFilter
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.geometric.sparse_depth_splat")
class SparseDepthSplat(keras.layers.Layer):
    """Densify a sparse depth map into a dense depth map and a confidence mask.

    Call the layer on a pair, ``layer((sparse_depth, validity_mask))``, both
    ``(B, H, W, 1)``. The depth values are masked, both signals are blurred by
    their own fixed Gaussian filter, and the blurred value is divided by the
    blurred weight. The two filters carry the same ``kernel_size`` and
    ``sigma`` but are separate instances, so each owns its kernel under Keras 3
    tracking.

    Architecture:

    .. code-block:: text

                sparse_depth                          validity_mask
                      │                                     │
                      ▼                                     ▼
            ┌───────────────────┐             ┌───────────────────┐
            │ depth * mask      │             │ cast to float     │
            └─────────┬─────────┘             └─────────┬─────────┘
                      ▼                                 ▼
            ┌───────────────────┐             ┌───────────────────┐
            │ GaussianFilter    │             │ GaussianFilter    │
            │  fixed kernel     │             │  fixed kernel     │
            └─────────┬─────────┘             └─────────┬─────────┘
              accumulated_value                 accumulated_weight
                      │                                 │
                      └─────┬───────────────────────────┤
                            ▼                           ▼
            ┌───────────────────────┐     ┌───────────────────┐
            │ value / max(w, eps)   │     │ clip to [0, 1]    │
            │  zero where w <= eps  │     │                   │
            └───────────┬───────────┘     └─────────┬─────────┘
                        ▼                           ▼
                   dense_depth                dense_confidence
                    [B,H,W,1]                     [B,H,W,1]

    :param kernel_size: Height and width of the fixed Gaussian window. Defaults
        to ``(9, 9)``.
    :type kernel_size: Tuple[int, int]
    :param sigma: Standard deviation of the Gaussian window, shared by both
        axes and forwarded to ``GaussianFilter``. Defaults to ``2.0``.
    :type sigma: float
    :param epsilon: Accumulated-weight threshold. A pixel whose weight is at or
        below it is treated as having no support, and its depth is forced to
        ``0.0``. Defaults to ``1e-8``.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        Tuple or list of two 4D tensors, each ``(batch_size, height, width,
        1)``: ``sparse_depth`` and ``validity_mask``.

    Output shape:
        Tuple of two 4D tensors, each ``(batch_size, height, width, 1)``:
        ``(dense_depth, dense_confidence)``. ``dense_confidence`` is bounded
        in ``[0, 1]``.

    :raises ValueError: If ``kernel_size`` is not length 2.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.geometric.sparse_depth_splat import (
            SparseDepthSplat,
        )

        depth = keras.ops.zeros((2, 32, 32, 1))
        mask = keras.ops.zeros((2, 32, 32, 1))
        dense_depth, dense_mask = SparseDepthSplat()((depth, mask))
        dense_depth.shape  # (2, 32, 32, 1)
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (9, 9),
            sigma: float = 2.0,
            epsilon: float = 1e-8,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if len(kernel_size) != 2:
            raise ValueError(f"kernel_size must be length 2, got {kernel_size}")
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)

        # Separate instances rather than one filter called twice, so each
        # builds and owns its own kernel weight under Keras 3 tracking.
        self._value_filter = GaussianFilter(
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            padding="same",
            trainable=False,
            name="value_gaussian",
        )
        self._weight_filter = GaussianFilter(
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            padding="same",
            trainable=False,
            name="weight_gaussian",
        )

    def build(self, input_shape: Tuple) -> None:
        """Build both inner ``GaussianFilter`` sublayers.

        Both filters always see a single-channel map, so the shape needed to
        build them does not depend on the caller's spatial resolution. Building
        them here rather than at first call keeps them built for an owner that
        only conditionally invokes this layer.

        :param input_shape: The pair of input shapes
            ``(sparse_depth_shape, validity_mask_shape)``.
        :type input_shape: list of tuple
        """
        # DECISION D-015: build both filters here, unconditionally; an owner
        # that never calls this layer would otherwise leave them unbuilt, which
        # Keras 3 reports as unbuilt state. See decisions.md.
        single_channel_shape = (None, None, None, 1)
        self._value_filter.build(single_channel_shape)
        self._weight_filter.build(single_channel_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Densify the sparse depth map.

        :param inputs: The pair ``(sparse_depth, validity_mask)``, each
            ``(B, H, W, 1)``.
        :type inputs: tuple of keras.KerasTensor
        :param training: Forwarded to the inner ``GaussianFilter`` calls.
        :type training: Optional[bool]
        :return: ``(dense_depth, dense_confidence)``, each ``(B, H, W, 1)``.
        :rtype: tuple of keras.KerasTensor
        """
        sparse_depth, validity_mask = inputs
        validity_mask = keras.ops.cast(validity_mask, sparse_depth.dtype)
        masked_depth = sparse_depth * validity_mask

        accumulated_value = self._value_filter(masked_depth, training=training)
        accumulated_weight = self._weight_filter(validity_mask, training=training)

        has_support = keras.ops.cast(
            accumulated_weight > self.epsilon, accumulated_value.dtype
        )
        safe_weight = keras.ops.maximum(accumulated_weight, self.epsilon)
        # The multiply gives exact 0.0, not a tiny quotient, where no sparse
        # sample fell inside the receptive field.
        dense_depth = (accumulated_value / safe_weight) * has_support
        dense_confidence = keras.ops.clip(accumulated_weight, 0.0, 1.0)

        return dense_depth, dense_confidence

    def compute_output_shape(
            self,
            input_shape: Tuple,
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return the shape of each output tensor, both unchanged.

        :param input_shape: The pair of input shapes.
        :type input_shape: list of tuple
        :return: ``(input_shape[0], input_shape[0])``; both outputs share the
            sparse-depth input's shape.
        :rtype: tuple
        """
        depth_shape = tuple(input_shape[0])
        return depth_shape, depth_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config plus every constructor argument.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
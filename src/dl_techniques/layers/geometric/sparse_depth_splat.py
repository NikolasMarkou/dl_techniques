"""Scatter-free densification of a sparse depth map via Gaussian-window
splatting.

Input contract (plan.md Step 5, this is the decision this module fixes):
    A **full-resolution, grid-aligned** sparse depth map, ``(B, H, W, 1)``,
    mostly zero/invalid, paired with a full-resolution validity mask,
    ``(B, H, W, 1)``, marking which pixels hold a real sparse sample. This is
    the simplest contract that avoids ragged/variable-length point lists,
    which ``keras.ops`` handles poorly, and it is exactly the shape LiDAR
    projected into an image plane, or KITTI's own sparse depth PNGs, already
    come in.

Build-time finding on decisions.md D-003 (scatter-free strategy):
    D-003 defaults to a one-hot-encode + matmul rasterization step specifically
    to avoid depending on an unverified/possibly-stale ``keras.ops``
    scatter/segment-sum primitive for "nearest grid cell" accumulation of a
    raw point LIST. **That concern does not apply to this module**: the input
    contract above is already a dense, grid-aligned map -- there is no point
    list to rasterize, and therefore no scatter-shaped operation anywhere in
    this layer at all, verified or not. D-003's fallback is simply moot here,
    not substituted; this is the documented build-time finding the plan asked
    for. All this layer needs is a spatial smoothing/normalization step, which
    it gets by reusing the existing
    :class:`~dl_techniques.layers.signal_processing.gaussian_filter.GaussianFilter`
    depthwise-convolution layer (backed by
    ``dl_techniques.utils.tensors.depthwise_gaussian_kernel``) rather than
    reimplementing Gaussian-kernel math a second time.

Algorithm (standard splat normalization, a.k.a. Shepard's method with a
Gaussian window):
    1. Zero the sparse depth values outside valid pixels (defensive; the
       caller's own mask already implies this).
    2. Convolve the masked depth values, and separately the validity mask,
       with the SAME fixed (non-trainable) Gaussian kernel.
    3. Divide the accumulated value by the accumulated weight. Where the
       accumulated weight is (numerically) zero -- no sparse sample falls
       within the kernel's receptive field at that pixel -- the result is
       forced to exactly ``0.0`` rather than a division artifact, per the
       Problem Statement's "must degrade to no depth prior, not
       divide-by-zero" edge case.

    The accumulated weight doubles as a densified confidence/validity mask:
    since the Gaussian kernel is normalized (sums to 1) and the input mask is
    valued in ``{0, 1}``, the accumulated weight is already bounded in
    ``[0, 1]`` -- 1.0 where every kernel tap hit a valid sample, 0.0 where
    none did.
"""

from typing import Any, Dict, Optional, Tuple

import keras

from dl_techniques.layers.signal_processing.gaussian_filter import GaussianFilter
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.geometric.sparse_depth_splat")
class SparseDepthSplat(keras.layers.Layer):
    """Densify a sparse depth map into a dense depth map + confidence mask.

    Call the layer on a pair: ``layer((sparse_depth, validity_mask))``, both
    ``(B, H, W, 1)``. See this module's docstring for the full input contract
    and algorithm.

    :param kernel_size: Height and width of the fixed Gaussian window.
    :type kernel_size: Tuple[int, int]
    :param sigma: Standard deviation of the Gaussian window (isotropic, one
        value shared by both axes). Forwarded to ``GaussianFilter``.
    :type sigma: float
    :param epsilon: Accumulated-weight floor below which a pixel is treated
        as having no support (output forced to ``0.0``, not a division
        artifact).
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        Tuple or list of two 4D tensors, each ``(batch_size, height, width,
        1)``: ``sparse_depth`` and ``validity_mask``.

    Output shape:
        Tuple of two 4D tensors, each ``(batch_size, height, width, 1)``:
        ``(dense_depth, dense_confidence)``. ``dense_confidence`` is bounded
        in ``[0, 1]``.

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

        # Two independent, non-trainable Gaussian filters -- one for the
        # (masked) depth values, one for the validity weights. They must be
        # separate layer instances (not one filter called twice) so each
        # builds/owns its own kernel weight cleanly under Keras 3 tracking.
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
        """Eagerly build both inner ``GaussianFilter`` sublayers.

        A caller that only conditionally invokes this layer's ``call()``
        (e.g. `OmniPoint`'s `conditioning.py`, when sparse depth is absent
        for a whole batch) would otherwise leave `_value_filter`/
        `_weight_filter` -- and their own tracked kernel weights -- unbuilt,
        which Keras 3 reports as "unbuilt state" on the OWNING layer once
        that layer is itself marked built. Building both filters explicitly
        here, independent of which branch `call()` takes at runtime, avoids
        that: both channels are always a single-channel map (the sparse
        depth value or its validity mask), so the shape needed to build them
        never depends on the caller's actual spatial resolution.

        :param input_shape: The pair of input shapes
            ``(sparse_depth_shape, validity_mask_shape)``.
        :type input_shape: list of tuple
        """
        # DECISION plan-2026-09-11T050223-1b47bcf6/D-015: build both filters
        # HERE, unconditionally -- do not rely on GaussianFilter's own
        # lazy build-at-first-call. A caller that never invokes this layer's
        # call() (an absent-modality branch) would otherwise leave these
        # sublayers unbuilt, which `pyproject.toml`'s `error::UserWarning`
        # turns into a hard test failure. See decisions.md.
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
        # Force exact 0.0 (not a tiny division artifact) wherever no sparse
        # sample fell within this pixel's receptive field -- the Problem
        # Statement's "must degrade to no depth prior, not divide-by-zero"
        # edge case.
        dense_depth = (accumulated_value / safe_weight) * has_support
        dense_confidence = keras.ops.clip(accumulated_weight, 0.0, 1.0)

        return dense_depth, dense_confidence

    def compute_output_shape(
            self,
            input_shape: Tuple,
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return the (unchanged) shape of each output tensor.

        :param input_shape: The pair of input shapes.
        :type input_shape: list of tuple
        :return: ``(input_shape[0], input_shape[0])`` (depth and mask share
            the sparse-depth input's shape).
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

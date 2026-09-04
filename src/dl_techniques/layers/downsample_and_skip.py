"""
One encoder junction: produce ``(skip, downsampled)`` from a single input.

This module holds :class:`DownsampleAndSkip`, the single home for the "skip
connection + downsample" decision made at every encoder level of the U-Net-shaped
denoisers in :mod:`dl_techniques.models.vision.bias_free_denoisers` (``bfunet`` and
``bfconvunext``). Both the raw-skip/pooling path and the Laplacian-pyramid split
path live here, so the branch is written once rather than once per model family.
"""

import keras
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.signal_processing.laplacian_filter import LaplacianPyramidLevel
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.downsample_and_skip")
class DownsampleAndSkip(keras.layers.Layer):
    """Produce ``(skip, downsampled)`` for one encoder junction.

    With ``use_laplacian_pyramid=False`` (the default): the skip is the
    pre-downsample tensor itself, and downsampling is ``MaxPooling2D(2, 2)``,
    ``AveragePooling2D(2, 2)`` (``pool_type='average'``), or a channel-preserving
    ``Conv2D(kernel_size=2, strides=2)`` (``pool_type='strided_conv'`` -- the
    only branch that carries weights; ``filters`` is taken from the input
    channel count at build time, not fixed at construction). Average pooling
    and a bias-free strided conv are both linear, keeping the encoder path
    linear where a bias-free / Miyasawa denoiser arm needs that.

    With ``use_laplacian_pyramid=True``: a channel-preserving, bias-free
    :class:`~dl_techniques.layers.signal_processing.laplacian_filter.LaplacianPyramidLevel`
    split. The full-resolution high-frequency band becomes the skip; the
    half-resolution low band continues down the encoder. ``pool_type`` is
    inert on this path -- the pyramid already pools linearly.

    Output order is ``(skip, downsampled)`` on both paths: the Laplacian path
    returns ``(high, low)``, high band first. Both outputs are rank-4, so a
    swapped tuple at a call site is shape-compatible and invisible to a
    shape-only assertion -- treat the order as a hard contract.

    This layer is a wrapper: its own name is the pooling layer's historical
    name (e.g. ``'bottleneck_downsample'``), and its inner sub-layer is named
    ``f'{name}_pool'``, ``f'{name}_conv'`` or ``f'{name}_pyramid'``,
    deterministically, so the graph is reproducible across builds.

    :param use_laplacian_pyramid: Select the Laplacian split path when ``True``,
        the raw-skip + pooling path when ``False``.
    :type use_laplacian_pyramid: bool
    :param laplacian_kernel_size: Gaussian blur kernel ``(height, width)`` handed to
        :class:`LaplacianPyramidLevel`. Inert when ``use_laplacian_pyramid`` is
        ``False``, but still recorded in ``get_config`` so a config round trip
        cannot silently lose it.
    :type laplacian_kernel_size: Tuple[int, int]
    :param pool_type: ``'max'`` (default), ``'average'`` or ``'strided_conv'``.
        Inert when ``use_laplacian_pyramid`` is ``True``. ``'strided_conv'``
        preserves the channel count rather than widening it, unlike the
        deleted ``ConvUNextModel._downsample``, since callers already widen
        channels as a separate step.
    :type pool_type: str
    :param use_bias: Whether the ``'strided_conv'`` downsample carries a bias vector.
        Defaults to ``True``. Inert on every other branch (pooling and the
        Laplacian pyramid are weightless). Pass ``False`` on a bias-free /
        degree-1 homogeneous arm -- a bias-free strided conv is linear and
        preserves ``f(a*x) = a*f(x)``.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the ``'strided_conv'`` kernel. Inert on
        every other branch.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for the ``'strided_conv'`` kernel. Inert on
        every other branch.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    :raises ValueError: If ``pool_type`` is not one of ``'max'``, ``'average'`` or
        ``'strided_conv'``.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``.

    Output shape:
        Tuple of two 4D tensors ``(skip, downsampled)``:
        ``skip`` is ``(batch_size, height, width, channels)`` on both paths and
        ``downsampled`` is ``(batch_size, height / 2, width / 2, channels)``.

    Example:

    .. code-block:: python

        junction = DownsampleAndSkip(
            use_laplacian_pyramid=False,
            laplacian_kernel_size=(5, 5),
            pool_type='max',
            name='bottleneck_downsample',
        )
        skip, x = junction(x)
    """

    _VALID_POOL_TYPES = ("max", "average", "strided_conv")

    def __init__(
            self,
            use_laplacian_pyramid: bool,
            laplacian_kernel_size: Tuple[int, int] = (5, 5),
            pool_type: str = "max",
            use_bias: bool = True,
            kernel_initializer: Union[
                str, keras.initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[
                Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if pool_type not in self._VALID_POOL_TYPES:
            raise ValueError(
                f"pool_type must be one of {self._VALID_POOL_TYPES}, "
                f"got {pool_type!r}"
            )
        if (not isinstance(laplacian_kernel_size, Sequence)
                or isinstance(laplacian_kernel_size, str)
                or len(laplacian_kernel_size) != 2):
            raise ValueError(
                "laplacian_kernel_size must be a length-2 sequence "
                f"(height, width), got {laplacian_kernel_size!r}"
            )

        self.use_laplacian_pyramid = bool(use_laplacian_pyramid)
        self.laplacian_kernel_size = tuple(laplacian_kernel_size)
        self.pool_type = pool_type
        self.use_bias = bool(use_bias)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Only the active branch's sub-layer is constructed; the others stay None.
        self.pyramid: Optional[LaplacianPyramidLevel] = None
        self.pool: Optional[keras.layers.Layer] = None
        self.conv: Optional[keras.layers.Conv2D] = None

        # DECISION plan-2026-08-14T092357-0e3d792d/D-009: the layer takes one name (the
        # pooling layer's historical one); inner sub-layers derive theirs from it, never auto-numbered. See decisions.md.
        if self.use_laplacian_pyramid:
            self.pyramid = LaplacianPyramidLevel(
                blur_kernel_size=self.laplacian_kernel_size,
                name=f"{self.name}_pyramid",
            )
        elif self.pool_type != "strided_conv":
            pool_cls = (
                keras.layers.AveragePooling2D if self.pool_type == "average"
                else keras.layers.MaxPooling2D
            )
            self.pool = pool_cls(
                pool_size=(2, 2),
                name=f"{self.name}_pool",
            )
        # The 'strided_conv' branch needs the input channel count, so its
        # sub-layer is created in build(), not here.

    def build(self, input_shape) -> None:
        # DECISION plan-2026-08-14T092357-0e3d792d/D-013: 'strided_conv' preserves
        # channels (filters from input_shape[-1]), never fuses with channel widening -- callers already widen separately. See decisions.md.
        if self.pool_type == "strided_conv" and not self.use_laplacian_pyramid:
            self.conv = keras.layers.Conv2D(
                filters=int(input_shape[-1]),
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"{self.name}_conv",
            )

        # Explicit build, not lazy auto-build: a lazily-built sub-layer can
        # silently drop weights on .keras reload; only strided_conv carries weights.
        self._downsample_sublayer().build(input_shape)
        super().build(input_shape)

    def _downsample_sublayer(self) -> keras.layers.Layer:
        """Return the single constructed sub-layer for this instance's branch."""
        if self.use_laplacian_pyramid:
            return self.pyramid
        if self.pool_type == "strided_conv":
            return self.conv
        return self.pool

    def call(self, inputs):
        if self.use_laplacian_pyramid:
            low, high = self.pyramid(inputs)
            return high, low
        return inputs, self._downsample_sublayer()(inputs)

    def compute_output_shape(self, input_shape):
        if self.use_laplacian_pyramid:
            low_shape, high_shape = self.pyramid.compute_output_shape(input_shape)
            return high_shape, low_shape
        return (
            tuple(input_shape),
            self._downsample_sublayer().compute_output_shape(input_shape),
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "use_laplacian_pyramid": self.use_laplacian_pyramid,
                "laplacian_kernel_size": self.laplacian_kernel_size,
                "pool_type": self.pool_type,
                "use_bias": self.use_bias,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer),
            }
        )
        return config

# ---------------------------------------------------------------------

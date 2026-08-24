"""One encoder junction: produce ``(skip, downsampled)`` from a single input.

This module holds :class:`DownsampleAndSkip`, the single home for the "skip
connection + downsample" decision made at every encoder level of the U-Net-shaped
denoisers in :mod:`dl_techniques.models.bias_free_denoisers` (``bfunet`` and
``bfconvunext``). Both the raw-skip/pooling path and the Laplacian-pyramid split
path live here, so the branch is written once rather than once per model family.
"""

import keras
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.laplacian_filter import LaplacianPyramidLevel

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.layers")
class DownsampleAndSkip(keras.layers.Layer):
    """Produce ``(skip, downsampled)`` for one encoder junction.

    OFF path (``use_laplacian_pyramid=False``, the default architecture): the skip
    is the pre-downsample tensor ITSELF and downsampling is ``MaxPooling2D(2, 2)``.
    With ``pool_type='average'`` the downsample uses ``AveragePooling2D(2, 2)``
    instead -- a LINEAR (and bias-free / homogeneous) operator, so the encoder path
    stays linear for the Miyasawa/Tweedie residual-as-score interpretation
    (MaxPooling is non-linear). Pooling layers are weightless, so the swap does not
    affect checkpoint weight transfer.

    With ``pool_type='strided_conv'`` the downsample is instead a LEARNED
    ``Conv2D(kernel_size=2, strides=2)`` -- the only branch of this layer that carries
    weights. It is channel-PRESERVING (``filters`` is taken from the input channel
    count at build time) and, with ``use_bias=False``, it is a linear and therefore
    degree-1 homogeneous operator, so it is legal on a bias-free/Miyasawa denoiser
    arm. See the ``pool_type`` parameter for the deliberate divergence from the
    deleted ``ConvUNextModel``, whose strided conv also CHANGED the channel count.

    ON path (``use_laplacian_pyramid=True``): a channel-preserving, bias-free
    :class:`~dl_techniques.layers.laplacian_filter.LaplacianPyramidLevel` split. The
    full-resolution high-frequency band becomes the skip; the half-resolution low
    band continues down the encoder. Bias-free and homogeneous by construction
    (fixed blur + average pool + bilinear upsample, zero learnable bias). The
    pyramid already pools linearly, so ``pool_type`` does not apply here.

    **Output order is ``(skip, downsampled)`` on BOTH paths** -- i.e. the ON path
    returns ``(high, low)``, high band first. The two outputs are both rank-4, so a
    swapped tuple at a call site is shape-compatible and invisible to any
    shape-only assertion; treat the order as a hard contract.

    **Sub-layer naming.** This layer is a wrapper: it inserts one extra level in the
    functional graph relative to the free function it replaced. The wrapper carries
    the name the caller passes (historically the POOLING layer's name, e.g.
    ``'bottleneck_downsample'``), and its inner sub-layer is named
    ``f'{name}_pool'``, ``f'{name}_conv'`` or ``f'{name}_pyramid'`` --
    deterministically derived, never left to Keras' auto-numbering, so the graph is
    reproducible across builds.

    :param use_laplacian_pyramid: Select the Laplacian split path when ``True``,
        the raw-skip + pooling path when ``False``.
    :type use_laplacian_pyramid: bool
    :param laplacian_kernel_size: Gaussian blur kernel ``(height, width)`` handed to
        :class:`LaplacianPyramidLevel`. Inert on the OFF path, but still recorded in
        ``get_config`` so a config round trip cannot silently lose it.
    :type laplacian_kernel_size: Tuple[int, int]
    :param pool_type: ``'max'`` (default), ``'average'`` or ``'strided_conv'``. Inert
        on the ON path. ``'strided_conv'`` builds a LEARNED, channel-PRESERVING
        ``Conv2D(filters=input_channels, kernel_size=2, strides=2)``. Channel
        preservation is a deliberate divergence from the deleted
        ``ConvUNextModel._downsample``, which used
        ``Conv2D(filters=next_level_filters, kernel_size=2, strides=2)`` and so fused
        the downsample with the channel widening: this layer has no way to know the
        next level's width, and its callers already perform channel adjustment as a
        separate step, which stays valid only if this branch preserves channels.
    :type pool_type: str
    :param use_bias: Whether the ``'strided_conv'`` downsample carries a bias vector.
        Defaults to ``True``. Inert on every other branch (pooling and the Laplacian
        pyramid are weightless). Pass ``False`` on a bias-free / degree-1 homogeneous
        arm -- a bias-free strided conv is a linear operator and preserves
        ``f(a*x) = a*f(x)``.
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

        # Exactly ONE sub-layer exists per instance -- the unused branches are not
        # constructed, so an OFF-path junction carries no pyramid sub-layer.
        self.pyramid: Optional[LaplacianPyramidLevel] = None
        self.pool: Optional[keras.layers.Layer] = None
        self.conv: Optional[keras.layers.Conv2D] = None

        # DECISION plan-2026-08-14T092357-0e3d792d/D-009: the wrapper takes the name the
        # free function gave the POOLING layer ('encoder_downsample_N' / 'bottleneck_
        # downsample'), and the inner op is named f'{self.name}_pool' / f'{self.name}_
        # pyramid'. The free function's SECOND name argument (`pyramid_name`, e.g.
        # 'encoder_pyramid_N') is deliberately GONE -- a Layer has one name, and the
        # pooling name is the one every caller's comments and checkpoint-compat notes
        # cite. Do NOT "restore symmetry" by adding a `pyramid_name`/`pool_name`
        # constructor argument: it would have to be serialized, and two names for one
        # layer is exactly the ambiguity this consolidation removes. Do NOT drop the
        # explicit inner names and let Keras auto-number them either -- auto-numbering
        # is build-order dependent, so the graph would stop being reproducible across
        # builds. See decisions.md D-009 (accepted graph changes C-1/C-2).
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
        # The 'strided_conv' branch needs the input channel count (it is
        # channel-PRESERVING), so its sub-layer is created in `build`, not here.

    def build(self, input_shape) -> None:
        # DECISION plan-2026-08-14T092357-0e3d792d/D-013: the 'strided_conv' branch is
        # channel-PRESERVING -- `filters` comes from `input_shape[-1]`, never from a
        # constructor argument. The deleted `ConvUNextModel._downsample` used
        # `Conv2D(filters=<next level's width>, kernel_size=2, strides=2)`, fusing the
        # downsample with the channel widening. Do NOT "restore" that: this Layer has
        # no way to know the next level's width, and BOTH of its callers already widen
        # channels in a separate, separately-named step (`encoder_level_N_channel_
        # adjust` / `MatchChannels`) which a channel-CHANGING junction would silently
        # duplicate or bypass. Do NOT add a `filters` constructor argument to work
        # around that either -- it would have to be serialized, and it would let a
        # caller build a junction whose output width contradicts the very next layer's
        # expectation with no error until the residual add fails. The skip is the RAW
        # INPUT here, exactly as on the pooling branch. See decisions.md D-013.
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

        # Build the sub-layer EXPLICITLY rather than relying on lazy auto-build:
        # this repo has a recorded defect class where lazily-built sub-layers
        # silently drop weights on `.keras` reload. The pooling and pyramid branches
        # are weightless, but the strided-conv branch is NOT, so the explicit build is
        # load-bearing there; it also makes compute_output_shape exact everywhere.
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

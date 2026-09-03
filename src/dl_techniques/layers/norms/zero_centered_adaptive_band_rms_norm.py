"""ZeroCenteredAdaptiveBandRMS: centering, RMS normalization, then an adaptive band scale.

This layer is :class:`AdaptiveBandRMS` with a mean subtraction in front of it.
The input is centered over the normalization axes, divided by the RMS of the
CENTERED tensor, then multiplied by a scale confined to ``[1 - alpha, 1]`` that
is computed from the input's own aggregate RMS.

Computation
-----------

For an input ``x`` reduced over ``axis``::

    mu            = mean(x)
    centered      = x - mu
    rms           = maximum(sqrt(mean(centered ** 2) + epsilon), epsilon)
    normalized    = centered / rms
    rms_stats     = mean(rms) over every non-batch axis    -> (batch, 1)
    log_rms       = log(rms_stats)
    band_logits   = Dense(num_params)(log_rms)
    scale         = (1 - alpha) + alpha * sigmoid(5.0 * band_logits)
    output        = normalized * scale

The default ``band_initializer="zeros"`` makes the ``Dense`` output zero, and
``sigmoid(0) = 0.5``, so the initial scale is the MIDPOINT of the band, not its
top. Measured output RMS on a ``(4, 32)`` input: ``0.95`` at ``alpha=0.1`` and
``0.75`` at ``alpha=0.5``, with no spread across rows to six decimals (measured
spread at most ``4.83e-07`` over shapes ``(4, 32)`` to ``(64, 128)``, which is
float32 rounding). Both ends of the band are reachable during training:
assigning the ``Dense`` bias ``-5`` gives an output RMS of ``0.900000`` and
``+5`` gives ``1.000000`` at ``alpha=0.1``.

Centering is what makes the statistics dtype matter here. Statistics run in
``keras.backend.result_type(input_dtype, "float32")``, which is float64 under a
float64 policy. Measured on the float64 input
``[[1e8+1, 1e8+2, 1e8+3, 1e8+4]]``: float32 statistics center it to exactly
``[[0, 0, 0, 0]]`` and the layer returns all zeros, while float64 statistics
center it to ``[[-1.5, -0.5, 0.5, 1.5]]`` and the layer returns
``[[-1.2745587, -0.4248529, 0.4248529, 1.2745587]]`` at ``alpha=0.1``. A
hardcoded ``"float32"`` would destroy the signal, not merely round it.

References
----------

[1] Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization."
    https://arxiv.org/abs/1910.07467
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Union, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.dtype_policy import statistics_dtype
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.norms.zero_centered_adaptive_band_rms_norm")
class ZeroCenteredAdaptiveBandRMS(keras.layers.Layer):
    """Center, normalize by the root mean square, then scale by an adaptive band factor.

    Subtracts the mean over ``axis``, divides by
    ``maximum(sqrt(mean(centered ** 2) + epsilon), epsilon)``, then multiplies by
    a scale confined to ``[1 - alpha, 1]``, where ``alpha`` is
    ``max_band_width``. The scale comes from an internal ``Dense`` fed with the
    log of the aggregate RMS, so it moves with the input. The output has the same
    shape and dtype as the input.

    This is :class:`AdaptiveBandRMS` plus the mean subtraction. The centering
    step is what separates it from :class:`AdaptiveBandRMS`, and the rest of the
    pipeline is the same.

    The layer works on any rank from 2 upward, with any axis spelling that does
    not name the batch axis. Its only weights live in the internal ``Dense``,
    which is created in ``build()`` because its width depends on the input shape.

    At the default ``band_initializer="zeros"`` the ``Dense`` output is zero and
    ``sigmoid(0) = 0.5``, so the scale starts at the middle of the band. Measured
    output RMS on a ``(4, 32)`` input: ``0.95`` at ``alpha=0.1`` and ``0.75`` at
    ``alpha=0.5``, with no spread across rows to six decimals (measured spread
    at most ``4.83e-07`` over shapes ``(4, 32)`` to ``(64, 128)``, float32
    rounding). Assigning the ``Dense`` bias ``-5`` gives ``0.900000`` and ``+5``
    gives ``1.000000`` at ``alpha=0.1``, so training reaches both ends of the
    band.

    Statistics run in ``keras.backend.result_type(input_dtype, "float32")``:
    float32 at minimum, float64 under a float64 policy. The result is cast back
    to the input dtype before it is returned.

    .. warning::
        **This layer does not support masking.** ``supports_masking`` stays
        ``False``. The aggregate RMS that drives the ``Dense`` is reduced over
        every non-batch axis, so ONE scale rescales all of a sample's positions
        together. Any position's value reaches every other position's output as
        soon as the ``Dense`` kernel is non-zero, which is the trained regime. A
        propagated Keras mask would be invalid on the output.

        **A default-constructed probe cannot see this.** The default
        ``band_initializer="zeros"`` pins the ``Dense`` output to a constant, so
        a leak probe on a fresh layer reads exactly ``0.0`` and the flag looks
        safe for the wrong reason. Measured on a ``(4, 5, 8)`` input, perturbing
        one ``(sample, token)`` slot, with the ``Dense`` kernel assigned from
        ``numpy.random.default_rng(1).normal`` (the ``_make_nontrivial`` helper
        in ``tests/test_layers/test_norms/test_the_norms_propagate_masks.py``):
        other positions move by up to ``1.864e-01``. With the untouched default
        kernel: exactly ``0.0``.

    .. warning::
        **Resolution lock.** ``build()`` sizes the internal ``Dense`` from the
        PRODUCT of the sizes at the normalized axes, so a normalized axis whose
        size varies between calls locks the layer to its build-time value.
        Measured: ``axis=(1, 2)`` built on ``(None, 8, 8, 3)`` gives
        ``Dense(units=64)``, and calling that built layer on ``(2, 16, 16, 3)``
        raises ``InvalidArgumentError``. On CPU the message is
        ``Incompatible shapes: [2,16,16,3] vs. [2,8,8,1] [Op:Mul]``; on GPU the
        same op reports ``required broadcastable shapes [Op:Mul]``.

        Two spellings are NOT locked. The default ``axis=-1`` sizes ``units``
        from the channel count, which does not vary, and the global spelling
        ``axis=(1, 2, 3)`` on a rank-4 input gives ``units=1``. Measured: both
        accept ``16x16`` after being built at ``8x8``.

        :class:`AdaptiveBandRMS` behaves identically here. Its
        ``_compute_param_shape_and_axes`` is a separate copy of the same logic,
        not a shared helper, so a change to one does not reach the other.

    **Architecture Overview:**

    .. code-block:: text

            input: x  (batch, ..., F)
                │
                ▼
        ┌────────────────────────────────────────────────────────────┐
        │ cast inputs to stat_dtype =                                │
        │ result_type(input dtype, "float32")                        │
        └───────┬────────────────────────────────────────────────────┘
                │ x_stat
                ▼
        ┌────────────────────────────────────────────────────────────┐
        │ mean = mean(x_stat) over axis, keepdims=True               │
        │ centered = x_stat - mean                                   │
        └───────┬────────────────────────────────────────────────────┘
                │ centered
                ▼
        ┌────────────────────────────────────────────────────────────┐
        │ mean_square = mean(centered ** 2) over axis, keepdims=True │
        │ rms = maximum(sqrt(mean_square + epsilon), epsilon)        │
        └───────┬────────────────────────────────────────────────────┘
                │ rms: (batch, ..., 1)
                ├─────────────────────────────────┐
                ▼                                 ▼
        ┌───────────────────┐                     │
        │ normalized =      │                     │
        │   centered / rms  │                     │
        └───────┬───────────┘                     │
                │                                 ▼
                │                 ┌──────────────────────────────────┐
                │                 │ rms_stats = mean(rms) over every │
                │                 │ non-batch axis, reshaped to      │
                │                 │ (batch, 1)                       │
                │                 └───────────────┬──────────────────┘
                │                                 │
                │                                 ▼
                │                 ┌──────────────────────────────────┐
                │                 │ log_rms = log(rms_stats)         │
                │                 └───────────────┬──────────────────┘
                │                                 │
                │                                 ▼
                │                 ┌──────────────────────────────────┐
                │                 │ band_logits =                    │
                │                 │   dense_layer(log_rms)           │
                │                 └───────────────┬──────────────────┘
                │                                 │
                │                                 ▼
                │                 ┌──────────────────────────────────┐
                │                 │ band = sigmoid(5.0 * band_logits)│
                │                 │ scale = (1 - max_band_width)     │
                │                 │         + max_band_width * band  │
                │                 └───────────────┬──────────────────┘
                │                                 │
                │                                 ▼
                │                 ┌──────────────────────────────────┐
                │                 │ scale_factors = reshape(scale,   │
                │                 │   (batch,) + param_shape)        │
                │                 └───────────────┬──────────────────┘
                ▼                                 ▼
        ┌────────────────────────────────────────────────────────────┐
        │ output = normalized * cast(scale_factors, stat_dtype)      │
        └───────┬────────────────────────────────────────────────────┘
                │
                ▼
        ┌────────────────────────────────────────────────────────────┐
        │ cast back to the input dtype                               │
        └───────┬────────────────────────────────────────────────────┘
                │
                ▼
            output: (batch, ..., F), same shape and dtype as x

    :param max_band_width: Thickness of the band, written ``alpha`` above. The
        scale is confined to ``[1 - alpha, 1]``. Must satisfy ``0 < alpha < 1``.
        Defaults to 0.1.
    :type max_band_width: float
    :param axis: Axis or axes reduced by the mean and RMS statistics. The default
        -1 reduces the last dimension. Can be an int or a tuple of ints. Axis 0
        is rejected in ``build()``.
    :type axis: Union[int, Tuple[int, ...]]
    :param epsilon: Constant added inside the square root, and also the floor the
        RMS is clamped to. Must be positive. Defaults to 1e-7.
    :type epsilon: float
    :param band_initializer: Initializer for the internal ``Dense`` kernel.
        Defaults to ``"zeros"``, which starts the scale at the MIDDLE of the
        band, not at 1.0.
    :type band_initializer: Union[str, keras.initializers.Initializer]
    :param band_regularizer: Regularizer for the internal ``Dense`` kernel.
        Defaults to None, matching :class:`AdaptiveBandRMS`.
    :type band_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar max_band_width: The configured band thickness.
    :vartype max_band_width: float
    :ivar axis: The configured normalization axis or axes, kept verbatim.
    :vartype axis: Union[int, Tuple[int, ...]]
    :ivar epsilon: The configured numerical constant.
    :vartype epsilon: float
    :ivar band_initializer: The resolved initializer for the ``Dense`` kernel.
    :vartype band_initializer: keras.initializers.Initializer
    :ivar band_regularizer: The resolved regularizer for the ``Dense`` kernel.
    :vartype band_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar dense_layer: The internal projection, or None until ``build()`` runs.
        It holds every weight this layer owns.
    :vartype dense_layer: Optional[keras.layers.Dense]

    :raises ValueError: If ``max_band_width`` is not strictly between 0 and 1.
    :raises ValueError: If ``epsilon`` is not positive.
    :raises TypeError: If ``axis`` is not an int or a sequence of ints.
    :raises ValueError: If ``axis`` names the batch dimension. Raised in
        ``build()``.
    :raises ValueError: If ``axis`` is out of bounds for the input rank. Raised
        in ``build()``.
    :raises ValueError: If a normalized axis has an undefined (None) size. Raised
        in ``build()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import ZeroCenteredAdaptiveBandRMS

        x = keras.random.normal((4, 16, 64))
        y = ZeroCenteredAdaptiveBandRMS(max_band_width=0.1)(x)
    """

    def __init__(
        self,
        max_band_width: float = 0.1,
        axis: Union[int, Tuple[int, ...]] = -1,
        epsilon: float = 1e-7,
        band_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        band_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        """Store the configuration and validate it.

        No weight is created here. The internal ``Dense`` needs the input shape,
        so it is built in ``build()``.

        :param max_band_width: Band thickness, in ``(0, 1)``.
        :type max_band_width: float
        :param axis: Normalization axis or axes.
        :type axis: Union[int, Tuple[int, ...]]
        :param epsilon: Positive numerical constant.
        :type epsilon: float
        :param band_initializer: Initializer for the ``Dense`` kernel.
        :type band_initializer: Union[str, keras.initializers.Initializer]
        :param band_regularizer: Regularizer for the ``Dense`` kernel.
        :type band_regularizer: Optional[keras.regularizers.Regularizer]
        :param kwargs: Forwarded to ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If max_band_width or epsilon is out of range.
        :raises TypeError: If axis is not an int or a sequence of ints.
        """
        super().__init__(**kwargs)

        self._validate_inputs(max_band_width, axis, epsilon)

        # Store ALL configuration parameters for get_config().
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon
        self.band_initializer = keras.initializers.get(band_initializer)
        self.band_regularizer = keras.regularizers.get(band_regularizer)

        # Shape computation results (set in build).
        self._param_shape: Optional[Tuple[int, ...]] = None
        self._scaling_axes: Optional[List[int]] = None

        # Sub-layer created in build() due to input-shape dependency.
        self.dense_layer: Optional[keras.layers.Dense] = None

        logger.debug(
            f"Initialized ZeroCenteredAdaptiveBandRMS: "
            f"max_band_width={max_band_width}, "
            f"axis={axis}, "
            f"epsilon={epsilon}"
        )

    def _validate_inputs(
        self,
        max_band_width: float,
        axis: Union[int, Tuple[int, ...]],
        epsilon: float,
    ) -> None:
        """Reject an out-of-range band width, a non-positive epsilon, or a bad axis type.

        This checks the axis TYPE only. Whether the axis is in bounds, and
        whether it names the batch dimension, needs the input rank and is checked
        in ``build()``.

        :param max_band_width: Band thickness to validate.
        :type max_band_width: float
        :param axis: Normalization axis or axes to validate.
        :type axis: Union[int, Tuple[int, ...]]
        :param epsilon: Epsilon value to validate.
        :type epsilon: float

        :raises ValueError: If max_band_width is not strictly between 0 and 1.
        :raises ValueError: If epsilon is not positive.
        :raises TypeError: If axis is not an int or a sequence of ints.
        """
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if isinstance(axis, (list, tuple)):
            if not all(isinstance(ax, int) for ax in axis):
                raise TypeError(
                    f"All elements in axis must be integers, got {axis}"
                )
        elif not isinstance(axis, int):
            raise TypeError(
                f"axis must be int or tuple of ints, got {type(axis)}"
            )

    def _compute_param_shape_and_axes(
        self,
        input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Tuple[int, ...], List[int]]:
        """Decide the broadcast shape of the scale, and which axes get their own scale.

        The returned ``param_shape`` excludes the batch dimension. Its product is
        the ``Dense`` width. A normalized axis contributes its own size; every
        other non-batch axis contributes 1, so the scale broadcasts along it.

        One case is special. When every non-batch axis is normalized and the rank
        is above 2, ``param_shape`` is all ones and the layer gets a single
        broadcasting scale. That spelling is the only multi-axis one that
        survives a resolution change, because ``units`` is then 1.

        :param input_shape: Shape of the input tensor, including the batch
            dimension.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``(param_shape, scaling_axes)``. ``param_shape`` is the scale's
            shape without the batch dimension; ``scaling_axes`` lists the axes
            that get an independent scale.
        :rtype: Tuple[Tuple[int, ...], List[int]]

        :raises ValueError: If an axis names the batch dimension.
        :raises ValueError: If an axis is out of bounds for the input rank.
        :raises ValueError: If a normalized axis has an undefined (None) size.
        """
        input_rank = len(input_shape)

        if isinstance(self.axis, int):
            axes = [self.axis]
        else:
            axes = list(self.axis)

        normalized_axes: List[int] = []
        for ax in axes:
            if ax < 0:
                ax = input_rank + ax
            if ax == 0:
                raise ValueError(
                    "axis 0 (batch dimension) cannot be normalized"
                )
            if ax < 0 or ax >= input_rank:
                raise ValueError(
                    f"axis {ax} is out of bounds for input with {input_rank} "
                    f"dimensions"
                )
            normalized_axes.append(ax)
        normalized_axes = sorted(set(normalized_axes))

        is_global = (
            input_rank > 2 and len(normalized_axes) == input_rank - 1
        )
        if is_global:
            param_shape = [1] * (input_rank - 1)
            return tuple(param_shape), []

        param_shape: List[int] = []
        scaling_axes: List[int] = []
        # Start at 1: the batch dimension never gets a scale of its own.
        for i in range(1, input_rank):
            if i in normalized_axes:
                if input_shape[i] is None:
                    raise ValueError(
                        f"Normalized axis {i} has an undefined (None) size; "
                        f"ZeroCenteredAdaptiveBandRMS needs a static dimension to "
                        f"size its scaling Dense layer. Got input_shape={input_shape}."
                    )
                param_shape.append(input_shape[i])
                scaling_axes.append(i)
            else:
                param_shape.append(1)
        return tuple(param_shape), scaling_axes

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create and build the internal ``Dense``, sized from the input shape.

        The ``Dense`` width is the product of ``param_shape``, so it depends on
        the input shape. That is why the sub-layer is created here rather than in
        ``__init__``. It also means a normalized axis whose size varies between
        calls locks the layer to its build-time value; see the class docstring's
        resolution-lock warning.

        The ``Dense`` input is always ``(None, 1)``, because the aggregated
        log-RMS is one number per sample.

        :param input_shape: Shape tuple of the input tensor. The batch dimension
            may be None.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the axis configuration is invalid for this rank.
        """
        if self.built:
            return

        self._param_shape, self._scaling_axes = (
            self._compute_param_shape_and_axes(input_shape)
        )

        num_params = 1
        for dim in self._param_shape:
            if dim is not None:
                num_params *= dim

        self.dense_layer = keras.layers.Dense(
            units=num_params,
            kernel_initializer=self.band_initializer,
            kernel_regularizer=self.band_regularizer,
            use_bias=True,
            name="band_dense",
        )
        # Aggregated log-RMS is always shape (batch, 1).
        self.dense_layer.build((None, 1))

        logger.debug(
            f"Built ZeroCenteredAdaptiveBandRMS: "
            f"param_shape={self._param_shape}, num_params={num_params}"
        )

        super().build(input_shape)

    def _aggregate_rms_statistics(
        self, rms_tensor: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Reduce the RMS tensor to one number per sample.

        The ``Dense`` takes a single statistic per sample, so the per-position
        RMS values are averaged over every non-batch axis and reshaped to
        ``(batch, 1)``. This reduction is why the layer cannot support masking:
        it mixes every position of a sample.

        :param rms_tensor: RMS tensor produced with ``keepdims=True``, so it has
            the same rank as the input.
        :type rms_tensor: keras.KerasTensor

        :return: Aggregated statistics of shape ``(batch, 1)``.
        :rtype: keras.KerasTensor
        """
        aggregation_axes = list(range(1, len(rms_tensor.shape)))
        if aggregation_axes:
            rms_stats = ops.mean(
                rms_tensor, axis=aggregation_axes, keepdims=True
            )
        else:
            rms_stats = rms_tensor
        return ops.reshape(rms_stats, [-1, 1])

    def _reshape_scaling_factors(
        self,
        scaling_factors: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Reshape the ``Dense`` output so it broadcasts against the input.

        The target is ``(batch,) + param_shape``, with the batch size read at
        call time so a dynamic batch works.

        :param scaling_factors: Dense output of shape ``(batch, num_params)``.
        :type scaling_factors: keras.KerasTensor

        :return: The same values shaped ``(batch,) + param_shape``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(scaling_factors)[0]
        target_shape = [batch_size] + list(self._param_shape)
        return ops.reshape(scaling_factors, target_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Apply centering, RMS normalization and the input-adaptive band scale.

        :param inputs: Input tensor of any rank from 2 upward. Centering and
            normalization run along the axes given at construction.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the internal ``Dense``. That ``Dense`` has
            no training-dependent behaviour, so the two modes agree.
        :type training: Optional[bool]

        :return: Centered and normalized tensor, same shape and dtype as
            ``inputs``. Its RMS lies in ``[1 - max_band_width, 1]``, up to the
            epsilon floor.
        :rtype: keras.KerasTensor
        """
        original_dtype = inputs.dtype

        # Statistics dtype: float32 at minimum for numerical stability under
        # mixed precision, and float64 when the layer really is float64. A
        # hardcoded "float32" here would silently run the statistics in float32
        # under a float64 policy, and for a centered layer that destroys the
        # signal rather than rounding it. Measured on the float64 input
        # [[1e8+1, 1e8+2, 1e8+3, 1e8+4]]: float32 statistics center to exactly
        # [[0, 0, 0, 0]] and the layer returns all zeros, while float64
        # statistics center to [[-1.5, -0.5, 0.5, 1.5]] and the layer returns
        # [[-1.2745587, -0.4248529, 0.4248529, 1.2745587]]. This also feeds the
        # internal Dense at the policy's dtype, so a float64 policy no longer
        # promotes a float32 tensor against a float64 kernel.
        stat_dtype = statistics_dtype(original_dtype)
        inputs_fp32 = ops.cast(inputs, stat_dtype)

        # Step 1: Zero-centering.
        mean = ops.mean(inputs_fp32, axis=self.axis, keepdims=True)
        centered = inputs_fp32 - mean

        # Step 2: RMS of centered inputs.
        mean_square = ops.mean(
            ops.square(centered), axis=self.axis, keepdims=True
        )
        rms = ops.maximum(
            ops.sqrt(mean_square + self.epsilon), self.epsilon
        )

        # Step 3: Normalize.
        normalized = centered / rms

        # Step 4: Aggregate RMS -> log -> Dense -> sigmoid -> band scale.
        rms_stats = self._aggregate_rms_statistics(rms)
        log_rms = ops.log(rms_stats)
        band_logits = self.dense_layer(log_rms, training=training)

        band_activation = ops.sigmoid(5.0 * band_logits)
        scale_factors = (1.0 - self.max_band_width) + (
            self.max_band_width * band_activation
        )

        # Step 5: Reshape and apply adaptive scaling.
        scale_factors = self._reshape_scaling_factors(scale_factors)
        # DECISION plan-2026-08-25T195813-d5a035ab/D-005: this cast is NOT redundant
        # with the stat_dtype cast above and must not be deleted. dense_layer returns
        # its COMPUTE dtype, float16 under mixed_float16, and `fp32 * fp16` raises
        # InvalidArgumentError (measured). Do NOT hardcode "float32" here: stat_dtype
        # keeps a float64 policy in float64. See decisions.md D-005.
        output = normalized * ops.cast(scale_factors, stat_dtype)

        return ops.cast(output, original_dtype)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this layer.

        ``axis`` is returned exactly as it was passed, so the config does not
        depend on whether ``build()`` has run.

        :return: Serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "max_band_width": self.max_band_width,
                "axis": self.axis,
                "epsilon": self.epsilon,
                "band_initializer": keras.initializers.serialize(
                    self.band_initializer
                ),
                "band_regularizer": keras.regularizers.serialize(
                    self.band_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------

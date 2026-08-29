"""Bias-free (variance-only) batch normalization.

``BiasFreeBatchNorm`` divides by a fixed running variance and adds nothing. It
exists to keep **degree-1 homogeneity** at inference:

.. math::
    f(\\alpha x) = \\alpha f(x)

for any scalar ``alpha``. Bias-free denoisers need this property. Their
theoretical reading rests on the Miyasawa / Tweedie relation, and the network
residual equals a scaled score estimate only while the map is homogeneous. A
network built from homogeneous, bias-free operations generalizes across noise
levels; one additive offset, whether a ``beta`` or a subtracted mean, destroys
that.

Mathematical Formulation
------------------------
    Inference (``training=False``):

    .. math::
        y = \\gamma \\cdot \\frac{x}{\\sqrt{\\text{running\\_var} + \\varepsilon}}

    with ``running_var`` a fixed constant that does not depend on ``x``. The map
    is therefore exactly linear in ``x``, and scaling the input by ``alpha``
    scales the output by exactly ``alpha``.

    Training (``training=True``):

    .. math::
        y = \\gamma \\cdot \\frac{x}{\\sqrt{\\text{batch\\_var} + \\varepsilon}}

    where ``batch_var`` is the per-batch variance over all non-channel axes, and
    the non-trainable ``running_var`` is updated by an exponential moving average.
    This mirrors the moving-statistic update of Keras ``BatchNormalization``, but
    for the variance only.

Why variance-only, with no mean and no beta
-------------------------------------------
Measured in this repository at inference on the nonzero-mean input
``keras.random.normal((16, 32), seed=0) + 1.0``, statistics primed by one
``training=True`` pass. The metric is ``max|f(a x) - a f(x)| / max|a f(x)|``:

.. code-block:: text

    layer                                      a = 2       a = 3    a = 1000
    BiasFreeBatchNorm(use_scale=True)      0.000e+00   7.658e-08   1.176e-07
    BiasFreeBatchNorm(use_scale=False)     0.000e+00   7.658e-08   1.176e-07
    keras BatchNormalization (primed)      1.590e-03   2.121e-03   3.178e-03
    keras BatchNormalization(center=False) 1.590e-03   2.121e-03   3.178e-03
    keras BatchNormalization (UNPRIMED)    0.000e+00   7.654e-08   1.176e-07
    keras LayerNormalization               4.998e-01   6.665e-01   9.990e-01

The input construction is part of the claim. The BatchNorm rows scale with how
far the input mean is from zero, so their digits move with the draw: over seeds
0, 1 and 2 the ``a = 2`` entry ranges ``1.590e-03`` to ``2.029e-03``. What holds
at every draw is the STRUCTURE -- the two BiasFreeBatchNorm rows and the
UNPRIMED row at float32 zero, the two primed BatchNorm rows equal to each other,
and the LayerNormalization row at ``|1 - a| / a`` to three digits. It is not
exact there: LayerNorm's own epsilon puts ``a = 2`` at ``4.998e-01`` against
``5.0000e-01`` and ``a = 3`` at ``6.665e-01`` against ``6.6667e-01``, while
``a = 1000`` matches ``9.9900e-01`` to every printed digit.

Two of those rows are controls, and they identify the mechanism rather than just
observing an effect:

* ``BatchNormalization(center=False)`` removes ``beta`` but keeps the moving-mean
  subtraction, and it is non-homogeneous by exactly the same amount -- measured
  bit-identical to the primed default at all three seeds. So the mean
  subtraction alone is enough to break the property.
* An UNPRIMED ``BatchNormalization``, whose ``moving_mean`` is still at its zero
  initializer so the subtraction is a no-op, is as homogeneous as this layer.
  After one training pass its ``moving_mean`` becomes nonzero (measured
  ``max|moving_mean| = 1.3164e-02`` at seed 0, ``1.32e-02`` to ``1.54e-02`` over
  seeds 0-2) and the property is gone. So it is the NONZERO moving mean that
  breaks it, not the layer's structure.

Stock Keras ``BatchNormalization`` creates and subtracts a ``moving_mean``
whatever ``center`` is set to. This layer therefore creates NO ``moving_mean``
and NO ``beta`` at all. Mean subtraction and additive offsets are the exact
mechanisms that would reintroduce the bug, so they are absent from the weight
list rather than gated off at runtime.

``LayerNormalization`` and the RMS family in this package also fail the test, at
``9.990e-01`` for ``a = 1000``, but for an unrelated reason: they are scale
INVARIANT (degree 0), so ``f(a x) = f(x)`` and the ratio to ``a f(x)`` tends to
1. Do not read their number as the same defect.

Homogeneity is an INFERENCE-time property (IMPORTANT)
-----------------------------------------------------
Degree-1 homogeneity holds only at inference (``training=False``). During
training the layer uses the per-batch variance, and since
``var(alpha x) = alpha^2 var(x)`` the ``alpha`` cancels in the ratio
``alpha x / sqrt(alpha^2 var(x))``. The training-mode forward pass is therefore
scale-INVARIANT (degree 0), not degree-1 homogeneous. Measured with
``alpha = 7``: ``max|f(7x) - f(x)| = 2.861e-06`` in training mode, while
``max|f(7x) - 7 f(x)| = 3.948e+01``. This train/inference split is unavoidable
for the whole BatchNorm family and is acceptable here because both the
deployment path and the Miyasawa reading are inference-time. Always probe
homogeneity with ``training=False``.

Note on freshly-initialized weights: ``running_var`` starts at ``1``. Homogeneity
still holds at that point, since a constant scale is still constant, but the
value is not yet data-meaningful until the EMA has seen real batches.

References
----------
    - Mohan et al. (2020), "Robust and Interpretable Blind Image Denoising via
      Bias-Free Convolutional Neural Networks." (bias-free / scaling-generalization)
    - Miyasawa (1961); Tweedie / Robbins empirical-Bayes score relation.
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.norms.bias_free_batch_norm")
class BiasFreeBatchNorm(keras.layers.Layer):
    """Variance-only, fixed-statistic normalization; degree-1 homogeneous at inference.

    Normalizes by a fixed, EMA-tracked running variance, with an optional
    learnable per-channel scale ``gamma``, no mean subtraction and no additive
    ``beta``. At inference this is exactly
    ``y = gamma * x / sqrt(running_var + eps)``, which is linear in ``x`` and so
    satisfies ``f(alpha x) = alpha f(x)``. Measured relative error at
    ``alpha = 1000``: ``9.001e-08``, against ``5.512e-03`` for stock
    ``keras.layers.BatchNormalization``. The module docstring above has the full
    table and the two controls that pin down the mechanism.

    Weight list, measured: ``use_scale=True`` gives ``['gamma', 'running_var']``
    with ``gamma`` trainable and ``running_var`` not; ``use_scale=False`` gives
    ``['running_var']`` and no trainable weight at all. There is no
    ``moving_mean`` and no ``beta`` in either case.

    .. important::
        Homogeneity holds at INFERENCE ONLY (``training=False``). During training
        the layer uses the per-batch variance and is scale-INVARIANT (degree 0),
        which is unavoidable for the BatchNorm family. Probe the homogeneity
        property with ``training=False``.

    .. note::
        **No masking support, and that is the intended setting.**
        ``supports_masking`` is left ``False`` even though stock
        ``keras.layers.BatchNormalization`` sets it. In training the batch
        variance is reduced over every non-channel axis, so perturbing one
        ``(sample, token)`` slot moves other tokens by up to ``1.761e+00`` and
        other SAMPLES by up to ``2.331e+00``. Measured on a ``(3, 5, 8)`` input;
        the inference path leaks exactly ``0.0``. A mask propagated through this
        layer would describe outputs that were computed from the padding it
        marks.

    **Architecture Overview:**

    .. code-block:: text

                    inputs: x   (B, ..., C)
                                │
                                ▼
        ┌──────────────────────────────────────────────────────┐
        │ stat_dtype   = result_type(x.dtype, float32)         │
        │ inputs_fp32  = cast(x, stat_dtype)                   │
        └───────────┬─────────────────────────┬────────────────┘
                    │ training=True           │ training=False/None
                    ▼                         ▼
        ┌─────────────────────────────┐  ┌────────────────────────────┐
        │ batch_var = var(            │  │ var_for_norm =             │
        │   inputs_fp32, over         │  │   running_var, a FIXED     │
        │   non-channel axes)         │  │   constant       (C,)      │
        │ running_var <- EMA          │  │                            │
        │ var_for_norm =              │  │                            │
        │   batch_var    (C,)         │  │                            │
        └─────────────┬───────────────┘  └─────────────┬──────────────┘
                      │                                │
                      ▼                                ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ var_b  = reshape(var_for_norm, broadcast_shape)             │
        │ output = inputs_fp32 / sqrt(var_b + epsilon)                │
        │ NO mean subtraction anywhere, and NO beta term              │
        └───────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ output = output * gamma_b     (optional; only when          │
        │                                use_scale is True)           │
        └───────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ cast(output, original_dtype)                                │
        └───────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
                    output: (B, ..., C)   SAME shape as x

    :param axis: Channel axis that keeps its own statistics and scale. The
        reduction runs over every OTHER axis. Defaults to ``-1``, channels-last.
    :type axis: int
    :param epsilon: Constant added to the variance for numerical stability. Must
        be strictly positive. Defaults to 1e-6.
    :type epsilon: float
    :param momentum: EMA momentum for the ``running_var`` update during training:
        ``running_var <- momentum * running_var + (1 - momentum) * batch_var``.
        Must be in ``[0, 1]``. Defaults to 0.99.
    :type momentum: float
    :param use_scale: Whether to add a learnable per-channel scale ``gamma``.
        When ``False`` the layer is a pure fixed-statistic divisor with no
        trainable weight. Defaults to ``True``.
    :type use_scale: bool
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar axis: The configured channel axis, stored exactly as passed.
    :vartype axis: int
    :ivar epsilon: The configured epsilon, stored exactly as passed.
    :vartype epsilon: float
    :ivar momentum: The configured EMA momentum, stored exactly as passed.
    :vartype momentum: float
    :ivar use_scale: The configured flag, stored exactly as passed.
    :vartype use_scale: bool
    :ivar running_var: The non-trainable ``(C,)`` inference statistic, or
        ``None`` until ``build()`` runs.
    :vartype running_var: Optional[keras.Variable]
    :ivar gamma: The trainable ``(C,)`` scale, or ``None`` when ``use_scale`` is
        ``False`` or ``build()`` has not run.
    :vartype gamma: Optional[keras.Variable]

    :raises ValueError: If ``epsilon`` is not strictly positive. Measured:
        ``epsilon=0.0`` raises ``ValueError: epsilon must be positive, got 0.0``.
        Raised in ``__init__``.
    :raises ValueError: If ``momentum`` is outside ``[0, 1]``. Raised in
        ``__init__``.
    :raises TypeError: If ``axis`` is not an ``int``. Measured: ``axis=-1.0``
        raises ``TypeError: axis must be an int, got <class 'float'>``. Raised in
        ``__init__``.
    :raises ValueError: If the channel dimension along ``axis`` is dynamic.
        Raised in ``build()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import BiasFreeBatchNorm

        x = keras.random.normal((8, 32, 32, 16))
        y = BiasFreeBatchNorm()(x, training=False)
    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-6,
        momentum: float = 0.99,
        use_scale: bool = True,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and store it.

        No weight is created here. ``running_var`` and ``gamma`` need the channel
        dimension, so they are created in ``build()``.

        :param axis: Channel axis that keeps its own statistics and scale.
        :type axis: int
        :param epsilon: Constant added to the variance. Must be strictly positive.
        :type epsilon: float
        :param momentum: EMA momentum, in ``[0, 1]``.
        :type momentum: float
        :param use_scale: Whether to add a learnable per-channel ``gamma``.
        :type use_scale: bool
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``epsilon`` is not strictly positive.
        :raises ValueError: If ``momentum`` is outside ``[0, 1]``.
        :raises TypeError: If ``axis`` is not an ``int``.
        """
        super().__init__(**kwargs)

        # Validate configuration early.
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        if not isinstance(axis, int):
            raise TypeError(f"axis must be an int, got {type(axis)}")

        # Store ALL configuration parameters - required for get_config().
        self.axis = axis
        self.epsilon = epsilon
        self.momentum = momentum
        self.use_scale = use_scale

        # Weights created in build().
        self.running_var = None
        self.gamma = None

        logger.debug(
            f"Initialized BiasFreeBatchNorm with axis={axis}, epsilon={epsilon}, "
            f"momentum={momentum}, use_scale={use_scale}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the fixed-statistic ``running_var`` and the optional ``gamma``.

        :param input_shape: Shape tuple; the channel axis must be static.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the channel dimension along ``axis`` is dynamic.
            Measured: ``build((None, None))`` raises ``ValueError:
            BiasFreeBatchNorm requires a static channel dimension along axis=-1,
            but input_shape (None, None) has a dynamic dimension there.``
        """
        if self.built:
            return

        ndims = len(input_shape)
        channel_axis = self.axis % ndims
        dim = input_shape[channel_axis]

        if dim is None:
            raise ValueError(
                f"BiasFreeBatchNorm requires a static channel dimension along "
                f"axis={self.axis}, but input_shape {input_shape} has a dynamic "
                f"dimension there."
            )

        # DECISION plan_2026-07-01_8054f023/D-001: variance-only by construction.
        # Create ONLY a non-trainable running_var (the fixed inference statistic,
        # EMA-updated in training) and an optional trainable gamma. Do NOT add a
        # moving_mean or a beta here: stock Keras BatchNormalization subtracts a
        # moving_mean whatever `center` says, and that subtraction is what makes it
        # NON-homogeneous. Re-measured here at inference on a (16, 32) input with a
        # nonzero mean, statistics primed by one training pass, f(1000 x) vs 1000 f(x):
        # stock BatchNorm rel err 5.512e-03, BatchNormalization(center=False) the same to
        # every digit, an UNPRIMED BatchNorm (moving_mean still 0) 8.777e-08, and this
        # layer 9.001e-08. Adding either weight would reintroduce the audited bug.
        # The originating plan directory is gone; this comment is the record.
        # DECISION plan-2026-08-18T123346-c3c4a681/D-002: moving statistics live
        # in the VARIABLE dtype, never the compute dtype, and are read with autocast
        # DISABLED. Under `mixed_float16` the variable dtype is float32 while the compute
        # dtype is float16, so without `autocast=False` a `self.running_var` read inside
        # `call()` comes back float16, which both loses EMA precision and mixes dtypes in
        # the update. Do NOT drop `dtype=`/`autocast=False`, and do NOT move the statistics
        # to the compute dtype to line the dtypes up - that turns the mixed_float16 path
        # into a float16 accumulator. Mirrors keras.layers.BatchNormalization.build.
        # See decisions.md D-002.
        self.running_var = self.add_weight(
            name="running_var",
            shape=(dim,),
            initializer="ones",
            trainable=False,
            dtype=self.variable_dtype,
            autocast=False,
        )

        if self.use_scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(dim,),
                initializer="ones",
                trainable=True,
                dtype=self.variable_dtype,
                autocast=False,
            )

        logger.debug(f"Built BiasFreeBatchNorm weights for channel dim {dim}")

        # Always call parent build at the end.
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply variance-only normalization.

        Measured after one training pass at ``momentum=0.9``: the EMA update
        matches ``0.9 * running_var + 0.1 * batch_var`` to ``0.000e+00``, and the
        inference output matches ``x / sqrt(running_var + eps)`` to ``9.537e-07``.

        :param inputs: Input tensor. Statistics and scale are kept along ``axis``;
            the reduction runs over every other axis.
        :type inputs: keras.KerasTensor
        :param training: If ``True``, use the per-batch variance and EMA-update
            ``running_var``. If ``False`` or ``None``, use the fixed
            ``running_var``, which is the homogeneous inference path.
        :type training: Optional[bool]

        :return: Normalized tensor with the same shape and dtype as ``inputs``.
        :rtype: keras.KerasTensor
        """
        original_dtype = inputs.dtype

        # Statistics dtype: float32 at minimum (numerical stability under
        # mixed precision), but float64 when the layer really is float64 -
        # a hardcoded "float32" here made the layer raise a dtype TypeError
        # under any non-float32 policy. See D-002 in build().
        stat_dtype = keras.backend.result_type(original_dtype, "float32")
        inputs_fp32 = ops.cast(inputs, stat_dtype)

        ndims = len(inputs.shape)
        channel_axis = self.axis % ndims
        reduction_axes = [i for i in range(ndims) if i != channel_axis]

        running_var = ops.cast(self.running_var, stat_dtype)

        if training:
            # Per-batch variance over all non-channel axes (shape: (C,)).
            batch_var = ops.var(inputs_fp32, axis=reduction_axes, keepdims=False)

            # EMA update of the fixed inference statistic (VARIANCE ONLY, mirroring
            # keras BatchNormalization's moving-stat update - but no moving_mean).
            # `assign` casts back down to the variable dtype.
            self.running_var.assign(
                self.momentum * running_var
                + (1.0 - self.momentum) * batch_var
            )
            var_for_norm = batch_var
        else:
            # Inference: fixed constant -> output is linear in inputs -> degree-1
            # homogeneous.
            var_for_norm = running_var

        # Broadcast the per-channel variance (and gamma) to the input rank. The
        # channel dim is static (enforced in build), so this reshape is safe.
        broadcast_shape = [1] * ndims
        broadcast_shape[channel_axis] = var_for_norm.shape[0]

        var_b = ops.reshape(var_for_norm, broadcast_shape)

        # NO mean subtraction anywhere - this is the whole point.
        output = inputs_fp32 / ops.sqrt(var_b + self.epsilon)

        if self.use_scale:
            gamma_b = ops.reshape(ops.cast(self.gamma, stat_dtype), broadcast_shape)
            output = output * gamma_b

        return ops.cast(output, original_dtype)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``input_shape`` unchanged. The layer is shape-preserving.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape as the input.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration for serialization.

        ``running_var`` and ``gamma`` are weights restored by Keras, so they are
        not included here.

        :return: A dictionary carrying ``axis``, ``epsilon``, ``momentum`` and
            ``use_scale`` on top of the base ``Layer`` config.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "use_scale": self.use_scale,
        })
        return config

# ---------------------------------------------------------------------

"""
Bias-Free (Variance-Only) Batch Normalization Layer.

This module implements ``BiasFreeBatchNorm``, a variance-only, fixed-statistic
normalization designed to preserve **degree-1 homogeneity** at inference:

.. math::
    f(\\alpha x) = \\alpha f(x)

for any scalar ``alpha``. This property is required by bias-free denoisers whose
theoretical interpretation rests on the Miyasawa / Tweedie relation (the network
residual equals a scaled score estimate only when the map is homogeneous). A
network built from strictly homogeneous, bias-free operations generalizes across
noise levels and admits the score-based interpretation; a single additive offset
(a ``beta``, or a subtracted running/batch mean) destroys that property.

Mathematical Formulation
------------------------
    Inference (``training=False``):

    .. math::
        y = \\gamma \\cdot \\frac{x}{\\sqrt{\\text{running\\_var} + \\varepsilon}}

    with ``running_var`` a FIXED (non-input-derived) constant. Because the
    denominator does not depend on ``x``, the map is exactly linear in ``x`` and
    therefore degree-1 homogeneous: scaling the input by ``alpha`` scales the
    output by exactly ``alpha``.

    Training (``training=True``):

    .. math::
        y = \\gamma \\cdot \\frac{x}{\\sqrt{\\text{batch\\_var} + \\varepsilon}}

    where ``batch_var`` is the per-batch variance over all non-channel axes, and
    the non-trainable ``running_var`` is updated by an exponential moving average
    (EMA), mirroring the moving-statistic update of Keras ``BatchNormalization``
    but for VARIANCE ONLY.

Why variance-only (no mean, no beta)
------------------------------------
Empirically (see ``findings/norm-homogeneity-mechanics.md``): plain
``LayerNormalization(center=False)``, the repo RMS family, and stock
``BatchNormalization(center=False)`` are all NON-homogeneous. Stock Keras
``BatchNormalization`` unconditionally creates and subtracts a ``moving_mean``
regardless of ``center`` — that additive offset breaks ``f(alpha x)=alpha f(x)``
(rel err in the hundreds). This layer therefore creates **NO** ``moving_mean``
and **NO** ``beta`` at all: mean-subtraction and additive offsets are the exact
mechanisms that would reintroduce the bug, so they are structurally absent rather
than merely gated off.

Homogeneity is an INFERENCE-time property (IMPORTANT)
-----------------------------------------------------
Degree-1 homogeneity holds only at inference (``training=False``). During training
the layer uses the per-batch variance, and since ``var(alpha x) = alpha^2 var(x)``
the ``alpha`` factor cancels in the ratio ``alpha x / sqrt(alpha^2 var(x))``,
making the training-mode forward pass scale-INVARIANT (degree-0), not degree-1
homogeneous. This train/inference split is architecturally unavoidable for the
entire BatchNorm family and is acceptable here because both the deployment path
and the Miyasawa interpretation are inference-time. Always probe homogeneity with
``training=False``.

Note on freshly-initialized weights: ``running_var`` starts at ``1``. Homogeneity
still holds trivially at that point (a constant scale is still constant), but the
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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BiasFreeBatchNorm(keras.layers.Layer):
    """
    Variance-only, fixed-statistic normalization; degree-1 homogeneous at inference.

    Normalizes inputs by a fixed (EMA-tracked) running variance with an optional
    learnable per-channel scale ``gamma`` and NO mean subtraction and NO additive
    ``beta``. At inference this is exactly ``y = gamma * x / sqrt(running_var + eps)``,
    which is linear in ``x`` and therefore satisfies ``f(alpha x) = alpha f(x)``.

    .. important::
        Homogeneity holds at INFERENCE ONLY (``training=False``). During training
        the layer uses the per-batch variance and is scale-INVARIANT (degree-0),
        which is architecturally unavoidable for the BatchNorm family. Probe the
        homogeneity property with ``training=False``.

    Unlike stock ``keras.layers.BatchNormalization``, this layer creates neither a
    ``moving_mean`` nor a ``beta`` weight — mean-subtraction and additive offsets
    are exactly what would break homogeneity, so they are structurally absent. See
    the module docstring and ``findings/norm-homogeneity-mechanics.md`` for the
    empirical rationale.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., C)   [normalized channel = axis]
                │
                ▼
        ┌────────────────────────────────────────────┐
        │ training=True : var = batch var over        │
        │                 non-channel axes;           │
        │                 EMA-update running_var       │
        │ training=False: var = running_var (constant) │
        └────────────────────────┬────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────┐
        │ y = x / sqrt(var + eps)   (NO mean subtract) │
        └────────────────────────┬────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────┐
        │ y = y * gamma   (if use_scale; NO beta add)  │
        └────────────────────────┬────────────────────┘
                                 │
                                 ▼
        Output: (batch, ..., C)

    :param axis: Channel axis over which per-channel statistics/scale are kept.
        Reduction happens over all OTHER axes. Default ``-1`` (NHWC channels-last).
    :type axis: int
    :param epsilon: Small constant added to the variance for numerical stability.
        Must be positive.
    :type epsilon: float
    :param momentum: EMA momentum for the ``running_var`` update during training.
        ``running_var <- momentum * running_var + (1 - momentum) * batch_var``.
        Must be in ``[0, 1]``.
    :type momentum: float
    :param use_scale: Whether to include a learnable per-channel scale ``gamma``.
        When ``False`` the layer is a pure fixed-statistic divisor.
    :type use_scale: bool
    :param kwargs: Additional keyword arguments passed to the parent ``Layer``.
    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-6,
        momentum: float = 0.99,
        use_scale: bool = True,
        **kwargs: Any
    ) -> None:
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
        """
        Create the fixed-statistic ``running_var`` and optional ``gamma`` weights.

        :param input_shape: Shape tuple; the channel axis must be static.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the channel dimension along ``axis`` is dynamic.
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
        # moving_mean or a beta here: stock Keras BatchNormalization's unconditional
        # moving_mean subtraction (and any additive beta) is exactly what makes it
        # NON-homogeneous (findings/norm-homogeneity-mechanics.md, rel err 500+).
        # Adding either weight would reintroduce the audited bug. See decisions.md D-001.
        self.running_var = self.add_weight(
            name="running_var",
            shape=(dim,),
            initializer="ones",
            trainable=False,
        )

        if self.use_scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(dim,),
                initializer="ones",
                trainable=True,
            )

        logger.debug(f"Built BiasFreeBatchNorm weights for channel dim {dim}")

        # Always call parent build at the end.
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply variance-only normalization.

        :param inputs: Input tensor. Statistics/scale are kept along ``axis``;
            reduction happens over every other axis.
        :type inputs: keras.KerasTensor
        :param training: If ``True``, use the per-batch variance and EMA-update
            ``running_var``. If ``False``/``None``, use the fixed ``running_var``
            (the homogeneous inference path).
        :type training: Optional[bool]
        :return: Normalized tensor with the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        original_dtype = inputs.dtype

        # Compute in float32 for numerical stability under mixed precision.
        inputs_fp32 = ops.cast(inputs, "float32")

        ndims = len(inputs.shape)
        channel_axis = self.axis % ndims
        reduction_axes = [i for i in range(ndims) if i != channel_axis]

        if training:
            # Per-batch variance over all non-channel axes (shape: (C,)).
            batch_var = ops.var(inputs_fp32, axis=reduction_axes, keepdims=False)

            # EMA update of the fixed inference statistic (VARIANCE ONLY, mirroring
            # keras BatchNormalization's moving-stat update - but no moving_mean).
            self.running_var.assign(
                self.momentum * self.running_var
                + (1.0 - self.momentum) * batch_var
            )
            var_for_norm = batch_var
        else:
            # Inference: fixed constant -> output is linear in inputs -> degree-1
            # homogeneous.
            var_for_norm = self.running_var

        # Broadcast the per-channel variance (and gamma) to the input rank. The
        # channel dim is static (enforced in build), so this reshape is safe.
        broadcast_shape = [1] * ndims
        broadcast_shape[channel_axis] = var_for_norm.shape[0]

        var_b = ops.reshape(var_for_norm, broadcast_shape)

        # NO mean subtraction anywhere - this is the whole point.
        output = inputs_fp32 / ops.sqrt(var_b + self.epsilon)

        if self.use_scale:
            gamma_b = ops.reshape(self.gamma, broadcast_shape)
            output = output * gamma_b

        return ops.cast(output, original_dtype)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (identity for a normalization layer).

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Same shape as the input.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor configuration for serialization.

        ``running_var`` and ``gamma`` are weights restored by Keras, so they are
        not included here.

        :return: Dictionary of constructor arguments.
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

"""
One normalization layer covering both RevIN and z-score standard scaling.

`UnifiedScaler` normalizes along any axis you pick. Set ``axis=1`` and you get
Reversible Instance Normalization for time series. Set ``axis=-1`` and you get
per-feature standard scaling. One layer, one API, both jobs.

**What it does:**

- Normalizes along any axis or combination of axes.
- Optionally applies a learnable scale ``γ`` and shift ``β``.
- Replaces NaN entries with a configurable value before computing statistics.
- Optionally keeps the statistics as non-trainable weights so they survive
  serialization.
- Inverts itself. `inverse_transform()` puts predictions back on the original
  scale; `denormalize()` is an alias for it.
- `reset_stats()` clears the stored statistics, `get_stats()` reads them back.

**Mathematical Foundation:**

For an input tensor `x` of shape `(batch, ..., features)`:

1. **Statistics:**
   - `μ = mean(x, axis=axis, keepdims=True)`
   - `σ = sqrt(var(x, axis=axis, keepdims=True) + eps)`

2. **Normalization:**
   - `x_norm = (x - μ) / σ`

3. **Optional affine transform** (if `affine=True`):
   - `output = γ ⊙ x_norm + β`

4. **Inverse:**
   - If affine: `x = (output - β) / γ_safe`
   - `x_original = x * σ + μ`

`γ_safe` is not plain `γ`. Training can drive `γ` to 0, and dividing by 0
returns inf or NaN. So the divide uses `γ` with its magnitude floored at `eps`
and its sign kept, treating `γ == 0` as positive. See
`UnifiedScaler.inverse_transform`.

Where:
- `⊙` is element-wise multiplication.
- `γ`, `β` are learnable parameters.
- `μ`, `σ` are computed over the axes named by `axis`.

**Use Cases:**

- **Time series forecasting:** instance normalization (`axis=1`) against
  distribution shift between series.
- **Feature preprocessing:** standard normalization (`axis=-1`) across
  multivariate features.
- **Online learning:** persistent statistics for streaming data.
- **Interpretability:** invert predictions back to the original data scale.

**References:**
    - Kim et al., "Reversible Instance Normalization for Accurate Time-Series
      Forecasting against Distribution Shift", ICLR 2022.
      https://arxiv.org/abs/2107.03445
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any, Dict
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.statistics.scaler")
class UnifiedScaler(keras.layers.Layer):
    """
    Normalization layer that covers both RevIN and standard z-score scaling.

    Each forward pass computes ``mean`` and ``std`` over the axes named by
    ``axis``, then returns ``(x - mean) / std``. Turn ``affine`` on and a
    learnable scale and shift are applied on top. NaN entries are replaced with
    ``nan_replacement`` before any statistic is computed.

    The statistics of the most recent call are kept on the instance, so
    ``inverse_transform`` can undo that exact call. Set ``store_stats`` and the
    batch-averaged statistics are also written into non-trainable weights, which
    means they survive saving and loading.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────┐
        │ inputs  (batch, ..., features)              │
        └──────────────────────┬──────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │ x = where(isnan(inputs), nan_replacement)   │
        └──────────────────────┬──────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │ mean = mean(x, axis)                        │
        │ std  = sqrt(mean((x - mean)^2) + eps)       │
        └──────────┬───────────────────────┬──────────┘
                   │                       │ (optional)
                   │                       ▼
                   │            ┌─────────────────────┐
                   │            │ stored_mean.assign  │
                   │            │ stored_std.assign   │
                   │            │ if store_stats      │
                   │            └─────────────────────┘
                   ▼
        ┌─────────────────────────────────────────────┐
        │ x_norm = (x - mean) / std                   │
        └──────────────────────┬──────────────────────┘
                               ▼
                           affine ?
                   ┌───────────┴───────────┐
                 False                   True
                   │                       │
                   ▼                       ▼
        ┌─────────────────────┐ ┌─────────────────────┐
        │ return x_norm       │ │ return              │
        │ (batch, ..., feat)  │ │  x_norm * a_weight  │
        │                     │ │  + a_bias           │
        └─────────────────────┘ └─────────────────────┘

    ``a_weight`` and ``a_bias`` are ``affine_weight`` and ``affine_bias``, the
    layer's only trainable weights. ``stored_mean`` and ``stored_std`` are
    non-trainable. Both output branches keep the input shape.

    **Inverse Path (inverse_transform):**

    .. code-block:: text

        ┌─────────────────────────────────────────────┐
        │ scaled_inputs                               │
        └──────────────────────┬──────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │ sign       = where(gamma < 0, -1.0, 1.0)    │
        │ gamma_safe = sign * max(abs(gamma), eps)    │
        │ x = (x - affine_bias) / gamma_safe          │
        │ (optional: only if affine)                  │
        └──────────────────────┬──────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │ x = x * std + mean                          │
        │ (std, mean) = the explicit pair if given,   │
        │              else (_last_std, _last_mean)   │
        └──────────────────────┬──────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │ original scale                              │
        └─────────────────────────────────────────────┘

    ``gamma`` is ``affine_weight``. ``_last_mean`` and ``_last_std`` come from
    the most recent ``call``, not from the stored weights, so the layer must be
    called before it can be inverted with no arguments. They are also per-sample
    tensors scoped to the trace that produced them, so after ``model.fit`` or
    ``model.predict`` they are gone and the no-argument path raises a
    ``RuntimeError`` explaining that. A caller who kept the statistics passes
    them back as ``inverse_transform(scaled, mean=m, std=s)``, which is the only
    exact way to invert after a traced call. Both documented personas need it:

    * **RevIN** (``axis=1``, ``affine=True``) -- the statistics are per-sample
      and per-feature.
    * **StandardScaler** (``axis=-1``, ``store_stats=True``) -- the statistics
      are reduced over the feature axis, still one set per row.

    Neither persona can invert with ``get_stats()``: those summaries average
    over the BATCH axis, which neither of them reduces over. Measured, they cost
    a mean absolute error of 1.02 and 1.40 respectively and up to 136x relative
    error. They are exact only at ``axis=0``.

    :param num_features: Number of features/channels. Defaults to ``None``,
        which infers it from the last input dimension.
    :type num_features: int | None
    :param axis: Axis or axes the statistics are computed over. Defaults to -1.
    :type axis: int | tuple[int, ...]
    :param eps: Small value added to the variance and used as the inverse
        divide floor. Defaults to 1e-5.
    :type eps: float
    :param affine: Whether to apply the learnable affine transform.
        Defaults to ``False``.
    :type affine: bool
    :param affine_weight_initializer: Initializer for the scale gamma.
        Defaults to "ones".
    :type affine_weight_initializer: str | keras.initializers.Initializer
    :param affine_bias_initializer: Initializer for the shift beta.
        Defaults to "zeros".
    :type affine_bias_initializer: str | keras.initializers.Initializer
    :param nan_replacement: Value that replaces NaN entries. Defaults to 0.0.
    :type nan_replacement: float
    :param store_stats: Whether to keep the statistics as persistent
        non-trainable weights. Defaults to ``False``.
    :type store_stats: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :ivar affine_weight: Trainable scale gamma, or ``None`` if ``affine`` is off.
    :vartype affine_weight: keras.Variable | None
    :ivar affine_bias: Trainable shift beta, or ``None`` if ``affine`` is off.
    :vartype affine_bias: keras.Variable | None
    :ivar stored_mean: Non-trainable batch-averaged mean, or ``None`` if
        ``store_stats`` is off.
    :vartype stored_mean: keras.Variable | None
    :ivar stored_std: Non-trainable batch-averaged standard deviation, or
        ``None`` if ``store_stats`` is off.
    :vartype stored_std: keras.Variable | None

    :raises ValueError: If ``num_features`` is not positive, or ``eps`` is not
        positive.

    Input shape:
        At least 2D tensor of shape ``(batch, ..., features)``.

    Output shape:
        Same shape as the input.

    Example:
        >>> scaler = UnifiedScaler(axis=1, affine=True, store_stats=True)
        >>> y = scaler(x)
        >>> x_again = scaler.inverse_transform(y)

    Note:
        ``training`` is accepted for API consistency. The layer behaves the same
        in both modes, and the stored statistics refresh on every call.
    """

    def __init__(
            self,
            num_features: Optional[int] = None,
            axis: Union[int, Tuple[int, ...]] = -1,
            eps: float = 1e-5,
            affine: bool = False,
            affine_weight_initializer: Union[str, keras.initializers.Initializer] = "ones",
            affine_bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            nan_replacement: float = 0.0,
            store_stats: bool = False,
            **kwargs: Any
    ) -> None:
        """Initialize the UnifiedScaler layer.

        Weights are not created here. ``build`` creates them, once the input
        shape is known. See the class docstring for the parameters.

        :raises ValueError: If ``num_features`` is not positive, or ``eps`` is
            not positive.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if num_features is not None and num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        # Store configuration
        self.num_features = num_features
        self.axis = axis if isinstance(axis, (tuple, list)) else (axis,)
        self.eps = eps
        self.affine = affine
        self.affine_weight_initializer = keras.initializers.get(affine_weight_initializer)
        self.affine_bias_initializer = keras.initializers.get(affine_bias_initializer)
        self.nan_replacement = nan_replacement
        self.store_stats = store_stats

        # Weight attributes (created in build)
        self.affine_weight = None
        self.affine_bias = None
        self.stored_mean = None
        self.stored_std = None

        # Statistics from last forward pass (for inverse transform)
        self._last_mean = None
        self._last_std = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights and validate the input shape.

        Creates ``affine_weight`` and ``affine_bias`` when ``affine`` is set,
        and ``stored_mean`` and ``stored_std`` when ``store_stats`` is set.
        Neither pair exists otherwise.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        :raises ValueError: If the input is less than 2D, if ``num_features``
            is ``None`` and the last dimension is undefined, or if ``axis``
            is out of range for the input rank.
        """
        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(
                f"UnifiedScaler expects at least 2D input, got shape {input_shape}"
            )

        # Infer num_features if not provided
        if self.num_features is None:
            self._inferred_num_features = input_shape[-1]
            if self._inferred_num_features is None:
                raise ValueError(
                    f"Last dimension of input must be defined when "
                    f"num_features=None, got input_shape={tuple(input_shape)}"
                )
        else:
            self._inferred_num_features = self.num_features

        # Validate axis configuration
        rank = len(input_shape)
        normalized_axes = tuple(
            ax if ax >= 0 else rank + ax for ax in self.axis
        )

        if any(ax >= rank or ax < 0 for ax in normalized_axes):
            raise ValueError(
                f"Invalid axis {self.axis} for input with rank {rank}"
            )

        # Calculate shape for statistics (keepdims after reduction)
        self._stats_shape = list(input_shape)
        for ax in sorted(normalized_axes, reverse=True):
            self._stats_shape[ax] = 1
        self._stats_shape = tuple(self._stats_shape)

        # Calculate shape for affine parameters (reduced dimensions)
        # Affine parameters should match the feature dimensions, not the normalized axes
        self._affine_shape = [
            dim for i, dim in enumerate(input_shape)
            if i not in normalized_axes
        ]
        # Handle case where all dimensions are normalized (unlikely but possible)
        if not self._affine_shape:
            self._affine_shape = [1]
        else:
            # Remove batch dimension
            self._affine_shape = self._affine_shape[1:]
        self._affine_shape = tuple(self._affine_shape)

        # Create affine parameters if enabled
        if self.affine:
            # For most common case (3D input, axis=1), shape is (num_features,)
            # For general case, shape matches non-normalized dimensions
            affine_param_shape = (self._inferred_num_features,) if input_shape[
                                                                       -1] == self._inferred_num_features else self._affine_shape

            self.affine_weight = self.add_weight(
                name="affine_weight",
                shape=affine_param_shape,
                initializer=self.affine_weight_initializer,
                trainable=True,
            )

            self.affine_bias = self.add_weight(
                name="affine_bias",
                shape=affine_param_shape,
                initializer=self.affine_bias_initializer,
                trainable=True,
            )

        # Create persistent statistics storage if enabled
        if self.store_stats:
            # Remove batch dimension for weight shape
            weight_shape = tuple(self._stats_shape[1:])

            self.stored_mean = self.add_weight(
                name="stored_mean",
                shape=weight_shape,
                initializer="zeros",
                trainable=False,
            )

            self.stored_std = self.add_weight(
                name="stored_std",
                shape=weight_shape,
                initializer="ones",
                trainable=False,
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply normalization to the inputs.

        Also records the statistics of this call on the instance, so
        ``inverse_transform`` can undo it.

        :param inputs: Input tensor to normalize.
        :type inputs: keras.KerasTensor
        :param training: Boolean for training mode. The layer behaves the same
            either way.
        :type training: bool | None
        :return: Normalized tensor, same shape as the input.
        :rtype: keras.KerasTensor
        """
        # Replace NaN values with specified replacement value
        x = ops.where(ops.isnan(inputs), self.nan_replacement, inputs)

        # Compute mean and standard deviation along specified axes
        mean = ops.mean(x, axis=self.axis, keepdims=True)

        # Compute variance using the stable two-pass formula
        variance = ops.mean(ops.square(x - mean), axis=self.axis, keepdims=True)
        # There is no ops.maximum(std, eps) floor here because it could never
        # fire. For any sane eps (1e-5, say), sqrt(variance + eps) >= sqrt(eps),
        # which is already larger than eps.
        std = ops.sqrt(variance + self.eps)

        # Apply z-score normalization
        x_norm = (x - mean) / std

        # Store current statistics for inverse transform
        self._last_mean = mean
        self._last_std = std

        # Update the persistent statistics.
        #
        # Assigning to a non-trainable variable inside call() is the standard
        # Keras 3 idiom, not an anti-pattern. keras.layers.BatchNormalization
        # does the same thing: self.moving_mean.assign(...) and
        # self.moving_variance.assign(...) in its own call(), at
        # layers/normalization/batch_normalization.py:257,260.
        #
        # There is no `if training:` gate, and that is on purpose. This layer
        # promises to refresh the stored stats on EVERY forward pass, inference
        # calls with training=None included. Do NOT add a training gate. It would
        # break test_scaler_stored_statistics, which asserts the stored stats
        # track plain inference calls. BatchNorm does gate on training; this
        # layer differs from it here.
        #
        # The gate is self.built alone. That skips the update during the symbolic
        # functional-API shape-inference pass, where the weights do not exist yet.
        #
        # This state mutation is graph-safe under the TF backend in every regime
        # that was measured: eager, tf.function, model.fit, and
        # jit_compile=True. XLA is not an exception. An earlier version of this
        # comment claimed the assign was "NOT supported under TF
        # jit_compile=True"; that was measured false and removed. It also matters
        # in practice, because model.fit compiles with jit_compile="auto" and so
        # reaches the XLA path by default.
        #
        # What XLA does require is a fully defined variable shape. stored_mean
        # and stored_std have one. A variable with any unknown dimension fails
        # tf2xla conversion on AssignVariableOp, measured. So do not make any
        # variable assigned here dynamically shaped.
        #
        # Behaviour on the stateless JAX backend is UNVERIFIED. This repo runs on
        # TF, so the claim was not tested either way and is not made here.
        #
        # The XLA half of this is pinned by TestTheStoredStatsAssignWorksUnderXla
        # in tests/test_layers/test_statistics/test_the_package_is_v2_compliant.py.
        if self.store_stats and self.built:
            # Average statistics across batch dimension for storage
            batch_mean = ops.mean(mean, axis=0)
            batch_std = ops.mean(std, axis=0)

            self.stored_mean.assign(batch_mean)
            self.stored_std.assign(batch_std)

        # Apply learnable affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm

    def _stats_are_usable(self) -> bool:
        """Report whether the per-call statistics can still be read.

        A tensor created inside a traced ``call`` is scoped to that trace.
        Touching it afterwards raises ``TypeError`` from whatever operation
        happens to reach it first, which tells the caller nothing. Probing it
        here turns that into one legible error.

        :return: True when ``_last_mean`` and ``_last_std`` can be used.
        :rtype: bool
        """
        try:
            ops.shape(self._last_mean)
            ops.shape(self._last_std)
        except Exception:
            return False
        return True

    def inverse_transform(
        self,
        scaled_inputs: keras.KerasTensor,
        mean: Optional[keras.KerasTensor] = None,
        std: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Transform normalized data back to the original scale.

        With no ``mean``/``std`` this uses the statistics of the most recent
        ``call``, so the layer has to have been called eagerly first. Those
        statistics are per-sample tensors scoped to the trace that made them, so
        after ``model.fit`` or ``model.predict`` there is nothing left to invert
        with and this raises a ``RuntimeError`` saying so.

        Supply BOTH ``mean`` and ``std`` to invert with statistics the caller
        kept, which is the only exact way to invert after a traced call. They
        must be the reduced statistics of the SAME rows being inverted, i.e. what
        a prior eager ``call`` produced, broadcastable against ``scaled_inputs``.
        Both documented personas need this:

        * RevIN (``axis=1``, ``affine=True``) -- per-sample, per-feature
          statistics of shape ``(batch, 1, features)``.
        * StandardScaler (``axis=-1``, ``store_stats=True``) -- statistics
          reduced over the feature axis, shape ``(batch, steps, 1)``.

        Do NOT pass ``get_stats()``'s ``stored_mean``/``stored_std`` here: those
        are averaged over the BATCH axis, which neither persona reduces over.
        Measured, they give mean absolute error 1.02 (``axis=1``) and 1.40
        (``axis=-1``) and a max relative error of 128-136x against the exact
        inverse. They are exact only at ``axis=0``.

        :param scaled_inputs: Normalized tensor to denormalize.
        :type scaled_inputs: keras.KerasTensor
        :param mean: Optional mean to invert with. Must be given together with
            ``std``. When given, the layer's own statistics are not consulted.
        :type mean: keras.KerasTensor | None
        :param std: Optional standard deviation to invert with. Must be given
            together with ``mean``.
        :type std: keras.KerasTensor | None
        :return: Tensor on the original scale.
        :rtype: keras.KerasTensor
        :raises ValueError: If exactly one of ``mean`` and ``std`` is given.
        :raises RuntimeError: If neither is given and the layer has no usable
            statistics, because it has not been called or was called under a
            trace.
        """
        if (mean is None) != (std is None):
            raise ValueError(
                "inverse_transform takes mean and std together or neither; got "
                f"mean={'a value' if mean is not None else None} and "
                f"std={'a value' if std is not None else None}. Inverting with "
                "one of the two would silently use the layer's other statistic, "
                "which belongs to a different batch."
            )

        if mean is not None:
            return self._undo_normalization(scaled_inputs, mean, std)

        if self._last_mean is None or self._last_std is None:
            raise RuntimeError(
                "Cannot perform inverse transformation: statistics not computed. "
                "Call the layer with input data first to compute statistics."
            )

        # The statistics are PER-SAMPLE, so they carry the batch axis and cannot
        # be held in a keras.Variable, which needs a fully defined shape. They
        # are therefore plain tensors, and a tensor produced inside a traced
        # call() dies with the trace. Detect that and say so, rather than
        # letting the caller hit `TypeError: ... is out of scope` from deep
        # inside the multiply below.
        if not self._stats_are_usable():
            raise RuntimeError(
                "Cannot perform inverse transformation: the stored statistics "
                "came from a traced call (tf.function, model.fit or predict) "
                "and do not outlive it. They are per-sample, so they belong to "
                "a batch that no longer exists. Call the layer eagerly on the "
                "data you want to invert, then call inverse_transform. For a "
                "persistent batch-averaged summary instead, construct with "
                "store_stats=True and read get_stats()."
            )

        return self._undo_normalization(
            scaled_inputs, self._last_mean, self._last_std
        )

    def _undo_normalization(
        self,
        scaled_inputs: keras.KerasTensor,
        mean: keras.KerasTensor,
        std: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Undo the affine transform and then the normalization, in that order.

        The single arithmetic path both ``inverse_transform`` entries share, so
        the explicit-statistics case cannot drift from the stored-statistics one.

        :param scaled_inputs: Normalized tensor to denormalize.
        :type scaled_inputs: keras.KerasTensor
        :param mean: The mean to add back. Broadcastable to ``scaled_inputs``.
        :type mean: keras.KerasTensor
        :param std: The standard deviation to multiply by. Broadcastable to
            ``scaled_inputs``.
        :type std: keras.KerasTensor
        :return: Tensor on the original scale.
        :rtype: keras.KerasTensor
        """
        x = scaled_inputs

        # Reverse the affine transform.
        # affine_weight (gamma) starts at ones but is trainable, and training can
        # drive it to 0. Dividing by 0 gives inf or NaN. So floor the magnitude of
        # the denominator at eps and keep its sign.
        if self.affine:
            gamma = self.affine_weight
            # ops.sign would return 0 for gamma == 0 and zero the denominator, so
            # this maps gamma == 0 to +1 instead.
            sign = ops.where(gamma < 0, -1.0, 1.0)
            gamma_safe = sign * ops.maximum(ops.abs(gamma), self.eps)
            x = (x - self.affine_bias) / gamma_safe

        # Reverse normalization: multiply by std and add mean
        x = x * std + mean

        return x

    def denormalize(
        self,
        scaled_inputs: keras.KerasTensor,
        mean: Optional[keras.KerasTensor] = None,
        std: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Denormalize a tensor. Calls ``inverse_transform`` and nothing else.

        The ``mean``/``std`` pair is forwarded unchanged; see
        ``inverse_transform`` for what they must be and for the batch-averaged
        ``get_stats()`` trap they must NOT be.

        :param scaled_inputs: Normalized tensor to denormalize.
        :type scaled_inputs: keras.KerasTensor
        :param mean: Optional mean to invert with, given together with ``std``.
        :type mean: keras.KerasTensor | None
        :param std: Optional standard deviation, given together with ``mean``.
        :type std: keras.KerasTensor | None
        :return: Tensor on the original scale.
        :rtype: keras.KerasTensor
        :raises ValueError: If exactly one of ``mean`` and ``std`` is given.
        :raises RuntimeError: If neither is given and the layer has no usable
            statistics.
        """
        return self.inverse_transform(scaled_inputs, mean=mean, std=std)

    def reset_stats(self) -> None:
        """Reset all stored statistics to their initial values.

        Drops the statistics of the last call, so ``inverse_transform`` raises
        until the layer is called again. If ``store_stats`` is on and the layer
        is built, ``stored_mean`` goes back to zeros and ``stored_std`` to ones.

        :return: Nothing.
        :rtype: None
        """
        # Clear instance variables used for inverse transform
        self._last_mean = None
        self._last_std = None

        # Reset persistent statistics if they exist
        if self.store_stats and self.built:
            if self.stored_mean is not None and self.stored_std is not None:
                self.stored_mean.assign(ops.zeros_like(self.stored_mean))
                self.stored_std.assign(ops.ones_like(self.stored_std))

    def get_stats(self) -> Optional[Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Get the persistent statistics weights.

        Returns ``None`` unless ``store_stats`` is on, the layer is built, and
        both weights exist. These are the batch-averaged weights, not the
        per-call statistics that ``inverse_transform`` uses.

        :return: Tuple of ``(stored_mean, stored_std)``, or ``None``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor] | None
        """
        if (not self.store_stats or not self.built or
                self.stored_mean is None or self.stored_std is None):
            return None

        return self.stored_mean, self.stored_std

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Normalization does not change shape, so the input shape is returned
        unchanged.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        :return: Output shape tuple, identical to the input.
        :rtype: tuple[int | None, ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "axis": self.axis[0] if len(self.axis) == 1 else self.axis,
            "eps": self.eps,
            "affine": self.affine,
            "affine_weight_initializer": keras.initializers.serialize(
                self.affine_weight_initializer
            ),
            "affine_bias_initializer": keras.initializers.serialize(
                self.affine_bias_initializer
            ),
            "nan_replacement": self.nan_replacement,
            "store_stats": self.store_stats,
        })
        return config

# ---------------------------------------------------------------------
"""``LinearEfficientEnsemble`` -- one shared kernel plus rank-1 per-member scaling.

Running ``k`` independent linear layers costs ``k`` kernels. This layer keeps a
single shared kernel of shape ``(input_dim, units)`` and gives each ensemble
member only two rank-1 vectors: an input scaling ``r`` of shape ``(k, input_dim)``
applied before the matmul and an output scaling ``s`` of shape ``(k, units)``
applied after it. Each member therefore sees its own effective weight matrix
``diag(r_i) @ W @ diag(s_i)`` while paying ``k * (input_dim + units)`` extra
parameters instead of ``k * input_dim * units``. This is the parameter-cheap half
of the TabM trade; :class:`dl_techniques.layers.tabular.nlinear.NLinear` is the
other half, storing ``n`` genuinely independent kernels instead.

The scaling vectors hold no ``scaling_initializer`` knob of their own. Which
distribution they are drawn from is decided entirely by ``init_distribution``,
resolved through
:func:`dl_techniques.layers.tabular._ensemble_scaling.ensemble_scaling_initializer`
-- and that choice is what decides whether the members start as different
functions at all: ``'random-signs'`` (the paper's default) and ``'normal'`` break
symmetry, ``'ones'`` does not.

Each of ``ensemble_scaling_in``, ``ensemble_scaling_out`` and ``use_bias`` gates
both the weight creation in ``build()`` and the matching read in ``call()``. The
two sides are deliberately kept in lockstep: a flag that is False creates no
weight at all, so an asymmetry would be an ``AttributeError`` at call time rather
than a silent numerical change.
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.tabular._ensemble_scaling import (
    EnsembleInitDistribution,
    ensemble_scaling_initializer,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.tabular.linear_efficient_ensemble")
class LinearEfficientEnsemble(keras.layers.Layer):
    """
    Efficient ensemble linear layer with rank-1 perturbations.

    This layer performs a shared linear transformation across ``k`` ensemble
    members, with optional learnable input scaling (``r``) and output scaling
    (``s``) vectors per member. The result is equivalent to applying unique
    diagonal transformations to the shared kernel for each member, providing
    ensemble diversity without the cost of ``k`` independent weight matrices.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, K, D_in]              │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Opt. input scaling: x * r[K,D]  │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Shared matmul: x @ W            │
        │  W [D_in, units] (shared)        │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Opt. output scaling: x * s[K,U] │
        │  + opt. bias [K, units]          │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Output [B, K, units]            │
        └──────────────────────────────────┘

    :param units: Output dimension. Must be positive.
    :type units: int
    :param k: Number of ensemble members. Must be positive.
    :type k: int
    :param use_bias: Whether to use bias.
    :type use_bias: bool
    :param ensemble_scaling_in: Whether to use input scaling.
    :type ensemble_scaling_in: bool
    :param ensemble_scaling_out: Whether to use output scaling.
    :type ensemble_scaling_out: bool
    :param init_distribution: How the per-member scaling vectors ``r`` and ``s``
        are initialized. ``'random-signs'`` draws from :math:`\\{-1, +1\\}` (the
        paper's default: members start as distinct, unit-magnitude sign patterns),
        ``'normal'`` draws from :math:`\\mathcal{N}(1, 0.1)`, and ``'ones'``
        starts every member at the identity perturbation — under which all ``k``
        members share one effective weight matrix at initialization.
    :type init_distribution: str
    :param kernel_initializer: Initializer for the main weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.
    :type kwargs: Any

    :raises ValueError: If ``units`` or ``k`` is not positive, or if
        ``init_distribution`` is not one of the three supported names.
    """

    def __init__(
            self,
            units: int,
            k: int,
            use_bias: bool = True,
            ensemble_scaling_in: bool = True,
            ensemble_scaling_out: bool = True,
            init_distribution: EnsembleInitDistribution = 'random-signs',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"units must be positive; got {units!r}")
        if k <= 0:
            raise ValueError(f"k must be positive; got {k!r}")

        self.units = units
        self.k = k
        self.use_bias = use_bias
        self.ensemble_scaling_in = ensemble_scaling_in
        self.ensemble_scaling_out = ensemble_scaling_out
        self.init_distribution = init_distribution
        # The scaling initializer is fully determined by `init_distribution`;
        # there is no separate knob to contradict it. That is also why
        # `get_config()` below serializes only `init_distribution`.
        self.scaling_initializer = ensemble_scaling_initializer(init_distribution)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the ensemble linear layer weights.

        Each of the three optional weights is created only when its flag is set,
        and :meth:`call` reads it behind the identical flag. See the D-011
        anchor below the kernel for why that conditionality is load-bearing.
        """
        input_dim = input_shape[-1]

        # Main weight matrix shared across ensemble members
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='kernel'
        )

        # DECISION plan-2026-09-07T095804-b821967f/D-011
        # The next three weights are created CONDITIONALLY, each behind the same
        # flag `call()` reads it behind. Do NOT "clean this up" into three
        # unconditional `add_weight` calls with the flags applied only in
        # `call()` (e.g. `r` always created and multiplied in as 1.0 when
        # `ensemble_scaling_in` is False). It reads simpler and it is wrong:
        # the set of weights `build()` creates IS the `.keras` weight layout, so
        # an unconditional weight changes the layout for every flag combination
        # and every archive written under the current one is refused on load.
        # All three flags are in `get_config()`, so all eight combinations are
        # reachable and archivable -- this is not a dead branch.
        # The lockstep is also the error-reporting mechanism: a False flag
        # leaves the attribute UNSET, so a `call()` that drifts out of step
        # raises `AttributeError` at the drift site instead of silently
        # multiplying by a weight the config says does not exist.
        # See decisions.md D-011 and plan.md § Edge cases.

        # Input scaling weights
        if self.ensemble_scaling_in:
            self.r = self.add_weight(
                shape=(self.k, input_dim),
                initializer=self.scaling_initializer,
                trainable=True,
                name='input_scaling'
            )

        # Output scaling weights
        if self.ensemble_scaling_out:
            self.s = self.add_weight(
                shape=(self.k, self.units),
                initializer=self.scaling_initializer,
                trainable=True,
                name='output_scaling'
            )

        # Bias weights
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.k, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name='bias'
            )

        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Forward pass through efficient ensemble layer.

            :param inputs: Input tensor of shape (batch_size, k, input_dim).
            :type inputs: keras.KerasTensor

            :return: Output tensor of shape (batch_size, k, units).
            :rtype: keras.KerasTensor
        """
        x = inputs

        # Every gate below mirrors the matching gate in `build()`; a weight that
        # was not created there is never read here.
        if self.ensemble_scaling_in:
            x = keras.ops.multiply(x, keras.ops.expand_dims(self.r, axis=0))

        x = keras.ops.einsum('bki,iu->bku', x, self.kernel)

        if self.ensemble_scaling_out:
            x = keras.ops.multiply(x, keras.ops.expand_dims(self.s, axis=0))

        if self.use_bias:
            x = keras.ops.add(x, keras.ops.expand_dims(self.bias, axis=0))

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int, int]:
        """Compute the output shape of the layer."""
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "k": self.k,
            "use_bias": self.use_bias,
            "ensemble_scaling_in": self.ensemble_scaling_in,
            "ensemble_scaling_out": self.ensemble_scaling_out,
            "init_distribution": self.init_distribution,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

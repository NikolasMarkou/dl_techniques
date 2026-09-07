"""``TabMMLPBlock`` -- one linear + activation + optional dropout stage of a TabM MLP.

A single block of the TabM backbone. In plain mode (``k is None``) it is a
``Dense``; in ensemble mode it is either a
:class:`dl_techniques.layers.tabular.linear_efficient_ensemble.LinearEfficientEnsemble`
(``ensemble_type='efficient'`` -- one shared kernel plus rank-1 per-member
scaling) or a :class:`dl_techniques.layers.tabular.nlinear.NLinear`
(``ensemble_type='packed'`` -- ``k`` genuinely independent kernels). The branch is
a config-time decision taken in ``__init__``, so the sub-layer exists before
``build()`` and the weight graph is stable across serialization.

The ``activation`` argument is stored through
:func:`dl_techniques.utils.activation_serialization.deserialize_activation` and
emitted through :func:`~dl_techniques.utils.activation_serialization.serialize_activation`,
the same pair
:class:`dl_techniques.layers.tabular.tabm_backbone.TabMBackbone` uses -- so a
string key survives ``get_config()`` verbatim and a callable or activation
``Layer`` round-trips as a config dict. Because that pair deliberately returns a
string unchanged, the *live* callable ``call()`` runs is resolved separately into
``self.activation_fn``; ``self.activation`` is the serializable value and
``self.activation_fn`` is what the forward path invokes (the split
``layers/ffn/gated_mlp.py`` uses for the same reason).
"""

import keras
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.tabular._ensemble_scaling import EnsembleInitDistribution
from dl_techniques.layers.tabular.linear_efficient_ensemble import LinearEfficientEnsemble
from dl_techniques.layers.tabular.nlinear import NLinear
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


# DECISION plan-2026-09-01T110541-dcc1574a/D-001: keep the ``TabM`` prefix; do not rename to
# ``MLPBlock``, which is ``layers/ffn/mlp.py``'s bare class name and FFN factory key. See decisions.md.
@register_dl_technique("dl_techniques.layers.tabular.tabm_mlp_block")
class TabMMLPBlock(keras.layers.Layer):
    """
    MLP block with optional efficient ensemble support.

    This layer implements a single MLP block (linear + activation + optional
    dropout) that can operate in plain mode (single model) or ensemble mode
    (with ``k`` members using ``LinearEfficientEnsemble``).

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [B, (K,) D]              │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Linear (Dense or Ensemble)     │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Activation                     │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Optional Dropout               │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Output [B, (K,) units]         │
        └─────────────────────────────────┘

    :param units: Number of units in the hidden layer.
    :type units: int
    :param k: Number of ensemble members (None for plain MLP).
    :type k: int or None
    :param ensemble_type: How the ``k`` members are realized when ``k`` is not
        ``None``. ``'efficient'`` uses a :class:`LinearEfficientEnsemble` (one
        shared kernel plus rank-1 per-member scaling); ``'packed'`` uses an
        :class:`NLinear`, i.e. ``k`` fully independent kernels, which costs ``k``
        times the backbone parameters and is the honest deep-ensemble baseline.
    :type ensemble_type: str
    :param ensemble_scaling_in: Whether the efficient ensemble applies per-member
        input scaling. Ignored when ``ensemble_type='packed'`` or ``k is None``.
    :type ensemble_scaling_in: bool
    :param ensemble_scaling_out: Whether the efficient ensemble applies per-member
        output scaling. Ignored when ``ensemble_type='packed'`` or ``k is None``.
    :type ensemble_scaling_out: bool
    :param init_distribution: Initialization of the per-member scaling vectors
        (see :class:`LinearEfficientEnsemble`).
    :type init_distribution: str
    :param activation: Activation, as a name string, a callable, or an
        activation ``Layer``. Stored verbatim for ``get_config()``; the live
        callable used by ``call()`` is resolved into ``activation_fn``.
    :type activation: str or Callable
    :param dropout_rate: Dropout rate, in ``[0, 1]``.
    :type dropout_rate: float
    :param use_bias: Whether to use bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.
    :type kwargs: Any

    :ivar activation: The activation as ``get_config()`` will emit it -- a
        string stays a string, because
        :func:`~dl_techniques.utils.activation_serialization.deserialize_activation`
        returns a non-``dict`` unchanged. Never assume this is callable.
    :vartype activation: Any
    :ivar activation_fn: The live callable ``call()`` invokes, resolved from
        ``activation``.
    :vartype activation_fn: Callable
    :ivar dropout: Always present, even at ``dropout_rate == 0`` (v2 s1.3); it is
        ``build()`` and ``call()`` that gate on the rate, not the construction.
    :vartype dropout: keras.layers.Dropout

    :raises ValueError: If ``ensemble_type`` is not ``'efficient'`` or ``'packed'``,
        if ``units`` is not positive, if ``k`` is given and not positive, or if
        ``dropout_rate`` is outside ``[0, 1]``.
    """

    def __init__(
            self,
            units: int,
            k: Optional[int] = None,
            ensemble_type: Literal['efficient', 'packed'] = 'efficient',
            ensemble_scaling_in: bool = True,
            ensemble_scaling_out: bool = True,
            init_distribution: EnsembleInitDistribution = 'random-signs',
            activation: Union[str, Callable[[Any], Any]] = 'relu',
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if ensemble_type not in ('efficient', 'packed'):
            raise ValueError(
                f"ensemble_type must be 'efficient' or 'packed'; got {ensemble_type!r}"
            )
        if units <= 0:
            raise ValueError(f"units must be positive; got {units!r}")
        if k is not None and k <= 0:
            raise ValueError(f"k must be positive when given; got {k!r}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1]; got {dropout_rate!r}")

        self.units = units
        self.k = k
        self.ensemble_type = ensemble_type
        self.ensemble_scaling_in = ensemble_scaling_in
        self.ensemble_scaling_out = ensemble_scaling_out
        self.init_distribution = init_distribution
        # `deserialize_activation` returns a string unchanged, so `self.activation`
        # is the SERIALIZABLE value (what `get_config` emits) and may be a str;
        # `self.activation_fn` is the live callable `call()` runs.
        self.activation = deserialize_activation(activation)
        self.activation_fn = (
            self.activation if callable(self.activation)
            else keras.activations.get(self.activation)
        )
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Create sub-layers in __init__ (units/k are config-known) so weights
        # are reliably created/restored across serialization.
        if self.k is None:
            self.linear = keras.layers.Dense(
                self.units,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='linear'
            )
        elif self.ensemble_type == 'packed':
            # k fully independent kernels; fan-in resolved at build time.
            self.linear = NLinear(
                n=self.k,
                input_dim=None,
                output_dim=self.units,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='linear'
            )
        else:
            self.linear = LinearEfficientEnsemble(
                self.units,
                self.k,
                use_bias=self.use_bias,
                ensemble_scaling_in=self.ensemble_scaling_in,
                ensemble_scaling_out=self.ensemble_scaling_out,
                init_distribution=self.init_distribution,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='linear'
            )

        # v2 s1.3: create unconditionally, gate build() and call() on the rate.
        # `Dropout` holds no weights, so a rate of 0 still costs nothing while the
        # object graph and the auto-numbered sibling names stay stable.
        self.dropout = keras.layers.Dropout(self.dropout_rate, name='dropout')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the MLP block layers."""
        # Explicitly build sub-layers for robust serialization
        self.linear.build(input_shape)
        linear_output_shape = self.linear.compute_output_shape(input_shape)

        # Build exactly what call() runs (v2 s8.1): a zero rate short-circuits both.
        if self.dropout_rate > 0:
            self.dropout.build(linear_output_shape)

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Forward pass through MLP block.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Training mode flag.
            :type training: bool or None

            :return: Output tensor after linear transformation, activation, and dropout.
            :rtype: keras.KerasTensor
        """
        x = self.linear(inputs)
        x = self.activation_fn(x)

        if self.dropout_rate > 0:
            x = self.dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        if self.k is None:
            return (input_shape[0], self.units)
        else:
            return (input_shape[0], self.k, self.units)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "k": self.k,
            "ensemble_type": self.ensemble_type,
            "ensemble_scaling_in": self.ensemble_scaling_in,
            "ensemble_scaling_out": self.ensemble_scaling_out,
            "init_distribution": self.init_distribution,
            "activation": serialize_activation(self.activation),
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

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
string key survives ``get_config()`` verbatim and a stateless callable
round-trips as a config dict. Because that pair deliberately returns a string
unchanged, the *live* callable ``call()`` runs is resolved separately into
``self.activation_fn``; ``self.activation`` is the serializable value and
``self.activation_fn`` is what the forward path invokes (the split
``layers/ffn/gated_mlp.py`` uses for the same reason).

An activation **``Layer`` instance is rejected** by ``__init__``. Two supported
forms remain: a name string and a stateless callable. See the class ``:raises``
list and decisions.md D-009 for the measurements behind that rejection.
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
        There is no ensemble at all when ``k is None``, so any value other than
        the ``'efficient'`` default is **rejected** in that case rather than
        silently discarded -- ``k is None`` always builds a plain ``Dense``.
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
    :param activation: Activation, as a name string (``'relu'``, ``'mish'``,
        ...) or as a **stateless** callable (``keras.activations.gelu``, ...).
        Stored verbatim for ``get_config()``; the live callable used by
        ``call()`` is resolved into ``activation_fn``. An activation ``Layer``
        instance is **not** accepted -- see ``:raises``.
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

    :raises ValueError: If ``ensemble_type`` is not ``'efficient'`` or
        ``'packed'``; if ``ensemble_type`` is non-default while ``k is None``
        (there is no ensemble to configure, and the setting would otherwise be
        discarded while ``get_config()`` still reported it); if ``units`` is not
        positive; if ``k`` is given and not positive; if ``dropout_rate`` is
        outside ``[0, 1]``; or if ``activation`` resolves to a
        ``keras.layers.Layer`` instance.
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
        # Resolved before validation, not after: `from_config` hands back a config
        # dict, and only the deserialized value can be checked for what it is.
        activation = deserialize_activation(activation)

        if ensemble_type not in ('efficient', 'packed'):
            raise ValueError(
                f"ensemble_type must be 'efficient' or 'packed'; got {ensemble_type!r}"
            )
        if k is None and ensemble_type != 'efficient':
            raise ValueError(
                f"ensemble_type={ensemble_type!r} is meaningless when k is None: with no "
                f"ensemble members this block is a plain Dense, so the setting would be "
                f"silently discarded while get_config() still reported {ensemble_type!r}. "
                f"Pass k to build an ensemble, or leave ensemble_type at its 'efficient' "
                f"default."
            )
        if units <= 0:
            raise ValueError(f"units must be positive; got {units!r}")
        if k is not None and k <= 0:
            raise ValueError(f"k must be positive when given; got {k!r}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1]; got {dropout_rate!r}")
        if isinstance(activation, keras.layers.Layer):
            raise ValueError(
                f"activation must be a name string (e.g. 'relu', 'mish') or a stateless "
                f"callable (e.g. keras.activations.gelu); got a "
                f"{type(activation).__name__} Layer instance. An activation Layer is not "
                f"supported here: build() does not build it, so a parameterised one such "
                f"as PReLU creates its variables lazily inside call() and they are absent "
                f"from the archive at save() time; and TabMBackbone would hand the one "
                f"instance to every block, which raises on the forward pass as soon as two "
                f"blocks differ in width. See decisions.md D-009."
            )

        self.units = units
        self.k = k
        self.ensemble_type = ensemble_type
        self.ensemble_scaling_in = ensemble_scaling_in
        self.ensemble_scaling_out = ensemble_scaling_out
        self.init_distribution = init_distribution
        # DECISION plan-2026-09-07T095804-b821967f/D-009: `activation` and `activation_fn`
        # are two attributes on purpose. `self.activation` is the SERIALIZABLE value (what
        # `get_config` emits) and is usually a `str`; `self.activation_fn` is the live
        # callable `call()` runs. Do NOT collapse `activation_fn` back into `activation`:
        # `deserialize_activation` returns a non-dict UNCHANGED (by design -- see
        # `utils/activation_serialization.py`, because a dl_techniques factory key such as
        # 'mish' must survive `get_config()` verbatim), so a single
        # `self.activation = deserialize_activation(activation)` leaves a plain string in
        # the attribute and `call()` raises `TypeError: 'str' object is not callable` on
        # the DEFAULT 'relu' path every shipped caller uses. Do NOT "fix" that by eagerly
        # storing `keras.activations.get(activation)` either: that is the pre-split shape,
        # and it destroys the factory key on the way into `get_config()`. Same split as
        # `layers/ffn/gated_mlp.py`. See decisions.md D-009.
        self.activation = activation
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
        """Compute the output shape of the layer.

            :param input_shape: Shape of the input, batch axis included.
            :type input_shape: tuple of (int or None)

            :return: Plain mode (``k is None``) maps only the LAST axis, so every
                leading axis is preserved; ensemble mode is always
                ``(batch, k, units)``.
            :rtype: tuple of (int or None)
        """
        # DECISION plan-2026-09-07T161712-985e4d31/D-005: the plain branch keeps every
        # leading axis because its ``self.linear`` is a ``keras.layers.Dense``, which
        # transforms only the last axis. It previously hardcoded
        # ``(input_shape[0], self.units)``, which is right at rank 2 -- the only rank the
        # shipped model ever reaches here -- and wrong at every other rank: measured,
        # ``TabMMLPBlock(units=8)`` on ``(2, 3, 6)`` RAN to ``(2, 3, 8)`` and PREDICTED
        # ``(2, 8)``.
        #
        # DO NOT add a matching formula to ``tabm_backbone.py``. ``TabMBackbone`` owns no
        # shape arithmetic: both its ``build()`` and its ``compute_output_shape()`` thread
        # ``current_shape = block.compute_output_shape(current_shape)`` through the block
        # list, so this ONE expression is the whole chain's arithmetic (guide v2 s3.4 --
        # shape arithmetic lives in exactly one pure helper). A second site in the
        # backbone would be the duplication s3.4 forbids, and would have to be kept in
        # lockstep with this one by hand. The delegation is not merely asserted here:
        # ``TestPlainModeShapeIsRankCorrect::test_the_backbone_inherits_the_block_formula_at_rank_3``
        # in ``tests/test_layers/test_tabular/test_tabm_blocks.py`` fails if it stops
        # holding.
        #
        # No rank GUARD was added either: a rank-3+ plain-mode call currently succeeds and
        # computes a numerically correct answer, so rejecting it would be a behaviour
        # regression for zero benefit. See decisions.md D-005.
        if self.k is None:
            return tuple(input_shape[:-1]) + (self.units,)
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

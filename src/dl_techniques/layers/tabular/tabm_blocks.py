"""TabM-style batched-ensemble building blocks: ``ScaleEnsemble``, ``LinearEfficientEnsemble``,
``NLinear``, and the ``TabMMLPBlock`` / ``TabMBackbone`` layers that assemble them into MLPs.

Training ``k`` independent models and averaging their predictions improves accuracy
and uncertainty estimates, but costs ``k`` times the parameters and compute. This
module represents all ``k`` members inside one model instead, sharing most weights
and giving each member only a small per-member perturbation. ``LinearEfficientEnsemble``
shares one kernel across members and multiplies it by rank-1 input/output scaling
vectors (`r`, `s`) per member; ``NLinear`` is the alternative with ``k`` fully
independent kernels via a single batched ``einsum``, used where true independence
matters more than parameter savings.

Every layer here can also run in plain (non-ensemble) mode by leaving ``k`` unset.
The scaling-vector `init_distribution` controls whether ensemble members start as
different functions at all: `'random-signs'` (the paper's choice) and `'normal'`
both break symmetry between members; `'ones'` makes every member's effective
weight matrix identical at initialization.
"""

import keras
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.tabular._ensemble_scaling import EnsembleInitDistribution
from dl_techniques.layers.tabular.linear_efficient_ensemble import LinearEfficientEnsemble
from dl_techniques.layers.tabular.nlinear import NLinear
from dl_techniques.layers.tabular.scale_ensemble import ScaleEnsemble
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


# DECISION plan-2026-09-01T110541-dcc1574a/D-001: keep the ``TabM`` prefix; do not rename to
# ``MLPBlock``, which is ``layers/ffn/mlp.py``'s bare class name and FFN factory key. See decisions.md.
@register_dl_technique("dl_techniques.layers.tabular.tabm_blocks")
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
    :param activation: Activation function.
    :type activation: str
    :param dropout_rate: Dropout rate.
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

    :raises ValueError: If ``ensemble_type`` is not ``'efficient'`` or ``'packed'``.
    """

    def __init__(
            self,
            units: int,
            k: Optional[int] = None,
            ensemble_type: Literal['efficient', 'packed'] = 'efficient',
            ensemble_scaling_in: bool = True,
            ensemble_scaling_out: bool = True,
            init_distribution: EnsembleInitDistribution = 'random-signs',
            activation: str = 'relu',
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
        self.units = units
        self.k = k
        self.ensemble_type = ensemble_type
        self.ensemble_scaling_in = ensemble_scaling_in
        self.ensemble_scaling_out = ensemble_scaling_out
        self.init_distribution = init_distribution
        self.activation = keras.activations.get(activation)
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

        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the MLP block layers."""
        # Explicitly build sub-layers for robust serialization
        self.linear.build(input_shape)
        linear_output_shape = self.linear.compute_output_shape(input_shape)

        if self.dropout is not None:
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
        x = self.activation(x)

        if self.dropout_rate > 0 and self.dropout is not None:
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
            "activation": keras.activations.serialize(self.activation),
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.tabular.tabm_blocks")
class TabMBackbone(keras.layers.Layer):
    """
    TabM backbone MLP with optional ensemble support.

    This layer stacks multiple ``TabMMLPBlock`` layers to form a complete backbone.
    It can operate in plain mode (single model) or ensemble mode by setting the
    ``k`` parameter.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [B, (K,) D]              │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  TabMMLPBlock(hidden_dims[0])   │
        ├─────────────────────────────────┤
        │  TabMMLPBlock(hidden_dims[1])   │
        ├─────────────────────────────────┤
        │  ...                            │
        ├─────────────────────────────────┤
        │  TabMMLPBlock(hidden_dims[-1])  │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Output [B, (K,) hidden[-1]]    │
        └─────────────────────────────────┘

    :param hidden_dims: List of hidden layer dimensions.
    :type hidden_dims: list[int]
    :param k: Number of ensemble members (None for plain MLP).
    :type k: int or None
    :param ensemble_type: ``'efficient'`` or ``'packed'`` (see :class:`TabMMLPBlock`).
    :type ensemble_type: str
    :param ensemble_scaling_in: Per-member input scaling in the efficient ensemble.
    :type ensemble_scaling_in: bool
    :param ensemble_scaling_out: Per-member output scaling in the efficient ensemble.
    :type ensemble_scaling_out: bool
    :param init_distribution: Initialization of the per-member scaling vectors.
    :type init_distribution: str
    :param activation: Activation function.
    :type activation: str
    :param dropout_rate: Dropout rate.
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
    """

    def __init__(
            self,
            hidden_dims: List[int],
            k: Optional[int] = None,
            ensemble_type: Literal['efficient', 'packed'] = 'efficient',
            ensemble_scaling_in: bool = True,
            ensemble_scaling_out: bool = True,
            init_distribution: EnsembleInitDistribution = 'random-signs',
            activation: str = 'relu',
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.k = k
        self.ensemble_type = ensemble_type
        self.ensemble_scaling_in = ensemble_scaling_in
        self.ensemble_scaling_out = ensemble_scaling_out
        self.init_distribution = init_distribution
        self.activation = deserialize_activation(activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Create all MLP blocks in __init__ (hidden_dims are config-known) so
        # weights are reliably created/restored across serialization.
        self.blocks = [
            TabMMLPBlock(
                units=units,
                k=self.k,
                ensemble_type=self.ensemble_type,
                ensemble_scaling_in=self.ensemble_scaling_in,
                ensemble_scaling_out=self.ensemble_scaling_out,
                init_distribution=self.init_distribution,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f'block_{i}'
            )
            for i, units in enumerate(self.hidden_dims)
        ]

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the backbone MLP blocks in computational order."""
        current_shape = input_shape
        for block in self.blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Forward pass through backbone MLP.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Training mode flag.
            :type training: bool or None

            :return: Output tensor after passing through all MLP blocks.
            :rtype: keras.KerasTensor
        """
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        shape = input_shape
        for block in self.blocks:
            shape = block.compute_output_shape(shape)
        return shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "hidden_dims": self.hidden_dims,
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

# ---------------------------------------------------------------------


"""``TabMBackbone`` -- the stack of ``TabMMLPBlock`` layers that forms a TabM MLP.

The backbone owns no math of its own: it constructs one
:class:`dl_techniques.layers.tabular.tabm_mlp_block.TabMMLPBlock` per entry of
``hidden_dims`` in ``__init__`` (the dimensions are config-known, so the weight
graph is stable across serialization) and threads the input through them in
order. Every ensemble knob -- ``k``, ``ensemble_type``, the two scaling flags and
``init_distribution`` -- is passed straight down to each block, so the backbone
is plain or ensembled exactly as its blocks are.

``activation`` is stored through
:func:`dl_techniques.utils.activation_serialization.deserialize_activation` and
emitted through :func:`~dl_techniques.utils.activation_serialization.serialize_activation`.
That pair returns a non-``dict`` unchanged, so ``self.activation`` is the
*serializable* value -- a string stays a string -- and it is handed down to the
blocks in that form. The backbone never invokes it: resolving the live callable
is ``TabMMLPBlock``'s job (``activation_fn`` there), which is why this module
needs no callable of its own.
"""

import keras
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.tabular._ensemble_scaling import EnsembleInitDistribution
from dl_techniques.layers.tabular.tabm_mlp_block import TabMMLPBlock
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.tabular.tabm_backbone")
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

    :param hidden_dims: List of hidden layer dimensions. Must be non-empty and
        every entry must be positive.
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
    :param activation: Activation, as a name string, a callable, or an activation
        ``Layer``. Stored verbatim and handed to every block in that form.
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

    :ivar activation: The activation as ``get_config()`` will emit it, and as it
        is handed down to each block -- a string stays a string, because
        :func:`~dl_techniques.utils.activation_serialization.deserialize_activation`
        returns a non-``dict`` unchanged. Never assume this is callable; the
        backbone does not call it, each ``TabMMLPBlock`` resolves its own
        ``activation_fn``.
    :vartype activation: Any
    :ivar blocks: One ``TabMMLPBlock`` per entry of ``hidden_dims``, in order.
    :vartype blocks: list[TabMMLPBlock]

    :raises ValueError: If ``hidden_dims`` is empty or holds a non-positive
        entry, or if ``dropout_rate`` is outside ``[0, 1]``. ``k`` and
        ``ensemble_type`` are validated by :class:`TabMMLPBlock` as the blocks
        are constructed.
    """

    def __init__(
            self,
            hidden_dims: List[int],
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
        if len(hidden_dims) == 0:
            raise ValueError(f"hidden_dims must be non-empty; got {hidden_dims!r}")
        for i, dim in enumerate(hidden_dims):
            if dim <= 0:
                raise ValueError(
                    f"hidden_dims entries must be positive; "
                    f"got {dim!r} at index {i} of {hidden_dims!r}"
                )
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1]; got {dropout_rate!r}")

        self.hidden_dims = hidden_dims
        self.k = k
        self.ensemble_type = ensemble_type
        self.ensemble_scaling_in = ensemble_scaling_in
        self.ensemble_scaling_out = ensemble_scaling_out
        self.init_distribution = init_distribution
        # `deserialize_activation` returns a string unchanged, so `self.activation`
        # is the SERIALIZABLE value and may be a str. It is handed to the blocks in
        # that form; each block resolves its own live callable (`activation_fn`).
        # Nothing here ever calls it.
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

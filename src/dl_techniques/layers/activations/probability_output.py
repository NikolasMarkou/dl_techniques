"""
One wrapper over six ways to turn a tensor into a probability distribution.

``ProbabilityOutput`` picks a strategy from a string and delegates to it, so
swapping softmax for sparsemax is a config change rather than a model edit.
Every strategy returns rows that sum to 1 over the chosen axis.

The six strategies, and what they eat:

- **softmax** -- ``keras.layers.Softmax``. Standard exponential
  normalization. Takes logits.
- **sparsemax** -- Euclidean projection onto the simplex. Produces exact
  zeros. Takes logits.
- **threshmax** -- confidence gating against ``1/N``, then renormalization.
  Suppresses low-confidence classes but does not zero them. Takes logits.
- **adaptive** -- entropy-driven temperature. Takes logits.
- **routing** -- deterministic, parameter-free hierarchical tree. Takes
  **features**, not logits, and does its own projection.
- **hierarchical** -- the same tree with a learned projection. Takes
  **features**.

``routing`` and ``hierarchical`` replace your final Dense layer, so the last
dimension of the output is ``output_dim``, not the input width. The other
four preserve the input shape.
"""

import keras
from typing import Optional, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .sparsemax import Sparsemax
from .thresh_max import ThreshMax
from .adaptive_softmax import AdaptiveTemperatureSoftmax
from .routing_probabilities import RoutingProbabilitiesLayer


# Type alias for supported probability types
ProbabilityType = Literal[
    "softmax",
    "sparsemax",
    "threshmax",
    "thresh_max",
    "adaptive",
    "adaptive_softmax",
    "routing",
    "deterministic_routing",
    "hierarchical",
    "hierarchical_routing",
]

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ProbabilityOutput(keras.layers.Layer):
    """Config-driven wrapper that delegates to one probability layer.

    Instantiates the strategy named by ``probability_type`` in ``__init__``
    and stores it as ``self.strategy_layer``. ``call`` forwards to it and
    nothing else. The wrapper owns no weights of its own; the strategy layer
    may own some (``hierarchical`` does, the rest do not).

    **Architecture Overview:**

    .. code-block:: text

                         x  [B, ..., D]
                                │
                                ▼
                  ┌───────────────────────────┐
                  │ strategy_layer            │
                  │ (built once in __init__)  │
                  └─────────────┬─────────────┘
                                │
                   ┌────────────┴────────────┐
                   │ logit strategies        │ routing strategies
                   ▼                         ▼
        ┌─────────────────────┐   ┌─────────────────────┐
        │ Softmax             │   │ RoutingProbabili-   │
        │ Sparsemax           │   │ tiesLayer, mode     │
        │ ThreshMax           │   │ deterministic or    │
        │ AdaptiveTempSoftmax │   │ trainable           │
        └──────────┬──────────┘   └──────────┬──────────┘
                   │ [B, ..., D]             │ [B, ..., output_dim]
                   └────────────┬────────────┘
                                ▼
                        y  rows sum to 1

    Only one branch exists in any given instance: the fork is resolved in
    ``__init__``, not per batch.

    **Accepted ``probability_type`` values and the layer each builds:**

    .. code-block:: text

        key                     builds
        softmax                 keras.layers.Softmax
        sparsemax               Sparsemax
        threshmax               ThreshMax
        thresh_max              ThreshMax
        adaptive                AdaptiveTemperatureSoftmax
        adaptive_softmax        AdaptiveTemperatureSoftmax
        routing                 RoutingProbabilitiesLayer
        deterministic_routing   RoutingProbabilitiesLayer
        hierarchical            RoutingProbabilitiesLayer(mode="trainable")
        hierarchical_routing    RoutingProbabilitiesLayer(mode="trainable")

    The value is lowercased before the lookup, so ``"SOFTMAX"`` is accepted
    and ``get_config`` writes back the lowercase spelling.

    Everything in ``type_config`` is forwarded to the strategy's constructor
    unchanged, except for ``softmax``, where only ``axis`` is read and any
    other key is dropped.

    Three behaviours to know before you rely on them, all measured:

    - **``mask`` reaches the softmax strategy only.** For every other type
      ``call`` drops it silently -- no warning, no error. Measured: a
      ``sparsemax`` instance returns bit-identical output with and without a
      mask.
    - **Only ``hierarchical`` validates ``output_dim``.** Constructing
      ``routing`` without it raises nothing; ``RoutingProbabilitiesLayer``
      defaults ``output_dim`` to the input width, so a ``(4, 16)`` input
      gives a ``(4, 16)`` output instead of the class count you meant.
    - **Routing strategies change the last dimension.** Measured with
      ``type_config={"output_dim": 5}``, a ``(4, 16)`` input gives ``(4, 5)``
      for both ``routing`` and ``hierarchical``, while all four logit
      strategies give ``(4, 16)``.

    :param probability_type: Which strategy to build. One of the ten keys in
        the table above. Defaults to ``"softmax"``.
    :type probability_type: ProbabilityType
    :param type_config: Constructor arguments for the chosen strategy.
        ``None`` means an empty dict. The valid keys are the chosen layer's
        own constructor arguments; consult that layer.
    :type type_config: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``probability_type`` is not one of the ten keys,
        or if it is ``"hierarchical"`` / ``"hierarchical_routing"`` and
        ``type_config`` has no ``"output_dim"``.

    :ivar strategy_layer: The delegate, created in ``__init__``.
    :vartype strategy_layer: keras.layers.Layer
    """

    #: Every accepted ``probability_type`` spelling, checked in ``__init__``.
    _SUPPORTED_TYPES: tuple[str, ...] = (
        "softmax",
        "sparsemax",
        "threshmax",
        "thresh_max",
        "adaptive",
        "adaptive_softmax",
        "routing",
        "deterministic_routing",
        "hierarchical",
        "hierarchical_routing",
    )

    def __init__(
            self,
            probability_type: ProbabilityType = "softmax",
            type_config: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Validate the type string and build the strategy layer.

        The strategy layer is created here rather than in ``build`` so that
        Keras 3 sees it as a sublayer from the start and tracks its weights.

        :param probability_type: Which strategy to build. Lowercased before
            the lookup. Defaults to ``"softmax"``.
        :type probability_type: ProbabilityType
        :param type_config: Constructor arguments for the chosen strategy.
            ``None`` means an empty dict.
        :type type_config: Optional[Dict[str, Any]]
        :param kwargs: Additional keyword arguments for the Layer base class.

        :raises ValueError: If ``probability_type`` is not one of the ten
            accepted keys, or if a hierarchical type is asked for without
            ``"output_dim"`` in ``type_config``. Note that ``routing`` is
            **not** checked for ``output_dim``.
        """
        super().__init__(**kwargs)

        self._probability_type = probability_type.lower()
        self._type_config = type_config if type_config is not None else {}

        if self._probability_type not in self._SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown probability_type '{self._probability_type}'. "
                f"Supported types: {list(self._SUPPORTED_TYPES)}"
            )

        if self._probability_type in ("hierarchical", "hierarchical_routing"):
            if "output_dim" not in self._type_config:
                raise ValueError(
                    "ProbabilityOutput with type='hierarchical' requires "
                    "'output_dim' in type_config."
                )

        self.strategy_layer: keras.layers.Layer = self._create_strategy_layer()

    def _create_strategy_layer(self) -> keras.layers.Layer:
        """Build the delegate named by ``self._probability_type``.

        Every branch forwards ``self._type_config`` verbatim, so an unknown
        key raises from the strategy's own constructor, not from here. The
        ``softmax`` branch is the exception: it reads only ``axis`` and
        ignores the rest.

        :return: The strategy layer.
        :rtype: keras.layers.Layer
        :raises ValueError: Never in practice -- the trailing raise is
            unreachable because ``__init__`` validates the type first. It is
            kept so a new key added to ``_SUPPORTED_TYPES`` without a branch
            here fails loudly instead of returning ``None``.
        """
        if self._probability_type == "softmax":
            axis = self._type_config.get("axis", -1)
            return keras.layers.Softmax(axis=axis, name="softmax")

        elif self._probability_type == "sparsemax":
            return Sparsemax(name="sparsemax", **self._type_config)

        elif self._probability_type in ("threshmax", "thresh_max"):
            return ThreshMax(name="threshmax", **self._type_config)

        elif self._probability_type in ("adaptive", "adaptive_softmax"):
            return AdaptiveTemperatureSoftmax(
                name="adaptive_softmax",
                **self._type_config
            )

        elif self._probability_type in ("routing", "deterministic_routing"):
            return RoutingProbabilitiesLayer(
                name="routing_probs",
                **self._type_config
            )

        elif self._probability_type in ("hierarchical", "hierarchical_routing"):
            return RoutingProbabilitiesLayer(
                name="hierarchical_routing",
                mode="trainable",
                **self._type_config
            )

        raise ValueError(f"Unhandled probability_type: {self._probability_type}")

    def build(self, input_shape: tuple) -> None:
        """Build the strategy layer against the same input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        """
        if self.built:
            return

        self.strategy_layer.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Forward to the strategy layer.

        :param inputs: Logits shaped ``(B, ..., C)`` for the four logit
            strategies, or features shaped ``(B, ..., D)`` for ``routing``
            and ``hierarchical``.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Forwarded to every
            strategy except ``softmax``.
        :type training: Optional[bool]
        :param mask: Mask tensor. Forwarded to ``softmax`` only. For any
            other strategy it is **dropped without a warning** -- measured,
            a ``sparsemax`` instance returns identical values with and
            without it.
        :type mask: Optional[keras.KerasTensor]
        :return: Probabilities summing to 1 over the strategy's axis. Same
            shape as ``inputs``, except for the routing strategies, which
            return ``output_dim`` on the last axis.
        :rtype: keras.KerasTensor
        """
        if self._probability_type == "softmax":
            return self.strategy_layer(inputs, mask=mask)

        return self.strategy_layer(inputs, training=training)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """Return the strategy layer's output shape.

        Delegated, because the routing strategies change the last dimension
        and the other four do not.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        :return: Shape of the output tensor.
        :rtype: tuple
        """
        return self.strategy_layer.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to rebuild the layer.

        ``probability_type`` is written back lowercased, which is what
        ``__init__`` stored, not necessarily what the caller passed.

        :return: The base Layer config plus ``probability_type`` and
            ``type_config``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "probability_type": self._probability_type,
            # Through keras, so a nested keras object in type_config (say a
            # kernel_regularizer for hierarchical routing) survives the round
            # trip. A plain primitive dict comes back unchanged.
            "type_config": keras.saving.serialize_keras_object(self._type_config),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ProbabilityOutput":
        """Rebuild the layer from a config dict.

        Overridden because ``get_config`` ran ``type_config`` through
        ``serialize_keras_object``; this undoes that before calling the
        constructor. The input ``config`` is copied, not mutated.

        :param config: Config dictionary from ``get_config()``.
        :type config: Dict[str, Any]
        :return: A new ``ProbabilityOutput``.
        :rtype: ProbabilityOutput
        """
        config = dict(config)
        if config.get("type_config") is not None:
            config["type_config"] = keras.saving.deserialize_keras_object(
                config["type_config"]
            )
        return cls(**config)

    @property
    def probability_type(self) -> str:
        """Return the strategy name, lowercased.

        :return: The resolved ``probability_type``.
        :rtype: str
        """
        return self._probability_type

    @property
    def type_config(self) -> Dict[str, Any]:
        """Return a copy of the strategy configuration.

        A copy, so a caller mutating the result cannot change what this
        layer will serialize.

        :return: The ``type_config`` dict.
        :rtype: Dict[str, Any]
        """
        return self._type_config.copy()

# ---------------------------------------------------------------------

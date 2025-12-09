"""
Unified Probability Output Layer.

This module provides a wrapper layer that consolidates various probability
distribution mechanisms (Softmax, Sparsemax, Adaptive Softmax, Routing, ThreshMax)
into a single, configuration-driven component.

It allows for seamless experimentation with different output strategies by
changing a configuration string rather than rewriting model architecture code.

Supported Strategies:
    - **softmax**: Standard exponential normalization.
    - **sparsemax**: Euclidean projection onto simplex (sparse outputs).
    - **threshmax**: Differentiable confidence thresholding (sparse, rank-preserving).
    - **adaptive**: Entropy-based temperature scaling (sharpens uncertain predictions).
    - **routing**: Deterministic, parameter-free tree (input: features, not logits).
    - **hierarchical**: Learnable tree (input: features, not logits).

Usage Note:
    - ``softmax``, ``sparsemax``, ``threshmax``, and ``adaptive`` expect input **logits**.
    - ``routing`` and ``hierarchical`` expect input **features** (they replace the final Dense layer).
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
from .routing_probabilities_hierarchical import HierarchicalRoutingLayer


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
    """
    Unified wrapper for probability output layers.

    This layer serves as a factory and container for various activation and
    classification heads. It instantiates the specific strategy based on the
    ``probability_type`` argument.

    **Strategy Logic**:

    - **"softmax"**: Standard ``keras.layers.Softmax``.
      Input: Logits. Config: ``{"axis": -1}``.

    - **"sparsemax"**: ``Sparsemax`` (L2 projection).
      Input: Logits. Config: ``{"axis": -1}``.

    - **"threshmax"**: ``ThreshMax`` (Differentiable confidence thresholding).
      Input: Logits. Config: ``{"axis": -1, "slope": 10.0, "trainable_slope": False}``.

    - **"adaptive"**: ``AdaptiveTemperatureSoftmax``.
      Input: Logits. Config: ``{"min_temp": 0.1, "max_temp": 1.0, "entropy_threshold": 0.5}``.

    - **"routing"**: ``RoutingProbabilitiesLayer`` (Deterministic).
      Input: **Features** (Not logits). Config: ``{"output_dim": int, "axis": -1}``.

    - **"hierarchical"**: ``HierarchicalRoutingLayer`` (Trainable).
      Input: **Features** (Not logits). Config: ``{"output_dim": int, "axis": -1}``.

    **Architecture**::

        Input(shape=[..., logits_or_features])
               |
        Strategy Layer (softmax/sparsemax/threshmax/adaptive/routing/hierarchical)
               |
        Output(shape=[..., output_dim or same as input])

    Args:
        probability_type: String identifier for the output strategy.
            One of ``"softmax"``, ``"sparsemax"``, ``"threshmax"``, ``"adaptive"``,
            ``"routing"``, ``"hierarchical"``. Aliases ``"thresh_max"``,
            ``"adaptive_softmax"``, ``"deterministic_routing"``, and
            ``"hierarchical_routing"`` are also accepted. Defaults to ``"softmax"``.
        type_config: Optional dictionary containing arguments specific to the
            chosen strategy. Keys depend on the selected ``probability_type``.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: ``(batch_size, ..., logits_dim)`` for logit-based
        strategies, or ``(batch_size, ..., features_dim)`` for routing strategies.

    Output shape:
        N-D tensor with shape: ``(batch_size, ..., output_dim)``.
        For logit-based strategies, ``output_dim == logits_dim``.
        For routing strategies, ``output_dim`` is specified in ``type_config``.

    Attributes:
        strategy_layer: The instantiated probability layer based on configuration.

    Example:
        .. code-block:: python

            # 1. Standard Softmax
            outputs = ProbabilityOutput(probability_type="softmax")(logits)

            # 2. ThreshMax (Sparse, learnable threshold)
            outputs = ProbabilityOutput(
                probability_type="threshmax",
                type_config={"trainable_slope": True, "slope": 10.0}
            )(logits)

            # 3. Adaptive (Dynamic temperature based on uncertainty)
            outputs = ProbabilityOutput(
                probability_type="adaptive",
                type_config={"min_temp": 0.05, "entropy_threshold": 0.4}
            )(logits)

            # 4. Hierarchical Routing (Replaces Dense head)
            features = last_hidden_layer_output
            outputs = ProbabilityOutput(
                probability_type="hierarchical",
                type_config={"output_dim": 1000}
            )(features)

    Note:
        For ``routing`` and ``hierarchical`` types, the input should be features
        (e.g., from a hidden layer), not class logits, as these strategies
        perform their own internal projection.
    """

    # Supported probability types for validation
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
        """
        Initialize the ProbabilityOutput layer.

        Args:
            probability_type: String identifier for the output strategy.
            type_config: Optional dictionary containing strategy-specific arguments.
            **kwargs: Additional keyword arguments for the Layer base class.

        Raises:
            ValueError: If ``probability_type`` is not supported or if required
                configuration is missing for certain types.
        """
        super().__init__(**kwargs)

        self._probability_type = probability_type.lower()
        self._type_config = type_config if type_config is not None else {}

        # Validate probability type
        if self._probability_type not in self._SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown probability_type '{self._probability_type}'. "
                f"Supported types: {list(self._SUPPORTED_TYPES)}"
            )

        # Validate required config for hierarchical routing
        if self._probability_type in ("hierarchical", "hierarchical_routing"):
            if "output_dim" not in self._type_config:
                raise ValueError(
                    "ProbabilityOutput with type='hierarchical' requires "
                    "'output_dim' in type_config."
                )

        # Create the strategy layer in __init__ (per Keras 3 best practices)
        self.strategy_layer: keras.layers.Layer = self._create_strategy_layer()

    def _create_strategy_layer(self) -> keras.layers.Layer:
        """
        Instantiate the internal layer based on configuration.

        Returns:
            The instantiated strategy layer.
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
            return HierarchicalRoutingLayer(
                name="hierarchical_routing",
                **self._type_config
            )

        # Should never reach here due to validation in __init__
        raise ValueError(f"Unhandled probability_type: {self._probability_type}")

    def build(self, input_shape: tuple) -> None:
        """
        Build the internal strategy layer.

        This method explicitly builds the sub-layer to ensure proper weight
        creation before serialization/deserialization cycles.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        self.strategy_layer.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass delegating to the selected strategy.

        Args:
            inputs: Input tensor. Shape depends on strategy type:
                - For logit-based strategies: ``(batch_size, ..., num_classes)``
                - For routing strategies: ``(batch_size, ..., features_dim)``
            training: Boolean indicating training mode. Passed to strategy layer
                if it supports training-specific behavior.
            mask: Optional mask tensor. Currently only supported by softmax.

        Returns:
            Probability distribution tensor with same batch dimensions as input.
        """
        if self._probability_type == "softmax":
            return self.strategy_layer(inputs, mask=mask)

        return self.strategy_layer(inputs, training=training)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute output shape based on strategy.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Shape tuple of the output tensor.
        """
        return self.strategy_layer.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Configuration dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update({
            "probability_type": self._probability_type,
            "type_config": self._type_config,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ProbabilityOutput":
        """
        Reconstruct the layer from configuration.

        Args:
            config: Configuration dictionary from ``get_config()``.

        Returns:
            New instance of ProbabilityOutput.
        """
        return cls(**config)

    @property
    def probability_type(self) -> str:
        """Return the configured probability type."""
        return self._probability_type

    @property
    def type_config(self) -> Dict[str, Any]:
        """Return the type-specific configuration."""
        return self._type_config.copy()

# ---------------------------------------------------------------------

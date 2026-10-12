"""Configurable transition modules for the Neural HMM.

The transition holds the initial distribution ``log pi`` and either a static
transition matrix ``log A`` of shape ``(states, states)`` or an
input-dependent one of shape ``(batch, time, states, states)``. The type is
chosen through :data:`TRANSITION_REGISTRY`; the topology (which transitions
are allowed at all) is orthogonal to the type and is enforced by additive
masks of ``0`` / ``-1e9`` applied before the log-softmax, so forbidden
transitions carry exactly zero probability rather than a small one.

An input-dependent transition starts from a plain HMM: its context projection
is zero-initialized, so training begins at the static solution and only moves
away when the gradient pays for it.

References:
    - Rabiner, 1989. A Tutorial on Hidden Markov Models and Selected
      Applications in Speech Recognition. Proceedings of the IEEE.
      (https://doi.org/10.1109/5.18626)
    - Bengio and Frasconi, 1995. An Input Output HMM Architecture.
      NeurIPS. (https://proceedings.neurips.cc/paper/1994/hash/)
"""

import copy
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Transition types accepted by :func:`create_transition_layer`.
TransitionType = Literal["static", "input_dependent"]

#: Topologies accepted by the transition layers.
TopologyType = Literal["ergodic", "left_to_right", "banded"]

#: Additive logit for a forbidden transition: zero probability after softmax.
NEGATIVE_LOGIT: float = -1e9

#: Marker text for the strict unknown-keyword error, matching the FFN factory.
STRICT_DROPPED_KEY_MARKER: str = "unsupported parameter(s)"

#: Config keys that pass through :func:`assemble_transition_config` untouched.
_TRANSITION_CONFIG_PASSTHROUGH_KEYS = ("type", "name")


# ---------------------------------------------------------------------
# topology masks
# ---------------------------------------------------------------------


def transition_mask(n_states: int, topology: str, band_width: int = 1) -> np.ndarray:
    """Build the ``(states, states)`` additive mask for a topology.

    Allowed transitions carry ``0``; forbidden ones carry
    :data:`NEGATIVE_LOGIT`.

    :param n_states: Number of hidden states.
    :type n_states: int
    :param topology: ``ergodic`` (all transitions), ``left_to_right``
        (upper-triangular, no returns) or ``banded`` (upper-triangular with at
        most ``band_width`` forward skips).
    :type topology: str
    :param band_width: Maximum forward skip for ``banded``.
    :type band_width: int
    :return: Float32 ``(states, states)`` additive mask.
    :rtype: np.ndarray
    :raises ValueError: If the topology is unknown or ``band_width`` is
        negative.
    """
    if topology == "ergodic":
        allowed = np.ones((n_states, n_states))
    elif topology == "left_to_right":
        allowed = np.triu(np.ones((n_states, n_states)))
    elif topology == "banded":
        if band_width < 0:
            raise ValueError(f"band_width must be non-negative, got {band_width}")
        allowed = np.triu(np.ones((n_states, n_states))) - np.triu(
            np.ones((n_states, n_states)), k=band_width + 1
        )
    else:
        raise ValueError(
            f"Unknown topology '{topology}'. "
            f"Available: ['ergodic', 'left_to_right', 'banded']."
        )
    return np.where(allowed > 0, 0.0, NEGATIVE_LOGIT).astype("float32")


def initial_mask(n_states: int, topology: str) -> np.ndarray:
    """Build the ``(states,)`` additive mask for the initial distribution.

    Non-ergodic topologies must start in state 0; ergodic ones may start
    anywhere.

    :param n_states: Number of hidden states.
    :type n_states: int
    :param topology: One of ``ergodic``, ``left_to_right``, ``banded``.
    :type topology: str
    :return: Float32 ``(states,)`` additive mask.
    :rtype: np.ndarray
    :raises ValueError: If the topology is unknown.
    """
    if topology not in ("ergodic", "left_to_right", "banded"):
        raise ValueError(
            f"Unknown topology '{topology}'. "
            f"Available: ['ergodic', 'left_to_right', 'banded']."
        )
    mask = np.zeros(n_states, dtype="float32")
    if topology != "ergodic":
        mask[1:] = NEGATIVE_LOGIT
    return mask


# ---------------------------------------------------------------------
# transition layers
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.neural_hmm.transitions")
class StaticTransition(keras.layers.Layer):
    """A time-invariant transition matrix with a topology mask.

    :param n_states: Number of hidden states.
    :type n_states: int
    :param topology: Allowed-transition topology.
    :type topology: str
    :param band_width: Maximum forward skip for ``banded``.
    :type band_width: int
    :param self_loop_init: Diagonal bias of the transition logits at
        initialization; 1 to 3 keeps states sticky early and avoids flicker.
    :type self_loop_init: float
    """

    def __init__(
        self,
        n_states: int,
        topology: str = "ergodic",
        band_width: int = 1,
        self_loop_init: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if n_states <= 0:
            raise ValueError(f"n_states must be positive, got {n_states}")
        if band_width < 0:
            raise ValueError(f"band_width must be non-negative, got {band_width}")
        self.n_states = n_states
        self.topology = topology
        self.band_width = band_width
        self.self_loop_init = self_loop_init

        # Deterministic functions of the config: plain tensors, rebuilt from
        # the config by from_config, never serialized as weights.
        self.pi_mask = keras.ops.convert_to_tensor(
            initial_mask(n_states, topology)
        )
        self.transition_mask = keras.ops.convert_to_tensor(
            transition_mask(n_states, topology, band_width)
        )
        self.pi_logits = self.add_weight(
            shape=(n_states,), initializer="zeros", name="pi_logits"
        )
        self.transition_logits = self.add_weight(
            shape=(n_states, n_states),
            initializer=keras.initializers.Constant(
                np.eye(n_states, dtype="float32") * self_loop_init
            ),
            name="transition_logits",
        )

    def build(self, input_shape: Any) -> None:
        """Mark the layer built; weights already exist from ``__init__``.

        :param input_shape: ``(batch, time, hidden_dim)``; unused beyond the
            built flag, since the parameters do not depend on it.
        :type input_shape: Any
        """
        if self.built:
            return
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Any:
        """Return the log initial distribution and log transition matrix.

        :param inputs: Context features ``(batch, time, hidden_dim)``; ignored
            by the static transition, accepted for a uniform call signature.
        :type inputs: keras.KerasTensor
        :param training: Unused.
        :type training: Optional[bool]
        :return: ``(log_pi, log_transition)`` with shapes ``(states,)`` and
            ``(states, states)``.
        :rtype: Any
        """
        log_pi = keras.ops.log_softmax(self.pi_logits + self.pi_mask)
        log_transition = keras.ops.log_softmax(
            self.transition_logits + self.transition_mask, axis=-1
        )
        return log_pi, log_transition

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Return the ``(log_pi, log_transition)`` shapes.

        :param input_shape: ``(batch, time, hidden_dim)``.
        :type input_shape: Any
        :return: ``((states,), (states, states))``.
        :rtype: Any
        """
        return ((self.n_states,), (self.n_states, self.n_states))

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "topology": self.topology,
                "band_width": self.band_width,
                "self_loop_init": self.self_loop_init,
            }
        )
        return config


@register_dl_technique("dl_techniques.models.neural_hmm.transitions")
class InputDependentTransition(keras.layers.Layer):
    """A transition matrix modulated by the context at each timestep.

    Adds a zero-initialized context offset to the static logits, so training
    starts from a plain HMM and the dependence only appears when the gradient
    pays for it. This is the IOHMM / regime-switching regime.

    :param n_states: Number of hidden states.
    :type n_states: int
    :param hidden_dim: Width of the context features.
    :type hidden_dim: int
    :param topology: Allowed-transition topology.
    :type topology: str
    :param band_width: Maximum forward skip for ``banded``.
    :type band_width: int
    :param self_loop_init: Diagonal bias of the static logits at
        initialization.
    :type self_loop_init: float
    :param use_bias: Whether the context projection uses bias.
    :type use_bias: bool
    """

    def __init__(
        self,
        n_states: int,
        hidden_dim: int,
        topology: str = "ergodic",
        band_width: int = 1,
        self_loop_init: float = 2.0,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if n_states <= 0:
            raise ValueError(f"n_states must be positive, got {n_states}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if band_width < 0:
            raise ValueError(f"band_width must be non-negative, got {band_width}")
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        self.topology = topology
        self.band_width = band_width
        self.self_loop_init = self_loop_init
        self.use_bias = use_bias

        self.pi_mask = keras.ops.convert_to_tensor(
            initial_mask(n_states, topology)
        )
        self.transition_mask = keras.ops.convert_to_tensor(
            transition_mask(n_states, topology, band_width)
        )
        self.pi_logits = self.add_weight(
            shape=(n_states,), initializer="zeros", name="pi_logits"
        )
        self.transition_logits = self.add_weight(
            shape=(n_states, n_states),
            initializer=keras.initializers.Constant(
                np.eye(n_states, dtype="float32") * self_loop_init
            ),
            name="transition_logits",
        )
        # Zero-initialized: the model starts as a plain HMM.
        self.context_projection = keras.layers.Dense(
            n_states * n_states,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            use_bias=use_bias,
            name="context_projection",
        )

    def build(self, input_shape: Any) -> None:
        """Build the context projection against the context width.

        :param input_shape: ``(batch, time, hidden_dim)``.
        :type input_shape: Any
        :raises ValueError: If the last axis does not match ``hidden_dim``.
        """
        if self.built:
            return
        shape = tuple(input_shape)
        if shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Context width ({shape[-1]}) must match hidden_dim "
                f"({self.hidden_dim})"
            )
        self.context_projection.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Any:
        """Return the log initial distribution and per-step log transitions.

        :param inputs: Context features ``(batch, time, hidden_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the context projection.
        :type training: Optional[bool]
        :return: ``(log_pi, log_transition)`` with shapes ``(states,)`` and
            ``(batch, time, states, states)``.
        :rtype: Any
        """
        log_pi = keras.ops.log_softmax(self.pi_logits + self.pi_mask)
        offset = self.context_projection(inputs, training=training)
        batch = keras.ops.shape(offset)[0]
        time = keras.ops.shape(offset)[1]
        offset = keras.ops.reshape(
            offset, (batch, time, self.n_states, self.n_states)
        )
        logits = self.transition_logits + self.transition_mask + offset
        return log_pi, keras.ops.log_softmax(logits, axis=-1)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Return the ``(log_pi, log_transition)`` shapes.

        :param input_shape: ``(batch, time, hidden_dim)``.
        :type input_shape: Any
        :return: ``((states,), (batch, time, states, states))``.
        :rtype: Any
        """
        shape = tuple(input_shape)
        batch, time = shape[0], shape[1]
        return (
            (self.n_states,),
            (batch, time, self.n_states, self.n_states),
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "hidden_dim": self.hidden_dim,
                "topology": self.topology,
                "band_width": self.band_width,
                "self_loop_init": self.self_loop_init,
                "use_bias": self.use_bias,
            }
        )
        return config


# ---------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------

#: Registry of transition types. Every entry carries ``class``,
#: ``description``, ``required_params``, ``optional_params`` and ``use_case``,
#: mirroring the FFN factory.
TRANSITION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "static": {
        "class": StaticTransition,
        "description": "Time-invariant transition matrix: a plain HMM. "
        "Declares no hidden_dim: the model routes its wrapper config through "
        "assemble_transition_config, which drops the context width for this "
        "type before it reaches the constructor.",
        "required_params": ["n_states"],
        "optional_params": {
            "topology": "ergodic",
            "band_width": 1,
            "self_loop_init": 2.0,
        },
        "use_case": "Stationary dynamics; the default.",
    },
    "input_dependent": {
        "class": InputDependentTransition,
        "description": "Per-timestep transition matrix modulated by a "
        "zero-initialized context projection (IOHMM / regime switching).",
        "required_params": ["n_states", "hidden_dim"],
        "optional_params": {
            "topology": "ergodic",
            "band_width": 1,
            "self_loop_init": 2.0,
            "use_bias": True,
        },
        "use_case": "Non-stationary dynamics driven by the context.",
    },
}


def get_transition_info() -> Dict[str, Dict[str, Any]]:
    """Return a deep copy of the transition registry.

    :return: The registry, safe to mutate.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return copy.deepcopy(TRANSITION_REGISTRY)


def validate_transition_config(transition_type: str, **kwargs: Any) -> None:
    """Validate a transition type and its keyword arguments.

    :param transition_type: A :data:`TRANSITION_REGISTRY` key.
    :type transition_type: str
    :param kwargs: Candidate constructor arguments.
    :type kwargs: Any
    :raises ValueError: If the type is unknown or a required parameter is
        missing.
    """
    info = TRANSITION_REGISTRY.get(transition_type)
    if info is None:
        raise ValueError(
            f"Unknown transition_type '{transition_type}'. "
            f"Available: {sorted(TRANSITION_REGISTRY)}."
        )
    missing = [p for p in info["required_params"] if p not in kwargs]
    if missing:
        raise ValueError(
            f"create_transition_layer('{transition_type}') is missing required "
            f"parameter(s) {missing}."
        )


def assemble_transition_config(
    transition_type: str,
    wrapper_config: Mapping[str, Any],
    caller_args: Optional[Mapping[str, Any]] = None,
    *,
    passthrough: Sequence[str] = _TRANSITION_CONFIG_PASSTHROUGH_KEYS,
) -> Dict[str, Any]:
    """Filter a wrapper's generic transition defaults, then merge the caller's.

    :param transition_type: A :data:`TRANSITION_REGISTRY` key.
    :type transition_type: str
    :param wrapper_config: The wrapper's own generic config; filtered.
    :type wrapper_config: Mapping[str, Any]
    :param caller_args: The caller's explicit args; never filtered.
    :type caller_args: Optional[Mapping[str, Any]]
    :param passthrough: Keys kept regardless of the registry intersection.
    :type passthrough: Sequence[str]
    :return: The assembled config dict for :func:`create_transition_layer`.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``transition_type`` is not registered.
    """
    info = TRANSITION_REGISTRY.get(transition_type)
    if info is None:
        raise ValueError(
            f"Unknown transition_type '{transition_type}'. "
            f"Available: {sorted(TRANSITION_REGISTRY)}."
        )
    accepted = (
        set(info["required_params"])
        | set(info["optional_params"])
        | set(passthrough)
    )
    config = {k: v for k, v in wrapper_config.items() if k in accepted}
    if caller_args:
        config.update(caller_args)
    return config


def create_transition_layer(
    transition_type: TransitionType,
    name: Optional[str] = None,
    **kwargs: Any,
) -> keras.layers.Layer:
    """Create a transition layer by type name.

    :param transition_type: A :data:`TRANSITION_REGISTRY` key.
    :type transition_type: TransitionType
    :param name: Optional layer name.
    :type name: Optional[str]
    :param kwargs: Constructor arguments for the transition class.
    :type kwargs: Any
    :return: An unbuilt transition layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If the type is unknown, a required parameter is
        missing, or an undeclared keyword is passed. Undeclared keywords
        raise; they are never silently dropped.
    """
    info = TRANSITION_REGISTRY.get(transition_type)
    if info is None:
        raise ValueError(
            f"Unknown transition_type '{transition_type}'. "
            f"Available: {sorted(TRANSITION_REGISTRY)}."
        )
    valid = set(info["required_params"]) | set(info["optional_params"])
    dropped = sorted(set(kwargs) - valid)
    if dropped:
        raise ValueError(
            f"create_transition_layer('{transition_type}'): "
            f"{len(dropped)} {STRICT_DROPPED_KEY_MARKER} {dropped}. "
            f"'{transition_type}' ({info['class'].__name__}) accepts only "
            f"{sorted(valid)}. Pre-filter wrapper defaults with "
            f"assemble_transition_config() instead of passing them here."
        )
    validate_transition_config(transition_type, **kwargs)
    # The static transition ignores the context width; it is accepted in the
    # registry so one wrapper config serves both types, and dropped here
    # rather than forwarded into a constructor that does not declare it.
    if transition_type == "static":
        kwargs.pop("hidden_dim", None)
    try:
        return info["class"](name=name, **kwargs)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Failed to create {transition_type} transition "
            f"({info['class'].__name__}). Original error: {e}"
        ) from e


def create_transition_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """Create a transition layer from a config dict carrying ``type``.

    :param config: Config dict with a ``type`` key; ``name`` flows through.
    :type config: Dict[str, Any]
    :return: An unbuilt transition layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``config`` is not a dict or lacks ``type``.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary, got {type(config)}")
    if "type" not in config:
        raise ValueError("Configuration must include 'type' key")
    config_copy = config.copy()
    transition_type = config_copy.pop("type")
    return create_transition_layer(transition_type, **config_copy)

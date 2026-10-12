"""Configurable emission modules for the Neural HMM.

The emission module maps the observation and the state-context features to a
per-state log-density. Two shapes are supported, driven by the ``obs_type``
config key:

- ``gaussian``: each state emits a Gaussian density :math:`\mathcal{N}(\mu_k,
  \sigma_k^2)`. The per-state parameters ``(\mu_k, \sigma_k)` are derived from a
  learned ``(K, E)`` state embedding, optionally conditioned on the encoder
  context ``h_t`` when ``emission_context`` is ``autoregressive``.

- ``categorical``: each state emits a categorical over a vocabulary of
  ``vocab_size`` tokens. The per-state logits are produced from the same state
  embedding, again optionally conditioned on context.

- ``poisson``: each state emits a Poisson distribution :math:`\text{Pois}(\lambda_k)`
  with rate ``\lambda_k > 0``. The rate is obtained from a softplus of the
  params output; stable training requires ``min_scale`` > 0.

All three share the ``state_emb`` / ``head`` pair; the only branch is the
mapping from params to a per-state log-density. The emission_context
``autoregressive`` mode was originally designed for the generative regime so
the encoder ``h_t`` only ever sees the past; when used discriminatively the
context is supplied directly from the (possibly bidirectional) encoder output.

References:
    - Bishop, 1994. Mixture Density Networks. Aston University Technical Report NCRG/94/004.
    - Tang et al., 2020. Neural HMMs for regime-switching time-series.
      (https://arxiv.org/abs/2006.00694)
    - Chen and Saad, 2016. Gaussian HMM with learnable state emission densities.
      Signal Processing.
"""

import copy
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.ffn.factory import create_ffn_layer, assemble_ffn_config

# ---------------------------------------------------------------------

#: Emission types accepted by :func:`create_emission_layer`.
EmissionType = Literal["gaussian", "categorical", "poisson"]

#: Additive logit for a forbidden transition: zero probability after softmax.
NEGATIVE_LOGIT: float = -1e9

#: Minimum scale clamp for Gaussian and Poisson emissions; avoids degenerate
#  densities where ``sigma`` or ``lambda`` collapse to zero.
MIN_SCALE: float = 1e-3

#: Marker text for the strict unknown-keyword error, matching the FFN factory.
STRICT_DROPPED_KEY_MARKER: str = "unsupported parameter(s)"

#: Config keys that pass through :func:`assemble_emission_config` untouched.
_EMISSION_CONFIG_PASSTHROUGH_KEYS = ("type", "name")


# ---------------------------------------------------------------------
# per-state log-density helpers
# ---------------------------------------------------------------------


def _gaussian_logprob(x: keras.KerasTensor, mu: keras.KerasTensor,
                       raw_sd: keras.KerasTensor) -> keras.KerasTensor:
    """Gaussian log-density per observation, summed over the feature dim.

    :math:`-\frac{1}{2}\Bigl(\frac{x-\mu}{\sigma}\Bigr)^2 - \log\sigma
    - \frac{1}{2}\log(2\pi)`

    :param x: Observation tensor ``(B, T, D)``.
    :type x: keras.KerasTensor
    :param mu: Mean per state ``(B, T, K, D)``.
    :type mu: keras.KerasTensor
    :param raw_sd: Raw scale before softplus ``(B, T, K, D)``.
    :type raw_sd: keras.KerasTensor
    :return: Log-probability per observation ``(B, T, K)``, summed over D.
    :rtype: keras.KerasTensor
    """
    sd = keras.ops.softplus(raw_sd) + MIN_SCALE
    z = (x[:, :, None] - mu) / sd
    logprob = -0.5 * keras.ops.square(z) - keras.ops.log(sd) - 0.5 * np.log(2 * np.pi)
    return keras.ops.sum(logprob, axis=-1)


def _categorical_logprob(x: keras.KerasTensor, logits: keras.KerasTensor,
                          vocab_size: int) -> keras.KerasTensor:
    """Categorical log-density per observation, summed over the vocab dim.

    :math:`\log p(x_t \mid z_t = k) = \log\text{softmax}(logits)_k` at
    the index ``x_t``.

    :param x: Integer observation tensor ``(B, T)``; each entry ``in [0,
        vocab_size)``.
    :type x: keras.KerasTensor
    :param logits: Per-state logits ``(B, T, K, vocab_size)``.
    :type logits: keras.KerasTensor
    :param vocab_size: Size of the observation vocabulary.
    :type vocab_size: int
    :return: Log-probability per observation ``(B, T, K)``, un-reduced.
    :rtype: keras.KerasTensor
    """
    logp = keras.ops.log_softmax(logits, axis=-1)
    onehot = keras.ops.one_hot(x, vocab_size)[:, :, None]  # (B,T,1,V)
    # logp is (B,T,K,V) after broadcasting; select the token class and sum over V=1
    return keras.ops.sum(logp * onehot, axis=-1)


def _poisson_logprob(x: keras.KerasTensor, rate: keras.KerasTensor) -> keras.KerasTensor:
    """Poisson log-density per observation, un-reduced.

    :math:`-\lambda + x \log\lambda - \log(x!)`.  The ``\log(x!)`` term is
    constant w.r.t. the model parameters, so it is omitted (it does not affect
    the gradient) and the return is the *unnormalized* log-probability that
    ``forward_logprob`` (below) adds into.

    :param x: Observation tensor ``(B, T)``, non-negative integers.
    :type x: keras.KerasTensor
    :param rate: Rate per state ``(B, T, K)``, strictly positive.
    :type rate: keras.KerasTensor
    :return: Un-normalized log-probability per observation ``(B, T, K)``.
    :rtype: keras.KerasTensor
    """
    return -rate + x[:, :, None] * keras.ops.log(rate)


# ---------------------------------------------------------------------
# emission layers
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.neural_hmm.emissions")
class GenerativeEmission(keras.layers.Layer):
    """Generative emission: ``log p(x_t | z_t = k)`` in the forward pass.

    Each state ``k`` emits a density over the observation. The density type
    (Gaussian / categorical / Poisson) is chosen at construction time through
    the ``obs_type`` config key; all three share the same ``state_emb`` /
    ``head`` backbone.

    When ``emission_context`` is ``autoregressive`` the encoder context
    ``h_t`` (causal, past-only) is broadcast and concatenated with the state
    embedding before the head, so each timestep sees a different context. When
    ``emission_context`` is ``none`` the same parameters are used for every
    timestep (the embedding alone sets the per-state density).

    The ``state_emb`` is a ``(K, E``) embedding looked up by state index; it is
    linearly projected by ``head`` to the parameter vector whose meaning depends
    on ``obs_type``:

    - ``gaussian``: ``head`` outputs ``2 * obs_dim``, split into ``mu`` and
      ``raw_sd``; ``sigma = softplus(raw_sd) + min_scale``.
    - ``categorical``: ``head`` outputs ``vocab_size`` logits per state.
    - ``poisson``: ``head`` outputs 1 value per state, the Poisson rate.

    In the discriminative regime the ``x`` argument is still supplied but only
    needed for the Gaussian branch (computes the log-likelihood). The ``h``
    argument carries the encoder context from the (possibly bidirectional)
    encoder.

    :param n_states: Number of hidden states ``K``.
    :type n_states: int
    :param hidden_dim: Dimensionality of the encoder context ``H``.
    :type hidden_dim: int
    :param state_emb_dim: Dimensionality of the state embedding ``E``.
    :type state_emb_dim: int
    :param obs_type: Observation type (``gaussian``, ``categorical``, ``poisson``).
    :type obs_type: str
    :param obs_dim: Observation dimension for Gaussian emissions.
    :type obs_dim: Optional[int]
    :param vocab_size: Vocabulary size for categorical emissions.
    :type vocab_size: Optional[int]
    :param emission_context: Whether to condition on autoregressive context (``none`` or ``autoregressive``).
    :type emission_context: str
    :param min_scale: Minimum scale for Gaussian/Poisson emissions.
    :type min_scale: float
    """

    def __init__(
        self,
        n_states: int,
        hidden_dim: int,
        state_emb_dim: int,
        obs_type: str,
        obs_dim: Optional[int] = None,
        vocab_size: Optional[int] = None,
        emission_context: str = "none",
        min_scale: float = MIN_SCALE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        self.state_emb_dim = state_emb_dim
        self.obs_type = obs_type
        self.obs_dim = obs_dim
        self.vocab_size = vocab_size
        self.emission_context = emission_context
        self.min_scale = min_scale

        # State embedding lookup table: (K, E)
        self.state_emb = self.add_weight(
            shape=(self.n_states, self.state_emb_dim),
            initializer="random_normal",
            name="state_emb",
        )

        # Parameter head: maps state embedding + optional context -> params
        P = {
            "gaussian": 2 * (self.obs_dim or 1),
            "categorical": self.vocab_size or 1,
            "poisson": 1,
        }[self.obs_type]

        self.head = keras.Sequential(
            [keras.layers.Dense(self.hidden_dim, activation="gelu"),
             keras.layers.Dense(P)],
            name="emission_head",
        )

    def call(self, x: keras.KerasTensor, h: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the per-state log-probability ``log p(x_t | z_t = k)``.

        :param x: Observation tensor ``(batch, time)`` of ints (gaussian: float
            ``(B, T, D)``, categorical: int, poisson: int).
        :type x: keras.KerasTensor
        :param h: Encoder context ``(batch, time, hidden_dim)``; ignored when
            ``emission_context`` is ``none``.
        :type h: keras.KerasTensor
        :return: Per-state log-probability ``(batch, time, K)``; each column k
            is ``log p(x_t | z_t = k)``.
        :rtype: keras.KerasTensor
        """
        K, E = self.n_states, self.state_emb_dim
        H = self.hidden_dim
        emb = keras.ops.broadcast_to(
            self.state_emb[None, None], (keras.ops.shape(x)[0], keras.ops.shape(x)[1], K, E)
        )
        if self.emission_context == "autoregressive" and h is not None:
            # causal context broadcast over states and time
            ctx = keras.ops.broadcast_to(
                h[:, :, None], (keras.ops.shape(x)[0], keras.ops.shape(x)[1], K, H)
            )
            params = self.head(keras.ops.concatenate([emb, ctx], axis=-1))
        else:
            params = self.head(emb)  # [1,1,K,P]

        if self.obs_type == "gaussian":
            mu, raw = keras.ops.split(params, 2, axis=-1)
            sd = keras.ops.softplus(raw) + self.min_scale
            # x may be (B,T,D) float or squeezed; unify to (B,T,D) if needed
            if x.shape.rank == 2:
                x_expanded = x[:, :, None]
            else:
                x_expanded = x
            return _gaussian_logprob(x_expanded, mu, raw)

        elif self.obs_type == "categorical":
            logits = params
            return _categorical_logprob(x, logits, self.vocab_size)

        elif self.obs_type == "poisson":
            rate = keras.ops.exp(params)
            rate = keras.ops.maximum(rate, MIN_SCALE)
            return _poisson_logprob(x, rate)

        else:
            raise ValueError(f"Unknown obs_type '{self.obs_type}'. "
                             f"Available: ['gaussian', 'categorical', 'poisson'].")

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Return ``(batch, time, K)``.

        :param input_shape: ``(batch, time)`` for int observations; ``(batch,
            time, D)`` for gaussian floats.
        :type input_shape: Any
        :return: ``(batch, time, K)``.
        :rtype: Any
        """
        shape = tuple(input_shape)
        return (shape[0], shape[1], self.n_states)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = {
            "type": "generative_emission",
            "n_states": self.n_states,
            "obs_type": self.obs_type,
            "obs_dim": self.obs_dim,
            "vocab_size": self.vocab_size,
            "emission_context": self.emission_context,
            "min_scale": self.min_scale,
            "hidden_dim": self.hidden_dim,
            "state_emb_dim": self.state_emb_dim,
        }
        base = super().get_config()
        return {**base, **config}


@register_dl_technique("dl_techniques.models.neural_hmm.emissions")
class DiscriminativeEmission(keras.layers.Layer):
    """Discriminative emission: state scores ``Dense(K)`` from ``h_t``.

    In the discriminative / CRF-style mode the model maximizes
    ``log p(y_{1:T} | x_{1:T})``. The emission does not produce a density over
    the observations; instead it emits a vector of logits (one per state) from
    the current encoder context ``h_t``.

    The emission layer is simply ``Dense(n_states)`` with no activation; the
    caller applies ``log_softmax`` per timestep when computing the forward
    algorithm score or the Viterbi score.

    :param n_states: Number of hidden states ``K``.
    :type n_states: int
    :param hidden_dim: Dimensionality of the encoder context ``H``.
    :type hidden_dim: int
    """

    def __init__(
        self,
        n_states: int,
        hidden_dim: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_states = n_states
        self.hidden_dim = hidden_dim

        self.dense = keras.layers.Dense(
            self.n_states,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name="state_scores",
        )

    def call(self, h: keras.KerasTensor) -> keras.KerasTensor:
        """Emit state scores from the encoder context.

        :param h: Encoder context ``(batch, time, hidden_dim)``.
        :type h: keras.KerasTensor
        :return: State scores ``(batch, time, n_states)``, unactivated.
        :rtype: keras.KerasTensor
        """
        return self.dense(h)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Return ``(batch, time, n_states)``.

        :param input_shape: ``(batch, time, hidden_dim)``.
        :type input_shape: Any
        :return: ``(batch, time, n_states)``.
        :rtype: Any
        """
        shape = tuple(input_shape)
        return (shape[0], shape[1], self.n_states)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = {"type": "discriminative_emission", "n_states": self.n_states}
        base = super().get_config()
        return {**base, **config}


# ---------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------

#: Registry of emission types. Every entry carries ``class``,
#: ``description``, ``required_params``, ``optional_params`` and ``use_case``,
#: mirroring the FFN factory. The key set, the :data:`EmissionType` alias and
#: each entry's parameter lists are public API.
EMISSION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gaussian": {
        "class": GenerativeEmission,
        "description": "Per-state Gaussian density with optional "
        "autoregressive context conditioning (generative mode).",
        "required_params": ["n_states", "hidden_dim", "state_emb_dim",
                           "obs_dim"],
        "optional_params": {
            "emission_context": "none",
            "min_scale": 1e-3,
        },
        "use_case": "Generative HMM: each state emits a Gaussian density.",
    },
    "categorical": {
        "class": GenerativeEmission,
        "description": "Per-state categorical over a vocabulary, optional "
        "autoregressive context conditioning.",
        "required_params": ["n_states", "hidden_dim", "state_emb_dim",
                           "vocab_size"],
        "optional_params": {
            "emission_context": "none",
        },
        "use_case": "Generative HMM with discrete emissions (e.g. text).",
    },
    "poisson": {
        "class": GenerativeEmission,
        "description": "Per-state Poisson rate with optional "
        "autoregressive context conditioning.",
        "required_params": ["n_states", "hidden_dim", "state_emb_dim"],
        "optional_params": {
            "emission_context": "none",
            "min_scale": 1e-3,
        },
        "use_case": "Generative HMM with count data.",
    },
    "discriminative": {
        "class": DiscriminativeEmission,
        "description": "State scores from encoder context, no observation "
        "density (CRF-style tagger). The caller applies log_softmax per timestep.",
        "required_params": ["n_states", "hidden_dim"],
        "optional_params": {},
        "use_case": "Discriminative HMM / neural CRF: maximize "
        "log p(y | x).",
    },
}


def get_emission_info() -> Dict[str, Dict[str, Any]]:
    """Return a deep copy of the emission registry.

    :return: The registry, safe to mutate.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return copy.deepcopy(EMISSION_REGISTRY)


def validate_emission_config(emission_type: str, **kwargs: Any) -> None:
    """Validate an emission type and its keyword arguments.

    :param emission_type: A :data:`EMISSION_REGISTRY` key.
    :type emission_type: str
    :param kwargs: Candidate constructor arguments.
    :type kwargs: Any
    :raises ValueError: If the type is unknown or a required parameter is
        missing.
    """
    info = EMISSION_REGISTRY.get(emission_type)
    if info is None:
        raise ValueError(
            f"Unknown emission_type '{emission_type}'. "
            f"Available: {sorted(EMISSION_REGISTRY)}."
        )
    missing = [p for p in info["required_params"] if p not in kwargs]
    if missing:
        raise ValueError(
            f"create_emission_layer('{emission_type}') is missing required "
            f"parameter(s) {missing}."
        )


def assemble_emission_config(
    emission_type: str,
    wrapper_config: Mapping[str, Any],
    caller_args: Optional[Mapping[str, Any]] = None,
    *,
    passthrough: Sequence[str] = _EMISSION_CONFIG_PASSTHROUGH_KEYS,
) -> Dict[str, Any]:
    """Filter a wrapper's generic emission defaults, then merge the caller's.

    :param emission_type: A :data:`EMISSION_REGISTRY` key.
    :type emission_type: str
    :param wrapper_config: The wrapper's own generic config; filtered.
    :type wrapper_config: Mapping[str, Any]
    :param caller_args: The caller's explicit args; never filtered.
    :type caller_args: Optional[Mapping[str, Any]]
    :param passthrough: Keys kept regardless of the registry intersection.
    :type passthrough: Sequence[str]
    :return: The assembled config dict for :func:`create_emission_layer`.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``emission_type`` is not registered.
    """
    info = EMISSION_REGISTRY.get(emission_type)
    if info is None:
        raise ValueError(
            f"Unknown emission_type '{emission_type}'. "
            f"Available: {sorted(EMISSION_REGISTRY)}."
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


def create_emission_layer(
    emission_type: EmissionType,
    name: Optional[str] = None,
    **kwargs: Any,
) -> keras.layers.Layer:
    """Create an emission layer by type name.

    :param emission_type: A :data:`EMISSION_REGISTRY` key.
    :type emission_type: EmissionType
    :param name: Optional layer name.
    :type name: Optional[str]
    :param kwargs: Constructor arguments for the emission class.
    :type kwargs: Any
    :return: An unbuilt emission layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If the type is unknown, a required parameter is
        missing, or an undeclared keyword is passed. Undeclared keywords
        raise; they are never silently dropped.
    """
    info = EMISSION_REGISTRY.get(emission_type)
    if info is None:
        raise ValueError(
            f"Unknown emission_type '{emission_type}'. "
            f"Available: {sorted(EMISSION_REGISTRY)}."
        )
    valid = set(info["required_params"]) | set(info["optional_params"])
    dropped = sorted(set(kwargs) - valid)
    if dropped:
        raise ValueError(
            f"create_emission_layer('{emission_type}'): "
            f"{len(dropped)} {STRICT_DROPPED_KEY_MARKER} {dropped}. "
            f"'{emission_type}' ({info['class'].__name__}) accepts only "
            f"{sorted(valid)}. Pre-filter wrapper defaults with "
            f"assemble_emission_config() instead of passing them here."
        )
    validate_emission_config(emission_type, **kwargs)
    # Add obs_type to kwargs so the class knows which emission type it is
    kwargs_with_type = dict(kwargs)
    kwargs_with_type["obs_type"] = emission_type
    try:
        return info["class"](name=name, **kwargs_with_type)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Failed to create {emission_type} emission "
            f"({info['class'].__name__}). Original error: {e}"
        ) from e


def create_emission_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """Create an emission layer from a config dict carrying ``type``.

    :param config: Config dict with a ``type`` key; ``name`` flows through.
    :type config: Dict[str, Any]
    :return: An unbuilt emission layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``config`` is not a dict or lacks ``type``.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary, got {type(config)}")
    if "type" not in config:
        raise ValueError("Configuration must include 'type' key")
    config_copy = config.copy()
    emission_type = config_copy.pop("type")
    return create_emission_layer(emission_type, **config_copy)
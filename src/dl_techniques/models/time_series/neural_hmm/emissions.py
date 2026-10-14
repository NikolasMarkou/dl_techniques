"""
Emission layers of the Neural HMM: the per-state log-potential ``log_emit`` of every
observation, of shape ``(B, T, K)``.

- :class:`GenerativeEmission` scores ``log p(x_t | z_t = k)``, a normalized density. Each
  state ``k`` owns a learned embedding ``e_k``; an ``mlp`` feed-forward head built by
  :func:`dl_techniques.layers.ffn.factory.create_ffn_layer` maps it, optionally
  concatenated with a per-step context ``h_t``, to the parameters of the state's
  distribution (Tran et al., 2016):

  - ``gaussian``: a diagonal Gaussian over ``obs_dim`` features, mean ``mu`` and scale
    ``softplus(raw) + min_scale``;
  - ``categorical``: a categorical over ``vocab_size`` symbols, ``log_softmax`` of the
    head's logits; a symbol outside ``[0, vocab_size)`` scores the finite
    :func:`~dl_techniques.utils.dtype_policy.mask_sentinel` (``-1e9`` in float32), so
    it can never pass as a probable observation;
  - ``poisson``: independent Poisson counts over ``obs_dim`` features, rate
    ``softplus(raw) + min_scale``, including the ``-log x!`` term, so the density is
    normalized.

  Without a context the head runs once on the ``(K, E)`` embedding table and every step
  shares the parameters; with one it runs on a ``(B, T, K, E + H)`` tensor.
- :class:`DiscriminativeEmission` scores each state from a ``(B, T, H)`` context with one
  ``Dense(K)``: unnormalized potentials for a linear-chain CRF (Lample et al., 2016).

Every output is in the accumulation dtype (float32 under a float16 policy): the head runs in
the compute dtype, and the density math runs after a cast, because a Gaussian log-density
``-0.5 * ((x - mu) / sd) ** 2`` overflows float16 once ``|x - mu| / sd`` exceeds about 362.

References:
    - Tran, Bisk, Vaswani, Marcu and Knight, 2016. Unsupervised Neural Hidden Markov
      Models. Proceedings of the Workshop on Structured Prediction for NLP.
      arXiv:1609.09007.
    - Lample, Ballesteros, Subramanian, Kawakami and Dyer, 2016. Neural Architectures for
      Named Entity Recognition. NAACL-HLT. arXiv:1603.01360.
"""

from typing import Any, Optional, Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn.factory import create_ffn_layer
from dl_techniques.utils.dtype_policy import accumulation_dtype, mask_sentinel
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tensors import log_gamma

from .inference import LogSpaceOutputSpec

# ---------------------------------------------------------------------

#: Observation types :class:`GenerativeEmission` accepts.
OBS_TYPES: Tuple[str, ...] = ("gaussian", "categorical", "poisson")

#: ``0.5 * log(2 * pi)``, the Gaussian normalizer per feature.
_HALF_LOG_TWO_PI: float = 0.5 * float(np.log(2.0 * np.pi))

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.neural_hmm.emissions")
class GenerativeEmission(LogSpaceOutputSpec, keras.layers.Layer):
    """Per-state observation log-densities ``log p(x_t | z_t = k)``.

    :param n_states: Number of hidden states ``K``; positive.
    :param obs_type: One of ``gaussian``, ``categorical``, ``poisson``.
    :param obs_dim: Feature count ``D`` of a ``gaussian`` or ``poisson`` observation;
        required for those types and must be ``None`` for ``categorical``.
    :param vocab_size: Symbol count ``V`` of a ``categorical`` observation; required for
        that type and must be ``None`` otherwise.
    :param state_emb_dim: Width ``E`` of the learned state embeddings; positive.
    :param hidden_dim: Hidden width of the head; positive.
    :param context_dim: Width ``H`` of the per-step context the head also reads; 0 (the
        default) for no context.
    :param min_scale: Floor added to the Gaussian scale and the Poisson rate; positive.
        Unused by ``categorical``.
    :param kwargs: Base ``keras.layers.Layer`` arguments.
    :raises ValueError: If an argument is out of range, missing for the ``obs_type``, or
        set where the ``obs_type`` does not read it.
    """

    def __init__(
        self,
        n_states: int,
        obs_type: str,
        obs_dim: Optional[int] = None,
        vocab_size: Optional[int] = None,
        state_emb_dim: int = 16,
        hidden_dim: int = 64,
        context_dim: int = 0,
        min_scale: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if obs_type not in OBS_TYPES:
            raise ValueError(f"Unknown obs_type '{obs_type}'. Available: {list(OBS_TYPES)}.")
        if obs_type == "categorical":
            if vocab_size is None or vocab_size <= 0 or obs_dim is not None:
                raise ValueError(
                    f"obs_type='categorical' needs a positive vocab_size and obs_dim=None, "
                    f"got vocab_size={vocab_size}, obs_dim={obs_dim}."
                )
        elif obs_dim is None or obs_dim <= 0 or vocab_size is not None:
            raise ValueError(
                f"obs_type='{obs_type}' needs a positive obs_dim and vocab_size=None, "
                f"got obs_dim={obs_dim}, vocab_size={vocab_size}."
            )
        for name, value in (("n_states", n_states), ("state_emb_dim", state_emb_dim), ("hidden_dim", hidden_dim)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if context_dim < 0:
            raise ValueError(f"context_dim must be non-negative, got {context_dim}.")
        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {min_scale}.")

        self.n_states = n_states
        self.obs_type = obs_type
        self.obs_dim = obs_dim
        self.vocab_size = vocab_size
        self.state_emb_dim = state_emb_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.min_scale = min_scale

        n_params = {"gaussian": 2 * (obs_dim or 0), "categorical": vocab_size, "poisson": obs_dim}[obs_type]
        # DECISION plan-2026-10-13T192257-1bb3bf1f/D-020: the head type is the LITERAL
        # "mlp" with literal kwargs. Do NOT make it a constructor knob (ffn_type /
        # ffn_args): a variable type is a 16th dynamic site that the static forwarding
        # sweep in test_package_api_contract.py cannot check, past its pinned ceiling of
        # 15 (D-136 of plan-2026-08-19T163559-499b6f0e). The model's ffn_type knob stays
        # live through the transformer encoder only.
        self.head = create_ffn_layer("mlp", name="head", hidden_dim=hidden_dim, output_dim=n_params)
        self.state_embedding = None

    def build(
        self,
        obs_shape: Tuple[Optional[int], ...],
        context_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> None:
        """Create the state embedding table and build the head.

        :param obs_shape: ``(B, T)`` for ``categorical``, ``(B, T, D)`` otherwise.
        :param context_shape: ``(B, T, H)`` when ``context_dim > 0``, else ``None``.
        :raises ValueError: If a shape disagrees with the config.
        """
        expected_rank = 2 if self.obs_type == "categorical" else 3
        if len(obs_shape) != expected_rank or (expected_rank == 3 and obs_shape[-1] not in (None, self.obs_dim)):
            raise ValueError(
                f"obs_type='{self.obs_type}' expects observations of rank {expected_rank}"
                f"{'' if expected_rank == 2 else f' with {self.obs_dim} features'}, got {tuple(obs_shape)}."
            )
        if self.context_dim:
            if context_shape is None or len(context_shape) != 3 or context_shape[-1] not in (None, self.context_dim):
                raise ValueError(
                    f"context_dim={self.context_dim} expects a (batch, time, {self.context_dim}) "
                    f"context, got {context_shape}."
                )
        self.state_embedding = self.add_weight(
            name="state_embedding",
            shape=(self.n_states, self.state_emb_dim),
            # Unit scale, not Keras' 0.05: states must start distinguishable, or every
            # state emits the same density and the gradient cannot tell them apart.
            initializer=keras.initializers.RandomNormal(stddev=1.0),
        )
        width = self.state_emb_dim + self.context_dim
        self.head.build((None, None, self.n_states, width) if self.context_dim else (self.n_states, width))
        super().build(obs_shape)

    def _head_features(self, context: Optional[keras.KerasTensor]) -> keras.KerasTensor:
        """Return the head's input: ``(K, E)`` alone, or ``(B, T, K, E + H)`` with a context.

        :param context: ``(B, T, H)`` context, or ``None`` when ``context_dim == 0``.
        :return: The feature tensor in the compute dtype.
        :raises ValueError: If a context is required but missing.
        """
        embedding = keras.ops.convert_to_tensor(self.state_embedding)
        if not self.context_dim:
            return embedding
        if context is None:
            raise ValueError(f"context_dim={self.context_dim}: call(obs, context) needs a context.")
        context = keras.ops.cast(context, embedding.dtype)
        per_state = keras.ops.zeros((self.n_states, 1), dtype=embedding.dtype)
        embedding = keras.ops.zeros_like(context[:, :, None, :1]) + embedding  # (B, T, K, E)
        context = context[:, :, None, :] + per_state  # (B, T, K, H)
        return keras.ops.concatenate([embedding, context], axis=-1)

    def call(
        self,
        obs: keras.KerasTensor,
        context: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Return ``log p(x_t | z_t = k)`` for every step and state.

        :param obs: ``(B, T)`` integer symbols in ``[0, V)`` for ``categorical`` (any other
            value, ``-1`` padding included, scores ``mask_sentinel`` in every state);
            ``(B, T, D)`` reals for ``gaussian``; ``(B, T, D)`` non-negative counts for
            ``poisson``.
        :param context: ``(B, T, H)`` per-step context when ``context_dim > 0``.
        :param training: Forwarded to the head.
        :return: ``(B, T, K)`` log-densities in the accumulation dtype.
        """
        params = self.head(self._head_features(context), training=training)
        # DECISION plan-2026-10-13T192257-1bb3bf1f/D-010: the head runs in the compute
        # dtype, the density math in the accumulation dtype (float32 under mixed_float16).
        # Do NOT move this cast below the density: ((x - mu) / sd) ** 2 overflows float16
        # (max 65504) once |x - mu| / sd > 362, which sd = min_scale = 1e-3 reaches at
        # |x - mu| > 0.362, and the forward recursion then sums an inf.
        accum = accumulation_dtype(self.compute_dtype)
        params = keras.ops.cast(params, accum)
        if not self.context_dim:
            params = params[None, None]  # (1, 1, K, P): broadcasts over batch and time

        if self.obs_type == "categorical":
            log_probs = keras.ops.log_softmax(params, axis=-1)  # (.., K, V)
            symbols = keras.ops.cast(obs, "int32")
            chosen = keras.ops.one_hot(symbols, self.vocab_size)[:, :, None, :] > 0
            # Select, never multiply by the one-hot: 0 * -inf is NaN.
            picked = keras.ops.sum(keras.ops.where(chosen, log_probs, keras.ops.zeros_like(log_probs)), axis=-1)
            # DECISION plan-2026-10-13T192257-1bb3bf1f/D-022: a symbol outside [0, V) has
            # an all-zero one-hot, so the sum above scores it 0 = log 1 (certain). Score
            # it mask_sentinel instead, so a bad symbol at a valid step makes log p(x) and
            # the loss huge. Do NOT assert in-graph (XLA drops tf.debugging asserts) and do
            # NOT use -inf (a step with every state at -inf makes the forward recursion and
            # its gradients NaN). The model's -1 padding also lands here; inference never
            # reads a padded step, so the sentinel there is inert.
            in_range = keras.ops.logical_and(symbols >= 0, symbols < self.vocab_size)[:, :, None]
            return keras.ops.where(in_range, picked, keras.ops.full_like(picked, mask_sentinel(accum)))

        x = keras.ops.cast(obs, accum)[:, :, None, :]  # (B, T, 1, D)
        if self.obs_type == "gaussian":
            mean, raw_scale = keras.ops.split(params, 2, axis=-1)
            scale = keras.ops.softplus(raw_scale) + self.min_scale
            z = (x - mean) / scale
            log_density = -0.5 * keras.ops.square(z) - keras.ops.log(scale) - _HALF_LOG_TWO_PI
        else:
            rate = keras.ops.softplus(params) + self.min_scale
            log_density = x * keras.ops.log(rate) - rate - log_gamma(x + 1.0)
        return keras.ops.sum(log_density, axis=-1)

    def compute_output_shape(
        self,
        obs_shape: Tuple[Optional[int], ...],
        context_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> Tuple[Optional[int], ...]:
        """Return ``(B, T, K)`` from the stored config.

        :param obs_shape: Observation shape ``(B, T, ...)``.
        :param context_shape: Context shape; does not change the output shape.
        :return: ``(B, T, K)``.
        """
        return obs_shape[0], obs_shape[1], self.n_states

    def get_config(self) -> dict:
        """Return every constructor argument.

        :return: The layer config.
        """
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "obs_type": self.obs_type,
                "obs_dim": self.obs_dim,
                "vocab_size": self.vocab_size,
                "state_emb_dim": self.state_emb_dim,
                "hidden_dim": self.hidden_dim,
                "context_dim": self.context_dim,
                "min_scale": self.min_scale,
            }
        )
        return config


@register_dl_technique("dl_techniques.models.neural_hmm.emissions")
class DiscriminativeEmission(LogSpaceOutputSpec, keras.layers.Layer):
    """Per-state potentials from a context, for a discriminative (CRF) Neural HMM.

    ``log_emit = Dense(K)(h)``: unnormalized scores, not densities; the CRF partition
    function normalizes over whole paths. The output is in the accumulation dtype.

    :param n_states: Number of hidden states ``K``; positive.
    :param kwargs: Base ``keras.layers.Layer`` arguments.
    :raises ValueError: If ``n_states`` is not positive.
    """

    def __init__(self, n_states: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if n_states <= 0:
            raise ValueError(f"n_states must be positive, got {n_states}.")
        self.n_states = n_states
        self.state_scores = keras.layers.Dense(n_states, name="state_scores")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the scoring projection.

        :param input_shape: Context shape ``(B, T, H)``.
        :raises ValueError: If the context is not rank 3.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected a context of shape (batch, time, features), got {tuple(input_shape)}.")
        self.state_scores.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Return the ``(B, T, K)`` state potentials.

        :param inputs: Context ``(B, T, H)``.
        :param training: Forwarded to the projection.
        :return: ``(B, T, K)`` potentials in the accumulation dtype.
        """
        scores = self.state_scores(inputs, training=training)
        return keras.ops.cast(scores, accumulation_dtype(self.compute_dtype))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return ``(B, T, K)`` from the stored config.

        :param input_shape: Context shape ``(B, T, H)``.
        :return: ``(B, T, K)``.
        """
        return input_shape[0], input_shape[1], self.n_states

    def get_config(self) -> dict:
        """Return every constructor argument.

        :return: The layer config.
        """
        config = super().get_config()
        config["n_states"] = self.n_states
        return config

"""Synthetic time-series generators for the Neural HMM trainer.

Unlike the existing ``src/dl_techniques/datasets/time_series/generator.py``'s
``_generate_regime_switching`` (which uses a deterministic cyclic advance
``(regime + 1) % regimes`` and discards the latent path), these generators:

1. Sample the state path from a *proper* transition matrix (row-stochastic,
   configurable per regime).
2. Return ``(x, states)`` so the trainer has ground-truth states for
   state-occupancy / segmentation metrics.
3. Support Gaussian observation models (the HMM's default ``obs_type='gaussian'``).

The design follows ``transform_sequences.py``'s contract of returning the ground-truth
latent alongside the sequence, so metrics can compute state-occupancy entropy,
Viterbi accuracy, etc.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Transition matrix samplers
# ---------------------------------------------------------------------


def _row_stochastic(rng: np.random.RandomState, dim: int) -> np.ndarray:
    """Return a row-stochastic matrix of shape ``(dim, dim)``.

    Each row sums to 1.0. The last row is set to a one-hot vector at the
    ``last`` index so the chain is irreducible (the last state cannot be
    left; this mirrors the recipe's ``self_loop_init`` / sticky-state
    convention).

    :param rng: Random state.
    :type rng: np.random.RandomState
    :param dim: Dimension of the square matrix.
    :type dim: int
    :return: Row-stochastic ``(dim, dim)`` float32 array.
    :rtype: np.ndarray
    """
    # Draw a Dirichlet row for each row except the last
    M = rng.dirichlet(np.ones(dim), size=dim - 1).astype("float32")
    # The last row is a one-hot at the last state (stickiness)
    last_row = np.zeros(dim, dtype="float32")
    last_row[-1] = 1.0
    M = np.vstack([M, last_row])
    return M


# ---------------------------------------------------------------------
# Observation models (matching the HMM's emission types)
# ---------------------------------------------------------------------


def _gaussian_observation(
    rng: np.random.RandomState,
    states: np.ndarray,          # (T,) int state indices
    means: np.ndarray,           # (K,) per-state mean
    stds: np.ndarray,            # (K,) per-state std dev
) -> np.ndarray:
    """Emit a Gaussian observation per timestep.

    :param rng: Random state for noise draw.
    :type rng: np.random.RandomState
    :param states: Timestep state indices ``(T,)``.
    :type states: np.ndarray
    :param means: Per-state mean ``(K,)``.
    :type means: np.ndarray
    :param stds: Per-state standard dev ``(K,)``.
    :type stds: np.ndarray
    :return: Observations ``(T,)`` float32.
    :rtype: np.ndarray
    """
    T = states.shape[0]
    noise = rng.normal(0.0, 1.0, size=T).astype("float32")
    # Map each timestep to its state's mean+std
    obs = means[states] + stds[states] * noise
    return obs


def _categorical_observation(
    rng: np.random.RandomState,
    states: np.ndarray,          # (T,) int state indices
    vocab_size: int,
    probs_per_state: Optional[np.ndarray] = None,  # (K, vocab_size) or None
) -> np.ndarray:
    """Emit a categorical observation per timestep.

    :param rng: Random state.
    :type rng: np.random.RandomState
    :param states: Timestep state indices ``(T,)``.
    :type states: np.ndarray
    :param vocab_size: Size of the observation vocabulary.
    :type vocab_size: int
    :param probs_per_state: Per-state categorical probabilities ``(K, V)``;
        if ``None`` uniform probs are used.
    :type probs_per_state: Optional[np.ndarray]
    :return: Integer observations ``(T,)`` in ``[0, vocab_size)``.
    :rtype: np.ndarray
    """
    T = states.shape[0]
    if probs_per_state is None:
        # Uniform over vocabulary for every state
        probs = np.full((T, vocab_size), 1.0 / vocab_size, dtype="float32")
    else:
        # Use the per-state probabilities; index by the state at each timestep
        probs = probs_per_state[states]  # (T, V)
        # Normalise each row
        probs = probs / probs.sum(axis=1, keepdims=True)
    # Sample categorical
    obs = rng.multinomial(1, probs, size=1).reshape(-1).astype("int32")
    return obs


# ---------------------------------------------------------------------
# Generator functions returned by the task table
# ---------------------------------------------------------------------


def _generate_gaussian_hmm(
    rng: np.random.RandomState,
    n_samples: int,
    n_states: int,
    transition_params: Dict[str, Any],
    emission_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a Gaussian-HMM synthetic sequence with ground-truth states.

    The state path is sampled from a row-stochastic transition matrix; the
    observations are Gaussian with per-state mean/std.

    :param rng: Random state.
    :type rng: np.random.RandomState
    :param n_samples: Number of timesteps ``T``.
    :type n_samples: int
    :param n_states: Number of hidden states ``K``.
    :type n_states: int
    :param transition_params: Dict with keys ``type`` transition matrix and
        optional ``switch_prob``. If ``type`` is ``'static'`` the matrix
        is constant; if ``'input_dependent'`` a simple two-regime switch is
        applied (the first half uses one matrix, the second half another).
    :type transition_params: Dict[str, Any]
    :param emission_params: Dict with keys ``means`` and ``stds``,
        each ``(K,)``.
    :type emission_params: Dict[str, Any]
    :return: Dict with keys ``x`` (observations ``(T,)``), ``states``
        (latent state path ``(T,)``), and ``transition_matrix``
        (``(K, K)`` if static, or ``(2, K, K)`` if input-dependent).
    :rtype: Dict[str, Any]
    """
    T = n_samples
    K = n_states

    # Build transition matrix
    trans_type = transition_params.get("type", "static")
    if trans_type == "static":
        transition_matrix = _row_stochastic(rng, K)
        # Apply self-loop bias: overwrite the diagonal to be sticky.
        # The recipe's ``self_loop_init`` of 2-3 means diagonal logits of 2-3,
        # which after log_softmax give ~0.9 probability to stay. We approximate
        # this by setting diagonal entries to 0.9 and renormalising.
        for k in range(K):
            transition_matrix[k, k] = 0.9
            transition_matrix[k, :] = transition_matrix[k, :] / transition_matrix[k, :].sum()
    elif trans_type == "input_dependent":
        # Two-regime switch at T//2
        transition_matrix_1 = _row_stochastic(rng, K)
        transition_matrix_2 = _row_stochastic(rng, K)
        for k in range(K):
            transition_matrix_1[k, k] = 0.9
            transition_matrix_1[k, :] = transition_matrix_1[k, :] / transition_matrix_1[k, :].sum()
            transition_matrix_2[k, k] = 0.9
            transition_matrix_2[k, :] = transition_matrix_2[k, :] / transition_matrix_2[k, :].sum()
        transition_matrix = np.tile(transition_matrix_1, (T // 2 + 1, 1))[:T]
        # Second regime overrides the back half
        if T > T // 2:
            transition_matrix[T // 2:] = np.tile(transition_matrix_2, (T - T // 2 + 1, 1))[: T - T // 2]
    else:
        raise ValueError(f"Unknown transition type '{trans_type}'.")

    # Sample the state path
    states = np.zeros(T, dtype="int32")
    # Initial state: sample from the first row of the transition matrix
    current = rng.randint(0, K)
    states[0] = current
    for t in range(1, T):
        # Transition probabilities from current state
        probs = transition_matrix[current]
        current = rng.choice(K, p=probs)
        states[t] = current

    # Emission parameters
    means = np.array(emission_params["means"], dtype="float32")  # (K,)
    stds = np.array(emission_params["stds"], dtype="float32")    # (K,)

    # Generate observations
    x = _gaussian_observation(rng, states, means, stds)

    # Pack result
    result: Dict[str, Any] = {
        "x": x,
        "states": states,
    }

    if trans_type == "static":
        result["transition_matrix"] = transition_matrix
    else:
        result["transition_matrix_1"] = transition_matrix_1
        result["transition_matrix_2"] = transition_matrix_2

    return result


def _generate_categorical_hmm(
    rng: np.random.RandomState,
    n_samples: int,
    n_states: int,
    transition_params: Dict[str, Any],
    emission_params: Dict[str, Any],
    vocab_size: int,
) -> Dict[str, Any]:
    """Generate a categorical-HMM synthetic sequence with ground-truth states.

    Same structure as ``_generate_gaussian_hmm`` but with discrete
    observations.

    :param rng: Random state.
    :type rng: np.random.RandomState
    :param n_samples: Number of timesteps ``T``.
    :type n_samples: int
    :param n_states: Number of hidden states ``K``.
    :type n_states: int
    :param transition_params: Same format as ``_generate_gaussian_hmm``.
    :type transition_params: Dict[str, Any]
    :param emission_params: Per-state categorical probabilities ``(K, V)``.
    :type emission_params: Dict[str, Any]
    :param vocab_size: Size of the observation vocabulary ``V``.
    :type vocab_size: int
    :return: Dict with keys ``x`` (integer observations ``(T,)``),
        ``states`` (latent path ``(T,)``), and transition matrices.
    :rtype: Dict[str, Any]
    """
    T = n_samples
    K = n_states

    # Build transition matrix (same logic as Gaussian version)
    trans_type = transition_params.get("type", "static")
    if trans_type == "static":
        transition_matrix = _row_stochastic(rng, K)
        for k in range(K):
            transition_matrix[k, k] = 0.9
            transition_matrix[k, :] = transition_matrix[k, :] / transition_matrix[k, :].sum()
    elif trans_type == "input_dependent":
        transition_matrix_1 = _row_stochastic(rng, K)
        transition_matrix_2 = _row_stochastic(rng, K)
        for k in range(K):
            transition_matrix_1[k, k] = 0.9
            transition_matrix_1[k, :] = transition_matrix_1[k, :] / transition_matrix_1[k, :].sum()
            transition_matrix_2[k, k] = 0.9
            transition_matrix_2[k, :] = transition_matrix_2[k, :] / transition_matrix_2[k, :].sum()
        transition_matrix = np.tile(transition_matrix_1, (T // 2 + 1, 1))[:T]
        if T > T // 2:
            transition_matrix[T // 2:] = np.tile(transition_matrix_2, (T - T // 2 + 1, 1))[: T - T // 2]
    else:
        raise ValueError(f"Unknown transition type '{trans_type}'.")

    # Sample the state path
    states = np.zeros(T, dtype="int32")
    current = rng.randint(0, K)
    states[0] = current
    for t in range(1, T):
        probs = transition_matrix[current]
        current = rng.choice(K, p=probs)
        states[t] = current

    # Per-state categorical probabilities (K, V)
    K_val = emission_params.get("probs_per_state", None)
    if K_val is None:
        # Uniform per state; we'll just note it
        probs_per_state = np.full((K, vocab_size), 1.0 / vocab_size, dtype="float32")
    else:
        probs_per_state = np.array(K_val, dtype="float32")

    # Generate observations
    x = _categorical_observation(rng, states, vocab_size, probs_per_state)

    result: Dict[str, Any] = {
        "x": x,
        "states": states,
    }

    if trans_type == "static":
        result["transition_matrix"] = transition_matrix
    else:
        result["transition_matrix_1"] = transition_matrix_1
        result["transition_matrix_2"] = transition_matrix_2

    return result


# ---------------------------------------------------------------------
# Trainer-visible task table (mirrors the TimeSeriesGenerator pattern)
# ---------------------------------------------------------------------

# These are the generator functions the trainer can dispatch by name.
# The trainer's data loader reads these by string key.

_GENERATORS: Dict[str, Any] = {
    "gaussian_hmm": _generate_gaussian_hmm,
    "categorical_hmm": _generate_categorical_hmm,
}


def available_generators() -> List[str]:
    """Return the names of available synthetic generators.

    :return: List of generator names.
    :rtype: List[str]
    """
    return sorted(_GENERATORS.keys())


def generate_dataset(
    generator_name: str,
    cfg: Dict[str, Any],
    **generator_kwargs: Any,
) -> Dict[str, Any]:
    """Dispatch to a registered synthetic generator.

    :param generator_name: Name from ``available_generators()``.
    :type generator_name: str
    :param cfg: Configuration dict passed to the generator (contains
        ``n_samples``, ``n_states``, emission/transition params).
    :type cfg: Dict[str, Any]
    :param generator_kwargs: Additional keyword args forwarded to the
        generator function.
    :type generator_kwargs: Any
    :return: Generator output dict (observations + states + optionally matrices).
    :rtype: Dict[str, Any]
    :raises ValueError: If ``generator_name`` is unrecognised.
    """
    gen = _GENERATORS.get(generator_name)
    if gen is None:
        raise ValueError(
            f"Unknown generator '{generator_name}'. "
            f"Available: {available_generators()}"
        )
    return gen(cfg["rng"], **cfg, **generator_kwargs)


# ---------------------------------------------------------------------
# Convenience: seed the RNG for a reproducible run
# ---------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed numpy's random state globally (called once per run).

    This is the same convention used by the other time-series trainers
    (nbeats, deepar, etc.) so that a run's data is reproducible.

    :param seed: Integer seed.
    :type seed: int
    """
    np.random.seed(seed)


# ---------------------------------------------------------------------
# CLI / one-shot sanity check
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity: generate a gaussian HMM sequence and print shapes.
    import sys
    from dl_techniques.train.neural_hmm.data import (
        generate_dataset, available_generators, set_global_seed,
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        print("Available generators:", available_generators())
    else:
        set_global_seed(42)
        cfg = dict(
            n_samples=100,
            n_states=5,
            means=[0.0, 1.0, -1.0, 2.0, -2.0],
            stds=[1.0, 1.0, 1.0, 1.0, 1.0],
            transition={"type": "static"},
        )
        data = generate_dataset("gaussian_hmm", cfg)
        print(f"x shape: {data['x'].shape}  states shape: {data['states'].shape}")
        print(f"x dtype: {data['x'].dtype}  states dtype: {data['states'].dtype}")
        # Verify states are within range
        print(f"states min/max: {data['states'].min()} / {data['states'].max()}")
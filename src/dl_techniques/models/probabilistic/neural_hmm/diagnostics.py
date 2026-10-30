"""Host-side checks and compiled, batched inference over a :class:`~.model.NeuralHMM`.

The model's :meth:`~.model.NeuralHMM.viterbi`, ``posteriors`` and ``log_likelihood`` run
eagerly on whatever they are given, one batch, with no batching. The helpers here run the
same computation through ``predict``, so a whole split goes through in compiled batches:

- :func:`decode` and :func:`score`: the Viterbi paths and the log-likelihoods of a split;
- :func:`state_diagnostics`: whether the hidden states are carrying information, which
  the likelihood alone cannot say (an autoregressive emission can reach a good likelihood
  by reading the history and ignoring the state);
- :func:`validate_inputs`: the input errors the in-graph sentinel makes loud only as a
  huge loss, or not at all, reported by name before any training.

Each call traces its function once, so pass a whole split rather than one batch at a time.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import keras
import numpy as np

from . import inference
from .model import NeuralHMM


class _CompiledInference(keras.Model):
    """Run a function of a model through ``predict``; never saved, and has no own weights.

    :param model: The model to read.
    :param compute: ``compute(model, inputs)``, returning arrays with a batch axis.
    :param kwargs: Base ``keras.Model`` arguments.
    """

    def __init__(self, model: NeuralHMM, compute: Callable[[NeuralHMM, Dict[str, Any]], Any], **kwargs: Any) -> None:
        super().__init__(dtype=model.dtype_policy, **kwargs)
        self.wrapped = model
        self.compute = compute

    def call(self, inputs: Dict[str, Any], training: bool = False) -> Any:
        """Return the wrapped function's result.

        :param inputs: The batch.
        :param training: Unused: inference is always in inference mode.
        :return: Whatever ``compute`` returns.
        """
        return self.compute(self.wrapped, inputs)


def _run(model: NeuralHMM, compute: Callable, inputs: Mapping[str, Any], batch_size: int) -> Any:
    """Run ``compute`` over ``inputs`` in compiled batches and concatenate the results.

    :param model: The model.
    :param compute: See :class:`_CompiledInference`.
    :param inputs: Arrays with a sequence axis first.
    :param batch_size: Sequences per batch.
    :return: ``compute``'s structure, of NumPy arrays over every sequence.
    """
    runner = _CompiledInference(model, compute, name="compiled_inference")
    # An lstm or gru encoder does not compile under XLA, whatever the model's own setting.
    runner.jit_compile = False
    return runner.predict(dict(inputs), batch_size=batch_size, verbose=0)


def decode(model: NeuralHMM, inputs: Mapping[str, Any], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Viterbi paths and scores of a whole split.

    :param model: The model.
    :param inputs: ``{"obs", "mask"}``, plus ``"inputs"`` with ``input_dim`` set.
    :param batch_size: Sequences per batch.
    :return: ``(path (N, T) int32 with -1 at padding, score (N,))``.
    """
    return tuple(_run(model, lambda m, batch: m.viterbi(batch), inputs, batch_size))


def score(
    model: NeuralHMM, inputs: Mapping[str, Any], states: Optional[np.ndarray] = None, batch_size: int = 32
) -> np.ndarray:
    """Return the per-sequence log-likelihood of a whole split.

    :param model: The model.
    :param inputs: ``{"obs", "mask"}``, plus ``"inputs"`` with ``input_dim`` set.
    :param states: ``None`` (generative) or ``(N, T)`` integer states (discriminative).
    :param batch_size: Sequences per batch.
    :return: ``(N,)``, as :meth:`~.model.NeuralHMM.log_likelihood`.
    :raises ValueError: If ``states`` does not match the mode.
    """
    if (states is not None) != (model.mode == "discriminative"):
        raise ValueError(f"mode={model.mode!r} takes states only in discriminative mode, where it needs them.")
    if states is None:
        return _run(model, lambda m, batch: m.log_likelihood(batch), inputs, batch_size)

    def compute(m: NeuralHMM, batch: Dict[str, Any]) -> keras.KerasTensor:
        return m.log_likelihood({k: v for k, v in batch.items() if k != "states"}, batch["states"])

    return _run(model, compute, {**inputs, "states": np.asarray(states)}, batch_size)


def state_diagnostics(
    model: NeuralHMM, inputs: Mapping[str, Any], batch_size: int = 32, seed: int = 0
) -> Dict[str, Any]:
    """Measure how much the hidden states are used, and whether they carry information.

    All-healthy numbers: an occupancy near the true state frequencies, a mean posterior
    entropy well below ``log K``, and a positive shuffled gap. A gap near 0 means shuffling
    the emissions over the states costs nothing, so the state sequence explains nothing
    that the per-step mixture does not: the emissions do not depend on the state, or the
    transitions ignore it.

    :param model: The model.
    :param inputs: ``{"obs", "mask"}``, plus ``"inputs"`` with ``input_dim`` set.
    :param batch_size: Sequences per batch.
    :param seed: Seed of the per-step permutations.
    :return: ``occupancy`` ``(K,)``, the mean posterior of each state over valid steps;
        ``occupancy_entropy`` and ``mean_posterior_entropy``, in nats, each at most
        ``log K``; ``state_shuffled_gap``, nats per valid step of ``log p(x)`` lost when
        ``log_emit`` is permuted over the states at every step independently, or ``None``
        in discriminative mode, whose ``log Z`` is not a likelihood.
    :raises ValueError: If there is no valid step.
    """
    generative = model.mode != "discriminative"
    arrays = dict(inputs)
    if generative:
        shape = (*np.shape(inputs["obs"])[:2], model.n_states)
        arrays["permutation"] = np.argsort(np.random.default_rng(seed).random(shape), axis=-1).astype("int32")

    def compute(m: NeuralHMM, batch: Dict[str, Any]) -> Dict[str, keras.KerasTensor]:
        plain = {k: v for k, v in batch.items() if k != "permutation"}
        out = m(plain, training=False)
        log_pi, log_A, log_emit, log_end = out["log_pi"], out["log_A"], out["log_emit"], out.get("log_end")
        mask = m._valid_mask(plain, log_emit)
        gamma = inference.state_posteriors(log_pi, log_A, log_emit, mask, log_end)
        safe = keras.ops.where(gamma > 0, gamma, keras.ops.ones_like(gamma))
        result = {
            "occupancy": keras.ops.sum(gamma, axis=1),
            "entropy": -keras.ops.sum(gamma * keras.ops.log(safe), axis=(1, 2)),
            "steps": keras.ops.sum(mask, axis=1),
        }
        if generative:
            shuffled = keras.ops.take_along_axis(log_emit, batch["permutation"], axis=-1)
            result["log_z"] = inference.forward_log_likelihood(log_pi, log_A, log_emit, mask, log_end)
            result["shuffled_log_z"] = inference.forward_log_likelihood(log_pi, log_A, shuffled, mask, log_end)
        return result

    raw = _run(model, compute, arrays, batch_size)
    steps = float(raw["steps"].astype(np.float64).sum())
    if steps == 0:
        raise ValueError("state_diagnostics needs at least one valid step.")
    occupancy = raw["occupancy"].astype(np.float64).sum(axis=0) / steps
    used = occupancy[occupancy > 0]
    gap = None
    if generative:
        gap = float((raw["log_z"].astype(np.float64) - raw["shuffled_log_z"].astype(np.float64)).sum() / steps)
    return {
        "occupancy": occupancy,
        "occupancy_entropy": float(-(used * np.log(used)).sum()),
        "mean_posterior_entropy": float(raw["entropy"].astype(np.float64).sum() / steps),
        "state_shuffled_gap": gap,
    }


def validate_inputs(model: NeuralHMM, inputs: Mapping[str, Any], states: Optional[np.ndarray] = None) -> None:
    """Refuse a split the model would read silently wrong; checks valid steps only.

    Padded steps may hold anything, as they may in the model. The checks are a non-prefix
    mask, a mask of another shape, a ``categorical`` symbol outside ``[0, vocab_size)``, a
    non-finite observation or exogenous input, a negative ``poisson`` count and a label
    outside ``[0, n_states)``.

    :param model: The model.
    :param inputs: ``{"obs", "mask"}``, plus ``"inputs"`` with ``input_dim`` set.
    :param states: Optional ``(N, T)`` integer labels.
    :raises ValueError: Naming every problem found, with a count and the first bad sequence.
    """
    obs = np.asarray(inputs["obs"])
    steps = obs.shape[:2]
    problems: List[str] = []
    valid = np.ones(steps, dtype=bool)
    if inputs.get("mask") is not None:
        mask = np.asarray(inputs["mask"])
        if mask.shape != steps:
            raise ValueError(f"mask has shape {mask.shape}, but obs has {steps} sequences and steps.")
        valid = mask != 0
        holes = valid[:, 1:] & ~valid[:, :-1]
        if holes.any():
            problems.append(f"mask is not a valid prefix in {_where(holes.any(axis=1))} (right padding only)")

    def check(bad: np.ndarray, what: str) -> None:
        bad = bad & (valid[:, :, None] if bad.ndim == 3 else valid)
        if bad.any():
            per_sequence = bad.reshape(steps[0], -1).any(axis=1)
            problems.append(f"{what} at {int(bad.sum())} valid entries, in {_where(per_sequence)}")

    if model.obs_type == "categorical":
        check((obs < 0) | (obs >= model.vocab_size), f"symbols outside [0, {model.vocab_size})")
    else:
        check(~np.isfinite(obs), "non-finite observations")
        if model.obs_type == "poisson":
            # NaN compares False here, so the check above alone reports it.
            with np.errstate(invalid="ignore"):
                check(obs < 0, "negative counts")
    if inputs.get("inputs") is not None:
        check(~np.isfinite(np.asarray(inputs["inputs"])), "non-finite exogenous inputs")
    if states is not None:
        labels = np.asarray(states)
        check((labels < 0) | (labels >= model.n_states), f"labels outside [0, {model.n_states})")
    if problems:
        raise ValueError("Invalid inputs: " + "; ".join(problems) + ".")


def _where(per_sequence: np.ndarray) -> str:
    """Name the bad sequences of a ``(N,)`` boolean array.

    :param per_sequence: Which sequences are bad.
    :return: Their count and the index of the first.
    """
    return f"{int(per_sequence.sum())} sequences (first: {int(np.argmax(per_sequence))})"

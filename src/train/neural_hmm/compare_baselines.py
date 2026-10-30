"""Measure the neural HMM against what it is meant to replace or improve on.

Three experiments, each row appended to ``--output`` as one JSON line as it finishes:

- ``baum-welch``: ``neural-hmm``, without and with k-means emissions, against Baum-Welch, on standardized and on shifted and
  scaled copies of the data, with ``K`` in ``--states``. NLLs are in the units of the
  transformed data, and every gap is to the true model under the same transform.
- ``crf``: ``neural-crf-tagger`` against a softmax tagger, the same model with its
  transitions frozen uniform, and against the CRF with a BIO mask, on data with hard label
  constraints.
- ``autoregressive``: ``autoregressive-neural-hmm`` against ``neural-hmm`` and Baum-Welch,
  on data whose history predicts ``x_t`` strongly, with :func:`state_diagnostics`.

Usage::

    python -m train.neural_hmm.compare_baselines --output comparison.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.models.probabilistic.neural_hmm import NeuralHMM, create_neural_hmm
from dl_techniques.models.probabilistic.neural_hmm import diagnostics, inference, initialization
from dl_techniques.optimization import optimizer_builder
from train.common import set_seeds
from train.neural_hmm import baselines
from train.neural_hmm.data import (
    HMMDataset, accuracy, matched_accuracy, sample_splits, true_nll_per_step,
)

EXPERIMENTS = ("baum-welch", "crf", "autoregressive")
OBS_DIM = 2
SHIFT, SCALE = 100.0, 5.0
#: O, B, I: I may only follow B or I, and a sequence may not start in I.
BIO_TRANSITIONS = [[True, True, False], [True, True, True], [True, True, True]]
BIO_STARTS = [True, True, False]
BIO_MEANS = np.array([[-1.0, 0.0], [1.0, 0.5], [1.0, -0.5]])
BIO_MOVES = np.array([[0.85, 0.15, 0.0], [0.0, 0.0, 1.0], [0.25, 0.05, 0.70]])
AR_RHO, AR_NOISE_SD = 0.9, 0.5


def _train(
        model: NeuralHMM, train: Dict[str, np.ndarray], val: Dict[str, np.ndarray], labels: Tuple[Any, Any],
        args: argparse.Namespace,
) -> Dict[str, Any]:
    """Fit with the trainer's defaults and keep the best validation weights.

    :param model: The model.
    :param train: Training inputs.
    :param val: Validation inputs.
    :param labels: Training and validation labels, or ``None`` each.
    :param args: Parsed arguments.
    :return: ``epochs`` run and ``fit_seconds``.
    """
    model.compile(optimizer=optimizer_builder(
        {"type": "adam", "gradient_clipping_by_norm": 1.0}, args.learning_rate))
    start = time.perf_counter()
    history = model.fit(
        train, labels[0], validation_data=(val, labels[1]), epochs=args.epochs, batch_size=args.batch_size, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True)],
    ).history
    return {"epochs": len(history["loss"]), "fit_seconds": time.perf_counter() - start}


def _as_inputs(split: HMMDataset) -> Dict[str, np.ndarray]:
    return {"obs": split.obs, "mask": split.mask}


def _nll(log_lik: np.ndarray, mask: np.ndarray) -> float:
    return float(-np.asarray(log_lik, dtype=np.float64).sum() / np.sum(mask))


def _score_neural(
        model: NeuralHMM, test: HMMDataset, true_nll: float, args: argparse.Namespace,
) -> Dict[str, Any]:
    inputs = _as_inputs(test)
    nll = _nll(diagnostics.score(model, inputs, batch_size=args.batch_size), test.mask)
    path = diagnostics.decode(model, inputs, args.batch_size)[0]
    report = diagnostics.state_diagnostics(model, inputs, args.batch_size)
    return {
        "nll_gap": nll - true_nll,
        "matched_accuracy": matched_accuracy(path, test.states, model.n_states),
        "occupancy_entropy": report["occupancy_entropy"],
        "mean_posterior_entropy": report["mean_posterior_entropy"],
        "state_shuffled_gap": report["state_shuffled_gap"],
    }


def _score_baum_welch(hmm: baselines.BaumWelchHMM, test: HMMDataset, true_nll: float) -> Dict[str, Any]:
    nll = _nll(baselines.log_likelihood(hmm, test.obs, test.mask), test.mask)
    p = baselines.potentials(hmm, test.obs, test.mask)
    path = keras.ops.convert_to_numpy(inference.viterbi_decode(p["log_pi"], p["log_A"], p["log_emit"], test.mask)[0])
    return {
        "nll_gap": nll - true_nll,
        "matched_accuracy": matched_accuracy(path, test.states, len(hmm.pi)),
        "iterations": len(hmm.history),
    }


def _transformed(split: HMMDataset, shift: np.ndarray, scale: np.ndarray) -> HMMDataset:
    return split._replace(obs=(split.obs * scale + shift).astype(np.float32))


def compare_with_baum_welch(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    """Yield one row per seed, state count, data transform and method.

    :param args: Parsed arguments.
    :return: Rows.
    """
    sizes = {"train": args.num_train, "val": args.num_val, "test": args.num_test}
    for seed in range(args.seeds):
        for n_states in args.states:
            splits = sample_splits(
                sizes, seed, seq_len=args.seq_len, min_seq_len=args.seq_len // 2, n_states=n_states,
                obs_type="gaussian", obs_dim=OBS_DIM, separation=4.0, stay_prob=0.9)
            hmm = splits["test"].true_hmm
            valid = splits["train"].obs[splits["train"].mask > 0]
            base_nll = true_nll_per_step(hmm, splits["test"].obs, splits["test"].mask)
            for data, (shift, scale) in {
                "standardized": (-valid.mean(axis=0) / valid.std(axis=0), 1.0 / valid.std(axis=0)),
                "shifted_scaled": (np.full(OBS_DIM, SHIFT), np.full(OBS_DIM, SCALE)),
            }.items():
                train, val, test = (_transformed(splits[name], shift, scale) for name in ("train", "val", "test"))
                # x' = scale x + shift, so log p(x') = log p(x) - sum log scale.
                true_nll = base_nll + float(np.log(scale).sum())
                row = {"experiment": "baum-welch", "seed": seed, "n_states": n_states, "data": data}
                set_seeds(seed)
                model = NeuralHMM(n_states=n_states, obs_type="gaussian", obs_dim=OBS_DIM, hidden_dim=args.hidden_dim)
                fit = _train(model, _as_inputs(train), _as_inputs(val), (None, None), args)
                yield {**row, "method": "neural-hmm", **fit, **_score_neural(model, test, true_nll, args)}
                set_seeds(seed)
                model = NeuralHMM(n_states=n_states, obs_type="gaussian", obs_dim=OBS_DIM, hidden_dim=args.hidden_dim)
                initialization.initialize_emissions_from_kmeans(model, _as_inputs(train), seed=seed)
                fit = _train(model, _as_inputs(train), _as_inputs(val), (None, None), args)
                yield {**row, "method": "neural-hmm-kmeans", **fit, **_score_neural(model, test, true_nll, args)}
                for init in baselines.INITS:
                    start = time.perf_counter()
                    fitted = baselines.fit_baum_welch(train.obs, train.mask, n_states, init=init, seed=seed)
                    fit = {"fit_seconds": time.perf_counter() - start}
                    yield {**row, "method": f"baum-welch-{init}", **fit, **_score_baum_welch(fitted, test, true_nll)}


def sample_bio(n_sequences: int, steps: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Draw noisy observations of label paths with hard BIO constraints.

    :param n_sequences: ``N``.
    :param steps: ``T``.
    :param seed: Seed.
    :return: ``(obs (N, T, 2) float32, labels (N, T) int32)``.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros((n_sequences, steps), dtype=np.int32)
    labels[:, 0] = rng.choice(2, n_sequences, p=[0.85, 0.15])
    for t in range(1, steps):
        draws = rng.random(n_sequences)[:, None]
        labels[:, t] = (draws > np.cumsum(BIO_MOVES[labels[:, t - 1]], axis=1)).sum(axis=1)
    return (BIO_MEANS[labels] + rng.normal(size=(n_sequences, steps, 2))).astype(np.float32), labels


def invalid_bio_rate(paths: np.ndarray) -> float:
    """Return the fraction of sequences breaking a BIO constraint.

    :param paths: ``(N, T)`` labels.
    :return: A rate in ``[0, 1]``.
    """
    return float((((paths[:, :-1] == 0) & (paths[:, 1:] == 2)).any(axis=1) | (paths[:, 0] == 2)).mean())


def compare_crf(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    """Yield one row per seed and tagger; the softmax tagger is the CRF with frozen transitions.

    Frozen at zero logits, ``log_A`` and ``log_pi`` are constants, so the CRF loss is the
    per-step cross entropy and Viterbi is the per-step argmax over the same encoder.

    :param args: Parsed arguments.
    :return: Rows.
    """
    for seed in range(args.crf_seeds):
        x_train, y_train = sample_bio(args.num_train, args.seq_len, seed * 3 + 1)
        x_val, y_val = sample_bio(args.num_val, args.seq_len, seed * 3 + 2)
        x_test, y_test = sample_bio(args.num_test, args.seq_len, seed * 3 + 3)
        masks = [np.ones(x.shape[:2], "float32") for x in (x_train, x_val, x_test)]
        for method in ("softmax", "crf", "crf-bio-mask"):
            set_seeds(seed)
            constraints = (
                {"allowed_transitions": BIO_TRANSITIONS, "allowed_starts": BIO_STARTS} if method == "crf-bio-mask"
                else {})
            model = create_neural_hmm(
                "neural-crf-tagger", n_states=3, obs_type="gaussian", obs_dim=OBS_DIM, hidden_dim=args.hidden_dim,
                max_seq_len=args.seq_len, self_loop_init=0.0, **constraints)
            model.transition.trainable = method != "softmax"
            fit = _train(
                model, {"obs": x_train, "mask": masks[0]}, {"obs": x_val, "mask": masks[1]}, (y_train, y_val), args)
            path = diagnostics.decode(model, {"obs": x_test, "mask": masks[2]}, args.batch_size)[0]
            yield {
                "experiment": "crf", "seed": seed, "method": method, **fit,
                "accuracy": accuracy(path, y_test),
                "sequence_exact": float((path == y_test).all(axis=1).mean()),
                "invalid_path_rate": invalid_bio_rate(path),
            }


def sample_autoregressive(
        n_sequences: int, steps: int, n_states: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw sticky states with ``x_t = rho x_{t-1} + (1 - rho) mu_z + noise``.

    :param n_sequences: ``N``.
    :param steps: ``T``.
    :param n_states: ``K``.
    :param seed: Seed.
    :return: ``(obs (N, T, 2) float32, states (N, T) int32, log_emit (N, T, K))``, the
        last being the true ``log p(x_t | x_{t-1}, z_t = k)``.
    """
    rng = np.random.default_rng(seed)
    means = (np.arange(n_states) - (n_states - 1) / 2.0)[:, None] * 4.0 / np.sqrt(2.0) * np.ones((n_states, 2))
    states = np.zeros((n_sequences, steps), dtype=np.int32)
    states[:, 0] = rng.integers(0, n_states, n_sequences)
    for t in range(1, steps):
        stay = rng.random(n_sequences) < 0.9
        states[:, t] = np.where(stay, states[:, t - 1], rng.integers(0, n_states, n_sequences))
    obs = np.zeros((n_sequences, steps, 2))
    obs[:, 0] = means[states[:, 0]] + rng.normal(size=(n_sequences, 2))
    for t in range(1, steps):
        center = AR_RHO * obs[:, t - 1] + (1.0 - AR_RHO) * means[states[:, t]]
        obs[:, t] = center + AR_NOISE_SD * rng.normal(size=(n_sequences, 2))
    centers = np.concatenate([
        np.broadcast_to(means, (n_sequences, 1, n_states, 2)),
        AR_RHO * obs[:, :-1, None] + (1.0 - AR_RHO) * means[None, None],
    ], axis=1)
    sds = np.full((n_sequences, steps, 1, 1), AR_NOISE_SD)
    sds[:, 0] = 1.0
    log_emit = -0.5 * np.sum(np.square((obs[:, :, None] - centers) / sds) + np.log(2.0 * np.pi * sds ** 2), axis=-1)
    return obs.astype(np.float32), states, log_emit


def compare_autoregressive(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    """Yield one row per seed and method on data whose history predicts ``x_t`` strongly.

    :param args: Parsed arguments.
    :return: Rows.
    """
    n_states = 3
    pi = np.full(n_states, 1.0 / n_states)
    transition = np.full((n_states, n_states), 0.1 / n_states) + 0.9 * np.eye(n_states)
    for seed in range(args.crf_seeds):
        x_train, z_train, _ = sample_autoregressive(args.num_train, args.seq_len, n_states, seed * 3 + 1)
        x_val, z_val, _ = sample_autoregressive(args.num_val, args.seq_len, n_states, seed * 3 + 2)
        x_test, z_test, log_emit = sample_autoregressive(args.num_test, args.seq_len, n_states, seed * 3 + 3)
        masks = [np.ones(x.shape[:2], "float32") for x in (x_train, x_val, x_test)]
        test = HMMDataset(x_test, masks[2], z_test, None)
        true_ll = baselines.forward_backward(pi, transition, log_emit, masks[2])[2]
        true_nll = _nll(true_ll, masks[2])
        true_path = keras.ops.convert_to_numpy(inference.viterbi_decode(
            np.log(pi)[None].repeat(len(x_test), axis=0),
            np.broadcast_to(np.log(transition), (*masks[2].shape, n_states, n_states)).copy(), log_emit, masks[2])[0])
        gamma = baselines.forward_backward(pi, transition, log_emit, masks[2])[0]
        shuffle = np.argsort(np.random.default_rng(seed).random(log_emit.shape), axis=-1)
        shuffled_ll = baselines.forward_backward(pi, transition, np.take_along_axis(log_emit, shuffle, axis=-1), masks[2])[2]
        row = {
            "experiment": "autoregressive", "seed": seed, "true_accuracy": accuracy(true_path, z_test),
            "true_state_shuffled_gap": _nll(shuffled_ll, masks[2]) - true_nll,
            "true_mean_posterior_entropy": float(-np.sum(gamma * np.log(np.maximum(gamma, 1e-300))) / masks[2].sum()),
        }
        for variant in ("autoregressive-neural-hmm", "neural-hmm"):
            set_seeds(seed)
            model = create_neural_hmm(
                variant, n_states=n_states, obs_type="gaussian", obs_dim=OBS_DIM, hidden_dim=args.hidden_dim)
            fit = _train(model, {"obs": x_train, "mask": masks[0]}, {"obs": x_val, "mask": masks[1]}, (None, None), args)
            yield {**row, "method": variant, **fit, **_score_neural(model, test, true_nll, args)}
        fitted = baselines.fit_baum_welch(x_train, masks[0], n_states, seed=seed)
        yield {**row, "method": "baum-welch-kmeans", **_score_baum_welch(fitted, test, true_nll)}


RUNNERS = {
    "baum-welch": compare_with_baum_welch, "crf": compare_crf, "autoregressive": compare_autoregressive,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="JSON-lines file, appended to.")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--states", type=int, nargs="+", default=[4, 16], help="K of the baum-welch experiment.")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds of the baum-welch experiment.")
    parser.add_argument("--crf-seeds", type=int, default=3, help="Seeds of the other two experiments.")
    parser.add_argument("--num-train", type=int, default=1000, help="Training sequences.")
    parser.add_argument("--num-val", type=int, default=200, help="Validation sequences.")
    parser.add_argument("--num-test", type=int, default=200, help="Test sequences.")
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Width of the neural models.")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Sequences per batch.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Adam learning rate.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Run the requested experiments, appending each row to ``--output`` as it finishes.

    :param argv: Arguments, or ``None`` for ``sys.argv``.
    :return: Every row of this run.
    """
    args = _build_parser().parse_args(argv)
    rows: List[Dict[str, Any]] = []
    for name in args.experiments:
        for row in RUNNERS[name](args):
            rows.append(row)
            with args.output.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
    return rows


if __name__ == "__main__":
    main()

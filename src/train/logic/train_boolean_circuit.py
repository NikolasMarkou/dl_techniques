"""
Validate ``LearnableNeuralCircuit`` end-to-end on synthetic boolean tasks.

Three tasks (CLI-selectable via ``--task``):

- ``parity``      — y = XOR(x1, ..., xK). Linearly inseparable at K>=2; the
                    classical XOR-generalization benchmark.
- ``majority``    — y = 1 iff sum(x) >= K/2. Linear baseline; sanity check.
- ``random_dnf``  — y = OR(AND-of-literals, ...) over a random k-term k-DNF
                    formula sampled at startup under a fixed seed.

After training we print the per-depth dominant operator picked by each inner
expert via ``LearnableNeuralCircuit.to_symbolic()`` — this is the empirical
proof that the circuit converged to an interpretable structure.

Usage::

    MPLBACKEND=Agg .venv/bin/python -m train.logic.train_boolean_circuit \\
        --task parity --num-bits 4 --epochs 50 --gpu 1

Plan: plan_2026-05-13_d256b568.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, Tuple

import keras
import numpy as np

from dl_techniques.layers.logic import LearnableNeuralCircuit
from dl_techniques.utils.logger import logger
from train.common import setup_gpu, create_callbacks


# ---------------------------------------------------------------------
# Data generators — each returns (X: (N, K) float32, y: (N, 1) float32)
# ---------------------------------------------------------------------

def gen_parity(num_samples: int, num_bits: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """y = XOR(x1, ..., xK) = (sum % 2)."""
    x = rng.integers(0, 2, size=(num_samples, num_bits)).astype(np.float32)
    y = (x.sum(axis=1) % 2 == 1).astype(np.float32).reshape(-1, 1)
    return x, y


def gen_majority(num_samples: int, num_bits: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """y = 1 if sum(x) >= K/2 else 0 (ties broken in favor of 1)."""
    x = rng.integers(0, 2, size=(num_samples, num_bits)).astype(np.float32)
    y = (x.sum(axis=1) >= (num_bits / 2.0)).astype(np.float32).reshape(-1, 1)
    return x, y


def _sample_dnf_formula(
    num_bits: int, num_terms: int, term_size: int, rng: np.random.Generator
) -> list:
    """Sample a k-term k-DNF: list of terms; each term is a list of
    (bit_index, polarity in {0, 1}) tuples interpreted as AND of literals.
    """
    formula = []
    for _ in range(num_terms):
        idx = rng.choice(num_bits, size=term_size, replace=False)
        pol = rng.integers(0, 2, size=term_size)
        formula.append(list(zip(idx.tolist(), pol.tolist())))
    return formula


def _eval_dnf(x: np.ndarray, formula: list) -> np.ndarray:
    """Evaluate a DNF formula on a batch of bit-vectors. Returns float32 (N,1)."""
    # OR of (AND over each term).
    n = x.shape[0]
    out = np.zeros(n, dtype=bool)
    for term in formula:
        term_truth = np.ones(n, dtype=bool)
        for bit_idx, polarity in term:
            literal = x[:, bit_idx] > 0.5 if polarity == 1 else x[:, bit_idx] < 0.5
            term_truth &= literal
        out |= term_truth
    return out.astype(np.float32).reshape(-1, 1)


def gen_random_dnf(num_samples: int, num_bits: int, rng: np.random.Generator,
                   num_terms: int = 3, term_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """y = OR(AND of literals) for a fixed random formula (seeded via rng)."""
    formula = _sample_dnf_formula(num_bits, num_terms, term_size, rng)
    x = rng.integers(0, 2, size=(num_samples, num_bits)).astype(np.float32)
    y = _eval_dnf(x, formula)
    return x, y


TASK_REGISTRY: Dict[str, Callable] = {
    "parity": gen_parity,
    "majority": gen_majority,
    "random_dnf": gen_random_dnf,
}


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model(
    num_bits: int,
    channels: int = 16,
    circuit_depth: int = 2,
    num_logic_ops: int = 2,
    num_arith_ops: int = 2,
    gate_entropy_coef: float = 0.0,
    diversity_coef: float = 0.0,
    learning_rate: float = 3e-3,
) -> keras.Model:
    """Bit-vector classifier with a LearnableNeuralCircuit middle.

    Architecture::

        Input(K,) -> Dense(channels, relu) -> LayerNorm
                  -> LearnableNeuralCircuit
                  -> Dense(1, sigmoid)
    """
    inputs = keras.Input(shape=(num_bits,), name="bits")
    x = keras.layers.Dense(channels, activation="relu", name="embed")(inputs)
    x = keras.layers.LayerNormalization(name="embed_norm")(x)
    x = LearnableNeuralCircuit(
        circuit_depth=circuit_depth,
        num_logic_ops_per_depth=num_logic_ops,
        num_arithmetic_ops_per_depth=num_arith_ops,
        use_residual=True,
        use_layer_norm=True,
        gate_entropy_coefficient=gate_entropy_coef,
        diversity_coefficient=diversity_coef,
        name="neural_circuit",
    )(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="head")(x)
    model = keras.Model(inputs, outputs, name=f"boolean_circuit_{num_bits}b")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------

def exact_accuracy_for_parity(model: keras.Model, num_bits: int) -> float:
    """For small K, enumerate ALL 2^K inputs and compute exact accuracy."""
    n = 1 << num_bits
    x = np.array(
        [[(i >> b) & 1 for b in range(num_bits)] for i in range(n)],
        dtype=np.float32,
    )
    y = (x.sum(axis=1) % 2 == 1).astype(np.float32).reshape(-1, 1)
    y_pred = (model.predict(x, verbose=0) > 0.5).astype(np.float32)
    return float((y_pred == y).mean())


def round_trip_check(model: keras.Model, X_test: np.ndarray, save_dir: str) -> float:
    """Save + reload + re-predict; return max-abs prediction diff."""
    path = os.path.join(save_dir, "model.keras")
    model.save(path)
    reloaded = keras.models.load_model(path)
    p1 = model.predict(X_test, verbose=0)
    p2 = reloaded.predict(X_test, verbose=0)
    return float(np.max(np.abs(p1 - p2)))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate LearnableNeuralCircuit on synthetic boolean tasks."
    )
    p.add_argument("--task", choices=list(TASK_REGISTRY.keys()), default="majority")
    p.add_argument("--num-bits", type=int, default=4)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--circuit-depth", type=int, default=2)
    p.add_argument("--num-logic-ops", type=int, default=2)
    p.add_argument("--num-arith-ops", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--train-samples", type=int, default=4096)
    p.add_argument("--test-samples", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--gate-entropy-coef", type=float, default=0.0)
    p.add_argument("--diversity-coef", type=float, default=0.0)
    p.add_argument("--results-dir-prefix", type=str, default="logic")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_gpu(args.gpu)
    keras.utils.set_random_seed(args.seed)
    rng_train = np.random.default_rng(args.seed)
    rng_test = np.random.default_rng(args.seed + 1)

    logger.info(
        f"task={args.task} num_bits={args.num_bits} channels={args.channels} "
        f"depth={args.circuit_depth} epochs={args.epochs} lr={args.lr}"
    )

    gen = TASK_REGISTRY[args.task]
    X_train, y_train = gen(args.train_samples, args.num_bits, rng_train)
    # For random_dnf the formula is sampled inside the generator from the rng;
    # use the same seed for the test set so the same formula is sampled.
    rng_test_for_dnf = np.random.default_rng(args.seed)
    if args.task == "random_dnf":
        # Replay the formula by re-seeding then drawing samples.
        # gen_random_dnf does: sample formula, then sample x. We can't easily
        # decouple the two without changing the API; instead generate a large
        # train pool and split deterministically.
        X_full, y_full = gen(args.train_samples + args.test_samples, args.num_bits,
                             np.random.default_rng(args.seed))
        X_train, y_train = X_full[: args.train_samples], y_full[: args.train_samples]
        X_test, y_test = X_full[args.train_samples:], y_full[args.train_samples:]
    else:
        X_test, y_test = gen(args.test_samples, args.num_bits, rng_test)

    # Sanity: log positive-class fraction (catches degenerate generators).
    logger.info(
        f"train positives: {y_train.mean():.3f}; test positives: {y_test.mean():.3f}"
    )

    model = build_model(
        num_bits=args.num_bits,
        channels=args.channels,
        circuit_depth=args.circuit_depth,
        num_logic_ops=args.num_logic_ops,
        num_arith_ops=args.num_arith_ops,
        gate_entropy_coef=args.gate_entropy_coef,
        diversity_coef=args.diversity_coef,
        learning_rate=args.lr,
    )
    model.summary(print_fn=logger.info)

    callbacks, results_dir = create_callbacks(
        model_name=f"{args.task}_k{args.num_bits}",
        results_dir_prefix=args.results_dir_prefix,
        monitor="val_accuracy",
        patience=max(10, args.epochs // 3),
        use_lr_schedule=False,
        include_analyzer=False,
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Final eval.
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"FINAL test loss={test_loss:.4f} acc={test_acc:.4f}")
    if args.task == "parity" and args.num_bits <= 12:
        exact_acc = exact_accuracy_for_parity(model, args.num_bits)
        logger.info(f"EXACT parity accuracy over all 2^{args.num_bits}: {exact_acc:.4f}")

    # Symbolic readout.
    nc = model.get_layer("neural_circuit")
    logger.info("LearnableNeuralCircuit.to_symbolic(top_k=3):")
    for line in nc.to_symbolic(top_k=3).split("\n"):
        logger.info(f"  {line}")

    # Save + reload + round-trip.
    max_diff = round_trip_check(model, X_test, results_dir)
    logger.info(f"save/load max-abs prediction diff: {max_diff:.2e}")
    assert max_diff < 1e-6, f"round-trip diff too large: {max_diff:.2e}"

    logger.info(f"DONE. Results in {results_dir}")


if __name__ == "__main__":
    main()

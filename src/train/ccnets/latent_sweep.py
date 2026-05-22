"""Latent-size sweep: how big should the CCNet latent cause E be?

Trains the MNIST CCNet at ``explanation_dim`` in {4, 8, 16, 32, 64, 128} and, for
each, measures three things on the test set:

  * reconstruction / generation MAE   -- can the Producer rebuild X? (the
    sufficiency lower bound: too-small E => underdetermined => blur)
  * label-sensitivity ``mean|x_gen(y) - x_gen(y')|`` -- does the Producer still
    depend on Y? (the disentanglement upper failure: too-large E can absorb Y,
    and this metric falls toward 0)
  * achieved KL divergence in nats    -- the *effective rate* E actually carries,
    as opposed to its dimensional ceiling.

Together these locate the usable band for dim(E): large enough to reconstruct,
small enough that E does not swallow the explicit cause Y. See the discussion in
``models/ccnets/PRINCIPLES_CCNETS.md`` (P3) and the H(X|Y) argument.

Run:
    MPLBACKEND=Agg .venv/bin/python -m train.ccnets.latent_sweep
"""

import os
from typing import Dict, List

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.models.ccnets import CCNetTrainer
from train.ccnets.mnist import (
    ModelConfig, TrainingConfig, DataConfig, ExperimentConfig,
    create_mnist_ccnet, prepare_mnist_data,
)

# --------------------------------------------------------------------- config
LATENT_DIMS = [4, 8, 16, 32, 64, 128]
SEED = 0
EPOCHS = 20
NUM_EVAL = 4000
RESULTS_DIR = "results/ccnets_latent_sweep"


# --------------------------------------------------------------------- eval
def evaluate(orchestrator, x_test, y_test, num_eval=NUM_EVAL) -> Dict[str, float]:
    """Reconstruction quality, label-sensitivity, and achieved KL rate."""
    x = x_test[:num_eval]
    y = y_test[:num_eval]
    y_oh = keras.utils.to_categorical(y, 10).astype("float32")
    y_shift = keras.utils.to_categorical((y + 1) % 10, 10).astype("float32")
    x_tf = tf.convert_to_tensor(x)

    # Full pipeline: reconstruction MAE + reasoner accuracy + latent stats.
    tensors = orchestrator.forward_pass(x_tf, tf.convert_to_tensor(y_oh), training=False)
    x_rec = keras.ops.convert_to_numpy(tensors["x_reconstructed"])
    y_inf = keras.ops.convert_to_numpy(tensors["y_inferred"])
    mu = keras.ops.convert_to_numpy(tensors["mu"])
    log_var = keras.ops.convert_to_numpy(tensors["log_var"])

    # Deterministic generation (uses mu, no sampling) -> clean, comparable.
    xg_true = keras.ops.convert_to_numpy(
        orchestrator.counterfactual_generation(x_tf, tf.convert_to_tensor(y_oh)))
    xg_shift = keras.ops.convert_to_numpy(
        orchestrator.counterfactual_generation(x_tf, tf.convert_to_tensor(y_shift)))

    # Achieved KL in nats: the effective rate E carries.
    kl = float(-0.5 * np.mean(
        np.sum(1.0 + log_var - np.square(mu) - np.exp(log_var), axis=1)))

    return {
        "reasoner_acc": float(np.mean(np.argmax(y_inf, axis=-1) == y)),
        "recon_mae": float(np.mean(np.abs(x_rec - x))),
        "gen_mae": float(np.mean(np.abs(xg_true - x))),
        "label_sensitivity": float(np.mean(np.abs(xg_true - xg_shift))),
        "kl_nats": kl,
    }


# --------------------------------------------------------------------- plot
def plot_sweep(dims: List[int], rows: List[Dict[str, float]], path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(dims, [r["recon_mae"] for r in rows], "o-", label="reconstruction MAE")
    axes[0].plot(dims, [r["gen_mae"] for r in rows], "s--", label="generation MAE")
    axes[0].set_title("Reconstruction quality\n(sufficiency: lower = better)")
    axes[0].set_ylabel("MAE")

    axes[1].plot(dims, [r["label_sensitivity"] for r in rows], "o-", color="darkred")
    axes[1].set_title("Label-sensitivity  mean|x_gen(y) − x_gen(y′)|\n"
                      "(disentanglement: → 0 means E absorbed Y)")
    axes[1].set_ylabel("MAE between label-swapped generations")

    axes[2].plot(dims, [r["kl_nats"] for r in rows], "o-", color="darkgreen")
    axes[2].set_title("Achieved KL (nats)\n(the effective rate E carries)")
    axes[2].set_ylabel("KL divergence (nats)")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("explanation_dim  (dim of E)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("CCNet MNIST — latent size dim(E) sweep", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(dims: List[int], rows: List[Dict[str, float]], path: str) -> str:
    lines = [
        "# CCNet latent-size sweep — MNIST",
        "",
        f"`explanation_dim` in {dims}. {EPOCHS} epochs each, seed {SEED}, "
        f"{NUM_EVAL}-sample test eval.",
        "",
        "| dim(E) | Reasoner acc | Recon MAE | Gen MAE | Label-sensitivity | KL (nats) |",
        "|-------:|-------------:|----------:|--------:|------------------:|----------:|",
    ]
    for d, r in zip(dims, rows):
        lines.append(
            f"| {d:>6} | {r['reasoner_acc']:.4f} | {r['recon_mae']:.4f} | "
            f"{r['gen_mae']:.4f} | {r['label_sensitivity']:.4f} | {r['kl_nats']:.3f} |"
        )
    lines += [
        "",
        "Reading the table:",
        "- **Recon / Gen MAE** falling then flattening marks the sufficiency lower "
        "bound — once dim(E) covers H(X|Y), more dimensions stop helping.",
        "- **Label-sensitivity** trending toward 0 marks the upper failure — E has "
        "enough capacity to encode Y itself, so the Producer stops needing the label.",
        "- **KL (nats)** is the effective rate; it should rise with dim(E) and then "
        "saturate as the KL term throttles further growth.",
        "- The usable band is where recon MAE is low **and** label-sensitivity is "
        "still high.",
    ]
    report = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(report + "\n")
    return report


# --------------------------------------------------------------------- main
def main() -> None:
    setup_gpu(None)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    rows: List[Dict[str, float]] = []
    for dim in LATENT_DIMS:
        logger.info(f"=== explanation_dim = {dim} ===")
        keras.utils.set_random_seed(SEED)
        config = ExperimentConfig(
            model=ModelConfig(explanation_dim=dim),
            training=TrainingConfig(epochs=EPOCHS, kl_annealing_epochs=EPOCHS // 3),
            data=DataConfig(batch_size=128),
        )
        orchestrator = create_mnist_ccnet(config)
        train_ds, _ = prepare_mnist_data(config.data)
        CCNetTrainer(orchestrator, kl_annealing_epochs=EPOCHS // 3).train(train_ds, EPOCHS)

        metrics = evaluate(orchestrator, x_test, y_test)
        rows.append(metrics)
        logger.info(f"dim={dim}: {metrics}")

    plot_sweep(LATENT_DIMS, rows, os.path.join(RESULTS_DIR, "latent_sweep.png"))
    report = write_report(LATENT_DIMS, rows, os.path.join(RESULTS_DIR, "sweep.md"))
    logger.info("\n" + report)


if __name__ == "__main__":
    main()

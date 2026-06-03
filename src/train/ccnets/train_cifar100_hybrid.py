"""Hybrid latent-verification CCNet on CIFAR-100 (training script).

A LeWM-inspired variant of the image CCNet. The plain CIFAR-100 CCNet
(``train_cifar100.py``) verifies the Producer purely in **pixel space** -- and we
saw it collapse to blurry class-average blobs, because a class label + modest
latent cannot determine a 32x32 natural photo (``PRINCIPLES_CCNETS.md``, P1/P2).

LeWM's key strength is that it verifies in **latent space** (JEPA-style): it
predicts/compares embeddings, not raw observations. This ports that as a
*hybrid*: the Producer still outputs pixels (so MNIST-style counterfactual
generation survives), but an additional verification term compares the produced
and real images in a learned **feature space**:

    producer_error  +=  w * ( ||phi(x_gen) - phi(x)||^2
                              + ||phi(x_recon) - phi(x)||^2 )
    explainer_error +=  w *   ||phi(x_gen) - phi(x)||^2

phi is the Reasoner's own image-feature backbone (``Cifar100Reasoner.image_features``)
-- a *trained, semantic* encoder, anchored against collapse by the classification
objective. The latent term is added only to the Producer and Explainer errors;
tape isolation (P7) keeps it out of ``reasoner_error``, so phi stays a stable
target. This is the JEPA "stop-gradient-ish target" obtained for free from the
three-tape design.

The hybrid orchestrator and the unified ``create_cifar100_ccnet(..., hybrid=True)``
factory live in ``dl_techniques.models.ccnets``; CIFAR-100 data prep / eval /
plotting are reused from the sibling ``train.ccnets.train_cifar100`` training
script (a sanctioned data-prep train-to-train edge, D-007).

Honest scope: phi is a *moving* target (the Reasoner trains throughout); an EMA or
frozen phi would be steadier. The latent weight is untuned.

Run:
    MPLBACKEND=Agg .venv/bin/python -m train.ccnets.train_cifar100_hybrid --gpu 0 --epochs 40
"""

import os
import argparse

import keras
import tensorflow as tf

from train.common import setup_gpu, set_seeds
from dl_techniques.utils.logger import logger
from dl_techniques.models.ccnets import CCNetTrainer
from dl_techniques.models.ccnets.architectures.cifar100 import (
    create_cifar100_ccnet,
    LATENT_WEIGHT,
)
# Data-prep / eval / plotting + training config are training-side; reuse them
# from the sibling CIFAR-100 training script (sanctioned data-prep edge, D-007).
from train.ccnets.train_cifar100 import (
    ExperimentConfig,
    prepare_cifar100,
    evaluate,
    plot_history,
)


# =====================================================================
# CONSTRUCTION
# =====================================================================

def build_hybrid_ccnet(config: ExperimentConfig):
    """Build the hybrid latent-verification CIFAR-100 orchestrator."""
    return create_cifar100_ccnet(
        config.model,
        loss_fn=config.training.loss_fn,
        learning_rates=config.training.learning_rates,
        gradient_clip_norm=config.training.gradient_clip_norm,
        explainer_weights=config.training.explainer_weights,
        reasoner_weights=config.training.reasoner_weights,
        producer_weights=config.training.producer_weights,
        hybrid=True,
        latent_weight=LATENT_WEIGHT,
    )


# =====================================================================
# RUN
# =====================================================================

def run_experiment(config: ExperimentConfig):
    results_dir = config.results_dir.replace("ccnets_cifar100", "ccnets_cifar100_hybrid")
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Building hybrid latent-verification CIFAR-100 CCNet "
                f"(latent_weight={LATENT_WEIGHT})...")
    orchestrator = build_hybrid_ccnet(config)

    logger.info("Loading CIFAR-100...")
    train_ds, x_test, y_test = prepare_cifar100(config)

    logger.info(f"Training for {config.training.epochs} epochs...")
    trainer = CCNetTrainer(orchestrator, kl_annealing_epochs=config.training.kl_annealing_epochs)
    trainer.train(train_ds, config.training.epochs)

    orchestrator.save_models(os.path.join(results_dir, "cifar100_hybrid_ccnet"))
    plot_history(trainer.history, os.path.join(results_dir, "training_history.png"))
    report = evaluate(orchestrator, x_test, y_test, results_dir)

    # Additional: the latent verification metric on a test batch.
    xb = tf.convert_to_tensor(x_test[:512])
    yb = tf.convert_to_tensor(keras.utils.to_categorical(y_test[:512], 100).astype("float32"))
    tensors = orchestrator.forward_pass(xb, yb, training=False)
    gen_latent, rec_latent = orchestrator._latent_losses(tensors)
    latent_line = (f"Latent (perceptual) verification MSE: "
                   f"gen = {float(gen_latent):.4f}, recon = {float(rec_latent):.4f}")
    logger.info(latent_line)

    full = report + "\n" + latent_line
    with open(os.path.join(results_dir, "report.txt"), "w") as fh:
        fh.write(full + "\n")
    logger.info(f"Artifacts saved to {results_dir}")
    return orchestrator


# =====================================================================
# ENTRY POINT
# =====================================================================

def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Map parsed CLI args onto the experiment config dataclasses."""
    config = ExperimentConfig()
    config.gpu = args.gpu
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size

    if args.smoke:
        config.model.explanation_dim = 16
        config.training.epochs = 1
        config.training.kl_annealing_epochs = 1
        config.data.batch_size = 64
        config.data.augment = False
        config.results_dir = "results/ccnets_cifar100_smoke"

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a hybrid latent-verification CCNet on CIFAR-100.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device index.")
    parser.add_argument('--epochs', type=int, default=None, help="Training epochs.")
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--smoke', action='store_true',
                        help="Tiny CI-safe run (1 epoch, small dims).")
    args = parser.parse_args()

    setup_gpu(args.gpu)
    set_seeds(args.seed)

    config = build_config(args)
    run_experiment(config)


if __name__ == "__main__":
    main()

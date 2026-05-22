"""Hybrid latent-verification CCNet on CIFAR-100 (prototype).

A LeWM-inspired variant of the image CCNet. The plain CIFAR-100 CCNet
(`cifar100.py`) verifies the Producer purely in **pixel space** -- and we saw it
collapse to blurry class-average blobs, because a class label + modest latent
cannot determine a 32x32 natural photo (`PRINCIPLES_CCNETS.md`, P1/P2).

LeWM's key strength is that it verifies in **latent space** (JEPA-style): it
predicts/compares embeddings, not raw observations. This prototype ports that as
a *hybrid*: the Producer still outputs pixels (so MNIST-style counterfactual
generation survives), but an additional verification term compares the produced
and real images in a learned **feature space**:

    producer_error  +=  w * ( ||phi(x_gen) - phi(x)||^2
                              + ||phi(x_recon) - phi(x)||^2 )
    explainer_error +=  w *   ||phi(x_gen) - phi(x)||^2

phi is the Reasoner's own image-feature backbone (`Cifar100Reasoner.image_features`)
-- a *trained, semantic* encoder, anchored against collapse by the classification
objective. The latent term is added only to the Producer and Explainer errors;
tape isolation (P7) keeps it out of `reasoner_error`, so phi stays a stable target
(the Reasoner backbone is trained only by classification + cooperative recon, not
by the perceptual loss). This is the JEPA "stop-gradient-ish target" obtained for
free from the three-tape design.

Honest scope: phi is a *moving* target (the Reasoner trains throughout); an EMA or
frozen phi would be steadier. The latent weight is untuned. This prototype tests
whether a semantic verification signal changes what the Producer learns versus
pure pixel L1.

Run:
    MPLBACKEND=Agg .venv/bin/python -m train.ccnets.cifar100_hybrid
"""

import os
from typing import Dict

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger
from dl_techniques.models.ccnets import CCNetConfig, CCNetOrchestrator, wrap_keras_model
from dl_techniques.models.ccnets.base import CCNetModelErrors
from dl_techniques.models.ccnets import CCNetTrainer
from train.common import setup_gpu
from train.ccnets.cifar100 import (
    ModelConfig, ExperimentConfig,
    Cifar100Explainer, Cifar100Reasoner, Cifar100Producer,
    prepare_cifar100, evaluate, plot_history,
)

# Weight on the latent (perceptual) verification term. Untuned; kept modest so
# the pixel loss still grounds generation. The run logs both magnitudes.
LATENT_WEIGHT = 0.25


# =====================================================================
# HYBRID ORCHESTRATOR
# =====================================================================

class HybridCCNetOrchestrator(CCNetOrchestrator):
    """CCNet orchestrator with an added latent-space (perceptual) verification term.

    `perceptual_model` must expose `image_features(x, training)` — used as the
    encoder phi. The latent term enters only the Producer and Explainer errors;
    the three-tape design then guarantees phi (the Reasoner backbone) is not
    updated by it.
    """

    def __init__(self, *args, perceptual_model, latent_weight=LATENT_WEIGHT, **kwargs):
        super().__init__(*args, **kwargs)
        self.perceptual_model = perceptual_model
        self.latent_weight = latent_weight

    def _latent_losses(self, tensors):
        phi_x = self.perceptual_model.image_features(tensors["x_input"], training=False)
        phi_gen = self.perceptual_model.image_features(tensors["x_generated"], training=False)
        phi_rec = self.perceptual_model.image_features(tensors["x_reconstructed"], training=False)
        gen_latent = keras.ops.mean(keras.ops.square(phi_gen - phi_x))
        rec_latent = keras.ops.mean(keras.ops.square(phi_rec - phi_x))
        return gen_latent, rec_latent

    def compute_model_errors(self, losses, tensors) -> CCNetModelErrors:
        errors = super().compute_model_errors(losses, tensors)   # pixel-space errors
        gen_latent, rec_latent = self._latent_losses(tensors)
        w = self.latent_weight
        return CCNetModelErrors(
            explainer_error=errors.explainer_error + w * gen_latent,
            reasoner_error=errors.reasoner_error,                      # unchanged: phi stays anchored
            producer_error=errors.producer_error + w * (gen_latent + rec_latent),
        )


# =====================================================================
# CONSTRUCTION
# =====================================================================

def create_hybrid_ccnet(config: ExperimentConfig) -> HybridCCNetOrchestrator:
    mc = config.model
    explainer = Cifar100Explainer(mc)
    reasoner = Cifar100Reasoner(mc)
    producer = Cifar100Producer(mc)

    dummy_x = keras.ops.zeros((1, 32, 32, mc.image_channels))
    dummy_y = keras.ops.zeros((1, mc.num_classes))
    dummy_e = keras.ops.zeros((1, mc.explanation_dim))
    explainer(dummy_x)
    reasoner(dummy_x, dummy_e)
    producer(dummy_y, dummy_e)

    ccnet_config = CCNetConfig(
        explanation_dim=mc.explanation_dim,
        loss_fn=config.training.loss_fn,
        learning_rates=config.training.learning_rates,
        gradient_clip_norm=config.training.gradient_clip_norm,
        explainer_weights=config.training.explainer_weights,
        reasoner_weights=config.training.reasoner_weights,
        producer_weights=config.training.producer_weights,
    )
    return HybridCCNetOrchestrator(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=ccnet_config,
        perceptual_model=reasoner,           # phi = the Reasoner's image features
        latent_weight=LATENT_WEIGHT,
    )


# =====================================================================
# RUN
# =====================================================================

def run_experiment(config: ExperimentConfig) -> HybridCCNetOrchestrator:
    setup_gpu(config.gpu)
    results_dir = config.results_dir.replace("ccnets_cifar100", "ccnets_cifar100_hybrid")
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Building hybrid latent-verification CIFAR-100 CCNet "
                f"(latent_weight={LATENT_WEIGHT})...")
    orchestrator = create_hybrid_ccnet(config)

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


if __name__ == "__main__":
    run_experiment(ExperimentConfig())

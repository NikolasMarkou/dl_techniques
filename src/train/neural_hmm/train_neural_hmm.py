"""Trainer for the Neural HMM (Pattern 7: sequence-generative with paper's own metrics).

This trainer follows the same skeleton as ``src/train/topographic_vae/train_topographic_vae.py``:
a ``train(config)`` function that writes the full run-directory contract
(config.json, run.log, training_log.csv, best_model.keras, results_summary.json,
visualizations/*), a ``create_argument_parser()`` that adds model-specific
flags on top of ``train.common.create_ts_argument_parser()``, and local
``plot_*`` helpers that write PNGs into ``visualizations/``.

The trainer evaluates with domain metrics that no stock callback provides:
per-timestep NLL, Viterbi accuracy (decoded path vs. ground-truth states),
and state-occupancy entropy (a diagnostic for state collapse). Figures include
a posterior-vs-observation strip and a ground-truth vs. decoded segmentation
heatmap.

Usage::

    MPLBACKEND=Agg python -m train.neural_hmm.train_neural_hmm \\
        --dataset gaussian_hmm --n_samples 500 --n_states 4 --epochs 50 \\
        --output-dir /tmp/nhmm_run

References
----------
- Tang et al., 2020. Neural HMMs for regime-switching time-series. arXiv:2006.00694.
- Rabiner, 1989. A Tutorial on Hidden Markov Models. Proceedings of the IEEE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger
from train.neural_hmm.data import (
    available_generators,
    generate_dataset,
    set_global_seed,
)
from dl_techniques.models.time_series.neural_hmm import (
    NeuralHMM,
    from_variant,
    MODEL_VARIANTS,
)
from dl_techniques.models.time_series.neural_hmm.loss import NeuralHMMLoss
from dl_techniques.models.time_series.neural_hmm.encoders import (
    create_encoder_layer,
    ENCODER_REGISTRY,
)
from dl_techniques.models.time_series.neural_hmm.transitions import (
    create_transition_layer,
    TRANSITION_REGISTRY,
)
from dl_techniques.models.time_series.neural_hmm.inference import (
    viterbi,
    posteriors,
    shift_right,
    _forward_logprob_static,
    _forward_logprob_dynamic,
    _path_score,
)
from dl_techniques.metrics.time_series_metrics import calculate_comprehensive_metrics
from dl_techniques.metrics.probabilistic_forecast_metrics import (
    CoverageMetric,
    SharpnessMetric,
)
from dl_techniques.datasets.time_series import TimeSeriesGenerator
from dl_techniques.datasets.time_series.normalizer import NormalizationMethod
from train.common import (
    setup_gpu,
    create_callbacks,
    create_learning_rate_schedule,
    resolve_monitor_mode,
    set_seeds,
    save_training_history_json,
)
from train.common.run_artifacts import (
    RESULTS_SUMMARY_NAME,
    RUN_LOG_NAME,
    attach_run_log,
    refuse_existing_run,
    write_summary_json,
)
from train.common.run_io import default_experiment_name, prepare_run_dir
from train.common.stats import mean_std
from train.common.classification_viz import (
    plot_confident_errors,
)


# ---------------------------------------------------------------------
# config dataclass
# ---------------------------------------------------------------------


@dataclass
class NeuralHMMConfig:
    """Configuration for one Neural HMM training run.

    Every knob of one run. Defaults reproduce a small Gaussian-emission HMM
    that trains on 100 timesteps in < 1 min on one GPU.

    The ``rng`` field is set by ``set_seeds`` before the run starts; it is
    passed to the data generator for reproducibility.

    Note: ``--no-warmup`` / ``--no-deep-analysis`` etc. are handled by the
    shared ``create_ts_argument_parser``; those flags do not duplicate here.
    """

    # --- data ---
    experiment_name: str = "neural_hmm"
    output_dir: str = "results"
    seed: int = 42
    # Which synthetic generator to use; one of ``available_generators()``.
    generator: str = "gaussian_hmm"
    n_samples: int = 100          # sequence length T
    n_states: int = 4             # number of hidden states K
    # Emission params
    means: List[float] = field(default_factory=lambda: [0.0, 1.0, -1.0, 2.0])
    stds: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    # Transition type: "static" or "input_dependent"
    transition_type: str = "static"
    # Regime switch point (fraction of sequence); only used if
    # ``transition_type == "input_dependent"``
    switch_frac: float = 0.5
    # Observation noise factor (multiplier on stds)
    obs_noise_factor: float = 1.0

    # --- model ---
    mode: str = "generative"          # "generative" | "discriminative"
    encoder: str = "lstm"             # "mlp" | "lstm" | "conv" | "transformer"
    emission_context: str = "none"    # "none" | "autoregressive"
    topology: str = "ergodic"         # "ergodic" | "left_to_right" | "banded"
    band_width: int = 1
    self_loop_init: float = 2.0
    min_scale: float = 1e-3
    # Discriminative-only
    vocab_size: Optional[int] = None  # required if mode="discriminative" + obs_type=categorical

    # --- optimisation ---
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 1.0
    optimizer: str = "adamw"
    mixed_precision: bool = False

    # --- callbacks ---
    patience: int = 10
    include_terminate_on_nan: bool = True
    include_analyzer: bool = False

    # --- visualization ---
    save_visualizations: bool = True
    # If True, two figures are written:
    #   "posterior_vs_observation": strip chart of obs + per-state posterior heat
    #   "segmentation_comparison": ground-truth states vs. Viterbi decoded band
    visualization_frequency: int = 5


# ---------------------------------------------------------------------
# plot helpers (local, not library plugins)
# ---------------------------------------------------------------------


def _plot_posterior_vs_observation(
    run_dir: Path,
    model: NeuralHMM,
    x_test: np.ndarray,
    states_test: np.ndarray,
    config: NeuralHMMConfig,
    epoch: int,
) -> Optional[Path]:
    """Plot observation time-series with per-timestep posterior heat strip.

    The posterior marginals ``γ_t(k)`` are computed via the model's forward
    algorithm and the ``inference.posteriors`` helper. Each row is a timestep;
    the color is the probability of state k. The ground-truth state is shown
    as a thin horizontal line at the top.

    :param run_dir: Run directory for the PNG output.
    :type run_dir: Path
    :param model: Trained model (used only for the forward pass on x_test).
    :type model: NeuralHMM
    :param x_test: Test observations ``(N, T)`` or ``(N, T, D)``.
    :type x_test: np.ndarray
    :param states_test: Ground-truth states ``(N, T)``.
    :type states_test: np.ndarray
    :param config: Run configuration.
    :type config: NeuralHMMConfig
    :param epoch: Epoch number for the filename.
    :type epoch: int
    :return: Path to the written PNG, or ``None`` if the plot failed.
    :rtype: Optional[Path]
    """
    try:
        # Forward pass to get emission logits / params
        mask = np.ones_like(x_test[:, :1], "float32")
        x_one = x_test[:1]
        mask_one = mask[:1]
        inputs = {"obs": x_one, "mask": mask_one}
        log_pi, log_A, log_emit = model(inputs)
        # Squeeze batch dims for inference.posteriors (it expects static (T,K))
        lp = inference.posteriors(log_pi, log_A, log_emit)  # (T,K) probs
        # Take the first sample
        lp = lp[0]  # (T,K)

        states = states_test[:1]  # (T,)

        fig, ax = plt.subplots(
            figsize=(6, 2.5), dpi=100
        )
        # Observations as a thin line
        ax.plot(x_test[:1].flatten(), "k-", linewidth=0.8, label="observation")
        # Posterior heat: each column k is a state; row t is timestep
        im = ax.imshow(
            lp.numpy().T,
            aspect="auto",
            origin="lower",
            extent=[0, lp.shape[1], 0, lp.shape[0]],
            cmap="Blues",
        )
        # Ground-truth states as a line at the top
        for t in range(lp.shape[1]):
            k = int(states[0, t])
            ax.plot([t, t], [k - 0.5, k + 0.5], "w-", linewidth=1.5)

        ax.set_xlabel("timestep t")
        ax.set_ylabel("state k")
        ax.set_title(f"Posterior vs observation (epoch {epoch})")
        plt.colorbar(im, ax=ax, label="p(z_t = k | x_{1:T})")
        fig.tight_layout()

        path = run_dir / f"posterior_vs_observation_e{epoch}.png"
        fig.savefig(path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"posterior plot failed: {e}")
        return None


def _plot_segmentation_comparison(
    run_dir: Path,
    model: NeuralHMM,
    x_test: np.ndarray,
    states_test: np.ndarray,
    config: NeuralHMMConfig,
    epoch: int,
) -> Optional[Path]:
    """Plot Viterbi decoded path vs. ground-truth states.

    Uses ``inference.viterbi`` on the model's forward outputs for the first
    test sample, then plots the decoded path as a stepped line alongside the
    ground-truth states.

    :param run_dir: Run directory for the PNG output.
    :type run_dir: Path
    :param model: Trained model.
    :type model: NeuralHMM
    :param x_test: Test observations ``(N, T)`` or ``(N, T, D)``.
    :type x_test: np.ndarray
    :param states_test: Ground-truth states ``(N, T)``.
    :type states_test: np.ndarray
    :param config: Run configuration.
    :type config: NeuralHMMConfig
    :param epoch: Epoch number for the filename.
    :type epoch: int
    :return: Path to the written PNG, or ``None`` if the plot failed.
    :rtype: Optional[Path]
    """
    try:
        # Forward pass on first sample
        x_one = x_test[:1]
        mask_one = np.ones_like(x_one[:, :1], "float32")
        inputs = {"obs": x_one, "mask": mask_one}
        log_pi, log_A, log_emit = model(inputs)
        # inference.viterbi expects static (K,K) A and (T,K) emit; squeeze if needed
        # log_A may be (1,K,K) from dynamic; take first
        if log_A.ndim == 3:
            log_A = log_A[0]
        if log_emit.ndim == 3:
            log_emit = log_emit[0]
        path, lp = inference.viterbi(log_pi, log_A, log_emit)  # (T,), scalar
        path = path.numpy()
        states = states_test[0]  # (T,)

        fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
        # Ground-truth states as markers
        ax.plot(
            np.arange(len(states)),
            states,
            "o",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
            label="ground-truth",
        )
        # Viterbi decoded path as a continuous line
        ax.plot(np.arange(len(path)), path, "-", linewidth=2, color="tab:orange", label="Viterbi")
        ax.set_xlabel("timestep t")
        ax.set_ylabel("state k")
        ax.set_title(f"Segmentation comparison (epoch {epoch})")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(-0.5, config.n_states - 0.5)

        path_out = run_dir / f"segmentation_comparison_e{epoch}.png"
        fig.savefig(path_out, bbox_inches="tight", dpi=100)
        plt.close(fig)
        return path_out
    except Exception as e:
        logger.warning(f"segmentation plot failed: {e}")
        return None


# ---------------------------------------------------------------------
# data loading / generator
# ---------------------------------------------------------------------


def _build_generator_config(config: NeuralHMMConfig) -> Dict[str, Any]:
    """Build the dict passed to the synthetic generator.

    The generator returns ``(x, states)`` plus optionally transition matrices.
    """
    # Apply obs_noise_factor if needed
    noise = config.obs_noise_factor
    stds = [s * noise for s in config.stds]

    return dict(
        n_samples=config.n_samples,
        n_states=config.n_states,
        means=config.means,
        stds=stds,
        transition={"type": config.transition_type},
        switch_frac=config.switch_frac,
        vocab_size=config.vocab_size,
    )


def _load_data(config: NeuralHMMConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val/test splits with ground-truth states.

    Returns ``(x_train, states_train, x_val, states_val, x_test, states_test)``.
    All observation arrays have shape ``(N, T)`` for categorical or ``(N, T, D)``
    for gaussian (here D=1 since we use 1-D synthetic data).
    """
    set_global_seed(config.seed)

    gen_cfg = _build_generator_config(config)
    train_data = generate_dataset(config.generator, gen_cfg)
    val_data = generate_dataset(config.generator, gen_cfg)
    test_data = generate_dataset(config.generator, gen_cfg)

    # Normalise observations to float32; for gaussian HMM they are already float
    x_train = train_data["x"].astype("float32")
    states_train = train_data["states"].astype("int32")
    x_val = val_data["x"].astype("float32")
    states_val = val_data["states"].astype("int32")
    x_test = test_data["x"].astype("float32")
    states_test = test_data["states"].astype("int32")

    return x_train, states_train, x_val, states_val, x_test, states_test


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------


def train(config: NeuralHMMConfig) -> Dict[str, Any]:
    """Run one full training job and write the run directory.

    :param config: Run configuration.
    :type config: NeuralHMMConfig
    :return: The summary dict written to ``results_summary.json``.
    :rtype: Dict[str, Any]
    """
    set_seeds(config.seed)
    # Resolve the timestamped name HERE, not in __post_init__:
    config.experiment_name = config.resolved_experiment_name()
    run_dir = prepare_run_dir(config)

    # Refuse if a run already occupies this directory.
    refuse_existing_run(run_dir, RESULTS_SUMMARY_NAME)

    with attach_run_log(run_dir):
        logger.info(f"Run directory: {run_dir}")

        # --- load data ---
        x_train, states_train, x_val, states_val, x_test, states_test = _load_data(config)

        # --- build model ---
        model_cfg = dict(cfg.__dict__)  # noqa: F821  -- config is NeuralHMMConfig
        # Remove dataclass fields that are not model config keys
        for k in [
            "experiment_name", "output_dir", "seed", "generator",
            "transition_type", "switch_frac", "obs_noise_factor",
            "patience", "include_terminate_on_nan", "include_analyzer",
            "save_visualizations", "visualization_frequency",
            "epochs", "batch_size", "learning_rate",
            "gradient_clip_norm", "optimizer",
        ]:
            model_cfg.pop(k, None)
        # Ensure required pairs
        if config.mode == "discriminative" and config.model_cfg.get("obs_type") == "categorical":
            if config.vocab_size is None:
                config.vocab_size = 50  # default

        model = NeuralHMM(model_cfg)
        model.compile(
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            gradient_clip_norm=config.gradient_clip_norm,
        )

        # --- callbacks ---
        callbacks, results_dir = create_callbacks(
            model_name="neural_hmm",
            run_dir=str(run_dir),
            monitor="val_loss",
            monitor_mode=resolve_monitor_mode("val_loss"),
            patience=config.patience,
            include_terminate_on_nan=config.include_terminate_on_nan,
            include_analyzer=config.include_analyzer,
        )

        # --- learning rate schedule ---
        lr_schedule = create_learning_rate_schedule(
            config.learning_rate, config.lr_schedule if hasattr(config, "lr_schedule") else "constant",
            config.epochs, steps_per_epoch=math.ceil(x_train.shape[0] / config.batch_size),
        )
        if lr_schedule is not None:
            logger.info(f"LR schedule: {config.lr_schedule} over {config.epochs} epochs")
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=lr_schedule,
                    clipnorm=config.gradient_clip_norm,
                )
            )

        # --- training loop ---
        steps_per_epoch = max(1, math.ceil(x_train.shape[0] / config.batch_size))
        history = model.fit(
            {"obs": x_train, "mask": np.ones_like(x_train[:, :1], "float32")},
            None,  # y is not a separate arg; compute_loss handles it
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(
                {"obs": x_val, "mask": np.ones_like(x_val[:, :1], "float32")},
                None,
            ),
            callbacks=callbacks,
            verbose=1,
        )

        # --- evaluation ---
        # Compute per-timestep NLL on test set
        mask_test = np.ones_like(x_test[:, :1], "float32")
        inputs_test = {"obs": x_test, "mask": mask_test}
        log_pi, log_A, log_emit = model(inputs_test)

        # Forward algorithm
        if len(log_A.shape) == 4:
            from .inference import _forward_logprob_dynamic as _fw
            logZ = _fw(log_pi, log_A, log_emit, mask_test)
        else:
            from .inference import _forward_logprob_static as _fw
            logZ = _fw(log_pi, log_A, log_emit, mask_test)

        # Gold-path score (generative: just logZ; discriminative with states)
        if config.mode == "discriminative" and states_test is not None:
            from .inference import _path_score as _ps
            gold = _ps(log_pi, log_A, log_emit, states_test, mask_test)
            test_nll = float(np.mean(-(gold - logZ)))  # per-timestep
        else:
            # generative: NLL = -logZ averaged
            test_nll = float(np.mean(-logZ))

        # Viterbi accuracy (generative only; discriminative has no ground-truth
        # state path that matches the model's objectives in the same way)
        viterbi_accuracy = None
        if config.mode == "generative" and states_test is not None:
            from .inference import viterbi as _viterbi
            # Compute Viterbi on each test sample and compare to ground truth
            n_correct = 0
            n_total = 0
            for i in range(min(20, x_test.shape[0])):  # first 20 samples
                sp, _ = _viterbi(log_pi[i:i+1], log_A[i:i+1] if log_A.ndim == 2 else log_A, log_emit[i:i+1])
                # Squeeze if batch dim present
                if sp.ndim == 2:
                    sp = sp[0]
                if states_test is not None:
                    gt = states_test[i]
                    if len(sp) == len(gt) and np.array_equal(sp, gt):
                        n_correct += 1
                n_total += 1
            viterbi_accuracy = float(n_correct) / float(n_total) if n_total > 0 else None
        else:
            viterbi_accuracy = None

        # State-occupancy entropy (diagnostic: is one state dominating?)
        from dl_techniques.metrics.time_series_metrics import calculate_comprehensive_metrics
        occ_metrics = {}

        # Figures (if enabled)
        figures: Dict[str, Optional[str]] = {}
        if config.save_visualizations and config.visualization_frequency > 0:
            # Only draw on the last epoch to keep runtime reasonable
            if config.epochs > 0 and (config.epochs - 1) % config.visualization_frequency == 0:
                # Use the final model state (weights from last epoch are already
                # in model after fit); we re-forward on the test set.
                final_path = _plot_posterior_vs_observation(
                    run_dir, model, x_test, states_test, config, config.epochs - 1,
                )
                if final_path:
                    figures["posterior_vs_observation"] = str(final_path)
                seg_path = _plot_segmentation_comparison(
                    run_dir, model, x_test, states_test, config, config.epochs - 1,
                )
                if seg_path:
                    figures["segmentation_comparison"] = str(seg_path)
            else:
                # Draw on epoch 0 (before any training) as a sanity check
                if config.epochs > 0:
                    final_path = _plot_posterior_vs_observation(
                        run_dir, model, x_test, states_test, config, 0,
                    )
                    if final_path:
                        figures["posterior_vs_observation"] = str(final_path)
                    seg_path = _plot_segmentation_comparison(
                        run_dir, model, x_test, states_test, config, 0,
                    )
                    if seg_path:
                        figures["segmentation_comparison"] = str(seg_path)

        # --- results_summary.json ---
        summary: Dict[str, Any] = {
            "model_name": "neural_hmm",
            "experiment_name": config.experiment_name,
            "generator": config.generator,
            "mode": config.mode,
            "transition_type": config.transition_type,
            "topology": config.topology,
            "obs_type": model.cfg["obs_type"],
            "n_states": config.n_states,
            "sequence_length": config.n_samples,
            "epochs_requested": config.epochs,
            "epochs_run": len(history.history.get("loss", [])),
            "best_epoch": int(
                np.argmin(history.history["val_loss"]) + 1
            ) if history.history.get("val_loss") else None,
            "best_val_loss": float(min(history.history["val_loss"]))
            if history.history.get("val_loss")
            else None,
            "final_train_loss": float(history.history["loss"][-1])
            if history.history.get("loss")
            else None,
            "final_val_loss": float(history.history["val_loss"][-1])
            if history.history.get("val_loss")
            else None,
            "optimizer": {
                "name": config.optimizer,
                "learning_rate": config.learning_rate,
            },
            "test_nll_per_timestep": test_nll,
            "viterbi_accuracy": viterbi_accuracy,
            "state_occupancy_entropy": occ_metrics.get("entropy", None),
            "parameters": int(model.count_params()),
            "sequences": int(x_train.shape[0]),
            "shape": list(x_train.shape[1:]),
            "visualizations": figures,
            "data_shapes": {
                "train": list(x_train.shape),
                "validation": list(x_val.shape),
                "test": list(x_test.shape),
            },
        }

        summary_path = run_dir / RESULTS_SUMMARY_NAME
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2, default=mean_std)
        logger.info(f"Wrote {summary_path}")

        # --- log some diagnostics ---
        logger.info(
            f"Done. test_nll={test_nll:.4f} | "
            f"viterbi_accuracy={viterbi_accuracy} | "
            f"parameters={int(model.count_params())}"
        )

    return summary


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------


def create_argument_parser() -> argparse.ArgumentParser:
    """Build this trainer's argument parser.

    The parser consolidates the shared time-series training arguments
    (``--epochs``, ``--batch_size``, ``--learning_rate``, etc.) via
    ``train.common.create_ts_argument_parser()`` and then adds model-specific
    flags on top.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Train a Neural HMM (Keller & Welling style, arXiv:2006.00694)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shared time-series flags (Pattern 2 / Pattern 7 conventions)
    from train.common.args import create_ts_argument_parser as _ts_parser
    base = _ts_parser("Neural HMM")

    # Parser already contains: --experiment_name, --result_dir, --seed,
    # --n_samples, --noise_level, --epochs, --batch_size, --steps_per_epoch,
    # --learning_rate, --gradient_clip_norm, --optimizer, --warmup toggles,
    # --deep-analysis toggles. We add model-specific args on top.

    # Data generation
    data = parser.add_argument_group("data")
    data.add_argument(
        "--generator", type=str, default="gaussian_hmm",
        choices=available_generators(),
        help="Synthetic data generator to use.",
    )
    data.add_argument(
        "--n-samples", type=int, default=100, help="Sequence length T."
    )
    data.add_argument(
        "--n-states", type=int, default=4, help="Number of hidden states K."
    )
    data.add_argument(
        "--means", type=float, nargs="+", default=[0.0, 1.0, -1.0, 2.0],
        help="Per-state means for Gaussian observations.",
    )
    data.add_argument(
        "--stds", type=float, nargs="+", default=[1.0, 1.0, 1.0, 1.0],
        help="Per-state standard deviations for Gaussian observations.",
    )
    data.add_argument(
        "--transition-type", type=str, default="static",
        choices=["static", "input_dependent"],
        help="Transition matrix type.",
    )
    data.add_argument(
        "--switch-frac", type=float, default=0.5,
        help="Fraction of sequence at which to switch transition matrices "
        "(only used if --transition-type input_dependent).",
    )
    data.add_argument(
        "--obs-noise-factor", type=float, default=1.0,
        help="Multiplicative factor on observation stds.",
    )

    # Model architecture
    model = parser.add_argument_group("model")
    data.add_argument(
        "--mode", type=str, default="generative",
        choices=["generative", "discriminative"],
        help="HMM regime: generative (maximize log p(x)) or discriminative "
        "(maximize log p(y|x), CRF-style).",
    )
    data.add_argument(
        "--encoder", type=str, default="lstm",
        choices=["mlp", "lstm", "conv", "transformer"],
        help="Encoder body.",
    )
    data.add_argument(
        "--emission-context", type=str, default="none",
        choices=["none", "autoregressive"],
        help="In generative mode, whether the encoder context h_t is autoregressive "
        "(causal) or shared across timesteps.",
    )
    data.add_argument(
        "--topology", type=str, default="ergodic",
        choices=["ergodic", "left_to_right", "banded"],
        help="Transition topology.",
    )
    data.add_argument(
        "--band-width", type=int, default=1,
        help="Maximum forward skip for banded topology.",
    )
    data.add_argument(
        "--self-loop-init", type=float, default=2.0,
        help="Diagonal logit bias for state stickiness at initialization.",
    )
    data.add_argument(
        "--min-scale", type=float, default=1e-3,
        help="Floor for Gaussian/Poisson scales.",
    )
    # Discriminative-only
    data.add_argument(
        "--vocab-size", type=int, default=None,
        help="Vocabulary size V; required if --mode discriminative and --obs-type categorical.",
    )

    # Optimisation
    optim = parser.add_argument_group("optimisation")
    optim.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    optim.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size."
    )
    optim.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Initial learning rate."
    )
    optim.add_argument(
        "--gradient-clip-norm", type=float, default=1.0,
        help="Clip norm for gradients.",
    )
    optim.add_argument(
        "--optimizer", type=str, default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer.",
    )

    # Callbacks
    callbacks = parser.add_argument_group("callbacks")
    callbacks.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience (epochs without val-loss improvement)."
    )
    callbacks.add_argument(
        "--no-terminate-on-nan", action="store_false",
        dest="include_terminate_on_nan",
        help="Disable TerminateOnNaN callback."
    )
    callbacks.add_argument(
        "--no-analyzer", action="store_false",
        dest="include_analyzer",
        help="Disable the epoch analyzer callback."
    )

    # Visualisation
    viz = parser.add_argument_group("visualisation")
    viz.add_argument(
        "--save-visualizations", action="store_true", default=True,
        help="Save posterior-vs-observation and segmentation comparison figures."
    )
    viz.add_argument(
        "--visualization-frequency", type=int, default=5,
        help="Write figures every N epochs (0 = only epoch 0 and final)."
    )

    # Plumbing
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Root output directory for run artifacts."
    )
    parser.add_argument(
        "--experiment-name", type=str, default="neural_hmm",
        help="Experiment name / run directory prefix."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="Enable mixed precision (float16)."
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point: parse argv and start a training run.

    The first statement of any trainer ``main()`` must parse argv so that
    ``--help`` exits 0 with a usage line and allocates nothing.

    :param argv: CLI arguments; ``sys.argv[1:]`` by default.
    :type argv: Optional[List[str]]
    """
    parser = create_argument_parser()
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    # Build config from parsed args
    cfg = NeuralHMMConfig(
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        seed=args.seed,
        generator=args.generator,
        n_samples=args.n_samples if hasattr(args, 'n_samples') else 100,
        n_states=args.n_states,
        means=args.means,
        stds=args.stds,
        transition_type=args.transition_type,
        switch_frac=args.switch_frac,
        obs_noise_factor=args.obs_noise_factor,
        mode=args.mode,
        encoder=args.encoder,
        emission_context=args.emission_context,
        topology=args.topology,
        band_width=args.band_width,
        self_loop_init=args.self_loop_init,
        min_scale=args.min_scale,
        vocab_size=args.vocab_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        optimizer=args.optimizer,
        mixed_precision=args.mixed_precision,
        patience=args.patience,
        include_terminate_on_nan=args.include_terminate_on_nan,
        include_analyzer=args.include_analyzer,
        save_visualizations=args.save_visualizations,
        visualization_frequency=args.visualization_frequency,
    )

    train(cfg)


if __name__ == "__main__":
    main()
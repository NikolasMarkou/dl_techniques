"""Training script for the Tiny Recursive Model (TRM).

Mirrors ``src/train/hrm/train_hrm.py``: a custom GradientTape ACT loop with
per-step apply_gradients (correct because TRM stops gradient on the inner
carry between steps), plus an HRM-style class-based trainer.

Supports two datasets:
- ``sample`` (default, no path required): synthetic uniform-random tokens
  for plumbing tests. Loss will not decrease meaningfully — the goal is to
  exercise the data → model → loss → optimizer → save/load pipeline.
- ``arc``: requires ``--arc-data-path``. Uses
  ``dl_techniques.datasets.arc.arc_keras.ARCSequence``; we wrap it to emit
  ``{"inputs", "labels"}`` int32 dicts (the underlying sequence returns
  ``(inputs, labels)`` tuples and normalizes inputs to floats — we undo the
  normalization for token-id consumption by TRM).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import keras
from keras import ops
import tensorflow as tf  # noqa: F401  # GradientTape (HRM precedent)

from dl_techniques.losses.hrm_loss import create_hrm_loss
from dl_techniques.metrics.hrm_metrics import HRMMetrics
from dl_techniques.models.tiny_recursive_model import TRM, create_trm
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
)
from dl_techniques.utils.logger import logger
from train.common import create_base_argument_parser, setup_gpu


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------


def create_sample_dataset(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    seed: Optional[int] = None,
) -> keras.utils.Sequence:
    """Synthetic uniform-random ``{"inputs", "labels"}`` dataset.

    Inputs and labels are both ``int32`` token ids in ``[2, vocab_size)``
    (matching the HRM sample dataset convention — 0 and 1 reserved for pad
    and EOS in the ARC tokenization).
    """

    class _SampleDataset(keras.utils.Sequence):
        def __init__(self) -> None:
            super().__init__()
            self._rng = np.random.default_rng(seed)
            self._num_batches = num_batches

        def __len__(self) -> int:
            return self._num_batches

        def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
            inputs = self._rng.integers(
                2, vocab_size, size=(batch_size, seq_len)
            ).astype(np.int32)
            labels = self._rng.integers(
                2, vocab_size, size=(batch_size, seq_len)
            ).astype(np.int32)
            return {"inputs": inputs, "labels": labels}

    return _SampleDataset()


def create_arc_dataset(
    arc_data_path: str,
    split: str,
    batch_size: int,
    vocab_size: int,
) -> keras.utils.Sequence:
    """Wrap ``ARCSequence`` to emit ``{"inputs", "labels"}`` int32 dicts."""
    from dl_techniques.datasets.arc.arc_keras import ARCSequence

    inner = ARCSequence(
        dataset_path=arc_data_path,
        split=split,
        subset="all",
        batch_size=batch_size,
        shuffle=(split == "train"),
        normalize_inputs=False,  # keep raw int token ids
    )

    class _ARCWrappedSequence(keras.utils.Sequence):
        def __init__(self) -> None:
            super().__init__()
            self._inner = inner

        def __len__(self) -> int:
            return len(self._inner)

        def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
            inputs, labels = self._inner[idx]
            return {
                "inputs": inputs.astype(np.int32),
                "labels": labels.astype(np.int32),
            }

        def on_epoch_end(self) -> None:
            self._inner.on_epoch_end()

    _ = vocab_size  # vocab_size is implied by the dataset metadata; trainer trusts user CLI
    return _ARCWrappedSequence()


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------


class TRMTrainer:
    """Trainer for the Tiny Recursive Model with ACT support.

    Mirrors the ``HRMTrainer`` shape but uses ``create_trm`` and the simpler
    TRM forward signature ``model(carry, batch, training=...)``.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        train_dataset: keras.utils.Sequence,
        val_dataset: Optional[keras.utils.Sequence] = None,
    ) -> None:
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model: TRM = self._create_model()

        self.loss_fn = create_hrm_loss(
            lm_loss_type=config.get("lm_loss_type", "stable_max"),
            q_loss_weight=config.get("q_loss_weight", 0.5),
            ignore_index=config.get("ignore_index", -100),
        )
        self.metrics = HRMMetrics(ignore_index=config.get("ignore_index", -100))

        self.optimizer, self.lr_schedule = self._create_optimizer()

        self.current_epoch = 0
        self.current_step = 0
        self.best_val_accuracy = 0.0

        logger.info(
            f"Initialized TRM Trainer with {self.model.count_params():,} parameters"
        )

    # -----------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------

    def _create_model(self) -> TRM:
        mc = self.config["model"]
        return create_trm(
            vocab_size=mc["vocab_size"],
            hidden_size=mc["hidden_size"],
            num_heads=mc["num_heads"],
            expansion=mc["expansion"],
            seq_len=mc["seq_len"],
            puzzle_emb_len=mc.get("puzzle_emb_len", 0),
            h_layers=mc.get("h_layers", 2),
            l_layers=mc.get("l_layers", 2),
            halt_max_steps=mc.get("halt_max_steps", 8),
            halt_exploration_prob=mc.get("halt_exploration_prob", 0.1),
            no_act_continue=mc.get("no_act_continue", True),
            rope_theta=mc.get("rope_theta", 10000.0),
            dropout_rate=mc.get("dropout_rate", 0.0),
            attention_dropout_rate=mc.get("attention_dropout_rate", 0.0),
        )

    def _create_optimizer(self):
        lr_cfg = self.config.get("learning_rate", {})
        lr_schedule = learning_rate_schedule_builder(
            {
                "type": lr_cfg.get("type", "cosine_decay"),
                "learning_rate": lr_cfg.get("initial_lr", 1e-4),
                "decay_steps": lr_cfg.get("decay_steps", 10000),
                "warmup_steps": lr_cfg.get("warmup_steps", 200),
                "warmup_start_lr": lr_cfg.get("warmup_start_lr", 1e-8),
                "alpha": lr_cfg.get("min_lr_ratio", 0.1),
            }
        )

        opt_cfg = self.config.get("optimizer", {})
        optimizer = optimizer_builder(
            {
                "type": opt_cfg.get("type", "adam"),
                "beta_1": opt_cfg.get("beta_1", 0.9),
                "beta_2": opt_cfg.get("beta_2", 0.95),
                "gradient_clipping_by_norm": opt_cfg.get("grad_clip_norm", 1.0),
            },
            lr_schedule,
        )
        return optimizer, lr_schedule

    # -----------------------------------------------------------------
    # Steps
    # -----------------------------------------------------------------

    @staticmethod
    def _prepare_batch(batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs = {"inputs": batch["inputs"]}
        targets = {"labels": batch["labels"]}
        return inputs, targets

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One TRM training step with per-step apply_gradients across the ACT unroll.

        Per-step apply is correct here because TRM stops gradient on the
        inner carry between steps (see ``TRMInner.call`` and the model
        docstring); each step's GradientTape is fully self-contained.
        """
        inputs, targets = self._prepare_batch(batch)

        carry = self.model.initial_carry(inputs)
        step_losses = []
        step_count = 0

        for _ in range(self.model.halt_max_steps):
            with tf.GradientTape() as tape:
                carry, outputs = self.model(carry, inputs, training=True)
                step_targets = dict(targets)
                step_targets.update({
                    "halted": carry["halted"],
                    "steps": carry["steps"],
                })
                step_loss = self.loss_fn(step_targets, outputs)
            grads = tape.gradient(step_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            self.metrics.update_state(step_targets, outputs)
            step_losses.append(float(step_loss))
            step_count += 1

            if bool(ops.all(carry["halted"])):
                break

        current_lr = float(self.optimizer.learning_rate)
        step_metrics = self.metrics.result()
        step_metrics.update({
            "loss": float(np.mean(step_losses)),
            "learning_rate": current_lr,
            "act_steps": float(step_count),
        })

        self.current_step += 1
        return step_metrics

    def evaluate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        inputs, targets = self._prepare_batch(batch)

        carry = self.model.initial_carry(inputs)
        step_losses = []
        step_count = 0

        for _ in range(self.model.halt_max_steps):
            carry, outputs = self.model(carry, inputs, training=False)
            step_targets = dict(targets)
            step_targets.update({
                "halted": carry["halted"],
                "steps": carry["steps"],
            })
            step_loss = self.loss_fn(step_targets, outputs)
            self.metrics.update_state(step_targets, outputs)
            step_losses.append(float(step_loss))
            step_count += 1
            if bool(ops.all(carry["halted"])):
                break

        eval_metrics = self.metrics.result()
        eval_metrics.update({
            "val_loss": float(np.mean(step_losses)),
            "val_act_steps": float(step_count),
        })
        return eval_metrics

    # -----------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        logger.info(f"Training epoch {self.current_epoch + 1}")
        self.metrics.reset_state()
        epoch_metrics: Dict[str, list] = {}

        for batch_idx in range(len(self.train_dataset)):
            batch = self.train_dataset[batch_idx]
            step_metrics = self.train_step(batch)
            for k, v in step_metrics.items():
                epoch_metrics.setdefault(k, []).append(v)
            if batch_idx % 100 == 0:
                logger.info(
                    f"Batch {batch_idx}: loss={step_metrics['loss']:.4f}, "
                    f"act_steps={step_metrics['act_steps']:.1f}"
                )

        try:
            self.train_dataset.on_epoch_end()
        except AttributeError:
            pass

        self.current_epoch += 1
        return {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

    def evaluate(self) -> Dict[str, float]:
        if self.val_dataset is None:
            return {}

        logger.info("Evaluating on validation set")
        self.metrics.reset_state()
        eval_metrics: Dict[str, list] = {}

        for batch_idx in range(len(self.val_dataset)):
            batch = self.val_dataset[batch_idx]
            step_metrics = self.evaluate_step(batch)
            for k, v in step_metrics.items():
                eval_metrics.setdefault(k, []).append(v)

        return {k: float(np.mean(v)) for k, v in eval_metrics.items()}

    def train(
        self,
        epochs: int,
        save_dir: str,
        eval_freq: int = 1,
        save_freq: int = 5,
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(save_dir, f"trm_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        logger.info(f"Starting training for {epochs} epochs at {run_dir}")

        for epoch in range(epochs):
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch + 1}/{epochs} - Train metrics:")
            for k, v in train_metrics.items():
                logger.info(f"  {k}: {v:.6f}")

            if (epoch + 1) % eval_freq == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Validation metrics:")
                    for k, v in eval_metrics.items():
                        logger.info(f"  {k}: {v:.6f}")
                    val_acc = eval_metrics.get("exact_accuracy", 0.0)
                    if val_acc > self.best_val_accuracy:
                        self.best_val_accuracy = val_acc
                        best_path = os.path.join(run_dir, "best_model.keras")
                        self.model.save(best_path)
                        logger.info(
                            f"Saved best model with validation exact_accuracy={val_acc:.4f}"
                        )

            if (epoch + 1) % save_freq == 0:
                ckpt = os.path.join(run_dir, f"checkpoint_epoch_{epoch + 1}.keras")
                self.model.save(ckpt)
                logger.info(f"Saved checkpoint: {ckpt}")

        final_path = os.path.join(run_dir, "final_model.keras")
        self.model.save(final_path)
        logger.info(f"Training complete. Final model saved: {final_path}")
        return run_dir


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    parser = create_base_argument_parser(
        description="Train Tiny Recursive Model (TRM)",
        default_dataset="sample",
        dataset_choices=["sample", "arc"],
    )

    # Model
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=900,
                        help="Sequence length (excluding puzzle_emb prefix)")
    parser.add_argument("--puzzle-emb-len", type=int, default=0,
                        help="Puzzle-embedding prefix length (zero-padded; B-11 residual)")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expansion", type=float, default=4.0)
    parser.add_argument("--h-layers", type=int, default=2)
    parser.add_argument("--l-layers", type=int, default=2)
    parser.add_argument("--halt-max-steps", type=int, default=8)
    parser.add_argument("--halt-exploration-prob", type=float, default=0.1)
    parser.add_argument("--no-act-continue", action="store_true", default=False,
                        help="Use simple halting (q_halt > 0). Default: Q-learning halting.")
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--attention-dropout-rate", type=float, default=0.0)

    # Loss / opt
    parser.add_argument("--lm-loss-type", type=str, default="stable_max",
                        choices=["stable_max", "sparse_categorical"])
    parser.add_argument("--q-loss-weight", type=float, default=0.5)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    # Data
    parser.add_argument("--num-train-batches", type=int, default=200)
    parser.add_argument("--num-val-batches", type=int, default=50)
    parser.add_argument("--arc-data-path", type=str, default=None,
                        help="Path to ARC-1 dataset directory (only used with --dataset arc)")

    # Output
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--save-freq", type=int, default=5)

    args = parser.parse_args()
    setup_gpu(gpu_id=args.gpu)

    # --- Build datasets ---
    if args.dataset == "arc":
        if args.arc_data_path is None or not os.path.isdir(args.arc_data_path):
            logger.warning(
                "--dataset arc requested but --arc-data-path is missing or "
                "invalid; falling back to synthetic 'sample' dataset."
            )
            args.dataset = "sample"

    if args.dataset == "sample":
        train_ds = create_sample_dataset(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_batches=args.num_train_batches,
            seed=0,
        )
        val_ds = create_sample_dataset(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_batches=args.num_val_batches,
            seed=1,
        )
    else:
        train_ds = create_arc_dataset(
            arc_data_path=args.arc_data_path,
            split="train",
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
        )
        val_ds = create_arc_dataset(
            arc_data_path=args.arc_data_path,
            split="test",
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
        )

    # --- Config dict (logged to config.json) ---
    config: Dict[str, Any] = {
        "dataset": args.dataset,
        "model": {
            "vocab_size": args.vocab_size,
            "seq_len": args.seq_len,
            "puzzle_emb_len": args.puzzle_emb_len,
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "expansion": args.expansion,
            "h_layers": args.h_layers,
            "l_layers": args.l_layers,
            "halt_max_steps": args.halt_max_steps,
            "halt_exploration_prob": args.halt_exploration_prob,
            "no_act_continue": bool(args.no_act_continue),
            "dropout_rate": args.dropout_rate,
            "attention_dropout_rate": args.attention_dropout_rate,
        },
        "learning_rate": {
            "type": "cosine_decay",
            "initial_lr": args.learning_rate,
            "decay_steps": max(args.epochs * args.num_train_batches, 1),
            "warmup_steps": min(200, max(args.num_train_batches // 5, 1)),
            "min_lr_ratio": 0.1,
        },
        "optimizer": {
            "type": "adam",
            "beta_1": 0.9,
            "beta_2": 0.95,
            "grad_clip_norm": args.grad_clip_norm,
        },
        "lm_loss_type": args.lm_loss_type,
        "q_loss_weight": args.q_loss_weight,
        "ignore_index": args.ignore_index,
    }

    trainer = TRMTrainer(config, train_ds, val_ds)
    trainer.train(
        epochs=args.epochs,
        save_dir=args.save_dir,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
    )


if __name__ == "__main__":
    main()

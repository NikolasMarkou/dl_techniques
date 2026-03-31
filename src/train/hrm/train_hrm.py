"""
Training script for Hierarchical Reasoning Model.

Uses a custom GradientTape training loop for ACT (Adaptive Computation Time).
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

import keras
from keras import optimizers

from dl_techniques.models.hierarchical_reasoning_model.model import (
    create_hierarchical_reasoning_model,
    HierarchicalReasoningModel
)
from dl_techniques.losses.hrm_loss import create_hrm_loss
from dl_techniques.metrics.hrm_metrics import HRMMetrics
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder
)
from dl_techniques.utils.logger import logger
from train.common import setup_gpu, create_base_argument_parser


class HRMTrainer:
    """Trainer for Hierarchical Reasoning Model with ACT support."""

    def __init__(
            self,
            config: Dict[str, Any],
            train_dataset: keras.utils.Sequence,
            val_dataset: Optional[keras.utils.Sequence] = None
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model = self._create_model()

        self.loss_fn = create_hrm_loss(
            lm_loss_type=config.get("lm_loss_type", "stable_max"),
            q_loss_weight=config.get("q_loss_weight", 0.5),
            ignore_index=config.get("ignore_index", -100)
        )

        self.metrics = HRMMetrics(
            ignore_index=config.get("ignore_index", -100)
        )

        self.optimizer, self.lr_schedule = self._create_optimizer()

        self.current_epoch = 0
        self.current_step = 0
        self.best_val_accuracy = 0.0

        logger.info(f"Initialized HRM Trainer with {self.model.count_params():,} parameters")

    def _create_model(self) -> HierarchicalReasoningModel:
        model_config = self.config.get("model", {})
        return create_hierarchical_reasoning_model(
            vocab_size=model_config["vocab_size"],
            seq_len=model_config["seq_len"],
            embed_dim=model_config.get("embed_dim", 512),
            num_puzzle_identifiers=model_config.get("num_puzzle_identifiers", 1000),
            puzzle_emb_dim=model_config.get("puzzle_emb_dim", 512),
            batch_size=model_config.get("batch_size", 32),
            h_layers=model_config.get("h_layers", 4),
            l_layers=model_config.get("l_layers", 4),
            h_cycles=model_config.get("h_cycles", 2),
            l_cycles=model_config.get("l_cycles", 2),
            num_heads=model_config.get("num_heads", 8),
            ffn_expansion_factor=model_config.get("ffn_expansion_factor", 4),
            pos_encodings=model_config.get("pos_encodings", "rope"),
            rope_theta=model_config.get("rope_theta", 10000.0),
            halt_max_steps=model_config.get("halt_max_steps", 16),
            halt_exploration_prob=model_config.get("halt_exploration_prob", 0.1),
            dropout_rate=model_config.get("dropout_rate", 0.0),
            use_bias=model_config.get("use_bias", False)
        )

    def _create_optimizer(self) -> Tuple[optimizers.Optimizer, Any]:
        lr_config = self.config.get("learning_rate", {})
        lr_schedule = learning_rate_schedule_builder({
            "type": lr_config.get("type", "cosine_decay"),
            "learning_rate": lr_config.get("initial_lr", 1e-4),
            "decay_steps": lr_config.get("decay_steps", 10000),
            "warmup_steps": lr_config.get("warmup_steps", 2000),
            "warmup_start_lr": lr_config.get("warmup_start_lr", 1e-8),
            "alpha": lr_config.get("min_lr_ratio", 0.1)
        })

        opt_config = self.config.get("optimizer", {})
        optimizer = optimizer_builder({
            "type": opt_config.get("type", "adam"),
            "beta_1": opt_config.get("beta_1", 0.9),
            "beta_2": opt_config.get("beta_2", 0.95),
            "gradient_clipping_by_norm": opt_config.get("grad_clip_norm", 1.0)
        }, lr_schedule)

        return optimizer, lr_schedule

    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs = {
            "token_ids": batch["inputs"],
            "puzzle_ids": batch["puzzle_identifiers"]
        }
        targets = {
            "labels": batch["labels"],
            "halted": batch.get("halted", None),
            "steps": batch.get("steps", None)
        }
        return inputs, targets

    @keras.utils.traceback_utils.filter_traceback
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step with ACT logic using GradientTape."""
        inputs, targets = self._prepare_batch(batch)

        with keras.utils.GradientTape() as tape:
            carry = self.model.initial_carry(inputs)
            total_loss = 0.0
            step_count = 0

            while step_count < self.model.halt_max_steps:
                carry, outputs, all_finished = self.model._forward_step(
                    carry, inputs, training=True
                )

                step_targets = dict(targets)
                step_targets.update({
                    "halted": carry["halted"],
                    "steps": carry["steps"]
                })

                step_loss = self.loss_fn(step_targets, outputs)
                total_loss += step_loss
                step_count += 1

                self.metrics.update_state(step_targets, outputs)

                if all_finished:
                    break

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        current_lr = float(self.optimizer.learning_rate)

        step_metrics = self.metrics.result()
        step_metrics.update({
            "loss": float(total_loss),
            "learning_rate": current_lr,
            "act_steps": float(step_count)
        })

        self.current_step += 1
        return step_metrics

    @keras.utils.traceback_utils.filter_traceback
    def evaluate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single evaluation step."""
        inputs, targets = self._prepare_batch(batch)

        carry = self.model.initial_carry(inputs)
        total_loss = 0.0
        step_count = 0

        while step_count < self.model.halt_max_steps:
            carry, outputs, all_finished = self.model._forward_step(
                carry, inputs, training=False
            )

            step_targets = dict(targets)
            step_targets.update({
                "halted": carry["halted"],
                "steps": carry["steps"]
            })

            step_loss = self.loss_fn(step_targets, outputs)
            total_loss += step_loss
            step_count += 1

            self.metrics.update_state(step_targets, outputs)

            if all_finished:
                break

        eval_metrics = self.metrics.result()
        eval_metrics.update({
            "val_loss": float(total_loss),
            "val_act_steps": float(step_count)
        })
        return eval_metrics

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        logger.info(f"Training epoch {self.current_epoch + 1}")
        self.metrics.reset_state()
        epoch_metrics = {}

        for batch_idx, batch in enumerate(self.train_dataset):
            step_metrics = self.train_step(batch)

            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

            if batch_idx % 100 == 0:
                logger.info(
                    f"Batch {batch_idx}: loss={step_metrics['loss']:.4f}, "
                    f"acc={step_metrics['accuracy']:.4f}, "
                    f"exact_acc={step_metrics['exact_accuracy']:.4f}"
                )

        epoch_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        self.current_epoch += 1
        return epoch_metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_dataset is None:
            return {}

        logger.info("Evaluating on validation set")
        self.metrics.reset_state()
        eval_metrics = {}

        for batch in self.val_dataset:
            step_metrics = self.evaluate_step(batch)
            for key, value in step_metrics.items():
                if key not in eval_metrics:
                    eval_metrics[key] = []
                eval_metrics[key].append(value)

        return {key: np.mean(values) for key, values in eval_metrics.items()}

    def train(
            self,
            epochs: int,
            save_dir: str,
            eval_freq: int = 1,
            save_freq: int = 5
    ):
        """Train the model for the specified number of epochs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, f"hrm_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            train_metrics = self.train_epoch()

            logger.info(f"Epoch {epoch + 1}/{epochs} - Train metrics:")
            for key, value in train_metrics.items():
                logger.info(f"  {key}: {value:.6f}")

            if (epoch + 1) % eval_freq == 0:
                eval_metrics = self.evaluate()

                if eval_metrics:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Validation metrics:")
                    for key, value in eval_metrics.items():
                        logger.info(f"  {key}: {value:.6f}")

                    val_accuracy = eval_metrics.get("exact_accuracy", 0.0)
                    if val_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = val_accuracy
                        best_model_path = os.path.join(save_dir, "best_model.keras")
                        self.model.save(best_model_path)
                        logger.info(f"Saved best model with validation accuracy: {val_accuracy:.4f}")

            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.keras")
                self.model.save(checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

        final_model_path = os.path.join(save_dir, "final_model.keras")
        self.model.save(final_model_path)
        logger.info(f"Training completed. Final model saved: {final_model_path}")


def create_sample_dataset(
        vocab_size: int = 100,
        seq_len: int = 50,
        batch_size: int = 32,
        num_batches: int = 100,
        num_puzzle_ids: int = 10
) -> keras.utils.Sequence:
    """Create a sample dataset for testing HRM training."""

    class SampleDataset(keras.utils.Sequence):
        def __init__(self):
            self.num_batches = num_batches

        def __len__(self):
            return self.num_batches

        def __getitem__(self, idx):
            return {
                "inputs": np.random.randint(2, vocab_size, size=(batch_size, seq_len)),
                "labels": np.random.randint(2, vocab_size, size=(batch_size, seq_len)),
                "puzzle_identifiers": np.random.randint(0, num_puzzle_ids, size=(batch_size,))
            }

    return SampleDataset()


def main():
    parser = create_base_argument_parser(
        description="Train Hierarchical Reasoning Model",
        default_dataset="sample",
        dataset_choices=["sample"],
    )
    parser.add_argument('--vocab-size', type=int, default=100, help='Vocabulary size')
    parser.add_argument('--seq-len', type=int, default=50, help='Sequence length')
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--h-layers', type=int, default=2, help='Number of high-level layers')
    parser.add_argument('--l-layers', type=int, default=2, help='Number of low-level layers')
    parser.add_argument('--h-cycles', type=int, default=2, help='Number of high-level cycles')
    parser.add_argument('--l-cycles', type=int, default=2, help='Number of low-level cycles')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--halt-max-steps', type=int, default=8, help='Maximum ACT steps')
    parser.add_argument('--num-puzzle-ids', type=int, default=10, help='Number of puzzle identifiers')
    parser.add_argument('--num-train-batches', type=int, default=200, help='Number of training batches')
    parser.add_argument('--num-val-batches', type=int, default=50, help='Number of validation batches')
    parser.add_argument('--save-dir', type=str, default='results', help='Checkpoint save directory')
    parser.add_argument('--eval-freq', type=int, default=2, help='Evaluation frequency (epochs)')
    parser.add_argument('--save-freq', type=int, default=5, help='Checkpoint save frequency (epochs)')
    parser.add_argument('--lm-loss-type', type=str, default='stable_max', help='Language model loss type')
    parser.add_argument('--q-loss-weight', type=float, default=0.5, help='Q-loss weight')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0, help='Gradient clipping norm')
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    config = {
        "model": {
            "vocab_size": args.vocab_size,
            "seq_len": args.seq_len,
            "embed_dim": args.embed_dim,
            "num_puzzle_identifiers": args.num_puzzle_ids,
            "puzzle_emb_dim": args.embed_dim,
            "batch_size": args.batch_size,
            "h_layers": args.h_layers,
            "l_layers": args.l_layers,
            "h_cycles": args.h_cycles,
            "l_cycles": args.l_cycles,
            "num_heads": args.num_heads,
            "ffn_expansion_factor": 4,
            "pos_encodings": "rope",
            "halt_max_steps": args.halt_max_steps,
            "halt_exploration_prob": 0.1,
            "dropout_rate": 0.0,
            "use_bias": False
        },
        "learning_rate": {
            "type": "cosine_decay",
            "initial_lr": args.learning_rate,
            "decay_steps": 5000,
            "warmup_steps": 1000,
            "min_lr_ratio": 0.1
        },
        "optimizer": {
            "type": "adam",
            "beta_1": 0.9,
            "beta_2": 0.95,
            "grad_clip_norm": args.grad_clip_norm
        },
        "lm_loss_type": args.lm_loss_type,
        "q_loss_weight": args.q_loss_weight,
        "ignore_index": -100
    }

    train_dataset = create_sample_dataset(
        vocab_size=config["model"]["vocab_size"],
        seq_len=config["model"]["seq_len"],
        batch_size=config["model"]["batch_size"],
        num_batches=args.num_train_batches,
        num_puzzle_ids=config["model"]["num_puzzle_identifiers"]
    )

    val_dataset = create_sample_dataset(
        vocab_size=config["model"]["vocab_size"],
        seq_len=config["model"]["seq_len"],
        batch_size=config["model"]["batch_size"],
        num_batches=args.num_val_batches,
        num_puzzle_ids=config["model"]["num_puzzle_identifiers"]
    )

    trainer = HRMTrainer(config, train_dataset, val_dataset)

    trainer.train(
        epochs=args.epochs,
        save_dir=args.save_dir,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq
    )


if __name__ == "__main__":
    main()

"""
Training script for Hierarchical Reasoning Model.
"""
import os
import json
import numpy as np
from typing import Dict, Tuple, Optional, Any


import keras
from keras import optimizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.hrm import (
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

# ---------------------------------------------------------------------

class HRMTrainer:
    """
    Trainer class for Hierarchical Reasoning Model.

    Handles model creation, training, evaluation, and checkpointing
    with support for ACT (Adaptive Computation Time) training.
    """

    def __init__(
            self,
            config: Dict[str, Any],
            train_dataset: keras.utils.Sequence,
            val_dataset: Optional[keras.utils.Sequence] = None
    ):
        """
        Initialize HRM trainer.

        Args:
            config: Training configuration dictionary
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
        """
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Initialize model
        self.model = self._create_model()

        # Initialize loss and metrics
        self.loss_fn = create_hrm_loss(
            lm_loss_type=config.get("lm_loss_type", "stable_max"),
            q_loss_weight=config.get("q_loss_weight", 0.5),
            ignore_index=config.get("ignore_index", -100)
        )

        self.metrics = HRMMetrics(
            ignore_index=config.get("ignore_index", -100)
        )

        # Initialize optimizer and scheduler
        self.optimizer, self.lr_schedule = self._create_optimizer()

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_accuracy = 0.0

        logger.info(f"Initialized HRM Trainer with {self.model.count_params():,} parameters")

    def _create_model(self) -> HierarchicalReasoningModel:
        """Create HRM model from configuration."""
        model_config = self.config.get("model", {})

        model = create_hierarchical_reasoning_model(
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

        return model

    def _create_optimizer(self) -> Tuple[optimizers.Optimizer, Any]:
        """Create optimizer and learning rate schedule."""
        # Learning rate schedule
        lr_config = self.config.get("learning_rate", {})
        lr_schedule = learning_rate_schedule_builder({
            "type": lr_config.get("type", "cosine_decay"),
            "learning_rate": lr_config.get("initial_lr", 1e-4),
            "decay_steps": lr_config.get("decay_steps", 10000),
            "warmup_steps": lr_config.get("warmup_steps", 2000),
            "warmup_start_lr": lr_config.get("warmup_start_lr", 1e-8),
            "alpha": lr_config.get("min_lr_ratio", 0.1)
        })

        # Optimizer
        opt_config = self.config.get("optimizer", {})
        optimizer = optimizer_builder({
            "type": opt_config.get("type", "adam"),
            "beta_1": opt_config.get("beta_1", 0.9),
            "beta_2": opt_config.get("beta_2", 0.95),
            "gradient_clipping_by_norm": opt_config.get("grad_clip_norm", 1.0)
        }, lr_schedule)

        return optimizer, lr_schedule

    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepare batch for training/evaluation.

        Args:
            batch: Raw batch from dataset

        Returns:
            Tuple of (inputs, targets) for model
        """
        # Inputs for model
        inputs = {
            "token_ids": batch["inputs"],
            "puzzle_ids": batch["puzzle_identifiers"]
        }

        # Targets for loss computation
        targets = {
            "labels": batch["labels"],
            "halted": batch.get("halted", None),
            "steps": batch.get("steps", None)
        }

        return inputs, targets

    @keras.utils.traceback_utils.filter_traceback
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step with ACT logic.

        Args:
            batch: Training batch

        Returns:
            Dictionary of loss and metric values
        """
        inputs, targets = self._prepare_batch(batch)

        with keras.utils.GradientTape() as tape:
            # Initialize carry state
            carry = self.model.initial_carry(inputs)

            total_loss = 0.0
            step_count = 0

            # Run ACT steps until all sequences halt
            while step_count < self.model.halt_max_steps:
                carry, outputs, all_finished = self.model._forward_step(
                    carry, inputs, training=True
                )

                # Compute loss for this step
                step_targets = dict(targets)
                step_targets.update({
                    "halted": carry["halted"],
                    "steps": carry["steps"]
                })

                step_loss = self.loss_fn(step_targets, outputs)
                total_loss += step_loss
                step_count += 1

                # Update metrics for halted sequences
                self.metrics.update_state(step_targets, outputs)

                if all_finished:
                    break

        # Compute gradients and apply optimizer
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Get current learning rate
        current_lr = float(self.optimizer.learning_rate)

        # Return step metrics
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
        """
        Single evaluation step.

        Args:
            batch: Evaluation batch

        Returns:
            Dictionary of loss and metric values
        """
        inputs, targets = self._prepare_batch(batch)

        # Run inference until convergence
        carry = self.model.initial_carry(inputs)

        total_loss = 0.0
        step_count = 0

        while step_count < self.model.halt_max_steps:
            carry, outputs, all_finished = self.model._forward_step(
                carry, inputs, training=False
            )

            # Compute loss
            step_targets = dict(targets)
            step_targets.update({
                "halted": carry["halted"],
                "steps": carry["steps"]
            })

            step_loss = self.loss_fn(step_targets, outputs)
            total_loss += step_loss
            step_count += 1

            # Update metrics
            self.metrics.update_state(step_targets, outputs)

            if all_finished:
                break

        # Return evaluation metrics
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

        # Training loop
        for batch_idx, batch in enumerate(self.train_dataset):
            step_metrics = self.train_step(batch)

            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

            # Log progress
            if batch_idx % 100 == 0:
                logger.info(
                    f"Batch {batch_idx}: loss={step_metrics['loss']:.4f}, "
                    f"acc={step_metrics['accuracy']:.4f}, "
                    f"exact_acc={step_metrics['exact_accuracy']:.4f}"
                )

        # Average metrics over epoch
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

            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in eval_metrics:
                    eval_metrics[key] = []
                eval_metrics[key].append(value)

        # Average metrics
        eval_metrics = {key: np.mean(values) for key, values in eval_metrics.items()}

        return eval_metrics

    def train(
            self,
            epochs: int,
            save_dir: str,
            eval_freq: int = 1,
            save_freq: int = 5
    ):
        """
        Train the model.

        Args:
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            eval_freq: Frequency of evaluation (in epochs)
            save_freq: Frequency of checkpoint saving (in epochs)
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Starting training for {epochs} epochs")

        # Training loop
        for epoch in range(epochs):
            # Train epoch
            train_metrics = self.train_epoch()

            # Log training metrics
            logger.info(f"Epoch {epoch + 1}/{epochs} - Train metrics:")
            for key, value in train_metrics.items():
                logger.info(f"  {key}: {value:.6f}")

            # Evaluate
            if (epoch + 1) % eval_freq == 0:
                eval_metrics = self.evaluate()

                if eval_metrics:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Validation metrics:")
                    for key, value in eval_metrics.items():
                        logger.info(f"  {key}: {value:.6f}")

                    # Check for best model
                    val_accuracy = eval_metrics.get("exact_accuracy", 0.0)
                    if val_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = val_accuracy
                        best_model_path = os.path.join(save_dir, "best_model.keras")
                        self.model.save(best_model_path)
                        logger.info(f"Saved best model with validation accuracy: {val_accuracy:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.keras")
                self.model.save(checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save final model
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
    """
    Create a sample dataset for testing HRM training.

    Args:
        vocab_size: Vocabulary size
        seq_len: Sequence length
        batch_size: Batch size
        num_batches: Number of batches
        num_puzzle_ids: Number of puzzle identifiers

    Returns:
        Sample dataset
    """

    class SampleDataset(keras.utils.Sequence):
        def __init__(self):
            self.num_batches = num_batches

        def __len__(self):
            return self.num_batches

        def __getitem__(self, idx):
            # Generate random batch
            inputs = np.random.randint(2, vocab_size, size=(batch_size, seq_len))
            labels = np.random.randint(2, vocab_size, size=(batch_size, seq_len))
            puzzle_ids = np.random.randint(0, num_puzzle_ids, size=(batch_size,))

            return {
                "inputs": inputs,
                "labels": labels,
                "puzzle_identifiers": puzzle_ids
            }

    return SampleDataset()


def main():
    """Main training function."""
    # Configuration
    config = {
        "model": {
            "vocab_size": 100,
            "seq_len": 50,
            "embed_dim": 256,
            "num_puzzle_identifiers": 10,
            "puzzle_emb_dim": 256,
            "batch_size": 16,
            "h_layers": 2,
            "l_layers": 2,
            "h_cycles": 2,
            "l_cycles": 2,
            "num_heads": 4,
            "ffn_expansion_factor": 4,
            "pos_encodings": "rope",
            "halt_max_steps": 8,
            "halt_exploration_prob": 0.1,
            "dropout_rate": 0.0,
            "use_bias": False
        },
        "learning_rate": {
            "type": "cosine_decay",
            "initial_lr": 1e-4,
            "decay_steps": 5000,
            "warmup_steps": 1000,
            "min_lr_ratio": 0.1
        },
        "optimizer": {
            "type": "adam",
            "beta_1": 0.9,
            "beta_2": 0.95,
            "grad_clip_norm": 1.0
        },
        "lm_loss_type": "stable_max",
        "q_loss_weight": 0.5,
        "ignore_index": -100
    }

    # Create datasets
    train_dataset = create_sample_dataset(
        vocab_size=config["model"]["vocab_size"],
        seq_len=config["model"]["seq_len"],
        batch_size=config["model"]["batch_size"],
        num_batches=200,
        num_puzzle_ids=config["model"]["num_puzzle_identifiers"]
    )

    val_dataset = create_sample_dataset(
        vocab_size=config["model"]["vocab_size"],
        seq_len=config["model"]["seq_len"],
        batch_size=config["model"]["batch_size"],
        num_batches=50,
        num_puzzle_ids=config["model"]["num_puzzle_identifiers"]
    )

    # Create trainer
    trainer = HRMTrainer(config, train_dataset, val_dataset)

    # Train model
    trainer.train(
        epochs=10,
        save_dir="hrm_checkpoints",
        eval_freq=2,
        save_freq=5
    )

# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
"""Multi-Task NTM Training Script.

A complete training script for training a single Neural Turing Machine (NTM)
to perform multiple algorithmic tasks simultaneously using task conditioning.

This script follows the framework structure:
1. Configuration via Dataclasses
2. Unified Data Generation Pipeline
3. Model Construction (NTMMultiTask)
4. Optimization and Callbacks
5. Training and Evaluation Orchestration

The model learns to switch strategies between tasks like Copy, Sorting,
Associative Recall, etc., based on a task-specific control input.
"""

import os
import keras
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ntm.ntm_interface import NTMConfig
from dl_techniques.models.ntm.model_multitask import NTMMultiTask

from .data_generators import (
    CopyTaskGenerator,
    CopyTaskConfig,
    AssociativeRecallGenerator,
    AssociativeRecallConfig,
    RepeatCopyGenerator,
    PriorityAccessGenerator,
    DynamicNGramGenerator,
    AlgorithmicTaskGenerator,
    AlgorithmicTaskConfig,
    TaskData
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class MultitaskNTMConfig:
    """Configuration for Multi-Task NTM Training.

    :param memory_size: Number of memory slots.
    :param memory_dim: Dimensionality of each memory slot.
    :param controller_dim: Hidden dimension of the controller (LSTM).
    :param num_heads: Number of read/write heads (symmetric).
    :param max_seq_length: Maximum sequence length for padding.
    :param max_vector_size: Maximum vector dimension for padding.
    :param batch_size: Training batch size.
    :param num_epochs: Number of training epochs.
    :param steps_per_epoch: Number of batches per epoch.
    :param validation_steps: Number of validation batches.
    :param learning_rate: Initial learning rate.
    :param clip_norm: Gradient clipping norm.
    :param save_dir: Root directory for outputs.
    :param task_map: Dictionary mapping task names to one-hot indices.
    """

    # Model Architecture
    memory_size: int = 128
    memory_dim: int = 20
    controller_dim: int = 256
    controller_type: str = "lstm"
    num_read_heads: int = 1
    num_write_heads: int = 1
    shift_range: int = 3
    epsilon: float = 1e-6

    # Data Dimensions (Normalization)
    max_seq_length: int = 100
    max_vector_size: int = 16

    # Training Hyperparameters
    batch_size: int = 64
    num_epochs: int = 100
    steps_per_epoch: int = 1000
    validation_steps: int = 100
    learning_rate: float = 1e-4
    clip_norm: float = 1.0

    # Paths
    save_dir: str = "results/multitask_ntm"
    checkpoint_dir: str = "results/multitask_ntm/checkpoints"
    log_dir: str = "results/multitask_ntm/logs"

    # Evaluation
    num_eval_samples: int = 1000

    # Task Registry
    task_map: Dict[str, int] = field(default_factory=lambda: {
        "copy": 0,
        "associative_recall": 1,
        "repeat_copy": 2,
        "priority_access": 3,
        "dynamic_ngram": 4,
        "insertion_sort": 5
    })

    @property
    def num_tasks(self) -> int:
        """Return total number of tasks."""
        return len(self.task_map)


# ---------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------

class UnifiedTaskGenerator(keras.utils.Sequence):
    """Unified Keras Sequence for multi-task data generation.

    Interleaves batches from different tasks and normalizes dimensions.
    """

    def __init__(
            self,
            config: MultitaskNTMConfig,
            mode: str = 'train'
    ) -> None:
        """Initialize the generator.

        :param config: Training configuration object.
        :param mode: 'train' or 'val' (affects random seeding).
        """
        self.config = config
        self.mode = mode
        self.batch_size = config.batch_size
        self.steps = config.steps_per_epoch if mode == 'train' else config.validation_steps

        # Seed logic: 'val' gets fixed seed for consistency
        seed = 42 if mode == 'val' else None
        self.rng = np.random.default_rng(seed)

        # Initialize Sub-Generators
        self.generators = {
            "copy": CopyTaskGenerator(CopyTaskConfig(
                sequence_length=10, vector_size=8, random_seed=seed
            )),
            "associative_recall": AssociativeRecallGenerator(AssociativeRecallConfig(
                num_items=6, key_size=8, value_size=8, random_seed=seed
            )),
            "repeat_copy": RepeatCopyGenerator(
                vector_size=8, random_seed=seed
            ),
            "priority_access": PriorityAccessGenerator(
                vector_size=8, random_seed=seed
            ),
            "dynamic_ngram": DynamicNGramGenerator(
                vocab_size=8, random_seed=seed
            ),
            "insertion_sort": AlgorithmicTaskGenerator(AlgorithmicTaskConfig(
                task_name="insertion_sort",
                train_size=16,
                test_size=16,
                random_seed=seed
            ))
        }

    def __len__(self) -> int:
        return self.steps

    def _generate_raw_data(self, task_key: str) -> TaskData:
        """Generate raw data for a specific task using randomized parameters."""
        gen = self.generators[task_key]

        if task_key == "copy":
            seq_len = self.rng.integers(5, 20)
            return gen.generate(self.batch_size, sequence_length=seq_len)
        elif task_key == "associative_recall":
            items = self.rng.integers(2, 6)
            return gen.generate(self.batch_size, num_items=items)
        elif task_key == "repeat_copy":
            seq_len = self.rng.integers(3, 8)
            repeats = self.rng.integers(2, 4)
            return gen.generate(self.batch_size, sequence_length=seq_len, num_repeats=repeats)
        elif task_key == "priority_access":
            num_items = self.rng.integers(3, 10)
            return gen.generate(self.batch_size, num_items=num_items)
        elif task_key == "dynamic_ngram":
            seq_len = self.rng.integers(10, 30)
            data = gen.generate(self.batch_size, sequence_length=seq_len)
            # Apply N-Gram mask (ignore context setup)
            if data.masks is None:
                mask = np.ones((self.batch_size, seq_len))
                mask[:, :2] = 0.0
                data.masks = mask
            return data
        elif task_key == "insertion_sort":
            size = self.rng.integers(5, 16)
            return gen.generate(self.batch_size, problem_size=size)
        else:
            raise ValueError(f"Unknown task key: {task_key}")

    def _pad_and_normalize(
            self,
            data: TaskData,
            task_key: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize dimensions to config max values."""
        inputs_raw = data.inputs
        targets_raw = data.targets
        masks_raw = data.masks

        curr_batch, curr_seq, _ = inputs_raw.shape
        target_dim = targets_raw.shape[-1]

        # --- Task Specific Adjustments ---
        if task_key == "associative_recall":
            # Reposition target to end of sequence
            seq_targets = np.zeros(
                (curr_batch, curr_seq, target_dim), dtype=np.float32
            )
            seq_targets[:, -1, :] = targets_raw
            targets_raw = seq_targets
            masks_raw = np.zeros((curr_batch, curr_seq), dtype=np.float32)
            masks_raw[:, -1] = 1.0

        if task_key == "insertion_sort" and masks_raw is None:
            masks_raw = np.ones((curr_batch, curr_seq), dtype=np.float32)

        # --- Generic Padding ---
        max_seq = self.config.max_seq_length
        max_vec = self.config.max_vector_size

        inputs_padded = np.zeros(
            (curr_batch, max_seq, max_vec), dtype=np.float32
        )
        targets_padded = np.zeros(
            (curr_batch, max_seq, max_vec), dtype=np.float32
        )
        masks_padded = np.zeros(
            (curr_batch, max_seq), dtype=np.float32
        )

        limit_seq = min(curr_seq, max_seq)
        limit_in_dim = min(inputs_raw.shape[-1], max_vec)
        limit_out_dim = min(target_dim, max_vec)

        inputs_padded[:, :limit_seq, :limit_in_dim] = \
            inputs_raw[:, :limit_seq, :limit_in_dim]
        targets_padded[:, :limit_seq, :limit_out_dim] = \
            targets_raw[:, :limit_seq, :limit_out_dim]

        if masks_raw is not None:
            masks_padded[:, :limit_seq] = masks_raw[:, :limit_seq]
        else:
            masks_padded[:, :limit_seq] = 1.0

        return inputs_padded, targets_padded, masks_padded

    def __getitem__(
            self,
            index: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Get a batch of data."""
        # 1. Select Task
        task_key = self.rng.choice(list(self.config.task_map.keys()))
        task_id = self.config.task_map[task_key]

        # 2. Generate Raw Data
        data = self._generate_raw_data(task_key)

        # 3. Normalize
        inputs, targets, masks = self._pad_and_normalize(data, task_key)

        # 4. Create Task Condition Vector
        task_one_hot = np.zeros(
            (self.batch_size, self.config.num_tasks), dtype=np.float32
        )
        task_one_hot[:, task_id] = 1.0

        # 5. Return dict for functional model inputs
        inputs_dict = {
            "sequence_in": inputs,
            "task_id_in": task_one_hot
        }

        return inputs_dict, targets, masks


def create_generators(
        config: MultitaskNTMConfig
) -> Tuple[UnifiedTaskGenerator, UnifiedTaskGenerator]:
    """Create train and validation generators.

    :param config: Training configuration.
    :return: Tuple of (train_generator, val_generator).
    """
    logger.info("Initializing Unified Task Generators...")
    train_gen = UnifiedTaskGenerator(config, mode='train')
    val_gen = UnifiedTaskGenerator(config, mode='val')
    return train_gen, val_gen


# ---------------------------------------------------------------------
# Model Creation
# ---------------------------------------------------------------------

def create_multitask_ntm_model(
        config: MultitaskNTMConfig
) -> keras.Model:
    """Create the Multi-Task NTM Keras model.

    :param config: Training configuration.
    :return: Compiled Keras model.
    """
    logger.info("Constructing Multi-Task NTM Model...")

    # 1. Prepare Interface Config
    ntm_config = NTMConfig(
        memory_size=config.memory_size,
        memory_dim=config.memory_dim,
        controller_dim=config.controller_dim,
        controller_type=config.controller_type,
        num_read_heads=config.num_read_heads,
        num_write_heads=config.num_write_heads,
        shift_range=config.shift_range,
        epsilon=config.epsilon
    )

    # 2. Define Inputs
    seq_input = keras.Input(
        shape=(config.max_seq_length, config.max_vector_size),
        name="sequence_in"
    )
    task_input = keras.Input(
        shape=(config.num_tasks,),
        name="task_id_in"
    )

    # 3. Instantiate Architecture Wrapper
    # This wrapper handles the concatenation of task IDs to inputs
    # and manages the internal NTM state.
    ntm_model = NTMMultiTask(
        ntm_config=ntm_config,
        output_dim=config.max_vector_size,
        num_tasks=config.num_tasks
    )

    # 4. Forward Pass
    outputs = ntm_model([seq_input, task_input])

    # 5. Output Activation
    # Sigmoid is generally safe for binary copy tasks and normalized sorting.
    final_output = keras.layers.Activation(
        "sigmoid", name="output_sigmoid"
    )(outputs)

    # 6. Build Model
    model = keras.Model(
        inputs=[seq_input, task_input],
        outputs=final_output,
        name="multitask_ntm"
    )

    num_params = model.count_params()
    logger.info(f"Model built with {num_params:,} parameters.")

    return model


# ---------------------------------------------------------------------
# Optimization and Compilation
# ---------------------------------------------------------------------

def compile_model(model: keras.Model, config: MultitaskNTMConfig) -> None:
    """Compile the model with optimizer and loss functions.

    :param model: The Keras model to compile.
    :param config: Training configuration.
    """
    logger.info("Compiling model...")

    optimizer = keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        clipnorm=config.clip_norm
    )

    # Use BinaryCrossentropy as the primary loss for bit-based tasks.
    # Monitor MSE for tasks like sorting.
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.MeanSquaredError(name="mse")
        ]
    )


def create_callbacks(
        config: MultitaskNTMConfig
) -> List[keras.callbacks.Callback]:
    """Create training callbacks.

    :param config: Training configuration.
    :return: List of callbacks.
    """
    logger.info("Creating training callbacks...")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.checkpoint_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=config.log_dir,
            histogram_freq=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(config.save_dir, "training_log.csv"),
            append=True
        )
    ]
    return callbacks


# ---------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------

def train_multitask_ntm(
        config: MultitaskNTMConfig
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Execute the full training pipeline.

    :param config: Training configuration.
    :return: Tuple of (trained_model, history).
    """
    logger.info("=" * 80)
    logger.info("Multi-Task Neural Turing Machine Training")
    logger.info("=" * 80)

    # 1. Create Generators
    train_gen, val_gen = create_generators(config)

    # 2. Build Model
    model = create_multitask_ntm_model(config)

    # 3. Compile
    compile_model(model, config)

    # 4. Callbacks
    callbacks = create_callbacks(config)

    # 5. Train
    logger.info("Starting training loop...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.num_epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Save Final Model
    final_path = os.path.join(config.save_dir, "multitask_ntm_final.keras")
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")

    return model, history


def evaluate_tasks(
        model: keras.Model,
        config: MultitaskNTMConfig
) -> None:
    """Evaluate the model on all tasks and report metrics.

    :param model: Trained model.
    :param config: Training configuration.
    """
    logger.info("=" * 60)
    logger.info(f"FINAL EVALUATION (N={config.num_eval_samples} samples/task)")
    logger.info("=" * 60)

    # Instantiate a generator in validation mode specifically for eval
    eval_gen = UnifiedTaskGenerator(config, mode='val')

    results = {}

    for task_name, task_id in config.task_map.items():
        # 1. Generate Raw Data
        raw_data = eval_gen._generate_raw_data(task_name)

        # 2. Normalize inputs manually to match model expectations
        # (UnifiedTaskGenerator internal method reused here for consistency)
        inputs, targets, masks = eval_gen._pad_and_normalize(raw_data, task_name)

        # 3. Prepare Task Vector
        task_one_hot = np.zeros(
            (config.batch_size, config.num_tasks), dtype=np.float32
        )
        task_one_hot[:, task_id] = 1.0

        # 4. Predict
        preds = model.predict(
            {"sequence_in": inputs, "task_id_in": task_one_hot},
            verbose=0
        )

        # 5. Compute Metrics
        mask_bool = masks.astype(bool)
        preds_flat = preds[mask_bool]
        targets_flat = targets[mask_bool]

        if task_name == "insertion_sort":
            mse = np.mean((preds_flat - targets_flat) ** 2)
            logger.info(f"Task: {task_name:<20} | MSE: {mse:.6f}")
            results[task_name] = mse
        else:
            preds_binary = (preds_flat > 0.5).astype(float)
            accuracy = np.mean(preds_binary == targets_flat)
            error_rate = 1.0 - accuracy
            logger.info(
                f"Task: {task_name:<20} | "
                f"Acc: {accuracy:.4f} | Error: {error_rate:.4f}"
            )
            results[task_name] = accuracy

    logger.info("=" * 60)


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------

def main() -> None:
    """Main entry point."""
    # 1. Setup Configuration
    config = MultitaskNTMConfig()

    logger.info("Training Configuration:")
    logger.info(f"  - Tasks: {list(config.task_map.keys())}")
    logger.info(f"  - Memory: {config.memory_size} slots x {config.memory_dim} dim")
    logger.info(f"  - Max Seq Length: {config.max_seq_length}")
    logger.info(f"  - Batch Size: {config.batch_size}")
    logger.info(f"  - Epochs: {config.num_epochs}")

    try:
        # 2. Train
        model, history = train_multitask_ntm(config)

        # 3. Evaluate
        evaluate_tasks(model, config)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()
"""
Multi-Task Neural Turing Machine (NTM) Training Script.

This script trains a single Neural Turing Machine to perform multiple
algorithmic tasks simultaneously using a task-conditioning mechanism.

It utilizes a unified data generator that interleaves batches from different
tasks (Copy, Sorting, Associative Recall, etc.), normalizing them to a fixed
sequence length and vector dimension. The model receives a task-specific
one-hot vector concatenated to the input sequence, allowing the controller
to adapt its read/write strategies per task.

Key Features:
    1. Centralized configuration for model, data, and training.
    2. Unified data pipeline handling heterogeneous task shapes.
    3. Task-conditioned inputs via broadcasting and concatenation.
    4. Comprehensive post-training evaluation metrics per task.
"""

import os
import numpy as np
import keras
from datetime import datetime
from typing import Tuple, Dict, Any

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------
from dl_techniques.utils.logger import logger
from dl_techniques.layers.ntm.ntm_interface import NTMConfig
from dl_techniques.models.ntm.model_multitask import NTMMultiTask

from .data_generators import (
    CopyTaskGenerator,
    AssociativeRecallGenerator,
    RepeatCopyGenerator,
    PriorityAccessGenerator,
    DynamicNGramGenerator,
    AlgorithmicTaskGenerator,
    TaskData
)
from .config import (
    CopyTaskConfig,
    AssociativeRecallConfig,
    AlgorithmicTaskConfig
)

# =====================================================================
# GLOBAL CONFIGURATION
# =====================================================================

# --- Task Registry ---
# Map task names to integer IDs for one-hot encoding
TASK_MAP = {
    "copy": 0,
    "associative_recall": 1,
    "repeat_copy": 2,
    "priority_access": 3,
    "dynamic_ngram": 4,
    "insertion_sort": 5
}
NUM_TASKS = len(TASK_MAP)

# --- Data Normalization ---
# All tasks must fit within these dimensions for the single model.
# MAX_VECTOR_SIZE accommodates Copy (8+markers), N-Gram (8), etc.
MAX_VECTOR_SIZE = 16
MAX_SEQ_LENGTH = 100

# --- Model Architecture ---
MEMORY_SIZE = 128
MEMORY_DIM = 20
CONTROLLER_DIM = 256
NUM_READ_HEADS = 1
NUM_WRITE_HEADS = 1
CONTROLLER_TYPE = "lstm"
SHIFT_RANGE = 3
EPSILON = 1e-6

# --- Training Hyperparameters ---
BATCH_SIZE = 64
STEPS_PER_EPOCH = 100
EPOCHS = 100
VALIDATION_STEPS = 20
LEARNING_RATE = 1e-4
CLIP_NORM = 10.0

# --- Evaluation Settings ---
NUM_EVAL_SAMPLES = 100

# --- Paths ---
CHECKPOINT_DIR = "checkpoints/multitask_ntm"
LOG_DIR = "logs/multitask_ntm"


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def pad_and_normalize(
    data: TaskData,
    task_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize dimensions for the multi-task model (Padding/Cropping).

    Handles task-specific output formatting (e.g., Associative Recall
    outputs only at the last timestep) and generic padding to MAX dimensions.

    Args:
        data: Raw TaskData object from a generator.
        task_key: Name of the task (for specific handling).

    Returns:
        tuple: (inputs_padded, targets_padded, masks_padded)
            inputs: (batch, MAX_SEQ_LENGTH, MAX_VECTOR_SIZE)
            targets: (batch, MAX_SEQ_LENGTH, MAX_VECTOR_SIZE)
            masks: (batch, MAX_SEQ_LENGTH)
    """
    inputs_raw = data.inputs
    targets_raw = data.targets
    masks_raw = data.masks

    curr_batch, curr_seq, _ = inputs_raw.shape
    target_dim = targets_raw.shape[-1]

    # --- Task Specific Adjustments ---

    # 1. Associative Recall: Generator outputs (Batch, Val_Dim)
    #    We need (Batch, Seq, Val_Dim) with target at the end.
    if task_key == "associative_recall":
        seq_targets = np.zeros(
            (curr_batch, curr_seq, target_dim), dtype=np.float32
        )
        seq_targets[:, -1, :] = targets_raw
        targets_raw = seq_targets

        masks_raw = np.zeros((curr_batch, curr_seq), dtype=np.float32)
        masks_raw[:, -1] = 1.0

    # 2. Insertion Sort: Ensure mask exists if generator doesn't provide one
    if task_key == "insertion_sort":
        if masks_raw is None:
            masks_raw = np.ones((curr_batch, curr_seq), dtype=np.float32)

    # --- Generic Padding ---

    # Initialize buffers with zeros
    inputs_padded = np.zeros(
        (curr_batch, MAX_SEQ_LENGTH, MAX_VECTOR_SIZE), dtype=np.float32
    )
    targets_padded = np.zeros(
        (curr_batch, MAX_SEQ_LENGTH, MAX_VECTOR_SIZE), dtype=np.float32
    )
    masks_padded = np.zeros(
        (curr_batch, MAX_SEQ_LENGTH), dtype=np.float32
    )

    # Determine limits to avoid overflow
    limit_seq = min(curr_seq, MAX_SEQ_LENGTH)
    limit_in_dim = min(inputs_raw.shape[-1], MAX_VECTOR_SIZE)
    limit_out_dim = min(target_dim, MAX_VECTOR_SIZE)

    # Copy data into buffers
    inputs_padded[:, :limit_seq, :limit_in_dim] = \
        inputs_raw[:, :limit_seq, :limit_in_dim]

    targets_padded[:, :limit_seq, :limit_out_dim] = \
        targets_raw[:, :limit_seq, :limit_out_dim]

    if masks_raw is not None:
        masks_padded[:, :limit_seq] = masks_raw[:, :limit_seq]
    else:
        # Fallback: assume valid if no mask provided
        masks_padded[:, :limit_seq] = 1.0

    return inputs_padded, targets_padded, masks_padded


# =====================================================================
# DATA PIPELINE
# =====================================================================

class UnifiedTaskGenerator(keras.utils.Sequence):
    """
    A unified Keras Sequence that generates batches for multiple tasks.

    It randomly selects a task for each batch, generates data using specific
    generators, and normalizes dimensions using `pad_and_normalize`.
    """

    def __init__(
        self,
        batch_size: int,
        steps_per_epoch: int,
        mode: str = 'train',
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.steps = steps_per_epoch
        self.mode = mode
        self.rng = np.random.default_rng(42 if mode == 'val' else None)
        seed = 42 if mode == 'val' else None

        # Initialize Generators
        self.generators = {
            "copy": CopyTaskGenerator(CopyTaskConfig(
                sequence_length=10, vector_size=8, random_seed=seed
            )),
            "associative_recall": AssociativeRecallGenerator(
                AssociativeRecallConfig(
                    num_items=6, key_size=8, value_size=8, random_seed=seed
                )
            ),
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
        """Helper to call specific generator with randomized parameters."""
        gen = self.generators[task_key]

        if task_key == "copy":
            seq_len = self.rng.integers(5, 20)
            return gen.generate(
                self.batch_size, sequence_length=seq_len
            )
        elif task_key == "associative_recall":
            items = self.rng.integers(2, 6)
            return gen.generate(
                self.batch_size, num_items=items
            )
        elif task_key == "repeat_copy":
            seq_len = self.rng.integers(3, 8)
            repeats = self.rng.integers(2, 4)
            return gen.generate(
                self.batch_size, sequence_length=seq_len, num_repeats=repeats
            )
        elif task_key == "priority_access":
            num_items = self.rng.integers(3, 10)
            return gen.generate(
                self.batch_size, num_items=num_items
            )
        elif task_key == "dynamic_ngram":
            seq_len = self.rng.integers(10, 30)
            data = gen.generate(
                self.batch_size, sequence_length=seq_len
            )
            # Apply N-Gram mask (ignore context setup, typically first 2)
            if data.masks is None:
                mask = np.ones((self.batch_size, seq_len))
                mask[:, :2] = 0.0
                data.masks = mask
            return data
        elif task_key == "insertion_sort":
            size = self.rng.integers(5, 16)
            return gen.generate(
                self.batch_size, problem_size=size
            )
        else:
            raise ValueError(f"Unknown task key: {task_key}")

    def __getitem__(
        self,
        index: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        # 1. Select Task
        task_key = self.rng.choice(list(TASK_MAP.keys()))
        task_id = TASK_MAP[task_key]

        # 2. Generate Raw Data
        data = self._generate_raw_data(task_key)

        # 3. Normalize
        inputs, targets, masks = pad_and_normalize(data, task_key)

        # 4. Create Task Condition Vector
        task_one_hot = np.zeros(
            (self.batch_size, NUM_TASKS), dtype=np.float32
        )
        task_one_hot[:, task_id] = 1.0

        # 5. Return formatted batch
        # Using dictionary inputs to satisfy Keras/TF signature requirements
        inputs_dict = {
            "sequence_in": inputs,
            "task_id_in": task_one_hot
        }

        return inputs_dict, targets, masks


# =====================================================================
# METRICS EVALUATION
# =====================================================================

def evaluate_all_tasks(model: keras.Model, num_samples: int = 100) -> None:
    """
    Runs a comprehensive evaluation loop over all supported tasks.

    Generates a validation batch for each task type, runs inference, and
    computes relevant metrics (Accuracy for discrete tasks, MSE for continuous).

    Args:
        model: The trained Keras model.
        num_samples: Number of test samples per task.
    """
    logger.info("=" * 60)
    logger.info(f"FINAL EVALUATION REPORT (N={num_samples} samples/task)")
    logger.info("=" * 60)

    # Reuse the generator logic in 'validation' mode (fixed seed internally)
    eval_gen = UnifiedTaskGenerator(num_samples, 1, mode='val')

    results: Dict[str, float] = {}

    for task_name, task_id in TASK_MAP.items():
        # 1. Generate Data
        raw_data = eval_gen._generate_raw_data(task_name)

        # 2. Prepare Inputs
        inputs, targets, masks = pad_and_normalize(raw_data, task_name)

        task_one_hot = np.zeros((num_samples, NUM_TASKS), dtype=np.float32)
        task_one_hot[:, task_id] = 1.0

        # 3. Predict
        preds = model.predict(
            {"sequence_in": inputs, "task_id_in": task_one_hot},
            verbose=0
        )

        # 4. Calculate Metrics
        mask_bool = masks.astype(bool)

        # Flatten based on mask to ignore padding in metrics
        preds_flat = preds[mask_bool]
        targets_flat = targets[mask_bool]

        if task_name == "insertion_sort":
            # Continuous data: Use Mean Squared Error
            mse = np.mean((preds_flat - targets_flat) ** 2)
            logger.info(f"Task: {task_name:<20} | MSE: {mse:.6f}")
            results[task_name] = mse
        else:
            # Binary data: Use Accuracy / Bit Error Rate
            # Threshold predictions at 0.5
            preds_binary = (preds_flat > 0.5).astype(float)
            accuracy = np.mean(preds_binary == targets_flat)
            error_rate = 1.0 - accuracy
            logger.info(
                f"Task: {task_name:<20} | "
                f"Acc: {accuracy:.4f} | Error: {error_rate:.4f}"
            )
            results[task_name] = accuracy

    logger.info("=" * 60)


# =====================================================================
# MAIN TRAINING LOOP
# =====================================================================

def main() -> None:
    logger.info("Initializing Expanded Multi-Task NTM Training...")

    # 1. Prepare Configuration
    ntm_config = NTMConfig(
        memory_size=MEMORY_SIZE,
        memory_dim=MEMORY_DIM,
        controller_dim=CONTROLLER_DIM,
        controller_type=CONTROLLER_TYPE,
        num_read_heads=NUM_READ_HEADS,
        num_write_heads=NUM_WRITE_HEADS,
        shift_range=SHIFT_RANGE,
        epsilon=EPSILON
    )

    # 2. Build Model
    seq_input = keras.Input(
        shape=(MAX_SEQ_LENGTH, MAX_VECTOR_SIZE),
        name="sequence_in"
    )
    task_input = keras.Input(
        shape=(NUM_TASKS,),
        name="task_id_in"
    )

    # Instantiate Wrapper
    ntm_model = NTMMultiTask(
        ntm_config=ntm_config,
        output_dim=MAX_VECTOR_SIZE,
        num_tasks=NUM_TASKS
    )

    # Functional API Construction
    # The wrapper expects a list: [sequence, task_id]
    outputs = ntm_model([seq_input, task_input])

    # Final activation
    # Sigmoid is standard for NTM copy tasks. For sorting normalized to [0,1],
    # sigmoid is acceptable.
    final_output = keras.layers.Activation(
        "sigmoid", name="output_sigmoid"
    )(outputs)

    model = keras.Model(
        inputs=[seq_input, task_input],
        outputs=final_output,
        name="multitask_ntm_expanded"
    )

    # 3. Compile
    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIP_NORM
    )

    # Use BinaryCrossentropy for stability across binary tasks.
    # Track MSE for sorting performance.
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.MeanSquaredError(name="mse")
        ]
    )

    model.summary()

    # 4. Data Generators
    logger.info(f"Setting up data generators for {NUM_TASKS} tasks...")
    train_gen = UnifiedTaskGenerator(BATCH_SIZE, STEPS_PER_EPOCH, mode='train')
    val_gen = UnifiedTaskGenerator(BATCH_SIZE, VALIDATION_STEPS, mode='val')

    # 5. Callbacks
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, timestamp),
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
        )
    ]

    # 6. Training Loop
    logger.info(f"Starting training for {EPOCHS} epochs...")
    try:
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        logger.info("Training complete.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current state...")
        model.save(os.path.join(CHECKPOINT_DIR, "interrupted_model.keras"))

    # 7. Final Evaluation
    evaluate_all_tasks(model, num_samples=NUM_EVAL_SAMPLES)


if __name__ == "__main__":
    main()
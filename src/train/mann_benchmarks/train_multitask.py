import os
import numpy as np
import keras
from datetime import datetime
from typing import Tuple, List, Dict, Any, Union

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------
from dl_techniques.utils.logger import logger
from dl_techniques.layers.ntm.ntm_interface import NTMConfig
from dl_techniques.models.ntm.model_multitask import NTMMultiTask

# Import generators from the benchmarks package
from .data_generators import (
    CopyTaskGenerator,
    AssociativeRecallGenerator,
    RepeatCopyGenerator,
    TaskData
)
from .config import (
    CopyTaskConfig,
    AssociativeRecallConfig
)

# =====================================================================
# CONFIGURATION
# =====================================================================

# ---------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------
MEMORY_SIZE = 128
MEMORY_DIM = 20
CONTROLLER_DIM = 256
NUM_READ_HEADS = 1
NUM_WRITE_HEADS = 1

# ---------------------------------------------------------------------
# Task Definitions & Normalization
# ---------------------------------------------------------------------
# We normalize all tasks to these dimensions to allow single-model training
MAX_VECTOR_SIZE = 16  # Must handle max(features) + markers across tasks
MAX_SEQ_LENGTH = 80  # Sufficient for RepeatCopy which produces long sequences

TASK_MAP = {
    "copy": 0,
    "associative_recall": 1,
    "repeat_copy": 2
}
NUM_TASKS = len(TASK_MAP)

# ---------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------
BATCH_SIZE = 64
STEPS_PER_EPOCH = 100
EPOCHS = 100
VALIDATION_STEPS = 20
LEARNING_RATE = 1e-4
CLIP_NORM = 5.0

# Paths
CHECKPOINT_DIR = "checkpoints/multitask_ntm"
LOG_DIR = "logs/multitask_ntm"


# =====================================================================
# UNIFIED DATA PIPELINE
# =====================================================================

class UnifiedTaskGenerator(keras.utils.Sequence):
    """
    A unified Keras Sequence that generates batches for multiple tasks.

    It handles:
    1. Randomly selecting a task per batch.
    2. Generating task-specific data.
    3. Padding sequences and feature dimensions to fixed sizes.
    4. Constructing masks and One-Hot task IDs.

    Returns batches in the format:
        ({'sequence_in': X, 'task_id_in': T}, target_sequence, sample_weights)
    """

    def __init__(self, batch_size: int, steps_per_epoch: int, mode: str = 'train', **kwargs):
        # Pass kwargs to super to suppress Keras 3 warnings regarding workers/multiprocessing
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.steps = steps_per_epoch
        self.mode = mode
        self.rng = np.random.default_rng(42 if mode == 'val' else None)

        # Initialize specific generators
        # 1. Copy Task
        self.copy_gen = CopyTaskGenerator(CopyTaskConfig(
            sequence_length=10,
            vector_size=8,
            random_seed=1 if mode == 'val' else None
        ))

        # 2. Associative Recall
        self.recall_gen = AssociativeRecallGenerator(AssociativeRecallConfig(
            num_items=6,
            key_size=8,
            value_size=8,
            random_seed=2 if mode == 'val' else None
        ))

        # 3. Repeat Copy
        self.repeat_gen = RepeatCopyGenerator(
            vector_size=8,
            random_seed=3 if mode == 'val' else None
        )

    def __len__(self) -> int:
        return self.steps

    def _pad_and_normalize(self, data: TaskData, task_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardizes dimensions for the multi-task model.

        Args:
            data: Raw TaskData object from a generator.
            task_key: 'copy', 'associative_recall', or 'repeat_copy'.

        Returns:
            inputs: (batch, MAX_SEQ_LENGTH, MAX_VECTOR_SIZE)
            targets: (batch, MAX_SEQ_LENGTH, MAX_VECTOR_SIZE)
            masks: (batch, MAX_SEQ_LENGTH)
        """
        inputs_raw = data.inputs
        targets_raw = data.targets
        masks_raw = data.masks

        curr_batch, curr_seq, curr_dim = inputs_raw.shape

        # Handle Associative Recall specific logic
        # AR generator outputs (Batch, Val_Dim) targets, need (Batch, Seq, Val_Dim)
        if task_key == "associative_recall":
            # Expand targets to sequence
            val_dim = targets_raw.shape[-1]
            seq_targets = np.zeros((curr_batch, curr_seq, val_dim), dtype=np.float32)
            # Target is expected at the last timestep
            seq_targets[:, -1, :] = targets_raw
            targets_raw = seq_targets

            # Create mask (only penalize last timestep)
            masks_raw = np.zeros((curr_batch, curr_seq), dtype=np.float32)
            masks_raw[:, -1] = 1.0

        # Create buffers
        inputs_padded = np.zeros((curr_batch, MAX_SEQ_LENGTH, MAX_VECTOR_SIZE), dtype=np.float32)
        targets_padded = np.zeros((curr_batch, MAX_SEQ_LENGTH, MAX_VECTOR_SIZE), dtype=np.float32)
        masks_padded = np.zeros((curr_batch, MAX_SEQ_LENGTH), dtype=np.float32)

        # Clip dimensions if they exceed MAX (robustness)
        limit_seq = min(curr_seq, MAX_SEQ_LENGTH)
        limit_dim = min(curr_dim, MAX_VECTOR_SIZE)
        limit_out_dim = min(targets_raw.shape[-1], MAX_VECTOR_SIZE)

        # Copy data into buffers
        inputs_padded[:, :limit_seq, :limit_dim] = inputs_raw[:, :limit_seq, :limit_dim]
        targets_padded[:, :limit_seq, :limit_out_dim] = targets_raw[:, :limit_seq, :limit_out_dim]

        if masks_raw is not None:
            masks_padded[:, :limit_seq] = masks_raw[:, :limit_seq]
        else:
            # Fallback if no mask: assume all data is valid
            masks_padded[:, :limit_seq] = 1.0

        return inputs_padded, targets_padded, masks_padded

    def __getitem__(self, index: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        # 1. Select Task
        task_key = self.rng.choice(list(TASK_MAP.keys()))
        task_id = TASK_MAP[task_key]

        # 2. Generate Raw Data
        if task_key == "copy":
            seq_len = self.rng.integers(5, 20)
            data = self.copy_gen.generate(self.batch_size, sequence_length=seq_len)

        elif task_key == "associative_recall":
            items = self.rng.integers(2, 6)
            data = self.recall_gen.generate(self.batch_size, num_items=items)

        elif task_key == "repeat_copy":
            seq_len = self.rng.integers(3, 8)
            repeats = self.rng.integers(2, 4)
            data = self.repeat_gen.generate(self.batch_size, sequence_length=seq_len, num_repeats=repeats)

        # 3. Normalize
        inputs, targets, masks = self._pad_and_normalize(data, task_key)

        # 4. Create Task Condition Vector
        task_one_hot = np.zeros((self.batch_size, NUM_TASKS), dtype=np.float32)
        task_one_hot[:, task_id] = 1.0

        # 5. Return formatted batch
        # Using Dictionary for inputs prevents TypeError in TF signature inference
        # keys must match Input(name=...) in the model
        inputs_dict = {
            "sequence_in": inputs,
            "task_id_in": task_one_hot
        }

        return inputs_dict, targets, masks


# =====================================================================
# TRAINING ROUTINE
# =====================================================================

def main():
    logger.info("Initializing Multi-Task NTM Training...")

    # 1. Prepare Configuration
    ntm_config = NTMConfig(
        memory_size=MEMORY_SIZE,
        memory_dim=MEMORY_DIM,
        controller_dim=CONTROLLER_DIM,
        controller_type="lstm",
        num_read_heads=NUM_READ_HEADS,
        num_write_heads=NUM_WRITE_HEADS,
        shift_range=3,
        epsilon=1e-6
    )

    # 2. Build Model
    # Explicit Input layers with names matching the generator keys
    seq_input = keras.Input(shape=(MAX_SEQ_LENGTH, MAX_VECTOR_SIZE), name="sequence_in")
    task_input = keras.Input(shape=(NUM_TASKS,), name="task_id_in")

    # Instantiate Wrapper
    ntm_model = NTMMultiTask(
        ntm_config=ntm_config,
        output_dim=MAX_VECTOR_SIZE,
        num_tasks=NUM_TASKS
    )

    # Functional API Construction
    # The wrapper expects a list [sequence, task_id]
    outputs = ntm_model([seq_input, task_input])

    # Final activation for binary data (Copy/Repeat) and approximate for floats
    final_output = keras.layers.Activation("sigmoid", name="output_sigmoid")(outputs)

    model = keras.Model(
        inputs=[seq_input, task_input],
        outputs=final_output,
        name="multitask_ntm_v1"
    )

    # 3. Compile
    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIP_NORM
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc")]
    )

    model.summary()

    # 4. Data Generators
    logger.info("Setting up data generators...")
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

    # 7. Verification / Sanity Check
    logger.info("Running post-training verification on Copy Task...")

    # Generate a single manual sample to verify inference
    test_gen = CopyTaskGenerator(CopyTaskConfig(sequence_length=8, vector_size=8))
    data = test_gen.generate(num_samples=1)

    # Pad input
    inp = data.inputs
    curr_seq = inp.shape[1]
    curr_dim = inp.shape[2]

    padded_inp = np.zeros((1, MAX_SEQ_LENGTH, MAX_VECTOR_SIZE), dtype=np.float32)
    padded_inp[0, :curr_seq, :curr_dim] = inp

    # Create Copy Task ID
    task_id = np.zeros((1, NUM_TASKS), dtype=np.float32)
    task_id[0, TASK_MAP['copy']] = 1.0

    # Predict using dictionary input for consistency
    preds = model.predict(
        {"sequence_in": padded_inp, "task_id_in": task_id},
        verbose=0
    )

    # Check shape
    logger.info(f"Input shape: {padded_inp.shape}")
    logger.info(f"Prediction shape: {preds.shape}")

    logger.info("Verification complete.")


if __name__ == "__main__":
    main()
"""Neural Turing Machine training on the copy task."""

import os
import keras
import numpy as np
from datetime import datetime

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.layers.ntm import create_ntm
from . import CopyTaskGenerator, CopyTaskConfig


# =====================================================================
# Configuration
# =====================================================================

# NTM Memory
MEMORY_SIZE = 128
MEMORY_DIM = 20

# Controller
CONTROLLER_DIM = 100
CONTROLLER_TYPE = 'lstm'

# Heads
NUM_READ_HEADS = 1
NUM_WRITE_HEADS = 1

# Copy Task
SEQUENCE_LENGTH = 20
VECTOR_SIZE = 8
NUM_SAMPLES = 100000

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
CLIP_NORM = 1.0
EPOCHS = 100
VALIDATION_SPLIT = 0.1

# LR Scheduler
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 25
MIN_LEARNING_RATE = 1e-6

# Early Stopping
EARLY_STOPPING_PATIENCE = 10

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_MONITOR = "val_loss"

# Evaluation
NUM_EVAL_SAMPLES = 20
SUCCESS_THRESHOLD = 0.9


# =====================================================================
# Training Pipeline
# =====================================================================

def main():
    if CopyTaskGenerator is None:
        logger.error("Cannot proceed without CopyTaskGenerator.")
        return

    setup_gpu()

    # Generate data
    logger.info("Generating Copy Task data...")
    config = CopyTaskConfig(
        sequence_length=SEQUENCE_LENGTH,
        vector_size=VECTOR_SIZE,
        num_samples=NUM_SAMPLES,
    )
    generator = CopyTaskGenerator(config)
    data = generator.generate()
    logger.info(f"Input: {data.inputs.shape}, Target: {data.targets.shape}")

    # Build model
    seq_len = data.inputs.shape[1]
    input_dim = data.inputs.shape[2]
    output_dim = data.targets.shape[2]

    inputs = keras.Input(shape=(seq_len, input_dim), name="input_sequence")
    ntm_layer = create_ntm(
        memory_size=MEMORY_SIZE, memory_dim=MEMORY_DIM, output_dim=output_dim,
        controller_dim=CONTROLLER_DIM, controller_type=CONTROLLER_TYPE,
        num_read_heads=NUM_READ_HEADS, num_write_heads=NUM_WRITE_HEADS,
        return_sequences=True,
    )
    x = ntm_layer(inputs)
    outputs = keras.layers.Activation('sigmoid', name="binary_output")(x)
    model = keras.Model(inputs, outputs, name="ntm_copy_task")

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM),
        loss='binary_crossentropy',
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    model.summary()

    # Callbacks
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{CHECKPOINT_DIR}/ntm_copy_{timestamp}.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath, monitor=CHECKPOINT_MONITOR, save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor=CHECKPOINT_MONITOR, patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=CHECKPOINT_MONITOR, factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE, min_lr=MIN_LEARNING_RATE, verbose=1
        )
    ]

    # Train
    logger.info("Starting training...")
    history = model.fit(
        data.inputs, data.targets, sample_weight=data.masks,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks, verbose=1,
    )

    # Reload and evaluate
    logger.info(f"Reloading model from {filepath}...")
    try:
        model = keras.models.load_model(filepath)
        logger.info("Model reloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        logger.warning("Continuing evaluation with in-memory model.")

    evaluate_model(model, data)


def evaluate_model(model, data, num_eval=NUM_EVAL_SAMPLES):
    """Detailed evaluation of NTM copy task performance."""
    indices = np.random.choice(len(data.inputs), num_eval, replace=False)
    eval_inputs = data.inputs[indices]
    eval_targets = data.targets[indices]
    eval_masks = data.masks[indices]

    preds = model.predict(eval_inputs, verbose=0)
    preds_binary = (preds > 0.5).astype(float)

    seq_accs = []
    bit_accs = []

    for i in range(num_eval):
        mask_boolean = eval_masks[i].astype(bool).flatten()
        p_valid = preds_binary[i].flatten()[mask_boolean]
        t_valid = eval_targets[i].flatten()[mask_boolean]

        if len(p_valid) == 0:
            continue

        seq_accs.append(1.0 if np.array_equal(p_valid, t_valid) else 0.0)
        bit_accs.append(np.mean(p_valid == t_valid))

    mean_bit_acc = np.mean(bit_accs)
    mean_seq_acc = np.mean(seq_accs)

    logger.info(f"Evaluation (N={num_eval}): Bit Acc={mean_bit_acc:.2%}, Sequence Acc={mean_seq_acc:.2%}")

    if mean_seq_acc > SUCCESS_THRESHOLD:
        logger.info("SUCCESS: NTM has solved the copy task!")
    else:
        logger.info("STATUS: NTM needs more training or tuning.")


if __name__ == '__main__':
    main()

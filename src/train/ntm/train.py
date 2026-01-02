import os
import keras
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ntm import create_ntm
from . import CopyTaskGenerator, CopyTaskConfig


# =====================================================================
# CONFIGURATION
# =====================================================================

# ---------------------------------------------------------------------
# NTM Memory Configuration
# ---------------------------------------------------------------------
# MEMORY_SIZE: Number of memory slots (rows) in the external memory matrix.
#              Larger values allow storing more information but increase
#              computational cost and memory usage.
MEMORY_SIZE = 128

# MEMORY_DIM: Dimensionality of each memory slot (columns).
#             Controls the amount of information each slot can hold.
MEMORY_DIM = 20

# ---------------------------------------------------------------------
# NTM Controller Configuration
# ---------------------------------------------------------------------
# CONTROLLER_DIM: Hidden dimension of the controller network.
#                 Higher values increase capacity but also parameters.
CONTROLLER_DIM = 100

# CONTROLLER_TYPE: Architecture of the controller network.
#                  Options: 'lstm', 'gru', or 'feedforward'
#                  LSTM/GRU provide temporal context; feedforward is simpler.
CONTROLLER_TYPE = 'lstm'

# ---------------------------------------------------------------------
# NTM Head Configuration
# ---------------------------------------------------------------------
# NUM_READ_HEADS: Number of read heads for parallel memory reads.
#                 Multiple heads can attend to different memory locations.
NUM_READ_HEADS = 1

# NUM_WRITE_HEADS: Number of write heads for parallel memory writes.
#                  Multiple heads can write to different locations.
NUM_WRITE_HEADS = 1

# ---------------------------------------------------------------------
# Copy Task Configuration
# ---------------------------------------------------------------------
# SEQUENCE_LENGTH: Length of the binary sequence to copy.
#                  Longer sequences are harder for the NTM to learn.
SEQUENCE_LENGTH = 20

# VECTOR_SIZE: Dimensionality of each vector in the sequence.
#              Each timestep contains VECTOR_SIZE binary values.
VECTOR_SIZE = 8

# NUM_SAMPLES: Total number of training samples to generate.
#              NTM requires many samples to converge properly.
NUM_SAMPLES = 50000

# ---------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------
# BATCH_SIZE: Number of samples per gradient update.
BATCH_SIZE = 32

# LEARNING_RATE: Initial learning rate for the optimizer.
#                NTM is sensitive to learning rate; smaller is often better.
LEARNING_RATE = 1e-4

# CLIP_NORM: Maximum gradient norm for gradient clipping.
#            Crucial for NTM stability due to complex gradient flow.
CLIP_NORM = 10.0

# EPOCHS: Maximum number of training epochs.
EPOCHS = 100

# VALIDATION_SPLIT: Fraction of training data used for validation.
VALIDATION_SPLIT = 0.1

# ---------------------------------------------------------------------
# Learning Rate Scheduler Configuration
# ---------------------------------------------------------------------
# LR_REDUCE_FACTOR: Factor by which to reduce LR when plateau is detected.
LR_REDUCE_FACTOR = 0.5

# LR_REDUCE_PATIENCE: Number of epochs with no improvement before reducing LR.
LR_REDUCE_PATIENCE = 5

# MIN_LEARNING_RATE: Minimum learning rate after reductions.
MIN_LEARNING_RATE = 1e-6

# ---------------------------------------------------------------------
# Early Stopping Configuration
# ---------------------------------------------------------------------
# EARLY_STOPPING_PATIENCE: Number of epochs with no improvement before stopping.
EARLY_STOPPING_PATIENCE = 10

# ---------------------------------------------------------------------
# Checkpointing Configuration
# ---------------------------------------------------------------------
# CHECKPOINT_DIR: Directory to save model checkpoints.
CHECKPOINT_DIR = "checkpoints"

# CHECKPOINT_MONITOR: Metric to monitor for saving best model.
CHECKPOINT_MONITOR = "val_loss"

# ---------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------
# NUM_EVAL_SAMPLES: Number of samples to use for detailed evaluation.
NUM_EVAL_SAMPLES = 20

# SUCCESS_THRESHOLD: Sequence accuracy threshold to consider task solved.
SUCCESS_THRESHOLD = 0.9


# =====================================================================
# TRAINING PIPELINE
# =====================================================================

def main():
    if CopyTaskGenerator is None:
        logger.error("Cannot proceed without CopyTaskGenerator.")
        return

    # 1. Generate Data
    logger.info("Generating Copy Task data...")
    config = CopyTaskConfig(
        sequence_length=SEQUENCE_LENGTH,
        vector_size=VECTOR_SIZE,
        num_samples=NUM_SAMPLES,
    )
    generator = CopyTaskGenerator(config)
    data = generator.generate()

    logger.info(f"Input shape:  {data.inputs.shape}")
    logger.info(f"Target shape: {data.targets.shape}")

    # 2. Build Model (Functional API)
    seq_len = data.inputs.shape[1]
    input_dim = data.inputs.shape[2]
    output_dim = data.targets.shape[2]

    inputs = keras.Input(shape=(seq_len, input_dim), name="input_sequence")

    ntm_layer = create_ntm(
        memory_size=MEMORY_SIZE,
        memory_dim=MEMORY_DIM,
        output_dim=output_dim,
        controller_dim=CONTROLLER_DIM,
        controller_type=CONTROLLER_TYPE,
        num_read_heads=NUM_READ_HEADS,
        num_write_heads=NUM_WRITE_HEADS,
        return_sequences=True,
    )

    x = ntm_layer(inputs)
    outputs = keras.layers.Activation('sigmoid', name="binary_output")(x)

    model = keras.Model(inputs, outputs, name="ntm_copy_task")

    # 3. Compile
    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIP_NORM
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    model.summary()

    # 4. Callbacks
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{CHECKPOINT_DIR}/ntm_copy_{timestamp}.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=CHECKPOINT_MONITOR,
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor=CHECKPOINT_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=CHECKPOINT_MONITOR,
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        )
    ]

    # 5. Train
    logger.info("Starting training...")
    history = model.fit(
        data.inputs,
        data.targets,
        sample_weight=data.masks,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Evaluation
    logger.info("Evaluating on hold-out set...")

    logger.info(f"Reloading model from {filepath} to verify serialization...")
    try:
        model = keras.models.load_model(filepath)
        logger.info("Model reloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        logger.warning("Continuing evaluation with current model in memory.")

    evaluate_model(model, data)


def evaluate_model(model, data, num_eval=NUM_EVAL_SAMPLES):
    """
    Detailed evaluation of NTM performance.

    :param model: Trained Keras model to evaluate.
    :param data: Dataset object containing inputs, targets, and masks.
    :param num_eval: Number of samples to use for evaluation.
    """
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
        p_flat = preds_binary[i].flatten()
        t_flat = eval_targets[i].flatten()

        p_valid = p_flat[mask_boolean]
        t_valid = t_flat[mask_boolean]

        if len(p_valid) == 0:
            continue

        is_perfect = np.array_equal(p_valid, t_valid)
        seq_accs.append(1.0 if is_perfect else 0.0)

        bit_acc = np.mean(p_valid == t_valid)
        bit_accs.append(bit_acc)

    mean_bit_acc = np.mean(bit_accs)
    mean_seq_acc = np.mean(seq_accs)

    logger.info("-" * 40)
    logger.info(f"Evaluation Results (N={num_eval})")
    logger.info("-" * 40)
    logger.info(f"Mean Bit Accuracy:     {mean_bit_acc:.2%}")
    logger.info(f"Perfect Sequence Acc:  {mean_seq_acc:.2%}")
    logger.info("-" * 40)

    if mean_seq_acc > SUCCESS_THRESHOLD:
        logger.info("SUCCESS: NTM has solved the copy task!")
    else:
        logger.info("STATUS: NTM needs more training or tuning.")


if __name__ == '__main__':
    main()
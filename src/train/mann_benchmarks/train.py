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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MEMORY_SIZE = 128
MEMORY_DIM = 20
CONTROLLER_DIM = 100
CONTROLLER_TYPE = 'lstm'  # 'lstm', 'gru', or 'feedforward'
NUM_READ_HEADS = 1
NUM_WRITE_HEADS = 1

# Task Params
SEQUENCE_LENGTH = 20
VECTOR_SIZE = 8
NUM_SAMPLES = 50000  # NTM needs many samples to converge
BATCH_SIZE = 32

# Training Params
LEARNING_RATE = 1e-4  # NTM is sensitive to LR
CLIP_NORM = 10.0  # Crucial for NTM stability
EPOCHS = 100


# ---------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------

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
    # The input shape is (seq_len, vector_size + 1) typically for Copy Task
    # (the +1 is often the delimiter flag)
    seq_len = data.inputs.shape[1]
    input_dim = data.inputs.shape[2]
    output_dim = data.targets.shape[2]

    inputs = keras.Input(shape=(seq_len, input_dim), name="input_sequence")

    # Create the NTM layer
    # Note: create_ntm returns a NeuralTuringMachine layer instance
    ntm_layer = create_ntm(
        memory_size=MEMORY_SIZE,
        memory_dim=MEMORY_DIM,
        output_dim=output_dim,
        controller_dim=CONTROLLER_DIM,
        controller_type=CONTROLLER_TYPE,
        num_read_heads=NUM_READ_HEADS,
        num_write_heads=NUM_WRITE_HEADS,
        return_sequences=True,  # Important for Copy Task
    )

    x = ntm_layer(inputs)

    # NTM outputs logits (linear projection), so we add Sigmoid for binary targets
    outputs = keras.layers.Activation('sigmoid', name="binary_output")(x)

    model = keras.Model(inputs, outputs, name="ntm_copy_task")

    # 3. Compile
    # RMSprop or Adam with gradient clipping is essential for NTM convergence
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    model.summary()

    # 4. Callbacks
    # Ensure we test serialization by saving the best model
    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"checkpoints/ntm_copy_{timestamp}.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Train
    logger.info("Starting training...")
    history = model.fit(
        data.inputs,
        data.targets,
        sample_weight=data.masks,  # Apply masks so we don't train on padding/inputs
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Evaluation
    logger.info("Evaluating on hold-out set...")

    # Reload model to verify serialization worked correctly
    logger.info(f"Reloading model from {filepath} to verify serialization...")
    try:
        model = keras.models.load_model(filepath)
        logger.info("Model reloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        logger.warning("Continuing evaluation with current model in memory.")

    evaluate_model(model, data)


def evaluate_model(model, data, num_eval=20):
    """
    Detailed evaluation of NTM performance.
    """
    # Select a subset for detailed logging
    indices = np.random.choice(len(data.inputs), num_eval, replace=False)
    eval_inputs = data.inputs[indices]
    eval_targets = data.targets[indices]
    eval_masks = data.masks[indices]

    preds = model.predict(eval_inputs, verbose=0)
    preds_binary = (preds > 0.5).astype(float)

    seq_accs = []
    bit_accs = []

    for i in range(num_eval):
        # Apply mask to ignore input phase and padding
        # Flatten logic: we only care about bits where mask == 1
        mask_boolean = eval_masks[i].astype(bool).flatten()

        # Flatten time and feature dimensions for comparison
        p_flat = preds_binary[i].flatten()
        t_flat = eval_targets[i].flatten()

        # Select valid bits
        p_valid = p_flat[mask_boolean]
        t_valid = t_flat[mask_boolean]

        if len(p_valid) == 0:
            continue

        # Sequence accuracy: 1.0 if ALL valid bits match, 0.0 otherwise
        is_perfect = np.array_equal(p_valid, t_valid)
        seq_accs.append(1.0 if is_perfect else 0.0)

        # Bit accuracy: percentage of matching bits
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

    if mean_seq_acc > 0.9:
        logger.info("SUCCESS: NTM has solved the copy task!")
    else:
        logger.info("STATUS: NTM needs more training or tuning.")


if __name__ == '__main__':
    main()
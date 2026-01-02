"""
Minimal NTM Quick Start Example.

Run: python quick_start_ntm.py
"""

import keras
import numpy as np

# Imports - adjust paths based on your project structure
from dl_techniques.layers.ntm import create_ntm
from . import CopyTaskGenerator, CopyTaskConfig


def main():
    # 1. Generate copy task data
    # Increase samples to allow convergence
    config = CopyTaskConfig(
        sequence_length=5,
        vector_size=4,
        num_samples=10000,
    )
    generator = CopyTaskGenerator(config)
    data = generator.generate()

    print(f"Input shape:  {data.inputs.shape}")
    print(f"Target shape: {data.targets.shape}")

    # 2. Create NTM model
    seq_len = data.inputs.shape[1]
    input_dim = data.inputs.shape[2]
    output_dim = data.targets.shape[2]

    inputs = keras.Input(shape=(seq_len, input_dim))
    x = create_ntm(
        memory_size=64,
        memory_dim=16,
        output_dim=output_dim,
        controller_dim=64,
        controller_type='lstm',
    )(inputs)
    outputs = keras.layers.Activation('sigmoid')(x)

    model = keras.Model(inputs, outputs)

    # Lower learning rate and keep clipping
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=10.0),
        loss='binary_crossentropy',
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    print(f"\nModel parameters: {model.count_params():,}")

    # 3. Train
    print("\nTraining...")
    model.fit(
        data.inputs,
        data.targets,
        sample_weight=data.masks,
        batch_size=64,
        epochs=30,
        validation_split=0.1,
        verbose=1,
    )

    # 4. Evaluate
    print("\nEvaluating...")
    num_eval = 20
    eval_inputs = data.inputs[:num_eval]
    eval_targets = data.targets[:num_eval]
    eval_masks = data.masks[:num_eval]

    preds = model.predict(eval_inputs, verbose=0)
    preds_binary = (preds > 0.5).astype(float)

    seq_accs = []
    bit_accs = []

    for i in range(num_eval):
        # Only evaluate on the output sequence part (where mask is 1)
        mask = eval_masks[i] == 1
        p = preds_binary[i][mask]
        t = eval_targets[i][mask]

        # Sequence accuracy (all bits match)
        if np.array_equal(p, t):
            seq_accs.append(1)
        else:
            seq_accs.append(0)

        # Bit accuracy (fraction of bits matching)
        bit_accs.append(np.mean(p == t))

    print(f"Mean Bit Accuracy (masked): {np.mean(bit_accs):.2%}")
    print(f"Sequence Accuracy (masked): {np.mean(seq_accs):.2%}")


if __name__ == '__main__':
    main()
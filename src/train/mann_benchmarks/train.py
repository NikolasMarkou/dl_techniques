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
    config = CopyTaskConfig(
        sequence_length=5,  # Short sequences for quick test
        vector_size=4,  # Small vectors
        num_samples=1000,
    )
    generator = CopyTaskGenerator(config)
    data = generator.generate()

    print(f"Input shape:  {data.inputs.shape}")  # (1000, 11, 6)
    print(f"Target shape: {data.targets.shape}")  # (1000, 11, 4)

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
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    print(f"\nModel parameters: {model.count_params():,}")

    # 3. Train
    print("\nTraining...")
    model.fit(
        data.inputs,
        data.targets,
        batch_size=32,
        epochs=20,
        validation_split=0.1,
        verbose=1,
    )

    # 4. Evaluate
    print("\nEvaluating...")
    preds = model.predict(data.inputs[:10], verbose=0)
    preds_binary = (preds > 0.5).astype(float)

    # Check sequence accuracy
    correct = np.all(np.isclose(preds_binary, data.targets[:10], atol=0.1), axis=(1, 2))
    print(f"Sequence accuracy (10 samples): {correct.mean():.2%}")


if __name__ == '__main__':
    main()
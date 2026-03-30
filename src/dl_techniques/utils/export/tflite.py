import os
import numpy as np

# 1. FORCE TENSORFLOW BACKEND (Must be done before importing keras)
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from model import TiRexCore  # Assuming your file is named model.py


def export_tirex_to_tflite(
        model_variant="small",
        input_length=168,
        features=1,
        prediction_length=24,
        output_path="tirex_model.tflite"
):
    print(f"Initializing TiRex-{model_variant} for export...")

    # 2. Initialize the model
    # We use fixed dimensions for TFLite to ensure better hardware acceleration
    model = TiRexCore.from_variant(
        variant=model_variant,
        prediction_length=prediction_length
    )

    # 3. Build the model by calling it with a concrete input shape
    # TFLite needs to know the exact input signature
    input_shape = (1, input_length, features)
    dummy_input = tf.ones(input_shape, dtype=tf.float32)
    _ = model(dummy_input)

    print("Converting to TFLite format...")

    # 4. Use the TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 5. Enable Optimization & Supported Ops
    # TiRex uses LSTMs and Attention, which often require 'Select TF Ops' (Flex)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS  # Enable Flex ops for LSTMs/Advanced Attention
    ]

    # Ensure the model handles the float32 precision common in time series
    converter.target_spec.supported_types = [tf.float32]

    # 6. Convert and Save
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Success! Model saved to: {output_path}")
    print(f"Model Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def verify_tflite(tflite_path, input_length, features):
    """Simple verification to ensure the TFLite model runs."""
    print("\nVerifying TFLite Inference...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare random input
    test_input = np.random.random((1, input_length, features)).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Input Shape: {test_input.shape}")
    print(f"Output Shape (Quantiles): {output_data.shape}")
    print("Verification complete.")


if __name__ == "__main__":
    # Settings should match your training config
    IN_LEN = 168
    FEATS = 1
    PRED_LEN = 24
    OUT_FILE = "tirex_small.tflite"

    export_tirex_to_tflite(
        model_variant="small",
        input_length=IN_LEN,
        features=FEATS,
        prediction_length=PRED_LEN,
        output_path=OUT_FILE
    )

    verify_tflite(OUT_FILE, IN_LEN, FEATS)
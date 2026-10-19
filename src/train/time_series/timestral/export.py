"""Export utilities for Timestral models.

Provides functions to export trained Timestral models to various formats
including Keras .keras, ONNX, and TensorFlow SavedModel.
"""

import os
from typing import Optional

import keras
import tensorflow as tf

from dl_techniques.models.time_series.timestral import TimestralModel, TimestralConfig


def export_to_keras(model: TimestralModel, output_path: str) -> None:
    """Export model to Keras .keras format.

    Args:
        model: Trained TimestralModel instance.
        output_path: Path to save the .keras file.
    """
    model.save(output_path)
    print(f"Model saved to {output_path}")


def export_to_onnx(
    model: TimestralModel,
    output_path: str,
    opset_version: int = 17,
    input_shape: Optional[tuple] = None,
) -> None:
    """Export model to ONNX format.

    Requires tf2onnx to be installed: pip install tf2onnx

    Args:
        model: Trained TimestralModel instance.
        output_path: Path to save the .onnx file.
        opset_version: ONNX opset version.
        input_shape: Input shape for the model (batch, seq_len, features).
            If None, uses (None, model.config.context_length, 1).
    """
    try:
        import tf2onnx
    except ImportError:
        raise ImportError("tf2onnx is required for ONNX export. Install with: pip install tf2onnx")

    if input_shape is None:
        input_shape = (None, model.config.context_length, 1)

    # Create a concrete function for the model
    spec = tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input")
    @tf.function(input_signature=[spec])
    def model_fn(x):
        return model(x, training=False)

    concrete_func = model_fn.get_concrete_function()
    onnx_model, _ = tf2onnx.convert.from_function(
        concrete_func,
        opset=opset_version,
        output_path=output_path,
    )
    print(f"ONNX model saved to {output_path}")


def export_to_savedmodel(model: TimestralModel, output_dir: str) -> None:
    """Export model to TensorFlow SavedModel format.

    Args:
        model: Trained TimestralModel instance.
        output_dir: Directory to save the SavedModel.
    """
    tf.saved_model.save(model, output_dir)
    print(f"SavedModel saved to {output_dir}")


def load_exported_model(path: str) -> TimestralModel:
    """Load a model exported via export_to_keras.

    Args:
        path: Path to the .keras file.

    Returns:
        Loaded TimestralModel instance.
    """
    return keras.models.load_model(path, custom_objects={"TimestralModel": TimestralModel})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Timestral model")
    parser.add_argument("checkpoint", help="Path to trained model checkpoint (.keras)")
    parser.add_argument("output", help="Output path")
    parser.add_argument("--format", choices=["keras", "onnx", "savedmodel"], default="keras")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    model = load_exported_model(args.checkpoint)

    if args.format == "keras":
        export_to_keras(model, args.output)
    elif args.format == "onnx":
        export_to_onnx(model, args.output, opset_version=args.opset)
    elif args.format == "savedmodel":
        export_to_savedmodel(model, args.output)
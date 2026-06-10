#!/usr/bin/env python3
"""
Standalone ONNX Export Script for PRISM Models.

This script exports a trained PRISM Keras model to ONNX format with the GPU
disabled, mirroring the tirex exporter (CudnnRNN / cuda-only ops cannot be
serialized portably). PRISM emits a single dense tensor (rank 3 in point mode
or rank 4 in quantile mode), so this exporter does NOT expose a --output-key
flag.

Usage:
    python -m train.time_series.prism.export --model_path results/experiment/best_model.keras
    python -m train.time_series.prism.export --model_path results/experiment/best_model.keras --output_path model.onnx --verify
"""

import os

# CRITICAL: Disable GPU BEFORE importing TensorFlow/Keras.
# This forces CPU-based implementations and avoids non-portable cuda ops.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import argparse
import numpy as np

import keras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PRISM Keras model to ONNX format (CPU-only)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the .keras model file"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output path for ONNX file. Defaults to <model_dir>/model.onnx."
    )
    parser.add_argument(
        "--opset_version", type=int, default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--input_length", type=int, default=None,
        help="Input sequence length (context_len). Auto-detected if not specified."
    )
    parser.add_argument(
        "--num_features", type=int, default=1,
        help="Number of input features (default: 1)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify ONNX output matches Keras output"
    )
    parser.add_argument(
        "--num_verify_samples", type=int, default=100,
        help="Number of samples for verification (default: 100)"
    )
    return parser.parse_args()


def detect_input_length(model: keras.Model) -> int:
    """Attempt to detect context_len from model config or input shape."""
    # PRISMModel stores context_len in get_config().
    config = model.get_config()
    if 'context_len' in config:
        return config['context_len']

    # Fallback: try the model's input shape.
    if hasattr(model, 'input_shape') and model.input_shape is not None:
        input_shape = model.input_shape
        if isinstance(input_shape, tuple) and len(input_shape) >= 2:
            if input_shape[1] is not None:
                return input_shape[1]

    print("Warning: Could not auto-detect context_len. Using default 168.")
    return 168


def export_to_onnx(
        model_path: str,
        output_path: str,
        opset_version: int = 17,
        input_length: int = None,
        num_features: int = 1
) -> str:
    """
    Export PRISM Keras model to ONNX format.

    Args:
        model_path: Path to .keras model file.
        output_path: Output path for ONNX file.
        opset_version: ONNX opset version.
        input_length: Input sequence length (auto-detected if None).
        num_features: Number of input features (default 1).

    Returns:
        Path to exported ONNX file.
    """
    print(f"Loading model from: {model_path}")
    model = keras.saving.load_model(model_path, compile=False)

    if input_length is None:
        input_length = detect_input_length(model)
    print(f"Using context_len: {input_length}, num_features: {num_features}")

    # Build with a concrete input shape (sets internal shapes for tracing).
    dummy_input = np.zeros((1, input_length, num_features), dtype=np.float32)
    _ = model(dummy_input, training=False)

    # Explicit input signature avoids symbolic-shape issues during tracing.
    input_signature = [
        keras.InputSpec(
            shape=(None, input_length, num_features),
            dtype="float32"
        )
    ]

    print(f"Exporting to ONNX (opset {opset_version}): {output_path}")
    model.export(
        output_path,
        format="onnx",
        input_signature=input_signature,
        opset_version=opset_version
    )

    print(f"ONNX export successful: {output_path}")
    return output_path


def verify_onnx(
        onnx_path: str,
        keras_model_path: str,
        input_length: int,
        num_features: int = 1,
        num_samples: int = 100
) -> bool:
    """
    Verify ONNX model outputs match Keras model.

    Handles both rank-3 (point) and rank-4 (quantile) PRISM outputs transparently
    via numpy.allclose on the single output tensor at rtol=atol=1e-4.

    Returns:
        True if outputs match within tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Skipping verification.")
        print("Install with: pip install onnxruntime")
        return False

    print(f"Verifying ONNX model: {onnx_path}")

    keras_model = keras.saving.load_model(keras_model_path, compile=False)
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )

    print(f"ONNX model inputs:")
    for inp in ort_session.get_inputs():
        print(f"  - {inp.name}: {inp.shape} ({inp.type})")

    print(f"ONNX model outputs:")
    for out in ort_session.get_outputs():
        print(f"  - {out.name}: {out.shape} ({out.type})")

    onnx_inputs = ort_session.get_inputs()
    output_name = ort_session.get_outputs()[0].name

    test_input = np.random.randn(num_samples, input_length, num_features).astype(np.float32)

    # Build feed dict: primary data input + any auxiliary shape tensors.
    input_feed = {onnx_inputs[0].name: test_input}
    for inp in onnx_inputs[1:]:
        inp_shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if inp.type == 'tensor(int32)':
            input_feed[inp.name] = np.zeros(inp_shape, dtype=np.int32)
        elif inp.type == 'tensor(int64)':
            input_feed[inp.name] = np.zeros(inp_shape, dtype=np.int64)
        else:
            input_feed[inp.name] = np.zeros(inp_shape, dtype=np.float32)
        print(f"  Feeding auxiliary input '{inp.name}': shape={inp_shape}")

    keras_preds = keras_model.predict(test_input, verbose=0)
    onnx_preds = ort_session.run([output_name], input_feed)[0]

    abs_diff = np.abs(keras_preds - onnx_preds)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    rtol, atol = 1e-4, 1e-4
    outputs_match = np.allclose(keras_preds, onnx_preds, rtol=rtol, atol=atol)

    print(f"Keras output shape: {keras_preds.shape}")
    print(f"ONNX output shape:  {onnx_preds.shape}")
    print(f"Max absolute diff:  {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    if outputs_match:
        print("VERIFICATION PASSED")
    else:
        print(f"VERIFICATION FAILED (tolerance: rtol={rtol}, atol={atol})")

    return outputs_match


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    if args.output_path is None:
        model_dir = os.path.dirname(args.model_path)
        args.output_path = os.path.join(model_dir, "model.onnx")

    try:
        onnx_path = export_to_onnx(
            model_path=args.model_path,
            output_path=args.output_path,
            opset_version=args.opset_version,
            input_length=args.input_length,
            num_features=args.num_features
        )
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if args.verify:
        input_length = args.input_length or detect_input_length(
            keras.saving.load_model(args.model_path, compile=False)
        )
        success = verify_onnx(
            onnx_path=onnx_path,
            keras_model_path=args.model_path,
            input_length=input_length,
            num_features=args.num_features,
            num_samples=args.num_verify_samples
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

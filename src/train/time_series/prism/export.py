#!/usr/bin/env python3
"""
Standalone ONNX Export Script for PRISM Models.

This script exports a trained PRISM Keras model to ONNX format with the GPU
disabled, mirroring the tirex exporter (CudnnRNN / cuda-only ops cannot be
serialized portably). PRISM emits a single dense tensor (rank 3 in point mode
or rank 4 in quantile mode), so this exporter does NOT expose a --output-key
flag.

The standard single-tensor-input export/verify core is delegated to the shared
``train.common.ts_export`` helper; this script keeps only the PRISM-specific
CLI surface, the ``main()`` orchestration, and the CPU pinning (which MUST be
set before importing TensorFlow/Keras and therefore stays here, not in the
shared module).

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

import keras

from dl_techniques.utils.logger import logger
from train.common.ts_export import (
    detect_input_length,
    export_standard_ts_model,
    verify_standard_ts_model,
)

# Fallback input length used when neither the model's input shape nor its
# ``context_len`` config key yields a value. Preserves the historical PRISM
# default of 168. NOTE: PRISMModel's get_config() key stays ``context_len``
# (model API is fixed); only the trainer-config field / CLI flag were renamed
# to input_length/prediction_length in Step 6 -- the model config key did NOT.
DEFAULT_INPUT_LENGTH = 168


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
        help="Input sequence length. Auto-detected if not specified."
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


def main():
    args = parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error("Model file not found: %s", args.model_path)
        sys.exit(1)

    # Determine output path
    if args.output_path is None:
        model_dir = os.path.dirname(args.model_path)
        args.output_path = os.path.join(model_dir, "model.onnx")

    # Resolve input length: explicit flag, else detect from the loaded model
    # (input shape, then the 'context_len' config key -- the PRISMModel API key
    # is unchanged -- then the PRISM default).
    input_length = args.input_length
    if input_length is None:
        model = keras.saving.load_model(args.model_path, compile=False)
        input_length = detect_input_length(
            model,
            config_keys=['context_len'],
            default=DEFAULT_INPUT_LENGTH,
        )

    # Export
    try:
        onnx_path = export_standard_ts_model(
            model_path=args.model_path,
            output_path=args.output_path,
            opset_version=args.opset_version,
            input_length=input_length,
            num_features=args.num_features,
        )
    except Exception as e:
        logger.error("Export failed: %s", e, exc_info=True)
        sys.exit(1)

    # Verify if requested
    if args.verify:
        success = verify_standard_ts_model(
            model_path=args.model_path,
            onnx_path=onnx_path,
            input_length=input_length,
            num_features=args.num_features,
            num_samples=args.num_verify_samples,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

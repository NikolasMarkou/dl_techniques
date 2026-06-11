#!/usr/bin/env python3
"""
Standalone ONNX Export Script for xLSTM Forecaster Models.

This script exports a trained Keras model to ONNX format with GPU disabled
to avoid CudnnRNN ops that are not portable to ONNX runtime.

The standard single-tensor-input export/verify core is delegated to the shared
``train.common.ts_export`` helper; this script keeps only the xLSTM-specific
CLI surface, the ``main()`` orchestration, and the CPU pinning (which MUST be
set before importing TensorFlow/Keras and therefore stays here, not in the
shared module).

The xLSTM forecaster may emit a quantile output of shape ``[B, H, Q]`` when the
quantile head is enabled. The single-tensor forward export still works; the
``--verify`` step simply compares Keras and ONNX outputs element-wise via
``np.allclose`` regardless of the output rank.

Usage:
    python export.py --model_path results/experiment/best_model.keras --output_path model.onnx
    python export.py --model_path results/experiment/best_model.keras  # outputs to same dir
"""

import os

# CRITICAL: Disable GPU BEFORE importing TensorFlow/Keras
# This forces CPU-based LSTM implementation instead of CudnnRNNV3
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import argparse

import keras

from dl_techniques.utils.logger import logger
# Importing the forecaster registers its custom objects so
# ``keras.saving.load_model`` can resolve the serialized model.
from dl_techniques.models.time_series.xlstm.forecaster import xLSTMForecaster  # noqa: F401
from train.common.ts_export import (
    detect_input_length,
    export_standard_ts_model,
    verify_standard_ts_model,
)

# Fallback input length used when neither the model's input shape nor its
# ``input_length`` config key yields a value. Mirrors the xLSTM trainer's
# default context window of 168.
DEFAULT_INPUT_LENGTH = 168


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export xLSTM Forecaster Keras model to ONNX format (CPU-only to avoid CudnnRNN)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the .keras model file"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output path for ONNX file. Defaults to model directory."
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
    # (input shape, then the 'input_length' config key, then the xLSTM default).
    input_length = args.input_length
    if input_length is None:
        model = keras.saving.load_model(args.model_path, compile=False)
        input_length = detect_input_length(
            model,
            config_keys=['input_length'],
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

    # Verify if requested.
    # DECISION plan_2026-06-11_84296249/D-003: the forward export above succeeds,
    # but onnxruntime may REJECT the produced graph for xLSTM models whose mLSTM
    # blocks contain a causal conv1d — tf2onnx 1.16.1 leaves it as an unconvertible
    # `StatefulPartitionedCall` (clamped to opset 15). Do NOT "fix" this in export.py
    # by post-processing the graph or downgrading the helper: the shared ts_export
    # core is correct (tirex round-trips allclose with it). This is a model/toolchain
    # constraint, kept model-specific per INV-8. See decisions.md D-003.
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

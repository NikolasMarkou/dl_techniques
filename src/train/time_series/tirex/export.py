#!/usr/bin/env python3
"""
Standalone ONNX Export Script for TiRex Models.

This script exports a trained Keras model to ONNX format with GPU disabled
to avoid CudnnRNN ops that are not portable to ONNX runtime.

The standard single-tensor-input export/verify core is delegated to the shared
``train.common.ts_export`` helper; this script keeps only the TiRex-specific
CLI surface, the ``main()`` orchestration, and the CPU pinning (which MUST be
set before importing TensorFlow/Keras and therefore stays here, not in the
shared module).

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
from train.common.ts_export import (
    create_ts_export_argument_parser,
    detect_input_length,
    export_standard_ts_model,
    verify_standard_ts_model,
)

# Fallback input length used when neither the model's input shape nor its
# ``input_length`` config key yields a value. Preserves the historical tirex
# default of 256.
DEFAULT_INPUT_LENGTH = 256


def parse_args() -> argparse.Namespace:
    parser = create_ts_export_argument_parser(
        description="Export TiRex Keras model to ONNX format (CPU-only to avoid CudnnRNN)"
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
    # (input shape, then the 'input_length' config key, then the tirex default).
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

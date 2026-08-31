#!/usr/bin/env python3
"""
Standalone ONNX Export Script for ETS Models.

Mirrors the prism/tirex exporters: the GPU is disabled before TensorFlow is
imported, and the standard single-tensor-input export/verify core is delegated
to ``train.common.ts_export``. ``ETSModel`` emits a single dense rank-3 tensor
``[B, H, 1]`` in every variant, so there is no ``--output-key`` flag.

One ETS-specific caveat: the model's forward pass is a ``keras.ops.scan`` over
the context window. Export therefore requires a STATIC input length, which is
also what the model itself requires (the initial-state derivation reads it).

Usage:
    python -m train.time_series.ets.export --model_path results/experiment/best_model.keras
    python -m train.time_series.ets.export --model_path results/experiment/best_model.keras --output_path model.onnx --verify
"""

import os

# CRITICAL: Disable GPU BEFORE importing TensorFlow/Keras.
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

# Fallback input length, used only when neither the model's input shape nor a
# recognised config key yields one. Matches the trainer's --input_length default.
DEFAULT_INPUT_LENGTH = 168


def parse_args() -> argparse.Namespace:
    parser = create_ts_export_argument_parser(
        description="Export ETS Keras model to ONNX format (CPU-only)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        logger.error("Model file not found: %s", args.model_path)
        sys.exit(1)

    if args.output_path is None:
        model_dir = os.path.dirname(args.model_path)
        args.output_path = os.path.join(model_dir, "model.onnx")

    # ETSModel.get_config() carries no context-length key -- the context length
    # is a property of the DATA, not of the model -- so detection falls through
    # to the model's built input shape, then to the default.
    input_length = args.input_length
    if input_length is None:
        model = keras.saving.load_model(args.model_path, compile=False)
        input_length = detect_input_length(
            model,
            config_keys=[],
            default=DEFAULT_INPUT_LENGTH,
        )

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

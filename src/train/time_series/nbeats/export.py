#!/usr/bin/env python3
"""
Standalone ONNX Export Script for N-BEATS Models.

This script exports a trained N-BEATS Keras model to ONNX format with the GPU
disabled (CPU-only) to avoid non-portable device-specific ops during tracing.

The standard single-tensor-input export/verify core is delegated to the shared
``train.common.ts_export`` helper; this script keeps only the N-BEATS-specific
CLI surface, the custom-object registration needed for ``load_model``, the
``main()`` orchestration, and the CPU pinning (which MUST be set before importing
TensorFlow/Keras and therefore stays here, not in the shared module).

N-BEATS input window naming: the model's input window is the ``backcast_length``
(N-BEATS paper terminology). For cross-trainer export-CLI uniformity the flag is
exposed as ``--input_length``, but auto-detection reads the model's actual config
key ``backcast_length``.

Usage:
    python export.py --model_path results/experiment/best_model.keras --output_path model.onnx
    python export.py --model_path results/experiment/best_model.keras  # outputs to same dir
"""

import os

# CRITICAL: Disable GPU BEFORE importing TensorFlow/Keras so the export trace
# uses portable CPU ops only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import argparse

import keras

# Importing the N-BEATS models registers their @keras.saving.register_keras_serializable
# classes so keras.saving.load_model can resolve NBeatsNet / NBeatsXNet from a
# saved .keras artifact. NBeatsNet is the model the standard trainer saves on the
# default reconstruction-on path; NBeatsXNet is imported too in case an exogenous
# variant is trained and saved.
#
# DECISION plan_2026-06-11_84296249/D-006
# NBeatsNet round-trips load + ONNX export cleanly, but `--verify` does NOT pass:
# on the default reconstruction-on path NBeatsNet.call() returns a 2-OUTPUT tuple
# (forecast [B,H,1], backcast-residual [B,C]), and the shared single-primary-output
# verify_standard_ts_model() cannot diff a multi-output model (keras_preds is a
# list of 2 inhomogeneous arrays). The EXPORT itself succeeds. Do NOT special-case
# multi-output diffing into the shared single-tensor verifier (pollutes the proven
# tirex/prism single-output pattern), and do NOT switch the trainer to a
# forecast-only save just to make verify pass (that is a trainer/model concern,
# not an export-CLI concern). Treat as a model-specific export-verify divergence,
# same non-gating class as D-003 (xlstm tf2onnx gap) / D-004 (mdn serialization).
# See decisions.md D-006.
from dl_techniques.models.time_series.nbeats import (  # noqa: F401
    NBeatsNet,
    NBeatsXNet,
)

from dl_techniques.utils.logger import logger
from train.common.ts_export import (
    create_ts_export_argument_parser,
    detect_input_length,
    export_standard_ts_model,
    verify_standard_ts_model,
)

# Fallback input length used when neither the model's input shape nor its
# ``backcast_length`` config key yields a value. Matches the N-BEATS training
# config default backcast_length of 168.
DEFAULT_INPUT_LENGTH = 168

# The N-BEATS model stores its input window length under this get_config() key
# (NBeatsNet.get_config() -> 'backcast_length'). detect_input_length tries the
# model input shape first, then this key.
NBEATS_LENGTH_CONFIG_KEY = "backcast_length"


def parse_args() -> argparse.Namespace:
    # NOTE: the shared parser's --input_length help reads "Input sequence
    # length"; for N-BEATS this length is the model's backcast_length (paper
    # terminology). Auto-detection reads the 'backcast_length' config key (see
    # NBEATS_LENGTH_CONFIG_KEY) and the module docstring documents this.
    parser = create_ts_export_argument_parser(
        description="Export N-BEATS Keras model to ONNX format (CPU-only)"
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
    # (input shape, then the 'backcast_length' config key, then the default).
    input_length = args.input_length
    if input_length is None:
        model = keras.saving.load_model(args.model_path, compile=False)
        input_length = detect_input_length(
            model,
            config_keys=[NBEATS_LENGTH_CONFIG_KEY],
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

#!/usr/bin/env python3
"""
Standalone ONNX Export Script for Multi-Task MDN Models.

This script exports a trained Keras model to ONNX format with GPU disabled
to avoid CudnnRNN / device-specific ops that are not portable to ONNX runtime.

The standard single-tensor-input export/verify core is delegated to the shared
``train.common.ts_export`` helper; this script keeps only the MDN-specific
CLI surface, the custom-object registration imports, the ``main()``
orchestration, and the CPU pinning (which MUST be set before importing
TensorFlow/Keras and therefore stays here, not in the shared module).

NOTE on the MDN output: the Multi-Task MDN emits a flat mixture-parameter vector
(the packed ``[mu, sigma, pi]`` for ``num_mixtures`` Gaussians over the H-step
horizon), NOT a forecast tensor. The single-tensor forward export core itself is
correct (the shared ``ts_export`` helper round-trips allclose for tirex).

# DECISION plan_2026-06-11_84296249/D-004: this script's STRUCTURE is the
# deliverable -- it registers the MDN custom objects, loads the model, and
# delegates to the proven shared helper. It cannot complete an end-to-end export
# for the CURRENT saved artifact because ``MultiTaskMDNModel`` (a BESPOKE trainer
# model, defined in train_mdn.py, not a library model) ships NO get_config/
# from_config: its __init__ requires positional ``num_tasks`` + ``config``, so
# ``keras.saving.load_model`` cannot revive it (TypeError before any export op).
# Do NOT "fix" this by adding get_config to MultiTaskMDNModel here -- that is a
# MODEL change, out of scope for this trainers-only plan (INV: no model edits),
# and additionally the MDN consumes a 2-tuple ``(sequence, task_ids)`` input that
# the single-tensor ts_export InputSpec does not model. Same model-specific
# export-divergence class as xLSTM's tf2onnx gap (D-003), kept model-specific per
# INV-8. The export wrapper is left ready for when the model gains a serialization
# contract. See decisions.md D-004.

Usage:
    python export.py --model_path results/experiment/best_model.keras --output_path model.onnx
    python export.py --model_path results/experiment/best_model.keras  # outputs to same dir
"""

import os

# CRITICAL: Disable GPU BEFORE importing TensorFlow/Keras
# This forces CPU-based implementations instead of device-specific ops.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import argparse

import keras

from dl_techniques.utils.logger import logger
# Importing these registers the MDN custom objects so
# ``keras.saving.load_model`` can resolve the serialized model. The saved model
# is a ``MultiTaskMDNModel`` (defined in the trainer module) wrapping an
# ``MDNModel`` core whose head is an ``MDNLayer``; all three must be importable.
from dl_techniques.models.time_series.mdn import MDNModel  # noqa: F401
from dl_techniques.layers.statistics.mdn_layer import MDNLayer  # noqa: F401
from train.time_series.mdn.train_mdn import MultiTaskMDNModel  # noqa: F401
from train.common.ts_export import (
    detect_input_length,
    export_standard_ts_model,
    verify_standard_ts_model,
)

# Fallback input length used when neither the model's input shape nor its
# ``input_length`` config key yields a value. Mirrors the MDN trainer's default
# context window of 120.
DEFAULT_INPUT_LENGTH = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Multi-Task MDN Keras model to ONNX format (CPU-only)"
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
    # (input shape, then the 'input_length' config key, then the MDN default).
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

    # Verify if requested. See the module-level D-004 anchor: for the current
    # bespoke MultiTaskMDNModel artifact, execution stops at load_model (missing
    # get_config), so this block is only reached once the model gains a
    # serialization contract. Do NOT post-process the graph or fork the shared
    # helper -- the ts_export core is proven (tirex round-trips allclose).
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

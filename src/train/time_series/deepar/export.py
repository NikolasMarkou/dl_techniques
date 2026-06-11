#!/usr/bin/env python3
"""
Standalone ONNX Export Script for DeepAR Models.

This script exports a trained Keras model to ONNX format with GPU disabled
to avoid CudnnRNN ops that are not portable to ONNX runtime. DeepAR is an
LSTM-based recurrent forecaster, so the same CPU-only export rationale that
applies to TiRex applies here (force the portable CPU LSTM kernel instead of
CudnnRNNV3). Unlike `keras.ops.scan`-based models (e.g. adaptive_ema), DeepAR's
LSTMs trace cleanly to ONNX -- NO scan-unroll workaround is needed.

Relation to the shared helper
------------------------------
The STANDARD single-tensor export core lives in ``train.common.ts_export``
(used by tirex / prism / mdn / nbeats / xlstm). DeepAR consumes a DICT input
(``{'target','covariates'}``), so it CANNOT use
``export_standard_ts_model`` / ``verify_standard_ts_model`` (single-tensor
``(batch, input_length, num_features)`` signatures) without contortion. Per
INV-8 the dict-``input_signature`` export and the dict-input verify bodies stay
model-specific here. The genuinely shared piece -- recovering the window length
from a loaded model -- is reused via :func:`detect_input_length`.

What is exported
----------------
The trained `.keras` checkpoint produced by `train_deepar.py` is a
`DeepARTrainingWrapper` (the model `ModelCheckpoint` saves). Its `call` runs the
base DeepAR in TRAINING mode (teacher-forced LSTM forward) and returns the `mu`
likelihood parameter. THIS is what gets exported: the deterministic,
dict-in / tensor-out training forward pass.

LIMITATION (documented): DeepAR's probabilistic forecast is produced by an
autoregressive Monte-Carlo SAMPLING loop (`DeepAR._prediction_mode`,
`model.py:418-498`) -- an eager Python loop over `num_samples x prediction_len`
that is NOT a static graph and is NOT ONNX-traceable. This script therefore does
NOT export the sampling/prediction path; it exports the training-mode forward
(the same forward `model.evaluate` uses). For probabilistic inference, run the
Keras model's `predict_forecast` directly. This mirrors TiRex's export, which
also targets the model's forward rather than any iterative decode.

Input signature
----------------
DeepAR / `DeepARTrainingWrapper` consume a DICT, not a bare tensor:
    {'target': (B, T, target_dim), 'covariates': (B, T, covariate_dim)}
where `T = input_length + prediction_length` (the teacher-forced window). The
input signature below reflects that dict contract.

Usage:
    python -m train.time_series.deepar.export --model_path results/experiment/best_model.keras --output_path model.onnx
    python -m train.time_series.deepar.export --model_path results/experiment/best_model.keras  # outputs to same dir
"""

import os

# CRITICAL: Disable GPU BEFORE importing TensorFlow/Keras
# This forces CPU-based LSTM implementation instead of CudnnRNNV3
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import argparse
import numpy as np

import keras

from dl_techniques.utils.logger import logger

# Shared: the window-length detection is the standard piece DeepAR can reuse
# (the dict-input export/verify below are model-specific per INV-8).
from train.common.ts_export import detect_input_length

# Importing the trainer module registers DeepAR + DeepARTrainingWrapper (and the
# DeepAR blocks) as Keras serializables, so `keras.saving.load_model` can resolve
# the wrapper checkpoint without an explicit `custom_objects` map.
from train.time_series.deepar.train_deepar import DeepARTrainingWrapper  # noqa: F401
from dl_techniques.models.time_series.deepar.model import DeepAR  # noqa: F401

# Fallback teacher-forced window length T used when neither the model's input
# shape nor its config keys yield a value.
DEFAULT_INPUT_LENGTH = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DeepAR Keras model to ONNX format "
                    "(CPU-only to avoid CudnnRNN; exports the training-mode "
                    "forward, NOT the autoregressive sampling loop)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the .keras model file (a saved DeepARTrainingWrapper)"
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
        help="Teacher-forced window length T (= input_length + prediction_length). "
             "Auto-detected if not specified."
    )
    parser.add_argument(
        "--target_dim", type=int, default=1,
        help="Target dimensionality (default: 1)"
    )
    parser.add_argument(
        "--covariate_dim", type=int, default=4,
        help="Number of covariate features (default: 4)"
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


def _unwrap_base(model: keras.Model) -> keras.Model:
    """Return the underlying DeepAR base if `model` is a training wrapper."""
    base = getattr(model, "base", None)
    if isinstance(base, DeepAR):
        return base
    return model


def detect_dims(model: keras.Model) -> int:
    """Attempt to detect the target_dim from the model/base config."""
    base = _unwrap_base(model)
    config = base.get_config()
    if 'target_dim' in config:
        return int(config['target_dim'])
    logger.warning("Could not auto-detect target_dim. Using default 1.")
    return 1


def export_to_onnx(
        model_path: str,
        output_path: str,
        opset_version: int = 17,
        input_length: int = None,
        target_dim: int = 1,
        covariate_dim: int = 4
) -> str:
    """
    Export a saved DeepAR (training-wrapper) model to ONNX format.

    Model-specific (INV-8): builds an explicit DICT input signature
    (``{'target','covariates'}``) rather than the shared single-tensor signature.
    The window length, when not supplied, is recovered via the shared
    :func:`detect_input_length` helper.

    Args:
        model_path: Path to .keras model file (a DeepARTrainingWrapper).
        output_path: Output path for ONNX file.
        opset_version: ONNX opset version.
        input_length: Teacher-forced window length T (auto-detected if None).
        target_dim: Target dimensionality (auto-detected from config if 1).
        covariate_dim: Number of covariate features.

    Returns:
        Path to exported ONNX file.
    """
    logger.info("Loading model from: %s", model_path)
    model = keras.saving.load_model(model_path, compile=False)

    # Detect dims where possible.
    detected_target_dim = detect_dims(model)
    if target_dim == 1 and detected_target_dim != 1:
        target_dim = detected_target_dim

    if input_length is None:
        # Shared length detection: probe input shape, then DeepAR's config keys
        # (the teacher-forced window has no single config key, so input_length +
        # prediction_length is the meaningful pairing; fall back to the default).
        input_length = detect_input_length(
            model,
            config_keys=['input_length'],
            default=DEFAULT_INPUT_LENGTH,
        )
    logger.info(
        "Using window T=%d, target_dim=%d, covariate_dim=%d",
        input_length, target_dim, covariate_dim,
    )

    # Build model with concrete dict input shape (training-mode forward).
    dummy_input = {
        "target": np.zeros((1, input_length, target_dim), dtype=np.float32),
        "covariates": np.zeros((1, input_length, covariate_dim), dtype=np.float32),
    }
    _ = model(dummy_input, training=False)

    # Explicit dict input signature so dynamic shapes resolve during tracing.
    input_signature = [
        {
            "target": keras.InputSpec(
                shape=(None, input_length, target_dim), dtype="float32"
            ),
            "covariates": keras.InputSpec(
                shape=(None, input_length, covariate_dim), dtype="float32"
            ),
        }
    ]

    logger.info("Exporting to ONNX (opset %d): %s", opset_version, output_path)
    model.export(
        output_path,
        format="onnx",
        input_signature=input_signature,
        opset_version=opset_version
    )

    logger.info("ONNX export successful: %s", output_path)
    return output_path


def verify_onnx(
        onnx_path: str,
        keras_model_path: str,
        input_length: int,
        target_dim: int = 1,
        covariate_dim: int = 4,
        num_samples: int = 100
) -> bool:
    """
    Verify ONNX model outputs match the Keras training-mode forward.

    Model-specific (INV-8): feeds the DICT input contract
    (``{'target','covariates'}``) and maps ONNX inputs to the right tensor by
    trailing dimension. The shared single-tensor verifier cannot serve this case.

    Returns:
        True if outputs match within tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnxruntime not installed. Skipping verification. "
            "Install with: pip install onnxruntime"
        )
        return False

    logger.info("Verifying ONNX model: %s", onnx_path)

    keras_model = keras.saving.load_model(keras_model_path, compile=False)
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )

    logger.info("ONNX model inputs:")
    for inp in ort_session.get_inputs():
        logger.info("  - %s: %s (%s)", inp.name, inp.shape, inp.type)

    logger.info("ONNX model outputs:")
    for out in ort_session.get_outputs():
        logger.info("  - %s: %s (%s)", out.name, out.shape, out.type)

    onnx_inputs = ort_session.get_inputs()
    output_name = ort_session.get_outputs()[0].name

    # Generate test data for the dict inputs.
    test_target = np.random.randn(
        num_samples, input_length, target_dim
    ).astype(np.float32)
    test_cov = np.random.randn(
        num_samples, input_length, covariate_dim
    ).astype(np.float32)
    keras_feed = {"target": test_target, "covariates": test_cov}

    # Map by trailing dimension to the right ONNX input name.
    input_feed = {}
    for inp in onnx_inputs:
        last_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else None
        if last_dim == target_dim:
            input_feed[inp.name] = test_target
        elif last_dim == covariate_dim:
            input_feed[inp.name] = test_cov
        else:
            # Auxiliary shape tensors used by Reshape ops.
            inp_shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            if inp.type == 'tensor(int32)':
                input_feed[inp.name] = np.zeros(inp_shape, dtype=np.int32)
            elif inp.type == 'tensor(int64)':
                input_feed[inp.name] = np.zeros(inp_shape, dtype=np.int64)
            else:
                input_feed[inp.name] = np.zeros(inp_shape, dtype=np.float32)
            logger.info(
                "  Feeding auxiliary input '%s': shape=%s", inp.name, inp_shape
            )

    keras_preds = keras_model.predict(keras_feed, verbose=0)
    onnx_preds = ort_session.run([output_name], input_feed)[0]

    abs_diff = np.abs(keras_preds - onnx_preds)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    rtol, atol = 1e-4, 1e-4
    outputs_match = np.allclose(keras_preds, onnx_preds, rtol=rtol, atol=atol)

    logger.info("Keras output shape: %s", keras_preds.shape)
    logger.info("ONNX output shape:  %s", onnx_preds.shape)
    logger.info("Max absolute diff:  %.2e", max_diff)
    logger.info("Mean absolute diff: %.2e", mean_diff)

    if outputs_match:
        logger.info("VERIFICATION PASSED")
    else:
        logger.error(
            "VERIFICATION FAILED (tolerance: rtol=%s, atol=%s)", rtol, atol
        )

    return outputs_match


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        logger.error("Model file not found: %s", args.model_path)
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
            target_dim=args.target_dim,
            covariate_dim=args.covariate_dim
        )
    except Exception as e:
        logger.error("Export failed: %s", e, exc_info=True)
        sys.exit(1)

    if args.verify:
        input_length = args.input_length or DEFAULT_INPUT_LENGTH
        success = verify_onnx(
            onnx_path=onnx_path,
            keras_model_path=args.model_path,
            input_length=input_length,
            target_dim=args.target_dim,
            covariate_dim=args.covariate_dim,
            num_samples=args.num_verify_samples
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

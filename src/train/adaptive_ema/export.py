#!/usr/bin/env python3
"""
Standalone ONNX Export Script for AdaptiveEMA Models.

Mirrors ``src/train/tirex/export.py``: forces CPU before importing TF/Keras,
optional numpy round-trip verification against the Keras model.

The AdaptiveEMA model emits a dict; ONNX export captures every key, but the
verification path needs a single tensor — pass ``--output-key`` to pick
which one to compare (default: the first ONNX output).

Usage:
    python -m train.adaptive_ema.export \\
        --model_path results/adaptive_ema_xxx/best_model.keras \\
        --opset_version 17 --verify --output-key slope_quantiles
"""

import os

# CRITICAL: disable GPU BEFORE importing TF/Keras to keep ONNX export portable.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import argparse
from typing import Optional

import numpy as np
import keras

from dl_techniques.utils.logger import logger

# Trigger registration of the two custom classes used by saved checkpoints.
# Without these imports `keras.saving.load_model` cannot resolve
# `AdaptiveEMATrainingWrapper` / `AdaptiveEMASlopeFilterModel` from the
# `Custom>` namespace embedded in the .keras config.
from dl_techniques.models.time_series.adaptive_ema import AdaptiveEMASlopeFilterModel  # noqa: F401
from train.adaptive_ema.train_adaptive_ema import AdaptiveEMATrainingWrapper  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export AdaptiveEMA Keras model to ONNX (CPU-only)."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the .keras model file."
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output path for the ONNX file. Defaults to model directory."
    )
    parser.add_argument(
        "--opset_version", type=int, default=17,
        help="ONNX opset version (default: 17)."
    )
    parser.add_argument(
        "--input_length", type=int, default=None,
        help="Input sequence length. Auto-detected if not specified."
    )
    parser.add_argument(
        "--num_features", type=int, default=1,
        help="Number of input features (default: 1)."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify a single ONNX output against the Keras model."
    )
    parser.add_argument(
        "--output-key", dest="output_key", type=str, default=None,
        help=(
            "Name of the dict output (or ONNX output) to verify. "
            "If omitted, the first ONNX output is used."
        ),
    )
    parser.add_argument(
        "--num_verify_samples", type=int, default=64,
        help="Number of samples to use during verification (default: 64)."
    )
    return parser.parse_args()


def detect_input_length(model: keras.Model) -> int:
    """Best-effort detection of input length from the model."""
    if hasattr(model, "input_shape") and model.input_shape is not None:
        input_shape = model.input_shape
        if isinstance(input_shape, tuple) and len(input_shape) >= 2:
            if input_shape[1] is not None:
                return int(input_shape[1])
    config = model.get_config()
    if "input_length" in config:
        return int(config["input_length"])
    if "base" in config:
        base_cfg = config["base"]
        if isinstance(base_cfg, dict) and "input_length" in base_cfg:
            return int(base_cfg["input_length"])
    logger.warning("Could not auto-detect input_length; defaulting to 128.")
    return 128


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 17,
    input_length: Optional[int] = None,
    num_features: int = 1,
) -> str:
    logger.info(f"Loading model from: {model_path}")
    model = keras.saving.load_model(model_path, compile=False)

    if input_length is None:
        input_length = detect_input_length(model)
    logger.info(
        f"Using input_length={input_length}, num_features={num_features}"
    )

    dummy_input = np.zeros(
        (1, input_length, num_features), dtype=np.float32
    )
    _ = model(dummy_input, training=False)

    input_signature = [
        keras.InputSpec(
            shape=(None, input_length, num_features), dtype="float32",
        )
    ]

    logger.info(
        f"Exporting to ONNX (opset {opset_version}): {output_path}"
    )
    # tf2onnx 1.16.1 cannot convert the `tf.while_loop` produced by
    # `keras.ops.scan` (raises `wire_while_body: couldn't find scan
    # output index for nodes`). Because we know the input length is
    # static at export time, we monkey-patch the EMA layer's `call`
    # with a Python-unrolled loop variant for the duration of
    # `model.export()` only — the saved Keras model on disk and the
    # in-memory model after export are unchanged.
    with _ema_unrolled_for_export(input_length):
        model.export(
            output_path,
            format="onnx",
            input_signature=input_signature,
            opset_version=opset_version,
        )
    logger.info(f"ONNX export successful: {output_path}")
    return output_path


from contextlib import contextmanager


@contextmanager
def _ema_unrolled_for_export(input_length: int):
    """Temporarily replace ``ExponentialMovingAverage.call`` with a
    Python-unrolled variant so that ``tf2onnx`` sees a finite sequence
    of elementwise ops instead of a `tf.scan` While loop it can't
    translate.
    """
    from dl_techniques.layers.time_series.ema_layer import ExponentialMovingAverage
    from keras import ops as _ops

    original_call = ExponentialMovingAverage.call

    def unrolled_call(self, inputs):
        ndim = len(inputs.shape)
        x = _ops.expand_dims(inputs, axis=-1) if ndim == 2 else inputs
        T = x.shape[1] if x.shape[1] is not None else input_length
        if T == 1:
            return inputs
        alpha = _ops.cast(self.alpha, dtype=x.dtype)
        oma = _ops.cast(1.0 - self.alpha, dtype=x.dtype)
        ema_prev = x[:, 0, :]
        ema_values = [ema_prev]
        for t in range(1, T):
            cur = alpha * x[:, t, :] + oma * ema_prev
            if self.adjust:
                w = _ops.cast(
                    1.0 - _ops.power(oma, _ops.cast(t + 1, x.dtype)),
                    dtype=x.dtype,
                )
                cur = cur / _ops.maximum(w, 1e-10)
            ema_values.append(cur)
            ema_prev = cur
        ema = _ops.stack(ema_values, axis=1)
        if ndim == 2:
            ema = _ops.squeeze(ema, axis=-1)
        return ema

    ExponentialMovingAverage.call = unrolled_call
    try:
        yield
    finally:
        ExponentialMovingAverage.call = original_call


def _select_keras_output(
    keras_outputs, output_key: Optional[str]
) -> np.ndarray:
    """Coerce Keras dict / tuple / tensor output to a single numpy array."""
    if isinstance(keras_outputs, dict):
        if output_key and output_key in keras_outputs:
            arr = keras_outputs[output_key]
        else:
            arr = next(iter(keras_outputs.values()))
    elif isinstance(keras_outputs, (list, tuple)):
        arr = keras_outputs[0]
    else:
        arr = keras_outputs
    return np.asarray(arr)


def verify_onnx(
    onnx_path: str,
    keras_model_path: str,
    input_length: int,
    num_features: int = 1,
    num_samples: int = 64,
    output_key: Optional[str] = None,
) -> bool:
    """Compare one ONNX output head against the Keras dict output."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnxruntime not installed; skipping verification. "
            "Install with: pip install onnxruntime"
        )
        return False

    logger.info(f"Verifying ONNX model: {onnx_path}")
    keras_model = keras.saving.load_model(keras_model_path, compile=False)
    ort_session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    logger.info("ONNX model inputs:")
    for inp in ort_session.get_inputs():
        logger.info(f"  - {inp.name}: {inp.shape} ({inp.type})")
    logger.info("ONNX model outputs:")
    onnx_output_names = [out.name for out in ort_session.get_outputs()]
    for name in onnx_output_names:
        logger.info(f"  - {name}")

    onnx_inputs = ort_session.get_inputs()
    if output_key and output_key in onnx_output_names:
        onnx_output_name = output_key
    else:
        onnx_output_name = onnx_output_names[0]
        if output_key:
            logger.warning(
                f"output_key='{output_key}' not in ONNX outputs; "
                f"falling back to '{onnx_output_name}'."
            )

    test_input = np.random.randn(
        num_samples, input_length, num_features
    ).astype(np.float32)

    input_feed = {onnx_inputs[0].name: test_input}
    for inp in onnx_inputs[1:]:
        inp_shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if inp.type == "tensor(int32)":
            input_feed[inp.name] = np.zeros(inp_shape, dtype=np.int32)
        elif inp.type == "tensor(int64)":
            input_feed[inp.name] = np.zeros(inp_shape, dtype=np.int64)
        else:
            input_feed[inp.name] = np.zeros(inp_shape, dtype=np.float32)
        logger.info(
            f"  Feeding auxiliary input '{inp.name}': shape={inp_shape}"
        )

    # The ONNX graph encodes the Python-unrolled EMA (see
    # `_ema_unrolled_for_export` — required because tf2onnx 1.16.1 cannot
    # convert `tf.scan`). For an apples-to-apples comparison, run the
    # Keras side through the SAME unrolled path; otherwise the
    # scan-vs-unrolled float32 chaos for adjust=True (documented in
    # plan_2026-05-12_5f0e087c/decisions.md D-003) would dominate.
    with _ema_unrolled_for_export(input_length):
        keras_preds_raw = keras_model.predict(test_input, verbose=0)
    keras_preds = _select_keras_output(keras_preds_raw, output_key)
    onnx_preds = ort_session.run([onnx_output_name], input_feed)[0]

    if keras_preds.shape != onnx_preds.shape:
        logger.warning(
            f"Shape mismatch: keras={keras_preds.shape}, onnx={onnx_preds.shape}. "
            "Verification cannot proceed."
        )
        return False

    abs_diff = np.abs(keras_preds - onnx_preds)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    rtol, atol = 1e-4, 1e-4
    outputs_match = np.allclose(keras_preds, onnx_preds, rtol=rtol, atol=atol)

    logger.info(f"Keras output shape: {keras_preds.shape}")
    logger.info(f"ONNX  output shape: {onnx_preds.shape}")
    logger.info(f"Max absolute diff:  {max_diff:.2e}")
    logger.info(f"Mean absolute diff: {mean_diff:.2e}")
    if outputs_match:
        logger.info("VERIFICATION PASSED")
    else:
        logger.error(
            f"VERIFICATION FAILED (tolerance: rtol={rtol}, atol={atol})"
        )
    return outputs_match


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
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
            num_features=args.num_features,
        )
    except Exception as exc:
        logger.error(f"Export failed: {exc}", exc_info=True)
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
            num_samples=args.num_verify_samples,
            output_key=args.output_key,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

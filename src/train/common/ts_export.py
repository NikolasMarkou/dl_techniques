#!/usr/bin/env python3
"""
Shared ONNX export helpers for standard single-tensor-input time-series models.

This module factors the ~95%-identical export core shared by the per-trainer
``export.py`` scripts (``tirex``, ``prism``, and the soon-to-be-added ``mdn`` /
``nbeats`` / ``xlstm``). All three steps of the standard export flow are exposed
as standalone functions:

  - :func:`detect_input_length` — recover the context/window length from a loaded
    model (input shape + a caller-supplied ordered list of config keys).
  - :func:`export_standard_ts_model` — load a ``.keras`` model, run a dummy
    forward to materialize shapes, and ``model.export`` it to ONNX with an
    explicit single-tensor ``(batch, input_length, num_features)`` input
    signature.
  - :func:`verify_standard_ts_model` — feed a single random input tensor through
    both Keras and ONNX Runtime and compare with :func:`numpy.allclose`.

Scope: the STANDARD single-tensor-input case only. Model-specific divergences
stay in the per-trainer ``export.py``:
  - ``deepar`` uses a dict ``input_signature`` (``{'target','covariates'}``);
  - ``adaptive_ema`` exports a scan-unrolled graph (``_ema_unrolled_for_export``)
    because tf2onnx cannot convert ``keras.ops.scan``.
Those scripts reuse only the pieces of this module that fit (e.g.
:func:`detect_input_length`, :func:`verify_standard_ts_model`'s comparison
logic where applicable).

Design notes:
  - This is a shared library: it does NOT mutate global process state on import.
    In particular it does NOT set ``CUDA_VISIBLE_DEVICES``. CPU-pinning (needed
    to avoid non-portable CudnnRNN ops) remains the caller's ``export.py``
    responsibility — those scripts set ``os.environ["CUDA_VISIBLE_DEVICES"]=""``
    BEFORE importing TensorFlow/Keras, which a shared module imported later
    cannot do anyway.
  - ``onnxruntime`` is an optional dependency: it is imported lazily inside
    :func:`verify_standard_ts_model` so that ``import ts_export`` (and therefore
    the export-only path) never hard-requires it.
  - All messaging goes through ``dl_techniques.utils.logger`` (no ``print``),
    per repo convention.
"""

from typing import List, Optional

import numpy as np

import keras

from dl_techniques.utils.logger import logger


def detect_input_length(
        model: keras.Model,
        config_keys: List[str],
        default: Optional[int] = None,
) -> int:
    """Detect the input (context/window) length of a loaded time-series model.

    Mirrors the 3-step detection logic of the per-trainer export scripts, with
    the config key(s) made a parameter so each caller can pass the key its model
    stores the length under (e.g. ``['input_length']`` for tirex,
    ``['context_len']`` for prism, ``['backcast_length']`` for nbeats).

    Detection order:
      1. ``model.input_shape`` — if it is a tuple with a concrete time dimension
         at axis 1, return it.
      2. Each key in ``config_keys`` (in order) against ``model.get_config()``;
         the first present key wins.
      3. ``default`` if provided.

    Args:
        model: A built/loaded Keras model.
        config_keys: Ordered list of ``get_config()`` keys to try after the
            input-shape probe. The first key present in the config is used.
        default: Fallback length returned (with a warning) when neither the
            input shape nor any config key yields a length. If ``None`` and
            nothing is detectable, a ``ValueError`` is raised.

    Returns:
        The detected input length.

    Raises:
        ValueError: If the length cannot be detected and ``default`` is ``None``.
    """
    # 1. Try the model's input shape (concrete time dim at axis 1).
    if getattr(model, "input_shape", None) is not None:
        input_shape = model.input_shape
        if isinstance(input_shape, tuple) and len(input_shape) >= 2:
            if input_shape[1] is not None:
                return int(input_shape[1])

    # 2. Try each config key in order.
    config = model.get_config()
    for key in config_keys:
        if key in config and config[key] is not None:
            return int(config[key])

    # 3. Fall back to the default, or fail loudly.
    if default is not None:
        logger.warning(
            "Could not auto-detect input length from input_shape or config "
            "keys %s. Using default %d.",
            config_keys, default,
        )
        return int(default)

    raise ValueError(
        f"Could not detect input length: input_shape has no concrete time "
        f"dimension and none of the config keys {config_keys} are present in "
        f"model.get_config(). Pass an explicit input_length or a default."
    )


def export_standard_ts_model(
        model_path: str,
        output_path: str,
        opset_version: int,
        input_length: int,
        num_features: int,
) -> str:
    """Export a standard single-input time-series Keras model to ONNX.

    Loads the model, runs a dummy forward pass to materialize internal shapes,
    builds an explicit single-tensor ``(batch, input_length, num_features)``
    input signature (avoiding symbolic-shape issues during tracing), and writes
    the ONNX graph via ``model.export(format="onnx", ...)``.

    This is the standard single-input flow shared by tirex / prism / mdn /
    nbeats / xlstm. Models with non-standard input signatures (deepar's dict
    input, adaptive_ema's scan-unroll) must NOT use this function.

    Args:
        model_path: Path to the ``.keras`` model file.
        output_path: Destination path for the ONNX file.
        opset_version: ONNX opset version to target.
        input_length: Concrete input sequence length for the input signature.
        num_features: Number of input features (last axis).

    Returns:
        The ``output_path`` the ONNX model was written to.
    """
    logger.info("Loading model from: %s", model_path)
    model = keras.saving.load_model(model_path, compile=False)

    logger.info(
        "Using input_length: %d, num_features: %d", input_length, num_features
    )

    # Dummy forward pass to set internal shapes for tracing.
    dummy_input = np.zeros((1, input_length, num_features), dtype=np.float32)
    _ = model(dummy_input, training=False)

    # Explicit input signature avoids symbolic-shape issues during tracing.
    input_signature = [
        keras.InputSpec(
            shape=(None, input_length, num_features),
            dtype="float32",
        )
    ]

    logger.info(
        "Exporting to ONNX (opset %d): %s", opset_version, output_path
    )
    model.export(
        output_path,
        format="onnx",
        input_signature=input_signature,
        opset_version=opset_version,
    )

    logger.info("ONNX export successful: %s", output_path)
    return output_path


def verify_standard_ts_model(
        model_path: str,
        onnx_path: str,
        input_length: int,
        num_features: int,
        num_samples: int,
        rtol: float = 1e-4,
        atol: float = 1e-4,
) -> bool:
    """Verify an exported ONNX model matches its Keras source on random input.

    Runs a single random ``(num_samples, input_length, num_features)`` tensor
    through both the Keras model and an ONNX Runtime ``CPUExecutionProvider``
    session and compares the primary output with :func:`numpy.allclose`. Any
    auxiliary ONNX inputs (e.g. shape tensors emitted by Reshape ops) are fed
    with zeros.

    ``onnxruntime`` is imported lazily here so that importing this module does
    not require it.

    Args:
        model_path: Path to the ``.keras`` model file.
        onnx_path: Path to the exported ONNX file.
        input_length: Input sequence length used to build the test tensor.
        num_features: Number of input features (last axis).
        num_samples: Batch size of the random verification tensor.
        rtol: Relative tolerance for :func:`numpy.allclose`.
        atol: Absolute tolerance for :func:`numpy.allclose`.

    Returns:
        ``True`` if the outputs match within tolerance, ``False`` otherwise
        (including when ``onnxruntime`` is not installed).
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

    keras_model = keras.saving.load_model(model_path, compile=False)
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    logger.info("ONNX model inputs:")
    for inp in ort_session.get_inputs():
        logger.info("  - %s: %s (%s)", inp.name, inp.shape, inp.type)

    logger.info("ONNX model outputs:")
    for out in ort_session.get_outputs():
        logger.info("  - %s: %s (%s)", out.name, out.shape, out.type)

    onnx_inputs = ort_session.get_inputs()
    output_name = ort_session.get_outputs()[0].name

    test_input = np.random.randn(
        num_samples, input_length, num_features
    ).astype(np.float32)

    # Primary data input + any auxiliary shape tensors (fed with zeros).
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
            "  Feeding auxiliary input '%s': shape=%s", inp.name, inp_shape
        )

    keras_preds = keras_model.predict(test_input, verbose=0)
    onnx_preds = ort_session.run([output_name], input_feed)[0]

    abs_diff = np.abs(keras_preds - onnx_preds)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

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

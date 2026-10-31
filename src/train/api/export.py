"""Write a trained Keras model in a deployment format: ``export``.

``export`` takes a model that is already in memory (the CLI loads one with
:func:`train.api.predictor.load_model`) and writes ONE new file or directory. The formats
are the literal :data:`FORMATS`; each reloads in its own runtime and gives the Keras
model's outputs on the same input to within float rounding (the test pins 1e-5 on one
batch of an untrained ``vit_pico``; other architectures are unmeasured).

Interface contract
------------------
``export(*, model, out, format, input_shape=None)``
    :param model: a built Keras model whose outputs are the logits (the same object a
        :class:`~train.api.predictor.Predictor` holds as ``.model``).
    :param out: where to write. ``keras`` needs a name ending ``.keras``, ``tflite`` ending
        ``.tflite``, ``onnx`` ending ``.onnx``; ``saved_model`` is a directory (any name).
        The parent directory is created. ``out`` must NOT exist: an existing file or
        directory is refused and left as it is. This module never overwrites and never
        deletes (invariant 5).
    :param format: one of :data:`FORMATS`.

        ``keras``
            ``model.save(out)``, a ``.keras`` archive.
        ``saved_model``
            ``model.export(out)``: a TensorFlow SavedModel directory with a ``serve``
            endpoint. Its batch size is free.
        ``tflite``
            ``tf.lite.TFLiteConverter.from_saved_model`` on a SavedModel. The converter
            needs a directory, so the SavedModel is written beside the result as
            ``<out>.saved_model`` and KEPT: removing it is the caller's act. That directory
            must not exist either. The interpreter's input has batch 1; resize it to run a
            larger batch.
        ``onnx``
            ``tf2onnx.convert.from_function`` on ``model(x, training=False)``, with a free
            batch dimension. Only if ``tf2onnx`` imports; otherwise a :class:`ConfigError`
            that says so, before anything is written.
    :param input_shape: ``(height, width, channels)`` without the batch. Needed by
        ``saved_model``, ``tflite`` and ``onnx`` when the model has no ``inputs`` of its own
        (a subclassed model such as the ViT); a functional model supplies it. ``keras``
        ignores it.
    :return: the :class:`pathlib.Path` written.
    :raises ConfigError: for an unknown format, a model that is not a built Keras model, a
        name of the wrong kind, an ``out`` (or ``<out>.saved_model``) that exists, a missing
        ``input_shape``, or ``onnx`` without ``tf2onnx``. A converter's own failure
        propagates, and leaves no ``out`` behind (a file is written last, in one piece).

``check_target(out, format)``
    :param out: the path the caller means to write.
    :param format: one of :data:`FORMATS`.
    :return: ``Path(out)``.
    :raises ConfigError: for an unknown format, a name of the wrong kind, an ``out`` (or
        ``<out>.saved_model`` for ``tflite``) that exists, or ``onnx`` without ``tf2onnx``.
        Everything ``export`` can refuse from the target alone, with no model and no write, so
        the command line can refuse before it loads a model (D-064). ``export`` calls it first.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); ``keras``,
``tensorflow`` and ``tf2onnx`` are imported inside the functions that use them.
"""

import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from .errors import ConfigError

# ---------------------------------------------------------------------

#: The formats ``export`` writes, in the order they are documented.
FORMATS = ("keras", "saved_model", "tflite", "onnx")

#: The name a file format must end with; ``saved_model`` is a directory and has none.
SUFFIX = {"keras": ".keras", "tflite": ".tflite", "onnx": ".onnx"}

#: The ending of the SavedModel directory written beside a ``tflite`` result.
TFLITE_SIBLING = ".saved_model"


def _refuse_existing(path: Path, key: str) -> None:
    if path.exists():
        raise ConfigError(
            f"Invalid value for {key!r}: {path} already exists; export never overwrites, "
            "so choose a path that is not there"
        )


def _input_spec(model: Any, input_shape: Optional[Sequence[int]]) -> Any:
    """Return the ``tf.TensorSpec`` ``(None, H, W, C)`` float32 the converters trace the model with."""
    import tensorflow as tf

    if input_shape is None:
        inputs = getattr(model, "inputs", None)
        if not inputs or len(inputs) != 1:
            raise ConfigError(
                "Invalid value for 'input_shape': this model does not report one input, so "
                "pass input_shape=(height, width, channels)"
            )
        input_shape = tuple(inputs[0].shape[1:])
    shape = tuple(input_shape)
    if not (
        len(shape) == 3 and all(isinstance(side, int) and not isinstance(side, bool) and side > 0 for side in shape)
    ):
        raise ConfigError(
            f"Invalid value for 'input_shape': expected (height, width, channels) of positive "
            f"ints, got {input_shape!r}"
        )
    return tf.TensorSpec((None, *shape), tf.float32, name="input")


def _export_saved_model(model: Any, directory: Path, spec: Any) -> None:
    model.export(str(directory), input_signature=[spec], verbose=False)


def _export_tflite(model: Any, out: Path, spec: Any) -> None:
    import tensorflow as tf

    # DECISION plan-2026-10-30T132432-a865fa08/D-063: the converter reads a SavedModel DIRECTORY,
    # and the intermediate is written beside the result and KEPT. Do NOT convert inside a
    # TemporaryDirectory (its cleanup is a deletion this package forbids) and do NOT switch
    # to ``from_keras_model`` to avoid the directory (the plan names ``from_saved_model``,
    # the path measured on vit_pico).
    sibling = out.with_name(out.name + TFLITE_SIBLING)
    _export_saved_model(model, sibling, spec)
    blob = tf.lite.TFLiteConverter.from_saved_model(str(sibling)).convert()
    # DECISION plan-2026-10-30T132432-a865fa08/D-063: ``"xb"`` creates the file and fails if it
    # exists; do NOT write with ``"wb"`` or ``Path.write_bytes``, which truncate a file that
    # appeared after the up-front check.
    with open(out, "xb") as handle:
        handle.write(blob)


def _export_onnx(model: Any, out: Path, spec: Any) -> None:
    import tensorflow as tf
    import tf2onnx

    function = tf.function(lambda batch: model(batch, training=False))
    proto, _ = tf2onnx.convert.from_function(function, input_signature=(spec,))
    with open(out, "xb") as handle:
        handle.write(proto.SerializeToString())


def _require_tf2onnx() -> None:
    try:
        import tf2onnx  # noqa: F401
    except ImportError as exc:
        raise ConfigError(
            "Invalid value for 'format': 'onnx' needs the tf2onnx package, which cannot be "
            f"imported ({exc}); install it, or choose one of: "
            + ", ".join(f for f in FORMATS if f != "onnx")
        ) from exc


def check_target(out: Union[str, "os.PathLike[str]"], format: str) -> Path:
    """Return ``Path(out)`` after every check the target alone allows; see the module contract."""
    if format not in FORMATS:
        raise ConfigError(
            f"Invalid value for 'format': expected one of {', '.join(map(repr, FORMATS))}, got {format!r}"
        )
    path = Path(out)
    suffix = SUFFIX.get(format)
    if suffix is not None and path.suffix != suffix:
        raise ConfigError(f"Invalid value for 'out': format {format!r} writes a file named *{suffix}, got {str(path)!r}")
    _refuse_existing(path, "out")
    if format == "tflite":
        _refuse_existing(path.with_name(path.name + TFLITE_SIBLING), "out")
    if format == "onnx":
        _require_tf2onnx()
    return path


def export(
    *,
    model: Any,
    out: Union[str, "os.PathLike[str]"],
    format: str,
    input_shape: Optional[Sequence[int]] = None,
) -> Path:
    """Write ``model`` to ``out`` as ``format``; see the module contract."""
    import keras

    path = check_target(out, format)
    if not isinstance(model, keras.Model) or not model.built:
        raise ConfigError(
            "Invalid value for 'model': export needs a built keras.Model (one that has been "
            f"called or loaded), got {type(model).__name__}"
        )
    spec = None if format == "keras" else _input_spec(model, input_shape)

    from dl_techniques.utils.logger import logger

    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "keras":
        model.save(str(path), overwrite=False)
    elif format == "saved_model":
        _export_saved_model(model, path, spec)
    elif format == "tflite":
        _export_tflite(model, path, spec)
    else:
        _export_onnx(model, path, spec)
    logger.info(f"Exported {format} to {path}")
    return path

"""Write a trained Keras model in a deployment format: ``export``.

``export`` takes a model that is already in memory (the CLI loads one with
:func:`train.api.predictor.load_model`) and writes ONE new file or directory. The formats
are the literal :data:`FORMATS`; each reloads in its own runtime and gives the Keras
model's outputs on the same input to within float rounding (the tests pin 1e-5 on an untrained
``vit_pico`` and a tolerance of 1e-4 on six families, see ``test_export_per_family.py``).

The module is named ``exporting`` so that ``train.api.export`` is always the FUNCTION on the
lazy package surface (D-068): a submodule named ``export`` would replace it on first import.

What goes in and what comes out
-------------------------------
Every format takes the input the run's model was TRAINED on, not a raw image: a float32
batch ``(batch, height, width, channels)`` whose pixels were scaled to ``[0, 1]`` and then
standardized, ``(pixels - mean) / std``, with the per-channel ``mean`` and ``std`` the run
recorded in ``task.json`` (``unit`` runs record zeros and ones). It returns the RAW LOGITS
with no softmax; the class names are in ``task.json`` too. A classification model gives one
row of class scores per image, ``(batch, classes)``. A SEGMENTATION model gives one score per
class per pixel, ``(batch, height, width, classes)``, at the record's ``input_size`` (height and
width are fixed to it, not necessarily equal; only the batch is free), and the mask is the argmax
over the last axis after the caller has resized the scores to the picture (D-081; a model of a
``mixed_float16`` run returns half-precision scores, as Keras does). Feeding ``[0, 255]`` or
unstandardized ``[0, 1]`` pixels gives numbers and no error.
The command line therefore writes ``<out>.task.json`` beside every export (:func:`export`'s
``record``), and its log says so.

Numerical agreement is RELATIVE to the scale of the scores, and for the four segmentation families it is a CONTRACT PER FAMILY
(D-029, which replaces D-088's one relative 1e-4 for every family, itself a correction of D-081's absolute 1e-4, measured on scores of scale 1:
an AccUNet's untrained logits are 3e-5, so its head was lifted to 1). For ``saved_model``, ``tflite`` and ``onnx``, read by their own
runtimes, the largest difference from Keras on the CPU as a share of the largest Keras score is at most (``EXPORT_BAND`` in
``test_segmentation_export.py``, between 2 and 10 times the largest gap measured for the family, on trained Oxford-IIIT Pet weights, CPU, 16 real
test images at 128x128, and on the test fixtures; ``tests/test_train/test_api/export_gap_evidence.json`` lists every row):

================= ========= =============================================================
family            band      largest gap measured (``saved_model``, ``tflite``, ``onnx``)
================= ========= =============================================================
``convunext``     5e-6      5.6e-7, 8.5e-7, 7.4e-7
``accunet``       1e-3      1.4e-4, 2.5e-4, 4.9e-5
``resnet_seg``    5e-6      3.9e-7, 1.8e-6, 1.4e-6
``convnext_seg``  5e-6      8.9e-7, 2.1e-6, 7.3e-7
================= ========= =============================================================

and the argmax agrees at 99.99% of the pixels or more (the worst measured is 99.9954%, an AccUNet ``tflite``). Three families are within a few
millionths of the scale; AccUNet is two orders of magnitude above them on trained weights (scores of 93 to 834 in the cells measured; why it is above is
not established here), which is why one number could not serve both: 1e-4 of the scale was too loose for the three and too tight for the file that measured
2.5e-4 (an AccUNet of the gate, 2 epochs, ``tflite``). The four AccUNet exports of the fix 9.3 reruns (4 epochs at the recipe's rate) are 3e-5 at most. The
review's AccUNet run (scale 220) differed from Keras by up to 9.2e-4 in ``saved_model`` (4.2e-6 of the scale), 2.8e-3 in ``tflite`` (1.3e-5) and 1.1e-2 in
``onnx`` (4.9e-5), with 99.999% of the argmaxes equal; ConvUNext's (scale 9.8) by 3.8e-6, 8.3e-6 and 6.2e-6. Read every absolute figure below as a figure
at a scale of 1, and the table as the contract on trained weights. Float32 models match Keras to about 1e-6 in ``keras``, ``saved_model`` and
``tflite``, and in ``onnx`` (the four segmentation families, ConvUNext ``tiny``, AccUNet, ``resnet_seg/resnet18``
and ``convnext_seg/cifar10``, in all four formats at 32x64, and ConvUNext ``tiny`` with real Oxford-IIIT Pet weights at
128x128 and 128x256 on 16 real images: 5e-6 at most on scores of scale 7, every argmax equal, D-081), with Keras run on the CPU (the check ``export`` makes of a ``tflite``
file written with select operators runs the Keras side on the CPU too, so a GPU machine compares
CPU with CPU). Serving a model on a GPU is a different computation: it differs from the CPU by about
1e-3 in the logits (measured on ``cliffordnet/nano`` of the ImageNet gate, GPU against CPU on the same
weights: up to 1.1e-3 on logits of scale 3.6 for 16 real validation images and 5.8e-4 on scale 3.0 for 16
noise images; the argmaxes were equal on all 16 of each), which no export format changes. The same holds for a segmentation model: ConvUNext ``tiny`` with the real Pet weights, Keras on the GPU against the CPU
files (or Keras on the CPU), 16 real images: up to 8.2e-3 on scores of scale 7.7, 1 pixel in 9,400 with another argmax (0.999893 agree). A trained AccUNet (scores up to 220) shows how much the device matters: ONE ``saved_model`` of its real Pet
weights, written by a CPU process and by a GPU process alike, differs from Keras in the same regime by 9.2e-4 on the CPU
(4.2e-6 of the scale, argmax agreement 1.0), by 9.3 to 9.4 on a GPU with TF32 on (4.2% to 4.3% of the scale, agreement
0.9957 to 0.9963) and by 2.5e-3 on a GPU with TF32 off (1.1e-5); against Keras on the CPU the TF32 file is 2.6 to 2.8 away
(1.2% to 1.3%, plan 98b866fb step 1, ``scratch_step1/accunet_export_regimes.json``). The file is not at fault: compare an
export with Keras on the CPU. The backbone families on real Pet weights (30 epochs, 128x128, 16 real test images, CPU,
plan 98b866fb step 2, ``scratch_step2/export_trained_*.json``): ``resnet_seg/resnet18`` (scores up to 16.0) differs from
Keras by 5.7e-6 in ``saved_model``, 1.0e-5 in ``tflite`` and 9.5e-6 in ``onnx``, and ``convnext_seg/tiny`` (scale 16.2) by
4.8e-6, 1.4e-5 and 7.6e-6, at most 9e-7 of the scale, with every argmax equal and each artifact answering the same twice
on one runtime object. A ``mixed_float16`` model matches in ``keras``, ``saved_model``
and ``tflite``, but its ``onnx`` export differs from Keras by up to about 1e-4 (9.0e-05
measured on ``vit_pico``): ONNX Runtime rounds the half-precision steps its own way.

Interface contract
------------------
``export(*, model, out, format, input_shape=None, record=None)``
    :param model: a built Keras model whose outputs are the logits (the same object a
        :class:`~train.api.predictor.Predictor` holds as ``.model``).
    :param out: where to write. ``keras`` needs a name ending ``.keras``, ``tflite`` ending
        ``.tflite``, ``onnx`` ending ``.onnx``; ``saved_model`` is a directory (any name).
        The parent directory is created. ``out`` must NOT exist (not even as a dangling
        symlink), and no ancestor may be a file: an existing file or directory is refused and
        left as it is. This module never overwrites and never deletes (invariant 5).
    :param format: one of :data:`FORMATS`.

        ``keras``
            ``model.save(out)``, a ``.keras`` archive.
        ``saved_model``
            A TensorFlow SavedModel directory with a ``serve`` endpoint, written with
            ``keras.export.ExportArchive`` and traced with ``training=False`` (D-066): a
            model with stochastic depth would otherwise serve a different random answer on
            every call. Its batch size is free.
        ``tflite``
            ``tf.lite.TFLiteConverter.from_saved_model`` on a SavedModel. The converter
            needs a directory, so the SavedModel is written beside the result as
            ``<out>.saved_model`` and KEPT: removing it is the caller's act. That directory
            must not exist either. The interpreter's input has batch 1; resize it to run a
            larger batch. The converter is tried with the built-in operators first. A model
            that needs TensorFlow operators (the CliffordNet families use ``Roll``) is
            converted again with ``SELECT_TF_OPS``, checked against the Keras model in the
            Python interpreter, and a WARNING is logged: the file then needs a runtime with
            the Flex delegate (the ``tensorflow`` pip package has it; a bare TFLite runtime
            does not). If that conversion also fails, a :class:`ConfigError` names the
            what it rejected (the operators it names, else its first line) and the kept
            SavedModel (D-067).
        ``onnx``
            ``tf2onnx.convert.from_function`` on ``model(x, training=False)``, with a free
            batch dimension. Only if ``tf2onnx`` imports; otherwise a :class:`ConfigError`
            that says so, before anything is written.
    :param input_shape: ``(height, width, channels)`` without the batch. Needed by
        ``saved_model``, ``tflite`` and ``onnx`` when the model has no ``inputs`` of its own
        (a subclassed model such as the ViT); a functional model supplies it. ``keras``
        ignores it.
    :param record: a checked ``task.json`` record (:mod:`train.api.predictor`), or ``None``.
        When given it is checked again with ``check_record`` (a record of an older layout is a
        ``ConfigError`` naming the retraining) and written, last, to ``<out>.task.json`` (``out`` plus
        :data:`~train.api.run_dir.TASK_SUFFIX`), which must not exist, and the log says where.
    :return: the :class:`pathlib.Path` written.
    :raises ConfigError: for an unknown format, a ``record`` that fails ``check_record``, a model
        that is not a built Keras model, a name of the wrong kind, an ``out`` (or
        ``<out>.saved_model``, or ``<out>.task.json`` with a ``record``) that exists or cannot be
        created, a missing ``input_shape``,
        ``onnx`` without ``tf2onnx``, or a ``tflite`` conversion that fails or disagrees with
        Keras. Any other converter failure propagates, and leaves no ``out`` behind (a file is
        written last, in one piece).

``check_target(out, format, with_record=False)``
    :param out: the path the caller means to write.
    :param format: one of :data:`FORMATS`.
    :param with_record: also require ``<out>.task.json`` to be free.
    :return: ``Path(out)``.
    :raises ConfigError: for an unknown format, a name of the wrong kind, an ``out`` (or
        ``<out>.saved_model`` for ``tflite``) that exists or sits under a file, or ``onnx``
        without ``tf2onnx``. Everything ``export`` can refuse from the target alone, with no
        model and no write, so the command line can refuse before it loads a model (D-064).
        ``export`` calls it first.

``check_free(path, key, command="export")``
    :param path: a path the caller means to create.
    :param key: the name of the option, for the message.
    :param command: the command, for the message.
    :raises ConfigError: when ``path`` exists (a dangling symlink counts), or an ancestor
        exists and is not a directory. Used by ``predict`` for its ``out`` as well.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); ``keras``,
``tensorflow`` and ``tf2onnx`` are imported inside the functions that use them.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from .errors import ConfigError
from .run_dir import TASK_SUFFIX

# ---------------------------------------------------------------------

#: The formats ``export`` writes, in the order they are documented.
FORMATS = ("keras", "saved_model", "tflite", "onnx")

#: The name a file format must end with; ``saved_model`` is a directory and has none.
SUFFIX = {"keras": ".keras", "tflite": ".tflite", "onnx": ".onnx"}

#: The ending of the SavedModel directory written beside a ``tflite`` result.
TFLITE_SIBLING = ".saved_model"

#: The largest difference, relative to ``max(1, |logit|)``, a ``tflite`` file written with
#: select operators may have from the Keras model on the check batch before it is refused.
TFLITE_CHECK_RTOL = 1e-4

#: How much of a converter's own first line a :class:`ConfigError` quotes when it names no operator.
_CONVERTER_LINE = 300


def check_free(path: Path, key: str, command: str = "export") -> None:
    """Raise :class:`ConfigError` unless ``path`` can be created; see the module contract."""
    if os.path.lexists(path):
        raise ConfigError(
            f"Invalid value for {key!r}: {path} already exists; {command} never overwrites or "
            "deletes, so move it away or choose a path that is not there"
        )
    for ancestor in path.parents:
        if os.path.lexists(ancestor):
            if not ancestor.is_dir():
                raise ConfigError(
                    f"Invalid value for {key!r}: {ancestor} exists and is not a directory, so "
                    f"{path} cannot be created under it"
                )
            return


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
    import keras

    # DECISION plan-2026-10-30T132432-a865fa08/D-066: the endpoint calls the model with
    # ``training=False``. Do NOT use ``model.export(...)``: it traces ``model(x)`` with
    # ``training=None``, and a layer that only skips its randomness on ``training is False``
    # (stochastic depth) keeps it live in ``serve``, so the artifact is random at inference
    # and reports success (CliffordNet, review-iter-3 concern 1).
    # DECISION plan-2026-10-31T180033-2faf890a/D-081
    # The segmentation U-Nets take this path unchanged: ConvUNext's StochasticDepth is live under
    # ``training=None`` too (8e-5 of spread against scores of scale 1). Do NOT check it with two FRESH
    # loads of the artifact: a fresh load restarts the random generator, so both loads draw the same
    # noise and the check passes on a random graph (the first version of the test did). Hold ONE runtime
    # object and call it twice, and scan the graph for random ops.
    # Plan 98b866fb D-044 made StochasticDepth and StochasticGradient inert unless ``training`` is truthy, so
    # ``training=None`` no longer leaves them live; the explicit ``training=False`` stays so that the artifact
    # does not depend on any layer's predicate. Do NOT drop it because the two layers were fixed.
    archive = keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint("serve", lambda batch: model(batch, training=False), [spec])
    archive.write_out(str(directory), verbose=False)


def _formats_but(name: str) -> str:
    """The other formats in a sentence: ``keras, saved_model or onnx``."""
    names = [f for f in FORMATS if f != name]
    return ", ".join(names[:-1]) + " or " + names[-1]


def _refuse_bfloat16_tflite(model: Any) -> None:
    """Refuse, before anything is written, a model that computes in bfloat16.

    :raises ConfigError: naming the format, the precision and the formats that carry the model.
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-066
    # A refusal, not an export, and BEFORE the SavedModel is written. The graph casts to bfloat16
    # and the interpreter of the tensorflow package has no kernel for CAST version 7 (reproduced
    # on a GPU, XLA auto-jit off and on: the check of the converted file died with a ValueError,
    # exit 1, and `<out>.saved_model` was left behind). Do NOT trace the model in float32 to get
    # a file: the variables are float32 whichever policy the layers compute in, but the graph
    # that was trained and checked is the bfloat16 one, and a file of another graph is not the
    # model. mixed_float16 and float32 export (measured, review-iter-2 concern 9).
    if getattr(model, "compute_dtype", None) == "bfloat16":
        raise ConfigError(
            "Invalid value for 'format': 'tflite' cannot carry a model that computes in bfloat16 "
            "(precision mixed_bfloat16): its graph casts to bfloat16 and the TFLite interpreter of "
            "the tensorflow package has no kernel for that cast, so the file could be neither "
            f"checked nor run. Nothing was written; choose {_formats_but('tflite')}, or train in "
            "float32 or mixed_float16"
        )


def _what_failed(error: BaseException) -> str:
    """Return a short sentence about a converter failure: the operators it names, else its first line."""
    text = str(error)
    operators = sorted(
        set(re.findall(r'fused\["([A-Za-z0-9_]+):"', text)) - {"StatefulPartitionedCall", "PartitionedCall"}
    )
    if operators:
        return "the converter rejected the operator(s) " + ", ".join(operators)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[0] if lines else type(error).__name__)[:_CONVERTER_LINE]


def _tflite_logits(blob: bytes, batch: Any) -> Any:
    """Return the logits the TFLite interpreter gives for ``batch`` (the Python build has Flex)."""
    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=blob)
    entry = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(entry["index"], batch.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(entry["index"], batch)
    interpreter.invoke()
    return np.asarray(interpreter.get_tensor(interpreter.get_output_details()[0]["index"]), np.float64)


def _check_tflite_against_keras(blob: bytes, model: Any, spec: Any, out: Path) -> None:
    """Raise :class:`ConfigError` when ``blob`` does not give the Keras model's logits.

    Both sides run on the CPU: the interpreter always does, and the Keras side is placed there
    with ``tf.device`` so a GPU machine compares CPU with CPU.
    """
    import numpy as np
    import tensorflow as tf

    batch = np.random.default_rng(0).random((2, *tuple(spec.shape[1:]))).astype(np.float32)
    # DECISION plan-2026-10-31T180033-2faf890a/D-042
    # The Keras side is placed on the CPU. Do NOT let it run on the default device: on a GPU
    # machine the model's own logits then come from GPU kernels (TF32 convolutions and
    # matmuls, other reduction orders) and differ from the CPU interpreter by about 1e-3
    # (measured: nano 4.9e-4 against an allowance of 2.9e-4), so a faithful file was refused
    # on exactly the machines that train. Do NOT widen the tolerance to hide it: the check
    # must still refuse a file that is really wrong (the mutation test breaks the file).
    with tf.device("/CPU:0"):
        expected = np.asarray(model(batch, training=False), np.float64)
    try:
        got = _tflite_logits(blob, batch)
    except ValueError as exc:
        # DECISION plan-2026-10-31T180033-2faf890a/D-066
        # The interpreter raises ValueError for an operator it has no kernel for ("Didn't find
        # op for builtin opcode 'CAST' version '7'" for a bfloat16 model). Do NOT let it reach
        # the user as a traceback and exit 1: the request (this model as tflite) cannot be
        # honoured, which is exit 2, and the SavedModel the converter read is still on disk.
        first = next((line.strip() for line in str(exc).splitlines() if line.strip()), type(exc).__name__)
        raise ConfigError(
            f"Invalid value for 'format': the 'tflite' file converted from this model cannot be run "
            f"by the TFLite interpreter ({first[:_CONVERTER_LINE]}), so {out} was not written. Its "
            f"SavedModel was kept at {out.with_name(out.name + TFLITE_SIBLING)}; move it away before "
            f"trying again, or choose {_formats_but('tflite')}"
        ) from None
    scale = max(1.0, float(np.abs(expected).max()))
    worst = float("inf") if got.shape != expected.shape else float(np.abs(got - expected).max())
    if not worst <= TFLITE_CHECK_RTOL * scale:
        raise ConfigError(
            f"Invalid value for 'format': the tflite file converted with TensorFlow operators "
            f"differs from the Keras model by {worst:.3g} on a check batch (allowed "
            f"{TFLITE_CHECK_RTOL * scale:.3g}), so {out} was not written"
        )


def _convert_tflite(sibling: Path, model: Any, spec: Any, out: Path) -> bytes:
    """Return the ``.tflite`` bytes of the SavedModel ``sibling``; see the module contract."""
    import tensorflow as tf
    from tensorflow.lite.python.convert_phase import ConverterError

    from dl_techniques.utils.logger import logger

    def convert(select: bool) -> bytes:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(sibling))
        if select:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
        return converter.convert()

    # DECISION plan-2026-10-30T132432-a865fa08/D-067: built-in operators first, then select
    # operators, then a readable refusal. Do NOT go straight to SELECT_TF_OPS (a model that
    # converts with built-ins would then need the Flex runtime for nothing), do NOT skip the
    # numeric check of the select-ops result, and do NOT let ConverterError escape: its text is
    # an MLIR dump of the whole graph.
    try:
        return convert(False)
    except ConverterError as builtin_failure:
        logger.info(
            f"tflite: built-in operators cannot express this model ({_what_failed(builtin_failure)}); "
            "retrying with select TF operators"
        )
    try:
        blob = convert(True)
    except ConverterError as select_failure:
        logger.debug(f"tflite converter error with select operators: {select_failure}")
        raise ConfigError(
            f"Invalid value for 'format': the tflite converter cannot convert this model, even "
            f"with select TF operators ({_what_failed(select_failure)}). Its SavedModel was kept at "
            f"{sibling}; move it away before trying again, or choose keras, saved_model or onnx"
        ) from None
    _check_tflite_against_keras(blob, model, spec, out)
    logger.warning(
        f"{out} needs select TF operators, so it runs only where the Flex delegate is available "
        "(the tensorflow pip package has it; a plain TFLite runtime does not)"
    )
    return blob


def _export_tflite(model: Any, out: Path, spec: Any) -> None:
    # DECISION plan-2026-10-30T132432-a865fa08/D-063: the converter reads a SavedModel DIRECTORY,
    # and the intermediate is written beside the result and KEPT. Do NOT convert inside a
    # TemporaryDirectory (its cleanup is a deletion this package forbids) and do NOT switch
    # to ``from_keras_model`` to avoid the directory (the plan names ``from_saved_model``,
    # the path measured on vit_pico).
    sibling = out.with_name(out.name + TFLITE_SIBLING)
    _export_saved_model(model, sibling, spec)
    blob = _convert_tflite(sibling, model, spec, out)
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


def _record_path(out: Path) -> Path:
    return out.with_name(out.name + TASK_SUFFIX)


def check_target(
    out: Union[str, "os.PathLike[str]"], format: str, with_record: bool = False
) -> Path:
    """Return ``Path(out)`` after every check the target alone allows; see the module contract."""
    if format not in FORMATS:
        raise ConfigError(
            f"Invalid value for 'format': expected one of {', '.join(map(repr, FORMATS))}, got {format!r}"
        )
    path = Path(out)
    suffix = SUFFIX.get(format)
    if suffix is not None and path.suffix != suffix:
        raise ConfigError(f"Invalid value for 'out': format {format!r} writes a file named *{suffix}, got {str(path)!r}")
    check_free(path, "out")
    if format == "tflite":
        check_free(path.with_name(path.name + TFLITE_SIBLING), "out")
    if with_record:
        check_free(_record_path(path), "out")
    if format == "onnx":
        _require_tf2onnx()
    return path


def export(
    *,
    model: Any,
    out: Union[str, "os.PathLike[str]"],
    format: str,
    input_shape: Optional[Sequence[int]] = None,
    record: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write ``model`` to ``out`` as ``format``; see the module contract."""
    import keras

    path = check_target(out, format, with_record=record is not None)
    if record is not None:
        # A record this package's reader would refuse (an older version, a missing key) is not
        # written beside a fresh model (D-073); nothing has been written yet.
        from .predictor import check_record

        check_record(dict(record), "<given>")
    if not isinstance(model, keras.Model) or not model.built:
        raise ConfigError(
            "Invalid value for 'model': export needs a built keras.Model (one that has been "
            f"called or loaded), got {type(model).__name__}"
        )
    spec = None if format == "keras" else _input_spec(model, input_shape)
    if format == "tflite":
        _refuse_bfloat16_tflite(model)

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
    if record is not None:
        beside = _record_path(path)
        with open(beside, "x") as handle:
            json.dump(dict(record), handle, indent=2)
            handle.write("\n")
        logger.info(
            f"Wrote {beside}: the exported model takes inputs standardized with the run's "
            "mean and std (pixels in [0, 1], then (x - mean) / std) and returns raw logits"
        )
    return path

"""Predict with a trained pretrain run: :class:`PretrainPredictor`, the ``.npy`` writer of ``predict`` and the export
row of a ``pretrain`` model.

``train.api.predictor.load_model`` returns a :class:`PretrainPredictor` when the run's ``task.json`` says
``pretrain``. A pretrain run trains a masked-image model (``et_mim``: masked-patch completion; ``beit_mim``: the code
ids of a frozen tokenizer), whose own output on a bare image is a reconstruction or code logits, not something to
serve. ``predict`` and ``export`` answer the BACKBONE's patch-token embeddings instead (U2, plan 5c6cc105 D-003): one
float32 row per patch, the patches in row-major order, ``N = (side / patch_size)^2`` rows per image for both families
(BEiT's class-token row is dropped). The image goes through what the pretrain evaluation stream applies (D-023): a
file decoded by TensorFlow, the bytes divided by 255, then ``train.common.image_streams.evaluation_transform`` (a
bilinear resize to the trained side and the record's standardization). The model is built at one size (its position
table is fixed at the build) and is never rebuilt at another.

Interface contract
------------------
``read_image(item)``
    :param item: a path (``str`` or ``Path``), a ``PIL.Image.Image``, or a ``numpy`` array ``(H, W)``, ``(H, W, 1)``,
        ``(H, W, 3)`` or ``(H, W, 4)``.
    :return: ``uint8`` ``(H, W, 3)``. A path is decoded with ``tf.io.decode_image(channels=3)``, the decoder of the
        Imagenette stream (no EXIF orientation, as in training); any other form by
        :func:`train.api.segmentation_predictor.rgb_of` (a grey image repeated, alpha dropped, a floating array in
        ``[0, 1]`` rounded to its byte).
    :raises ConfigError: for a path that is not a file or that TensorFlow cannot read or decode (its message named),
        and for what ``rgb_of`` refuses.

``model_input(images, record)``
    :param images: ``uint8`` ``(H, W, 3)`` arrays of any size (:func:`read_image`).
    :param record: a checked pretrain record.
    :return: the float32 batch ``(n, side, side, 3)`` the backbone is fed: each image divided by 255, then
        ``evaluation_transform(side, mean, std)`` with the record's numbers.

``PretrainPredictor(model, record)`` / ``PretrainPredictor.predict(x)``
    :param model: the loaded MIM model; its ``backbone`` maps images to tokens (BEiT's with a class token first).
    :param record: a pretrain record that passed :func:`train.api.predictor.check_record` (another task is refused).
    :param x: one image or several (a list or tuple, or a 4-D array), each a form :func:`read_image` takes.
    :return: ``{"embeddings": [array, ...]}``, ALWAYS a list, one float32 ``(N, width)`` array per image in the order
        given. Images run :data:`CHUNK` at a time through ``model.backbone(..., training=False)``.
    :raises ConfigError: for an unreadable image, an empty list, or a backbone whose tokens are not ``N`` patch rows.

``check_built_size(model, record, model_file, task_file)`` / ``check_output(output, record)``
    The ``MODEL_CHECKS`` and ``OUTPUT_CHECKS`` rows of ``train.api.predictor``. :raises ConfigError: when the model has
    no ``backbone``, or the record's ``input_size`` and ``channels`` are not the ones the backbone was built with (its
    ``get_config()``), before the model is called; and when the MIM model's output on the zero image does not have
    ``N`` rows, which is how a ``patch_size`` that is not the model's is found (at the built size, ``N`` is a function
    of the patch).

``write_predictions(predictor, files, out)``
    Create the directory ``out`` (it must not exist), write ``<stem>.npy`` (the float32 ``(N, width)`` array,
    ``numpy.save``) for each of ``files`` and, LAST, ``index.json``: ``{"output": "patch_embeddings", "note": ...,
    "embeddings": [{"path", "file", "shape": [N, width]}, ...]}``. A run that fails half way leaves its arrays and no
    ``index.json``. :raises ConfigError: for two files with one stem (naming both), or any error of ``predict``.

``served(model, record)`` / ``check_export(record, format, shape)`` / ``export_note(record)``
    The ``EXPORT_RULES`` row (``shape_of`` is the record's ``(side, side, channels)``). ``served`` returns the
    functional ``keras.Model`` image ``(side, side, 3)`` -> patch tokens over the run's backbone, which every format
    runs. ``check_export`` raises :class:`ConfigError`, before anything is written, for a ``shape`` that is not the
    record's and for a cell of :data:`EXPORT_REFUSALS`. Measured on tiny runs of both families at 32 px (plan
    5c6cc105 D-024): the keras file equals the predictor exactly; saved_model, onnx and BEiT's tflite are within 7e-7
    of the scale.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only; ``tensorflow``, ``keras``, ``numpy`` and
``train.common`` are imported inside functions.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .depth_predictor import ARRAY_SUFFIX
from .errors import ConfigError
from .exporting import check_free
from .predictor import HOPFIELD_TFLITE_CAUSE, PRETRAIN, PRETRAIN_OUTPUTS, check_record, refuse_export_cell
from .pretrain_args import BEIT, ET
from .run_dir import TASK_FILE
from .segmentation_predictor import INDEX_NAME, items_of, mask_names, rgb_of

# DECISION plan-2026-11-19T081214-5c6cc105/D-013
# The task's predictor, served function, export check and writer live in this one module, like every other task's.
# Do NOT add a registered layer for the served function: the functional model over the existing backbone layer saves
# and reloads bit-exactly for both families (measured on tiny runs: a reload gap of 0.0), so there is no second
# abstraction to pay for.

#: Images turned into tensors and run through the backbone at a time.
CHUNK = 8

#: The rows a family's backbone returns BEFORE its patch tokens: BEiT's class token.
LEADING_ROWS = {ET: 0, BEIT: 1}

#: What an answer of ``predict`` is, in ``index.json`` and in the export note.
OUTPUT_NOTE = (
    "patch-token embeddings of the backbone: one float32 row per patch, the patches in row-major order, a class-token "
    "row dropped"
)

# DECISION plan-2026-11-19T081214-5c6cc105/D-024
# The et_mim tflite cell is refused HERE, before the SavedModel sibling is written. Do NOT let the export try it: the
# shared numeric check refuses the file (0.935 off on its check batch, built-in and select operators alike) only after
# `<out>.saved_model` exists, with no cause named. The cause is TensorFlow 2.18's converter on the Hopfield update
# (einsum, relu, einsum: 0.58 off in a plain-TensorFlow repro, each einsum alone exact, the matmul form exact). Do NOT
# drop the row because a new TensorFlow is installed: `test_the_et_mim_tflite_refusal_still_has_its_reason` says when.
#: The export cells that are refused, ``(family, format) -> reason``, each measured.
EXPORT_REFUSALS: Dict[Tuple[str, str], str] = {
    (ET, "tflite"): f"{HOPFIELD_TFLITE_CAUSE}, and the numeric check would refuse it; choose keras, saved_model or onnx",
}


def read_image(item: Any) -> Any:
    """Return one image as ``uint8`` ``(H, W, 3)``; see the module contract."""
    if not isinstance(item, (str, os.PathLike)):
        return rgb_of(item)
    import tensorflow as tf

    path = os.fspath(item)
    if not os.path.isfile(path):
        raise ConfigError(f"Invalid image: {path!r} is not a file")
    # DECISION plan-2026-11-19T081214-5c6cc105/D-023
    # A file is decoded by TensorFlow, the Imagenette stream's decoder (equal to its decode_jpeg on 4 of 4 JPEGs).
    # Do NOT use `rgb_of`'s PIL decode for a path (0 of 4 equal, up to 8 levels off) and do NOT use the classification
    # predictor's `decode_image_file`: it scales by convert_image_dtype, 1 ulp off `/ 255` on 126 of 256 levels.
    try:
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    except tf.errors.OpError as exc:
        raise ConfigError(
            f"Invalid image: {path!r} cannot be read as an image by tf.io.decode_image: {exc.message}"
        ) from exc
    return image.numpy()


def model_input(images: Sequence[Any], record: Mapping[str, Any]) -> Any:
    """Return the float32 batch ``(n, side, side, 3)`` the backbone is fed for ``uint8`` ``images``; see above."""
    import numpy as np
    import tensorflow as tf

    from train.common.image_streams import evaluation_transform

    transform = evaluation_transform(record["input_size"][0], record["mean"], record["std"])
    return np.stack([np.asarray(transform(tf.cast(image, tf.float32) / 255.0)) for image in images])


def patch_tokens(tokens: Any, record: Mapping[str, Any]) -> Any:
    """Return ``tokens`` ``(batch, rows, width)`` without the family's leading rows (a tensor or a Keras tensor)."""
    # DECISION plan-2026-11-19T081214-5c6cc105/D-003
    # The answer is the BACKBONE's tokens, the class row dropped. Do NOT serve the MIM model's own output (a
    # reconstruction for ET, code logits for BEiT, which are no image without the tokenizer's decoder), and do NOT keep
    # BEiT's class row: both families give N = (side / patch)^2 rows in one layout, and the class token stays reachable
    # through the archive.
    rows = LEADING_ROWS[record["family"]]
    return tokens[:, rows:, :] if rows else tokens


def _patches(record: Mapping[str, Any]) -> int:
    height, width = record["input_size"]
    return (height // record["patch_size"]) * (width // record["patch_size"])


def check_built_size(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    """Raise :class:`ConfigError` unless the record's size and channels are the ones the backbone was built at."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names a {PRETRAIN} model but {model_file} holds a "
            f"{type(model).__name__} with no backbone; the file and the model are not from one run"
        )
    config = backbone.get_config()
    built = [int(side) for side in config["input_shape"]]
    if built != [*record["input_size"], record["channels"]]:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names input_size {record['input_size']} and {record['channels']} "
            f"channels but {model_file} is built at {built}; its backbone takes only the size it was built at, so the "
            "file and the model are not from one run"
        )


def check_output(output: Any, record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless the MIM model's ``output`` on one zero image has ``N`` rows (the patch)."""
    height, width = record["input_size"]
    rows = _patches(record)
    shape = tuple(output.shape) if not isinstance(output, (list, tuple, dict)) else None
    if shape is None or len(shape) != 3 or shape[:2] != (1, rows):
        seen = f"{type(output).__name__} of {len(output)} outputs" if shape is None else str(shape)
        raise ConfigError(
            f"The model outputs {seen} for one {height}x{width} image but {TASK_FILE} names a {PRETRAIN} model of "
            f"{rows} patches of {record['patch_size']} px, (1, {rows}, width); the file and the model are not from "
            "one run"
        )


class PretrainPredictor:
    """A loaded pretrain model and the preprocessing it was trained with; see the module contract.

    :param model: the loaded MIM model; its ``backbone`` gives the tokens.
    :param record: a pretrain record that passed :func:`train.api.predictor.check_record`.
    """

    def __init__(self, model: Any, record: Mapping[str, Any]) -> None:
        check_record(dict(record), "<given>")
        if record["task"] != PRETRAIN:
            raise ConfigError(
                f"Invalid {TASK_FILE} <given>: the task is {record['task']!r} but this predictor serves "
                f"{PRETRAIN!r} only"
            )
        self.model = model
        self.record = dict(record)

    def _embeddings(self, images: Sequence[Any]) -> Any:
        """Return the float32 embeddings ``(n, N, width)`` of ``uint8`` ``images`` (:func:`read_image`)."""
        import numpy as np

        tokens = self.model.backbone(model_input(images, self.record), training=False)
        answer = np.asarray(patch_tokens(tokens, self.record), dtype=np.float32)
        rows = _patches(self.record)
        if answer.ndim != 3 or answer.shape[:2] != (len(images), rows):
            raise ConfigError(
                f"The backbone gives {answer.shape} for {len(images)} images but {TASK_FILE} names {rows} patch rows "
                "per image; the file and the model are not from one run"
            )
        return answer

    def predict(self, x: Any) -> Dict[str, Any]:
        """Return ``{"embeddings": [array, ...]}`` for ``x``; see the module contract."""
        items = items_of(x)
        answers: List[Any] = []
        for start in range(0, len(items), CHUNK):
            answers.extend(self._embeddings([read_image(item) for item in items[start : start + CHUNK]]))
        return {"embeddings": answers}


# ---------------------------------------------------------------------
# The command line's output
# ---------------------------------------------------------------------


def write_predictions(predictor: PretrainPredictor, files: Sequence[Path], out: Path) -> None:
    """Write one ``.npy`` per file and ``index.json`` into the new directory ``out``; see the module contract."""
    # DECISION plan-2026-11-19T081214-5c6cc105/D-025
    # A writer of its own beside depth's, sharing what the two share (mask_names, check_free, INDEX_NAME, the .npy
    # suffix). Do NOT generalize depth's write_predictions for it: depth's is bound to its answer (the "disparity" key,
    # the map's size, its output word and note), and the two answers need not move together.
    import numpy as np

    names = mask_names(files, ARRAY_SUFFIX)
    check_free(out, "out", "predict")
    out.mkdir(parents=True)
    entries: List[Dict[str, Any]] = []
    for start in range(0, len(files), CHUNK):
        chunk = files[start : start + CHUNK]
        answers = predictor.predict([str(f) for f in chunk])["embeddings"]
        for file, name, one in zip(chunk, names[start : start + CHUNK], answers):
            # "xb" creates the file and fails if it exists; numpy.save(path) would truncate one.
            with open(out / name, "xb") as handle:
                np.save(handle, one)
            entries.append({"path": str(file), "file": name, "shape": [int(side) for side in one.shape]})
    with open(out / INDEX_NAME, "x") as handle:
        json.dump({"output": PRETRAIN_OUTPUTS[0], "note": OUTPUT_NOTE, "embeddings": entries}, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def served(model: Any, record: Mapping[str, Any]) -> Any:
    """Return the functional ``keras.Model`` image ``(side, side, channels)`` -> patch tokens over the backbone."""
    import keras

    height, width = record["input_size"]
    image = keras.Input((height, width, record["channels"]), name="image")
    tokens = patch_tokens(model.backbone(image, training=False), record)
    return keras.Model(image, tokens, name="patch_embeddings")


def check_export(record: Mapping[str, Any], format: str, shape: Sequence[Optional[int]]) -> None:
    """Raise :class:`ConfigError` for another size or a refused cell; see the module contract."""
    wanted = (*record["input_size"], record["channels"])
    if tuple(shape) != wanted:
        raise ConfigError(
            f"Invalid value for 'input_shape': {tuple(shape)} is not the trained size {wanted} (height, width, "
            f"channels) of this {PRETRAIN} model; its backbone takes only the size it was built at, so it is exported "
            "at that size and a file of another size would be a new model"
        )
    refuse_export_cell(EXPORT_REFUSALS, record, format)


def export_note(record: Mapping[str, Any]) -> str:
    """Return the sentence logged beside an exported pretrain model."""
    height, width = record["input_size"]
    return (
        f"the exported model takes float32 images (batch, {height}, {width}, {record['channels']}), the trained size "
        "only, with pixels scaled to [0, 1] and then standardized with the mean and std of the record, "
        f"(x - mean) / std; it returns (batch, {_patches(record)}, width) {OUTPUT_NOTE}"
    )

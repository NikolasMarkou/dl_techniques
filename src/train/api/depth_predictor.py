"""Predict with a trained depth run: :class:`DepthPredictor`, the ``.npy`` writer of ``predict`` and the export row of a
``depth_estimation`` model.

``train.api.predictor.load_model`` returns a :class:`DepthPredictor` when the run's ``task.json`` says
``depth_estimation``. The model was built at one size, the crops it trained on (``input_size`` of the record): its DINOv2
position table is fixed at the build, so it takes no other size (D-009). The predictor therefore resizes each image to
that size, standardizes it as training did, runs the model and resizes the map back to the image's own size (D-012). The
model is never rebuilt at another size: serving the trained artifact is this module's job, and starting a run from an
earlier run's weights at another size is a phase of its own (D-020).

Interface contract
------------------
``DepthPredictor(model, record)``
    :param model: the loaded Keras model; its input is ``(batch, height, width, 3)`` at the record's ``input_size`` and its
        output ``(batch, height, width, 1)``.
    :param record: a depth record that passed :func:`train.api.predictor.check_record`.
    :raises ConfigError: for a record of another task.

``DepthPredictor.predict(x)``
    :param x: one image or several. An image is a path (``str`` or ``Path``), a ``PIL.Image.Image``, or a ``numpy`` array
        ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``; several are a list or tuple of those, or a 4-D array.
        Every form is read by the dense tasks' one decoder (:func:`train.api.segmentation_predictor.rgb_of`: a file through
        ``segmentation_data.decode_image``, EXIF orientation applied, then RGB, so a grey image is repeated and an RGBA
        image loses its alpha; a floating array must lie in ``[0, 1]`` and is rounded to its byte).
    :return: ``{"disparity": [map, ...]}``, ALWAYS a list, one ``numpy`` ``float32`` ``(height, width)`` map per image in
        the order given, at the size of THAT image (after its EXIF orientation): relative disparity, larger is nearer,
        defined up to a scale and a shift per image (the loss is affine-invariant), never metric depth.
    :raises ConfigError: for an unreadable file, an unsupported array, an empty list, or a model whose output is not one
        map per image at the record's size.

    The image is resized to ``input_size`` with PIL's bilinear filter on its bytes (the loader's resize,
    ``train.common.megadepth.load_and_process_pair``), mapped to ``[-1, 1]`` as the loader maps a byte (``/ 127.5 - 1``)
    and standardized by ``train.common.megadepth.standardize_imagenet`` (the function training calls); the model runs
    with ``training=False``; the map is resized to the image's size with PIL's bilinear filter on float32 values. A whole
    image resized to the crop size is not what the model trained on (crops of full-resolution images): the answer is
    coarser than a tiled one would be, and the README says so.

``check_built_size(model, record, model_file, task_file)`` / ``check_output(output, record)``
    The ``MODEL_CHECKS`` and ``OUTPUT_CHECKS`` rows of ``train.api.predictor``. :raises ConfigError: when the record's
    ``input_size`` and ``channels`` are not the size the model was built at (its ``get_config()["image_shape"]``), before
    the model is called; and when the output on the zero image is not one ``(1, height, width, 1)`` map.

``write_predictions(predictor, files, out)``
    Create the directory ``out`` (it must not exist), write ``<stem>.npy`` (the float32 map, ``numpy.save``) for each of
    ``files`` and, LAST, ``index.json``: ``{"output": "relative_disparity", "note": ..., "disparity": [{"path", "file",
    "size": [height, width]}, ...]}``. Images are answered :data:`CHUNK` at a time. A run that fails half way leaves its
    arrays and no ``index.json``: the index is the mark of a complete answer.
    :raises ConfigError: for two files with one stem (naming both), or any error of ``predict``.

``check_export(record, format, shape)`` / ``export_note(record)``
    The ``EXPORT_RULES`` row (its ``shape_of`` is the record's ``(height, width, channels)``). Every format exports at the
    trained size: ``keras``, ``saved_model``, ``tflite`` and ``onnx`` each give the Keras map within a band measured
    relative to its scale (D-018: keras exactly, saved_model 1e-5, onnx 3e-5, tflite 5e-5; the test of this module holds
    the bands). :raises ConfigError: from ``check_export`` for a ``shape`` that is not the record's, before anything is
    written. ``export_note`` is the sentence logged beside the file: what it takes and returns.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only; ``tensorflow``, ``numpy``,
``PIL`` and ``train.common`` are imported inside functions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import ConfigError
from .exporting import check_free
from .predictor import DEPTH, DEPTH_OUTPUTS, check_record
from .run_dir import TASK_FILE
from .segmentation_predictor import INDEX_NAME, items_of, mask_names, rgb_of

#: Images turned into tensors and run through the model at a time.
CHUNK = 8

#: The suffix of one written map.
ARRAY_SUFFIX = ".npy"

#: What a map of ``predict`` is, in ``index.json`` and in the export note.
OUTPUT_NOTE = (
    "relative disparity: larger is nearer, defined up to scale and shift per image (the loss is affine-invariant), "
    "not metric depth"
)


def check_built_size(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    """Raise :class:`ConfigError` unless the record's size and channels are the ones the model was built at."""
    # DECISION plan-2026-11-17T110935-9c7b9096/D-009
    # The record is held to the size the model was BUILT at, before the model is called. Do NOT leave it to the call on
    # zeros: a size that is not a multiple of the patch (30 for a 28 px model) gives the same 2 x 2 patch grid and an
    # answer, so the record would load and lie; and do NOT rebuild the model at the record's size (another size is a new
    # model, the init_from phase of D-020, never what predict serves).
    built = [int(side) for side in model.get_config()["image_shape"]]
    named = [*record["input_size"], record["channels"]]
    if built != named:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names input_size {record['input_size']} and {record['channels']} "
            f"channels but {model_file} is built at {built}; a Depth Anything takes only the size it was built at, so "
            "the file and the model are not from one run"
        )


def check_output(output: Any, record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless ``output`` is ONE map ``(1, height, width, 1)`` at the record's size."""
    height, width = record["input_size"]
    shape = tuple(output.shape) if not isinstance(output, (list, tuple, dict)) else None
    if shape != (1, height, width, 1):
        seen = f"{type(output).__name__} of {len(output)} outputs" if shape is None else str(shape)
        raise ConfigError(
            f"The model outputs {seen} for one {height}x{width} image but {TASK_FILE} names a depth model (one map of "
            f"shape (1, {height}, {width}, 1)); the file and the model are not from one run"
        )


class DepthPredictor:
    """A loaded depth model and the preprocessing it was trained with; see the module contract.

    :param model: the Keras model; its output is one relative-disparity map per image at the record's size.
    :param record: a depth record that passed :func:`train.api.predictor.check_record`.
    """

    def __init__(self, model: Any, record: Mapping[str, Any]) -> None:
        check_record(dict(record), "<given>")
        if record["task"] != DEPTH:
            raise ConfigError(
                f"Invalid {TASK_FILE} <given>: the task is {record['task']!r} but this predictor serves {DEPTH!r} only"
            )
        self.model = model
        self.record = dict(record)

    def model_input(self, images: Sequence[Any]) -> Any:
        """Return the float32 batch ``(N, height, width, 3)`` the model is fed for ``uint8`` RGB ``images``."""
        import numpy as np
        from PIL import Image

        from train.common.megadepth import standardize_imagenet

        height, width = self.record["input_size"]
        rows = []
        for image in images:
            # The loader's resize of an RGB file (bilinear on the bytes), then its byte-to-[-1, 1] map and its
            # standardization: one function, the one training calls.
            resized = np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR), dtype=np.float32)
            rows.append(standardize_imagenet(resized / 127.5 - 1.0))
        return np.stack(rows)

    def _maps(self, images: Sequence[Any]) -> Any:
        """Return the model's float32 maps ``(N, height, width)`` for ``uint8`` RGB ``images``."""
        import numpy as np

        height, width = self.record["input_size"]
        output = self.model(self.model_input(images), training=False)
        maps = np.asarray(output, dtype=np.float32)
        if maps.shape != (len(images), height, width, 1):
            raise ConfigError(
                f"The model outputs {maps.shape} for {len(images)} images of {height}x{width} but {TASK_FILE} names a "
                f"depth model (one map of ({height}, {width}, 1) per image); the file and the model are not from one run"
            )
        return maps[..., 0]

    @staticmethod
    def _at_size(one: Any, size: Tuple[int, int]) -> Any:
        """Return the map ``one`` resized bilinearly to ``size`` ``(height, width)``, float32."""
        import numpy as np
        from PIL import Image

        # DECISION plan-2026-11-17T110935-9c7b9096/D-012
        # The MAP goes back to the image's own size (bilinear, float32); the model ran at the trained size. Do NOT run the
        # model at the image's size (it takes only the size it was built at, D-009), do NOT quantize the map to a PNG (its
        # scale is arbitrary per image), and do NOT tile here (a seam blend for a deliverable the bar does not judge).
        height, width = size
        return np.asarray(Image.fromarray(one).resize((width, height), Image.BILINEAR), dtype=np.float32)

    def predict(self, x: Any) -> Dict[str, Any]:
        """Return ``{"disparity": [map, ...]}`` for ``x``; see the module contract."""
        items = items_of(x)
        answers: List[Any] = []
        for start in range(0, len(items), CHUNK):
            images = [rgb_of(item) for item in items[start : start + CHUNK]]
            for image, one in zip(images, self._maps(images)):
                answers.append(self._at_size(one, (int(image.shape[0]), int(image.shape[1]))))
        return {"disparity": answers}


# ---------------------------------------------------------------------
# The command line's output
# ---------------------------------------------------------------------


def _write_array(path: Path, array: Any) -> None:
    import numpy as np

    # "xb" creates the file and fails if it exists; numpy.save(path) would truncate one.
    with open(path, "xb") as handle:
        np.save(handle, array)


def write_predictions(predictor: DepthPredictor, files: Sequence[Path], out: Path) -> None:
    """Write one ``.npy`` per file and ``index.json`` into the new directory ``out``; see the module contract."""
    names = mask_names(files, ARRAY_SUFFIX)
    check_free(out, "out", "predict")
    out.mkdir(parents=True)
    entries: List[Dict[str, Any]] = []
    for start in range(0, len(files), CHUNK):
        chunk = files[start : start + CHUNK]
        answers = predictor.predict([str(f) for f in chunk])["disparity"]
        for file, name, one in zip(chunk, names[start : start + CHUNK], answers):
            _write_array(out / name, one)
            entries.append({"path": str(file), "file": name, "size": [int(one.shape[0]), int(one.shape[1])]})
    with open(out / INDEX_NAME, "x") as handle:
        json.dump({"output": DEPTH_OUTPUTS[0], "note": OUTPUT_NOTE, "disparity": entries}, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def check_export(record: Mapping[str, Any], format: str, shape: Sequence[Optional[int]]) -> None:
    """Raise :class:`ConfigError` unless ``shape`` is the trained size; see the module contract."""
    # DECISION plan-2026-11-17T110935-9c7b9096/D-018
    # Every format is exported, at the trained size ONLY, refused by name for any other. Do NOT leave another size to
    # the converters: asked at 42 or 30 px for a 28 px model, saved_model, tflite and onnx all WRITE a file (measured,
    # D-027): the 42 px file fails when it is run at 42, and the 30 px file answers a 30 px image with a 28 x 28 map,
    # with no error. Do NOT refuse a format here (all four are within their bands at 252 and 28 px). The model is never
    # rebuilt at another size (D-020).
    wanted = (*record["input_size"], record["channels"])
    if tuple(shape) != wanted:
        raise ConfigError(
            f"Invalid value for 'input_shape': {tuple(shape)} is not the trained size {wanted} (height, width, "
            f"channels) of this {DEPTH} model; a Depth Anything takes only the size it was built at, so it is "
            "exported at that size and a file of another size would be a new model"
        )


def export_note(record: Mapping[str, Any]) -> str:
    """Return the sentence logged beside an exported depth model."""
    height, width = record["input_size"]
    return (
        f"the exported model takes float32 images (batch, {height}, {width}, 3), the trained size {record['input_size']} "
        "only, with pixels scaled to [0, 1] and then standardized with the ImageNet mean and std, (x - mean) / std (no "
        f"other size and no raw pixels); it returns (batch, {height}, {width}, 1) {OUTPUT_NOTE}"
    )

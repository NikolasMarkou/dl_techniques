"""Predict with a trained tokenizer run: :class:`TokenizerPredictor`, the ``.npy`` writer of ``predict`` and the export row of a
``beit_tokenizer`` model.

``train.api.predictor.load_model`` returns a :class:`TokenizerPredictor` when the run's ``task.json`` says ``beit_tokenizer``. A tokenizer's
product is the integer CODE GRID of an image: one id per ``patch_size`` x ``patch_size`` cell, the index of the codebook entry nearest the
cell's encoding, in row-major order: the targets a ``beit_mim`` run is trained on. ``predict`` and ``export`` answer it, never the
reconstruction (the quality of the tokenizer is the run's summary). The image goes through what the pretrain evaluation stream applies, by
the pretrain predictor's own reader and transform (a file decoded by TensorFlow, the bytes divided by 255, a bilinear resize to the trained
side and the record's standardization), so the tokenizer sees what ``beit_mim`` shows it. The model is built at one size and is never
rebuilt at another.

Interface contract
------------------
``TokenizerPredictor(model, record)`` / ``TokenizerPredictor.predict(x)``
    :param model: the loaded ``VQVAERotationTrick``; its ``encode_to_indices`` gives the codes.
    :param record: a ``beit_tokenizer`` record that passed :func:`train.api.predictor.check_record` (another task is refused).
    :param x: one image or several (a list or tuple, or a 4-D array), each a form :func:`train.api.pretrain_predictor.read_image` takes (a
        path, a PIL image or an array).
    :return: ``{"codes": [array, ...]}``, ALWAYS a list, one int32 ``(grid, grid)`` array per image in the order given, ``grid = side //
        patch_size``. Images run :data:`CHUNK` at a time.
    :raises ConfigError: for an unreadable image, an empty list, or a model whose grid is not the record's.

``check_built_model(model, record, model_file, task_file)`` / ``check_output(output, record)``
    The ``MODEL_CHECKS`` and ``OUTPUT_CHECKS`` rows of ``train.api.predictor``. :raises ConfigError: from the first, before the model is
    called on anything the record could make fail, when the record's ``input_size`` and ``channels``, ``patch_size`` or ``num_embeddings``
    are not the ones in the model's own config, and when the model's code grid for a zero image is not ``side // patch_size`` (the grid
    is proved by running it, never read off the config); from the second when the reconstruction of one zero image is not
    ``(1, height, width, channels)``.

``write_predictions(predictor, files, out)``
    Create the directory ``out`` (it must not exist), write ``<stem>.npy`` (the int32 ``(grid, grid)`` array, ``numpy.save``) for each of
    ``files`` and, LAST, ``index.json``: ``{"output": "code_ids", "note": ..., "codes": [{"path", "file", "shape": [grid, grid]}, ...]}``.
    A run that fails half way leaves its arrays and no ``index.json``. :raises ConfigError: for two files with one stem (naming both), or
    any error of ``predict``.

``served(model, record)`` / ``check_export(record, format, shape)`` / ``export_note(record)``
    The ``EXPORT_RULES`` row (``shape_of`` is the record's ``(side, side, channels)``). ``served`` returns the layer every format runs,
    :class:`train.common.tokenizer_data.TokenizerCodes`: standardized image to int32 code grid. ``check_export`` raises
    :class:`ConfigError`, before anything is written, for a ``shape`` that is not the record's; it refuses no cell of the table. Measured
    (plan-2026-11-25T124419-fa94f59c, ``scratch_export/measured.md``): each of keras, saved_model, tflite and onnx gave Keras's code at every
    position of 64 and of 512 images, for a tiny and a default-size model, untrained and trained. The limit, which ``export_note`` states and a
    test keeps true: a position whose two nearest codes are within float32 rounding of each other can take the other one in a file.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only; ``numpy`` and ``train.common`` are imported inside functions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .depth_predictor import ARRAY_SUFFIX, write_array
from .errors import ConfigError
from .exporting import check_free
from .predictor import BEIT_TOKENIZER, TOKENIZER_OUTPUTS, check_record
from .pretrain_predictor import model_input, read_image
from .run_dir import TASK_FILE
from .segmentation_predictor import INDEX_NAME, items_of, mask_names

#: Images run through the model at a time: the distances of one chunk are ``chunk x cells x codes`` floats (8 x 196 x 8,192 at 224 px).
CHUNK = 8

#: What an answer of ``predict`` is, in ``index.json``.
OUTPUT_NOTE = (
    "code ids: one int32 per cell of the image's patch grid, row-major, the index of the nearest entry of the codebook; the image is "
    "resized to the trained side and standardized as the evaluation stream does"
)


def _grid_of(record: Mapping[str, Any]) -> int:
    """The side of the code grid: the record's square side over its patch (``record_problems`` holds both)."""
    return record["input_size"][0] // record["patch_size"]


def check_built_model(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    """Raise :class:`ConfigError` unless the record's size, patch and codebook are the model's and its code grid is the record's."""
    # DECISION plan-2026-11-25T124419-fa94f59c/D-044
    # The record is held to the model's OWN config BEFORE the model is called on anything: the size it was built at
    # (`input_shape_config`), the encoder's total stride (`downsample_factor`) and the codebook (`num_embeddings`). Then the code grid is
    # PROVED by running the model on a zero image, here and not in `check_output`: the call `load_model` makes on zeros returns the
    # RECONSTRUCTION, which shows no grid, and a config can say a stride the layers do not have (the adapter proves the grid the same way
    # at build time, D-032). Do NOT read the grid off `downsample_factor` alone, and do NOT leave the size to the zero-input call: a
    # record of another size would then fail inside Keras with a message about a layer.
    config = model.get_config()
    named = [*record["input_size"], record["channels"]]
    built = config.get("input_shape_config")
    problems: List[str] = []
    if built is None or [int(side) for side in built] != named:
        shown = "no size it reports" if built is None else str([int(side) for side in built])
        problems.append(f"input_size {record['input_size']} and {record['channels']} channels but {model_file} is built at {shown}")
    if config.get("downsample_factor") != record["patch_size"]:
        problems.append(
            f"patch_size {record['patch_size']} but the encoder of {model_file} has a total stride of {config.get('downsample_factor')}")
    if config.get("num_embeddings") != record["num_embeddings"]:
        problems.append(
            f"num_embeddings {record['num_embeddings']} but {model_file}'s codebook holds {config.get('num_embeddings')} codes")
    if problems:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names {'; and '.join(problems)}; the file and the model are not from one run")
    import numpy as np

    grid = tuple(int(side) for side in model.encode_to_indices(np.zeros((1, *named), np.float32)).shape[1:])
    if grid != (_grid_of(record), _grid_of(record)):
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names a patch_size of {record['patch_size']} on {record['input_size']}, a code grid of "
            f"{_grid_of(record)} x {_grid_of(record)}, but {model_file} gives {grid} for a zero image; the file and the model are not "
            "from one run"
        )


def check_output(output: Any, record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless ``output``, the model's call on one zero image, is ONE image at the record's size."""
    height, width = record["input_size"]
    wanted = (1, height, width, record["channels"])
    shape = tuple(output.shape) if not isinstance(output, (list, tuple, dict)) else None
    if shape != wanted:
        seen = f"{type(output).__name__} of {len(output)} outputs" if shape is None else str(shape)
        raise ConfigError(
            f"The model outputs {seen} for one {height}x{width} image but {TASK_FILE} names a tokenizer whose reconstruction is "
            f"{wanted}; the file and the model are not from one run"
        )


class TokenizerPredictor:
    """A loaded tokenizer and the preprocessing it was trained with; see the module contract.

    :param model: the loaded ``VQVAERotationTrick``.
    :param record: a ``beit_tokenizer`` record that passed :func:`train.api.predictor.check_record`.
    """

    def __init__(self, model: Any, record: Mapping[str, Any]) -> None:
        check_record(dict(record), "<given>")
        if record["task"] != BEIT_TOKENIZER:
            raise ConfigError(
                f"Invalid {TASK_FILE} <given>: the task is {record['task']!r} but this predictor serves {BEIT_TOKENIZER!r} only")
        self.model = model
        self.record = dict(record)

    def _codes(self, images: Sequence[Any]) -> List[Any]:
        """Return the int32 grids ``(grid, grid)`` of ``uint8`` ``images`` (:func:`train.api.pretrain_predictor.read_image`)."""
        import numpy as np

        codes = np.asarray(self.model.encode_to_indices(model_input(images, self.record)), dtype=np.int32)
        grid = _grid_of(self.record)
        if codes.shape != (len(images), grid, grid):
            raise ConfigError(
                f"The model gives {codes.shape} for {len(images)} images but {TASK_FILE} names a {grid} x {grid} code grid per image; "
                "the file and the model are not from one run"
            )
        return list(codes)

    def predict(self, x: Any) -> Dict[str, Any]:
        """Return ``{"codes": [array, ...]}`` for ``x``; see the module contract."""
        items = items_of(x)
        answers: List[Any] = []
        for start in range(0, len(items), CHUNK):
            answers.extend(self._codes([read_image(item) for item in items[start : start + CHUNK]]))
        return {"codes": answers}


# ---------------------------------------------------------------------
# The command line's output
# ---------------------------------------------------------------------


def write_predictions(predictor: TokenizerPredictor, files: Sequence[Path], out: Path) -> None:
    """Write one ``.npy`` per file and ``index.json`` into the new directory ``out``; see the module contract."""
    # DECISION plan-2026-11-25T124419-fa94f59c/D-047
    # A writer of its own beside depth's and pretrain's, sharing what they share (mask_names, check_free, INDEX_NAME, the .npy suffix and the
    # no-truncate write_array), for the reason plan 5c6cc105 D-025 gave: each is bound to its answer's key, its note and its index entry, and
    # the answers need not move together. Do NOT generalize pretrain's writer for it, and do NOT copy write_array back here.
    names = mask_names(files, ARRAY_SUFFIX)
    check_free(out, "out", "predict")
    out.mkdir(parents=True)
    entries: List[Dict[str, Any]] = []
    for start in range(0, len(files), CHUNK):
        chunk = files[start : start + CHUNK]
        answers = predictor.predict([str(f) for f in chunk])["codes"]
        for file, name, one in zip(chunk, names[start : start + CHUNK], answers):
            write_array(out / name, one)
            entries.append({"path": str(file), "file": name, "shape": [int(side) for side in one.shape]})
    with open(out / INDEX_NAME, "x") as handle:
        json.dump({"output": TOKENIZER_OUTPUTS[0], "note": OUTPUT_NOTE, "codes": entries}, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def served(model: Any, record: Mapping[str, Any]) -> Any:
    """Return the layer every format runs: a standardized image to its int32 code grid; see the module contract."""
    from train.common.tokenizer_data import TokenizerCodes

    return TokenizerCodes(model)


def check_export(record: Mapping[str, Any], format: str, shape: Sequence[Optional[int]]) -> None:
    """Raise :class:`ConfigError` for another size; no cell of the table is refused (see the module contract)."""
    wanted = (*record["input_size"], record["channels"])
    if tuple(shape) != wanted:
        raise ConfigError(
            f"Invalid value for 'input_shape': {tuple(shape)} is not the trained size {wanted} (height, width, channels) of this "
            f"{BEIT_TOKENIZER} model; its code grid is the patch grid of that size, so it is exported at that size and a file of "
            "another size would be a new model"
        )


def export_note(record: Mapping[str, Any]) -> str:
    """Return the sentence logged beside an exported tokenizer."""
    height, width = record["input_size"]
    patch, grid = record["patch_size"], _grid_of(record)
    return (
        f"the exported model takes float32 images (batch, {height}, {width}, {record['channels']}), the trained size only, with pixels "
        "scaled to [0, 1] and then standardized with the mean and std of the record, (x - mean) / std; it returns int32 code ids "
        f"(batch, {grid}, {grid}): for each cell of {patch} x {patch} pixels, in row-major order, the index of the nearest of the "
        f"{record['num_embeddings']} codes of the codebook. A position whose two nearest codes are within float32 rounding of each other can "
        "take the other one in another runtime or file (every position of the models measured agreed, and a codebook built to hold such "
        "pairs does not); the .keras file holds a layer of the train package, so it loads only where train is importable"
    )

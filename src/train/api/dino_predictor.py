"""Predict with a trained dino run: :class:`DinoPredictor`, the ``.npy`` writer of ``predict`` and the export row of a ``dino`` model.

``train.api.predictor.load_model`` returns a :class:`DinoPredictor` when the run's ``task.json`` says ``dino``. The file a dino run saves is the
TRAINING model (a student, an EMA teacher, both with projection heads, taking a view set: ``(batch, 2 + n_local_crops, H, W, C)``), and what a
user wants of it is a feature extractor: image in, the student's CLS embedding out (the post-final-norm token, ``embedding_dim`` wide, the vector
the k-NN probe scores; never the projection head's output, which is a training-time device). The feature model is DERIVED on every load, as
``train.dino.knn_eval`` derives it. The image goes through what the pretrain evaluation stream applies, by the pretrain predictor's own reader and
transform (a file decoded by TensorFlow, the bytes divided by 255, a bilinear resize to the trained side and the record's standardization), so the
model sees what the probe showed it.

Interface contract
------------------
``feature_model(model)``
    :return: ``keras.Model(student.inputs[0], student.head.input)``: image ``(None, H, W, C)`` to embedding ``(None, embedding_dim)``.
    :raises ConfigError: for a model with no student or no projection head (not a dino file).

``DinoPredictor(model, record)`` / ``DinoPredictor.predict(x)``
    :param model: the loaded ``DINOTrainingModel``. :param record: a ``dino`` record that passed :func:`train.api.predictor.check_record`.
    :param x: one image or several (a list or tuple, or a 4-D array), each a form :func:`train.api.pretrain_predictor.read_image` takes.
    :return: ``{"embeddings": [array, ...]}``, ALWAYS a list, one float32 ``(embedding_dim,)`` array per image in the order given. Images run
        :data:`CHUNK` at a time. :raises ConfigError: for an unreadable image, an empty list, or a model whose embedding is not the record's width.

``check_built_model(model, record, model_file, task_file)`` / ``check_output(output, record)``
    The ``MODEL_CHECKS`` and ``OUTPUT_CHECKS`` rows of ``train.api.predictor``. :raises ConfigError: from the first, BEFORE the model is called on
    a view set the record could make fail, when the record's view count, size and channels, patch, head width or embedding width are not the
    model's own (the view count from its config, the patch grid and the widths PROVED by running the student's patch layer and the model on a
    zero view set); from the second when the packed output of the zero view set is not ``(2 * views - 2, 2 * dino_out_dim)``.

``write_predictions(predictor, files, out)``
    Create the directory ``out`` (it must not exist), write ``<stem>.npy`` (the float32 ``(embedding_dim,)`` array, ``numpy.save``) for each of
    ``files`` and, LAST, ``index.json``: ``{"output": "cls_embeddings", "note": ..., "embeddings": [{"path", "file", "shape"}, ...]}``. A run that
    fails half way leaves its arrays and no ``index.json``. :raises ConfigError: for two files with one stem, or any error of ``predict``.

``served(model, record)`` / ``check_export(record, format, shape)`` / ``export_note(record)``
    The ``EXPORT_RULES`` row (``shape_of`` is the record's ``(side, side, channels)``). ``served`` returns the functional feature model, run with
    ``training=False``. ``check_export`` raises :class:`ConfigError`, before anything is written, for a ``shape`` that is not the record's; it
    refuses no cell of the table (D-028: the four formats were measured at both sizes, the worst gap 1.7e-6).

Import rule
-----------
Module scope imports the standard library and ``train.api`` only; ``numpy``, ``keras`` and ``train.common`` are imported inside functions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .depth_predictor import ARRAY_SUFFIX, write_array
from .dino_args import TASK as DINO
from .errors import ConfigError
from .exporting import check_free
from .predictor import DINO_OUTPUTS, check_record
from .pretrain_predictor import model_input, read_image
from .run_dir import TASK_FILE
from .segmentation_predictor import INDEX_NAME, items_of, mask_names

#: Images run through the model at a time.
CHUNK = 16

#: What an answer of ``predict`` is, in ``index.json``.
OUTPUT_NOTE = (
    "CLS embeddings: one float32 vector per image, the student's post-final-norm CLS token (the vector the k-NN probe scores, never the projection "
    "head's output); the image is resized to the trained side and standardized as the evaluation stream does"
)


def feature_model(model: Any) -> Any:
    """Return the functional image-to-embedding model of a loaded dino training model; see the module contract."""
    import keras

    student = getattr(model, "student", None)
    head = getattr(student, "head", None)
    if student is None or head is None:
        raise ConfigError(
            f"The loaded model is a {type(model).__name__} with no student that has a projection head, so there is no backbone and head "
            "boundary to read the CLS embedding from; it is not the file of a dino run")
    # DECISION plan-2026-11-26T224507-e3381378/D-012
    # The embedding is `student.head.input`, the post-final-norm CLS token, NEVER `student.output` (the `dino_out_dim`-wide projection) and
    # never the teacher: `train.dino.knn_eval._build_feature_model` reads the same tensor, so a prediction is what the probe scored. A swap to
    # the output is silent (plausible numbers of another width). Do NOT cache the model across loads: it is bound to the loaded student.
    return keras.Model(inputs=student.inputs[0], outputs=head.input)


def check_built_model(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    """Raise :class:`ConfigError` unless the record's views, size, patch, head and embedding are the model's; see the module contract."""
    import keras
    import numpy as np

    config = model.get_config()
    views = 2 + record["n_local_crops"]
    problems: List[str] = []
    if 2 + config.get("n_local_views", -1) != views:
        problems.append(f"n_local_crops {record['n_local_crops']} ({views} views) but {model_file} was built for {2 + config.get('n_local_views', -1)} views")
    student = getattr(model, "student", None)
    if student is None:
        raise ConfigError(f"Invalid {TASK_FILE} {task_file}: {model_file} holds a {type(model).__name__} with no student; it is not a dino model")
    built = [int(side) if side is not None else None for side in student.inputs[0].shape[1:]]
    named = [*record["input_size"], record["channels"]]
    if built != named:
        problems.append(f"input_size {record['input_size']} and {record['channels']} channels but {model_file} is built at {built}")
    if problems:
        raise ConfigError(f"Invalid {TASK_FILE} {task_file}: it names {'; and '.join(problems)}; the file and the model are not from one run")
    zero = np.zeros((1, *named), np.float32)
    tokens = int(student.get_layer("patch_embed")(keras.ops.convert_to_tensor(zero)).shape[1])
    expected_tokens = (record["input_size"][0] // record["patch_size"]) ** 2
    width = int(feature_model(model).output.shape[-1])
    if tokens != expected_tokens:
        problems.append(f"patch_size {record['patch_size']} ({expected_tokens} tokens) but {model_file} cuts the image into {tokens} tokens")
    if width != record["embedding_dim"]:
        problems.append(f"embedding_dim {record['embedding_dim']} but {model_file} gives an embedding of {width}")
    if problems:
        raise ConfigError(f"Invalid {TASK_FILE} {task_file}: it names {'; and '.join(problems)}; the file and the model are not from one run")


def check_output(output: Any, record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless ``output``, the model's call on one zero view set, is ``(2 * views - 2, 2 * dino_out_dim)``."""
    views = 2 + record["n_local_crops"]
    wanted = (2 * views - 2, 2 * record["dino_out_dim"])
    shape = tuple(output.shape) if not isinstance(output, (list, tuple, dict)) else None
    if shape != wanted:
        seen = f"{type(output).__name__} of {len(output)} outputs" if shape is None else str(shape)
        raise ConfigError(
            f"The model outputs {seen} for one view set of {views} views but {TASK_FILE} names a head of {record['dino_out_dim']} outputs "
            f"({wanted}); the file and the model are not from one run"
        )


class DinoPredictor:
    """A loaded dino model and the preprocessing it was trained with; see the module contract.

    :param model: the loaded ``DINOTrainingModel``.
    :param record: a ``dino`` record that passed :func:`train.api.predictor.check_record`.
    """

    def __init__(self, model: Any, record: Mapping[str, Any]) -> None:
        check_record(dict(record), "<given>")
        if record["task"] != DINO:
            raise ConfigError(f"Invalid {TASK_FILE} <given>: the task is {record['task']!r} but this predictor serves {DINO!r} only")
        self.model = model
        self.record = dict(record)
        self.features = feature_model(model)

    def _embeddings(self, images: Sequence[Any]) -> List[Any]:
        """Return the float32 embeddings ``(embedding_dim,)`` of ``uint8`` ``images`` (:func:`read_image`)."""
        import numpy as np

        answer = np.asarray(self.features(model_input(images, self.record), training=False), dtype=np.float32)
        width = self.record["embedding_dim"]
        if answer.shape != (len(images), width):
            raise ConfigError(
                f"The model gives {answer.shape} for {len(images)} images but {TASK_FILE} names an embedding of {width}; the file and the model "
                "are not from one run"
            )
        return list(answer)

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


def write_predictions(predictor: DinoPredictor, files: Sequence[Path], out: Path) -> None:
    """Write one ``.npy`` per file and ``index.json`` into the new directory ``out``; see the module contract."""
    # DECISION plan-2026-11-26T224507-e3381378/D-047
    # A writer of its own beside depth's, pretrain's and the tokenizer's, sharing what they share (mask_names, check_free, INDEX_NAME, the .npy
    # suffix and the no-truncate write_array); each is bound to its answer's key, its note and its index entry. Do NOT generalize one for the
    # others and do NOT copy write_array back here.
    names = mask_names(files, ARRAY_SUFFIX)
    check_free(out, "out", "predict")
    out.mkdir(parents=True)
    entries: List[Dict[str, Any]] = []
    for start in range(0, len(files), CHUNK):
        chunk = files[start : start + CHUNK]
        answers = predictor.predict([str(f) for f in chunk])["embeddings"]
        for file, name, one in zip(chunk, names[start : start + CHUNK], answers):
            write_array(out / name, one)
            entries.append({"path": str(file), "file": name, "shape": [int(side) for side in one.shape]})
    with open(out / INDEX_NAME, "x") as handle:
        json.dump({"output": DINO_OUTPUTS[0], "note": OUTPUT_NOTE, "embeddings": entries}, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def served(model: Any, record: Mapping[str, Any]) -> Any:
    """Return the functional feature model every format runs: a standardized image to its CLS embedding; see the module contract."""
    # DECISION plan-2026-11-26T224507-e3381378/D-063
    # `exporting` traces THIS model but archives the TRAINING model it was given, so a SavedModel (and the directory beside a tflite file) also carries
    # the teacher's and the heads' weights: 117 MB against 22 MB for the keras, tflite and onnx files at the default size (measured, scratch_export). Do NOT
    # try to shrink it from here (the function returns a model; what the archive tracks is `exporting`'s) and do NOT drop the sentence of `export_note`
    # that says so until a test (`test_the_limit_the_export_note_states_is_still_real...`) fails.
    return feature_model(model)


def check_export(record: Mapping[str, Any], format: str, shape: Sequence[Optional[int]]) -> None:
    """Raise :class:`ConfigError` for another size; no cell of the table is refused (see the module contract)."""
    wanted = (*record["input_size"], record["channels"])
    if tuple(shape) != wanted:
        raise ConfigError(
            f"Invalid value for 'input_shape': {tuple(shape)} is not the trained size {wanted} (height, width, channels) of this {DINO} model; "
            "its position table is the patch grid of that size, so it is exported at that size and a file of another size would be a new model"
        )


def export_note(record: Mapping[str, Any]) -> str:
    """Return the sentence logged beside an exported dino feature model."""
    height, width = record["input_size"]
    return (
        f"the exported model takes float32 images (batch, {height}, {width}, {record['channels']}), the trained size only, with pixels scaled to "
        "[0, 1] and then standardized with the mean and std of the record, (x - mean) / std; it returns the student's CLS embedding "
        f"(batch, {record['embedding_dim']}) (not the projection head's output, and no teacher); the .keras file holds the functional feature "
        "model of the student, built from dl_techniques layers only, so it loads without the train package; a SavedModel (and the directory beside "
        "a tflite file) also holds the training model's weights, the teacher's and the heads', so it is several times the size of the other files"
    )

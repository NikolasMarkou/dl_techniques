"""Reload a trained run and predict with it: ``task.json``, ``load_model`` and ``Predictor``.

A run directory holds ``task.json`` beside its models: what a model needs to be used
without the config that trained it (the class names, the input geometry, and the
per-channel mean and standard deviation the training inputs were standardized with,
including the statistics of a folder or csv run's fit split, D-060). ``load_model``
reads it, imports the family's model code so its classes are registered, loads the
Keras file and returns a :class:`Predictor`, which applies the loader's own
preprocessing to a path, a PIL image or an array and returns class names and
probabilities.

Interface contract
------------------
``task.json`` (schema version 1), a JSON object with exactly these keys:

``schema_version``
    ``1``. Any other value is refused: a reader never guesses at a newer layout.
``task``, ``model``, ``family``, ``variant``
    The task word, ``"family/variant"``, and its two halves.
``class_names``
    One text per class, in the order of the model's outputs.
``input_shape``, ``channels``
    ``[height, width, channels]`` of the model's input; ``channels`` repeats the last.
``normalize``
    ``"standardize"`` or ``"unit"`` (what the run asked for).
``mean``, ``std``
    One number per channel: what the training inputs were standardized with, as the
    data loader reports it. ``unit`` is zeros and ones. A folder or csv run records the
    statistics of ONE pass over its fit split, not an ImageNet constant.
``eval_transform``
    ``{"kind", "source_channels"}``: how an image of any size becomes the model's
    input before standardization (:data:`EVAL_KINDS`, the evaluation pipelines of the
    loader), and the channel count the files are read with (``1`` for MNIST, whose
    single channel is repeated up to ``channels``).

``build_task_record(...)``
    :return: the record for a finished data load (not checked; see its docstring).

``check_record(record, source)``
    :param record: a parsed ``task.json``.
    :param source: its path, for the messages.
    :raises ConfigError: for a wrong ``schema_version``, a missing or unknown key,
        a value of the wrong kind, a ``mean`` or ``std`` of the wrong length, a ``std``
        that is not positive, or an ``eval_transform`` whose kind is not in :data:`EVAL_KINDS`.

``load_model(path)``
    :param path: a run directory (its ``best_model.keras``) or a ``.keras`` file whose
        directory holds ``task.json``.
    :return: a :class:`Predictor`. Compiles nothing.
    :raises ConfigError: for a missing path, ``task.json`` or model file, a bad record,
        or a family that is not registered.

``Predictor.predict(x, top_k=None, threshold=None)``
    :param x: one image or several. A single image is a path (``str`` or ``Path``), a
        ``PIL.Image.Image``, or a 2-D or 3-D ``numpy`` array; several are a list or tuple
        of those, or a 4-D array. A ``uint8`` array is divided by 255; a floating array
        must already lie in ``[0, 1]`` and is refused otherwise; any other dtype is
        refused. A path is read with ``channels`` of ``eval_transform.source_channels``.
    :param top_k: keep the ``top_k`` best classes of each image; ``None`` keeps all.
    :param threshold: drop classes whose probability is below it; ``None`` drops none.
    :return: ``{"labels", "scores"}``. For a single image both are flat lists, best class
        first (ties keep class order); for several images, one such list per image.
        ``scores`` are softmax probabilities of the model's logits, as plain floats.
    :raises ConfigError: for an unreadable file, an unsupported array, an empty list, a
        ``top_k`` that is not a positive int, a ``threshold`` outside ``[0, 1]``, or a
        model whose output width is not the number of class names.

``Predictor.model``
    The loaded Keras model.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); every
``keras``, ``tensorflow`` and ``train.common`` name is imported inside a function.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import registry
from .errors import ConfigError
from .run_dir import TASK_FILE

# ---------------------------------------------------------------------

#: The only ``schema_version`` this module reads and writes.
SCHEMA_VERSION = 1

#: The keys of a record, in the order they are written.
RECORD_KEYS = (
    "schema_version",
    "task",
    "model",
    "family",
    "variant",
    "class_names",
    "input_shape",
    "channels",
    "normalize",
    "mean",
    "std",
    "eval_transform",
)

#: The evaluation transforms this predictor reproduces, by the loader code each one is:
#: ``resize`` is ``train.common.classification_data.resizer`` (the in-memory datasets,
#: folder and csv data); ``resize_centre_crop`` is
#: ``dl_techniques.datasets.vision.crops.resize_and_centre_crop`` (Imagenette);
#: ``imagenet_centre_crop`` is ``dl_techniques.datasets.vision.imagenet2012.center_crop``
#: (ImageNet's antialiased crop).
EVAL_KINDS = ("resize", "resize_centre_crop", "imagenet_centre_crop")

#: The evaluation transform each data source records.
# DECISION plan-2026-10-30T132432-a865fa08/D-062: ImageNet records ``imagenet_centre_crop`` and
# the predictor calls the loader's own ``center_crop`` for it. Do NOT record ``resize`` for
# ImageNet and do NOT approximate its crop with Imagenette's (an unrounded shorter side and no
# antialias): a prediction through another crop looks fine and is wrong. Each kind names the
# one loader function the predictor calls, so a new kind is a new row here and a branch in
# ``Predictor._shaped``, never a refusal at read.
EVAL_KIND_OF_DATA = {
    "mnist": "resize",
    "cifar10": "resize",
    "cifar100": "resize",
    "folder": "resize",
    "csv": "resize",
    "imagenette": "resize_centre_crop",
    "imagenet": "imagenet_centre_crop",
}

#: Images turned into tensors and run through the model at a time.
CHUNK = 32

_NORMALIZE = ("standardize", "unit")


def build_task_record(
    *,
    task: str,
    model: str,
    family: str,
    variant: str,
    class_names: Sequence[Any],
    input_shape: Sequence[int],
    normalize: str,
    mean: Any,
    std: Any,
    data_kind: str,
    source_channels: int,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a run; see the module contract.

    :param data_kind: a key of :data:`EVAL_KIND_OF_DATA`: the dataset name, or ``folder``
        or ``csv``.
    :param source_channels: the channels the files are read with.
    :return: the record. It is NOT checked: ImageNet's antialiased centre crop is recorded
        truthfully; every kind it can hold is one :func:`check_record` accepts.
    """
    shape = [int(side) for side in input_shape]
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "model": model,
        "family": family,
        "variant": variant,
        "class_names": [str(name) for name in class_names],
        "input_shape": shape,
        "channels": shape[2],
        "normalize": normalize,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "eval_transform": {
            "kind": EVAL_KIND_OF_DATA[data_kind],
            "source_channels": int(source_channels),
        },
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def record_problems(record: Mapping[str, Any]) -> List[str]:
    """Return one line per way ``record`` is not a valid version-1 record (empty when valid)."""
    lines: List[str] = []
    version = record.get("schema_version")
    if version != SCHEMA_VERSION or isinstance(version, bool):
        return [
            f"'schema_version' is {version!r}; this reader understands {SCHEMA_VERSION} only"
        ]
    missing = [key for key in RECORD_KEYS if key not in record]
    unknown = sorted(set(record) - set(RECORD_KEYS))
    if missing:
        lines.append(f"missing key(s) {', '.join(map(repr, missing))}")
    if unknown:
        lines.append(f"unknown key(s) {', '.join(map(repr, unknown))}")
    if missing:
        return lines
    for key in ("task", "model", "family", "variant"):
        if not isinstance(record[key], str) or not record[key]:
            lines.append(f"{key!r} must be a non-empty text, got {record[key]!r}")
    names = record["class_names"]
    if not (isinstance(names, list) and names and all(isinstance(n, str) for n in names)):
        lines.append(f"'class_names' must be a non-empty list of texts, got {names!r}")
    shape = record["input_shape"]
    shape_ok = (
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(side, int) and not isinstance(side, bool) and side > 0 for side in shape)
        and shape[0] == shape[1]
    )
    if not shape_ok:
        lines.append(f"'input_shape' must be [side, side, channels] of positive ints, got {shape!r}")
    elif record["channels"] != shape[2]:
        lines.append(f"'channels' is {record['channels']!r} but 'input_shape' has {shape[2]}")
    if record["normalize"] not in _NORMALIZE:
        lines.append(f"'normalize' must be one of {', '.join(_NORMALIZE)}, got {record['normalize']!r}")
    width = shape[2] if shape_ok else None
    for key in ("mean", "std"):
        values = record[key]
        if not (isinstance(values, list) and values and all(_is_number(v) for v in values)):
            lines.append(f"{key!r} must be a list of finite numbers, got {values!r}")
        elif width is not None and len(values) != width:
            lines.append(f"{key!r} holds {len(values)} numbers for {width} channels")
    std = record["std"]
    if isinstance(std, list) and all(_is_number(v) for v in std) and any(v <= 0 for v in std):
        lines.append(f"'std' must be positive, got {std!r}")
    transform = record["eval_transform"]
    if not isinstance(transform, dict) or set(transform) != {"kind", "source_channels"}:
        lines.append(
            f"'eval_transform' must be an object with 'kind' and 'source_channels', got {transform!r}"
        )
    else:
        if transform["kind"] not in EVAL_KINDS:
            lines.append(
                f"'eval_transform.kind' is {transform['kind']!r}, which this predictor does not "
                f"reproduce (it reproduces {', '.join(EVAL_KINDS)})"
            )
        if transform["source_channels"] not in (1, 3) or isinstance(transform["source_channels"], bool):
            lines.append(
                f"'eval_transform.source_channels' must be 1 or 3, got {transform['source_channels']!r}"
            )
    return lines


def check_record(record: Any, source: Union[str, Path]) -> None:
    """Raise :class:`ConfigError` unless ``record`` is a valid version-1 record; see the module contract."""
    lines = ["not a JSON object"] if not isinstance(record, dict) else record_problems(record)
    if lines:
        raise ConfigError(f"Invalid {TASK_FILE} {source}: " + "; ".join(lines))


def read_record(path: Path) -> Dict[str, Any]:
    """Return the checked record in the file ``path``.

    :raises ConfigError: when the file is missing, is not JSON, or fails :func:`check_record`.
    """
    if not path.is_file():
        raise ConfigError(
            f"Invalid value for 'model': {path} does not exist; a model is loaded from a run "
            f"directory, or from a .keras file whose directory holds {TASK_FILE}"
        )
    try:
        record = json.loads(path.read_text())
    except ValueError as exc:
        raise ConfigError(f"Invalid {TASK_FILE} {path}: not JSON ({exc})") from exc
    check_record(record, path)
    return record


def _model_files(path: Path) -> Tuple[Path, Path]:
    """Return ``(model file, task file)`` for a run directory or a ``.keras`` file."""
    from train.common.callbacks import best_checkpoint_path

    if path.is_dir():
        model_file = Path(best_checkpoint_path(str(path)))
        if not model_file.is_file():
            raise ConfigError(
                f"Invalid value for 'model': {path} holds no {model_file.name} (the run had "
                "checkpoint_args.save_best off?); pass the .keras file you want, e.g. "
                f"{path / 'final_model.keras'}"
            )
        return model_file, path / TASK_FILE
    if path.is_file():
        return path, path.parent / TASK_FILE
    raise ConfigError(f"Invalid value for 'model': {path} is neither a run directory nor a file")


def load_model(path: Union[str, Path]) -> "Predictor":
    """Load a trained model and its ``task.json``; see the module contract."""
    import keras

    from . import recipes

    model_file, task_file = _model_files(Path(path))
    record = read_record(task_file)
    # The family's model code is imported BEFORE the Keras file is read, so the classes
    # the archive names are registered whatever the caller imported (D-051).
    recipes.register_builtin_models()
    spec = registry.MODELS.get(record["family"])
    if spec is None:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: the family {record['family']!r} is not registered "
            f"(registered: {', '.join(sorted(registry.MODELS)) or 'none'})"
        )
    spec.load_factory()
    # DECISION plan-2026-10-30T132432-a865fa08/D-061: ``compile=False``. Do NOT load the compile config: a
    # predictor needs no loss or metric, and a metric class the archive names (the
    # library's half-precision top-k, D-051) would be one more thing that can fail a
    # reload that needs only the weights.
    return Predictor(keras.models.load_model(str(model_file), compile=False), record)


def _softmax(logits: Any) -> Any:
    import numpy as np

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class Predictor:
    """A loaded model and the preprocessing it was trained with; see the module contract.

    :param model: the Keras model; its outputs are logits.
    :param record: a record that passed :func:`check_record`.
    """

    def __init__(self, model: Any, record: Mapping[str, Any]) -> None:
        check_record(dict(record), "<given>")
        self.model = model
        self.record = dict(record)
        self.class_names: List[str] = list(record["class_names"])

    # -----------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------

    def _to_source_channels(self, image: Any) -> Any:
        """Return ``image`` ``(H, W, C)`` with ``eval_transform.source_channels`` channels."""
        import tensorflow as tf

        wanted = self.record["eval_transform"]["source_channels"]
        have = int(image.shape[-1])
        if have == wanted:
            return image
        if have == 3 and wanted == 1:
            return tf.image.rgb_to_grayscale(image)
        return tf.repeat(image, wanted, axis=-1)

    def _from_array(self, array: Any) -> Any:
        """Return one array as a float32 ``[0, 1]`` tensor ``(H, W, source_channels)``."""
        import numpy as np
        import tensorflow as tf

        if array.dtype != np.uint8 and not np.issubdtype(array.dtype, np.floating):
            raise ConfigError(
                f"Invalid image: an array must be uint8 or floating point, got {array.dtype}"
            )
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3 or array.shape[-1] not in (1, 3) or min(array.shape[:2]) < 1:
            raise ConfigError(
                "Invalid image: an array must be (H, W), (H, W, 1) or (H, W, 3), got shape "
                f"{tuple(array.shape)}"
            )
        if array.dtype == np.uint8:
            image = tf.image.convert_image_dtype(tf.constant(array), tf.float32)
        else:
            if not (np.all(array >= 0.0) and np.all(array <= 1.0)):
                raise ConfigError(
                    "Invalid image: a floating array must lie in [0, 1] (the scale of a "
                    "uint8 array divided by 255), got values outside it or NaN"
                )
            image = tf.constant(array.astype(np.float32))
        return self._to_source_channels(image)

    def _from_path(self, path: Union[str, "os.PathLike[str]"]) -> Any:
        import tensorflow as tf

        from train.common.classification_data import decode_image_file

        if not os.path.isfile(path):
            raise ConfigError(f"Invalid image: {os.fspath(path)!r} is not a file")
        channels = self.record["eval_transform"]["source_channels"]
        try:
            return decode_image_file(os.fspath(path), channels)
        except tf.errors.InvalidArgumentError as exc:
            raise ConfigError(
                f"Invalid image: {os.fspath(path)!r} is not an image file tf.io.decode_image reads"
            ) from exc

    def _one(self, item: Any) -> Any:
        """Return one image (a path, a PIL image or a 2-D or 3-D array) as a tensor."""
        import numpy as np
        from PIL import Image

        if isinstance(item, (str, os.PathLike)):
            return self._from_path(item)
        if isinstance(item, Image.Image):
            mode = "L" if self.record["eval_transform"]["source_channels"] == 1 else "RGB"
            return self._from_array(np.asarray(item.convert(mode)))
        if isinstance(item, np.ndarray) and item.ndim in (2, 3):
            return self._from_array(item)
        raise ConfigError(
            "Invalid image: expected a path, a PIL image, or a 2-D or 3-D numpy array, got "
            f"{type(item).__name__}"
            + (f" of shape {tuple(item.shape)}" if hasattr(item, "shape") else "")
        )

    def _images_of(self, x: Any) -> Tuple[List[Any], bool]:
        """Return ``(images, several)`` for the argument of :meth:`predict`."""
        import numpy as np

        if isinstance(x, np.ndarray) and x.ndim == 4:
            return [self._one(image) for image in x], True
        if isinstance(x, (list, tuple)):
            if not x:
                raise ConfigError("Invalid image: the list of images is empty")
            return [self._one(item) for item in x], True
        return [self._one(x)], False

    # -----------------------------------------------------------------
    # Preprocessing and scores
    # -----------------------------------------------------------------

    def _shaped(self, image: Any) -> Any:
        """Return ``image`` resized the way the evaluation pipeline did, ``(S, S, channels)``."""
        import tensorflow as tf

        from dl_techniques.datasets.vision.crops import resize_and_centre_crop
        from dl_techniques.datasets.vision.imagenet2012 import center_crop
        from train.common.classification_data import resizer

        side = self.record["input_shape"][0]
        kind = self.record["eval_transform"]["kind"]
        if kind == "resize":
            image = resizer(side)(image)
        elif kind == "resize_centre_crop":
            image = resize_and_centre_crop(image, side)
        else:
            image = center_crop(image, side)
        wanted = self.record["channels"]
        if int(image.shape[-1]) != wanted:
            image = tf.repeat(image, wanted, axis=-1)
        return image

    def probabilities(self, images: Sequence[Any]) -> Any:
        """Return the softmax probabilities ``(N, classes)`` of ``images`` (float64 ``numpy``).

        :param images: tensors from :meth:`_one`: float32 in ``[0, 1]``, any size.
        """
        import numpy as np
        import tensorflow as tf

        mean = tf.constant(np.asarray(self.record["mean"], np.float32))
        std = tf.constant(np.asarray(self.record["std"], np.float32))
        rows = []
        for start in range(0, len(images), CHUNK):
            batch = tf.stack([self._shaped(image) for image in images[start : start + CHUNK]])
            logits = self.model((batch - mean) / std, training=False)
            rows.append(np.asarray(logits, dtype=np.float64))
        logits = np.concatenate(rows, axis=0)
        if logits.ndim != 2 or logits.shape[1] != len(self.class_names):
            raise ConfigError(
                f"The model outputs {tuple(logits.shape[1:])} per image but {TASK_FILE} names "
                f"{len(self.class_names)} classes; the file and the model are not from one run"
            )
        return _softmax(logits)

    def predict(
        self, x: Any, top_k: Optional[int] = None, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Return ``{"labels", "scores"}`` for ``x``; see the module contract."""
        import numpy as np

        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
            raise ConfigError(f"Invalid value for 'top_k': expected a positive int or None, got {top_k!r}")
        if threshold is not None and (
            isinstance(threshold, bool) or not _is_number(threshold) or not 0 <= threshold <= 1
        ):
            raise ConfigError(
                f"Invalid value for 'threshold': expected a number in [0, 1] or None, got {threshold!r}"
            )
        images, several = self._images_of(x)
        probabilities = self.probabilities(images)
        labels: List[List[str]] = []
        scores: List[List[float]] = []
        for row in probabilities:
            order = np.argsort(-row, kind="stable")[:top_k]
            kept = [int(i) for i in order if threshold is None or row[i] >= threshold]
            labels.append([self.class_names[i] for i in kept])
            scores.append([float(row[i]) for i in kept])
        if several:
            return {"labels": labels, "scores": scores}
        return {"labels": labels[0], "scores": scores[0]}

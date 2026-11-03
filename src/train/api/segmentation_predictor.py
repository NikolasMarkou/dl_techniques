"""Predict with a trained segmentation run: :class:`SegmentationPredictor` and the PNG writer of ``predict``.

``train.api.predictor.load_model`` returns a :class:`SegmentationPredictor` when the run's
``task.json`` says ``image_segmentation``. It applies the preprocessing the run trained with
(the decoder of ``train.common.segmentation_data``, the same bilinear resize to ``input_size``, the
record's ``mean`` and ``std``), runs the model with ``training=False`` and returns, for every image,
one class id per pixel at the size of THAT image.

Interface contract
------------------
``SegmentationPredictor(model, record)``
    :param model: the loaded Keras model; its output is ``(batch, height, width, classes)`` logits at
        the record's ``input_size``.
    :param record: a segmentation record that passed :func:`train.api.predictor.check_record`.
    :raises ConfigError: for a record of another task, or one whose ``channels`` is not 3 (the
        decoder always yields RGB; no segmentation run records another count).

``SegmentationPredictor.predict(x)``
    :param x: one image or several. An image is a path (``str`` or ``Path``), a ``PIL.Image.Image``,
        or a ``numpy`` array ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``; several are a
        list or tuple of those, or a 4-D array. A path and a PIL image are read by ONE function,
        ``segmentation_data.image_to_rgb`` (EXIF orientation applied, then PIL's conversion to RGB, so
        an RGBA image loses its alpha and a grey one is repeated), which is also the decoder of the
        training pipeline; an array is taken as it is: a ``uint8`` array as bytes, a floating array
        must lie in ``[0, 1]`` and is rounded to its 8-bit value (so the three forms of one picture
        feed the model the same bytes), and an RGBA array loses its alpha.
    :return: ``{"masks": [mask, ...], "class_names": [...]}``, ALWAYS a list of masks, one per image in
        the order given (a single image gives a list of one: a mask is already a 2-D array, and an
        answer that changed shape with the number of images would be a trap). A mask is a ``numpy``
        ``(height, width)`` array of class ids, ``uint8``, or ``uint16`` when the model has more than
        255 classes; ``height`` and ``width`` are those of the image AFTER its EXIF orientation. When
        the record's ``output`` is ``"logits"`` the key is ``"logits"`` instead and each entry is the
        float32 ``(height, width, classes)`` score map.
    :raises ConfigError: for an unreadable file, an unsupported array, an empty list, or a model whose
        output is not ``(batch, height, width, classes)`` at the record's sizes.

    The logits are resized to the image's size BILINEARLY and the argmax is taken AFTER, never before
    (D-080): an argmax resized by nearest neighbour is blocky at the model's resolution and discards
    the margin between classes, which is what the bilinear map of scores keeps. The resize runs in
    pieces of classes when ``height * width * classes`` is large (:data:`MAX_RESIZE_ELEMENTS`) and
    the pieces are combined exactly (first maximum wins, as ``argmax`` does), so a 150-class model on a
    12-megapixel photograph does not need gigabytes at once.

``resize_argmax(logits, size, max_elements=MAX_RESIZE_ELEMENTS)``
    :param logits: ``(h, w, classes)`` float32 scores.
    :param size: ``(height, width)`` to resize to.
    :param max_elements: the most floats one resized piece may hold.
    :return: ``int32`` ``(height, width)`` ids of the bilinear resize's per-pixel maximum, equal to
        the argmax of the whole resized map.

``check_png_output(record)``
    :raises ConfigError: unless the record's ``output`` is ``masks``.

``mask_names(files)``
    :param files: the input paths of a ``predict`` run.
    :return: ``["<stem>.png", ...]``, one per file.
    :raises ConfigError: when two files have one stem (``a/cat.jpg`` and ``b/cat.png``), naming
        both; the outputs would overwrite each other and nothing is ever overwritten.

``write_predictions(predictor, files, out)``
    Create the directory ``out`` (it must not exist), write ``<stem>.png`` for each of ``files`` (grey,
    8-bit; 16-bit above 255 classes) and, LAST, ``index.json``:
    ``{"class_names": [...], "masks": [{"path", "mask", "size": [height, width]}, ...]}``. Images
    are read and answered :data:`CHUNK` at a time. A run that fails half way leaves its PNGs and no
    ``index.json``: the index is the mark of a complete answer.

    :raises ConfigError: for a collision of stems, a record whose ``output`` is ``"logits"`` (a score
        map is not a PNG; use ``predict`` from Python), or any error of ``predict``.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only (D-012); ``tensorflow``, ``numpy``,
``PIL`` and ``train.common`` are imported inside functions. ``train.common.segmentation_data``
imports TensorFlow at module scope and costs 0.0 s over ``classification_data``, which every ``predict``
already imports for its file listing (D-078's measured trigger for splitting that module: not met,
because a model cannot run without TensorFlow either).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from .errors import ConfigError
from .exporting import check_free
from .predictor import SEGMENTATION, check_record
from .run_dir import TASK_FILE

#: Images turned into tensors and run through the model at a time (a segmentation batch is a map per image).
CHUNK = 8

#: The most floats one resized piece of the logits may hold (256 MiB as float32).
MAX_RESIZE_ELEMENTS = 1 << 26

#: The name of the index :func:`write_predictions` writes last.
INDEX_NAME = "index.json"

#: Above this many classes a mask needs 16 bits.
MAX_BYTE_CLASSES = 255


def mask_dtype(num_classes: int) -> str:
    """Return the numpy dtype name of a mask of ``num_classes`` classes: ``uint8`` or ``uint16``."""
    return "uint8" if num_classes <= MAX_BYTE_CLASSES else "uint16"


def check_output(logits: Any, record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless ``logits`` is ``(N, height, width, classes)`` for the record."""
    shape = tuple(logits.shape)
    wanted = (*record["input_size"], len(record["class_names"]))
    if len(shape) != 4 or shape[1:] != wanted:
        raise ConfigError(
            f"The model outputs {shape[1:]} per image but {TASK_FILE} names input_size "
            f"{list(record['input_size'])} and {len(record['class_names'])} classes, so it should "
            f"output {wanted}; the file and the model are not from one run"
        )


def resize_argmax(logits: Any, size: Tuple[int, int], max_elements: int = MAX_RESIZE_ELEMENTS) -> Any:
    """Return the class ids of ``logits`` resized bilinearly to ``size``; see the module contract."""
    import tensorflow as tf

    # DECISION plan-2026-10-31T180033-2faf890a/D-080
    # The scores are resized, THEN the argmax is taken. Do NOT take the argmax at the model's
    # resolution and resize the ids by nearest neighbour: that is blocky at the model's resolution
    # and throws away the margin between classes that the bilinear map of scores keeps (a hand
    # fixture where the two disagree pins it). Do NOT resize all classes at once: a 150-class model
    # on a 12-megapixel photograph is 7 GiB; pieces of classes combined with a strict `>` give the
    # answer of one piece (first maximum wins, as `argmax` does).
    height, width = size
    classes = int(logits.shape[-1])
    step = max(1, max_elements // max(1, height * width))
    best = ids = None
    for start in range(0, classes, step):
        piece = tf.image.resize(
            tf.convert_to_tensor(logits)[None, :, :, start : start + step], (height, width), method="bilinear"
        )[0]
        local = tf.argmax(piece, axis=-1, output_type=tf.int32) + start
        top = tf.reduce_max(piece, axis=-1)
        if best is None:
            best, ids = top, local
        else:
            # A strict comparison: on a tie the earlier class keeps the pixel, as ``argmax`` does.
            better = top > best
            best, ids = tf.where(better, top, best), tf.where(better, local, ids)
    return ids.numpy()


def _array_to_rgb(array: Any) -> Any:
    """Return an array ``(H, W)`` or ``(H, W, 1|3|4)`` as ``uint8`` ``(H, W, 3)``."""
    import numpy as np

    if array.dtype != np.uint8 and not np.issubdtype(array.dtype, np.floating):
        raise ConfigError(f"Invalid image: an array must be uint8 or floating point, got {array.dtype}")
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4) or min(array.shape[:2]) < 1:
        raise ConfigError(
            "Invalid image: an array must be (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), got "
            f"shape {tuple(array.shape)}"
        )
    if array.dtype != np.uint8:
        if not (np.all(array >= 0.0) and np.all(array <= 1.0)):
            raise ConfigError(
                "Invalid image: a floating array must lie in [0, 1] (the scale of a uint8 array "
                "divided by 255), got values outside it or NaN"
            )
        array = np.rint(array.astype(np.float64) * 255.0).astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    elif array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.ascontiguousarray(array)


class SegmentationPredictor:
    """A loaded segmentation model and the preprocessing it was trained with; see the module contract.

    :param model: the Keras model; its outputs are per-pixel logits.
    :param record: a segmentation record that passed :func:`train.api.predictor.check_record`.
    """

    def __init__(self, model: Any, record: Mapping[str, Any]) -> None:
        check_record(dict(record), "<given>")
        if record["task"] != SEGMENTATION:
            raise ConfigError(
                f"Invalid {TASK_FILE} <given>: the task is {record['task']!r} but this predictor "
                f"serves {SEGMENTATION!r} only"
            )
        if record["channels"] != 3:
            raise ConfigError(
                f"Invalid {TASK_FILE} <given>: 'channels' is {record['channels']}; the segmentation "
                "decoder yields RGB images, so only 3 can be fed to the model"
            )
        self.model = model
        self.record = dict(record)
        self.class_names: List[str] = list(record["class_names"])

    # -----------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------

    def _rgb_of(self, item: Any) -> Any:
        """Return one image (a path, a PIL image or a 2-D or 3-D array) as ``uint8`` ``(H, W, 3)``."""
        import numpy as np
        from PIL import Image

        from train.common.segmentation_data import PairError, decode_image, image_to_rgb

        if isinstance(item, (str, os.PathLike)):
            if not os.path.isfile(item):
                raise ConfigError(f"Invalid image: {os.fspath(item)!r} is not a file")
            try:
                return decode_image(os.fspath(item))
            except PairError as exc:
                raise ConfigError(f"Invalid image: {exc}") from exc
        if isinstance(item, Image.Image):
            return image_to_rgb(item)
        if isinstance(item, np.ndarray) and item.ndim in (2, 3):
            return _array_to_rgb(item)
        raise ConfigError(
            "Invalid image: expected a path, a PIL image, or a 2-D or 3-D numpy array, got "
            f"{type(item).__name__}"
            + (f" of shape {tuple(item.shape)}" if hasattr(item, "shape") else "")
        )

    @staticmethod
    def _items_of(x: Any) -> List[Any]:
        """Return the images of the argument of :meth:`predict`, still undecoded."""
        import numpy as np

        if isinstance(x, np.ndarray) and x.ndim == 4:
            if len(x) == 0:
                raise ConfigError(f"Invalid image: the array of images is empty (shape {tuple(x.shape)})")
            return list(x)
        if isinstance(x, (list, tuple)):
            if not x:
                raise ConfigError("Invalid image: the list of images is empty")
            return list(x)
        return [x]

    # -----------------------------------------------------------------
    # The model
    # -----------------------------------------------------------------

    def _logits(self, images: Sequence[Any]) -> Any:
        """Return the model's float32 logits ``(N, height, width, classes)`` for ``uint8`` RGB ``images``."""
        import numpy as np
        import tensorflow as tf

        from train.common.segmentation_data import resize_image_bytes

        size = tuple(self.record["input_size"])
        mean = tf.constant(np.asarray(self.record["mean"], np.float32))
        std = tf.constant(np.asarray(self.record["std"], np.float32))
        # The training pipeline's own resize (`resize_pair`): bilinear, antialiased, rounded to a byte.
        batch = tf.stack([resize_image_bytes(tf.constant(image), size) / 255.0 for image in images])
        logits = tf.cast(self.model((batch - mean) / std, training=False), tf.float32)
        check_output(logits, self.record)
        return logits

    def _answer(self, logits: Any, size: Tuple[int, int]) -> Any:
        """Return the mask (or the score map) of one image's ``logits`` at ``size``."""
        import tensorflow as tf

        if self.record["output"] == "logits":
            return tf.image.resize(logits[None], size, method="bilinear")[0].numpy()
        return resize_argmax(logits, size).astype(mask_dtype(len(self.class_names)))

    def predict(self, x: Any) -> Dict[str, Any]:
        """Return ``{"masks", "class_names"}`` (or ``{"logits", ...}``) for ``x``; see the module contract."""
        items = self._items_of(x)
        answers: List[Any] = []
        for start in range(0, len(items), CHUNK):
            images = [self._rgb_of(item) for item in items[start : start + CHUNK]]
            logits = self._logits(images)
            for image, one in zip(images, logits):
                answers.append(self._answer(one, (int(image.shape[0]), int(image.shape[1]))))
        key = "logits" if self.record["output"] == "logits" else "masks"
        return {key: answers, "class_names": list(self.class_names)}


# ---------------------------------------------------------------------
# The command line's output
# ---------------------------------------------------------------------

def mask_names(files: Sequence[Union[str, "os.PathLike[str]"]]) -> List[str]:
    """Return ``<stem>.png`` for each of ``files``; see the module contract."""
    names: List[str] = []
    seen: Dict[str, str] = {}
    for file in files:
        name = Path(file).stem + ".png"
        if name in seen:
            raise ConfigError(
                f"Invalid value for 'data': {seen[name]} and {os.fspath(file)} have the same name "
                f"without its extension, so both would be written to {name}; rename one (predict "
                "never overwrites)"
            )
        seen[name] = os.fspath(file)
        names.append(name)
    return names


def check_png_output(record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless the record's ``output`` is ``masks`` (a score map is no PNG)."""
    if record["output"] != "masks":
        raise ConfigError(
            f"Invalid value for 'model': this record's output is {record['output']!r}, a score map "
            "per image, which is not a PNG; call SegmentationPredictor.predict from Python"
        )


def _write_png(path: Path, mask: Any) -> None:
    from PIL import Image

    # "xb" creates the file and fails if it exists; Image.save(path) would truncate one.
    with open(path, "xb") as handle:
        Image.fromarray(mask).save(handle, format="PNG")


def write_predictions(predictor: SegmentationPredictor, files: Sequence[Path], out: Path) -> None:
    """Write one PNG per file and ``index.json`` into the new directory ``out``; see the module contract."""
    check_png_output(predictor.record)
    names = mask_names(files)
    check_free(out, "out", "predict")
    out.mkdir(parents=True)
    entries: List[Dict[str, Any]] = []
    for start in range(0, len(files), CHUNK):
        chunk = files[start : start + CHUNK]
        for file, name, mask in zip(
            chunk, names[start : start + CHUNK], predictor.predict([str(f) for f in chunk])["masks"]
        ):
            _write_png(out / name, mask)
            entries.append({"path": str(file), "mask": name, "size": [int(mask.shape[0]), int(mask.shape[1])]})
    with open(out / INDEX_NAME, "x") as handle:
        json.dump({"class_names": list(predictor.class_names), "masks": entries}, handle, indent=2)
        handle.write("\n")

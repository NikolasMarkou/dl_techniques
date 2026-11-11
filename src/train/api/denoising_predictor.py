"""Predict with a trained denoising run: :class:`DenoisingPredictor`, the PNG writer of ``predict`` and the
export rules of an ``image_denoising`` model.

``train.api.predictor.load_model`` returns a :class:`DenoisingPredictor` when the run's ``task.json`` says
``image_denoising``. A denoiser takes an image in ``[0, 1]`` (the domain it trained on: no mean, no standard
deviation, no sigma input, so the model is BLIND to the noise level) and returns the denoised image at the
size of that image. Nothing is clipped, standardized or rescaled on the way out.

Interface contract
------------------
``DenoisingPredictor(model, record)``
    :param model: the loaded Keras model; its input is ``(batch, height, width, channels)`` of any size
        (the run saves it flexible) and its one output is the image of the same shape.
    :param record: a denoising record that passed :func:`train.api.predictor.check_record`.
    :raises ConfigError: for a record of another task.

``DenoisingPredictor.predict(x, tile=None, tile_overlap=None)``
    :param x: one image or several. An image is a path (``str`` or ``Path``), a ``PIL.Image.Image``, or a
        ``numpy`` array ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)``; several are a list or tuple
        of those, or a 4-D array. A path is read by ``classification_data.decode_image_file`` (the decoder of
        the training pipeline, at the record's channel count: no EXIF rotation, as in training). A PIL image
        and a ``uint8`` array are the same bytes divided by 255, and a colour image given to a one-channel
        model is made grey with the loader's own integer rule (``rgb8_to_grey8``), so the three forms of one
        picture feed the model the same numbers. A FLOATING array is taken as it is and is NOT required to lie
        in ``[0, 1]``: a denoiser is fed noisy images, and a run with ``data.noise_clip=false`` trained on
        values outside the unit interval; only a non-finite value is refused. An RGBA input loses its alpha,
        and a grey input given to a three-channel model is repeated.
    :param tile: ``None`` (the default: the whole image in one forward pass) or the side, in pixels, of the
        square tiles the image is cut into. A multiple of the record's ``size_multiple``. An axis shorter than
        ``tile`` is one tile on that axis (an image smaller than the tile is a whole-image pass).
    :param tile_overlap: the pixels two neighbouring tiles share, a multiple of ``size_multiple`` below
        ``tile``; ``None`` is ``tile // 4`` rounded down to a multiple of ``size_multiple`` (zero for a tile
        under four strides: tiling WITHOUT overlap, whose seams are what the overlap removes). Given without
        ``tile`` it is refused.
    :return: ``{"denoised": [image, ...]}``, ALWAYS a list, one ``numpy`` ``float32`` array ``(height, width,
        channels)`` per input in the order given, at the size of THAT input, RAW: not clipped to ``[0, 1]``
        (PNG files made from it are clipped, at the uint8 boundary only).
    :raises ConfigError: for an unreadable file, an unsupported array, an empty list, a ``tile`` or
        ``tile_overlap`` that is not valid, a model whose output is not one image of the input's shape, and
        a forward pass that ran out of memory (one ``tf.errors.ResourceExhaustedError``, said to be a lack of
        memory and naming ``tile``).

    The image is reflect-padded to a multiple of ``size_multiple`` on each side (a U-Net fails on any other
    size: measured, plan d483ffab step 2(c)), the model runs with ``training=False`` and EAGERLY (a compiled
    call would build one XLA program per new size: 3 to 44 s each and up to 17 GB, step 2(c)), and the result
    is cropped back. A tiled pass cuts the PADDED image into tiles of ``tile`` pixels at a stride of
    ``tile - tile_overlap`` (the last tile of an axis is moved back so every tile has the same size), runs each,
    and blends the tiles with FIXED linear ramps over the overlap: the sum of ``weight * output`` divided by the
    sum of the weights. The weights do not depend on the pixels, so a tiled pass of a model that is
    homogeneous (``f(a x) = a f(x)``, every bias-free family) is homogeneous too.

``check_png_output(record)`` / ``write_predictions(predictor, files, out)``
    The command line's output. ``write_predictions`` creates the directory ``out`` (it must not exist), writes
    ``<stem>.png`` for each of ``files`` (8 bit, grey or RGB as the model's channels) and, LAST,
    ``index.json``: ``{"images": [{"path", "image", "size": [height, width]}, ...], "png": ...}``. A PNG is the
    raw output clipped to ``[0, 1]`` and rounded to a byte: the clip is at this boundary and nowhere else.
    :raises ConfigError: for two files with one stem (naming both), or any error of ``predict``.

``export_rule_of(record, format, shape)`` / ``export_note(record)``
    The row of ``train.api.predictor.EXPORT_RULES`` for this task: refusals of a (family, format, fixed or
    dynamic) cell, and the sentence the CLI logs beside an export. See :data:`EXPORT_REFUSALS`.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only (D-012); ``tensorflow``, ``numpy``, ``PIL``
and ``train.common`` are imported inside functions.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import ConfigError
from .exporting import check_free
from .predictor import DENOISING, GREY16_MODES, GREY_MODES, WIDE_MODES, check_record
from .run_dir import TASK_FILE
from .segmentation_predictor import INDEX_NAME, _write_png, mask_names

#: Images turned into tensors and run through the model at a time by ``write_predictions``.
CHUNK = 8

#: The default overlap of a tiled pass is this share of the tile (``tile // OVERLAP_DIVISOR``).
OVERLAP_DIVISOR = 4

#: What a PNG of ``predict`` says about its pixels, in ``index.json``.
PNG_NOTE = "the raw output clipped to [0, 1] and rounded to 8 bits; predict() returns the raw values"


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _array_to_float(array: Any, channels: int) -> Any:
    """Return an array ``(H, W)`` or ``(H, W, 1|3|4)`` as float32 ``(H, W, channels)``; see :meth:`predict`."""
    import numpy as np
    import tensorflow as tf

    from train.common.classification_data import GREY_WEIGHTS, rgb8_to_grey8

    if array.dtype != np.uint8 and not np.issubdtype(array.dtype, np.floating):
        raise ConfigError(f"Invalid image: an array must be uint8 or floating point, got {array.dtype}")
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4) or min(array.shape[:2]) < 1:
        raise ConfigError(
            f"Invalid image: an array must be (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), got shape {tuple(array.shape)}"
        )
    if array.dtype != np.uint8 and not np.all(np.isfinite(array)):
        raise ConfigError("Invalid image: a floating array holds NaN or an infinite value")
    if array.shape[-1] == 4:
        array = array[..., :3]  # the alpha is dropped, as PIL's convert("RGB") drops it
    if array.shape[-1] == 3 and channels == 1:
        if array.dtype == np.uint8:
            array = rgb8_to_grey8(array)  # the loader's own integer rule (classification_data.GREY_WEIGHTS)
        else:
            weights = np.asarray(GREY_WEIGHTS, np.float64) / float(1 << 15)
            array = (array.astype(np.float64) @ weights)[..., None]
    if array.dtype == np.uint8:
        # The decoder's own byte to float step, so a path, a PIL image and a uint8 array give identical floats.
        array = tf.image.convert_image_dtype(tf.constant(array), tf.float32).numpy()
    array = np.asarray(array, np.float32)
    if array.shape[-1] != channels:  # one channel given to a three-channel model
        array = np.repeat(array, channels, axis=-1)
    return np.ascontiguousarray(array)


def _pil_to_float(image: Any, channels: int) -> Any:
    """Return a PIL image as float32 ``(H, W, channels)`` by the same bytes a file of it would give."""
    import numpy as np

    if image.mode in WIDE_MODES:
        raise ConfigError(
            f"Invalid image: a PIL image of mode {image.mode} holds 32-bit values and its bit depth cannot be "
            "preserved; convert it to an 8-bit mode ('L' or 'RGB') yourself, or pass the file's path"
        )
    if image.mode in GREY16_MODES:
        # The high byte of each sample, as tf.io.decode_image reads a 16-bit file (PIL's convert() clips to 255).
        return _array_to_float((np.asarray(image).astype(np.uint16) >> 8).astype(np.uint8), channels)
    grey = channels == 1 and image.mode in GREY_MODES
    return _array_to_float(np.asarray(image.convert("L" if grey else "RGB")), channels)


def _path_to_float(path: Any, channels: int) -> Any:
    import tensorflow as tf

    from train.common.classification_data import decode_image_file

    if not os.path.isfile(path):
        raise ConfigError(f"Invalid image: {os.fspath(path)!r} is not a file")
    try:
        return decode_image_file(os.fspath(path), channels).numpy()
    except tf.errors.InvalidArgumentError as exc:
        raise ConfigError(f"Invalid image: {os.fspath(path)!r} is not an image file tf.io.decode_image reads") from exc


# ---------------------------------------------------------------------
# Padding, tiles and blending (pure numpy)
# ---------------------------------------------------------------------


def reflect_pad(image: Any, multiple: int) -> Any:
    """Return ``image`` ``(H, W, C)`` reflect-padded at the end of each side to a multiple of ``multiple``."""
    import numpy as np

    pad_h = (multiple - image.shape[0] % multiple) % multiple
    pad_w = (multiple - image.shape[1] % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")


def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    """Return where the tiles of one axis of ``length`` pixels start: every ``stride``, the last one moved back.

    An axis not longer than ``tile`` is one tile at 0. Every tile but possibly the last is ``stride`` after the
    one before, and the last ends exactly at ``length``, so all tiles have the size ``min(tile, length)``.
    """
    if length <= tile:
        return [0]
    return [*range(0, length - tile, stride), length - tile]


def ramp_weights(start: int, size: int, length: int, overlap: int) -> Any:
    """Return the 1-D blending weights of the tile ``[start, start + size)`` of an axis of ``length`` pixels.

    1 inside; on a side that has a neighbouring tile, a linear ramp over ``overlap`` pixels that rises
    ``1/(overlap+1) .. overlap/(overlap+1)`` towards the inside (never 0: a pixel that only one tile covers
    keeps a positive weight). The image's own edges are not ramped. The weights are FIXED numbers that do not
    depend on the pixels.
    """
    import numpy as np

    weights = np.ones(size, np.float64)
    if overlap > 0:
        ramp = np.arange(1, overlap + 1, dtype=np.float64) / (overlap + 1)
        if start > 0:
            weights[:overlap] = ramp
        if start + size < length:
            weights[size - overlap :] = np.minimum(weights[size - overlap :], ramp[::-1])
    return weights


# ---------------------------------------------------------------------
# The predictor
# ---------------------------------------------------------------------


class DenoisingPredictor:
    """A loaded denoiser; see the module contract.

    :param model: the Keras model; its one output is the denoised image.
    :param record: a denoising record that passed :func:`train.api.predictor.check_record`.
    """

    def __init__(self, model: Any, record: Mapping[str, Any]) -> None:
        check_record(dict(record), "<given>")
        if record["task"] != DENOISING:
            raise ConfigError(
                f"Invalid {TASK_FILE} <given>: the task is {record['task']!r} but this predictor serves {DENOISING!r} only"
            )
        self.model = model
        self.record = dict(record)
        self.channels: int = int(record["channels"])
        self.size_multiple: int = int(record["size_multiple"])

    # -----------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------

    def _image_of(self, item: Any) -> Any:
        """Return one image (a path, a PIL image or a 2-D or 3-D array) as float32 ``(H, W, channels)``."""
        import numpy as np
        from PIL import Image

        if isinstance(item, (str, os.PathLike)):
            return _path_to_float(item, self.channels)
        if isinstance(item, Image.Image):
            return _pil_to_float(item, self.channels)
        if isinstance(item, np.ndarray) and item.ndim in (2, 3):
            return _array_to_float(item, self.channels)
        raise ConfigError(
            "Invalid image: expected a path, a PIL image, or a 2-D or 3-D numpy array, got "
            f"{type(item).__name__}" + (f" of shape {tuple(item.shape)}" if hasattr(item, "shape") else "")
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

    def _tiling(self, tile: Optional[int], tile_overlap: Optional[int]) -> Optional[Tuple[int, int]]:
        """Return ``(tile, overlap)`` or ``None`` for a whole-image pass; refuse what cannot be tiled."""
        multiple = self.size_multiple
        if tile is None:
            if tile_overlap is not None:
                raise ConfigError("Invalid value for 'tile_overlap': it needs 'tile' (it is the overlap of the tiles)")
            return None
        if not _is_int(tile) or tile < 1:
            raise ConfigError(f"Invalid value for 'tile': expected a positive int or None, got {tile!r}")
        if tile % multiple:
            raise ConfigError(
                f"Invalid value for 'tile': {tile} is not a multiple of {multiple}, the size_multiple of this model "
                f"({self.record['family']}); a U-Net takes only sides that are, so try {max(1, tile // multiple) * multiple}"
            )
        if tile_overlap is None:
            tile_overlap = (tile // OVERLAP_DIVISOR) // multiple * multiple
        if not _is_int(tile_overlap) or tile_overlap < 0:
            raise ConfigError(f"Invalid value for 'tile_overlap': expected a whole number of 0 or more, got {tile_overlap!r}")
        if tile_overlap % multiple:
            raise ConfigError(
                f"Invalid value for 'tile_overlap': {tile_overlap} is not a multiple of {multiple}, the size_multiple "
                "of this model; tiles that start off the model's stride see another sampling of the image"
            )
        if tile_overlap >= tile:
            raise ConfigError(f"Invalid value for 'tile_overlap': {tile_overlap} must be below tile ({tile})")
        return tile, tile_overlap

    # -----------------------------------------------------------------
    # The model
    # -----------------------------------------------------------------

    def _forward(self, batch: Any, how: str) -> Any:
        """Return the model's float32 output for ``batch`` ``(N, H, W, C)``; the one place memory can run out."""
        import tensorflow as tf

        try:
            output = self.model(batch, training=False)
        except tf.errors.ResourceExhaustedError as exc:
            # DECISION plan-2026-11-10T095307-d483ffab/D-030
            # Out of memory is ONE error type caught here and turned into the one remedy this API has: tile. Do NOT
            # catch more (a broad handler would hide a real fault as a memory problem) and do NOT retry with a
            # smaller tile on its own (the tile changes the numbers a little; the caller should choose that).
            raise ConfigError(
                f"Out of memory running the model on {how}; process the image in tiles by calling "
                f"DenoisingPredictor.predict(x, tile=<a multiple of {self.size_multiple}>), for example tile=256 "
                "(the command line predicts whole images)"
            ) from exc
        if isinstance(output, (list, tuple)) or tuple(output.shape) != tuple(batch.shape):
            seen = f"{len(output)} outputs" if isinstance(output, (list, tuple)) else str(tuple(output.shape))
            raise ConfigError(
                f"The model outputs {seen} for an input of shape {tuple(batch.shape)} but {TASK_FILE} names a denoiser "
                "(one output of the input's shape); the file and the model are not from one run"
            )
        return tf.cast(output, tf.float32).numpy()

    def _whole(self, padded: Any) -> Any:
        return self._forward(padded[None], f"a {padded.shape[0]}x{padded.shape[1]} image")[0]

    def _tiled(self, padded: Any, tile: int, overlap: int) -> Any:
        """Run ``padded`` ``(H, W, C)`` tile by tile and blend; see the module contract."""
        import numpy as np

        height, width = padded.shape[:2]
        stride = tile - overlap
        total = np.zeros(padded.shape, np.float64)
        weight_sum = np.zeros((height, width, 1), np.float64)
        for top in tile_starts(height, tile, stride):
            size_h = min(tile, height)
            wy = ramp_weights(top, size_h, height, overlap)
            for left in tile_starts(width, tile, stride):
                size_w = min(tile, width)
                wx = ramp_weights(left, size_w, width, overlap)
                piece = self._forward(
                    padded[top : top + size_h, left : left + size_w][None],
                    f"a {size_h}x{size_w} tile of a {height}x{width} image",
                )[0]
                weight = (wy[:, None] * wx[None, :])[..., None]
                total[top : top + size_h, left : left + size_w] += weight * piece
                weight_sum[top : top + size_h, left : left + size_w] += weight
        return (total / weight_sum).astype(np.float32)

    def predict(self, x: Any, tile: Optional[int] = None, tile_overlap: Optional[int] = None) -> Dict[str, Any]:
        """Return ``{"denoised": [...]}`` for ``x``; see the module contract."""
        tiling = self._tiling(tile, tile_overlap)
        denoised: List[Any] = []
        for item in self._items_of(x):
            image = self._image_of(item)
            height, width = image.shape[:2]
            padded = reflect_pad(image, self.size_multiple)
            result = self._whole(padded) if tiling is None else self._tiled(padded, *tiling)
            denoised.append(result[:height, :width, :])  # the crop back: the answer is at the input's own size
        return {"denoised": denoised}


# ---------------------------------------------------------------------
# The command line's output
# ---------------------------------------------------------------------


def check_png_output(record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless the record's ``output`` is ``images`` (the only one a denoising run has)."""
    if record["output"] != "images":
        raise ConfigError(
            f"Invalid value for 'model': this record's output is {record['output']!r}, which predict cannot write as PNG files"
        )


def to_uint8(image: Any) -> Any:
    """Return the raw ``(H, W, C)`` float output as a ``uint8`` picture: clipped to ``[0, 1]``, then rounded.

    The ONE place a denoiser's output is clipped: the file format cannot hold a value outside the byte range.
    A one-channel image is returned ``(H, W)`` (a grey PNG).
    """
    import numpy as np

    pixels = np.rint(np.clip(np.asarray(image, np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)
    return pixels[..., 0] if pixels.shape[-1] == 1 else pixels


def write_predictions(predictor: DenoisingPredictor, files: Sequence[Path], out: Path) -> None:
    """Write one PNG per file and ``index.json`` into the new directory ``out``; see the module contract."""
    check_png_output(predictor.record)
    names = mask_names(files)
    check_free(out, "out", "predict")
    out.mkdir(parents=True)
    entries: List[Dict[str, Any]] = []
    for start in range(0, len(files), CHUNK):
        chunk = files[start : start + CHUNK]
        for file, name, image in zip(
            chunk, names[start : start + CHUNK], predictor.predict([str(f) for f in chunk])["denoised"]
        ):
            _write_png(out / name, to_uint8(image))
            entries.append({"path": str(file), "image": name, "size": [int(image.shape[0]), int(image.shape[1])]})
    with open(out / INDEX_NAME, "x") as handle:
        json.dump({"images": entries, "png": PNG_NOTE}, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------
# Export rules
# ---------------------------------------------------------------------

#: The (family, format, mode) cells of ``export`` that are REFUSED, each with the measured reason (plan d483ffab
#: step 2(b), re-proved by ``test_denoising_predictor_and_export.py``: each row has a test that builds the
#: artifact anyway and fails if the reason stops being true). ``mode`` is ``fixed`` (an integer height and
#: width) or ``dynamic`` (a ``None`` side). Every cell NOT listed exports and matches Keras: dynamic to 2.3e-6
#: (relative) at two sizes for every family and format, fixed to 2.3e-6 at its own size.
EXPORT_REFUSALS: Dict[Tuple[str, str, str], str] = {
    ("bfunet", "tflite", "fixed"): (
        "a fixed-shape tflite file is silently wrong at any size but its own: the interpreter resizes its input "
        "without an error and a U-Net then returns numbers that are 1.0 (relative) off (measured on bfunet/tiny)"
    ),
    ("bfconvunext", "tflite", "fixed"): (
        "a fixed-shape tflite file is silently wrong at any size but its own: the interpreter resizes its input "
        "without an error and a U-Net then returns numbers that are 0.52 (relative) off (measured on bfconvunext/tiny)"
    ),
    ("convunext_biased", "tflite", "fixed"): (
        "a fixed-shape tflite file is silently wrong at any size but its own: the interpreter resizes its input "
        "without an error and a U-Net then returns numbers that are 0.56 (relative) off (measured on "
        "convunext_biased/tiny)"
    ),
    ("scunet", "onnx", "dynamic"): (
        "an onnx file of SCUNet with a free height and width runs only where both sides are multiples of 64 (it "
        "fails in a Tile node at 96x80, 100x100 and 65x65; measured on scunet/tiny), and the file alone cannot "
        "say so; export it at a fixed height and width, or use saved_model or tflite, which take any size"
    ),
}


def export_mode(shape: Sequence[Optional[int]]) -> str:
    """Return ``dynamic`` when the height or the width is ``None`` and ``fixed`` otherwise."""
    return "dynamic" if shape[0] is None or shape[1] is None else "fixed"


def check_export(record: Mapping[str, Any], format: str, shape: Sequence[Optional[int]]) -> None:
    """Raise :class:`ConfigError` for a cell of the export table that is refused; see :data:`EXPORT_REFUSALS`.

    :param record: the checked denoising record.
    :param format: one of ``train.api.exporting.FORMATS``.
    :param shape: ``(height, width, channels)`` the export traces; ``None`` for a free height or width.
    :raises ConfigError: for a fixed side that is not a multiple of the record's ``size_multiple`` (a U-Net cannot
        be traced there), for a channel count other than the record's, and for a row of :data:`EXPORT_REFUSALS`.
    """
    if format == "keras":
        return  # the archive carries the model's own flexible input; shape and format are not in question
    mode = export_mode(shape)
    if shape[2] != record["channels"]:
        raise ConfigError(
            f"Invalid value for 'input_shape': {tuple(shape)} has {shape[2]} channels but this denoiser takes "
            f"{record['channels']}"
        )
    multiple = record["size_multiple"]
    for side, name in zip(shape[:2], ("height", "width")):
        if side is not None and side % multiple:
            raise ConfigError(
                f"Invalid value for 'input_shape': the {name} {side} is not a multiple of {multiple}, the "
                f"size_multiple of {record['family']}/{record['variant']}; a model of this family cannot be traced "
                f"at it (pad the image to a multiple of {multiple} and use that size)"
            )
    reason = EXPORT_REFUSALS.get((record["family"], format, mode))
    if reason is not None:
        raise ConfigError(f"Invalid value for 'format': {format!r} of a {mode} {record['family']} model is refused: {reason}")


def export_note(record: Mapping[str, Any]) -> str:
    """Return the sentence the command line logs beside an exported denoiser (what goes in, what comes out)."""
    multiple = record["size_multiple"]
    return (
        f"the exported denoiser takes float32 images (batch, height, width, {record['channels']}) with pixels in [0, 1] "
        "(no mean, no std, no sigma input: it is blind) and returns the denoised image RAW, not clipped to [0, 1]; "
        f"pad each side to a multiple of {multiple} before calling and crop after"
    )

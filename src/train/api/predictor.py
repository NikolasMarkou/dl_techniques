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
``task.json`` (schema version 2), a JSON object. Its keys depend on its ``task`` (the record
carries the task word, so a reader dispatches on it and needs no second source). A classification
record has EXACTLY the keys and order of a version 1 record; only ``schema_version`` moved.
:data:`RECORD_KEYS` lists them per task. Every key of the task must be there, and no other, apart
from the OPTIONAL ``variant_overrides``.

``schema_version``
    ``2``. Any other value is refused: a reader never guesses at a newer layout, and there is no
    reader for version 1 (D-004). A version 1 record gets one sentence naming the file and saying
    the run has to be trained again; ``predict``, ``export`` and ``resume`` all reach it
    (:func:`check_record`, :func:`refuse_old_record`).
``task``, ``model``, ``family``, ``variant``
    The task word (``image_classification`` or ``image_segmentation``), ``"family/variant"``, and
    its two halves. A task this reader does not know is named and refused.
``class_names``
    One text per class, in the order of the model's outputs. **Limit, stated because it cannot
    be checked:** the reader verifies the COUNT (against the model's output width) and that no
    name repeats, never the ORDER. A record whose names are a permutation of the right ones (two
    classes swapped by hand or by a bug in whatever wrote it) loads and predicts, and reports the
    wrong name for those classes; no property of the model or of the record tells the two apart.
    Trust the file only as far as you trust its writer; ``export`` and a training run write it
    from the data they loaded.
``input_shape`` (classification), ``input_size`` (segmentation), ``channels``
    Classification: ``input_shape`` is ``[side, side, channels]`` of the model's input and
    ``channels`` repeats the last. Segmentation: ``input_size`` is ``[height, width]``, each above
    zero and not necessarily equal, and ``channels`` is the model's input channel count; there is
    no ``input_shape``, so the two cannot disagree.
``normalize``
    ``"standardize"`` or ``"unit"`` (what the run asked for).
``mean``, ``std``
    One number per channel: what the training inputs were standardized with, as the
    data loader reports it. ``unit`` is zeros and ones. A folder or csv run records the
    statistics of ONE pass over its fit split, not an ImageNet constant. **Bounds:**
    ``MEAN_RANGE`` (0 to 1) for each ``mean`` and ``STD_RANGE`` (1e-6 to 1) for each ``std``.
    The loader standardizes images that lie in ``[0, 1]``, so the mean of its pixels lies in
    ``[0, 1]`` and their standard deviation cannot pass 0.5 (``unit`` records 1); the loader
    itself refuses a folder or csv channel whose deviation is under 1e-6. A value outside these
    is not a statistic of such images, and the predictor would divide by it: a ``mean`` of 1e30
    turns every input into -1e30 and the probabilities into NaN, with no error. It is refused
    at load.
``eval_transform`` (classification only)
    ``{"kind", "source_channels"}``: how an image of any size becomes the model's
    input before standardization (:data:`EVAL_KINDS`, the evaluation pipelines of the
    loader), and the channel count the files are read with (``1`` for MNIST, whose
    single channel is repeated up to ``channels``).
``ignore_index`` (segmentation only)
    ``null``, or a whole number of 0 or more that is not a class index (outside
    ``0 .. len(class_names) - 1``): the mask value that marks a pixel as void. A mask file cannot
    hold a negative, and a class index would be a class the model cannot learn.
``mask_threshold`` (segmentation only)
    ``null``, or the finite number that turns a grey mask into two classes. It needs exactly two
    ``class_names`` and no ``ignore_index`` (D-014: an explicit number, two classes).
``output`` (segmentation only)
    ``"masks"`` or ``"logits"``: what the predictor returns for an image
    (:class:`~train.api.segmentation_predictor.SegmentationPredictor` reads it).

``variant_overrides`` (OPTIONAL)
    ``{table key: value}``: the keys of the variant's table (``MODEL_VARIANTS[variant]``) that the
    run's resolved ``model_args`` set to another value, so the model is the variant WITH these
    changes. Written ONLY when there is at least one (a default run's ``task.json`` is
    byte-identical to one written before the key existed); a record may omit it, and an empty or
    malformed one is refused. :func:`check_model_against_record` compares the model with the
    variant's table updated by it, so an override is honoured and a wrong variant or an override the
    model does not have is still refused. ``export`` carries it into ``<out>.task.json``.

``build_task_record(...)`` / ``build_segmentation_record(...)``
    :return: the record for a finished data load (not checked; see their docstrings). The first
        writes a classification record, the second a segmentation record.

``check_record(record, source)``
    :param record: a parsed ``task.json``.
    :param source: its path, for the messages.
    :raises ConfigError: for a wrong ``schema_version`` (an older one in the sentence described
        above), a ``task`` this reader does not know, a missing or unknown key for that task, a
        value of the wrong kind (``variant_overrides`` included: a non-empty object of text keys
        and JSON values), repeated class names, a ``model`` that is not ``family/variant``, a
        ``mean`` or ``std`` of the wrong length, a ``std`` that is not positive, a ``mean`` or
        ``std`` outside the bounds above, ``normalize`` of ``unit`` with a ``mean`` that is not
        zeros or a ``std`` that is not ones, and the task's own keys: a classification record's
        ``input_shape`` that is not square or an ``eval_transform`` whose kind is not in
        :data:`EVAL_KINDS`; a segmentation record's ``input_size``, ``channels``, ``ignore_index``,
        ``mask_threshold`` (with its two-class and no-void conditions) or ``output``.

``refuse_old_record(path, prefix)``
    :param path: a ``task.json`` on disk.
    :param prefix: how the message begins (``"Cannot resume <dir>"``).
    :raises ConfigError: when the file is JSON whose ``schema_version`` is a whole number below
        :data:`SCHEMA_VERSION`; returns for a missing file, a file that is not JSON (the reader
        names that) and any other version (the reader refuses those). One sentence naming the
        file and the retraining. ``resume`` calls it before it loads the data.

``record_differences(earlier, now)``
    :param earlier: a checked record (a run's ``task.json``).
    :param now: a record built from the data loaded now (:func:`build_task_record` or
        :func:`build_segmentation_record`).
    :return: one ``"<key>: <earlier> -> <now>"`` line per key of ``DATA_KEYS[task]`` that
        differs: for classification the class names (as a list, in order), the geometry, the
        normalization and the per-channel ``mean`` and ``std`` (within a relative ``1e-5``); for
        segmentation the class names, ``input_size``, the normalization, ``mean`` and ``std``,
        ``ignore_index`` and ``mask_threshold``. ``output`` is how the predictor answers, not what
        the run trained on, and is not compared. Records of two tasks differ in ``task`` alone.
        A resume compares with this before it trains on the data (D-069).

``load_model(path)``
    :param path: a run directory (its ``best_model.keras``) or a ``.keras`` file. A
        ``.keras`` file reads the first record that exists of ``<file>.task.json`` (what
        ``export`` writes beside its output), ``<stem>.task.json``, and ``task.json`` in its
        directory.
    :return: a :class:`Predictor` for an ``image_classification`` record, a
        :class:`~train.api.segmentation_predictor.SegmentationPredictor` for an
        ``image_segmentation`` one (dispatch on ``record["task"]``). Compiles nothing. The model is
        called once on zeros, so a model whose output width (classification) or output map
        (segmentation: ``input_size`` and the class count) does not match the record is refused here.
    :raises ConfigError: for a missing path, ``task.json`` or model file, a model file that
        is not an intact ``.keras`` archive (one cut short by a kill, or a zip with no
        ``config.json`` or no weights, is named, not a Keras traceback), a bad record, a
        ``task`` this reader does not know, a family that is not registered or does not train the
        record's task, a family or variant the model is not (:func:`check_model_against_record`),
        an ``input_shape`` / ``input_size`` or ``channels`` the model does not take, or a
        class-count mismatch. All of them are raised by ``load_model``, none waits for the first
        prediction.

``record_beside(path)`` / ``input_shape_of(record)``
    :param path: what ``load_model`` takes.
    :return: the checked record of that run or file, without loading the model (the command line
        reads the task before it loads anything); and the model input a record names,
        ``(height, width, channels)``: ``input_shape`` for classification, ``input_size`` and
        ``channels`` for segmentation.

``Predictor.predict(x, top_k=None, threshold=None)``
    :param x: one image or several. A single image is a path (``str`` or ``Path``), a
        ``PIL.Image.Image``, or a 2-D or 3-D ``numpy`` array; several are a list or tuple
        of those, or a 4-D array. A ``uint8`` array is divided by 255; a floating array
        must already lie in ``[0, 1]`` and is refused otherwise; any other dtype is
        refused. A 4-channel array is RGBA and loses its alpha channel, as a PIL image does. A path is read with ``channels`` of ``eval_transform.source_channels``.
        **Grey models** (``source_channels`` 1): a colour array or PIL image is made grey on its
        8-bit pixels with the loader's rule, ``(9797 R + 19234 G + 3737 B) >> 15``
        (:func:`train.common.classification_data.rgb8_to_grey8`; libpng's integer weights as
        ``tf.io.decode_image(channels=1)`` applies them to a PNG, equal to it on 65,536 random
        colours), and a floating array is rounded to its 8-bit values first, as storing it as a
        PNG would. So a PNG's pixels (RGB, RGBA, palette) come out as the loader's within 1e-5
        (measured: identical). **Measured limit, JPEG:** a JPEG given as a PATH is decoded by the
        loader's own call and is exact; a JPEG already decoded to an array or PIL image is not,
        because the loader's grey is the JPEG's Y plane, which RGB output no longer holds, and
        PIL's and TensorFlow's JPEG decoders differ besides. Against the loader's decode of the
        same file, in levels of 255: random colour noise at quality 75, worst pixel 14, mean 0.56;
        at quality 95, 13 and 1.15; smooth images, worst 1 to 2, mean 0.5 to 0.6 (about 0.002 of
        the scale). That is above 1e-5 and cannot be brought under it from decoded pixels; pass the
        file's path when the exact value matters. ``tf.image.rgb_to_grayscale`` and PIL's ``convert("L")`` (the earlier
        calls) differ from the loader on PNG pixels too: one level in a quarter and in a half of
        the pixels.
        **16-bit sources, by PIL mode:** ``I;16`` (and ``I;16L``, ``I;16B``, ``I;16N``) is
        converted as the decoder reads the file, each sample's high byte (``value >> 8``, equal
        to the loader on the same file), for one and three channels. ``I`` and ``F`` (32-bit
        integers and floats) are REFUSED with a :class:`ConfigError` naming the mode: the loader
        reads no file of that depth. A 16-bit RGB or RGBA PNG opens in PIL as 8-bit ``RGB`` or
        ``RGBA``, which cannot be told from an 8-bit file, so it is neither refused nor exact for
        a grey model: the decoder makes the grey from the 16-bit samples before it strips them, and
        the 8-bit pixels no longer hold that. **Measured limit:** at most 1 level of 255 against the loader's
        decode of the same file (a path is exact; a three-channel model is exact).
    :param top_k: keep the ``top_k`` best classes of each image; ``None`` keeps all.
    :param threshold: drop classes whose probability is below it; ``None`` drops none.
    :return: ``{"labels", "scores"}``. For a single image both are flat lists, best class
        first (ties keep class order); for several images, one such list per image.
        ``scores`` are softmax probabilities of the model's logits, as plain floats.
    :raises ConfigError: for an unreadable file, an unsupported or empty array, an empty list, a
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
from .run_dir import TASK_FILE, TASK_SUFFIX, check_model_file

# ---------------------------------------------------------------------

# DECISION plan-2026-10-31T180033-2faf890a/D-073: a record has the keys of ITS task and no others, and
# version 1 is not read. Do NOT make one flat record with every key and a null for the other task's
# (a key nothing reads is a lie in the file), do NOT keep ``input_shape`` beside ``input_size`` in a
# segmentation record (one fact held twice, and a square-only check on one of them), and do NOT add a
# version 1 reader "for the old runs": D-004 priced that, every old run directory loses predict,
# export and resume, and the standing rule is no legacy loaders.
#: The only ``schema_version`` this module reads and writes. Version 1 (classification only, one
#: layout) is not read: the number moved when segmentation records were added, a v1 run is
#: refused at predict, export and resume, and there is no converter (D-004, the standing rule
#: against legacy readers).
SCHEMA_VERSION = 2

CLASSIFICATION = "image_classification"
SEGMENTATION = "image_segmentation"

#: The task :class:`Predictor` serves; ``load_model`` returns a
#: :class:`~train.api.segmentation_predictor.SegmentationPredictor` for a segmentation record, so
#: a record of any other task is refused by the class that does not serve it.
SERVED_TASK = CLASSIFICATION

#: The keys of a record per task, in the order they are written. The classification tuple is the
#: version 1 tuple: byte-identical output apart from ``schema_version`` is a standing requirement.
RECORD_KEYS: Dict[str, Tuple[str, ...]] = {
    CLASSIFICATION: (
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
    ),
    SEGMENTATION: (
        "schema_version",
        "task",
        "model",
        "family",
        "variant",
        "class_names",
        "input_size",
        "channels",
        "normalize",
        "mean",
        "std",
        "ignore_index",
        "mask_threshold",
        "output",
    ),
}

#: What ``output`` may say in a segmentation record.
SEGMENTATION_OUTPUTS = ("masks", "logits")

#: Keys a record MAY carry (written only when they hold something); every other key is required.
OPTIONAL_RECORD_KEYS = ("variant_overrides",)

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

#: The range of a record's per-channel ``mean`` and ``std``. The loader's images lie in [0, 1], so
#: their mean does too and their deviation cannot pass 0.5 (``unit`` records exactly 1); 1e-6 is
#: the loader's own floor for a folder or csv channel (``classification_data`` refuses less).
#: A value outside is no statistic of such images and the predictor divides by it.
MEAN_RANGE = (0.0, 1.0)
STD_RANGE = (1e-6, 1.0)

#: The PIL modes that already are grey (read as ``L``); every other mode is read as RGB.
GREY_MODES = ("L", "LA", "1")

#: The 16-bit grey PIL modes (a 16-bit grey PNG opens as ``I;16``; the others differ in byte
#: order). Read as the decoder reads the same file: each sample stripped to its high byte.
GREY16_MODES = ("I;16", "I;16L", "I;16B", "I;16N")

#: The PIL modes of 32-bit integers and floats. REFUSED: the loader reads no file of that depth,
#: so there is no decoder rule to reproduce.
WIDE_MODES = ("I", "F")


def _record_of(
    task: str, values: Mapping[str, Any], variant_overrides: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Return the record of ``task``: ``values`` in the order of ``RECORD_KEYS[task]``, plus the optional key."""
    record = {key: values[key] for key in RECORD_KEYS[task]}
    if variant_overrides:
        record["variant_overrides"] = _plain(dict(variant_overrides))
    return record


def _statistics(mean: Any, std: Any) -> Dict[str, Any]:
    return {"mean": [float(value) for value in mean], "std": [float(value) for value in std]}


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
    variant_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a classification run; see the module contract.

    :param task: written as given (the classification task passes ``image_classification``);
        the LAYOUT is the classification one whatever the word.
    :param variant_overrides: the output of :func:`overrides_of_variant`; the record gets the
        ``variant_overrides`` key only when this is not empty.

    :param data_kind: a key of :data:`EVAL_KIND_OF_DATA`: the dataset name, or ``folder``
        or ``csv``.
    :param source_channels: the channels the files are read with.
    :return: the record. It is NOT checked: ImageNet's antialiased centre crop is recorded
        truthfully; every kind it can hold is one :func:`check_record` accepts.
    """
    shape = [int(side) for side in input_shape]
    return _record_of(
        CLASSIFICATION,
        {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "model": model,
            "family": family,
            "variant": variant,
            "class_names": [str(name) for name in class_names],
            "input_shape": shape,
            "channels": shape[2],
            "normalize": normalize,
            **_statistics(mean, std),
            "eval_transform": {
                "kind": EVAL_KIND_OF_DATA[data_kind],
                "source_channels": int(source_channels),
            },
        },
        variant_overrides,
    )


def build_segmentation_record(
    *,
    model: str,
    family: str,
    variant: str,
    class_names: Sequence[Any],
    input_size: Sequence[int],
    channels: int,
    normalize: str,
    mean: Any,
    std: Any,
    ignore_index: Optional[int] = None,
    mask_threshold: Optional[float] = None,
    output: str = "masks",
    variant_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a segmentation run; see the module contract.

    :param input_size: ``(height, width)`` of the model's input, not necessarily equal.
    :param channels: the model's input channels.
    :param ignore_index: the void value of the masks, or ``None``.
    :param mask_threshold: the grey-mask threshold of a two-class run, or ``None``.
    :param output: one of :data:`SEGMENTATION_OUTPUTS`.
    :return: the record. It is NOT checked (:func:`check_record` does that).
    """
    return _record_of(
        SEGMENTATION,
        {
            "schema_version": SCHEMA_VERSION,
            "task": SEGMENTATION,
            "model": model,
            "family": family,
            "variant": variant,
            "class_names": [str(name) for name in class_names],
            "input_size": [int(side) for side in input_size],
            "channels": int(channels),
            "normalize": normalize,
            **_statistics(mean, std),
            "ignore_index": ignore_index,
            "mask_threshold": mask_threshold,
            "output": output,
        },
        variant_overrides,
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_json_plain(value: Any) -> bool:
    """Whether ``value`` is None, a bool, a finite number, a text, or a list or text-keyed dict of those."""
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_plain(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_plain(item) for key, item in value.items())
    return False


def overrides_of_variant(table_row: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the keys of ``table_row`` that ``kwargs`` set to another value, as JSON-plain values.

    Interface contract: ``table_row`` is ``MODEL_VARIANTS[variant]`` and ``kwargs`` the factory
    keywords a run was built with (the resolver's output: the recipe's ``model_args``, the computed
    keys and the caller's ``model_args``). A key counts only when it is in the table AND its value
    differs from the table's (tuples read as lists), so a run that restates a table value, and
    every run that touches no table key, gets ``{}`` and therefore no ``variant_overrides`` key.
    Pure; imports nothing heavy.

    :param table_row: the variant's table.
    :param kwargs: the factory keywords.
    :return: ``{key: value}`` in the table's key order; ``{}`` when nothing was overridden.
    """
    return {
        key: _plain(kwargs[key])
        for key in table_row
        if key in kwargs and _plain(kwargs[key]) != _plain(table_row[key])
    }


def _is_whole(value: Any) -> bool:
    """Whether ``value`` is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def older_record_reason(version: Any) -> Optional[str]:
    """Return the clause for a record of an older layout, or ``None`` when ``version`` is not one.

    Interface contract: ``version`` is a ``schema_version`` as parsed from JSON. An older layout is
    a whole number from 1 up to (not including) :data:`SCHEMA_VERSION`; anything else (a newer
    number, a bool, text, a missing value) is ``None`` and keeps the plain refusal. The clause is
    the one place the sentence is worded, so ``predict``, ``export`` and ``resume`` say the same.
    """
    if _is_whole(version) and 1 <= version < SCHEMA_VERSION:
        return (
            f"an older layout; this reader understands {SCHEMA_VERSION} only and cannot upgrade it, "
            "so the run has to be trained again"
        )
    return None


def _version_line(version: Any) -> str:
    older = older_record_reason(version)
    if older is not None:
        return f"'schema_version' is {version}, {older}"
    return f"'schema_version' is {version!r}; this reader understands {SCHEMA_VERSION} only"


def _header_problems(record: Mapping[str, Any], keys: Tuple[str, ...]) -> Tuple[List[str], bool]:
    """Return ``(lines, complete)``: the key-set problems, and whether no key is missing."""
    missing = [key for key in keys if key not in record]
    unknown = sorted(set(record) - set(keys) - set(OPTIONAL_RECORD_KEYS))
    lines: List[str] = []
    if missing:
        lines.append(f"missing key(s) {', '.join(map(repr, missing))}")
    if unknown:
        lines.append(f"unknown key(s) {', '.join(map(repr, unknown))}")
    return lines, not missing


def _identity_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of ``task``, ``model``, ``family``, ``variant`` and ``class_names``."""
    lines: List[str] = []
    for key in ("task", "model", "family", "variant"):
        if not isinstance(record[key], str) or not record[key]:
            lines.append(f"{key!r} must be a non-empty text, got {record[key]!r}")
    names = record["class_names"]
    if not (isinstance(names, list) and names and all(isinstance(n, str) for n in names)):
        lines.append(f"'class_names' must be a non-empty list of texts, got {names!r}")
    elif len(set(names)) != len(names):
        repeated = sorted({n for n in names if names.count(n) > 1})
        lines.append(f"'class_names' repeats {', '.join(map(repr, repeated))}; each class needs its own name")
    if all(isinstance(record[key], str) and record[key] for key in ("model", "family", "variant")):
        if record["model"] != f"{record['family']}/{record['variant']}":
            lines.append(
                f"'model' is {record['model']!r} but 'family' and 'variant' give "
                f"{record['family'] + '/' + record['variant']!r}"
            )
    return lines


def _classification_geometry(record: Mapping[str, Any]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, channels)`` for ``input_shape`` and ``channels`` (``None`` when unusable)."""
    shape = record["input_shape"]
    shape_ok = (
        isinstance(shape, list)
        and len(shape) == 3
        and all(_is_whole(side) and side > 0 for side in shape)
        and shape[0] == shape[1]
    )
    if not shape_ok:
        return [f"'input_shape' must be [side, side, channels] of positive ints, got {shape!r}"], None
    if record["channels"] != shape[2]:
        return [f"'channels' is {record['channels']!r} but 'input_shape' has {shape[2]}"], shape[2]
    return [], shape[2]


def _segmentation_geometry(record: Mapping[str, Any]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, channels)`` for ``input_size`` and ``channels`` (``None`` when unusable)."""
    lines: List[str] = []
    size = record["input_size"]
    if not (isinstance(size, list) and len(size) == 2 and all(_is_whole(side) and side > 0 for side in size)):
        lines.append(f"'input_size' must be [height, width] of positive ints, got {size!r}")
    channels = record["channels"]
    if not (_is_whole(channels) and channels > 0):
        lines.append(f"'channels' must be a positive int, got {channels!r}")
        return lines, None
    return lines, channels


def _statistics_problems(record: Mapping[str, Any], width: Optional[int]) -> List[str]:
    """Return the problems of ``normalize``, ``mean`` and ``std`` against ``width`` channels."""
    lines: List[str] = []
    if record["normalize"] not in _NORMALIZE:
        lines.append(f"'normalize' must be one of {', '.join(_NORMALIZE)}, got {record['normalize']!r}")
    for key in ("mean", "std"):
        values = record[key]
        if not (isinstance(values, list) and values and all(_is_number(v) for v in values)):
            lines.append(f"{key!r} must be a list of finite numbers, got {values!r}")
        elif width is not None and len(values) != width:
            lines.append(f"{key!r} holds {len(values)} numbers for {width} channels")
    mean, std = record["mean"], record["std"]
    if isinstance(std, list) and all(_is_number(v) for v in std) and any(v <= 0 for v in std):
        lines.append(f"'std' must be positive, got {std!r}")
    for key, (low, high) in (("mean", MEAN_RANGE), ("std", STD_RANGE)):
        values = record[key]
        if not (isinstance(values, list) and all(_is_number(v) for v in values)):
            continue  # named above
        # A std of zero or less is already named above; do not say it twice.
        outside = [v for v in values if not low <= v <= high and (key == "mean" or v > 0)]
        if outside:
            lines.append(
                f"{key!r} must lie in [{low:g}, {high:g}] (the loader standardizes images in "
                f"[0, 1]), got {values!r}"
            )
    numbers = all(isinstance(v, list) and all(_is_number(x) for x in v) for v in (mean, std))
    if record["normalize"] == "unit" and numbers:
        if any(v != 0 for v in mean) or any(v != 1 for v in std):
            lines.append(
                f"'normalize' is 'unit' (no standardization) but 'mean' is {mean!r} and 'std' is "
                f"{std!r}; unit means zeros and ones"
            )
    return lines


def _overrides_problems(record: Mapping[str, Any]) -> List[str]:
    if "variant_overrides" not in record:
        return []
    overrides = record["variant_overrides"]
    if (
        isinstance(overrides, dict)
        and overrides
        and all(isinstance(key, str) and key and _is_json_plain(value) for key, value in overrides.items())
    ):
        return []
    return [
        "'variant_overrides' must be a non-empty object of text keys and JSON values "
        f"(it is written only when a model_args key changed a key of the variant's table), "
        f"got {_shown(overrides)}"
    ]


def _eval_transform_problems(record: Mapping[str, Any]) -> List[str]:
    lines: List[str] = []
    transform = record["eval_transform"]
    if not isinstance(transform, dict) or set(transform) != {"kind", "source_channels"}:
        lines.append(
            f"'eval_transform' must be an object with 'kind' and 'source_channels', got {transform!r}"
        )
        return lines
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


def _segmentation_tail_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of ``ignore_index``, ``mask_threshold`` and ``output``."""
    lines: List[str] = []
    names = record["class_names"]
    classes = len(names) if isinstance(names, list) and names else None
    ignore = record["ignore_index"]
    if ignore is not None and not (_is_whole(ignore) and ignore >= 0):
        lines.append(f"'ignore_index' must be null or a whole number of 0 or more, got {ignore!r}")
    elif ignore is not None and classes is not None and ignore < classes:
        lines.append(
            f"'ignore_index' is {ignore}, a class index; it must lie outside the class indices "
            f"0..{classes - 1}"
        )
    threshold = record["mask_threshold"]
    if threshold is not None:
        if not _is_number(threshold):
            lines.append(f"'mask_threshold' must be null or a finite number, got {threshold!r}")
        else:
            if classes is not None and classes != 2:
                lines.append(f"'mask_threshold' needs exactly 2 classes, the record names {classes}")
            if ignore is not None:
                lines.append(
                    "'mask_threshold' and 'ignore_index' cannot both be set: a thresholded mask "
                    "has two classes and no void"
                )
    if record["output"] not in SEGMENTATION_OUTPUTS:
        lines.append(
            f"'output' must be one of {', '.join(SEGMENTATION_OUTPUTS)}, got {record['output']!r}"
        )
    return lines


def record_problems(record: Mapping[str, Any]) -> List[str]:
    """Return one line per way ``record`` is not a valid version-2 record (empty when valid)."""
    version = record.get("schema_version")
    if version != SCHEMA_VERSION or isinstance(version, bool):
        return [_version_line(version)]
    task = record.get("task")
    keys = RECORD_KEYS.get(task) if isinstance(task, str) else None
    if keys is None:
        return [f"'task' is {task!r}; this reader knows {', '.join(RECORD_KEYS)}"]
    lines, complete = _header_problems(record, keys)
    if not complete:
        return lines
    lines += _identity_problems(record)
    geometry, width = (
        _classification_geometry(record) if task == CLASSIFICATION else _segmentation_geometry(record)
    )
    lines += geometry
    lines += _statistics_problems(record, width)
    lines += _overrides_problems(record)
    lines += (
        _eval_transform_problems(record) if task == CLASSIFICATION else _segmentation_tail_problems(record)
    )
    return lines


def check_record(record: Any, source: Union[str, Path]) -> None:
    """Raise :class:`ConfigError` unless ``record`` is a valid version-2 record; see the module contract."""
    lines = ["not a JSON object"] if not isinstance(record, dict) else record_problems(record)
    if lines:
        raise ConfigError(f"Invalid {TASK_FILE} {source}: " + "; ".join(lines))


def refuse_old_record(path: Path, prefix: str) -> None:
    """Raise :class:`ConfigError` when ``path`` is a record of an older layout; see the module contract."""
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return
    older = older_record_reason(record.get("schema_version")) if isinstance(record, dict) else None
    if older is not None:
        raise ConfigError(
            f"{prefix}: {path} has 'schema_version' {record['schema_version']}, {older}"
        )


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


#: The keys of a record that describe the DATA a run was trained on, per task (a resume compares
#: them). A segmentation run adds the void and the threshold: a different ``ignore_index`` is
#: other labels. ``output`` is how the predictor answers, not what the run saw.
DATA_KEYS: Dict[str, Tuple[str, ...]] = {
    CLASSIFICATION: ("class_names", "input_shape", "normalize", "eval_transform", "mean", "std"),
    SEGMENTATION: (
        "class_names", "input_size", "normalize", "mean", "std", "ignore_index", "mask_threshold",
    ),
}


def _shown(value: Any) -> str:
    text = json.dumps(value, default=repr)
    return text if len(text) <= 100 else text[:97] + "..."


def record_differences(earlier: Mapping[str, Any], now: Mapping[str, Any]) -> List[str]:
    """List the ``DATA_KEYS`` where ``now`` differs from ``earlier``; see the module contract."""
    if earlier["task"] != now["task"]:
        return [f"task: {_shown(earlier['task'])} -> {_shown(now['task'])}"]
    lines = []
    for key in DATA_KEYS[now["task"]]:
        before, after = earlier[key], now[key]
        if key in ("mean", "std"):
            same = len(before) == len(after) and all(
                math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-7) for a, b in zip(before, after)
            )
        else:
            same = before == after
        if not same:
            lines.append(f"{key}: {_shown(before)} -> {_shown(after)}")
    return lines


def _task_candidates(model_file: Path) -> Tuple[Path, ...]:
    """Return the records a ``.keras`` file may be read with, best first (see the module contract)."""
    # DECISION plan-2026-10-30T132432-a865fa08/D-068: ``<file>.task.json`` first, because that is
    # the name ``export`` writes for ``<out>``; ``<stem>.task.json`` and ``task.json`` after it.
    # Do NOT read only ``task.json``: an exported file moved away from its run has none.
    return (
        model_file.with_name(model_file.name + TASK_SUFFIX),
        model_file.with_name(model_file.stem + TASK_SUFFIX),
        model_file.parent / TASK_FILE,
    )


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
        candidates = _task_candidates(path)
        for candidate in candidates:
            if candidate.is_file():
                return path, candidate
        raise ConfigError(
            f"Invalid value for 'model': {path} has no record beside it; a .keras file is read "
            f"with the first of {', '.join(c.name for c in candidates)} that exists in "
            f"{path.parent} (export writes {candidates[0].name})"
        )
    raise ConfigError(f"Invalid value for 'model': {path} is neither a run directory nor a file")


def record_beside(path: Union[str, Path]) -> Dict[str, Any]:
    """Return the checked record of a run directory or ``.keras`` file; see the module contract."""
    return read_record(_model_files(Path(path))[1])


def input_shape_of(record: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return ``(height, width, channels)``, the model input a checked record names."""
    if record["task"] == SEGMENTATION:
        return (*record["input_size"], record["channels"])
    return tuple(record["input_shape"])


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
    if record["task"] not in spec.tasks:
        # A record whose family trains another task is not one a run wrote (the trainer refuses the
        # pair before it claims the directory, D-072); say so here rather than let the family check
        # below compare a model with a variants table that is not its own.
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: the family {record['family']!r} trains "
            f"{' and '.join(spec.tasks)}, not {record['task']!r}; the file is not from one run"
        )
    spec.load_factory()
    # DECISION plan-2026-10-30T132432-a865fa08/D-061: ``compile=False``. Do NOT load the compile config: a
    # predictor needs no loss or metric, and a metric class the archive names (the
    # library's half-precision top-k, D-051) would be one more thing that can fail a
    # reload that needs only the weights.
    check_model_file(model_file, "Invalid value for 'model'")
    model = keras.models.load_model(str(model_file), compile=False)
    check_model_against_record(model, record, spec, model_file, task_file)
    output = _run_on_zeros(model, record, model_file, task_file)
    if record["task"] == SEGMENTATION:
        from .segmentation_predictor import SegmentationPredictor, check_output

        check_output(output, record)
        return SegmentationPredictor(model, record)
    _check_width(output, record)
    return Predictor(model, record)


def _plain(value: Any) -> Any:
    """Return ``value`` with every tuple a list, so a config and a variant table compare equal."""
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def check_model_against_record(
    model: Any, record: Mapping[str, Any], spec: "registry.ModelSpec", model_file: Path, task_file: Path
) -> None:
    """Raise :class:`ConfigError` unless ``model`` is the family and variant ``record`` names.

    Interface contract: ``model`` is a loaded Keras model, ``record`` a checked record, ``spec`` the
    registered :class:`~train.api.registry.ModelSpec` of ``record["family"]``. The family is the
    class named by ``spec.variants_attr`` (``model`` must be an instance of it); the variant must
    be a key of that class's ``MODEL_VARIANTS``. The EXPECTED table is the variant's table updated
    by the record's ``variant_overrides`` (each override key must be a key of the table, and the
    model's ``get_config()`` must report it), and every key of the expected table that the model's
    ``get_config()`` also holds must be equal. Measured on all six families (the smallest variant of
    each, against every variant's table): a wrong variant differs in at least one such key.
    A run trained with ``model_args`` that override table keys is honoured through
    ``variant_overrides`` (D-032, which replaces the refusal D-028 accepted for it); a model
    that differs from its variant in a key the record does not list is refused. Two variants that
    differ only in keys the record overrides are the same architecture and cannot be told apart.

    A family whose models are functional (the four segmentation families: no class to be an instance of,
    no table keys in ``get_config()``) is checked by what its layers show: ``spec.observe(model)`` names
    the table keys the layers carry, or ``None`` for a model that holds no layer of the family, and
    each key it returns must equal the expected table's. A family with an ``observe`` is checked this way
    even when its ``variants_attr`` names a class (``resnet_seg`` reads ``ResNet.MODEL_VARIANTS``; its model
    wraps a ``ResNet``, it is not one). The keys a layer does not show
    (ConvUNext's ``drop_path_rate``) and a model that differs in one the record does not list are not
    seen; two variants that differ only there are not told apart (D-088).
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-028
    # The family and the variant of the record are compared with the MODEL, not only with each
    # other (``record_problems`` already checks ``model == family/variant``): a record that names
    # another family or variant still predicts, with a lie in its metadata. Do NOT compare the
    # whole config: ``model_args`` that are not part of a variant table are free.
    # DECISION plan-2026-10-31T180033-2faf890a/D-032
    # Supersedes D-028's accepted refusal of a model whose table keys ``model_args`` overrode.
    # Do NOT refuse such a model: the run records the overridden keys in ``variant_overrides``
    # and the EXPECTED table is the variant's updated by them. Refusing it made a run that
    # trained for hours unable to ``predict`` or ``export`` (step 14.7, measured on 5 of 6
    # families). A wrong variant, a foreign key and an override the model lacks stay refused.
    from .registry import load_attr

    module_name, _, attr_path = spec.variants_attr.partition(":")
    family, variant = record["family"], record["variant"]
    # DECISION plan-2026-10-31T180033-2faf890a/D-080
    # A family whose table is a module-level mapping (``module:TABLE``, no class: the functional
    # U-Nets of the segmentation task) has no class to be an instance of and no table keys in its
    # ``get_config()`` (a functional model reports its layers). Do NOT rebuild the variant and
    # compare weight shapes (a ``model_args`` key that is not a table key, ``kernel_size`` say,
    # changes the shapes and the record does not carry it: D-032's failure, a run trained for hours
    # that cannot ``predict``), and do NOT compare ``model.name`` (``model_args.model_name`` renames
    # the model).
    # DECISION plan-2026-10-31T180033-2faf890a/D-088
    # D-080 stopped there and README.md said the family "cannot be checked". It can: the two
    # families share no layer class (ConvUNext: ConvStem, DownsampleAndSkip, ConvNextV2Block; AccUNet:
    # AccUNet), so `spec.observe(model)` reads the family and the table keys the layers show, and a
    # mismatch is refused. Do NOT go back to "the variant is in the table and the model takes the
    # size": a convunext/tiny file loaded under records naming convunext/base, convunext/xlarge and
    # accunet/base, and `export` wrote the wrong record beside the export (review-iter-3 concern 3).
    # DECISION plan-2026-11-03T221936-98b866fb/D-012
    # A family with `observe` is checked by what its layers show EVEN WHEN its table lives on a class
    # (`resnet_seg` and `convnext_seg` share the classifier's `ResNet.MODEL_VARIANTS` and
    # `ConvNeXtV1.MODEL_VARIANTS`: one table, not a copy). Their model is a functional wrapper, so it is not
    # an instance of that class and `isinstance` would refuse every honest file. Do NOT copy the table into
    # `segmentation_models` to make `owner` None, and do NOT drop the `isinstance` check for the classifiers.
    owner = load_attr(f"{module_name}:{attr_path.rsplit('.', 1)[0]}") if "." in attr_path else None
    observed = spec.observe is not None
    if owner is not None and not observed and not isinstance(model, owner):
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names the family {family!r} ({owner.__name__}) "
            f"but {model_file} holds a {type(model).__name__}; the file and the model are not "
            "from one run"
        )
    table = spec.load_variants()
    if variant not in table:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: {variant!r} is not a variant of {family!r} "
            f"(known: {', '.join(sorted(table))})"
        )
    wanted = table[variant]
    overrides = record.get("variant_overrides") or {}
    if overrides and not isinstance(wanted, Mapping):
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it records variant_overrides but {variant!r} of "
            f"{family!r} has no table of keys to override"
        )
    if isinstance(wanted, Mapping):
        foreign = [key for key in overrides if key not in wanted]
        if foreign:
            raise ConfigError(
                f"Invalid {TASK_FILE} {task_file}: variant_overrides names {', '.join(map(repr, foreign))}, "
                f"which {variant!r} of {family!r} does not have (it has {', '.join(map(repr, wanted))})"
            )
        if owner is None or observed:
            _check_observed(model, spec, {**wanted, **overrides}, overrides, record, model_file, task_file)
            return
        built = model.get_config()
        unreported = [key for key in overrides if key not in built]
        if unreported:
            raise ConfigError(
                f"Invalid {TASK_FILE} {task_file}: variant_overrides names {', '.join(map(repr, unreported))} "
                f"but {model_file} does not report that key, so the override cannot be true of it"
            )
        expected = {**wanted, **overrides}
        differing = [
            f"{key} is {_shown(_plain(built[key]))} in the model, {_shown(_plain(value))} in "
            f"{variant!r}{' with the recorded overrides' if overrides else ''}"
            for key, value in expected.items()
            if key in built and _plain(built[key]) != _plain(value)
        ]
        if differing:
            raise ConfigError(
                f"Invalid {TASK_FILE} {task_file}: it names the variant {variant!r} but "
                f"{model_file} is not that variant ({'; '.join(differing)}); the file and the "
                "model are not from one run"
            )


def _check_observed(
    model: Any, spec: "registry.ModelSpec", expected: Mapping[str, Any], overrides: Mapping[str, Any],
    record: Mapping[str, Any], model_file: Path, task_file: Path,
) -> None:
    """Refuse a functional ``model`` whose layers are not the family and variant of ``record``.

    Interface contract: ``expected`` is the variant's table updated by the record's overrides,
    ``spec.observe`` the family's reader (``None``: nothing can be read, no check). The reader's keys
    that are also in ``expected`` must be equal. Raises :class:`ConfigError` naming the file, the
    family or the differing keys; the file and the model are "not from one run".
    """
    observer = spec.load_observer()
    if observer is None:
        return
    family, variant = record["family"], record["variant"]
    seen = observer(model)
    if seen is None:
        own = sorted({
            type(layer).__name__ for layer in getattr(model, "layers", None) or []
            if not type(layer).__module__.startswith(("keras", "tensorflow"))
        })
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names the family {family!r} but {model_file} holds "
            f"no layer of that family (its own layers: {', '.join(own[:8]) or 'none'}); the file and "
            "the model are not from one run"
        )
    differing = [
        f"{key} is {_shown(_plain(value))} in the model, {_shown(_plain(expected[key]))} in "
        f"{variant!r}{' with the recorded overrides' if overrides else ''}"
        for key, value in seen.items()
        if key in expected and _plain(value) != _plain(expected[key])
    ]
    if differing:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names the variant {variant!r} of {family!r} but "
            f"{model_file} is not that variant ({'; '.join(differing)}); the file and the model are "
            "not from one run"
        )


def _run_on_zeros(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> Any:
    """Return ``model`` called on one zero input of the record's shape, or raise :class:`ConfigError`.

    A model that cannot take the record's ``input_shape`` (another side, another channel count)
    fails inside Keras or TensorFlow with a message about a layer; here it is named as what it
    is, at load, not at the first prediction.
    """
    import tensorflow as tf

    try:
        return model(_zeros_of(record), training=False)
    except (ValueError, tf.errors.InvalidArgumentError) as exc:
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        key = "input_size" if record["task"] == SEGMENTATION else "input_shape"
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: {model_file} does not take the {key} "
            f"{record[key]} it names ({type(exc).__name__}: {first[:150]}); the file "
            "and the model are not from one run"
        ) from exc


def _zeros_of(record: Mapping[str, Any]) -> Any:
    """Return one all-zero input of the record's shape, to ask a loaded model its output width."""
    import numpy as np

    return np.zeros((1, *input_shape_of(record)), np.float32)


def _check_width(logits: Any, record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless ``logits`` is ``(N, classes)`` for the record's classes."""
    shape = tuple(logits.shape)
    if len(shape) != 2 or shape[1] != len(record["class_names"]):
        raise ConfigError(
            f"The model outputs {shape[1:]} per image but {TASK_FILE} names "
            f"{len(record['class_names'])} classes; the file and the model are not from one run"
        )


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
        if record["task"] != SERVED_TASK:
            raise ConfigError(
                f"Invalid {TASK_FILE} <given>: the task is {record['task']!r} but this predictor "
                f"serves {SERVED_TASK!r} only"
            )
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
        if int(image.shape[-1]) == wanted:
            return image
        # Colour to grey is NOT done here: it happens on the 8-bit pixels in `_from_array`.
        return tf.repeat(image, wanted, axis=-1)

    def _from_array(self, array: Any) -> Any:
        """Return one array as a float32 ``[0, 1]`` tensor ``(H, W, source_channels)``."""
        import numpy as np
        import tensorflow as tf

        from train.common.classification_data import rgb8_to_grey8

        if array.dtype != np.uint8 and not np.issubdtype(array.dtype, np.floating):
            raise ConfigError(
                f"Invalid image: an array must be uint8 or floating point, got {array.dtype}"
            )
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4) or min(array.shape[:2]) < 1:
            raise ConfigError(
                "Invalid image: an array must be (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), got "
                f"shape {tuple(array.shape)}"
            )
        if array.shape[-1] == 4:
            # DECISION plan-2026-10-30T132432-a865fa08/D-068: an RGBA array loses its alpha channel,
            # exactly as ``PIL.Image.convert("RGB")`` does for a PIL image. Do NOT composite it
            # over a background: the two input forms would then give different pixels.
            array = array[..., :3]
        if array.dtype != np.uint8 and not (np.all(array >= 0.0) and np.all(array <= 1.0)):
            raise ConfigError(
                "Invalid image: a floating array must lie in [0, 1] (the scale of a "
                "uint8 array divided by 255), got values outside it or NaN"
            )
        if array.shape[-1] == 3 and self.record["eval_transform"]["source_channels"] == 1:
            # DECISION plan-2026-10-31T180033-2faf890a/D-019
            # Grey is made on the 8-bit pixels with the loader's own integer rule, the stage
            # and the rounding at which `tf.io.decode_image(channels=1)` makes it for a PNG.
            # Do NOT use `tf.image.rgb_to_grayscale` (float luma, other weights: a level off
            # in a quarter of the pixels) or PIL's `convert("L")` (rounds: a level off in half).
            # A float array is rounded to the 8-bit value first, as storing it as a PNG would.
            # A JPEG already decoded to RGB cannot be exact: its Y plane is gone (module docstring).
            if array.dtype != np.uint8:
                array = np.rint(array.astype(np.float64) * 255.0).astype(np.uint8)
            array = rgb8_to_grey8(array)
        if array.dtype == np.uint8:
            image = tf.image.convert_image_dtype(tf.constant(array), tf.float32)
        else:
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
            if item.mode in WIDE_MODES:
                raise ConfigError(
                    f"Invalid image: a PIL image of mode {item.mode} holds 32-bit values and its "
                    "bit depth cannot be preserved (the loader reads no such file); convert it "
                    "to an 8-bit mode ('L' or 'RGB') yourself, or pass the file's path"
                )
            if item.mode in GREY16_MODES:
                # DECISION plan-2026-10-31T180033-2faf890a/D-028
                # A 16-bit grey image is read as `tf.io.decode_image` reads the file: the high
                # byte of each sample. Do NOT hand it to PIL's `convert`: that CLIPS to 255 and
                # was 254 levels off in 99% of the pixels (review-iter-1 concern 4).
                high = (np.asarray(item).astype(np.uint16) >> 8).astype(np.uint8)
                return self._from_array(high)
            # A colour image keeps its colours (alpha dropped) so `_from_array` converts them
            # to grey the loader's way; only an image that is already grey is read as grey.
            grey = self.record["eval_transform"]["source_channels"] == 1 and item.mode in GREY_MODES
            return self._from_array(np.asarray(item.convert("L" if grey else "RGB")))
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
            if len(x) == 0:
                raise ConfigError(f"Invalid image: the array of images is empty (shape {tuple(x.shape)})")
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
        _check_width(logits, self.record)
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

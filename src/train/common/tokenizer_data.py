"""Data for the ``beit_tokenizer`` task: image-only batches, the constant-image baseline and the code-id pass.

The BEiT stage-0 tokenizer, the library ``VQVAERotationTrick``, is trained to reconstruct the images it is given, so its data is the
plain ``(image, label)`` stream of :func:`train.common.image_streams.load_image_streams` with no element map: the model's own
``train_step`` and ``test_step`` read the image and ignore the label. ``train.common.pretrain_data`` is keyed on a mask and has no use
here. This module adds to that loader three things the task needs: the batches object the run sequence reads, with a ``with_stream``
for a resume (the shape of ``PretrainBatches``); the constant-image baseline the reconstruction bar is cut from; and one pass of the
tokenizer over the selection split, for its codebook usage.

This module imports TensorFlow; ``train.api`` imports it inside functions only.

Interface contract
------------------
``load_tokenizer_batches(*, dataset, image_size, batch_size, seed, stream_seed, validation_split, max_samples, augment)``
    Keyword only; no argument has a default (the task's config is the one home of the defaults).
    :param dataset: one of ``train.common.image_streams.SUPPORTED_DATASETS``.
    :param seed: the split seed: the selection and test rows are a function of it alone.
    :param stream_seed: the seed of the train shuffle and the train augmentation; ``None`` means ``seed``. Never reaches the split.
    :param validation_split, max_samples, augment: as in ``load_image_streams``; the cap bounds the train pool, never the test split.
    :return: a :class:`TokenizerBatches`.
    :raises TypeError: for a seed or a stream seed that is not an ``int`` (a ``bool`` is not), before any data is read.
    :raises ValueError: for a selection or test stream whose length is not the one pass it should state (both lengths are named).

``TokenizerBatches``
    What the run sequence reads: the three streams (``(image, label)`` batches; ``train_ds`` is infinite and drops the ragged batch,
    ``select_ds`` and ``test_ds`` are finite, never augmented, and keep every row), ``steps_per_epoch``, ``val_steps``, the three
    row counts, ``input_shape``, ``num_classes`` (``None``: a tokenizer has no classes), the two provenance sentences, the split seed,
    the resolved stream seed and ``request``, the keyword arguments the batches were built from.

``TokenizerBatches.with_stream(completed_steps)``
    :return: the same request on the train stream of ``derive_stream_seed(seed, completed_steps)``: the same split and the same
        select and test streams, a train stream that is neither a replay of the fresh run nor of another count. The data is loaded again.
    :raises TypeError / ValueError: for ``None``, a ``bool``, a non-``int`` or a negative count.

``constant_image_baseline(select_ds)``
    :param select_ds: one pass of a finite stream of ``(image, label)`` batches (or of image batches), the selection stream.
    :return: a :class:`ConstantImageBaseline`: the reconstruction loss of the best constant image on those images, in the metric's own
        reduction, as float32. The candidates are the per-channel mean (``kind = "channel_mean"``) and the per-pixel mean image
        (``"mean_image"``), both fitted in sample on ``select_ds``; the baseline is the smaller. It equals the model's own
        ``evaluate`` of a constant decoder within the repo's float32 tie band (``tests/test_train/test_api/test_common_tokenizer_data.py``).
    :raises ValueError: for an empty or an endless stream.

``code_ids_of(model, select_ds)``
    :param model: a model with ``encode_to_indices`` (a ``VQVAERotationTrick``), run eagerly in inference mode.
    :return: the integer code ids of EVERY image of one pass of ``select_ds``, in row order, shape ``(n, grid_h, grid_w)``.
    :raises TypeError: for a model with no ``encode_to_indices``. :raises ValueError: for an empty or an endless stream.

``per_cell_usage(code_ids)``
    :param code_ids: an integer array of shape ``(images, grid_h, grid_w)``, the output of :func:`code_ids_of` (non-negative ids).
    :return: a :class:`CellUsage`: ``entropy``, the mean over the grid's cells of the entropy (nats) of the codes that cell gave to the
        images, and ``perplexity``, its exp: the number of equally used codes a cell is worth on average, H(code | cell) in effective
        codes. A code that is the cell index alone is worth exactly 1 however many cells there are, where its POOLED perplexity
        (:func:`train.common.pretrain_data.tokenizer_usage`) is the number of cells.
    :raises ValueError: for an array that is not a non-empty 3-D array of non-negative integers.

``MIN_PER_CELL_PERPLEXITY`` / ``per_cell_meets_floor(usage)``
    The second floor of the tokenizer's bar (D-067): 8 effective codes per cell. ``per_cell_meets_floor`` judges the ENTROPY against the
    log of the constant read when it is called (as ``pretrain_run.usage_meets_floor`` does for the pooled floor, which this does not touch:
    ``beit_mim`` re-judges a tokenizer file by that one).

``TokenizerCodes(model)``
    A Keras layer, the function an exported tokenizer file runs (``train.api.tokenizer_predictor.served``): ``layer(images)`` is
    ``model.encode_to_indices(images)``, the int32 code grid ``(batch, height // patch, width // patch)`` of float32 images
    ``(batch, height, width, channels)`` standardized as the run's streams are. Registered under ``train_tokenizer``, a package of the
    ``train`` tree, which the wheel does not ship: a ``.keras`` file that holds the layer loads only where ``train`` is importable. Its config
    names the model; the weights ride in the archive beside it. ``tests/test_train/test_api/test_beit_tokenizer_predictor_and_export.py``
    holds it to the model's own codes in all four export formats.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from train.common.classification_data import DEFAULT_VALIDATION_SPLIT
from train.common.image_streams import load_image_streams
from train.common.pretrain_data import derive_stream_seed, states_its_length
from train.common.pretrain_run import ENTROPY_ROUNDING_TOLERANCE

#: The two constants the baseline tries: one number per channel, and one number per pixel and channel.
CHANNEL_MEAN, MEAN_IMAGE = "channel_mean", "mean_image"


@dataclass(frozen=True, eq=False)
class TokenizerBatches:
    """What the ``beit_tokenizer`` task consumes; the fields the run sequence reads (see the module contract).

    :param train_ds: infinite, batched ``(image, label)`` stream; the task passes ``steps_per_epoch`` to ``fit``.
    :param select_ds: the selection split, finite, one pass of ``val_steps`` batches, ragged last batch kept.
    :param test_ds: the test split, finite, one pass of ``ceil(n_test / batch_size)`` batches.
    :param steps_per_epoch, val_steps: batches in an epoch and in a selection pass.
    :param n_train, n_select, n_test: rows of the three splits.
    :param input_shape: ``(image_size, image_size, 3)``.
    :param num_classes: ``None``: a tokenizer has no classes (``check_holds_no_classes`` reads it).
    :param selection_split, test_split: plain-language provenance of the two splits.
    :param seed, stream_seed: the split seed and the (resolved) stream seed.
    :param request: the keyword arguments :func:`load_tokenizer_batches` was called with, so that :meth:`with_stream` rebuilds the
        same request on another stream seed.
    """

    train_ds: "tf.data.Dataset"
    select_ds: "tf.data.Dataset"
    test_ds: "tf.data.Dataset"
    steps_per_epoch: int
    val_steps: int
    n_train: int
    n_select: int
    n_test: int
    input_shape: Tuple[int, int, int]
    num_classes: Optional[int]
    selection_split: str
    test_split: str
    seed: int
    stream_seed: int
    request: Mapping[str, Any] = field(repr=False)

    def with_stream(self, completed_steps: int) -> "TokenizerBatches":
        """This request on the train stream of ``derive_stream_seed(seed, completed_steps)``."""
        # `None` is refused here because `derive_stream_seed` reads it as "a fresh run" and would hand back the fresh stream, the
        # replay of epoch 0 that a resume exists to avoid.
        if completed_steps is None:
            raise TypeError("completed_steps must be an int, got None: a resume asks for a stream by the steps it completed")
        stream_seed = derive_stream_seed(self.seed, completed_steps)
        return load_tokenizer_batches(**{**self.request, "stream_seed": stream_seed})


def load_tokenizer_batches(
        *, dataset: str, image_size: int, batch_size: int, seed: int, stream_seed: Optional[int],
        validation_split: Optional[float], max_samples: Optional[int], augment: bool,
) -> TokenizerBatches:
    """The three tokenizer pipelines of ``dataset``; see the module contract."""
    request: Dict[str, Any] = dict(
        dataset=dataset, image_size=image_size, batch_size=batch_size, seed=seed, stream_seed=stream_seed,
        validation_split=validation_split, max_samples=max_samples, augment=augment)
    # `derive_stream_seed(x)` with no step count is `x` itself, after the check that it is an int and not a bool.
    seed = derive_stream_seed(seed)
    stream = seed if stream_seed is None else derive_stream_seed(stream_seed)
    # DECISION plan-2026-11-25T124419-fa94f59c/D-026
    # The SPLIT is keyed on `seed` and the shuffle and augmentation on `stream`. Do NOT pass `stream` as the split seed: a resume
    # moves the stream seed, so the selection rows would change under the weights (the same weights would score another
    # val_reconstruction_loss after a kill, the best-checkpoint reload check would warn on every resumed run, and rows of the
    # selection split would leak into the train stream). Do NOT add an element map: the model's own steps read the image.
    streams = load_image_streams(
        dataset, image_size=image_size, batch_size=batch_size, seed=seed, stream_seed=stream,
        validation_split=validation_split, max_samples=max_samples, augment=augment)
    states_its_length("select_ds", streams.select_ds, streams.val_steps)
    states_its_length("test_ds", streams.test_ds, streams.test_steps)

    split = (f"validation_split={DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split}"
             " of the train set (seeded permutation)")
    return TokenizerBatches(
        train_ds=streams.train_ds, select_ds=streams.select_ds, test_ds=streams.test_ds,
        steps_per_epoch=streams.steps_per_epoch, val_steps=streams.val_steps,
        n_train=streams.n_train, n_select=streams.n_select, n_test=streams.n_test,
        input_shape=(image_size, image_size, 3), num_classes=None,
        selection_split=split, test_split="the dataset's test set",
        seed=seed, stream_seed=stream, request=request)


def _one_pass(stream: "tf.data.Dataset") -> Iterator["tf.Tensor"]:
    """The image batches of one pass of a finite stream; a batch may be ``(image, label)``. An endless stream is refused, not looped."""
    if int(stream.cardinality()) == int(tf.data.INFINITE_CARDINALITY):
        raise ValueError("the stream is endless: one pass of it never ends, so there is nothing to fit or to count")
    for batch in stream:
        yield batch[0] if isinstance(batch, (tuple, list)) else batch


@dataclass(frozen=True)
class ConstantImageBaseline:
    """The best constant image's reconstruction loss on the selection stream, and which constant it was.

    :param kind: :data:`CHANNEL_MEAN` or :data:`MEAN_IMAGE`, whichever scored lower (a tie names the channel mean, the simpler constant).
    :param reconstruction_loss: that lower value, float32 as a Python float: what a model that reconstructs a constant image scores.
    :param channel_mean, mean_image: the two candidates' values, each rounded to float32.
    """

    kind: str
    reconstruction_loss: float
    channel_mean: float
    mean_image: float

    def as_dict(self) -> Dict[str, Any]:
        """Plain values, for ``results_summary.json``."""
        return asdict(self)


def constant_image_baseline(select_ds: "tf.data.Dataset") -> ConstantImageBaseline:
    """The :class:`ConstantImageBaseline` of one pass of ``select_ds``; see the module contract."""
    # DECISION plan-2026-11-25T124419-fa94f59c/D-027
    # The baseline must equal what the model's own `evaluate` reports for a constant reconstruction, so it uses the METRIC's reduction:
    # the mean over batches of the per-batch pixel mean (Keras `Mean` takes one value per batch), so the ragged last batch counts as
    # much as a whole one. Do NOT pool every pixel into one mean: that weights the last batch by its rows and is off by far more than
    # the float32 band. Two passes, float64: pass one fits the constants in sample on the SELECTION images (the best a constant can
    # do there: fitting them on the train rows would make a weaker bar), pass two scores them. The result is rounded to float32 because
    # the bar compares float32 numbers (`forecast_run.beats_bar`).
    rows, total = 0, None
    for images in _one_pass(select_ds):
        images = np.asarray(images, np.float64)
        total = images.sum(axis=0) if total is None else total + images.sum(axis=0)
        rows += len(images)
    if total is None or rows == 0:
        raise ValueError("the selection stream is empty: there are no images to fit a constant image to")
    mean_image = total / rows
    channel_mean = total.sum(axis=(0, 1)) / (rows * total.shape[0] * total.shape[1])
    channel_losses, image_losses = [], []
    for images in _one_pass(select_ds):
        images = np.asarray(images, np.float64)
        channel_losses.append(np.mean((images - channel_mean) ** 2))
        image_losses.append(np.mean((images - mean_image) ** 2))
    channel, image = float(np.float32(np.mean(channel_losses))), float(np.float32(np.mean(image_losses)))
    return ConstantImageBaseline(
        kind=MEAN_IMAGE if image < channel else CHANNEL_MEAN, reconstruction_loss=min(channel, image),
        channel_mean=channel, mean_image=image)


def code_ids_of(model: Any, select_ds: "tf.data.Dataset") -> np.ndarray:
    """The code ids of every image of one pass of ``select_ds``, in row order; see the module contract."""
    encode = getattr(model, "encode_to_indices", None)
    if encode is None:
        raise TypeError(
            f"code_ids_of needs a model with encode_to_indices (a VQVAERotationTrick), got {type(model).__name__}")
    parts = [np.asarray(encode(images)) for images in _one_pass(select_ds)]
    if not parts:
        raise ValueError("the selection stream is empty: there are no images to encode")
    return np.concatenate(parts)


# DECISION plan-2026-11-25T124419-fa94f59c/D-067
# The codebook part of the bar has a SECOND floor, per cell, because the pooled one (`pretrain_run.MIN_TOKENIZER_PERPLEXITY`, 64) cannot tell
# codes that describe content from codes that describe position: CIFAR-10 at 32 px and patch 4 has 8 x 8 = 64 cells, so a code that is the
# cell index alone has a pooled perplexity of exactly 64 and meets it. MEASURED (scratch_step11): the cell index alone is worth 1.0 per
# cell, plus one balanced image-wide bit 2.0, two bits 4.0, three bits 8.0 (2 ** bits); the trained arms of Step 2 (750 steps, 5,000
# selection images) 36.07 and 46.53 per cell (92 and 90 percent of their pooled 39.04 and 51.73), the 2-epoch tiny run 15.6. 8 is the
# power of two nearest the geometric mean of 2.0 (the cheapest code the review named) and 36.07 (the lowest trained value): 4x above
# that code, 2x above two image-wide bits, 4.5x under the trained arms. Do NOT lower it to 2 (position plus one bit then meets it), do NOT
# make it a share of the pooled perplexity or of the codebook (the number is about what one cell carries, not about the book), and do NOT
# touch the pooled floor: `beit_mim` re-judges a tokenizer by it, and it rests on two lost tokenizers.
#: Effective codes a cell must use, on average over the cells, for the codebook part of the bar.
MIN_PER_CELL_PERPLEXITY = 8.0


@dataclass(frozen=True)
class CellUsage:
    """How a tokenizer's codes are used WITHIN a cell, averaged over the cells of the code grid.

    :param cells: positions of the grid (``grid_h * grid_w``). :param images: images counted.
    :param entropy: nats; the mean over the cells of the entropy of that cell's code histogram. :param perplexity: ``exp(entropy)``.
    """

    cells: int
    images: int
    entropy: float
    perplexity: float

    def as_dict(self) -> Dict[str, Any]:
        """Plain numbers, for ``results_summary.json``."""
        return asdict(self)


def per_cell_usage(code_ids: Any) -> CellUsage:
    """The :class:`CellUsage` of a grid of code ids ``(images, grid_h, grid_w)``; see the module contract."""
    ids = np.asarray(code_ids)
    if ids.ndim != 3 or ids.size == 0:
        raise ValueError(f"code ids must be a non-empty (images, grid_h, grid_w) array, got shape {ids.shape}")
    if ids.dtype.kind not in "iu":
        raise ValueError(f"code ids must be integers, got dtype {ids.dtype}")
    if ids.min() < 0:
        raise ValueError(f"code ids must be non-negative, got a minimum of {int(ids.min())}")
    images = ids.shape[0]
    flat = ids.reshape(images, -1).astype(np.int64)  # (images, cells)
    cells, book = flat.shape[1], int(flat.max()) + 1
    # One bincount over (cell, code) pairs: the histogram of every cell at once.
    counts = np.bincount((np.arange(cells, dtype=np.int64) * book + flat).reshape(-1), minlength=cells * book).reshape(cells, book)
    share = counts / images
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(share > 0, share * np.log(share), 0.0)
    entropy = float(-terms.sum(axis=1).mean())
    return CellUsage(cells=cells, images=images, entropy=entropy, perplexity=float(np.exp(entropy)))


def per_cell_meets_floor(usage: CellUsage) -> bool:
    """Whether ``usage`` reaches :data:`MIN_PER_CELL_PERPLEXITY`, judged on the entropy as the pooled floor is; see the module contract."""
    # The constant is read HERE, at the call (a test may patch it); the comparison is on the entropy with the pooled floor's slack, because
    # `exp(log(8))` is not exactly 8 in float64 and a code that uses exactly 8 codes per cell must meet the floor.
    return usage.entropy >= math.log(MIN_PER_CELL_PERPLEXITY) - ENTROPY_ROUNDING_TOLERANCE


@keras.saving.register_keras_serializable(package="train_tokenizer")
class TokenizerCodes(keras.layers.Layer):
    """An image to its code grid, as a layer: what an exported tokenizer file runs; see the module contract."""

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # DECISION plan-2026-11-25T124419-fa94f59c/D-045
        # The function a file runs is a REGISTERED layer around the loaded model, not a functional `keras.Model(image,
        # model.encode_to_indices(image))`: that model is built (the shape inference passes) and fails when it RUNS, because the quantizer's
        # `get_codebook_indices` reshapes to a shape tuple read off the symbolic input, `(None, 8, 8)`, and the `None` batch reaches the
        # Reshape op as a value ("Attempt to convert a value (None) with an unsupported type", measured). Inside a layer's `call` the
        # shape is dynamic and the same method runs. Do NOT wrap the call in a `keras.layers.Lambda` (it saves and does not load under
        # Keras's default safe mode, `forecast_run.ServedForecaster`'s measurement), and do NOT move this class into `dl_techniques` to
        # make the `.keras` file load without `train` (a library change outside this task).
        self.model = model

    def call(self, images: Any) -> Any:
        return self.model.encode_to_indices(images)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "model": keras.saving.serialize_keras_object(self.model)}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TokenizerCodes":
        return cls(**{**config, "model": keras.saving.deserialize_keras_object(config["model"])})

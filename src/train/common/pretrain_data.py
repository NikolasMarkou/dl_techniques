"""Data for the ``pretrain`` task: masked elements on fixed evaluation masks and a stream seed.

ET masked completion and BEiT masked image modelling train on an element built per image,
``((image, mask), target, sample_weight)``. The legacy trainers draw that mask from a shared
stateful generator, so the same weights score a different ``val_loss`` on every pass, and a
resumed ``fit`` replays its stream from the first element. This module builds the same three
pipelines on the stateless maps of ``dl_techniques.datasets.vision`` and
:func:`train.common.image_streams.load_image_streams`, so that ``val_loss`` is a function of the
weights alone and a resume trains on a fresh stream.

This module imports TensorFlow; ``train.api`` imports it inside functions only.

Interface contract
------------------
``load_pretrain_batches(*, method, dataset, image_size, patch_size, batch_size, seed, stream_seed,
validation_split, max_samples, augment, mask_ratio, mask_token_frac, num_mask_patches,
min_mask_patches_per_block, tokenizer_fn)``
    Keyword only; no mask argument has a default (the config is the one home of the defaults).
    :param method: one of :data:`METHODS`. ``et_masked`` takes ``mask_ratio`` and
        ``mask_token_frac``; ``beit_mim`` takes ``num_mask_patches``,
        ``min_mask_patches_per_block`` and ``tokenizer_fn``. The other method's arguments must be
        ``None``.
    :param seed: the split seed, and the key of the select and test masks.
    :param stream_seed: the seed of the train shuffle, the train augmentation and the train mask;
        ``None`` means ``seed``. Never reaches the split.
    :param tokenizer_fn: BATCHED, as ``train.beit.common.load_frozen_tokenizer`` returns it:
        ``(B, H, W, C)`` images to ``(B, gh, gw)`` integer code ids.
    :return: a :class:`PretrainBatches`.
    :raises ValueError: for an unknown method, a missing or foreign mask argument, an image size the
        patch size does not divide, or a select or test stream whose length is not the one pass the
        record states (both lengths are named).
    :raises TypeError: for a seed that is not an ``int`` (a ``bool`` is not).

``PretrainBatches.with_stream(completed_steps)``
    :return: the same request on the train stream of ``derive_stream_seed(seed, completed_steps)``:
        the same split, the same select and test streams, a train stream that is neither a replay
        of the fresh run nor of another count. The data is loaded again.

``derive_stream_seed(seed, completed_steps=None)``
    :return: ``seed`` itself when ``completed_steps`` is ``None`` (a fresh run), otherwise the
        first 4 bytes of ``sha256(f"{seed}:{completed_steps}")`` as an unsigned integer.
    :raises TypeError / ValueError: for a ``bool`` or a non-``int``, and a negative count.

``states_its_length(name, ds, expected)``
    The check that a finite stream states the length of one pass; :func:`load_pretrain_batches` and
    ``train.common.tokenizer_data.load_tokenizer_batches`` call it (one definition, two callers).
    :param name: the stream's name in the refusal. :param ds: a batched ``tf.data.Dataset``.
    :param expected: the batches of one pass.
    :return: ``None``. :raises ValueError: when ``ds.cardinality()`` is not ``expected`` (an unknown cardinality is shown
        as ``unknown``); both numbers are named, because Keras logs ``n / (n + 1)`` of the loss over a stream of unknown length.

``masked_code_ids(select_ds)``
    :return: a 1-D integer array of the BEiT target ids at the masked positions of every element of
        a finite stream (one pass). :raises ValueError: for a stream whose target is not integer.

``tokenizer_usage(code_ids)``
    :return: the :class:`TokenizerUsage` of an integer array of any shape: positions counted,
        distinct codes, perplexity ``exp(entropy)``, top-1 share and entropy (nats) of the code
        histogram. A pure function. :raises ValueError: for an empty, non-integer, boolean or
        negative array.
"""

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.beit_masking import make_stateless_beit_mim_map_fn
from dl_techniques.datasets.vision.masked_patches import make_stateless_masked_patch_map_fn
from train.common.classification_data import DEFAULT_VALIDATION_SPLIT
from train.common.image_streams import ElementMapFn, load_image_streams

ET_MASKED, BEIT_MIM = "et_masked", "beit_mim"

#: The two pretrain methods: ET masked-patch completion and BEiT masked image modelling.
METHODS: Tuple[str, ...] = (ET_MASKED, BEIT_MIM)

#: One salt per stream, so one seed gives the three streams three different mask sequences.
TRAIN_SALT, SELECT_SALT, TEST_SALT = "train", "select", "test"

_ET_ARGS = ("mask_ratio", "mask_token_frac")
_BEIT_ARGS = ("num_mask_patches", "min_mask_patches_per_block", "tokenizer_fn")


def _plain_int(name: str, value: Any) -> int:
    """``value`` as an ``int``; a ``bool`` and anything that is not an ``int`` are refused."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {value!r}")
    return int(value)


def derive_stream_seed(seed: int, completed_steps: Optional[int] = None) -> int:
    """The stream seed of a run that has completed ``completed_steps`` optimizer steps.

    ``None`` (a fresh run) returns ``seed`` itself; a count returns the first 4 bytes of
    ``sha256(f"{seed}:{completed_steps}")`` as an unsigned integer (so a resume at step 0 is
    still not the fresh run). Distinct counts give distinct seeds (tested over a range).
    """
    # DECISION plan-2026-11-12T015747-46e5318f/D-004
    # A resumed fit starts a FRESH iterator, so an infinite stream replays from its first
    # element; the resume therefore wants another stream. The seed that moves is the STREAM
    # seed only (shuffle, augmentation, train mask). Do NOT derive the split from it
    # (`seed + step` re-splits and leaks selection rows into train) and do NOT key the select
    # or test mask on it (the same weights would score a different val_loss after a kill).
    seed = _plain_int("seed", seed)
    if completed_steps is None:
        return seed
    steps = _plain_int("completed_steps", completed_steps)
    if steps < 0:
        raise ValueError(f"completed_steps must be non-negative, got {steps}")
    return int.from_bytes(hashlib.sha256(f"{seed}:{steps}".encode()).digest()[:4], "big")


@dataclass(frozen=True, eq=False)
class PretrainBatches:
    """What the pretrain task consumes; the fields the run sequence reads.

    :param train_ds: infinite, batched ``((image, mask), target, sample_weight)`` stream; the
        task passes ``steps_per_epoch`` to ``fit``. Its mask is keyed on ``(stream_seed, train
        salt, counter)``.
    :param select_ds: the selection split, finite, one pass of ``val_steps`` batches, ragged last
        batch kept. Its mask is keyed on ``(seed, select salt, row position)``.
    :param test_ds: the test split, finite, one pass of ``ceil(n_test / batch_size)`` batches, keyed on
        ``(seed, test salt, row position)``.
    :param steps_per_epoch, val_steps: batches in an epoch and in a selection pass.
    :param n_train, n_select, n_test: rows of the three splits.
    :param input_shape: ``(image_size, image_size, 3)``.
    :param num_classes: ``None``: a pretrain model has no classes.
    :param selection_split, test_split: plain-language provenance of the two splits.
    :param seed, stream_seed: the split seed and the (resolved) stream seed.
    :param request: the keyword arguments :func:`load_pretrain_batches` was called with (the batch size and the
        ``max_samples`` cap among them), so that :meth:`with_stream` rebuilds the same request on another stream seed.
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

    def with_stream(self, completed_steps: int) -> "PretrainBatches":
        """This request on the train stream of ``derive_stream_seed(seed, completed_steps)``."""
        stream_seed = derive_stream_seed(self.seed, _plain_int("completed_steps", completed_steps))
        return load_pretrain_batches(**{**self.request, "stream_seed": stream_seed})


def _check_request(method: str, args: Mapping[str, Any]) -> None:
    """Refuse a method that is not one of :data:`METHODS`, a missing argument and a foreign one."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    own, foreign = (_ET_ARGS, _BEIT_ARGS) if method == ET_MASKED else (_BEIT_ARGS, _ET_ARGS)
    for name in own:
        if args[name] is None:
            raise ValueError(f"method {method!r} needs {name}; none was given")
    for name in foreign:
        if args[name] is not None:
            raise ValueError(
                f"{name} belongs to the {BEIT_MIM if method == ET_MASKED else ET_MASKED!r} method, "
                f"not to {method!r}")


def _keyed_maps(
        method: str, *, image_size: int, patch_size: int, seed: int, stream: int,
        args: Mapping[str, Any]) -> Tuple[ElementMapFn, ElementMapFn, ElementMapFn]:
    """The train, select and test element maps, each keyed on its own ``(seed, salt)``."""
    grid = image_size // patch_size
    tokenizer_fn: Optional[Callable[..., Any]] = args["tokenizer_fn"]

    def one_map(salt: str, key: int) -> ElementMapFn:
        if method == ET_MASKED:
            return make_stateless_masked_patch_map_fn(
                patch_size, image_size, args["mask_ratio"], args["mask_token_frac"],
                seed=key, salt=salt)
        return make_stateless_beit_mim_map_fn(
            # the map hands over ONE unbatched image; the tokenizer wants a batch
            lambda image: tokenizer_fn(image[None])[0], (grid, grid),
            args["num_mask_patches"], args["min_mask_patches_per_block"], seed=key, salt=salt)

    # DECISION plan-2026-11-12T015747-46e5318f/D-005
    # The SELECT and TEST masks are keyed on the SPLIT seed (`seed`) and the row position, the
    # TRAIN mask on the stream seed and the counter. Do NOT key select or test on `stream`: it
    # changes on every resume, so the same weights would score a different val_loss after a
    # kill, the best-checkpoint reload check would warn on every resumed run and the bar would
    # not be comparable across it. Do NOT share one salt: the three streams must not see the
    # same masks at the same index.
    return one_map(TRAIN_SALT, stream), one_map(SELECT_SALT, seed), one_map(TEST_SALT, seed)


def states_its_length(name: str, ds: "tf.data.Dataset", expected: int) -> None:
    """Refuse a finite stream whose cardinality is not the length of one pass.

    Keras logs ``n / (n + 1)`` of the loss over a dataset of unknown length, so both numbers
    are named. MEASURED: ``enumerate().map().batch()`` keeps a known length, only ``filter``
    loses it, so this is a statement that is true today and a refusal if it stops being true.
    """
    got = int(ds.cardinality())
    if got != expected:
        shown = "unknown" if got == int(tf.data.UNKNOWN_CARDINALITY) else str(got)
        raise ValueError(
            f"{name} has cardinality {shown}, but one pass is {expected} batches: Keras would "
            "score it wrongly (a dataset of unknown length logs n / (n + 1) of the loss)")


def load_pretrain_batches(
        *, method: str, dataset: str, image_size: int, patch_size: int, batch_size: int,
        seed: int, stream_seed: Optional[int], validation_split: Optional[float],
        max_samples: Optional[int], augment: bool,
        mask_ratio: Optional[float] = None, mask_token_frac: Optional[float] = None,
        num_mask_patches: Optional[int] = None, min_mask_patches_per_block: Optional[int] = None,
        tokenizer_fn: Optional[Callable[..., Any]] = None,
) -> PretrainBatches:
    """The three pretrain pipelines of ``dataset``; see the module contract."""
    request: Dict[str, Any] = dict(
        method=method, dataset=dataset, image_size=image_size, patch_size=patch_size,
        batch_size=batch_size, seed=seed, stream_seed=stream_seed,
        validation_split=validation_split, max_samples=max_samples, augment=augment,
        mask_ratio=mask_ratio, mask_token_frac=mask_token_frac,
        num_mask_patches=num_mask_patches,
        min_mask_patches_per_block=min_mask_patches_per_block, tokenizer_fn=tokenizer_fn)
    _check_request(method, request)
    seed = _plain_int("seed", seed)
    stream = seed if stream_seed is None else _plain_int("stream_seed", stream_seed)
    if image_size % patch_size != 0:
        raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")

    train_map, select_map, test_map = _keyed_maps(
        method, image_size=image_size, patch_size=patch_size, seed=seed, stream=stream,
        args=request)
    streams = load_image_streams(
        dataset, image_size=image_size, batch_size=batch_size, seed=seed, stream_seed=stream,
        validation_split=validation_split, max_samples=max_samples, augment=augment,
        indexed_element_map_fn=train_map, select_indexed_map_fn=select_map,
        test_indexed_map_fn=test_map)
    states_its_length("select_ds", streams.select_ds, streams.val_steps)
    states_its_length("test_ds", streams.test_ds, streams.test_steps)

    split = (f"validation_split={DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split}"
             " of the train set (seeded permutation)")
    return PretrainBatches(
        train_ds=streams.train_ds, select_ds=streams.select_ds, test_ds=streams.test_ds,
        steps_per_epoch=streams.steps_per_epoch, val_steps=streams.val_steps,
        n_train=streams.n_train, n_select=streams.n_select, n_test=streams.n_test,
        input_shape=(image_size, image_size, 3), num_classes=None,
        selection_split=split, test_split="the dataset's test set",
        seed=seed, stream_seed=stream, request=request)


def masked_code_ids(select_ds: "tf.data.Dataset") -> np.ndarray:
    """The BEiT target ids at the masked positions of one pass of ``select_ds``, as one 1-D array."""
    spec = select_ds.element_spec
    if not spec[1].dtype.is_integer:
        raise ValueError(
            f"masked_code_ids reads integer target ids, but the stream's target is {spec[1].dtype.name}")
    parts = []
    for (_, mask), ids, _ in select_ds:
        parts.append(ids.numpy()[mask.numpy().astype(bool)])
    return np.concatenate(parts) if parts else np.zeros((0,), np.int64)


@dataclass(frozen=True)
class TokenizerUsage:
    """How a tokenizer's codes are used over a set of positions.

    :param n_codes: positions counted. :param distinct: distinct codes seen.
    :param perplexity: ``exp(entropy)``, the number of equally used codes the histogram is worth.
    :param top1_share: the share of the most frequent code. :param entropy: nats.
    """

    n_codes: int
    distinct: int
    perplexity: float
    top1_share: float
    entropy: float

    def as_dict(self) -> Dict[str, Any]:
        """Plain numbers, for ``results_summary.json``."""
        return asdict(self)


def tokenizer_usage(code_ids: Any) -> TokenizerUsage:
    """The :class:`TokenizerUsage` of an integer array of code ids of any shape."""
    ids = np.asarray(code_ids)
    if ids.size == 0:
        raise ValueError("code ids are empty: there is no usage to report")
    if ids.dtype.kind not in "iu":
        raise ValueError(f"code ids must be integers, got dtype {ids.dtype}")
    if ids.min() < 0:
        raise ValueError(f"code ids must be non-negative, got a minimum of {int(ids.min())}")
    counts = np.bincount(ids.reshape(-1).astype(np.int64)).astype(np.float64)
    counts = counts[counts > 0]
    share = counts / counts.sum()
    entropy = float(-(share * np.log(share)).sum())
    return TokenizerUsage(
        n_codes=int(ids.size), distinct=int(len(counts)), perplexity=float(np.exp(entropy)),
        top1_share=float(share.max()), entropy=entropy)

"""Data for the ``dino`` task: the multi-crop train stream, plain selection and test streams, and the k-NN probe's cached streams.

A DINO run reads four kinds of stream. The student and the teacher train on the multi-crop element ``(views, label)`` of
:func:`dl_techniques.datasets.vision.multi_crop.make_stateless_multi_crop_map_fn`; the selection and test splits are plain ``(image, label)``
batches (the probe scores the representation of a plain image, not of an augmented view); and the probe's memory bank and queries are
``.take(n).cache()`` of the plain fit rows and of the selection rows. ``train.dino.train_dino.build_dataset`` and ``build_knn_datasets`` do this
from the legacy 39-field ``TrainingConfig`` and seed the split, the shuffle and the crops from ONE seed, so they cannot give a resumed run a
stream of its own. This module calls the same two primitives (``load_image_streams`` and the stateless multi-crop map) and adds what a
resume needs: a batches object whose :meth:`DinoBatches.with_stream` moves the shuffle and the crops and nothing else.

This module imports TensorFlow; ``train.api`` imports it inside functions only.

Interface contract
------------------
``load_dino_batches(*, dataset, image_size, n_local_crops, batch_size, seed, stream_seed, validation_split, max_samples,
knn_bank_batches, knn_query_batches)``
    Keyword only; no argument has a default (the task's config is the one home of the defaults).
    :param image_size: the global crop size, which is also the local crop size and the size of the plain images (every view of the model has
        the same pixel resolution).
    :param n_local_crops: local crops per image (0 or more); a view count of ``2 + n_local_crops``.
    :param seed: the split seed: the selection, test and probe rows are a function of it alone.
    :param stream_seed: the seed of the train shuffle AND of the stateless crops; ``None`` means ``seed``. Never reaches the split or the probe.
    :param validation_split, max_samples: as in ``load_image_streams``; the cap bounds the train pool, never the test split.
    :param knn_bank_batches, knn_query_batches: the probe's bank is the first ``knn_bank_batches`` batches of the plain fit stream; its
        queries are the first ``knn_query_batches`` batches of the selection split (all of it when the split is shorter).
    :return: a :class:`DinoBatches`.
    :raises TypeError: for a seed, stream seed, crop count or batch count that is not an ``int`` (a ``bool`` is not), before any data is read.
    :raises ValueError: for a negative crop count, a batch count below 1, or a finite stream whose length is not the one pass it should
        state (both lengths are named).

``DinoBatches``
    What the run sequence, the trainer and the probe read: ``train_ds`` (infinite, ``(views, label)`` with ``views`` of shape
    ``(batch, 2 + n_local_crops, S, S, 3)``, whole batches only), ``select_ds`` and ``test_ds`` (finite plain ``(image, label)``, never
    augmented, every row kept), ``probe_bank_ds`` and ``probe_query_ds`` (cached, finite, plain), ``loss_batch_ds`` (one multi-crop batch of
    the first selection rows, for the epoch-zero loss), ``steps_per_epoch``, ``val_steps``,
    ``test_steps``, the three row counts, ``bank_batches`` and ``query_batches`` (the batches the two probe streams hold), ``input_shape``
    ``(S, S, 3)`` (one plain image), ``n_views``, ``num_classes`` (``None``: SSL has no classes), the two provenance sentences, the split seed,
    the resolved stream seed and ``request``, the keyword arguments the batches were built from.

``DinoBatches.with_stream(completed_steps)``
    :return: the same request on the train stream of ``derive_stream_seed(seed, completed_steps)``: the same split, the same plain streams,
        the same probe rows, a train stream whose shuffle and crops are neither a replay of the fresh run nor of another count. The data is
        loaded again.
    :raises TypeError / ValueError: for ``None``, a ``bool``, a non-``int`` or a negative count.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import tensorflow as tf

from dl_techniques.datasets.vision.multi_crop import make_stateless_multi_crop_map_fn
from train.common.classification_data import DEFAULT_VALIDATION_SPLIT
from train.common.image_streams import load_image_streams
from train.common.pretrain_data import derive_stream_seed, states_its_length

#: The two global crops every DINO view set starts with (views 0 and 1 are the teacher's).
N_GLOBAL_VIEWS = 2


@dataclass(frozen=True, eq=False)
class DinoBatches:
    """What the ``dino`` task consumes; the fields the run sequence and the probe read (see the module contract).

    :param train_ds: infinite, batched ``(views, label)``; the task passes ``steps_per_epoch`` to ``fit``. The crops are keyed on
        ``(stream_seed, counter, row)`` and the shuffle on ``stream_seed``.
    :param select_ds: the selection split, finite, one pass of ``val_steps`` plain batches, ragged last batch kept.
    :param test_ds: the test split, finite, one pass of ``test_steps`` plain batches.
    :param probe_bank_ds, probe_query_ds: ``bank_batches`` plain batches of the fit rows and ``query_batches`` plain batches of the selection
        rows, each ``.take(n).cache()`` of a stream on the split seed, so every probe of the run reads the same rows.
    :param loss_batch_ds: ONE multi-crop batch of ``(views, label)``: the first selection batch put through the same stateless multi-crop map,
        keyed on ``(stream seed, row)``, the same on every pass. Epoch zero feeds it to a throwaway loss. It is read from the selection split
        and not from a train or a bank stream: every pass of those is a new shuffle (and a half-read ``.cache()`` is dropped and refilled from
        a new pass), so a batch read from them for a diagnostic would change what ``fit`` and the probe see.
    :param steps_per_epoch, val_steps, test_steps: batches in an epoch and in a selection and a test pass.
    :param n_train, n_select, n_test: rows of the three splits.
    :param bank_batches, query_batches: batches of the two probe streams.
    :param input_shape: ``(image_size, image_size, 3)``, one plain image (a view has the same shape).
    :param n_views: ``2 + n_local_crops``.
    :param num_classes: ``None``: the task has no classes (the probe's labels are only read by the k-NN vote).
    :param selection_split, test_split: plain-language provenance of the two splits.
    :param seed, stream_seed: the split seed and the (resolved) stream seed.
    :param request: the keyword arguments :func:`load_dino_batches` was called with, so that :meth:`with_stream` rebuilds the same request
        on another stream seed.
    """

    train_ds: "tf.data.Dataset"
    select_ds: "tf.data.Dataset"
    test_ds: "tf.data.Dataset"
    probe_bank_ds: "tf.data.Dataset"
    probe_query_ds: "tf.data.Dataset"
    loss_batch_ds: "tf.data.Dataset"
    steps_per_epoch: int
    val_steps: int
    test_steps: int
    n_train: int
    n_select: int
    n_test: int
    bank_batches: int
    query_batches: int
    input_shape: Tuple[int, int, int]
    n_views: int
    num_classes: Optional[int]
    selection_split: str
    test_split: str
    seed: int
    stream_seed: int
    request: Mapping[str, Any] = field(repr=False)

    def with_stream(self, completed_steps: int) -> "DinoBatches":
        """This request on the train stream of ``derive_stream_seed(seed, completed_steps)``."""
        # `None` is refused here because `derive_stream_seed` reads it as "a fresh run" and would hand back the fresh stream, the
        # replay of epoch 0 that a resume exists to avoid.
        if completed_steps is None:
            raise TypeError("completed_steps must be an int, got None: a resume asks for a stream by the steps it completed")
        stream_seed = derive_stream_seed(self.seed, completed_steps)
        return load_dino_batches(**{**self.request, "stream_seed": stream_seed})


def _count(name: str, value: Any, *, at_least: int) -> int:
    """``value`` as an ``int`` of at least ``at_least``; a ``bool`` and anything that is not an ``int`` are refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < at_least:
        raise ValueError(f"{name} must be at least {at_least}, got {value}")
    return value


def load_dino_batches(
        *, dataset: str, image_size: int, n_local_crops: int, batch_size: int, seed: int, stream_seed: Optional[int],
        validation_split: Optional[float], max_samples: Optional[int], knn_bank_batches: int, knn_query_batches: int,
) -> DinoBatches:
    """The DINO pipelines of ``dataset``; see the module contract."""
    request: Dict[str, Any] = dict(
        dataset=dataset, image_size=image_size, n_local_crops=n_local_crops, batch_size=batch_size, seed=seed,
        stream_seed=stream_seed, validation_split=validation_split, max_samples=max_samples,
        knn_bank_batches=knn_bank_batches, knn_query_batches=knn_query_batches)
    # `derive_stream_seed(x)` with no step count is `x` itself, after the check that it is an int and not a bool.
    seed = derive_stream_seed(seed)
    stream = seed if stream_seed is None else derive_stream_seed(stream_seed)
    n_local_crops = _count("n_local_crops", n_local_crops, at_least=0)
    bank_asked = _count("knn_bank_batches", knn_bank_batches, at_least=1)
    query_asked = _count("knn_query_batches", knn_query_batches, at_least=1)

    # DECISION plan-2026-11-26T224507-e3381378/D-038
    # Two loads, and the line between them is the point. The TRAIN load is keyed on `stream` (the shuffle) and its multi-crop map on `stream`
    # (the crops); the PLAIN load, which holds the selection and test streams and the probe's bank and queries, is keyed on `seed` alone and
    # never sees `stream`. Both are SPLIT on `seed`. Do NOT key anything of the plain load on `stream`: a resume moves the stream seed, so the
    # selection rows would change under the weights (the same weights would score another k-NN accuracy after a kill) and the bank would
    # change, and the run's probe would stop being comparable across the kill. Do NOT keep the map seed fixed while the shuffle moves: the
    # resume would replay the crops of the fresh run on a reshuffled order. Do NOT take the bank from the selection stream: a query would find
    # itself at cosine 1.0. `augment=False` is the DINO recipe: its augmentation is the multi-crop map alone, and the loader's own flip and
    # pad-crop would stack a second one under it.
    map_fn = make_stateless_multi_crop_map_fn(
        global_crop_size=image_size, seed=stream, local_crop_size=image_size, n_local_crops=n_local_crops)
    train = load_image_streams(
        dataset, image_size=image_size, batch_size=batch_size, seed=seed, stream_seed=stream,
        validation_split=validation_split, max_samples=max_samples, augment=False, indexed_element_map_fn=map_fn)
    plain = load_image_streams(
        dataset, image_size=image_size, batch_size=batch_size, seed=seed, stream_seed=None,
        validation_split=validation_split, max_samples=max_samples, augment=False)
    states_its_length("select_ds", plain.select_ds, plain.val_steps)
    states_its_length("test_ds", plain.test_ds, plain.test_steps)

    bank_batches, query_batches = bank_asked, min(query_asked, plain.val_steps)
    probe_bank = plain.train_ds.take(bank_batches).cache()
    probe_query = plain.select_ds.take(query_batches).cache()
    states_its_length("probe_bank_ds", probe_bank, bank_batches)
    states_its_length("probe_query_ds", probe_query, query_batches)
    # DECISION plan-2026-11-26T224507-e3381378/D-038
    # One batch of multi-crop views for the epoch-zero loss, from the first SELECTION batch (unshuffled, so the same on every pass) through the
    # same map. Do NOT read it from `train.train_ds`: every iteration of that stream is a new shuffle drawn from one seed generator, so a
    # batch read for a diagnostic would change the order `fit` then trains on. Do NOT read it from `probe_bank` either: `.take(1)` over a
    # `.cache()` is a PARTIAL pass, tf.data drops a half-filled cache and refills it from the next pass of the shuffled source, and the bank
    # would then hold other rows than the ones a full read cached (measured by `test_the_loss_batch_...`). The probe streams are read in
    # full or not at all.
    loss_batch = plain.select_ds.take(1).unbatch().enumerate().map(
        lambda index, element: map_fn(index, *element)).batch(batch_size)

    split = (f"validation_split={DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split}"
             " of the train set (seeded permutation)")
    return DinoBatches(
        train_ds=train.train_ds, select_ds=plain.select_ds, test_ds=plain.test_ds,
        probe_bank_ds=probe_bank, probe_query_ds=probe_query, loss_batch_ds=loss_batch,
        steps_per_epoch=train.steps_per_epoch, val_steps=plain.val_steps, test_steps=plain.test_steps,
        n_train=train.n_train, n_select=plain.n_select, n_test=plain.n_test,
        bank_batches=bank_batches, query_batches=query_batches,
        input_shape=(image_size, image_size, 3), n_views=N_GLOBAL_VIEWS + n_local_crops, num_classes=None,
        selection_split=split, test_split="the dataset's test set",
        seed=seed, stream_seed=stream, request=request)

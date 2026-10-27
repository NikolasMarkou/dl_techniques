"""State a finite ``tf.data`` dataset's length by counting it once.

Why a dataset reaching ``fit``/``evaluate`` must state its length
------------------------------------------------------------------
Keras 3.8 gives a dataset of UNKNOWN cardinality (a ``from_generator`` source,
or a ``filter``/``flat_map``/packing step) two silent failure modes (MEASURED,
plan-2026-10-14T062616-81938e81 D-258 and D-265):

* **No ``steps_per_epoch``/``validation_steps``.** The epoch end is found by one
  more train/test call that raises ``OutOfRange``; when that step is not
  XLA-compiled (``jit_compile=False``, or ``"auto"`` on a CPU-only machine) and
  the batch axis is STATIC (``drop_remainder=True`` or a fixed-batch generator
  signature), the loss and ``Mean`` metric trackers' sample-count update has no
  data dependency and still runs, so every logged loss and metric is
  ``n / (n + 1)`` of the mean over ``n`` batches. A dynamic batch axis makes the
  count data-dependent, and an XLA step never runs; both log the true mean.
* **``steps_per_epoch`` equal to the dataset's length.** ``fit`` keeps one
  iterator across epochs and resets it only once it runs dry, so every second
  epoch trains on ZERO batches and logs a loss of 0.0.

A stated count (``tf.data.experimental.assert_cardinality``) removes both: the
epoch iterator stops at the count without the extra call and starts a fresh
iterator for the next epoch. A never-ending source states
``tf.data.INFINITE_CARDINALITY`` (or ends in ``repeat()``) and is driven with
``steps_per_epoch``.

:func:`state_counted_length` is for a finite source whose count is only
knowable by running it (a packed token stream): it counts one pass and states
that count. When the count is computable without a pass, state it directly.
"""

import numpy as np
import tensorflow as tf


def state_counted_length(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Count a finite dataset once and state the count as its cardinality.

    Interface contract (callers: ``train.common.nlp.preprocess_clm_packed_dataset``
    and :func:`dl_techniques.datasets.byte_lm.build_byte_clm_dataset`, each on
    its non-repeating path): ``dataset`` must be FINITE and yield the same
    NUMBER of elements on every pass (shuffles are fine; a draw that changes the
    count is not, the assertion then raises mid-epoch). Returns ``dataset``
    unchanged when its cardinality is already known; otherwise runs it once to
    the end (one full pass of its cost, at call time) and returns it with
    ``assert_cardinality(count)`` applied. A count of 0 is stated as 0.

    :param dataset: A finite dataset.
    :type dataset: tf.data.Dataset
    :return: The same dataset with a known cardinality.
    :rtype: tf.data.Dataset
    :raises ValueError: If ``dataset`` states an infinite cardinality: there
        is nothing to count, and the caller must pass ``steps_per_epoch``.
    """
    cardinality = int(dataset.cardinality())
    if cardinality == int(tf.data.INFINITE_CARDINALITY):
        raise ValueError(
            "state_counted_length: the dataset is infinite (it repeats); "
            "count only a finite dataset and drive an infinite one with "
            "steps_per_epoch"
        )
    if cardinality >= 0:
        return dataset
    count = int(dataset.reduce(np.int64(0), lambda total, _: total + 1))
    return dataset.apply(tf.data.experimental.assert_cardinality(count))

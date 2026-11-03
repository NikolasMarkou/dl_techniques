"""The loss and the metrics a segmentation run compiles with.

One home for what the ``image_segmentation`` task of ``train.api`` and any later segmentation
trainer put in ``model.compile``: stock Keras pieces for everything Keras can already do with a void
label, and one registered metric (:class:`~dl_techniques.metrics.void_aware_accuracy.VoidAwareAccuracy`)
for the one thing it cannot.

Interface contract
------------------
``SEGMENTATION_METRIC_NAMES``
    ``("accuracy", "miou")``: the keys of the history (and, with ``val_``, of the validation
    columns) whatever the ignore index is, so ``val_miou`` is the monitor in both cases. Known
    without building a metric, which a task's ``metric_names`` hook needs before the GPU exists.

``segmentation_loss(ignore_index)``
    :param ignore_index: the void label value, or ``None``.
    :return: ``SparseCategoricalCrossentropy(from_logits=True, ignore_class=ignore_index)``. The model
        emits LOGITS ``(B, H, W, C)`` (a softmax head would be softmaxed twice) and the targets are
        sparse ``(B, H, W)`` ids. A void pixel contributes nothing to the value or the gradient;
        the value of a batch is the mean over its valid pixels, and a batch with none is ``0``.

``segmentation_metrics(num_classes, ignore_index)``
    :param num_classes: classes (void is not one).
    :param ignore_index: the void label value, or ``None``.
    :return: ``[accuracy, miou]``. ``miou`` is ``MeanIoU(num_classes, sparse_y_true=True,
        sparse_y_pred=False, ignore_class=ignore_index, name="miou")`` (a confusion matrix pooled
        over every pixel since the reset). ``accuracy`` is the stock ``SparseCategoricalAccuracy``
        when ``ignore_index`` is ``None`` and ``VoidAwareAccuracy(ignore_index)`` otherwise: the stock
        metric would score each void pixel as wrong.
    :raises ValueError: from Keras, for ``num_classes`` below 2.
"""

from typing import List, Optional

import keras

from dl_techniques.metrics.void_aware_accuracy import VoidAwareAccuracy

#: The names of the compiled metrics, whatever the ignore index.
SEGMENTATION_METRIC_NAMES = ("accuracy", "miou")


def segmentation_loss(ignore_index: Optional[int]) -> keras.losses.Loss:
    """The cross-entropy of logits against sparse targets, void left out; see the module contract."""
    return keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=ignore_index)


def segmentation_metrics(num_classes: int, ignore_index: Optional[int]) -> List[keras.metrics.Metric]:
    """``[accuracy, miou]`` with void left out; see the module contract."""
    # DECISION plan-2026-10-31T180033-2faf890a/D-076
    # Stock Keras for the loss and mIoU, ONE custom metric for accuracy, and only when there is a
    # void label. `SparseCategoricalAccuracy` scores void as wrong (measured 0.5 where 2 of 4
    # pixels are void and the others right) and has no `ignore_class`; with no void label it is the
    # same number as ours and loads with no registration, so it stays. Do NOT use the stock accuracy
    # under an ignore index "because its name matches", and do NOT drop `ignore_class` from MeanIoU
    # (a 255 label then raises `indices[...] does not index into shape`).
    if ignore_index is None:
        accuracy = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    else:
        accuracy = VoidAwareAccuracy(ignore_index, name="accuracy")
    miou = keras.metrics.MeanIoU(
        num_classes=num_classes, sparse_y_true=True, sparse_y_pred=False,
        ignore_class=ignore_index, name="miou")
    return [accuracy, miou]

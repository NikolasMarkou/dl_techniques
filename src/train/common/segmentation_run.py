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

``segmentation_loss(name, ignore_index, **knobs)``
    :param name: ``ce``, ``dice``, ``ce_dice`` or ``focal`` (``train.api.segmentation_args.LOSS_KNOBS``
        has the same four keys; a test holds them equal).
    :param ignore_index: the void label value, or ``None``.
    :param knobs: ``dice_smooth`` (read by ``dice`` and ``ce_dice``; default 1e-5) or ``focal_gamma`` (read
        by ``focal``; default 2.0). Pass only the ones given: which loss reads which is
        ``train.api.segmentation_args.LOSS_KNOBS``, the config refuses the rest by name.
    :return: for ``ce``, ``SparseCategoricalCrossentropy(from_logits=True, ignore_class=ignore_index)``;
        for the others a :mod:`~dl_techniques.losses.sparse_segmentation_loss` loss (``SparseDiceLoss``,
        ``SparseCeDiceLoss``, ``SparseFocalLoss``). The model emits LOGITS ``(B, H, W, C)`` (a softmax
        head would be softmaxed twice) and the targets are sparse ``(B, H, W)`` ids. A void pixel
        contributes nothing to the value or the gradient. The value of a ``ce`` batch is the mean over
        ITS valid pixels (a batch with none is ``0``); the others are one value per IMAGE, averaged by
        Keras (an image with none is ``0``).
    :raises ValueError: for an unknown ``name``.
    :raises TypeError: for a knob the named loss does not take (Python's own refusal: the task config
        refuses these first, with the dotted key).

``segmentation_metrics(num_classes, ignore_index)``
    :param num_classes: classes (void is not one).
    :param ignore_index: the void label value, or ``None``.
    :return: ``[accuracy, miou]``. ``miou`` is ``MeanIoU(num_classes, sparse_y_true=True,
        sparse_y_pred=False, ignore_class=ignore_index, name="miou")`` (a confusion matrix pooled
        over every pixel since the reset). ``accuracy`` is the stock ``SparseCategoricalAccuracy``
        when ``ignore_index`` is ``None`` and ``VoidAwareAccuracy(ignore_index)`` otherwise: the stock
        metric would score each void pixel as wrong.
    :raises ValueError: from Keras, for ``num_classes`` below 2.

``class_pixel_counts(dataset, num_classes, ignore_index)``
    :param dataset: a batched dataset of ``(image, mask)``; every batch is read once.
    :return: ``int64`` ``(num_classes,)``: the pixels of each class, void left out.
    :raises ValueError: for a mask value that is neither a class nor ``ignore_index``.

``confusion_from_dataset(model, dataset, num_classes, ignore_index)``
    :param model: a model whose call returns per-pixel logits ``(B, H, W, C)``.
    :param dataset: a batched dataset of ``(image, mask)``; every batch is read once.
    :return: ``int64`` ``(C, C)``, entry ``[t, p]`` the pixels of true class ``t`` predicted ``p`` (argmax of
        the logits, the model run with ``training=False``), pooled over the whole dataset, void left out. Its
        mIoU and pixel accuracy are the ones Keras' ``MeanIoU`` and ``VoidAwareAccuracy`` report for the same pixels.

``majority_class_baseline(counts, class_names)``
    :param counts: the result of :func:`class_pixel_counts`.
    :return: the scores of predicting the most frequent class (the lowest id on a tie) for every
        valid pixel: ``{"class", "class_name", "miou", "pixel_accuracy", "per_class_iou",
        "pixels"}``. ``miou`` is ``keras.metrics.MeanIoU``'s, which leaves a class with no true and
        no predicted pixel out of the mean (``train.convunext.segmentation_viz.class_scores`` is
        the one implementation of that rule). It is the floor a segmenter must clear, and the
        default ``metric_args.success_threshold`` of the segmentation task.
    :raises ValueError: when ``counts`` holds no pixel (every pixel void, or no pixel).

``beats_baseline(reached, baseline)``
    :param reached: the best ``val_miou`` of a run (Keras' ``float32``).
    :param baseline: the ``miou`` of :func:`majority_class_baseline` (``float64``).
    :return: whether ``reached`` is STRICTLY above ``baseline``, both taken to ``float32`` and the
        baseline raised by :data:`TIE_ULPS` ``float32`` steps of its own size. A model that predicts
        the majority class everywhere IS the baseline, and Keras' pooled ``float32`` result for it
        differs from the ``float64`` figure by up to 0.91 ``float32`` steps (measured over 200
        random selection sets, 2 to 150 classes), so a bare ``>=`` called it a success in 8 of 20
        cases and a bare ``>`` would in some. ``NaN`` is not above anything.
"""

from typing import Any, Dict, List, Optional, Sequence

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.metrics.void_aware_accuracy import VoidAwareAccuracy

#: The names of the compiled metrics, whatever the ignore index.
SEGMENTATION_METRIC_NAMES = ("accuracy", "miou")

#: ``float32`` steps (``numpy.spacing`` of the baseline) by which ``val_miou`` must clear the
#: majority-class baseline. Measured: Keras' ``float32`` mIoU of the constant majority predictor
#: lies within 0.91 steps of the exact value; 16 steps leave room for the ``float32`` accumulation of
#: pixel counts above 2**24 and are still about 1e-6 of an mIoU, far below any learning.
TIE_ULPS = 16


def _ce_loss(ignore_index: Optional[int]) -> keras.losses.Loss:
    return keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=ignore_index)


def _dice_loss(ignore_index: Optional[int], dice_smooth: Optional[float] = None) -> keras.losses.Loss:
    from dl_techniques.losses.sparse_segmentation_loss import DEFAULT_DICE_SMOOTH, SparseDiceLoss

    return SparseDiceLoss(
        ignore_index=ignore_index, smooth=DEFAULT_DICE_SMOOTH if dice_smooth is None else dice_smooth)


def _ce_dice_loss(ignore_index: Optional[int], dice_smooth: Optional[float] = None) -> keras.losses.Loss:
    from dl_techniques.losses.sparse_segmentation_loss import DEFAULT_DICE_SMOOTH, SparseCeDiceLoss

    return SparseCeDiceLoss(
        ignore_index=ignore_index, smooth=DEFAULT_DICE_SMOOTH if dice_smooth is None else dice_smooth)


def _focal_loss(ignore_index: Optional[int], focal_gamma: Optional[float] = None) -> keras.losses.Loss:
    from dl_techniques.losses.sparse_segmentation_loss import DEFAULT_FOCAL_GAMMA, SparseFocalLoss

    return SparseFocalLoss(
        ignore_index=ignore_index, gamma=DEFAULT_FOCAL_GAMMA if focal_gamma is None else focal_gamma)


_LOSS_BUILDERS = {"ce": _ce_loss, "dice": _dice_loss, "ce_dice": _ce_dice_loss, "focal": _focal_loss}


def segmentation_loss(name: str, ignore_index: Optional[int], **knobs: float) -> keras.losses.Loss:
    """The loss of logits against sparse targets, void left out; see the module contract."""
    if name not in _LOSS_BUILDERS:
        raise ValueError(f"unknown segmentation loss {name!r}; expected one of {tuple(_LOSS_BUILDERS)}")
    return _LOSS_BUILDERS[name](ignore_index, **knobs)


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


def class_pixel_counts(
    dataset: "tf.data.Dataset", num_classes: int, ignore_index: Optional[int]
) -> np.ndarray:
    """The pixels of each class in a batched ``(image, mask)`` dataset, void left out."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for masks in dataset.map(lambda image, mask: mask).as_numpy_iterator():
        flat = masks.reshape(-1).astype(np.int64)
        if ignore_index is not None:
            flat = flat[flat != ignore_index]
        if flat.size and (flat.min() < 0 or flat.max() >= num_classes):
            raise ValueError(
                f"a mask holds the value {int(flat.min() if flat.min() < 0 else flat.max())}, "
                f"which is neither a class (0..{num_classes - 1}) nor the ignore index "
                f"{ignore_index!r}"
            )
        counts += np.bincount(flat, minlength=num_classes)
    return counts


def confusion_from_dataset(
    model: keras.Model, dataset: "tf.data.Dataset", num_classes: int, ignore_index: Optional[int]
) -> np.ndarray:
    """The pooled pixel confusion of ``model`` over a batched ``(image, mask)`` dataset; see the module contract."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, masks in dataset.as_numpy_iterator():
        logits = keras.ops.convert_to_numpy(model(images, training=False))
        predicted = logits.argmax(axis=-1).reshape(-1).astype(np.int64)
        truth = masks.reshape(-1).astype(np.int64)
        if ignore_index is not None:
            keep = truth != ignore_index
            truth, predicted = truth[keep], predicted[keep]
        confusion += np.bincount(truth * num_classes + predicted, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
    return confusion


def majority_class_baseline(counts: np.ndarray, class_names: Sequence[str]) -> Dict[str, Any]:
    """The scores of the constant majority-class predictor; see the module contract."""
    # DECISION plan-2026-10-31T180033-2faf890a/D-077
    # Imported here: train.convunext.segmentation_viz is a standalone module (keras, numpy and
    # matplotlib only) that holds the ONE implementation of IoU in the package (its docstring).
    # Do NOT re-derive the formula (`n_majority / N / C` is wrong whenever a class is absent from
    # the selection split): Keras leaves a class with an empty union out of the mean, and
    # `test_the_baseline_equals_keras_mean_iou_of_the_constant_prediction` holds the two equal.
    from train.convunext.segmentation_viz import class_scores

    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() == 0:
        raise ValueError("no valid pixel: every pixel is void, so there is no majority class")
    majority = int(np.argmax(counts))
    confusion = np.zeros((len(counts), len(counts)), dtype=np.int64)
    confusion[:, majority] = counts
    scores = class_scores(confusion)
    return {
        "class": majority,
        "class_name": str(class_names[majority]),
        "miou": scores["miou"],
        "pixel_accuracy": scores["pixel_accuracy"],
        "per_class_iou": scores["per_class_iou"],
        "pixels": int(counts.sum()),
    }


def beats_baseline(reached: float, baseline: float) -> bool:
    """Whether ``reached`` is strictly above ``baseline`` in ``float32``; see the module contract."""
    # DECISION plan-2026-10-31T180033-2faf890a/D-087
    # The constant majority predictor scores exactly the baseline, so a tie must be a "no". Do NOT
    # compare Keras' float32 against the float64 baseline with `>=` (8 of 20 constant models
    # passed) or with a bare `>` (a 1-step difference of rounding decides): compare in ONE dtype
    # and demand TIE_ULPS steps.
    reached32, baseline32 = np.float32(reached), np.float32(baseline)
    return bool(reached32 > baseline32 + np.float32(TIE_ULPS) * np.spacing(baseline32))

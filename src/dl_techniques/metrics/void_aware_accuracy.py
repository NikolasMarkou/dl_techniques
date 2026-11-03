"""Pixel accuracy that leaves void pixels out.

The stock :class:`keras.metrics.SparseCategoricalAccuracy` has no ``ignore_class``: it scores every
pixel, so a void pixel (a label such as 255 that marks "not a class") can never be predicted
and counts as wrong (measured: 0.5 where 2 of 4 pixels are void and the other two are right).
:class:`keras.metrics.MeanIoU` and :class:`keras.losses.SparseCategoricalCrossentropy` both take
``ignore_class``; this is the accuracy that does the same.

Interface contract
------------------
:class:`VoidAwareAccuracy`
    A streaming :class:`keras.metrics.Metric`: the fraction of pixels whose label is not
    ``ignore_index`` that the argmax of ``y_pred`` gets right, pooled over every batch since the
    last reset (a pixel count, not a mean of per-batch means).

    :param ignore_index: the label value of void pixels (a whole number outside the classes).
    :param name: metric name; ``"accuracy"`` by default, so a run that sets an ignore index and a
        run that sets none log the same key.
    :param dtype: state dtype, as every Keras metric.

    ``update_state(y_true, y_pred, sample_weight=None)``: ``y_true`` is integer class ids
    ``(B, H, W)`` (a trailing axis of 1 is dropped) and ``y_pred`` the logits or probabilities
    ``(B, H, W, C)``, any float dtype. ``sample_weight`` is ``(B,)`` or the shape of ``y_true`` and
    multiplies the pixel's count; a void pixel has weight 0 whatever it is.

    ``result()`` is ``0`` for a metric that has seen no valid pixel (a fully void batch), never
    ``NaN``. Everything is ``keras.ops``, so it runs under ``jit_compile=True``.

    Registered with ``@register_dl_technique``, so a ``.keras`` model compiled with it reloads
    wherever ``dl_techniques`` is importable (the archive names this module). Use it ONLY when
    the run has an ignore index: with none, the stock metric is the same number and needs no
    registration.

    Failure mode: a ``y_true`` whose rank is neither that of ``y_pred`` minus 1 nor equal to it
    with a last axis of 1 fails in the comparison with Keras' shape error.
"""

from typing import Any, Dict, Optional

import keras

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.metrics.void_aware_accuracy")
class VoidAwareAccuracy(keras.metrics.Metric):
    """Pixel accuracy over the pixels that are not void; see the module contract."""

    def __init__(self, ignore_index: int, name: str = "accuracy", dtype: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.ignore_index = int(ignore_index)
        self.correct = self.add_weight(name="correct", shape=(), initializer="zeros")
        self.valid = self.add_weight(name="valid", shape=(), initializer="zeros")

    def update_state(self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None) -> None:
        """Add one batch: its right pixels and its non-void pixels (weighted)."""
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true)
        if len(y_true.shape) == len(y_pred.shape) and y_true.shape[-1] == 1:
            y_true = keras.ops.squeeze(y_true, axis=-1)
        labels = keras.ops.cast(y_true, "int32")
        predicted = keras.ops.cast(keras.ops.argmax(y_pred, axis=-1), "int32")
        # DECISION plan-2026-10-31T180033-2faf890a/D-076
        # The void test is on the LABEL and the weight of a void pixel is multiplied in, never
        # taken from the prediction: a void pixel's logits are arbitrary (the model has no class
        # to give it), so anything that reads them would make the number depend on them. Do NOT
        # score `predicted == labels` over every pixel and subtract void afterwards, and do NOT
        # drop the `valid` factor from `hits` (a void label that happens to equal the argmax
        # would count as right).
        valid = keras.ops.cast(keras.ops.not_equal(labels, self.ignore_index), self.dtype)
        hits = keras.ops.cast(keras.ops.equal(predicted, labels), self.dtype) * valid
        if sample_weight is not None:
            weight = keras.ops.cast(sample_weight, self.dtype)
            while len(weight.shape) < len(labels.shape):
                weight = keras.ops.expand_dims(weight, axis=-1)
            valid = valid * weight
            hits = hits * weight
        self.correct.assign_add(keras.ops.sum(hits))
        self.valid.assign_add(keras.ops.sum(valid))

    def result(self) -> Any:
        """Right pixels over non-void pixels, ``0`` before any non-void pixel."""
        return keras.ops.divide_no_nan(self.correct, self.valid)

    def reset_state(self) -> None:
        """Forget every batch."""
        self.correct.assign(0.0)
        self.valid.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["ignore_index"] = self.ignore_index
        return config

# ---------------------------------------------------------------------

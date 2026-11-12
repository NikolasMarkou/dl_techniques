"""The loss and the constant baselines of the ``pretrain`` task.

The success bar of a pretrain run is cut from the ``val_loss`` of a CONSTANT Keras model
evaluated under the run's own compiled loss on the fixed select stream, so the bar and
``val_loss`` share one reduction (a closed-form numpy baseline would be a second copy of the
loss). This module imports TensorFlow and Keras; ``train.api`` imports it inside functions only.

Interface contract
------------------
``pretrain_loss(method)``
    :return: a new stock Keras loss: ``MeanSquaredError`` for ``et_masked`` (the mask is a
        ``sample_weight`` of the stream, ``N / n_loss`` on the loss set) and
        ``SparseCategoricalCrossentropy(from_logits=True)`` for ``beit_mim`` (``N / n_masked`` on
        the masked positions, so ``val_loss`` is the mean over examples of the per-example
        masked-token cross-entropy). :raises ValueError: for another method.

``ConstantPredictor(value, output_shape)``
    A non-saved ``keras.Model`` with no weights whose output is ``value`` broadcast to
    ``(batch, *output_shape)`` for every input; the batch is read off the first input.
    :raises ValueError: if ``value`` does not broadcast to ``output_shape``.

``constant_baseline(method, select_ds, *, vocab_size=None)``
    :param select_ds: one pass of the select stream of :class:`train.common.pretrain_data.PretrainBatches`.
    :param vocab_size: the BEiT codebook size (required for ``beit_mim``).
    :return: a :class:`Baseline`. ET: the zero-patch predictor (``kind="zero_patches"``); BEiT:
        constant logits of the in-sample log-probabilities of the masked positions' target ids
        (``kind="unigram_codes"``). ``val_loss`` is ``model.evaluate`` of that constant model.
    :raises ValueError: for a BEiT call with no ``vocab_size``, no masked position, or a target id
        outside ``[0, vocab_size)``.
    The constant BEiT logits broadcast to ``(batch, N, V)``: 32 * 196 * 8192 * 4 bytes, about
    205 MB per evaluation batch at imagenette with the 8192-code head.

``epoch_zero_val_loss(sidecar_path)``
    :param sidecar_path: ``<run>/checkpoints/fit_state.json``, the fit sidecar.
    :return: the ``val_loss`` of the UNTRAINED model on the select stream, which the fresh run's epoch-zero
        evaluation wrote there (``initial.metrics.loss``); a resume reads the same number.
    :raises ValueError: if the sidecar has no such value (a missing file, or a run that never evaluated epoch zero).
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from train.common.pretrain_data import BEIT_MIM, ET_MASKED, METHODS, masked_code_ids
from train.common.resume_state import read_fit_state

#: The floor of an in-sample probability before its log, so a code the select split never
#: masks cannot give ``-inf`` logits. A masked target always has a positive in-sample count, so
#: the floor changes no loss on the stream the logits were counted on.
_PROBABILITY_FLOOR = 1e-30


def pretrain_loss(method: str) -> "keras.losses.Loss":
    """A new stock loss for ``method``; see the module contract."""
    if method == ET_MASKED:
        return keras.losses.MeanSquaredError()
    if method == BEIT_MIM:
        return keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")


class ConstantPredictor(keras.Model):
    """A weightless model whose output is one constant, broadcast over the batch."""

    def __init__(self, value: Any, output_shape: Tuple[int, ...], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        value = np.asarray(value, dtype=np.float32)
        shape = tuple(int(n) for n in output_shape)
        if np.broadcast_shapes(value.shape, shape) != shape:
            raise ValueError(f"a constant of shape {value.shape} does not broadcast to {shape}")
        self._constant_value = value
        self._constant_shape = shape

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        first = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        batch = keras.ops.shape(first)[0]
        return keras.ops.broadcast_to(
            keras.ops.convert_to_tensor(self._constant_value), (batch,) + self._constant_shape)


@dataclass(frozen=True)
class Baseline:
    """The constant predictor's ``val_loss`` on the select stream and what predictor it was."""

    kind: str
    val_loss: float


def _evaluate(model: "keras.Model", select_ds: "tf.data.Dataset", method: str) -> float:
    """``model.evaluate`` of ``select_ds`` under the run's loss, the ``val_loss`` of the run."""
    model.compile(optimizer="sgd", loss=pretrain_loss(method))
    return float(model.evaluate(select_ds, verbose=0, return_dict=True)["loss"])


def constant_baseline(
        method: str, select_ds: "tf.data.Dataset", *, vocab_size: Optional[int] = None) -> Baseline:
    """The constant model's ``val_loss`` over one pass of ``select_ds``; see the module contract."""
    target_shape = tuple(select_ds.element_spec[1].shape)[1:]          # (N, P) or (N,)
    if method == ET_MASKED:
        model = ConstantPredictor(np.zeros(target_shape, np.float32), target_shape)
        return Baseline("zero_patches", _evaluate(model, select_ds, method))
    if method != BEIT_MIM:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    if vocab_size is None:
        raise ValueError("the BEiT baseline needs vocab_size, the tokenizer's codebook size")
    ids = masked_code_ids(select_ds)
    if ids.size == 0:
        raise ValueError("the select stream masks no position: there is no unigram to count")
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError(
            f"a target id ({int(ids.min())} to {int(ids.max())}) is outside [0, vocab_size="
            f"{vocab_size}): the tokenizer's codebook is larger than the vocab_size given")
    share = np.bincount(ids, minlength=vocab_size) / ids.size
    log_probs = np.log(np.maximum(share, _PROBABILITY_FLOOR)).astype(np.float32)
    model = ConstantPredictor(log_probs, (*target_shape, int(vocab_size)))
    return Baseline("unigram_codes", _evaluate(model, select_ds, method))


def epoch_zero_val_loss(sidecar_path: Any) -> float:
    """The untrained model's ``val_loss`` as the fit sidecar holds it; see the module contract."""
    # DECISION plan-2026-11-12T015747-46e5318f/D-009
    # The epoch-zero term of the bar is read from the SIDECAR, always, and never from an attribute the first
    # process set: a resume (and a reporting-only resume) never calls `initial_loss_check`, so the attribute is
    # empty there, and re-evaluating the weights a resume restored would measure the trained model, not the
    # untrained one. Do NOT cache it on the trainer "to save a read".
    state = read_fit_state(sidecar_path)
    initial = state.get("initial")
    metrics = initial.get("metrics") if isinstance(initial, dict) else None
    loss = metrics.get("loss") if isinstance(metrics, dict) else None
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise ValueError(
            f"the epoch-zero val_loss is not in {sidecar_path}: the fit sidecar holds no initial.metrics.loss, so the "
            "success bar cannot be cut (a run writes it before its first epoch)"
        )
    return float(loss)

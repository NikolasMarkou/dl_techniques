"""The contrastive stock-``fit`` bridge shared by the CLIP-family trainers.

``train.clip.train_clip`` (CLIP, MobileCLIP; the CliffordCLIP trainer shared it before it was
removed) trains a model whose output is a dict of two logit matrices through stock ``compile``/``fit``.
Keras hands each compiled loss and metric ONE leaf of a structured output, so
`StackedCLIPLogits` stacks the two matrices into one ``(N, 2, N)`` tensor and
the loss and metrics here unstack it (decisions.md D-036 of
plan-2026-10-14T062616-81938e81). The bridge lived in the CLIP trainer and the
CliffordCLIP trainer imported it from there; it moved here so neither trainer
imports the other (Step 12 Part F item 9, D-086).

Not re-exported from ``train.common``: importing it pulls in the CLIP losses
and metrics, which most trainers never use. Import it as
``from train.common.contrastive import ...``.
"""

from typing import Any, Dict, Optional

import keras
import tensorflow as tf

from dl_techniques.losses.clip_contrastive_loss import CLIPContrastiveLoss
from dl_techniques.losses.siglip_contrastive_loss import SigLIPContrastiveLoss
from dl_techniques.metrics.clip_accuracy import CLIPAccuracy, CLIPRecallAtK
from dl_techniques.utils.clip_utils import compute_clip_logits


# ---------------------------------------------------------------------
# The bridge: a CLIP-family output dict to stock compile/fit
# ---------------------------------------------------------------------
#
# DECISION plan-2026-10-14T062616-81938e81/D-036: train through stock
# `compile`/`fit` on a stacking wrapper. Do NOT bring back a GradientTape loop
# or a `train_step` override (the old `CLIPTrainer`; the step-override census
# in test_package_api_contract.py no longer lists this file). Do NOT compile
# the CLIP itself: Keras' `CompileLoss` and `CompileMetrics` hand ONE leaf of
# a structured output to each loss or metric, and the contrastive loss needs
# BOTH logit matrices. MEASURED on a tiny CLIP (Keras 3.8): one loss object
# over the dict raises; a per-key loss dict on `logits_per_image` trains, but
# a per-key metric dict is dropped without a word
# (`CompileMetrics._flat_metrics == [None]`, no metric column). Do NOT save
# the wrapper: `main` saves the CLIP it wraps. `StackedCLIPLogits` stacks the
# two matrices into one `(N, 2, N)` tensor; the loss and the metrics unstack
# it and call the unmodified `CLIPContrastiveLoss` and CLIP metrics. See
# decisions.md D-036.


def unstack_clip_logits(stacked: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
    """Split the ``(N, 2, N)`` output of `StackedCLIPLogits`.

    Interface contract (callers: `StackedCLIPContrastiveLoss`,
    `StackedCLIPAccuracy`, `StackedCLIPRecallAtK`, tests):

    Args:
        stacked: ``(N, 2, N)``; index 0 of axis 1 is ``logits_per_image``,
            index 1 is ``logits_per_text``.

    Returns:
        ``{'logits_per_image': (N, N), 'logits_per_text': (N, N)}``, the dict
        `CLIPContrastiveLoss` and the CLIP metrics read.
    """
    return {
        'logits_per_image': stacked[:, 0],
        'logits_per_text': stacked[:, 1],
    }


# NOT @register_dl_technique: never saved. The run saves the CLIP it wraps
# (`final_model.keras`), the reusable artefact; the wrapper is training
# scaffold (the SuperPoint/LightGlue trainers record the same ruling).
class StackedCLIPLogits(keras.Model):
    """Training view of a CLIP-family model: one ``(N, 2, N)`` logits tensor.

    Interface contract (callers: `compile_for_contrastive_training`, the
    MobileCLIP ``fit`` proofs): ``call`` returns
    ``stack([logits_per_image, logits_per_text], axis=1)``. A model whose
    output has no logits (MobileCLIP v1 emits ``image_features``,
    ``text_features`` and ``logit_scale``) gets them from
    `compute_clip_logits`, the helper CLIP and MobileCLIP v2 call in their own
    ``call``. Every weight is the wrapped model's, so a step trains exactly
    the wrapped model, ``logit_scale`` included.

    Args:
        clip_model: The model to train; called with ``{'image', 'text'}``.
    """

    def __init__(self, clip_model: keras.Model, **kwargs: Any) -> None:
        # An explicit name: the default is derived from the class name, and a
        # leading-underscore class name gives a scope name TensorFlow rejects
        # when `fit` traces the step.
        kwargs.setdefault('name', 'stacked_clip_logits')
        super().__init__(**kwargs)
        self.clip_model = clip_model

    def call(self, inputs: Dict[str, keras.KerasTensor],
             training: Optional[bool] = None) -> keras.KerasTensor:
        outputs = self.clip_model(inputs, training=training)
        if 'logits_per_image' not in outputs:
            outputs['logits_per_image'], outputs['logits_per_text'] = (
                compute_clip_logits(
                    outputs['image_features'], outputs['text_features'],
                    outputs['logit_scale'],
                )
            )
        return keras.ops.stack(
            [outputs['logits_per_image'], outputs['logits_per_text']], axis=1)


class StackedCLIPContrastiveLoss(keras.losses.Loss):
    """A contrastive loss on the stacked logits of `StackedCLIPLogits`.

    Interface contract (callers: `compile_for_contrastive_training`, the
    MobileCLIP ``fit`` proofs, tests): ``inner`` is any `keras.losses.Loss`
    whose ``call(y_true, y_pred)`` accepts the
    ``{'logits_per_image', 'logits_per_text'}`` dict and returns one value per
    sample (`CLIPContrastiveLoss`, `SigLIPContrastiveLoss`); the default is
    ``CLIPContrastiveLoss()``. ``y_true`` is ignored: the targets of a
    contrastive batch are its own diagonal. ``call`` returns the per-sample
    vector of the inner loss's ``call`` (not its ``__call__``, which converts a
    ``None`` label and raises, decisions.md D-030 of
    plan-2026-08-19T163559-499b6f0e), so the reduction is applied once, by
    this loss's own ``__call__``. An inner loss must not apply a temperature
    of its own (``CLIPContrastiveLoss`` keeps ``apply_temperature=False``,
    ``SigLIPContrastiveLoss`` ``temperature=1.0``): the model already
    multiplies by its learnable ``exp(logit_scale)``, and a second temperature
    would compound it.
    """

    def __init__(self, inner: Optional[keras.losses.Loss] = None,
                 name: str = 'clip_contrastive_loss', **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.inner = CLIPContrastiveLoss() if inner is None else inner

    def call(self, y_true: Any, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        return self.inner.call(None, unstack_clip_logits(y_pred))


class StackedCLIPAccuracy(CLIPAccuracy):
    """`CLIPAccuracy` on the stacked logits; ``y_true`` is ignored."""

    def update_state(self, y_true: Any = None, y_pred: Any = None,
                     sample_weight: Any = None) -> None:
        super().update_state(y_pred=unstack_clip_logits(y_pred))


class StackedCLIPRecallAtK(CLIPRecallAtK):
    """`CLIPRecallAtK` on the stacked logits; ``y_true`` is ignored."""

    def update_state(self, y_true: Any = None, y_pred: Any = None,
                     sample_weight: Any = None) -> None:
        super().update_state(y_pred=unstack_clip_logits(y_pred))


def compile_for_contrastive_training(
    clip_model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    inner_loss: Optional[keras.losses.Loss] = None,
) -> StackedCLIPLogits:
    """Wrap ``clip_model`` and compile it for stock ``fit``.

    Interface contract (callers: ``train.clip.train_clip.main``, tests): returns a compiled
    `StackedCLIPLogits` with loss ``StackedCLIPContrastiveLoss(inner_loss)``
    (``inner_loss=None`` is ``CLIPContrastiveLoss()``) and metrics
    ``clip_i2t_accuracy`` and ``clip_i2t_recall@5``. Feed its ``fit`` the
    batches of `as_contrastive_fit_batches`. ``clip_model`` itself is not
    compiled; save it, not the wrapper.
    """
    wrapper = StackedCLIPLogits(clip_model)
    wrapper.compile(
        optimizer=optimizer,
        loss=StackedCLIPContrastiveLoss(inner_loss),
        metrics=[StackedCLIPAccuracy(), StackedCLIPRecallAtK(k=5)],
    )
    return wrapper


def as_contrastive_fit_batches(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Contrastive batches to ``({'image', 'text'}, zeros(N))``.

    Interface contract (callers: both CLIP trainers, tests): an
    element is either an ``(images, tokens)`` pair or an
    ``{'image', 'text'}`` dict (the batches of
    ``train.common.make_image_text_tf_dataset``). The zeros are the ``y``
    that ``fit`` needs to call a loss at all (a ``None`` label raises inside
    `keras.losses.Loss.__call__`); the loss and the metrics ignore it.
    """
    def to_fit_inputs(*element):
        if len(element) == 1:
            inputs = element[0]
        else:
            images, tokens = element
            inputs = {'image': images, 'text': tokens}
        dummy = tf.zeros(tf.shape(inputs['text'])[:1], dtype=tf.float32)
        return inputs, dummy

    return dataset.map(to_fit_inputs)


def build_contrastive_loss(loss: str, label_smoothing: float) -> keras.losses.Loss:
    """The inner contrastive loss for ``--loss``.

    Interface contract (callers: tests; the CliffordCLIP trainer that used it was removed):
    returns a `keras.losses.Loss` whose ``call(None,
    {'logits_per_image', 'logits_per_text'})`` is the per-sample loss, the
    shape `StackedCLIPContrastiveLoss` requires of its ``inner``. Neither
    loss applies a temperature of its own.

    Args:
        loss: ``"clip"`` or ``"siglip"`` (the ``--loss`` choices).
        label_smoothing: CLIP label smoothing; must be ``0.0`` for
            ``"siglip"``, which has no smoothing (the parser enforces it).

    Returns:
        The loss.

    Raises:
        ValueError: For an unknown ``loss``, or label smoothing with
            ``"siglip"``.
    """
    # DECISION plan-2026-07-15T114613-5add9baa/D-002: --loss selects CLIP vs
    # SigLIP (both consume the same {logits_per_image, logits_per_text} dict
    # schema); default "clip" is byte-identical to the prior hardcoded path
    # (O1). SigLIP is built with temperature=1.0 / use_learnable_temperature=
    # False on purpose -- CliffordCLIP already applies exp(logit_scale) in its
    # forward pass, so do NOT enable a second temperature here (it would
    # compound the scaling, exactly as apply_temperature=False guards on the
    # CLIP path). See decisions.md D-002.
    if loss == "siglip":
        if label_smoothing:
            raise ValueError(
                f"--loss siglip has no label smoothing; got "
                f"label_smoothing={label_smoothing}"
            )
        return SigLIPContrastiveLoss(
            temperature=1.0,
            use_learnable_temperature=False,
            name="siglip_contrastive_loss",
        )
    if loss == "clip":
        return CLIPContrastiveLoss(
            apply_temperature=False,
            label_smoothing=label_smoothing,
            name="clip_contrastive_loss",
        )
    raise ValueError(f"unknown contrastive loss {loss!r}; expected 'clip' or 'siglip'")

"""Every ``--loss`` choice the CLI advertises must survive its own first step.

``ContrastiveCliffordCLIP._contrastive_loss`` once called
``self.loss_fn.reduced_loss(...)``, a method defined ONLY on
``CLIPContrastiveLoss``. ``--loss siglip`` is an advertised choice and builds a
``SigLIPContrastiveLoss``, which has no such method, so that path died on the
first batch with::

    AttributeError: 'SigLIPContrastiveLoss' object has no attribute 'reduced_loss'

(plan-2026-08-30T203107-30455f66 D-001). The wrapper and its hand-written
``train_step`` are gone since plan-2026-10-14T062616-81938e81 D-045: the trainer
builds the inner loss with ``build_contrastive_loss`` and trains through the
stock-``fit`` bridge of ``train.common.contrastive``
(``compile_for_contrastive_training``), whose ``StackedCLIPContrastiveLoss``
calls the inner loss's ``call``, the polymorphic reduction D-001 introduced.

This module therefore:

*   builds the loss through the REAL CLI path
    (``parse_arguments([...])`` -> ``build_contrastive_loss``),
*   runs one stock ``fit`` step through the REAL bridge for EVERY choice the
    parser advertises, so a future ``--loss`` entry is covered the day it is
    added,
*   pins the CLIP path's scalar on fixed logits at the values measured on the
    pre-D-001 tree, which is what makes both moves (D-001, then the bridge)
    provably value-neutral for the default choice, and
*   pins the per-loss ``--label-smoothing`` resolution (SigLIP has none).
"""

import numpy as np
import pytest

import keras
import tensorflow as tf

from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP
from train.cliffordnet.train_clip import (
    _build_arg_parser,
    parse_arguments,
)
from train.common.contrastive import (
    StackedCLIPContrastiveLoss,
    StackedCLIPLogits,
    build_contrastive_loss,
    compile_for_contrastive_training,
)

IMAGE_SIZE = 32
CONTEXT = 8
VOCAB = 64
BATCH = 4


def _advertised_loss_choices():
    """The ``--loss`` choices the CLI itself advertises, read from the parser.

    Derived, never hard-coded: no advertised choice may depend on a method
    only one loss class happens to own, so the guard's population must grow
    automatically with the parser's.
    """
    parser = _build_arg_parser()
    for action in parser._actions:  # noqa: SLF001 - argparse has no public API
        if "--loss" in action.option_strings:
            return list(action.choices)
    raise AssertionError("train_clip.py no longer advertises a --loss flag")


def _tiny_clip() -> CliffordCLIP:
    """Smallest CliffordCLIP that still produces both logit matrices."""
    return CliffordCLIP(
        image_size=IMAGE_SIZE,
        vision_patch_size=4,
        vision_stage_channels=[8, 8],
        vision_stage_depths=[1, 1],
        vision_stochastic_depth_rate=0.0,
        vocab_size=VOCAB,
        context_length=CONTEXT,
        text_channels=16,
        text_depth=2,
        text_stochastic_depth_rate=0.0,
        embed_dim=16,
        dropout_rate=0.0,
    )


def _fit_batches() -> tf.data.Dataset:
    rng = np.random.default_rng(0)
    inputs = {
        "image": rng.standard_normal(
            (BATCH, IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32"),
        "text": rng.integers(0, VOCAB, size=(BATCH, CONTEXT)).astype("int32"),
    }
    return tf.data.Dataset.from_tensors(
        (inputs, np.zeros((BATCH,), dtype="float32")))


def _fixed_logits():
    """Deterministic square logit pair, independent of any model init.

    The sum is asserted by the caller so a numpy-generator change cannot
    silently rebase the pinned CLIP values below.
    """
    rng = np.random.default_rng(1234)
    logits_per_image = rng.standard_normal((5, 5)).astype("float32")
    return logits_per_image, logits_per_image.T.copy()


def _stacked(logits_per_image, logits_per_text):
    """The ``(N, 2, N)`` layout `StackedCLIPLogits` emits."""
    return keras.ops.stack(
        [keras.ops.convert_to_tensor(logits_per_image),
         keras.ops.convert_to_tensor(logits_per_text)], axis=1)


def _bridge_loss_value(inner, logits_per_image, logits_per_text) -> np.ndarray:
    n = logits_per_image.shape[0]
    return np.array(StackedCLIPContrastiveLoss(inner)(
        np.zeros((n,), dtype="float32"),
        _stacked(logits_per_image, logits_per_text)))


@pytest.mark.parametrize("loss_choice", _advertised_loss_choices())
def test_every_advertised_loss_choice_completes_one_fit_step(loss_choice):
    """CLI -> inner loss -> bridge -> one stock ``fit`` step, for any choice."""
    args = parse_arguments(["--loss", loss_choice])
    assert args.loss == loss_choice

    keras.utils.set_random_seed(0)
    wrapper = compile_for_contrastive_training(
        _tiny_clip(),
        keras.optimizers.SGD(learning_rate=1e-3),
        inner_loss=build_contrastive_loss(args.loss, args.label_smoothing),
    )
    assert isinstance(wrapper, StackedCLIPLogits)
    history = wrapper.fit(_fit_batches(), epochs=1, verbose=0)

    losses = history.history["loss"]
    assert len(losses) == 1, losses
    assert np.isfinite(losses[0]), f"--loss {loss_choice}: loss is {losses[0]!r}"


@pytest.mark.parametrize("loss_choice", _advertised_loss_choices())
def test_the_bridge_reduces_every_advertised_choice_to_a_finite_scalar(
    loss_choice,
):
    """The bridge loss itself, on fixed logits, for every choice.

    Isolates the reduction from model initialisation: the D-001 defect was in
    the reduction call, not in the forward pass.
    """
    args = parse_arguments(["--loss", loss_choice])
    value = _bridge_loss_value(
        build_contrastive_loss(args.loss, args.label_smoothing),
        *_fixed_logits(),
    )
    assert value.ndim == 0, f"--loss {loss_choice}: got shape {value.shape}"
    assert np.isfinite(float(value)), f"--loss {loss_choice}: {value!r}"


@pytest.mark.parametrize(
    "label_smoothing,expected",
    [(0.0, 2.8179826736450195), (0.1, 2.7563607692718506)],
)
def test_the_clip_path_scalar_is_unchanged_by_the_bridge(
    label_smoothing, expected
):
    """The default ``--loss clip`` value must not move when the call site does.

    MEASURED on the pre-D-001 tree (``self.loss_fn.reduced_loss(...)``) with the
    exact fixture below and hard-coded here, so this comparison survives
    without a worktree: ``atol=1e-6, rtol=0``. The same pins guarded the D-001
    polymorphic call site; they now guard the stock-``fit`` bridge.
    """
    logits_per_image, logits_per_text = _fixed_logits()
    assert float(logits_per_image.sum()) == pytest.approx(
        1.7727731466293335, abs=1e-6, rel=0
    ), "the fixed-logit fixture drifted; the pinned CLIP values below are void"

    value = float(_bridge_loss_value(
        build_contrastive_loss("clip", label_smoothing),
        logits_per_image, logits_per_text,
    ))
    assert value == pytest.approx(expected, abs=1e-6, rel=0), (
        f"CLIP path moved: {value!r} vs pre-fix {expected!r} "
        f"(label_smoothing={label_smoothing})"
    )


def test_the_label_smoothing_default_belongs_to_the_clip_loss():
    """``--label-smoothing`` defaults to 0.1 for CLIP and is refused for SigLIP.

    SigLIPContrastiveLoss has no smoothing, so the old shared default of 0.1
    was silently ignored on ``--loss siglip``.
    """
    assert parse_arguments([]).label_smoothing == 0.1
    assert parse_arguments(["--label-smoothing", "0.0"]).label_smoothing == 0.0
    assert parse_arguments(["--loss", "siglip"]).label_smoothing == 0.0
    with pytest.raises(SystemExit):
        parse_arguments(["--loss", "siglip", "--label-smoothing", "0.1"])
    with pytest.raises(ValueError, match="no label smoothing"):
        build_contrastive_loss("siglip", 0.1)
    with pytest.raises(ValueError, match="unknown contrastive loss"):
        build_contrastive_loss("infonce", 0.0)


def test_no_inner_loss_adds_a_second_temperature():
    """CliffordCLIP multiplies by its learnable ``exp(logit_scale)`` itself."""
    clip_loss = build_contrastive_loss("clip", 0.1)
    assert clip_loss.apply_temperature is False
    assert clip_loss.label_smoothing == 0.1
    siglip_loss = build_contrastive_loss("siglip", 0.0)
    assert siglip_loss.temperature == 1.0
    assert siglip_loss.use_learnable_temperature is False

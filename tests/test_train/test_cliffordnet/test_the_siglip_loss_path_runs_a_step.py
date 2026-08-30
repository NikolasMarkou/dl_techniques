"""Every ``--loss`` choice the CLI advertises must survive its own first step.

``ContrastiveCliffordCLIP._contrastive_loss`` used to call
``self.loss_fn.reduced_loss(...)`` -- a method defined ONLY on
``CLIPContrastiveLoss`` (``clip_contrastive_loss.py:518``). ``--loss siglip``
is an advertised choice (``train_clip.py`` ``--loss`` / ``choices=[...]``) and
builds a ``SigLIPContrastiveLoss``, which has no such method, so that path died
on the first batch -- train OR eval -- with the reproduced::

    AttributeError: 'SigLIPContrastiveLoss' object has no attribute 'reduced_loss'

Three green tests in ``tests/test_train/test_clip/`` were blind to it because
none of them ever CONSTRUCTED the siglip branch: they exercised
``CLIPContrastiveLoss.reduced_loss`` directly. So a ``hasattr`` assertion would
be the wrong guard here -- the defect lives in the construction-plus-call-site
PAIR, and only driving both reproduces it.

This module therefore:

*   builds the loss through the REAL CLI path
    (``_build_arg_parser().parse_args([...])`` -> ``ContrastiveCliffordCLIP``),
*   drives it through the REAL call site (``_contrastive_loss``),
*   runs a real ``train_step`` for EVERY choice the parser advertises, so a
    future ``--loss`` entry is covered the day it is added, and
*   pins the CLIP path's scalar on fixed logits, which is what makes the
    polymorphic-reduction fix provably value-neutral for the default choice.

See ``plans/plan-2026-08-30T203107-30455f66/decisions.md`` D-001.
"""

import numpy as np
import pytest

import keras
import tensorflow as tf

from dl_techniques.models.vision_language.clip.clifford_clip import CliffordCLIP
from train.cliffordnet.train_clip import (
    ContrastiveCliffordCLIP,
    _build_arg_parser,
)

IMAGE_SIZE = 32
CONTEXT = 8
VOCAB = 64
BATCH = 4


def _advertised_loss_choices():
    """The ``--loss`` choices the CLI itself advertises, read from the parser.

    Derived, never hard-coded: the whole point of the fix is that no advertised
    choice may depend on a method only one loss class happens to own, so the
    guard's population must grow automatically with the parser's.
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


def _batch():
    rng = np.random.default_rng(0)
    return {
        "image": tf.constant(
            rng.standard_normal((BATCH, IMAGE_SIZE, IMAGE_SIZE, 3)),
            dtype="float32",
        ),
        "text": tf.constant(
            rng.integers(0, VOCAB, size=(BATCH, CONTEXT)), dtype="int32"
        ),
    }


def _fixed_logits():
    """Deterministic square logit pair, independent of any model init.

    The sum is asserted by the caller so a numpy-generator change cannot
    silently rebase the pinned CLIP values below.
    """
    rng = np.random.default_rng(1234)
    logits_per_image = rng.standard_normal((5, 5)).astype("float32")
    return logits_per_image, logits_per_image.T.copy()


@pytest.mark.parametrize("loss_choice", _advertised_loss_choices())
def test_every_advertised_loss_choice_completes_one_training_step(loss_choice):
    """The CLI path -> wrapper -> ``train_step`` must not raise, for any choice.

    RED before the fix for ``loss_choice == "siglip"`` with
    ``AttributeError: 'SigLIPContrastiveLoss' object has no attribute
    'reduced_loss'``; green for ``"clip"`` both before and after.
    """
    args = _build_arg_parser().parse_args(["--loss", loss_choice])
    assert args.loss == loss_choice

    clip_model = _tiny_clip()
    input_shape = {
        "image": (None, IMAGE_SIZE, IMAGE_SIZE, 3),
        "text": (None, CONTEXT),
    }
    clip_model.build(input_shape)

    wrapper = ContrastiveCliffordCLIP(
        clip_model=clip_model,
        label_smoothing=args.label_smoothing,
        loss=args.loss,
    )
    wrapper.build(input_shape)
    wrapper.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3))

    logs = wrapper.train_step(_batch())

    value = float(np.array(logs["loss"]))
    assert np.ndim(np.array(logs["loss"])) == 0, (
        f"--loss {loss_choice}: train_step returned a non-scalar loss "
        f"{np.array(logs['loss']).shape}"
    )
    assert np.isfinite(value), f"--loss {loss_choice}: loss is {value!r}"


@pytest.mark.parametrize("loss_choice", _advertised_loss_choices())
def test_the_call_site_reduces_every_advertised_choice_to_a_finite_scalar(
    loss_choice,
):
    """``_contrastive_loss`` itself, on fixed logits, for every choice.

    Isolates the call site from model initialisation: the defect was in the
    reduction call, not in the forward pass.
    """
    logits_per_image, logits_per_text = _fixed_logits()
    wrapper = ContrastiveCliffordCLIP(clip_model=None, loss=loss_choice)

    value = np.array(
        wrapper._contrastive_loss(  # noqa: SLF001 - the site under test
            {
                "logits_per_image": keras.ops.convert_to_tensor(
                    logits_per_image
                ),
                "logits_per_text": keras.ops.convert_to_tensor(logits_per_text),
            }
        )
    )

    assert value.ndim == 0, f"--loss {loss_choice}: got shape {value.shape}"
    assert np.isfinite(float(value)), f"--loss {loss_choice}: {value!r}"


@pytest.mark.parametrize(
    "label_smoothing,expected",
    [(0.0, 2.8179826736450195), (0.1, 2.7563607692718506)],
)
def test_the_clip_path_scalar_is_unchanged_by_the_polymorphic_call_site(
    label_smoothing, expected
):
    """The default ``--loss clip`` value must not move when the call site does.

    MEASURED on the pre-fix tree (``self.loss_fn.reduced_loss(...)``) with the
    exact fixture below and hard-coded here, so this comparison survives
    without a worktree: ``atol=1e-6, rtol=0``.
    """
    logits_per_image, logits_per_text = _fixed_logits()
    assert float(logits_per_image.sum()) == pytest.approx(
        1.7727731466293335, abs=1e-6, rel=0
    ), "the fixed-logit fixture drifted; the pinned CLIP values below are void"

    wrapper = ContrastiveCliffordCLIP(
        clip_model=None, label_smoothing=label_smoothing, loss="clip"
    )
    value = float(
        np.array(
            wrapper._contrastive_loss(  # noqa: SLF001 - the site under test
                {
                    "logits_per_image": keras.ops.convert_to_tensor(
                        logits_per_image
                    ),
                    "logits_per_text": keras.ops.convert_to_tensor(
                        logits_per_text
                    ),
                }
            )
        )
    )

    assert value == pytest.approx(expected, abs=1e-6, rel=0), (
        f"CLIP path moved: {value!r} vs pre-fix {expected!r} "
        f"(label_smoothing={label_smoothing})"
    )


def test_the_clip_loss_class_still_owns_reduced_loss():
    """``CLIPContrastiveLoss.reduced_loss`` has other callers; it stays.

    ``src/train/clip/train_clip.py:300`` and ``:345`` call it, and
    ``tests/test_train/test_clip/test_the_trainer_steps_do_not_raise.py:64``
    asserts on it. The fix makes the CLIFFORD call site stop REQUIRING it; it
    does not delete it.
    """
    from dl_techniques.losses.clip_contrastive_loss import CLIPContrastiveLoss

    assert hasattr(CLIPContrastiveLoss, "reduced_loss")

"""SC-10: the stage-1 -> stage-2 warm start transfers the trunk, or ABORTS the run.

**Why this file exists at all.** ``load_weights_from_checkpoint`` looks like it guards the
zero-layer case itself -- it ends with a ``ValueError("No overlapping layers ...")``. It
does not, for this caller. Its condition is::

    not report.loaded and not report.skipped_by_prefix and not report.shape_mismatch

and BEiT passes ``skip_prefixes=("decoder_", "head_")``, so the source MIM checkpoint's
``decoder_norm`` / ``decoder_head`` layers ALWAYS populate ``skipped_by_prefix``. The
guard is therefore permanently disarmed here, and a checkpoint whose backbone is absent
or renamed transfers ZERO trunk weights, returns a perfectly ordinary report, and trains
from random init while the command line says "pretrained". This was MEASURED at step 6 of
this plan, and ``test_the_library_helper_alone_does_NOT_raise`` below re-measures it, so
the claim cannot rot silently: if the library ever grows a real guard, that test goes RED
and this file's premise is re-examined rather than quietly assumed.

The three cases:

* **(a) absent / renamed backbone** -> zero-layer trunk transfer -> MUST raise. This is
  the case the library does not catch, so it proves OUR guard.
* **(b) config-mismatched checkpoint** -> ``shape_mismatch`` on the trunk -> MUST raise.
  ``strict=False`` records a mismatch and moves on, leaving the trunk at init.
* **happy path** -> trunk weight VALUES equal post-transfer (not merely "some layer
  loaded", not merely matching shapes or counts).
"""

from pathlib import Path
from typing import Any, Dict

import keras
import numpy as np
import pytest

from dl_techniques.models.beit import (
    BACKBONE_NAME,
    BeitForMaskedImageModeling,
    BeitModel,
    create_beit_classifier,
    create_beit_mim,
)
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from train.beit.train_classification import (
    WARM_START_SKIP_PREFIXES,
    warm_start_encoder,
)

# ---------------------------------------------------------------------------
# A deliberately tiny backbone. The warm start is a NAME + SHAPE contract, so its
# correctness does not depend on width or depth -- and a 12-layer `tiny` would make this
# file slow for nothing.
# ---------------------------------------------------------------------------

IMAGE_SIZE = 32
PATCH_SIZE = 16
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 4
VOCAB_SIZE = 32
NUM_CLASSES = 5

SMALL_BACKBONE: Dict[str, Any] = {
    "hidden_size": 32,
    "num_layers": 2,
    "num_heads": 2,
    "intermediate_size": 64,
    # Determinism for the value-equality assertion: `training=None` is NOT inference for
    # StochasticDepth, and a non-zero ramp would make a forward comparison stochastic.
    "drop_path_rate": 0.0,
}


def _build_mim(**backbone_overrides: Any) -> BeitForMaskedImageModeling:
    model = create_beit_mim(
        variant="tiny",
        input_shape=INPUT_SHAPE,
        patch_size=PATCH_SIZE,
        vocab_size=VOCAB_SIZE,
        **{**SMALL_BACKBONE, **backbone_overrides},
    )
    model.build([(None,) + INPUT_SHAPE, (None, NUM_PATCHES)])
    return model


def _build_classifier(**backbone_overrides: Any):
    model = create_beit_classifier(
        variant="tiny",
        input_shape=INPUT_SHAPE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        **{**SMALL_BACKBONE, **backbone_overrides},
    )
    # H-12: BUILT before any transfer -- load_weights_from_checkpoint needs weight-shaped
    # target layers for set_weights to land.
    model.build((None,) + INPUT_SHAPE)
    return model


def _save(model: keras.Model, tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    model.save(path)
    return str(path)


def _renamed_backbone_mim() -> BeitForMaskedImageModeling:
    """A structurally valid MIM model whose trunk carries a DIFFERENT layer name.

    This is what a renamed backbone, a hand-assembled checkpoint, or an unrelated
    architecture looks like to a name-matched transfer: everything loads cleanly except
    the one layer that matters.
    """
    backbone = BeitModel(
        input_shape=INPUT_SHAPE,
        patch_size=PATCH_SIZE,
        scale="tiny",
        name="some_other_backbone",
        **SMALL_BACKBONE,
    )
    model = BeitForMaskedImageModeling(backbone=backbone, vocab_size=VOCAB_SIZE)
    model.build([(None,) + INPUT_SHAPE, (None, NUM_PATCHES)])
    return model


# ---------------------------------------------------------------------------
# Happy path -- VALUES, not shapes or counts
# ---------------------------------------------------------------------------

class TestWarmStartHappyPath:

    def test_trunk_weight_values_are_equal_post_transfer(self, tmp_path) -> None:
        source = _build_mim()
        ckpt = _save(source, tmp_path, "mim.keras")
        source_trunk = source.get_layer(BACKBONE_NAME).get_weights()

        target = _build_classifier()
        target_trunk_before = target.get_layer(BACKBONE_NAME).get_weights()

        # The guard on the guard: if the two trunks already agreed at init, an equality
        # assertion after the transfer would pass with the transfer DELETED.
        assert len(source_trunk) == len(target_trunk_before) > 0
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(source_trunk, target_trunk_before)
        ), "source and target trunks are identical at init -- the assertion below is vacuous"

        report = warm_start_encoder(target, ckpt)

        assert BACKBONE_NAME in report.loaded
        after = target.get_layer(BACKBONE_NAME).get_weights()
        assert len(after) == len(source_trunk)
        for i, (got, want) in enumerate(zip(after, source_trunk)):
            np.testing.assert_array_equal(
                got, want, err_msg=f"trunk weight array {i} did not transfer")

    def test_the_mim_decoder_is_skipped_not_transferred(self, tmp_path) -> None:
        source = _build_mim()
        ckpt = _save(source, tmp_path, "mim.keras")
        target = _build_classifier()

        report = warm_start_encoder(target, ckpt)

        assert set(WARM_START_SKIP_PREFIXES) == {"decoder_", "head_"}
        assert any(n.startswith("decoder_") for n in report.skipped_by_prefix), (
            "the MIM decoder must be SKIPPED by prefix; an empty skip list would mean the "
            "prefixes no longer match the head's layer names"
        )
        assert not any(n.startswith("decoder_") for n in report.loaded)

    def test_the_classifier_still_predicts_after_the_transfer(self, tmp_path) -> None:
        """A transfer that lands wrong-shaped or half-applied weights can still satisfy a
        name-based report. Run the model."""
        source = _build_mim()
        ckpt = _save(source, tmp_path, "mim.keras")
        target = _build_classifier()
        warm_start_encoder(target, ckpt)

        logits = target(np.zeros((2,) + INPUT_SHAPE, dtype="float32"), training=False)
        assert tuple(logits.shape) == (2, NUM_CLASSES)
        assert np.all(np.isfinite(np.asarray(logits)))


# ---------------------------------------------------------------------------
# (a) Zero-layer transfer -- the case the library helper does NOT catch
# ---------------------------------------------------------------------------

class TestWarmStartRefusesAZeroLayerTransfer:

    def test_the_library_helper_alone_does_NOT_raise(self, tmp_path) -> None:
        """MEASUREMENT, re-run on every CI pass: this is the premise of our own guard.

        If this ever goes RED, ``load_weights_from_checkpoint`` grew a real no-overlap
        guard for a non-empty ``skip_prefixes`` and the wrapper's rationale must be
        revisited -- rather than the wrapper being quietly assumed redundant.
        """
        ckpt = _save(_renamed_backbone_mim(), tmp_path, "renamed.keras")
        target = _build_classifier()

        report = load_weights_from_checkpoint(
            target=target,
            ckpt_path=ckpt,
            skip_prefixes=WARM_START_SKIP_PREFIXES,
            strict=False,
        )
        assert BACKBONE_NAME not in report.loaded, (
            "the renamed-backbone fixture is not producing a zero-layer trunk transfer"
        )
        assert report.skipped_by_prefix, (
            "the no-overlap guard is disarmed BY skipped_by_prefix being non-empty; if it "
            "is empty here the measurement no longer demonstrates what it claims"
        )

    def test_warm_start_encoder_raises(self, tmp_path) -> None:
        ckpt = _save(_renamed_backbone_mim(), tmp_path, "renamed.keras")
        target = _build_classifier()

        with pytest.raises(RuntimeError, match=r"contains no layer named 'beit_backbone'"):
            warm_start_encoder(target, ckpt)

    def test_the_trunk_is_untouched_when_the_warm_start_refuses(self, tmp_path) -> None:
        """The raise must happen INSTEAD of a partial transfer, not after one."""
        ckpt = _save(_renamed_backbone_mim(), tmp_path, "renamed.keras")
        target = _build_classifier()
        before = [w.copy() for w in target.get_layer(BACKBONE_NAME).get_weights()]

        with pytest.raises(RuntimeError):
            warm_start_encoder(target, ckpt)

        after = target.get_layer(BACKBONE_NAME).get_weights()
        for i, (got, want) in enumerate(zip(after, before)):
            np.testing.assert_array_equal(
                got, want, err_msg=f"trunk weight array {i} moved despite the refusal")


# ---------------------------------------------------------------------------
# (b) Config drift -- shape mismatch on the trunk
# ---------------------------------------------------------------------------

class TestWarmStartRefusesAMismatchedCheckpoint:

    def test_a_wider_source_trunk_raises(self, tmp_path) -> None:
        source = _build_mim(hidden_size=64, intermediate_size=128)
        ckpt = _save(source, tmp_path, "mim_wide.keras")
        target = _build_classifier()  # hidden_size 32

        with pytest.raises(RuntimeError, match=r"trunk shapes do not match"):
            warm_start_encoder(target, ckpt)

    def test_a_deeper_source_trunk_raises(self, tmp_path) -> None:
        source = _build_mim(num_layers=4)
        ckpt = _save(source, tmp_path, "mim_deep.keras")
        target = _build_classifier()  # num_layers 2

        with pytest.raises(RuntimeError, match=r"trunk shapes do not match"):
            warm_start_encoder(target, ckpt)

    def test_strict_false_would_otherwise_leave_the_trunk_at_init(self, tmp_path) -> None:
        """Names the mechanism the raise defends against: with ``strict=False`` the
        library records the mismatch and returns normally, trunk untouched."""
        source = _build_mim(hidden_size=64, intermediate_size=128)
        ckpt = _save(source, tmp_path, "mim_wide.keras")
        target = _build_classifier()

        report = load_weights_from_checkpoint(
            target=target,
            ckpt_path=ckpt,
            skip_prefixes=WARM_START_SKIP_PREFIXES,
            strict=False,
        )
        mismatched = [name for name, _, _ in report.shape_mismatch]
        assert BACKBONE_NAME in mismatched
        assert BACKBONE_NAME not in report.loaded

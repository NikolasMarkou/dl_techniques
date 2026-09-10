r"""Guards for ``src/train/doc_res/infer_doc_res.py``.

What this file is defending, and why each arm can go red
--------------------------------------------------------
Inference post-processing is the part of a port where a defect keeps the
output's SHAPE, its DTYPE and its VALUE RANGE while changing the picture. Every
arm below is chosen for that property -- a wrong answer that a shape assertion
cannot see:

* **The padding side.** ``pad_to_multiple`` pads TOP and LEFT because
  ``crop_padding`` removes TOP and LEFT. Padding at the bottom/right instead
  returns a correctly-sized page shifted by up to seven pixels.
  :func:`test_an_identity_model_reproduces_the_page_exactly` closes this by
  round-tripping a real page through the real pipeline with a model that
  returns its own RGB input: any pad/crop disagreement moves content.
* **argmax vs a sigmoid threshold.** Both emit ``{0, 255}`` at the right shape.
  :class:`TestBinarizationIsArgmaxNotAThreshold` constructs the logit pair the
  two DISAGREE on and asserts which answer arrives, so swapping in a threshold
  is red rather than merely different.
* **The 1600 px branch.** A homomorphic re-composition and a plain upsample
  both produce a full-resolution uint8 page.
  :class:`TestTheHomomorphicBranchIsLive` proves the branch is selected at the
  boundary AND that taking it changes the answer -- a branch that produced the
  same thing as the fallback would be decoration.
* **x/y in the dewarping remap.** Swapping the two flow channels keeps the
  shape, the dtype and the range (D-021 says so in ``dtsprompt.py``).
  :class:`TestDewarpingDoesNotSwapXAndY` uses a page that is CONSTANT along one
  axis, so a shift along that axis is a no-op and a shift along the other is
  not; the swap therefore inverts the outcome.
* **Stage order in ``end2end``.** Asserted by INSTRUMENTING the real
  :func:`restore`, never by reading the source, and including the claim that
  each stage is fed the previous stage's OUTPUT rather than the original page.
* **Table-driven dispatch.** :class:`TestTheTableDrivesTheBehaviour` fabricates
  a ``TaskSpec`` with a different ``postprocess`` / a different
  ``n_supervised_channels`` and measures that the behaviour follows the table.
  Without it, "reads from ``TASKS``" is a claim about source text.

Nothing here trains, loads a checkpoint, allocates a GPU or writes outside
``tmp_path``. The two arms that need a network use a fake callable, because
what is under test is the arithmetic around the network, not the network.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from dl_techniques.datasets.document_restoration import tasks as tasks_module
from dl_techniques.datasets.document_restoration.dtsprompt import _resize_bilinear
from dl_techniques.datasets.document_restoration.tasks import (
    POSTPROCESS_ARGMAX_BINARY,
    POSTPROCESS_FLOW_REMAP,
    POSTPROCESS_MODES,
    TASKS,
    get_task,
    task_names,
)
from train.doc_res import infer_doc_res as inference
from train.doc_res.common import BINARY_INK_CLASS_INDEX, PIXEL_SCALE

#: Spatial extents whose pad-then-crop round trip must be exact. Includes the
#: brief's 33/100/255, every residue class mod 8 (so no remainder is untested),
#: and the two already-legal cases 8 and 256 where the padding must be ZERO --
#: a divisor bug that always padded would still round-trip, but would resize.
ROUND_TRIP_EXTENTS: Tuple[int, ...] = (
    1, 3, 8, 9, 15, 17, 33, 100, 254, 255, 256, 257,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


class _FakeModel:
    """A callable standing in for DocRes; records what it was handed.

    Interface contract: ``__call__(x, training=None) -> np.ndarray``. That is
    the whole surface :func:`inference.predict` uses, which is the point --
    every arm below tests the arithmetic AROUND the network, and a real DocRes
    would only add minutes and a GPU to arms that do not depend on its weights.
    """

    def __init__(self, respond) -> None:
        self.respond = respond
        self.inputs: List[np.ndarray] = []

    def __call__(self, x: Any, training: Any = None) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        self.inputs.append(array)
        return self.respond(array)


def _identity_model() -> _FakeModel:
    """Returns the RGB half of its own input, so the pipeline must be lossless."""
    return _FakeModel(lambda x: x[..., :3])


def _page(height: int, width: int, seed: int = 0) -> np.ndarray:
    """A random ``(h, w, 3)`` uint8 page."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _plan(
        original: np.ndarray,
        pad_h: int = 0,
        pad_w: int = 0,
        resized: bool = False,
        model_size: Tuple[int, int] = None,
) -> inference.InputPlan:
    """A hand-built :class:`InputPlan`.

    Used by the post-processing arms so they can fabricate the exact prediction
    that discriminates, instead of hoping a real network emits one.
    """
    height, width = (
        model_size
        if model_size is not None
        else (original.shape[0] + pad_h, original.shape[1] + pad_w)
    )
    return inference.InputPlan(
        array=np.zeros((1, height, width, 6), dtype=np.float32),
        prompt=np.zeros((height, width, 3), dtype=np.float32),
        original=original,
        pad_h=pad_h,
        pad_w=pad_w,
        resized=resized,
    )


def _spec_with(name: str, **overrides: Any):
    """A ``TaskSpec`` copy with fields replaced -- a fabricated table row."""
    return dataclasses.replace(get_task(name), **overrides)


# ---------------------------------------------------------------------
# --help allocates nothing
# ---------------------------------------------------------------------


class _Sentinel:
    """Records contact instead of doing the expensive thing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError(f"sentinel {self.name!r} was called")


#: Everything a real inference run does that costs something: claiming a GPU,
#: constructing or loading a 15 M-parameter model, walking the input directory,
#: reading and restoring a page.
_EXPENSIVE: Tuple[str, ...] = (
    "setup_gpu",
    "load_docres_model",
    "collect_input_images",
    "restore_image_file",
    "load_mask",
)


def _install_sentinels(monkeypatch) -> Dict[str, _Sentinel]:
    out: Dict[str, _Sentinel] = {}
    for attribute in _EXPENSIVE:
        sentinel = _Sentinel(attribute)
        # `setattr` raises on a misspelled name, so installing IS the check
        # that every sentinel covers something that exists.
        monkeypatch.setattr(inference, attribute, sentinel)
        out[attribute] = sentinel
    return out


def test_the_sentinels_cover_names_that_exist(monkeypatch):
    """A sentinel over a misspelled attribute would never fire."""
    sentinels = _install_sentinels(monkeypatch)
    assert len(sentinels) == len(_EXPENSIVE)
    assert all(s.calls == 0 for s in sentinels.values())


def test_help_prints_usage_and_allocates_nothing(monkeypatch, capsys):
    """``--help`` prints ``usage:`` and reaches no GPU, model or filesystem."""
    sentinels = _install_sentinels(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["infer_doc_res.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        inference.main()

    reached = {name: s.calls for name, s in sentinels.items() if s.calls}
    assert not reached, (
        f"--help reached {reached} before argparse could exit. "
        "`args = parse_arguments(argv)` must be the FIRST statement of main()."
    )
    assert excinfo.value.code == 0, f"--help exited {excinfo.value.code!r}"

    printed = capsys.readouterr().out
    assert printed.startswith("usage:"), (
        "--help printed no `usage:` line. An exit code of 0 is not evidence: a "
        "script with no parser at all runs its whole job and exits 0 anyway. "
        f"stdout was {printed[:200]!r}"
    )
    for flag in ("--input", "--task", "--checkpoint", "--gpu", "--output-dir"):
        assert flag in printed, f"--help does not advertise {flag}"


def test_a_missing_input_fails_before_the_gpu_and_the_model(monkeypatch, tmp_path):
    """The input check runs before anything is allocated.

    Measured through behaviour: a ``--input`` that names nothing must raise
    while the GPU and model sentinels are still untouched.
    """
    # Capture the REAL function BEFORE the sentinels are installed: reading
    # `inference.collect_input_images` afterwards hands back the sentinel, and
    # restoring a sentinel over itself makes the arm pass for the wrong reason.
    real_collect = inference.collect_input_images
    sentinels = _install_sentinels(monkeypatch)
    monkeypatch.setattr(inference, "collect_input_images", real_collect)

    with pytest.raises(FileNotFoundError):
        inference.main(["--input", str(tmp_path / "absent.png")])

    reached = {
        name: s.calls
        for name, s in sentinels.items()
        if s.calls and name != "collect_input_images"
    }
    assert not reached, f"a missing --input reached {reached}"


def test_every_task_plus_end2end_is_offered_and_a_typo_is_rejected(capsys):
    """The task surface is the TABLE plus the one composite, and nothing else."""
    parser = inference.build_parser()
    action = next(a for a in parser._actions if a.dest == "task")
    assert list(action.choices) == list(task_names()) + [inference.END2END]

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--input", "x.png", "--task", "binarisation"])
    assert excinfo.value.code == 2
    message = capsys.readouterr().err
    for name in task_names():
        assert name in message, message


# ---------------------------------------------------------------------
# padding: an EXACT spatial round trip
# ---------------------------------------------------------------------


@pytest.mark.parametrize("height", ROUND_TRIP_EXTENTS)
@pytest.mark.parametrize("width", ROUND_TRIP_EXTENTS)
def test_pad_then_crop_restores_the_exact_spatial_size(height, width):
    """Output size equals input size EXACTLY, for every residue mod 8."""
    array = _page(height, width, seed=height * 1000 + width)
    padded, pad_h, pad_w = inference.pad_to_multiple(array)

    assert padded.shape[0] % 8 == 0 and padded.shape[1] % 8 == 0, (
        f"{(height, width)} padded to {padded.shape[:2]}, which DocRes refuses"
    )
    assert padded.shape[0] - pad_h == height
    assert padded.shape[1] - pad_w == width

    restored = inference.crop_padding(padded, pad_h, pad_w)
    assert restored.shape == array.shape, (
        f"{(height, width)} -> {padded.shape[:2]} -> {restored.shape[:2]}"
    )
    np.testing.assert_array_equal(restored, array)


@pytest.mark.parametrize("extent", (8, 16, 256))
def test_an_already_legal_extent_is_not_padded_at_all(extent):
    """A divisor bug that always padded would still round-trip -- but resize."""
    padded, pad_h, pad_w = inference.pad_to_multiple(_page(extent, extent))
    assert (pad_h, pad_w) == (0, 0)
    assert padded.shape[:2] == (extent, extent)


def test_the_padding_is_added_at_the_top_and_the_left():
    """The SIDE, which a round trip alone cannot see.

    ``crop_padding`` removes ``[pad_h:, pad_w:]``, so padding anywhere else
    returns a correctly-shaped page whose content has moved. Upstream's
    ``stride_integral`` replicates the top row and the left column
    (``crop_merge_image.py:119-129``).
    """
    array = _page(13, 11, seed=7)
    padded, pad_h, pad_w = inference.pad_to_multiple(array)
    assert (pad_h, pad_w) == (3, 5)

    # The original occupies the BOTTOM-RIGHT of the padded array.
    np.testing.assert_array_equal(padded[pad_h:, pad_w:], array)
    # ... and the padding is a replication of the original's edges.
    for row in range(pad_h):
        np.testing.assert_array_equal(padded[row, pad_w:], array[0])
    for column in range(pad_w):
        np.testing.assert_array_equal(padded[pad_h:, column], array[:, 0])


@pytest.mark.parametrize("height,width", ((33, 100), (255, 41), (100, 255)))
def test_an_identity_model_reproduces_the_page_exactly(height, width):
    """The whole pipeline, end to end, must be lossless on an identity model.

    This is the strongest padding guard in the file: a model that returns its
    own RGB input means every byte of the output is traceable to a byte of the
    input, so a pad/crop side disagreement, an off-by-one crop, a double
    normalisation or a stray resize all move content and fail here. It also
    pins the ``/255 -> clip -> *255`` round trip to the identity.
    """
    page = _page(height, width, seed=height + width)
    result = inference.restore(_identity_model(), page, "deblurring")

    assert result.image.shape == page.shape, (
        f"a {page.shape} page came back {result.image.shape}"
    )
    np.testing.assert_array_equal(result.image, page)
    assert result.stages == ("deblurring",)


def test_the_model_never_sees_a_shape_docres_would_refuse():
    """What the padding is FOR: DocRes raises on a non-multiple of 8 (D-016)."""
    model = _identity_model()
    inference.restore(model, _page(33, 100, seed=3), "deblurring")

    (fed,) = model.inputs
    assert fed.shape[0] == 1 and fed.shape[3] == 6, fed.shape
    assert fed.shape[1] % 8 == 0 and fed.shape[2] % 8 == 0, (
        f"the model was handed {fed.shape[1:3]}, which DocRes refuses"
    )


# ---------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,target",
    (((37, 53), (19, 71)), ((16, 16), (64, 64)), ((100, 40), (100, 40))),
)
def test_the_float_resize_rounds_to_the_prompt_modules_resize(source, target):
    """The two resamplers are ONE convention, measured -- not kept in lockstep.

    ``dtsprompt._resize_bilinear`` is this function followed by a round-and-clip
    to uint8. Duplication of a rule across two modules is a defect unless the
    agreement is checked; this checks it at the bit level rather than trusting
    two comments to stay true.
    """
    array = _page(*source, seed=11)
    mine = np.clip(
        np.rint(inference.resize_bilinear(array, *target)), 0, 255
    ).astype(np.uint8)
    np.testing.assert_array_equal(mine, _resize_bilinear(array, *target))


def test_the_remap_reproduces_an_identity_warp_and_zeroes_the_outside():
    """``remap_bilinear`` at OpenCV's default ``BORDER_CONSTANT`` of 0."""
    page = _page(9, 7, seed=5)
    grid_y, grid_x = np.meshgrid(
        np.arange(9, dtype=np.float64), np.arange(7, dtype=np.float64),
        indexing="ij",
    )
    np.testing.assert_array_equal(
        inference.remap_bilinear(page, grid_x, grid_y), page
    )

    # A sample two columns past the right edge has no valid tap at all.
    outside = inference.remap_bilinear(page, grid_x + 100.0, grid_y)
    assert np.all(outside == 0), (
        "a warp reaching past the page must darken, not smear the border in"
    )


# ---------------------------------------------------------------------
# binarization: argmax over a logit PAIR, not a threshold
# ---------------------------------------------------------------------


def _sigmoid_threshold_reference(prediction: np.ndarray) -> np.ndarray:
    """The plausible WRONG implementation this file exists to keep out.

    Reads ONE channel, squashes it and cuts at 0.5. Same output shape, same
    dtype, same ``{0, 255}`` range as the real post-processor.
    """
    probability = 1.0 / (1.0 + np.exp(-prediction[..., BINARY_INK_CLASS_INDEX]))
    return np.where(
        probability >= 0.5, inference.INK_LEVEL_U8, inference.BACKGROUND_LEVEL_U8
    ).astype(np.uint8)


class TestBinarizationIsArgmaxNotAThreshold:
    """The two disagree, and this pins WHICH answer arrives."""

    #: Both logits negative, but channel 1 wins the pairwise comparison. argmax
    #: says class 1; a sigmoid on channel 1 says 0.27 < 0.5, i.e. class 0.
    DISCRIMINATING = (-3.0, -1.0)

    def _predict(self, logits: Tuple[float, float], shape=(4, 6)) -> np.ndarray:
        prediction = np.zeros(shape + (3,), dtype=np.float32)
        prediction[..., 0] = logits[0]
        prediction[..., 1] = logits[1]
        prediction[..., 2] = 99.0  # the unsupervised channel must be IGNORED
        return prediction

    def test_the_output_is_exactly_two_levels(self):
        rng = np.random.default_rng(2)
        prediction = rng.normal(size=(12, 10, 3)).astype(np.float32)
        page = inference.postprocess(
            prediction, _plan(_page(12, 10, seed=2)), get_task("binarization")
        )
        assert page.dtype == np.uint8
        assert set(np.unique(page)).issubset({0, 255}), np.unique(page)
        assert set(np.unique(page)) == {0, 255}, (
            "a binarized page that is all one level cannot discriminate; the "
            "random logits above must produce both classes"
        )

    def test_argmax_and_a_threshold_disagree_on_the_discriminating_pair(self):
        """Anti-vacuity: without this, the arm below could be comparing equals."""
        prediction = self._predict(self.DISCRIMINATING)
        page = inference.postprocess(
            prediction, _plan(_page(4, 6)), get_task("binarization")
        )
        threshold = _sigmoid_threshold_reference(prediction)

        assert not np.array_equal(page, threshold), (
            "the reference threshold agrees with argmax here, so the guard "
            "below would pass against a threshold implementation"
        )

    def test_the_shipped_answer_is_the_argmax_one(self):
        prediction = self._predict(self.DISCRIMINATING)
        page = inference.postprocess(
            prediction, _plan(_page(4, 6)), get_task("binarization")
        )
        # class 1 wins the pair, and class 1 is ink (common.BINARY_INK_CLASS_INDEX)
        assert BINARY_INK_CLASS_INDEX == 1
        assert np.all(page == inference.INK_LEVEL_U8), (
            f"logits {self.DISCRIMINATING} -> class 1 -> ink. Got "
            f"{np.unique(page)}; a sigmoid threshold on channel 1 would give "
            f"{np.unique(_sigmoid_threshold_reference(prediction))}"
        )

    def test_the_softmax_upstream_applies_cannot_move_the_argmax(self):
        """Upstream's ``argmax(softmax(x))``; softmax is strictly increasing."""
        rng = np.random.default_rng(4)
        logits = rng.normal(scale=5.0, size=(64, 64, 2))
        shifted = logits - logits.max(axis=-1, keepdims=True)
        softmax = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        np.testing.assert_array_equal(
            np.argmax(logits, axis=-1), np.argmax(softmax, axis=-1)
        )

    def test_the_ink_class_constant_is_live(self, monkeypatch):
        """Flipping ``BINARY_INK_CLASS_INDEX`` flips the picture.

        The constant is shared with the training target
        (``common._ground_truth_two_class``), so a decorative copy here would
        let the two drift into disagreement about what class 1 means.
        """
        prediction = self._predict(self.DISCRIMINATING)
        plan = _plan(_page(4, 6))
        spec = get_task("binarization")

        before = inference.postprocess(prediction, plan, spec)
        monkeypatch.setattr(inference, "BINARY_INK_CLASS_INDEX", 0)
        after = inference.postprocess(prediction, plan, spec)

        assert np.all(before == 0) and np.all(after == 255), (
            f"the ink-class constant is not read: {np.unique(before)} vs "
            f"{np.unique(after)}"
        )

    def test_the_internal_class_order_diverges_but_the_written_page_does_not(
            self,
    ):
        """D-032. The divergence from upstream is INTERNAL and stays internal.

        Upstream's ink is class **0**, and that is traceable from source:
        ``loaders/docres_loader.py:114-123`` thresholds the GT to ``{0, 255}``
        then divides by 255 (ink -> 0), and ``inference.py:250-252`` argmaxes
        then multiplies by 255 (class 1 -> white background). This port fixes
        ink at class 1 instead -- deliberately, because flipping it now would
        invert every page the shipped binarization checkpoint produces.

        What makes that safe is the second half of this guard: the WRITTEN page
        carries upstream's own polarity regardless. Ink is 0 and background is
        255, the DIBCO convention the ground truth uses, so no external
        consumer can observe the internal order. If this guard ever goes red on
        its second half, the divergence has escaped and the constant must be
        flipped (and every checkpoint retrained).
        """
        assert BINARY_INK_CLASS_INDEX == 1, (
            "upstream is 0; this port's 1 is a recorded, deliberate divergence "
            "(D-032) -- do not change it without retraining the checkpoints"
        )
        assert inference.INK_LEVEL_U8 == 0
        assert inference.BACKGROUND_LEVEL_U8 == 255

    def test_the_padding_is_cropped_off_a_binarized_page(self):
        page = _page(33, 41, seed=9)
        result = inference.restore(
            _FakeModel(lambda x: np.zeros(x.shape[:3] + (3,), np.float32)),
            page, "binarization",
        )
        assert result.image.shape == page.shape[:2], (
            f"a {page.shape[:2]} page binarized to {result.image.shape}"
        )


# ---------------------------------------------------------------------
# the >=1600 px homomorphic branch
# ---------------------------------------------------------------------


class TestTheHomomorphicBranchIsLive:
    """The branch is SELECTED at the boundary and it CHANGES the answer."""

    def test_the_boundary_is_inclusive_and_selects_the_resized_branch(self):
        """``max(h, w) >= 1600`` resizes; one pixel below it pads.

        Runs the real ``build_input_plan`` on the real deshadowing spec, so
        this measures the shipped selection rather than a re-typed condition.
        """
        below = inference.build_input_plan(
            _page(1599, 40, seed=1), get_task("deshadowing")
        )
        assert below.resized is False, "1599 px must be padded, not resized"
        assert below.array.shape[1:3] == (1600, 40), below.array.shape

        cap = get_task("deshadowing").max_input_size
        assert cap == 1600, cap
        at = inference.build_input_plan(
            _page(cap, 40, seed=1), get_task("deshadowing")
        )
        assert at.resized is True, (
            f"{cap} px must take the resized branch; upstream's condition is "
            "`if max(w,h) < MAX_SIZE: pad`"
        )
        assert at.array.shape[1:3] == (cap, cap)

    def test_the_homomorphic_output_differs_from_a_plain_upsample(self):
        """A branch that agreed with the fallback would be decoration.

        The prediction is a smooth low-frequency field and the original page
        carries high-frequency detail the 1600 px prediction cannot: upsampling
        ``pred`` returns the prediction's detail stretched to page size, while
        dividing the ORIGINAL by the upsampled ratio field keeps the page's own
        detail. The two are not close, and this measures how far apart.
        """
        size, height, width = 64, 200, 160
        original = _page(height, width, seed=13)
        ramp = np.linspace(0.3, 0.9, size, dtype=np.float32)
        prediction = np.repeat(
            np.repeat(ramp[:, None, None], size, axis=1), 3, axis=2
        )
        plan = _plan(original, resized=True, model_size=(size, size))

        homomorphic = inference.postprocess(
            prediction, plan, get_task("deshadowing")
        )
        upsampled = np.clip(
            np.rint(
                inference.resize_bilinear(
                    np.clip(prediction, 0, 1) * PIXEL_SCALE, height, width
                )
            ),
            0, 255,
        ).astype(np.uint8)

        assert homomorphic.shape == original.shape
        difference = np.abs(
            homomorphic.astype(int) - upsampled.astype(int)
        ).mean()
        assert difference > 10.0, (
            "the homomorphic re-composition produced essentially the plain "
            f"upsample (mean |delta| = {difference:.3f}); the branch is not "
            "doing what inference.py:163-166 does"
        )

    def test_detail_the_low_resolution_prediction_cannot_carry_survives(self):
        """The claim behind the branch, stated as a measurement.

        A one-pixel-tall bright row cannot be represented in a 64 px-tall
        prediction. The homomorphic path divides a SMOOTH field out of the
        FULL-resolution original, so the row survives; upsampling the
        prediction discards it entirely. That difference is the whole reason
        ``inference.py:163-166`` exists.
        """
        size, height, width = 64, 200, 160
        original = np.full((height, width, 3), 100, dtype=np.uint8)
        feature_row = height // 2
        original[feature_row] = 200
        prediction = np.full((size, size, 3), 0.4, dtype=np.float32)
        plan = _plan(original, resized=True, model_size=(size, size))

        homomorphic = inference.postprocess(
            prediction, plan, get_task("deshadowing")
        ).astype(float)
        upsampled = inference.resize_bilinear(
            np.clip(prediction, 0, 1) * PIXEL_SCALE, height, width
        )

        kept = homomorphic[feature_row].mean() - homomorphic[feature_row - 2].mean()
        lost = abs(
            upsampled[feature_row].mean() - upsampled[feature_row - 2].mean()
        )
        assert kept > 40.0, (
            f"the one-pixel feature came through at only {kept:.1f} grey "
            "levels of contrast; the full-resolution original is not reaching "
            "the output"
        )
        assert lost < 2.0, (
            f"the control is not a control: a plain upsample of the prediction "
            f"already carries {lost:.1f} grey levels of the feature"
        )

    def test_a_zero_prediction_does_not_divide_by_zero(self):
        """Upstream's ``pred[pred==0] = 1``: the prediction is a DIVISOR here."""
        size = 32
        out = inference.postprocess(
            np.zeros((size, size, 3), dtype=np.float32),
            _plan(_page(80, 60, seed=19), resized=True, model_size=(size, size)),
            get_task("deshadowing"),
        )
        assert np.isfinite(out.astype(float)).all()
        assert out.dtype == np.uint8

    def test_below_the_cap_the_padding_is_cropped_instead(self):
        """The control: the other side of the same branch."""
        page = _page(37, 45, seed=21)
        result = inference.restore(_identity_model(), page, "deshadowing")
        np.testing.assert_array_equal(result.image, page)


# ---------------------------------------------------------------------
# dewarping: x and y are not swapped
# ---------------------------------------------------------------------


class TestDewarpingDoesNotSwapXAndY:
    """Same defect class as step 9's ``base_coordinate_grid`` guard (D-021).

    The oracle is a page that is CONSTANT along one axis: shifting the flow
    along that axis is a no-op, shifting it along the other is not. Swapping
    the two flow channels therefore inverts which shift does nothing, and no
    shape, dtype or range assertion can see it.
    """

    SIZE = 64
    HEIGHT = 80
    WIDTH = 48
    #: A flow offset in normalised units. 0.25 of the axis is far larger than
    #: the box blur can smear and far smaller than the page.
    OFFSET = 0.25
    #: Rows/columns discarded before comparing. An offset warp pulls content in
    #: from OFF the page along the offset axis, and OpenCV's border rule
    #: (reproduced here) fills that with black -- a difference that has nothing
    #: to do with the channel order and would swamp both arms. The margin is
    #: ``OFFSET`` of the axis plus two pixels of slack.
    MARGIN = 0.25

    def _column_ramp(self) -> np.ndarray:
        """Varies along x (columns), CONSTANT down y (rows)."""
        ramp = np.linspace(0, 255, self.WIDTH).astype(np.uint8)
        return np.repeat(
            np.repeat(ramp[None, :, None], self.HEIGHT, axis=0), 3, axis=2
        )

    def _row_ramp(self) -> np.ndarray:
        """Varies along y (rows), CONSTANT across x (columns)."""
        ramp = np.linspace(0, 255, self.HEIGHT).astype(np.uint8)
        return np.repeat(
            np.repeat(ramp[:, None, None], self.WIDTH, axis=1), 3, axis=2
        )

    def _warp(self, page: np.ndarray, channel: int) -> np.ndarray:
        """Post-process a flow that is a constant offset in ONE channel."""
        prediction = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.float32)
        prediction[..., channel] = self.OFFSET
        return inference.postprocess(
            prediction,
            _plan(page, resized=True, model_size=(self.SIZE, self.SIZE)),
            get_task("dewarping"),
        )

    def _identity_warp(self, page: np.ndarray) -> np.ndarray:
        return inference.postprocess(
            np.zeros((self.SIZE, self.SIZE, 3), dtype=np.float32),
            _plan(page, resized=True, model_size=(self.SIZE, self.SIZE)),
            get_task("dewarping"),
        )

    def _interior(self, array: np.ndarray, channel: int) -> np.ndarray:
        """Drop the band the offset pulls in from off the page, plus a border."""
        rows = self.HEIGHT - int(self.MARGIN * self.HEIGHT) - 2
        columns = self.WIDTH - int(self.MARGIN * self.WIDTH) - 2
        if channel == 0:
            return array[1:-1, 1:columns]
        return array[1:rows, 1:-1]

    def _delta(self, page: np.ndarray, channel: int) -> float:
        """Mean |change| over the interior when ``channel`` carries the offset."""
        warped = self._interior(self._warp(page, channel), channel)
        return float(
            np.abs(warped.astype(int) - self._interior(page, channel).astype(int)).mean()
        )

    def test_a_zero_flow_is_very_nearly_the_identity_warp(self):
        """Anti-vacuity: without the offset the warp must be (almost) a no-op.

        The base coordinate grid IS the identity map. "Almost" is exact and
        upstream-faithful, not slop: the grid is normalised by its own extent
        at 256 px, resized to ``(h, w)`` under OpenCV's half-pixel convention
        and multiplied back by ``(w, h)``, which lands on ``j + 0.5 - W/512``
        rather than ``j``. Upstream's ``inference.py:123`` does the identical
        three steps and carries the identical sub-half-pixel offset. If this
        arm failed outright, every arm below would be measuring the grid rather
        than the channel order.
        """
        page = self._column_ramp()
        residual = np.abs(
            self._identity_warp(page)[1:-1, 1:-1].astype(int)
            - page[1:-1, 1:-1].astype(int)
        ).mean()
        assert residual < 4.0, (
            f"a zero flow moved the page by {residual:.2f} grey levels, which "
            "is more than the sub-pixel grid offset can account for"
        )

    def test_channel_zero_moves_content_along_x_and_not_along_y(self):
        """Channel 0 is x: it changes a column ramp and leaves a row ramp alone."""
        column = self._delta(self._column_ramp(), 0)
        row = self._delta(self._row_ramp(), 0)
        assert column > 30.0, (
            f"an x offset barely moved a column ramp (mean |delta|={column:.2f})"
        )
        assert row < 5.0, (
            f"an x offset moved a ROW ramp (mean |delta|={row:.2f}) -- channel "
            "0 is being applied to the y axis, i.e. x and y are swapped"
        )

    def test_channel_one_moves_content_along_y_and_not_along_x(self):
        """Channel 1 is y: the exact mirror of the arm above."""
        row = self._delta(self._row_ramp(), 1)
        column = self._delta(self._column_ramp(), 1)
        assert row > 30.0, (
            f"a y offset barely moved a row ramp (mean |delta|={row:.2f})"
        )
        assert column < 5.0, (
            f"a y offset moved a COLUMN ramp (mean |delta|={column:.2f}) -- "
            "channel 1 is being applied to the x axis"
        )

    def test_the_output_is_the_original_pages_size_not_the_networks(self):
        """The flow is predicted at 256 and the page comes back full size."""
        out = self._identity_warp(self._column_ramp())
        assert out.shape == (self.HEIGHT, self.WIDTH, 3), out.shape

    def test_the_flow_is_smoothed_before_the_warp(self):
        """The 15 box-blur passes are reproduced, not silently dropped.

        A single-pixel spike in the predicted flow must not survive to the warp.
        Measured by comparing against the same run with the passes set to zero.
        """
        page = self._column_ramp()
        prediction = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.float32)
        prediction[self.SIZE // 2, self.SIZE // 2, 0] = 5.0
        plan = _plan(page, resized=True, model_size=(self.SIZE, self.SIZE))
        spec = get_task("dewarping")

        smoothed = inference.postprocess(prediction, plan, spec)
        assert inference.FLOW_SMOOTHING_PASSES == 15

        import unittest.mock as mock

        with mock.patch.object(inference, "FLOW_SMOOTHING_PASSES", 0):
            unsmoothed = inference.postprocess(prediction, plan, spec)

        assert not np.array_equal(smoothed, unsmoothed), (
            "smoothing the predicted flow changed nothing, so the 15 blur "
            "passes are not reaching the warp"
        )


# ---------------------------------------------------------------------
# end2end: the ORDER, measured
# ---------------------------------------------------------------------


class TestEnd2EndChainsThreeStagesInOrder:
    """Order asserted by INSTRUMENTATION, never by reading the source."""

    def _record(self, monkeypatch) -> List[Tuple[str, np.ndarray]]:
        seen: List[Tuple[str, np.ndarray]] = []
        real = inference.restore

        def spy(model, page, task, mask=None):
            seen.append((task, np.asarray(page).copy()))
            return real(model, page, task, mask=mask)

        monkeypatch.setattr(inference, "restore", spy)
        return seen

    def test_the_three_stages_run_in_the_declared_order(self, monkeypatch):
        seen = self._record(monkeypatch)
        page = _page(40, 32, seed=23)
        mask = np.full(page.shape[:2], 255, dtype=np.uint8)

        result = inference.restore_end2end(_identity_model(), page, mask=mask)

        assert [name for name, _ in seen] == list(inference.END2END_STAGES)
        assert result.stages == inference.END2END_STAGES
        assert inference.END2END_STAGES == (
            "dewarping", "deshadowing", "appearance",
        )
        for name in inference.END2END_STAGES:
            assert name in TASKS, f"{name!r} is not a TASKS key"

    def test_each_stage_is_fed_the_previous_stages_output(self, monkeypatch):
        """Chaining, not three independent runs on the original page.

        A shim that ran each stage on ``page`` would produce the same three
        names in the same order and pass an order-only assertion.
        """
        seen = self._record(monkeypatch)
        page = _page(40, 32, seed=29)
        mask = np.full(page.shape[:2], 255, dtype=np.uint8)

        # Each stage lightens its input, so every stage's input must differ
        # from the one before it -- and from the original.
        model = _FakeModel(lambda x: np.clip(x[..., :3] * 1.2, 0.0, 1.0))
        inference.restore_end2end(model, page, mask=mask)

        inputs = [array for _, array in seen]
        np.testing.assert_array_equal(inputs[0], page)
        assert not np.array_equal(inputs[1], page), (
            "the deshadowing stage was handed the ORIGINAL page; the stages "
            "are not chained"
        )
        assert not np.array_equal(inputs[2], inputs[1]), (
            "the appearance stage was handed the deshadowing stage's input "
            "rather than its output"
        )

    def test_dewarping_without_a_mask_says_what_is_missing(self):
        """The mask comes from MBD, which this port does not include."""
        with pytest.raises(ValueError, match="mask"):
            inference.restore(_identity_model(), _page(32, 32), "dewarping")

    def test_end2end_is_not_a_task_and_is_never_looked_up_in_the_table(self):
        assert inference.END2END not in TASKS
        with pytest.raises(ValueError, match="unknown DocRes task"):
            get_task(inference.END2END)


# ---------------------------------------------------------------------
# the TABLE drives the behaviour
# ---------------------------------------------------------------------


class TestTheCapIsATableFieldNotAMode:
    """D-033. ``max_input_size`` is per TASK; deblurring does not carry it.

    The homomorphic re-composition is a multiplicative-illumination-field
    model. It is right for a shadow and physically wrong for a blur, which is a
    convolution -- and upstream's ``deblurring()`` (``inference.py:200-227``)
    has no ``MAX_SIZE`` branch at all. Since most document photographs exceed
    1600 px, this is deblurring's COMMON path, so these guards watch the common
    case rather than an edge case.
    """

    def test_deblurring_is_padded_at_and_above_the_size_that_caps_deshadowing(
            self,
    ):
        """Same page, two tasks, two branches -- and no task string compared."""
        cap = get_task("deshadowing").max_input_size
        page = _page(cap, 40, seed=7)

        deshadow = inference.build_input_plan(page, get_task("deshadowing"))
        deblur = inference.build_input_plan(page, get_task("deblurring"))

        assert deshadow.resized is True
        assert deblur.resized is False, (
            "deblurring must NOT be resized-and-re-composed: a blur is a "
            "convolution, not a multiplicative field (upstream has no cap here)"
        )
        assert deblur.array.shape[1:3] == (cap, 40), deblur.array.shape

    def test_the_shipped_table_gives_the_cap_to_exactly_the_upstream_two(self):
        """The literal binding, pinned. Upstream: `inference.py:141-166`."""
        capped = {
            name for name in task_names()
            if get_task(name).max_input_size is not None
        }
        assert capped == {"deshadowing", "appearance"}, capped

    def test_a_deblurring_page_above_the_cap_keeps_every_original_pixel(self):
        """The consequence that matters, measured end to end on the plan.

        A padded plan crops back to the original; a resized plan cannot. This
        asserts the network sees the page's own resolution.
        """
        cap = get_task("deshadowing").max_input_size
        page = _page(cap + 9, 40, seed=11)
        plan = inference.build_input_plan(page, get_task("deblurring"))
        assert plan.resized is False
        # `pad_to_multiple` pads up to a multiple of 8, never down.
        assert plan.array.shape[1] >= page.shape[0]
        assert plan.array.shape[1] - plan.pad_h == page.shape[0]
        assert plan.array.shape[2] - plan.pad_w == page.shape[1]

    def test_the_branch_follows_the_FIELD_not_the_task(self):
        """A fabricated row moves the cap; the shipped code follows it.

        This is what makes the guard above a table test rather than two
        hard-coded expectations: giving deblurring a small cap resizes it, and
        taking deshadowing's away pads it.
        """
        page = _page(200, 64, seed=17)

        capped_deblur = inference.build_input_plan(
            page, _spec_with("deblurring", max_input_size=128)
        )
        assert capped_deblur.resized is True
        assert capped_deblur.array.shape[1:3] == (128, 128)

        uncapped_deshadow = inference.build_input_plan(
            page, _spec_with("deshadowing", max_input_size=None)
        )
        assert uncapped_deshadow.resized is False

    def test_a_non_positive_cap_is_rejected_by_the_table_validator(self):
        """A zero cap would resize every page to 0x0; the table refuses it."""
        bad = _spec_with("deshadowing", max_input_size=0)
        with pytest.raises(ValueError, match="max_input_size"):
            tasks_module._validate((bad,))


class TestTheTableDrivesTheBehaviour:
    """Change the table, change the behaviour. Otherwise it is just a comment."""

    def test_both_dispatch_tables_cover_exactly_the_declared_modes(self):
        assert set(inference.INPUT_BUILDERS) == set(POSTPROCESS_MODES)
        assert set(inference.POSTPROCESSORS) == set(POSTPROCESS_MODES)

    def test_changing_a_specs_postprocess_changes_the_post_processing(self):
        """A fabricated row: deblurring's spec, re-pointed at argmax."""
        prediction = np.zeros((8, 8, 3), dtype=np.float32)
        prediction[..., 1] = 1.0
        plan = _plan(_page(8, 8, seed=31))

        as_image = inference.postprocess(
            prediction, plan, get_task("deblurring")
        )
        as_binary = inference.postprocess(
            prediction, plan,
            _spec_with("deblurring", postprocess=POSTPROCESS_ARGMAX_BINARY),
        )
        assert as_image.ndim == 3 and as_binary.ndim == 2
        assert set(np.unique(as_binary)) == {0}

    def test_changing_a_specs_postprocess_changes_the_input_preparation(self):
        """A 33x33 page is padded for one mode and resized to 256 for another."""
        page = _page(33, 33, seed=37)
        padded = inference.build_input_plan(page, get_task("binarization"))
        assert padded.array.shape[1:3] == (40, 40) and not padded.resized

        rerouted = inference.build_input_plan(
            page,
            _spec_with(
                "binarization",
                postprocess=POSTPROCESS_FLOW_REMAP,
                requires_mask=False,
            ),
        )
        assert rerouted.array.shape[1:3] == (
            inference.DEWARP_INPUT_SIZE, inference.DEWARP_INPUT_SIZE,
        )

    def test_the_supervised_slice_comes_from_the_table(self):
        """``n_supervised_channels`` decides which channels the argmax sees.

        The real binarization row supervises 2 channels; a fabricated 3-channel
        row makes the third channel win, which it never can today.
        """
        prediction = np.zeros((4, 4, 3), dtype=np.float32)
        prediction[..., 1] = 1.0
        prediction[..., 2] = 9.0  # only reachable if the slice widens
        plan = _plan(_page(4, 4, seed=41))

        two = inference.postprocess(prediction, plan, get_task("binarization"))
        three = inference.postprocess(
            prediction, plan,
            _spec_with("binarization", n_supervised_channels=3),
        )
        assert np.all(two == inference.INK_LEVEL_U8), np.unique(two)
        assert np.all(three == inference.BACKGROUND_LEVEL_U8), np.unique(three)

    def test_the_prompt_generator_comes_from_the_table(self, monkeypatch):
        """``spec.prompt_fn`` is called; nothing re-derives a prompt per task."""
        calls: List[str] = []

        def fake_prompt(img: np.ndarray) -> np.ndarray:
            calls.append("called")
            return np.full(img.shape[:2] + (3,), 7, dtype=np.uint8)

        plan = inference.build_input_plan(
            _page(16, 16, seed=43),
            _spec_with("binarization", prompt_fn=fake_prompt),
        )
        assert calls == ["called"]
        np.testing.assert_allclose(
            plan.array[0, ..., 3:], 7.0 / PIXEL_SCALE, atol=1e-6
        )

    def test_no_task_name_is_compared_anywhere_in_the_module(self):
        """The rule, checked against the source as a backstop.

        Source text is weak evidence (LESSONS: a byte-identity guard measures
        text, not behaviour), which is why every arm above measures behaviour
        instead. This one exists only to catch the easiest regression -- someone
        adding ``if task == "deblurring"`` -- and it is deliberately the LAST
        line of defence, not the first.
        """
        source = Path(inference.__file__).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )
        for name in task_names():
            for pattern in (f'== "{name}"', f'== \'{name}\'', f'"{name}" =='):
                assert pattern not in code, (
                    f"the module compares a task string ({pattern}); task "
                    "specificity belongs in TASKS"
                )


# ---------------------------------------------------------------------
# files and the loud untrained-model warning
# ---------------------------------------------------------------------


def test_an_absent_checkpoint_is_loud_and_not_silent(caplog, monkeypatch):
    """An untrained model must never be mistaken for a trained one."""
    built: List[str] = []
    monkeypatch.setattr(
        inference, "create_doc_res",
        lambda variant: built.append(variant) or _identity_model(),
    )
    with caplog.at_level("WARNING"):
        inference.load_docres_model(None, "docres")

    assert built == ["docres"]
    text = caplog.text.upper()
    assert "NO CHECKPOINT" in text and "NOISE" in text, caplog.text


def test_a_named_checkpoint_that_does_not_exist_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.load_docres_model(str(tmp_path / "absent.keras"))


def test_collect_input_images_takes_a_file_or_a_directory(tmp_path):
    from PIL import Image

    for name in ("b.png", "a.tif", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    for name in ("b.png", "a.tif"):
        Image.fromarray(_page(4, 4)).save(tmp_path / name)

    assert inference.collect_input_images(tmp_path / "b.png") == [
        tmp_path / "b.png"
    ]
    assert inference.collect_input_images(tmp_path) == [
        tmp_path / "a.tif", tmp_path / "b.png",
    ]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no image"):
        inference.collect_input_images(empty)


def test_a_written_page_round_trips_at_the_input_size(tmp_path):
    """The file-level contract: what goes in at (33, 100) comes out at (33, 100)."""
    from PIL import Image

    source = tmp_path / "page.png"
    page = _page(33, 100, seed=47)
    Image.fromarray(page).save(source)

    out_path = inference.restore_image_file(
        _identity_model(), source, tmp_path / "out", "deblurring",
        save_dtsprompt=True,
    )
    assert out_path == tmp_path / "out" / "page_deblurring.png"
    with Image.open(out_path) as handle:
        written = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    np.testing.assert_array_equal(written, page)

    for index in (1, 2, 3):
        assert (tmp_path / "out" / f"page_deblurring_prompt{index}.png").is_file()


def test_main_restores_a_directory_without_a_checkpoint(tmp_path, monkeypatch):
    """The real ``main`` path, with only the GPU and the model stubbed out."""
    from PIL import Image

    inputs = tmp_path / "in"
    inputs.mkdir()
    for index, size in enumerate(((33, 41), (24, 24))):
        Image.fromarray(_page(*size, seed=index)).save(inputs / f"p{index}.png")

    monkeypatch.setattr(inference, "setup_gpu", lambda gpu_id=None: None)
    monkeypatch.setattr(
        inference, "load_docres_model", lambda *a, **k: _identity_model()
    )

    inference.main([
        "--input", str(inputs), "--output-dir", str(tmp_path / "out"),
        "--task", "deblurring",
    ])

    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert written == ["p0_deblurring.png", "p1_deblurring.png"]


def test_the_tasks_table_is_the_one_this_module_reads():
    """``inference`` imports the live table, not a copy."""
    assert inference.get_task is tasks_module.get_task
    assert set(TASKS) == set(task_names())

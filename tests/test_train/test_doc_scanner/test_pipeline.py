r"""Data-pipeline, objective and optimizer guards for `src/train/doc_scanner/`.

The defect class this exists to catch
-------------------------------------
The rectifier's supervision target is a FOUR-channel stack, ``[f_gt, g]``, and
every plausible way of getting it wrong -- swapping the two halves, storing a
normalised map instead of an absolute-pixel one, transposing x and y --
preserves the shape, the dtype, the finiteness, the value range and the loss's
own willingness to consume it. The recorded failure mode of this repo is
precisely that class: one ``ops.roll`` sign error survived 249 tests because
every one of them was shape, config or serialization.

So the ``y_true`` half is NOT asserted against a comment restating the
convention. It is asserted against
``tests/test_datasets/test_document_rectification/backward_map_convention.py``
-- the shared instrument that IS the ``f_gt`` / ``g`` contract, already used by
both dataset modules, whose own ability to fail is proven in
``test_backward_map_convention.py``. If the trainer and the dataset ever
disagree about which half is which, they disagree with the same instrument.

TWO THINGS MEASURED HERE THAT ARE NOT VISIBLE ANYWHERE ELSE
-----------------------------------------------------------
1. ``optimizer_builder`` RENAMES the clipping keys. A literal ``"clipnorm"``
   key is silently ignored and the run trains unclipped with no error and no
   warning (``src/train/CLAUDE.md``). :class:`TestTheOptimizerRecipes` reads
   the constructed optimizer, not the config dict.
2. ``DocScannerRectifier`` uses Keras' ``training`` flag to select its output
   RANK, so a plain ``fit(validation_data=...)`` hands the compiled loss a
   rank-4 tensor at the first validation batch.
   :class:`TestTheValidationRankDispatch` pins both arms and pins that the
   rank-5 guard is still live.

Nothing here writes into repo-root ``results/``: every run directory goes to
``tmp_path``. The autouse guard in ``tests/conftest.py`` ASSERTS on a write
there, so a regression is a teardown ERROR rather than a silent mess.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, List

import keras
import numpy as np
import pytest

from train.doc_scanner import common
from train.doc_scanner.common import (
    RECTIFIER_TARGET_CHANNELS,
    SEGMENTER_HEAD_COUNT,
    STAGE_RECTIFIER,
    STAGE_SEGMENTER,
    DocScannerRectifierObjective,
    DocScannerTrainingConfig,
)

from ...test_datasets.test_document_rectification.backward_map_convention import (  # noqa: TID252
    assert_is_backward_map,
    assert_is_forward_map,
)

#: Every sample in this module is generated at 32 px, the smallest legal size:
#: the rectifier needs a multiple of 8 and the segmenter pools five times, so
#: 32 is the binding constraint and `__post_init__` enforces it.
TINY = 32

#: Two pages is enough to exercise the page draw and cheap enough that the
#: whole module stays under a minute on CPU.
PAGE_COUNT = 2


@pytest.fixture(scope="module")
def page_root(tmp_path_factory) -> str:
    """A tiny synthetic page corpus written to a tmp dir.

    Deliberately NOT the staged DIBCO tree: this module must be runnable on a
    machine with no dataset volume mounted, and a generated page exercises the
    same code path.
    """
    pytest.importorskip("PIL", reason="load_rgb needs Pillow (the 'data' extra)")
    from PIL import Image

    root = tmp_path_factory.mktemp("doc_scanner_pages")
    rng = np.random.default_rng(7)
    for index in range(PAGE_COUNT):
        page = np.full((96, 72, 3), 245, dtype=np.uint8)
        # A few dark bars, so the page is not a constant field: a constant page
        # makes several of the assertions below trivially true.
        for row in rng.integers(4, 90, size=8):
            page[int(row):int(row) + 3, 6:66] = 40
        Image.fromarray(page).save(root / f"page_{index}.png")
    return str(root)


def _config(page_root: str, stage: str, **overrides: Any) -> DocScannerTrainingConfig:
    """A tiny, tmp-rooted config for one stage."""
    base = replace(
        common.stage_defaults(stage),
        pages_root=page_root,
        backgrounds_root=os.path.join(page_root, "no-backgrounds-here"),
        image_size=TINY,
        batch_size=2,
        epochs=1,
        steps_per_epoch=2,
        validation_steps=1,
        synthetic_samples=8,
        shuffle_buffer=4,
        val_split=0.25,
    )
    return replace(base, **overrides)


# ---------------------------------------------------------------------
# the numpy sample producers
# ---------------------------------------------------------------------


class TestTheSyntheticSampleContract:
    """Shape, dtype, range and determinism of one generated sample."""

    def _pages(self, page_root: str) -> List[str]:
        return [str(path) for path in common.collect_page_paths(
            _config(page_root, STAGE_SEGMENTER)
        )]

    def test_the_page_corpus_is_found(self, page_root):
        assert len(self._pages(page_root)) == PAGE_COUNT

    def test_the_segmenter_sample_is_an_image_and_a_binary_mask(self, page_root):
        config = _config(page_root, STAGE_SEGMENTER)
        image, target = common.synthetic_sample(
            config, self._pages(page_root), [], 0
        )
        assert image.shape == (TINY, TINY, 3) and image.dtype == np.float32
        assert 0.0 <= image.min() and image.max() <= 1.0
        assert target.shape == (TINY, TINY, 1) and target.dtype == np.float32
        assert set(np.unique(target)) <= {0.0, 1.0}, (
            "the segmenter's target is a page FOOTPRINT, not a soft map; "
            f"found values {np.unique(target)[:8]}"
        )
        assert 0.0 < target.mean() < 1.0, (
            "a mask that is all page or all background supervises nothing"
        )

    def test_the_rectifier_sample_is_the_four_channel_flow_stack(self, page_root):
        config = _config(page_root, STAGE_RECTIFIER)
        image, target = common.synthetic_sample(
            config, self._pages(page_root), [], 0
        )
        assert image.shape == (TINY, TINY, 3)
        assert target.shape == (TINY, TINY, RECTIFIER_TARGET_CHANNELS)
        assert target.dtype == np.float32
        assert np.isfinite(target).all()

    def test_the_same_index_reproduces_the_same_sample(self, page_root):
        """Determinism is what makes the train/validation split meaningful."""
        config = _config(page_root, STAGE_RECTIFIER)
        pages = self._pages(page_root)
        first = common.synthetic_sample(config, pages, [], 3)
        second = common.synthetic_sample(config, pages, [], 3)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_a_different_index_gives_a_different_sample(self, page_root):
        """Anti-vacuity for the arm above: the seed actually reaches the draw."""
        config = _config(page_root, STAGE_RECTIFIER)
        pages = self._pages(page_root)
        assert not np.array_equal(
            common.synthetic_sample(config, pages, [], 3)[1],
            common.synthetic_sample(config, pages, [], 4)[1],
        )

    def test_a_different_config_seed_gives_a_different_sample(self, page_root):
        """`--seed` must reach the generator, not only `set_seeds`."""
        pages = self._pages(page_root)
        assert not np.array_equal(
            common.synthetic_sample(
                _config(page_root, STAGE_RECTIFIER, seed=1), pages, [], 3)[1],
            common.synthetic_sample(
                _config(page_root, STAGE_RECTIFIER, seed=2), pages, [], 3)[1],
        )

    def test_the_producer_does_not_touch_the_global_numpy_rng(self, page_root):
        """A tf.data worker drawing from global state would be irreproducible.

        Measured by DRAWING from the legacy global generator either side of a
        sample: if the producer consumed any global state, the second draw
        would differ.
        """
        state = np.random.get_state()
        try:
            np.random.seed(1234)
            before = np.random.random(4)
            np.random.seed(1234)
            common.synthetic_sample(
                _config(page_root, STAGE_RECTIFIER),
                self._pages(page_root), [], 0,
            )
            after = np.random.random(4)
        finally:
            np.random.set_state(state)
        np.testing.assert_array_equal(before, after)

    def test_an_empty_page_list_is_refused_by_name(self, page_root):
        with pytest.raises(ValueError, match="at least one page raster"):
            common.synthetic_sample(
                _config(page_root, STAGE_RECTIFIER), [], [], 0
            )


class TestTheRectifierTargetMatchesTheLossContract:
    """The ``[f_gt, g]`` halves, asserted against the SHARED instrument.

    ``backward_map_convention`` is the one statement of the contract, used by
    ``synthetic_warp`` and ``uvdoc`` alike. Restating it in a comment here
    would hold nothing -- comments do not run, and the failure being guarded
    against (the two halves swapped) is invisible to shape, dtype, range and
    finiteness alike.
    """

    def _halves(self, page_root):
        config = _config(page_root, STAGE_RECTIFIER)
        pages = [str(path) for path in common.collect_page_paths(config)]
        _, target = common.synthetic_sample(config, pages, [], 11)
        return target[..., :2], target[..., 2:]

    def test_the_first_two_channels_are_the_backward_map(self, page_root):
        f_gt, _ = self._halves(page_root)
        assert_is_backward_map(f_gt, TINY, TINY, "rectifier_target[..., :2]")

    def test_the_last_two_channels_are_the_forward_map(self, page_root):
        _, g = self._halves(page_root)
        assert_is_forward_map(g, TINY, TINY, "rectifier_target[..., 2:]")

    def test_the_stack_order_is_not_reversible_without_the_instrument_noticing(
        self, page_root
    ):
        """RED proof for the two arms above: the swap MUST be caught.

        A swapped stack keeps the shape, the dtype, the pixel units and the
        finiteness. What it does not keep is the range: ``f_gt`` lives inside
        the frame and ``g`` does not, so the instrument's range check is what
        separates them -- which is exactly why this test drives the instrument
        rather than re-deriving a threshold.
        """
        f_gt, g = self._halves(page_root)
        swapped = np.concatenate([g, f_gt], axis=-1)
        with pytest.raises(AssertionError):
            assert_is_backward_map(
                swapped[..., :2], TINY, TINY, "swapped rectifier_target"
            )

    def test_the_halves_are_not_equal(self, page_root):
        """Anti-vacuity: a warp so mild that f_gt == g would pass everything."""
        f_gt, g = self._halves(page_root)
        assert np.abs(f_gt - g).max() > 1.0, (
            "f_gt and g are within a pixel of each other -- the sample carries "
            "essentially no warp and the guards above cannot distinguish them"
        )

    def test_rectifier_target_preserves_its_two_inputs(self):
        """The packer itself, on hand-built halves."""
        f_gt = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        g = -f_gt
        stacked = common.rectifier_target(f_gt, g)
        assert stacked.shape == (2, 3, 4)
        np.testing.assert_array_equal(stacked[..., :2], f_gt)
        np.testing.assert_array_equal(stacked[..., 2:], g)


class TestTheSegmenterTargetFansOutToEveryHead:
    """Seven deep-supervision heads, seven copies of ONE mask."""

    def test_the_fan_out_count_matches_the_model(self, page_root):
        """SEVEN is the model's, not this module's, and it is checked as such."""
        from dl_techniques.models.vision.image_restoration.doc_scanner.model import (
            create_doc_scanner_segmenter,
        )
        model = create_doc_scanner_segmenter("docscanner-l")
        outputs = model(np.zeros((1, TINY, TINY, 3), dtype="float32"))
        assert len(outputs) == SEGMENTER_HEAD_COUNT

    def test_every_copy_is_the_same_mask(self):
        mask = np.zeros((2, 2, 1), dtype=np.float32)
        fanned = common.segmenter_target(mask)
        assert len(fanned) == SEGMENTER_HEAD_COUNT
        assert all(item is mask for item in fanned)


# ---------------------------------------------------------------------
# tf.data
# ---------------------------------------------------------------------


class TestTheDataset:
    """Batched shapes, the target structure, and the split."""

    def test_the_segmenter_dataset_yields_seven_targets(self, page_root):
        config = _config(page_root, STAGE_SEGMENTER)
        dataset = common.create_dataset(config, list(range(4)), is_training=True)
        images, targets = next(iter(dataset.take(1)))
        assert tuple(images.shape) == (config.batch_size, TINY, TINY, 3)
        assert len(targets) == SEGMENTER_HEAD_COUNT
        for target in targets:
            assert tuple(target.shape) == (config.batch_size, TINY, TINY, 1)

    def test_the_rectifier_dataset_yields_one_four_channel_target(self, page_root):
        config = _config(page_root, STAGE_RECTIFIER)
        dataset = common.create_dataset(config, list(range(4)), is_training=True)
        images, target = next(iter(dataset.take(1)))
        assert tuple(images.shape) == (config.batch_size, TINY, TINY, 3)
        assert tuple(target.shape) == (
            config.batch_size, TINY, TINY, RECTIFIER_TARGET_CHANNELS,
        )

    def test_an_empty_worklist_is_refused_by_name(self, page_root):
        with pytest.raises(ValueError, match="no samples for the"):
            common.create_dataset(
                _config(page_root, STAGE_RECTIFIER), [], is_training=True
            )

    def test_the_split_is_disjoint(self, page_root):
        config = _config(page_root, STAGE_RECTIFIER, synthetic_samples=100)
        train_items, validation_items = common._split_worklist(
            config, list(range(config.synthetic_samples))
        )
        assert set(train_items).isdisjoint(validation_items)
        assert len(train_items) + len(validation_items) == 100
        assert len(validation_items) == 25

    def test_a_degenerate_split_is_refused_rather_than_silently_empty(
        self, page_root
    ):
        config = _config(page_root, STAGE_RECTIFIER, val_split=0.001)
        with pytest.raises(ValueError, match="both sides must be"):
            common._split_worklist(config, list(range(8)))


# ---------------------------------------------------------------------
# objective + optimizer
# ---------------------------------------------------------------------


class TestTheValidationRankDispatch:
    """``training`` selects the rectifier's output RANK; validation sees 4.

    Without :class:`DocScannerRectifierObjective` a rectifier run trains for a
    whole epoch and then dies at the first validation batch. That was MEASURED
    before this class existed, and the third arm below reproduces the raise so
    the dispatch is known to be load-bearing rather than decorative.
    """

    def _batch(self, iterations: int):
        rng = np.random.default_rng(3)
        y_true = rng.uniform(0.0, 8.0, size=(2, 8, 8, 4)).astype("float32")
        y_pred = rng.uniform(0.0, 8.0, size=(2, iterations, 8, 8, 2)).astype(
            "float32"
        )
        return y_true, y_pred

    def test_the_sequence_arm_matches_the_shipped_loss_exactly(self):
        from dl_techniques.losses import DocScannerFlowSequenceLoss

        y_true, y_pred = self._batch(12)
        objective = DocScannerRectifierObjective(iters=12)
        reference = DocScannerFlowSequenceLoss(iters=12)
        assert float(objective(y_true, y_pred)) == pytest.approx(
            float(reference(y_true, y_pred)), rel=0, abs=1e-6
        )

    def test_the_rank_four_arm_is_the_last_iteration_term(self):
        """``gamma ** 0 == 1.0``: the final map's own weight, unscaled."""
        from dl_techniques.losses import DocScannerFlowSequenceLoss

        y_true, sequence = self._batch(12)
        final = sequence[:, -1]
        objective = DocScannerRectifierObjective(iters=12)
        single = DocScannerFlowSequenceLoss(iters=1)
        assert float(objective(y_true, final)) == pytest.approx(
            float(single(y_true, final[:, None])), rel=0, abs=1e-6
        )

    def test_the_shipped_loss_still_refuses_rank_four(self):
        """The guard this class routes AROUND must still be live.

        If someone "simplifies" by relaxing `DocScannerFlowSequenceLoss` to
        accept rank 4, the defect it exists to catch -- a whole training run
        done at `training=False`, i.e. one iteration instead of twelve -- comes
        back silently.
        """
        from dl_techniques.losses import DocScannerFlowSequenceLoss

        y_true, sequence = self._batch(12)
        with pytest.raises(ValueError, match="whole refinement sequence"):
            DocScannerFlowSequenceLoss(iters=12)(y_true, sequence[:, -1])

    def test_the_two_arms_are_not_the_same_number(self):
        """Anti-vacuity: `loss` and `val_loss` really are different scales."""
        y_true, sequence = self._batch(12)
        objective = DocScannerRectifierObjective(iters=12)
        assert float(objective(y_true, sequence)) != pytest.approx(
            float(objective(y_true, sequence[:, -1])), rel=1e-3
        )

    def test_it_round_trips_through_its_config(self):
        objective = DocScannerRectifierObjective(iters=5)
        restored = DocScannerRectifierObjective.from_config(
            objective.get_config()
        )
        assert restored.iters == 5


def _schedule_of(optimizer) -> keras.optimizers.schedules.LearningRateSchedule:
    """The SCHEDULE an optimizer was built with, not its value right now.

    ``optimizer.learning_rate`` is a property that EVALUATES the schedule at
    the optimizer's current step and hands back a scalar tensor, so
    ``optimizer.learning_rate(0)`` is a ``TypeError`` rather than the value at
    step 0. The schedule object itself is the private attribute; reaching for
    it here is deliberate, and asserting on the evaluated scalar instead would
    measure step 0 of every schedule and could not tell a step drop from a
    cosine.

    :param optimizer: A built optimizer.
    :return: Its learning-rate schedule.
    """
    schedule = optimizer._learning_rate
    assert isinstance(
        schedule, keras.optimizers.schedules.LearningRateSchedule
    ), (
        f"the optimizer was built with a bare {type(schedule).__name__} rather "
        "than a schedule -- every recipe in this port carries one"
    )
    return schedule


class TestTheOptimizerRecipes:
    """The two recipes, read off the CONSTRUCTED optimizer.

    ``optimizer_builder`` RENAMES the clipping keys
    (``gradient_clipping_by_norm`` -> ``global_clipnorm``), so a literal
    ``"clipnorm"`` key is silently ignored and the run trains unclipped with no
    error. Asserting on the config dict would pass against exactly that bug.
    """

    def test_the_segmenter_uses_adam(self, page_root):
        optimizer = common.build_optimizer(_config(page_root, STAGE_SEGMENTER))
        assert isinstance(optimizer, keras.optimizers.Adam)
        assert not isinstance(optimizer, keras.optimizers.AdamW)

    def test_the_rectifier_uses_adamw_with_the_configured_decay(self, page_root):
        config = _config(page_root, STAGE_RECTIFIER, weight_decay=3e-4)
        optimizer = common.build_optimizer(config)
        assert isinstance(optimizer, keras.optimizers.AdamW)
        assert float(optimizer.weight_decay) == pytest.approx(3e-4)

    @pytest.mark.parametrize("stage", [STAGE_SEGMENTER, STAGE_RECTIFIER])
    def test_the_clip_reaches_global_clipnorm_not_clipnorm(self, page_root, stage):
        config = _config(page_root, stage, gradient_clipping=0.75)
        optimizer = common.build_optimizer(config)
        assert optimizer.global_clipnorm == pytest.approx(0.75), (
            "the clip did not reach `global_clipnorm`. optimizer_builder maps "
            "`gradient_clipping_by_norm` onto it; a literal `clipnorm` key is "
            "silently ignored and the run trains unclipped"
        )

    def test_the_segmenter_schedule_holds_then_drops(self, page_root):
        """The paper's step drop, not a continuous decay.

        An ``ExponentialDecay(decay_rate=0.1)`` without ``staircase`` has the
        same two numbers in its config and is already an order of magnitude
        down half way to the drop. This asserts the FLAT part, which is what
        separates the two.
        """
        config = _config(
            page_root, STAGE_SEGMENTER, steps_per_epoch=10, lr_drop_epoch=3,
            learning_rate=1e-4, lr_drop_factor=0.1,
        )
        schedule = _schedule_of(common.build_optimizer(config))
        assert float(schedule(0)) == pytest.approx(1e-4)
        assert float(schedule(29)) == pytest.approx(1e-4), (
            "the learning rate moved BEFORE the drop epoch -- that is a "
            "continuous decay wearing the paper's numbers"
        )
        assert float(schedule(31)) == pytest.approx(1e-5)

    def test_the_rectifier_warms_up_linearly_from_near_zero(self, page_root):
        config = _config(
            page_root, STAGE_RECTIFIER, steps_per_epoch=10, epochs=10,
            warmup_epochs=2, learning_rate=1e-4,
        )
        schedule = _schedule_of(common.build_optimizer(config))
        start = float(schedule(0))
        middle = float(schedule(10))
        peak = float(schedule(20))
        assert start < middle < peak
        assert peak == pytest.approx(1e-4, rel=1e-3)
        # Linear: the half-way point of the ramp is half the peak.
        assert middle == pytest.approx(0.5 * peak, rel=1e-2), (
            "the warmup ramp is not linear. THE CURVE IS THIS REPO'S CHOICE "
            "(the paper gives only a length) and it is WarmupSchedule's, which "
            "is linear -- if this changed, the docstring's claim is now false"
        )

    def test_zero_warmup_epochs_starts_at_the_peak(self, page_root):
        """Anti-vacuity for the arm above: the knob is what moves the ramp."""
        config = _config(
            page_root, STAGE_RECTIFIER, steps_per_epoch=10, epochs=10,
            warmup_epochs=0, learning_rate=1e-4,
        )
        schedule = _schedule_of(common.build_optimizer(config))
        assert float(schedule(0)) == pytest.approx(1e-4, rel=1e-3)

    def test_no_layer_carries_a_kernel_regularizer(self, page_root):
        """Never double weight decay: AdamW's decay is the ONLY decay.

        Walked over the BUILT model rather than asserted from the source, so a
        regularizer added inside any sub-layer of either stage is caught.
        """
        config = _config(page_root, STAGE_RECTIFIER)
        model = common.build_model(config)
        model(np.zeros((1, TINY, TINY, 3), dtype="float32"))
        offenders = [
            layer.name
            for layer in model._flatten_layers()
            if getattr(layer, "kernel_regularizer", None) is not None
        ]
        assert not offenders, (
            f"{offenders} carry a kernel_regularizer while the optimizer is "
            "AdamW: the L2 penalty inflates the loss AND the optimizer decays "
            "the same parameter again"
        )


# ---------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------


@pytest.mark.parametrize("stage", [STAGE_SEGMENTER, STAGE_RECTIFIER])
def test_a_short_real_fit_decreases_and_stays_finite(page_root, tmp_path, stage):
    """Stock ``fit()``, both stages, writing ONLY into ``tmp_path``.

    Four epochs of three steps is not a training result and is not read as one.
    What it measures is that the whole chain -- generator, tf.data, model,
    loss, optimizer, callbacks -- composes, that gradients move the loss in the
    right direction, and that nothing goes non-finite at step 0 (the plan's
    named edge case: the composed two-step warp behind ``L_line`` can leave the
    image domain for a badly-initialised network).

    THE KNOBS ARE NOT THE PAPER'S, DELIBERATELY. ``warmup_epochs=0`` and
    ``learning_rate=1e-3`` are set because the paper's own recipe (a 4.8%
    linear warmup at a 1e-4 peak) spends the whole of a 12-step probe at an
    effectively zero learning rate, and a "the loss fell" assertion under those
    knobs would measure noise. ``steps_per_epoch=3`` at ``batch_size=2`` is one
    full pass over the six training samples, so each epoch sees the same set.
    MEASURED before it was written down: 4.82 -> 3.14 (segmenter) and
    714 -> 206 (rectifier).
    """
    config = _config(
        page_root, stage, epochs=4, steps_per_epoch=3, validation_steps=1,
        warmup_epochs=0, learning_rate=1e-3, output_dir=str(tmp_path),
    )
    model, history, results_dir = common.train(config)

    losses = history.history["loss"]
    assert len(losses) == 4
    assert all(np.isfinite(value) for value in losses), losses
    assert all(np.isfinite(value) for value in history.history["val_loss"]), (
        history.history["val_loss"]
    )
    assert losses[-1] < losses[0], (
        f"the {stage} loss did not fall over two epochs: {losses}"
    )

    assert str(tmp_path) in results_dir, (
        f"the run directory {results_dir!r} is outside tmp_path -- a test must "
        "never write into repo-root results/"
    )
    for artifact in ("best_model.keras", "config.json"):
        assert os.path.exists(os.path.join(results_dir, artifact)), artifact


# ---------------------------------------------------------------------
# UVDoc, when it is staged
# ---------------------------------------------------------------------


_UVDOC_STAGED = os.path.exists(common.DEFAULT_UVDOC_ROOT)


@pytest.mark.skipif(
    not _UVDOC_STAGED,
    reason=f"UVDoc is not staged at {common.DEFAULT_UVDOC_ROOT}",
)
class TestTheUvdocArm:
    """The other corpus, when the 27.5 GB archive is present.

    Skipped rather than mocked: the whole point of this arm is that the REAL
    archive's members densify into the SAME contract the synthetic generator
    emits, and a mock would assert that against itself.
    """

    def test_one_render_matches_the_same_contract_as_a_synthetic_sample(self):
        config = replace(
            common.stage_defaults(STAGE_RECTIFIER),
            data_source=common.SOURCE_UVDOC,
            image_size=TINY,
            max_geometries=4,
        )
        ids = common.collect_uvdoc_sample_ids(config)
        assert ids
        image, target = common.uvdoc_sample(config, ids, 0)
        assert image.shape == (TINY, TINY, 3)
        assert target.shape == (TINY, TINY, RECTIFIER_TARGET_CHANNELS)
        assert_is_backward_map(target[..., :2], TINY, TINY, "uvdoc f_gt")
        assert_is_forward_map(target[..., 2:], TINY, TINY, "uvdoc g")

    def test_the_geometry_cache_makes_the_second_read_cheap(self):
        """The 6h-vs-1.2h fact, measured rather than asserted from a comment."""
        import time

        config = replace(
            common.stage_defaults(STAGE_RECTIFIER),
            data_source=common.SOURCE_UVDOC,
            image_size=TINY,
            max_geometries=4,
        )
        ids = common.collect_uvdoc_sample_ids(config)
        with common.UVDocSource(config.uvdoc_root) as source:
            name = source.geometry_for_sample(ids[0])

        common._cached_geometry.cache_clear()
        start = time.perf_counter()
        common.cached_uvdoc_geometry(config, name)
        cold = time.perf_counter() - start

        start = time.perf_counter()
        common.cached_uvdoc_geometry(config, name)
        warm = time.perf_counter() - start

        assert warm < 0.1 * cold, (
            f"the cached read took {warm:.4f}s against a cold {cold:.4f}s -- "
            "the geometry cache is not being hit, and an epoch over 20,000 "
            "renders pays 20,000 densifications instead of 4,032"
        )

"""The segmentation verdict is STRICTLY above the majority-class baseline, in one dtype (D-087).

Review-iter-3 concern 1. The constant majority-class predictor IS the baseline, so a verdict that
lets it through is the failure the bar exists to prevent. ``run_sequence.report`` compared Keras'
``float32`` ``val_miou`` with the ``float64`` baseline by ``>=``, and the production
``MeanIoU`` of a model that predicts the majority class everywhere was judged a success in 8 of 20
random selection sets (for example 0.18360255658626556 against 0.18360254786036034, one ``float32`` step
apart).

What this file holds:

* the constant majority model, scored by the run's own ``segmentation_metrics`` on 20 random selection
  sets and 4 class counts, is NEVER a success; a model that is right everywhere is;
* the boundary, walked in ``float32`` steps of the baseline: the baseline itself, a float64 value a hair
  above it, and ``TIE_ULPS`` steps above are NO; one step more is YES; a baseline of 1.0 cannot be
  beaten and a ``NaN`` is not above anything;
* the hook: the segmentation task uses the strict rule against the baseline and the shared ``>=`` for a
  bar the caller stated; the base class keeps ``>=`` (so a value equal to a stated bar reaches it);
* a finished run says it in its notes, and its verdict is the function's answer on its own numbers;
* the bar comes from the SELECTION split: with a test split whose majority class is another, the
  summary's ``majority_class_baseline`` is the selection split's (a baseline worked out from the test
  pairs would let test data set the verdict).

The tests write only under ``tmp_path``.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from train.api.tasks.image_segmentation import ImageSegmentationTrainer
from train.api.trainer import TaskTrainer
from train.common import segmentation_run as sr

from . import _segmentation_fixtures as fx
from .test_image_segmentation_task import SIZE, world  # noqa: F401  (the registry fixture)


def constant_majority_scores(seed: int, classes: int):
    """The production ``MeanIoU`` of a model that says the majority class everywhere, and the baseline."""
    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet(np.ones(classes) * 2.0)
    masks = rng.choice(classes, size=(37, 16, 16), p=probabilities).astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((np.zeros((37, 1), np.float32), masks)).batch(16)
    baseline = sr.majority_class_baseline(
        sr.class_pixel_counts(dataset, classes, None), [str(i) for i in range(classes)])
    _, miou = sr.segmentation_metrics(classes, None)
    for start in range(0, 37, 16):
        chunk = masks[start:start + 16]
        logits = np.zeros(chunk.shape + (classes,), np.float32)
        logits[..., baseline["class"]] = 5.0
        miou.update_state(chunk, logits)
    return float(miou.result()), baseline["miou"]


@pytest.mark.parametrize("classes", [2, 3, 21, 150])
def test_the_constant_majority_model_is_never_a_success(classes):
    verdicts, apart = [], []
    for seed in range(20):
        reached, baseline = constant_majority_scores(seed, classes)
        apart.append(abs(reached - baseline))
        verdicts.append(sr.beats_baseline(reached, baseline))
    assert not any(verdicts), "a model that predicts the majority class everywhere passed the bar"
    # The control that makes the test mean something: the two figures DO differ (float32 against
    # float64), so a bare `reached >= baseline` is a coin flip on this input.
    assert max(apart) > 0.0


def test_the_unfixed_comparison_would_have_passed_it_in_some_cases():
    """The reproduction of the review: `reached >= baseline` is true for some constant models (classes=3)."""
    scores = [constant_majority_scores(seed, 3) for seed in range(40)]
    assert sum(reached >= baseline for reached, baseline in scores) > 0


def test_a_model_that_is_right_everywhere_is_a_success():
    assert sr.beats_baseline(1.0, 0.1836025565) is True
    assert sr.beats_baseline(0.2, 0.1836025565) is True


@pytest.mark.parametrize("baseline", [0.0002, 0.0558805, 0.1836025478, 0.5, 0.9])
def test_the_boundary_in_float32_steps_of_the_baseline(baseline):
    base32 = np.float32(baseline)
    step = float(np.spacing(base32))
    edge = float(base32) + sr.TIE_ULPS * step  # exact in float32
    assert sr.beats_baseline(baseline, baseline) is False
    assert sr.beats_baseline(float(base32), baseline) is False
    assert sr.beats_baseline(float(base32) + step, baseline) is False
    assert sr.beats_baseline(edge, baseline) is False, "the band's own edge is a tie"
    assert sr.beats_baseline(edge + step, baseline) is True, "one step past the band is above"
    assert sr.beats_baseline(baseline - 10 * step, baseline) is False


def test_the_two_figures_of_the_reviews_failing_case_are_one_float32_step_apart_and_a_no():
    reached, baseline = 0.18360255658626556, 0.18360254786036034
    assert reached > baseline, "float64 sees `reached` above"
    assert np.float32(reached) > np.float32(baseline), "and so does a bare float32 comparison"
    assert sr.beats_baseline(reached, baseline) is False


def test_a_baseline_of_one_cannot_be_beaten_and_nan_is_not_above_anything():
    assert sr.beats_baseline(1.0, 1.0) is False
    assert sr.beats_baseline(float("nan"), 0.2) is False
    assert sr.beats_baseline(0.9, float("nan")) is False


def stub_config(threshold):
    return SimpleNamespace(metric_args=SimpleNamespace(success_threshold=threshold))


def test_the_segmentation_task_uses_the_strict_rule_for_the_baseline_and_the_shared_one_for_a_stated_bar():
    trainer = object.__new__(ImageSegmentationTrainer)
    baseline = 0.1836025478
    assert trainer.bar_is_met(stub_config(None), baseline, baseline) is False
    assert trainer.bar_is_met(stub_config(None), 0.5, baseline) is True
    # A bar the caller stated is a bar to REACH: equal reaches it.
    assert trainer.bar_is_met(stub_config(0.5), 0.5, 0.5) is True
    assert trainer.bar_is_met(stub_config(0.5), 0.4999, 0.5) is False


def test_the_shared_default_keeps_reaching_the_bar_a_success():
    trainer = object.__new__(TaskTrainer)
    assert trainer.bar_is_met(None, 0.5, 0.5) is True
    assert trainer.bar_is_met(None, 0.4999, 0.5) is False
    assert trainer.bar_is_met(None, 0.5001, 0.5) is True


# ---------------------------------------------------------------------
# A finished run: the notes, the verdict, and the split the bar comes from
# ---------------------------------------------------------------------

def other_majority_world(root):
    """A train set whose majority class is 0 and a TEST set whose majority class is 1."""
    def write(prefix, count, widths):
        images, masks = root / f"{prefix}images", root / f"{prefix}masks"
        for index in range(count):
            mask = np.zeros((SIZE, SIZE), np.uint8)
            edge = 0
            for value, width in enumerate(widths):
                mask[:, edge:edge + width] = value
                edge += width
            fx.write_pair(images, masks, f"p{index:03d}", mask)
        return images, masks

    return {"clean": (write("train_", 40, (20, 8, 4)), write("test_", 8, (4, 24, 4)))}


def test_the_bar_is_the_selection_splits_not_the_test_splits(world, tmp_path):  # noqa: F811
    from train.api.tasks.image_segmentation import train_image_segmentation
    from . import test_image_segmentation_task as base

    folders = other_majority_world(tmp_path / "data")
    (train_images, train_masks), (test_images, test_masks) = folders["clean"]
    result = train_image_segmentation(
        out=str(tmp_path / "run"), model="convunext/tiny", epochs=1, batch_size=8,
        data=base.data_of(folders), transform_args={"image_size": SIZE},
        analysis_args={"model_analysis": False, "figures": False})
    summary = json.loads(result.summary_path.read_text())
    baseline = summary["majority_class_baseline"]
    assert baseline["class"] == 0, "the selection split's majority class, not the test split's (class 1)"
    assert baseline["miou"] == pytest.approx(0.625 / 3, abs=1e-12)
    assert summary["success_threshold"] == baseline["miou"]
    assert summary["success"] is sr.beats_baseline(summary["success_value"], summary["success_threshold"])
    notes = " ".join(summary["notes"])
    assert "STRICTLY above that baseline" in notes and f"{sr.TIE_ULPS} float32 steps" in notes


# ---------------------------------------------------------------------
# The sequence asks the trainer
# ---------------------------------------------------------------------

def test_the_run_sequence_takes_the_verdict_from_the_trainers_hook(tmp_path):
    """`run_sequence.report` must not compare the numbers itself: the hook is where a task says "above"."""
    from train.api.tasks.image_classification import ImageClassificationTrainer

    from tests.classification_fakes import use_tiny_raw

    from .test_task_trainer import make_config

    class Never(ImageClassificationTrainer):
        def bar_is_met(self, config, reached, threshold):
            return False

    stated = {"metric_args.success_threshold": 0.0}
    with pytest.MonkeyPatch.context() as patch:
        from train.api import recipes, registry

        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        plain = ImageClassificationTrainer(make_config(tmp_path / "plain", epochs=1, **stated)).run()
        never = Never(make_config(tmp_path / "never", epochs=1, **stated)).run()
    assert plain.success is True, "control: a bar of 0.0 is reached by any run, under the shared >= rule"
    assert never.success is False
    assert json.loads(never.summary_path.read_text())["success"] is False

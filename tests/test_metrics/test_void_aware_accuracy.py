"""Tests for VoidAwareAccuracy: pixel accuracy that leaves void pixels out.

The fixture is two 2x2 images of two classes with two void pixels (label 255). The probabilities
are written down, so every expectation is a hand count: 6 valid pixels, 4 of them right.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.metrics.void_aware_accuracy import VoidAwareAccuracy

VOID = 255
#: ``(2, 2, 2, 2)``: image, row, column, class probability. The void pixels' rows are arbitrary.
PROBS = np.array(
    [
        [[[0.8, 0.2], [0.4, 0.6]], [[0.5, 0.5], [0.3, 0.7]]],
        [[[0.1, 0.9], [0.5, 0.5]], [[0.6, 0.4], [0.9, 0.1]]],
    ],
    dtype="float32",
)
#: Image 0: right, right, void, WRONG (class 0 predicted 1). Image 1: right, void, WRONG, right.
LABELS = np.array([[[0, 1], [VOID, 0]], [[1, VOID], [1, 0]]], dtype="int32")
VOID_AT = LABELS == VOID


def score(logits, labels=LABELS, **kwargs):
    metric = VoidAwareAccuracy(VOID, **kwargs)
    metric.update_state(labels, logits)
    return float(metric.result())


class TestVoidAwareAccuracy:
    def test_it_counts_the_valid_pixels_only(self):
        assert score(PROBS) == pytest.approx(4 / 6)

    def test_the_stock_metric_counts_each_void_pixel_as_wrong_which_is_why_this_exists(self):
        stock = keras.metrics.SparseCategoricalAccuracy()
        stock.update_state(LABELS, PROBS)
        assert float(stock.result()) == pytest.approx(4 / 8)

    def test_it_is_named_accuracy_so_a_run_with_and_without_void_logs_one_key(self):
        assert VoidAwareAccuracy(VOID).name == "accuracy"
        assert VoidAwareAccuracy(VOID, name="pixel_accuracy").name == "pixel_accuracy"

    @pytest.mark.parametrize("scale", [1.0, 1e3, 1e9])
    def test_what_the_model_says_at_a_void_pixel_changes_nothing(self, scale):
        rng = np.random.default_rng(0)
        for _ in range(20):
            logits = PROBS.copy()
            logits[VOID_AT] = rng.normal(size=(int(VOID_AT.sum()), 2)).astype("float32") * scale
            assert score(logits) == pytest.approx(4 / 6)

    def test_a_void_label_that_equals_the_argmax_is_not_counted_as_right(self):
        labels = np.array([[[1, 0]]], dtype="int32")  # ignore_index 0: the second pixel is void
        logits = np.array([[[[0.0, 1.0], [1.0, 0.0]]]], dtype="float32")  # argmax 1 and 0
        metric = VoidAwareAccuracy(0)
        metric.update_state(labels, logits)
        assert float(metric.result()) == 1.0
        assert float(metric.valid) == 1.0

    def test_a_fully_void_batch_gives_zero_and_not_nan_and_does_not_poison_later_batches(self):
        metric = VoidAwareAccuracy(VOID)
        metric.update_state(np.full((2, 2, 2), VOID, "int32"), PROBS)
        assert float(metric.result()) == 0.0
        metric.update_state(LABELS, PROBS)
        assert float(metric.result()) == pytest.approx(4 / 6)

    def test_it_pools_pixels_over_batches_and_is_not_a_mean_of_batch_means(self):
        metric = VoidAwareAccuracy(VOID)
        metric.update_state(LABELS[:1], PROBS[:1])  # 2 right of 3
        metric.update_state(LABELS[1:], PROBS[1:])  # 2 right of 3
        assert float(metric.result()) == pytest.approx(4 / 6)
        uneven = VoidAwareAccuracy(VOID)
        one_right = np.array([[[0, VOID], [VOID, VOID]]], dtype="int32")
        uneven.update_state(one_right, PROBS[:1])  # 1 pixel, right
        uneven.update_state(LABELS[1:], PROBS[1:])  # 3 pixels, 2 right
        assert float(uneven.result()) == pytest.approx(3 / 4), "a mean of batch means would be 5/6"

    def test_a_sample_weight_per_image_or_per_pixel_multiplies_the_count(self):
        metric = VoidAwareAccuracy(VOID)
        metric.update_state(LABELS, PROBS, sample_weight=np.array([1.0, 0.0], "float32"))
        assert float(metric.result()) == pytest.approx(2 / 3)
        pixel = np.ones(LABELS.shape, "float32")
        pixel[0, 0, 0] = 0.0
        pixel[VOID_AT] = 7.0  # a void pixel stays out whatever its weight is
        metric = VoidAwareAccuracy(VOID)
        metric.update_state(LABELS, PROBS, sample_weight=pixel)
        assert float(metric.result()) == pytest.approx(3 / 5)

    def test_it_takes_labels_with_a_trailing_axis_and_any_integer_or_float_dtype(self):
        for labels in (LABELS[..., None], LABELS.astype("uint8"), LABELS.astype("float32")):
            assert score(PROBS, labels=labels) == pytest.approx(4 / 6)

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float64"])
    def test_it_takes_logits_of_any_float_dtype(self, dtype):
        assert score(tf.cast(np.log(PROBS), dtype)) == pytest.approx(4 / 6)

    def test_it_runs_under_xla(self):
        metric = VoidAwareAccuracy(VOID)

        @tf.function(jit_compile=True)
        def step(labels, logits):
            metric.update_state(labels, logits)
            return metric.result()

        assert float(step(LABELS, PROBS)) == pytest.approx(4 / 6)

    def test_reset_state_forgets_every_batch(self):
        metric = VoidAwareAccuracy(VOID)
        metric.update_state(LABELS, PROBS)
        metric.reset_state()
        assert float(metric.correct) == 0.0 and float(metric.valid) == 0.0
        metric.update_state(LABELS, PROBS)
        assert float(metric.result()) == pytest.approx(4 / 6), "the first batch is not carried into the next epoch"

    def test_the_config_round_trips_with_the_ignore_index(self):
        rebuilt = VoidAwareAccuracy.from_config(VoidAwareAccuracy(VOID, name="pixel_accuracy").get_config())
        assert rebuilt.ignore_index == VOID and rebuilt.name == "pixel_accuracy"

    def test_it_is_registered_under_its_module_path(self):
        assert (
            keras.saving.get_registered_name(VoidAwareAccuracy)
            == "dl_techniques.metrics.void_aware_accuracy>VoidAwareAccuracy"
        )

    def test_a_compiled_model_with_it_saves_and_reloads(self, tmp_path):
        model = keras.Sequential([keras.layers.Input((2, 2, 2)), keras.layers.Identity()])
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=VOID),
            metrics=[VoidAwareAccuracy(VOID)],
        )
        model.save(tmp_path / "m.keras")
        reloaded = keras.models.load_model(tmp_path / "m.keras")
        scores = reloaded.evaluate(PROBS, LABELS, batch_size=2, verbose=0, return_dict=True)
        assert scores["accuracy"] == pytest.approx(4 / 6)

"""Tests for HalfPrecisionTopKAccuracy: the stock top-k metric for half-precision logits."""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.metrics.half_precision_top_k import HalfPrecisionTopKAccuracy

LABELS = np.array([1, 2, 0, 3], dtype="int32")
LOGITS = np.array(
    [
        [0.1, 0.9, 0.0, 0.0],  # label 1: top-1
        [0.5, 0.4, 0.1, 0.0],  # label 2: rank 3
        [0.2, 0.7, 0.1, 0.0],  # label 0: rank 2
        [0.1, 0.2, 0.3, 0.4],  # label 3: top-1
    ],
    dtype="float32",
)


class TestHalfPrecisionTopKAccuracy:
    def test_the_stock_metric_rejects_half_precision_which_is_why_this_exists(self):
        stock = keras.metrics.SparseTopKCategoricalAccuracy(k=2)
        with pytest.raises((TypeError, tf.errors.InvalidArgumentError), match="InTopKV2"):
            stock.update_state(LABELS, LOGITS.astype("float16"))

    @pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
    def test_it_scores_logits_of_any_float_dtype(self, dtype):
        metric = HalfPrecisionTopKAccuracy(k=2, name="top2_accuracy")
        metric.update_state(LABELS, tf.cast(LOGITS, dtype))
        # rows 0, 2 and 3 have the label in their top two; row 1 has it at rank 3.
        assert float(metric.result()) == 0.75

    def test_it_equals_the_stock_metric_on_float32(self):
        ours = HalfPrecisionTopKAccuracy(k=3)
        stock = keras.metrics.SparseTopKCategoricalAccuracy(k=3)
        for metric in (ours, stock):
            metric.update_state(LABELS, LOGITS)
        assert float(ours.result()) == float(stock.result()) == 1.0

    def test_a_sample_weight_is_honoured(self):
        metric = HalfPrecisionTopKAccuracy(k=1)
        weights = np.array([1.0, 1.0, 1.0, 0.0], dtype="float32")
        metric.update_state(LABELS, LOGITS.astype("float16"), sample_weight=weights)
        np.testing.assert_allclose(float(metric.result()), 1.0 / 3.0, rtol=1e-6)

    def test_reset_state_clears_the_count(self):
        metric = HalfPrecisionTopKAccuracy(k=1)
        metric.update_state(LABELS, LOGITS)
        metric.reset_state()
        assert float(metric.result()) == 0.0

    def test_the_config_round_trips_with_name_and_k(self):
        metric = HalfPrecisionTopKAccuracy(k=5, name="top5_accuracy")
        rebuilt = HalfPrecisionTopKAccuracy.from_config(metric.get_config())
        assert rebuilt.name == "top5_accuracy" and rebuilt.k == 5

    def test_it_is_registered_under_its_module_path(self):
        assert (
            keras.saving.get_registered_name(HalfPrecisionTopKAccuracy)
            == "dl_techniques.metrics.half_precision_top_k>HalfPrecisionTopKAccuracy"
        )

    def test_a_compiled_model_with_it_saves_and_reloads(self, tmp_path):
        model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(10)])
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[HalfPrecisionTopKAccuracy(k=5, name="top5_accuracy")],
        )
        model.train_on_batch(np.zeros((4, 4), "float32"), np.zeros(4, "int32"))
        model.save(tmp_path / "m.keras")
        reloaded = keras.models.load_model(tmp_path / "m.keras")
        scores = reloaded.evaluate(
            np.zeros((4, 4), "float32"), np.zeros(4, "int32"), verbose=0, return_dict=True
        )
        assert set(scores) == {"loss", "top5_accuracy"}

"""Per-pixel calibration under 16-bit precision, one-channel masks, subclassed models, and the draw.

Plan 2faf890a step 14.3, review-iter-1 concerns 3, 13, 14, 15 and the three survivors of the
step-12 mutants (``an_rng_each_batch``, ``an_cap_check_off``, ``an_activation_each_batch``).

* Concern 3. Under ``mixed_float16`` a 21-class (and 150-class) mask model failed calibration with
  "a multi-label head needs a (n_samples, 21) indicator matrix": the float16 softmax row-sum error
  exceeds the 1e-3 tolerance of ``is_multilabel_output``, so a softmax was read as independent
  sigmoids. ``mixed_bfloat16`` failed on both the per-pixel and the rank-2 path (row-sum error
  0.0078). The scores are now read in float32, a 16-bit softmax gets a tolerance of a few epsilons
  of its dtype and is renormalised in float32. A float32 model's numbers do not move.
* Concern 13. The activation was decided from the first batch of 16 images only. A logits model
  whose first batch scored inside [0, 1] was never softmaxed (Brier 13.2, impossible). It is now
  decided from every pixel of every batch, the way a rank-2 model is decided from all its rows.
* Concern 14. A single-channel mask ``(N, H, W, 1)`` (sigmoid or logits) is a binary mask: it
  becomes ``[1 - p, p]`` per pixel. Checked against the hand fixture of the two-class tests.
* Concern 15. A subclassed model has no static ``output_shape``; its rank is read from one real
  call when a sample is given. A rank-2 subclassed model is unchanged.
"""

import keras
import numpy as np
import pytest

from dl_techniques.analyzer import DataInput, ModelAnalyzer
from dl_techniques.analyzer.utils import (
    OutputStatistics, classify_output_activation, reduced_precision_epsilon, to_full_precision)
from tests.test_analyzer.test_per_pixel_calibration import (
    HAND, P1, VOID, Y_CLEAN, Y_VOID, _cache_entry, _calibration_config, _x)


def _analyze(tmp_path, model, x, y, **config):
    analyzer = ModelAnalyzer({"m": model}, _calibration_config(**config), output_dir=str(tmp_path / "out"))
    return analyzer, analyzer.analyze(data=DataInput(x_data=x, y_data=y))


def _reasons(results):
    return results.analysis_failures


def _expected_numbers(p_correct_free_probs, labels):
    """ECE-free reference: Brier of ``probs`` (P, C) against integer ``labels`` (P,), by a plain loop."""
    total = 0.0
    for row, label in zip(p_correct_free_probs, labels):
        total += sum((float(v) - (1.0 if c == label else 0.0)) ** 2 for c, v in enumerate(row))
    return total / len(labels)


# ---------------------------------------------------------------------
# concern 3: 16-bit precision
# ---------------------------------------------------------------------

def _mask_model(classes, policy, final=None, size=16):
    keras.utils.set_random_seed(0)
    inputs = keras.Input((size, size, 3))
    h = keras.layers.Conv2D(8, 3, padding="same", activation="relu", dtype=policy)(inputs)
    outputs = keras.layers.Conv2D(classes, 1, activation=final, dtype=policy)(h)
    model = keras.Model(inputs, outputs, name="m")
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=final is None))
    return model


def _mask_images(classes, n=32, size=16):
    x = np.random.default_rng(0).random((n, size, size, 3), dtype=np.float32)
    return x, np.clip((x[..., 0] * classes).astype(np.int32), 0, classes - 1)


@pytest.mark.parametrize("policy", ["mixed_float16", "mixed_bfloat16"])
@pytest.mark.parametrize("final", [None, "softmax"])
@pytest.mark.parametrize("classes", [3, 21, 150])
def test_a_16_bit_mask_model_is_calibrated_and_matches_its_float32_twin(tmp_path, policy, final, classes):
    x, y = _mask_images(classes)
    model = _mask_model(classes, policy, final)
    assert reduced_precision_epsilon(model.predict(x[:2], verbose=0).dtype) is not None, "not a 16-bit output"
    _, results = _analyze(tmp_path / "low", model, x, y)
    assert _reasons(results) == {}, _reasons(results)
    low = results.calibration_metrics["m"]

    twin = _mask_model(classes, "float32", final)
    twin.set_weights(model.get_weights())
    _, ref = _analyze(tmp_path / "full", twin, x, y)
    full = ref.calibration_metrics["m"]
    assert low["ece"] == pytest.approx(full["ece"], abs=0.02)
    assert low["brier_score"] == pytest.approx(full["brier_score"], abs=0.02)


def test_the_float16_softmax_row_sum_error_is_beyond_the_old_tolerance():
    """The premise of the fix, measured: a float16 softmax head of 150 classes is off by more than 1e-3."""
    model = _mask_model(150, "mixed_float16", "softmax")
    x, _ = _mask_images(150)
    out = model.predict(x, verbose=0)
    assert out.dtype == np.float16
    worst = float(np.abs(out.astype(np.float32).sum(axis=-1) - 1.0).max())
    assert worst > 1e-3, worst
    assert worst < 4 * reduced_precision_epsilon(out.dtype)
    kind, _ = classify_output_activation(out)
    assert kind == "softmax"


def test_a_16_bit_softmax_is_renormalised_a_16_bit_sigmoid_is_not():
    rng = np.random.default_rng(0)
    softmax = np.exp(rng.normal(size=(500, 40)))
    softmax = (softmax / softmax.sum(-1, keepdims=True)).astype(np.float16)
    sigmoid = (1.0 / (1.0 + np.exp(-rng.normal(size=(500, 40))))).astype(np.float16)
    assert classify_output_activation(softmax)[0] == "softmax"
    assert classify_output_activation(sigmoid)[0] == "sigmoid"
    assert classify_output_activation(rng.normal(size=(500, 40)).astype(np.float16))[0] == "logits"


@pytest.mark.parametrize("policy", ["mixed_float16", "mixed_bfloat16"])
@pytest.mark.parametrize("classes", [10, 1000])
def test_a_16_bit_rank_2_model_is_calibrated(tmp_path, policy, classes):
    """The pre-existing rank-2 bfloat16 failure (row-sum error 0.0078) is fixed with the same cast."""
    keras.utils.set_random_seed(0)
    inputs = keras.Input((16, 16, 3))
    h = keras.layers.GlobalAveragePooling2D()(keras.layers.Conv2D(8, 3, activation="relu", dtype=policy)(inputs))
    model = keras.Model(inputs, keras.layers.Dense(classes, dtype=policy)(h), name="m")
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    x = np.random.default_rng(0).random((32, 16, 16, 3), dtype=np.float32)
    y = np.random.default_rng(1).integers(0, classes, 32)
    _, results = _analyze(tmp_path, model, x, y)
    assert _reasons(results) == {}, _reasons(results)
    assert np.isfinite(results.calibration_metrics["m"]["ece"])


def test_a_float32_array_is_passed_through_as_the_same_object():
    array = np.random.default_rng(0).random((4, 3), dtype=np.float32)
    assert to_full_precision(array) is array
    assert to_full_precision(array.astype(np.float16)).dtype == np.float32


# ---------------------------------------------------------------------
# concern 13: the activation is decided from every batch
# ---------------------------------------------------------------------

def _identity_model(shape):
    inputs = keras.Input(shape)
    model = keras.Model(inputs, keras.layers.Activation("linear")(inputs), name="m")
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


@pytest.mark.parametrize("inside", ["first", "last"])
def test_a_first_batch_inside_the_unit_interval_cannot_decide_for_the_rest(tmp_path, inside):
    """Logits model whose first (or last) batch scores in [0, 0.2): that batch alone reads as sigmoid.

    40 images are the batches [0:16], [16:32], [32:40]. Deciding from the first batch fails the
    ``first`` arrangement, deciding from the last one fails ``last``, and deciding from each batch
    softmaxes some batches and not others.
    """
    rng = np.random.default_rng(0)
    x = rng.random((40, 8, 8, 3), dtype=np.float32) * 5.0
    if inside == "first":
        x[:16] *= 0.04
    else:
        x[32:] *= 0.04
    y = (x[..., 0] > 0.5).astype(np.int32)
    _, results = _analyze(tmp_path, _identity_model((8, 8, 3)), x, y)
    cal = results.calibration_metrics["m"]

    logits = x.reshape(-1, 3).astype(np.float64)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    reference = _expected_numbers(probs, y.reshape(-1))
    assert cal["brier_score"] <= 2.0
    assert cal["brier_score"] == pytest.approx(reference, abs=1e-5)


def test_output_statistics_over_batches_equal_one_call_on_the_whole_array():
    rng = np.random.default_rng(0)
    softmax = np.exp(rng.normal(size=(300, 7)))
    arrays = {
        "softmax": softmax / softmax.sum(-1, keepdims=True),
        "sigmoid": 1.0 / (1.0 + np.exp(-rng.normal(size=(300, 7)))),
        "logits": rng.normal(size=(300, 7)) * 3,
        "logits_inside_unit_in_first_rows": np.concatenate(
            [rng.random((100, 7)) * 0.1, rng.normal(size=(200, 7)) * 3]),
    }
    for name, array in arrays.items():
        statistics = OutputStatistics()
        for start in range(0, 300, 64):
            statistics.update(array[start:start + 64])
        assert statistics.classify() == classify_output_activation(array), name


@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
def test_a_multi_channel_sigmoid_mask_is_refused_with_its_true_cause(tmp_path, policy):
    """Independent per-class probabilities with class-index masks: say so, not 'needs an indicator matrix'."""
    x, y = _mask_images(3)
    _, results = _analyze(tmp_path, _mask_model(3, policy, "sigmoid"), x, y)
    reason = _reasons(results)["calibration"]["m"]
    assert "independent per-class probabilities" in reason and "do not sum to 1" in reason
    assert "indicator matrix" not in reason


def test_a_pinned_logits_activation_overrides_what_the_values_suggest(tmp_path):
    """The remedy the message names: scores in [0, 1] that are really logits are softmaxed when pinned."""
    x, y = _mask_images(3)
    _, results = _analyze(tmp_path, _mask_model(3, "float32", "sigmoid"), x, y, output_activation="logits")
    assert _reasons(results) == {}


# ---------------------------------------------------------------------
# concern 14: a single-channel mask is a binary mask
# ---------------------------------------------------------------------

def _logit(p):
    return np.log(p / (1.0 - p)).astype(np.float32)


@pytest.mark.parametrize("form", ["sigmoid", "logits"])
@pytest.mark.parametrize("tag, y", [("clean", Y_CLEAN), ("void", Y_VOID)])
@pytest.mark.parametrize("trailing", [False, True])
def test_a_single_channel_mask_equals_the_two_class_hand_computation(tmp_path, form, tag, y, trailing):
    x = _x(P1) if form == "sigmoid" else _logit(_x(P1))
    labels = y[..., None] if trailing else y
    _, results = _analyze(tmp_path, _identity_model((2, 2, 1)), x, labels, ignore_label=VOID)
    assert _reasons(results) == {}, _reasons(results)
    cal = results.calibration_metrics["m"]
    hand = HAND[tag]
    assert cal["ece"] == pytest.approx(hand["ece"], abs=1e-5)
    assert cal["brier_score"] == pytest.approx(hand["brier"], abs=1e-5)
    assert (cal["pixels_total"], cal["pixels_void"], cal["pixels_used"]) == (
        hand["total"], hand["void"], hand["used"])


def test_a_single_channel_label_outside_zero_and_one_is_refused_naming_the_range(tmp_path):
    labels = Y_CLEAN.copy()
    labels[0, 0, 0] = 2
    _, results = _analyze(tmp_path, _identity_model((2, 2, 1)), _x(P1), labels)
    reason = _reasons(results)["calibration"]["m"]
    assert "[0, 2)" in reason


def test_the_label_shape_error_prints_the_arrays_not_one_batch():
    """40 images are three batches: the message must name (40, ...) shapes, never a 16-image batch."""
    analyzer = ModelAnalyzer.__new__(ModelAnalyzer)
    analyzer.config = _calibration_config()
    analyzer._multi_input_models = set()

    class Stub:
        output_shape = (None, 2, 2, 1)

        def predict(self, x, verbose=0):
            return np.asarray(x)

    x = np.tile(P1, (20, 1, 1))[..., None]
    with pytest.raises(ValueError) as error:
        analyzer._cache_pixel_predictions("stub", Stub(), x, np.zeros((40, 2, 3), dtype=np.int32))
    text = str(error.value)
    assert "(40, 2, 3)" in text and "(40, 2, 2, 1)" in text and "(40, 2, 2)" in text
    assert "(16," not in text


# ---------------------------------------------------------------------
# concern 15: a subclassed model
# ---------------------------------------------------------------------

class SubclassedMask(keras.Model):
    def __init__(self, classes=3):
        super().__init__(name="m")
        self.head = keras.layers.Conv2D(classes, 1)

    def call(self, inputs, training=None):
        return self.head(inputs)


class SubclassedClassifier(keras.Model):
    def __init__(self, classes=3):
        super().__init__(name="m")
        self.pool = keras.layers.GlobalAveragePooling2D()
        self.dense = keras.layers.Dense(classes)

    def call(self, inputs, training=None):
        return self.dense(self.pool(inputs))


def _built(model, x):
    model(x[:1])
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


def test_a_subclassed_mask_model_is_per_pixel_when_a_sample_is_given():
    x, _ = _mask_images(3, n=4)
    model = _built(SubclassedMask(), x)
    with pytest.raises(AttributeError):
        model.output_shape  # the reason it was not recognised
    assert ModelAnalyzer.outputs_per_pixel(model) is False
    assert ModelAnalyzer.outputs_per_pixel(model, sample=x[:1]) is True


def test_a_subclassed_rank_2_model_stays_on_the_original_path():
    x, _ = _mask_images(3, n=4)
    model = _built(SubclassedClassifier(), x)
    assert ModelAnalyzer.outputs_per_pixel(model, sample=x[:1]) is False


def test_a_sample_the_model_cannot_take_is_not_per_pixel_and_does_not_raise():
    x, _ = _mask_images(3, n=4)
    model = _built(SubclassedMask(), x)
    assert ModelAnalyzer.outputs_per_pixel(model, sample=np.zeros((1, 5), dtype=np.float32)) is False


def test_a_static_shape_is_never_overridden_by_the_sample():
    inputs = keras.Input((4, 4, 3))
    functional = keras.Model(inputs, keras.layers.Conv2D(3, 1)(inputs))
    assert ModelAnalyzer.outputs_per_pixel(functional, sample=np.zeros((1, 5), dtype=np.float32)) is True


def test_the_analyzer_serves_a_subclassed_mask_model_through_the_pixel_path(tmp_path):
    x, y = _mask_images(3, n=20)
    analyzer, results = _analyze(tmp_path, _built(SubclassedMask(), x), x, y)
    assert _reasons(results) == {}, _reasons(results)
    cal = results.calibration_metrics["m"]
    assert cal["pixels_used"] == 20 * 16 * 16 and "m" in analyzer._per_pixel_models


def test_a_subclassed_rank_2_model_is_analyzed_as_before(tmp_path):
    x, _ = _mask_images(3, n=20)
    y = np.random.default_rng(0).integers(0, 3, 20)
    analyzer, results = _analyze(tmp_path, _built(SubclassedClassifier(), x), x, y)
    assert _reasons(results) == {}, _reasons(results)
    assert "pixels_used" not in results.calibration_metrics["m"] and not analyzer._per_pixel_models


# ---------------------------------------------------------------------
# the survivors of the step-12 mutants
# ---------------------------------------------------------------------

def test_the_draw_gives_each_batch_its_own_keys():
    """Mutant ``an_rng_each_batch``: the same keys in every batch tie the pixel at one position in
    batch 0 with the pixel at that position in batch 1, so they are kept TOGETHER. The marginal
    chance of each pixel stays 1 in 4 (the existing uniformity test cannot see it); the JOINT
    chance of such a pair does not: uniform without replacement gives (40/160) * (39/159) = 0.0613,
    tied keys give about 0.25.

    40 images are the batches [0:16], [16:32], [32:40]; every pixel has its own value, which is how
    a kept pixel is found again. 300 seeded draws x 64 pairs: expected 1177, sd 33; the mutant
    lands near 4800.
    """
    p = (0.05 + 0.005 * np.arange(160)).astype(np.float32).reshape(40, 2, 2)
    labels = np.zeros((40, 2, 2), dtype=np.int32)
    draws, together = 300, 0
    for seed in range(draws):
        kept = _cache_entry(p[..., None], labels, max_pixels=40, seed=seed)["predictions"][:, 1]
        ids = {int(round((float(v) - 0.05) / 0.005)) for v in kept}
        together += sum(1 for i in range(64) if i in ids and (i + 64) in ids)
    pairs = draws * 64
    expected = pairs * (40 / 160) * (39 / 159)
    sigma = np.sqrt(pairs * (40 / 160) * (39 / 159) * (1 - (40 / 160) * (39 / 159)))
    assert abs(together - expected) < 6 * sigma, (together, expected, sigma)


@pytest.mark.parametrize("max_pixels", [0, -3])
def test_a_cap_below_one_is_refused_naming_the_option(max_pixels):
    """Mutant ``an_cap_check_off``: without the check, 0 kept pixels ends as 'no labelled pixel'."""
    with pytest.raises(ValueError, match="max_pixels must be at least 1"):
        _cache_entry(_x(), Y_CLEAN, max_pixels=max_pixels, seed=0)

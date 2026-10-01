"""Tests for :mod:`dl_techniques.losses.harmonic_loss`.

The oracle is a brute-force float64 numpy transcription of the paper's
definition (explicit ``||x - w||`` and ``d^-n`` normalisation). It shares no code
with the implementation under test, which works in log space on an expanded
squared distance.
"""

import keras
import numpy as np
import pytest

from dl_techniques.losses import HarmonicCausalLMLoss, harmonic_logits
from dl_techniques.losses.masked_causal_lm_loss import MaskedCausalLMLoss


def _oracle_log_probs(x, w, n):
    """log p_i = log(d_i^-n / sum_j d_j^-n), straight from the definition."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    d = np.linalg.norm(x[..., None, :] - w, axis=-1)
    unnorm = d ** (-float(n))
    return np.log(unnorm / unnorm.sum(axis=-1, keepdims=True))


def _log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _data(seed=0, b=3, t=5, h=16, v=40):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(b, t, h)).astype("float32"),
        rng.normal(size=(v, h)).astype("float32"),
    )


@pytest.mark.parametrize("n", [1.0, 4.0, 32.0, 64.0])
def test_logits_match_the_brute_force_definition(n):
    x, w = _data()
    got = _log_softmax(keras.ops.convert_to_numpy(harmonic_logits(x, w, n)))
    np.testing.assert_allclose(got, _oracle_log_probs(x, w, n), atol=1e-4)


def test_exponent_is_the_paper_n_not_n_over_two():
    # d^-n, not (d^2)^-n: halving the exponent must change the answer.
    x, w = _data()
    full = _log_softmax(keras.ops.convert_to_numpy(harmonic_logits(x, w, 8.0)))
    half = _log_softmax(keras.ops.convert_to_numpy(harmonic_logits(x, w, 4.0)))
    assert np.abs(full - half).max() > 0.1


def test_joint_scale_invariance():
    x, w = _data()
    a = _log_softmax(keras.ops.convert_to_numpy(harmonic_logits(x, w, 16.0)))
    b = _log_softmax(keras.ops.convert_to_numpy(harmonic_logits(3.0 * x, 3.0 * w, 16.0)))
    np.testing.assert_allclose(a, b, atol=1e-3)


def test_query_on_a_class_weight_is_finite_and_wins():
    _, w = _data()
    x = w[7][None, None, :].copy()
    lg = keras.ops.convert_to_numpy(harmonic_logits(x, w, 768.0))
    assert np.isfinite(lg).all()
    assert lg.argmax(-1)[0, 0] == 7


def test_rounding_negative_squared_distance_is_clamped():
    # Large-norm queries equal to weights: |x|^2+|w|^2-2xw has rounding error far above
    # the true distance, so many entries come out negative in float32 without the clamp.
    rng = np.random.default_rng(1)
    w = (1000.0 * rng.normal(size=(64, 256))).astype("float32")
    x = w[None, :, :].copy()
    lg = keras.ops.convert_to_numpy(harmonic_logits(x, w, 64.0))
    assert np.isfinite(lg).all()
    assert (lg.argmax(-1)[0] == np.arange(64)).all()


def test_half_precision_input_returns_float32_and_float64_stays():
    x, w = _data()
    assert harmonic_logits(x.astype("float16"), w.astype("float16"), 4.0).dtype == "float32"
    assert harmonic_logits(x.astype("float64"), w.astype("float64"), 4.0).dtype == "float64"


@pytest.mark.parametrize("kwargs", [{"exponent": 0.0}, {"exponent": -1.0}, {"exponent": 1.0, "eps": 0.0}])
def test_nonpositive_hyperparameters_raise(kwargs):
    x, w = _data()
    with pytest.raises(ValueError):
        harmonic_logits(x, w, **kwargs)


def test_loss_is_negative_log_harmonic_probability_with_masking():
    x, w = _data()
    n = 8.0
    y = np.random.default_rng(2).integers(0, w.shape[0], size=x.shape[:2])
    y[0, :2] = -1
    lg = harmonic_logits(x, w, n)
    got = float(HarmonicCausalLMLoss()(y, lg))
    logp = _oracle_log_probs(x, w, n)
    keep = y != -1
    picked = np.take_along_axis(logp, np.maximum(y, 0)[..., None], axis=-1)[..., 0]
    assert got == pytest.approx(float(-picked[keep].mean()), abs=1e-4)


def test_gradients_are_finite_at_gpt2_scale_exponent():
    import tensorflow as tf

    x, w = _data(h=64, v=50)
    xv, wv = tf.Variable(x), tf.Variable(w)
    y = np.random.default_rng(3).integers(0, 50, size=x.shape[:2])
    with tf.GradientTape() as tape:
        loss = HarmonicCausalLMLoss()(y, harmonic_logits(xv, wv, 128.0))
    gx, gw = tape.gradient(loss, [xv, wv])
    assert np.isfinite(gx.numpy()).all() and np.isfinite(gw.numpy()).all()
    assert np.abs(gw.numpy()).max() > 0


def test_config_round_trip_and_from_logits_cannot_be_disabled():
    loss = HarmonicCausalLMLoss(ignore_index=-100, label_smoothing=0.1)
    config = loss.get_config()
    assert "from_logits" not in config
    clone = HarmonicCausalLMLoss.from_config(config)
    assert clone.ignore_index == -100 and clone.label_smoothing == 0.1 and clone.from_logits
    assert HarmonicCausalLMLoss(from_logits=False).from_logits is True
    assert isinstance(keras.losses.deserialize(keras.losses.serialize(loss)), HarmonicCausalLMLoss)


def test_is_the_masked_ce_over_harmonic_logits():
    x, w = _data()
    y = np.random.default_rng(4).integers(0, w.shape[0], size=x.shape[:2])
    lg = harmonic_logits(x, w, 8.0)
    assert float(HarmonicCausalLMLoss()(y, lg)) == pytest.approx(
        float(MaskedCausalLMLoss()(y, lg)), abs=1e-6
    )

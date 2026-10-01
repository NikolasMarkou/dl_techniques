"""GPT2 ``head_type='harmonic'``: distance-scored LM head (arXiv:2502.01628)."""

import os

import keras
import numpy as np
import pytest

from dl_techniques.losses import HarmonicCausalLMLoss, harmonic_logits
from dl_techniques.models.language.gpt2.gpt2 import GPT2

_KW = dict(vocab_size=128, embed_dim=32, depth=2, num_heads=4,
           max_seq_len=32, dropout_rate=0.0, attention_dropout_rate=0.0)


def _tokens(seed=0):
    return np.random.default_rng(seed).integers(0, 128, (2, 16)).astype("int32")


def _np(t):
    return keras.ops.convert_to_numpy(t)


def test_default_head_is_the_linear_head_and_unchanged():
    model = GPT2(**_KW)
    assert model.head_type == "linear"
    out = model(_tokens(), training=False)
    emb = model.decoder.word_embeddings.embeddings
    np.testing.assert_allclose(
        _np(out["logits"]),
        _np(out["last_hidden_state"]) @ _np(emb).T,
        atol=1e-4,
    )


def test_harmonic_logits_are_distance_scores_on_the_embedding_table():
    model = GPT2(head_type="harmonic", harmonic_exponent=8.0, **_KW)
    out = model(_tokens(), training=False)
    emb = _np(model.decoder.word_embeddings.embeddings)
    h = _np(out["last_hidden_state"]).astype("float64")
    d = np.linalg.norm(h[:, :, None, :] - emb.astype("float64"), axis=-1)
    expected = -8.0 * np.log(d)
    got = _np(out["logits"]).astype("float64")
    # logits are defined up to a per-position constant (softmax shift)
    np.testing.assert_allclose(
        got - got.max(-1, keepdims=True),
        expected - expected.max(-1, keepdims=True),
        atol=1e-3,
    )


def test_harmonic_head_adds_no_parameters():
    counts = []
    for kwargs in ({}, {"head_type": "harmonic"}):
        model = GPT2(**_KW, **kwargs)
        model(_tokens(), training=False)
        counts.append(model.count_params())
    assert counts[0] == counts[1]


def test_exponent_defaults_to_twice_the_width():
    assert GPT2(head_type="harmonic", **_KW).harmonic_exponent == 64.0


def test_loss_on_model_output_is_finite_and_trainable_one_step():
    import tensorflow as tf

    model = GPT2(head_type="harmonic", **_KW)
    x = _tokens()
    y = np.roll(x, -1, axis=1)
    loss_fn = HarmonicCausalLMLoss()
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model(x, training=True)["logits"])
    grads = tape.gradient(loss, model.trainable_variables)
    assert np.isfinite(float(loss))
    assert all(g is not None and np.isfinite(_np(g)).all() for g in grads)


@pytest.mark.parametrize("kwargs", [
    {"head_type": "bogus"},
    {"head_type": "harmonic", "tie_word_embeddings": False},
    {"head_type": "harmonic", "harmonic_exponent": 0.0},
    {"head_type": "harmonic", "harmonic_eps": 0.0},
])
def test_invalid_head_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        GPT2(**_KW, **kwargs)


def test_old_config_without_head_keys_still_loads():
    config = GPT2(**_KW).get_config()
    for key in ("head_type", "harmonic_exponent", "harmonic_eps"):
        config.pop(key)
    assert GPT2.from_config(config).head_type == "linear"


def test_keras_round_trip_keeps_the_harmonic_head(tmp_path):
    model = GPT2(head_type="harmonic", harmonic_exponent=12.0, harmonic_eps=1e-5, **_KW)
    x = _tokens()
    before = _np(model(x, training=False)["logits"])
    path = os.path.join(str(tmp_path), "gpt2_harmonic.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    assert (loaded.head_type, loaded.harmonic_exponent, loaded.harmonic_eps) == ("harmonic", 12.0, 1e-5)
    np.testing.assert_allclose(before, _np(loaded(x, training=False)["logits"]), atol=1e-3)


def test_mixed_float16_logits_are_float32():
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = GPT2(head_type="harmonic", **_KW)
        assert model(_tokens(), training=False)["logits"].dtype == "float32"
    finally:
        keras.mixed_precision.set_global_policy("float32")

"""A Timestral model computes in the dtype it is given, and refuses float16.

Until 2026-10-20 (plan-2026-10-14T062616-81938e81 D-178) ``TimestralModel(config,
dtype=X)`` built float32 weights and computed in float32 for every X: the
sublayers were constructed without the model's policy, so they took the global
one. Measured on the released 5M weights (scratchpad ``step4/timestral``,
GPU 1), with every sublayer under one policy: bfloat16 forecasts stay finite and
close (median MAE 0.1410 vs 0.1416 in float32 on synthetic series with a known
continuation), float16 forecasts stay finite but are 62x worse (MAE 8.82). So the
policy is threaded through every sublayer, and a float16 compute dtype raises.
"""

import numpy as np
import pytest

import keras

from dl_techniques.models.time_series.timestral.config import TimestralConfig
from dl_techniques.models.time_series.timestral.model import TimestralModel


def _config():
    return TimestralConfig(
        patch_size=8, d_model=32, num_heads=4, mlp_hidden_dim=64,
        block_schedule=("time", "time_variate"), quantile_knots=(0.1, 0.5, 0.9),
        block_size=4, context_length=64, attn_bias=True, mlp_bias=True,
        variate_attn_gate="sdpa_sigmoid")


def _forecast(model):
    rng = np.random.RandomState(0)
    context = np.cumsum(rng.randn(3, 32), axis=-1).astype("float32")
    observed = np.ones((3, 32), dtype=bool)
    groups = np.array([0, 0, 1], dtype="int32")
    return keras.ops.convert_to_numpy(
        model.predict_block(context, observed, groups, horizon_patches=2)).astype("float64")


def _sublayers(model):
    return [layer for layer in model._flatten_layers(include_self=False)]


@pytest.mark.parametrize("policy, variable_dtype", [
    ("bfloat16", "bfloat16"),
    ("mixed_bfloat16", "float32"),
    ("float32", "float32"),
])
def test_the_model_policy_reaches_every_sublayer(policy, variable_dtype):
    model = TimestralModel(_config(), dtype=policy)
    model.build((None, 32))
    compute = keras.DTypePolicy(policy).compute_dtype
    layers = _sublayers(model)
    assert len(layers) > 20
    wrong = [(layer.path, layer.dtype_policy.name) for layer in layers
             if layer.dtype_policy.name != policy]
    assert wrong == []
    assert {w.dtype for w in model.weights} == {variable_dtype}
    assert all(layer.compute_dtype == compute for layer in layers)


def test_a_bfloat16_model_forecasts_close_to_float32():
    keras.utils.set_random_seed(0)
    reference = TimestralModel(_config())
    reference.build((None, 32))
    model = TimestralModel(_config(), dtype="bfloat16")
    model.build((None, 32))
    model.set_weights([keras.ops.convert_to_numpy(w) for w in reference.weights])
    out32, out16 = _forecast(reference), _forecast(model)
    assert np.isfinite(out16).all()
    assert not np.array_equal(out16, out32)  # it really computed in bfloat16
    assert np.abs(out16 - out32).max() <= 0.05 * np.abs(out32).max()


def test_a_bfloat16_model_round_trips_with_its_dtype(tmp_path):
    model = TimestralModel(_config(), dtype="bfloat16")
    model.build((None, 32))
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    assert {w.dtype for w in loaded.weights} == {"bfloat16"}
    np.testing.assert_array_equal(_forecast(loaded), _forecast(model))


@pytest.mark.parametrize("policy", ["float16", "mixed_float16"])
def test_a_float16_model_is_refused(policy):
    with pytest.raises(ValueError, match="float16"):
        TimestralModel(_config(), dtype=policy)


def test_a_float16_global_policy_is_refused():
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy("mixed_float16")
    try:
        with pytest.raises(ValueError, match="float16"):
            TimestralModel(_config())
    finally:
        keras.config.set_dtype_policy(previous)

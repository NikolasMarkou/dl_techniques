"""Checkpoint loading, writing and re-saving.

Until 2026-10-19 (plan-2026-10-14T062616-81938e81 D-166):

- ``load_model`` built the model through one dummy ``predict_block`` and never marked it
  built, so ``model.save`` wrote no weights (the reload held 0 of 30) and a later
  ``model.build`` raised.
- It matched checkpoint entries to ``Variable.path``, which carries the model name only
  when the model was built explicitly, so an explicitly built model's export did not load.
- No released checkpoint could load: their names are the reference PyTorch state-dict
  names, with ``(out, in)`` Linear weights.
"""

import numpy as np
import pytest

import keras

from dl_techniques.models.time_series.timestral import create_timestral_model, load_model
from dl_techniques.models.time_series.timestral.loading import (
    COPY, LINEAR, QKV, QKV_BIAS, WEIGHTS_FILE, load_safetensors, reference_weight_map,
    save_checkpoint, save_safetensors, to_keras_layout, to_reference_layout,
)


def _tiny(**overrides):
    kwargs = dict(prediction_length=16, patch_size=8, quantile_levels=(0.1, 0.5, 0.9),
                  d_model=32, num_heads=4, mlp_hidden_dim=64,
                  block_schedule=("time", "time_variate"), block_size=4, context_length=64,
                  attn_bias=True)
    kwargs.update(overrides)
    return create_timestral_model("5M", **kwargs)


def _predict(model):
    rng = np.random.RandomState(0)
    context = np.cumsum(rng.randn(3, 32), axis=-1).astype("float32")
    observed = np.ones((3, 32), dtype=bool)
    return keras.ops.convert_to_numpy(model.predict_block(
        context, observed, np.array([0, 0, 1], dtype="int32"), horizon_patches=2))


def _values(model):
    return [keras.ops.convert_to_numpy(w) for w in model.weights]


@pytest.fixture
def exported(tmp_path):
    keras.utils.set_random_seed(0)
    model = _tiny()
    model.build((None, 32))
    return model, save_checkpoint(model, tmp_path / "ckpt")


def test_a_loaded_model_is_built_and_builds_again(exported):
    _, directory = exported
    loaded = load_model(directory)
    assert loaded.built
    count = len(loaded.weights)
    loaded.build((None, 32))
    assert len(loaded.weights) == count


def test_a_loaded_model_saves_and_reloads_every_weight(exported, tmp_path):
    donor, directory = exported
    loaded = load_model(directory)
    path = tmp_path / "loaded.keras"
    loaded.save(str(path))
    reloaded = keras.models.load_model(str(path), compile=False)
    assert [w.path for w in reloaded.weights] == [w.path for w in loaded.weights]
    for got, want, weight in zip(_values(reloaded), _values(loaded), loaded.weights):
        np.testing.assert_array_equal(got, want, err_msg=weight.path)
    np.testing.assert_array_equal(_predict(reloaded), _predict(donor))


def test_the_checkpoint_round_trip_is_exact(exported):
    donor, directory = exported
    loaded = load_model(directory)
    for got, want, weight in zip(_values(loaded), _values(donor), donor.weights):
        np.testing.assert_array_equal(got, want, err_msg=weight.path)
    np.testing.assert_array_equal(_predict(loaded), _predict(donor))


@pytest.mark.parametrize("how", ["lazy", "explicit", "loaded", "keras_reload"])
def test_checkpoint_names_do_not_depend_on_how_the_model_was_built(how, tmp_path):
    """One naming for every model: lazy paths are unprefixed, explicit ones are not."""
    keras.utils.set_random_seed(0)
    model = _tiny()
    if how == "lazy":
        _predict(model)
    else:
        model.build((None, 32))
    if how in ("loaded", "keras_reload"):
        model = load_model(save_checkpoint(model, tmp_path / "first"))
    if how == "keras_reload":
        model.save(str(tmp_path / "m.keras"))
        model = keras.models.load_model(str(tmp_path / "m.keras"), compile=False)
    expected = _predict(model)
    names = sorted(load_safetensors(save_checkpoint(model, tmp_path / "ckpt") / WEIGHTS_FILE))
    assert names == sorted(reference_weight_map(model))
    assert names[0] == "head.projection.hidden_layer.weight"
    assert not any(name.startswith(model.name) for name in names)
    np.testing.assert_array_equal(_predict(load_model(tmp_path / "ckpt")), expected)


def test_the_fused_qkv_follows_the_reference_head_split():
    """The reference splits the fused output as (3, head_dim, heads), this port as (3, heads, head_dim).

    Reference ``Attention._qkv``: ``wQKV(x).reshape(batch, seq, 3, head_dim, num_heads)``
    then ``permute(2, 0, 1, 4, 3)``, so torch output row ``c * D + d * H + h`` is
    projection ``c`` (q, k, v), head ``h``, dimension ``d``. Here ``Attention._qkv``
    reshapes to ``(3, num_heads, head_dim)``, so that row is Keras column
    ``c * D + h * head_dim + d``.
    """
    heads, head_dim, width_in = 4, 8, 5
    d_model = heads * head_dim
    for c, h, d in [(0, 1, 2), (1, 3, 0), (2, 0, 7)]:
        torch_weight = np.zeros((3 * d_model, width_in), dtype="float32")
        torch_weight[c * d_model + d * heads + h, 3] = 1.0
        kernel = to_keras_layout(torch_weight, QKV, heads)
        assert kernel.shape == (width_in, 3 * d_model)
        assert np.argwhere(kernel).tolist() == [[3, c * d_model + h * head_dim + d]]
        torch_bias = np.zeros(3 * d_model, dtype="float32")
        torch_bias[c * d_model + d * heads + h] = 1.0
        assert np.argwhere(to_keras_layout(torch_bias, QKV_BIAS, heads)).ravel().tolist() == [
            c * d_model + h * head_dim + d]


@pytest.mark.parametrize("layout", [COPY, LINEAR, QKV, QKV_BIAS])
def test_the_layouts_invert(layout):
    rng = np.random.RandomState(1)
    shape = (24,) if layout in (COPY, QKV_BIAS) else (24, 5)
    array = rng.randn(*shape).astype("float32")
    np.testing.assert_array_equal(
        to_reference_layout(to_keras_layout(array, layout, 2), layout, 2), array)


def test_the_loader_refuses_a_missing_or_unused_tensor(exported):
    _, directory = exported
    tensors = load_safetensors(directory / WEIGHTS_FILE)
    tensors["transformer.layers.9.mlp.w3.weight"] = np.zeros((2, 2), "float32")
    save_safetensors(tensors, directory / WEIGHTS_FILE)
    with pytest.raises(ValueError, match="unused entries"):
        load_model(directory)
    del tensors["transformer.layers.9.mlp.w3.weight"]
    del tensors["head.projection.hidden_layer.weight"]
    save_safetensors(tensors, directory / WEIGHTS_FILE)
    with pytest.raises(ValueError, match="missing variables"):
        load_model(directory)


def test_safetensors_round_trip(tmp_path):
    tensors = {"b": np.arange(6, dtype="float32").reshape(2, 3), "a": np.array([True, False]),
               "c": np.float64(2.5) * np.ones(())}
    save_safetensors(tensors, tmp_path / "t.safetensors")
    raw = (tmp_path / "t.safetensors").read_bytes()
    assert int(np.frombuffer(raw[:8], dtype="<u8")[0]) % 8 == 0
    back = load_safetensors(tmp_path / "t.safetensors")
    assert sorted(back) == ["a", "b", "c"]
    for key, value in tensors.items():
        assert back[key].dtype == value.dtype
        np.testing.assert_array_equal(back[key], value)


@pytest.mark.parametrize("dtype", ["float64", "float16", "bfloat16"])
def test_load_model_has_no_dtype_knob(exported, dtype):
    """Until 2026-10-20 (D-173) ``load_model(dtype=...)`` accepted a dtype and ignored it.

    The sublayers take the global policy, not the model's, so honouring it would mean
    threading a policy through every layer; the model was validated in float32 only.
    The argument is gone: passing it raises, and a loaded model is float32.
    """
    _, directory = exported
    with pytest.raises(TypeError, match="dtype"):
        load_model(directory, dtype=dtype)
    assert {w.dtype for w in load_model(directory).weights} == {"float32"}

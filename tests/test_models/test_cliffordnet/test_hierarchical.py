"""HierarchicalCliffordNet: stages of CliffordNetBlock with a stride-2 downsampler between them.

Guard for plan-2026-10-23T132722-77331e7a (D-001). Source of the architecture:
github.com/ParaMind2025/CAN ``model_hier.py``, read through a fetch summary on
2026-10-23 and not byte-checked: patch-1 stem, stages of CliffordNet blocks, a
depthwise 3x3 stride-2 conv + BN + SiLU + 1x1 conv between stages, a linear
drop-path ramp over ALL blocks (not per stage), a pooled-LayerNorm head. No
accuracy or parameter count of this model is published, so the parameter count
here is a regression pin of what was measured, not a paper value.

The checks are oracles, not shape-only asserts: each stage's feature map is read,
the downsampler is recomputed by hand from its own weights, and the drop-path
ramp is read off the blocks.
"""

import copy

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.models.vision.cliffordnet.hierarchical import (
    HierarchicalCliffordNet,
    create_hierarchical_cliffordnet,
)

# A small model: widths 8/16/32, depths 1/2/1, 16x16 input with patch 1 -> 16, 8, 4.
TINY = dict(
    num_classes=10,
    stage_depths=[1, 2, 1],
    stage_dims=[8, 16, 32],
    shifts=[1, 2],
    stochastic_depth_rate=0.3,
)


def _images(batch=2, size=16, seed=0):
    return np.random.default_rng(seed).normal(size=(batch, size, size, 3)).astype("float32")


def _tiny(**overrides):
    model = HierarchicalCliffordNet(**{**TINY, **overrides})
    model.build((None, 16, 16, 3))
    return model


# ---------------------------------------------------------------------
# Architecture oracles
# ---------------------------------------------------------------------


def test_each_stage_halves_the_resolution_and_takes_its_width():
    model = _tiny()
    features = model.extract_stage_features(_images(), training=False)
    shapes = [tuple(f.shape) for f in features]
    assert shapes == [(2, 16, 16, 8), (2, 8, 8, 16), (2, 4, 4, 32)]


def test_patch_size_2_starts_at_half_resolution():
    model = HierarchicalCliffordNet(**{**TINY, "patch_size": 2})
    model.build((None, 16, 16, 3))
    shapes = [tuple(f.shape) for f in model.extract_stage_features(_images(), training=False)]
    assert shapes == [(2, 8, 8, 8), (2, 4, 4, 16), (2, 2, 2, 32)]


def test_the_drop_path_ramp_is_linear_over_all_blocks_not_per_stage():
    model = _tiny(stochastic_depth_rate=0.3)
    rates = [
        entry["drop_path"].drop_path_rate
        for stage in model.stages
        for entry in stage
    ]
    assert len(rates) == 4
    np.testing.assert_allclose(rates, np.linspace(0.0, 0.3, 4), atol=1e-6)


def test_a_zero_drop_path_rate_gives_every_block_zero():
    model = _tiny(stochastic_depth_rate=0.0)
    assert all(
        entry["drop_path"].drop_path_rate == 0.0
        for stage in model.stages
        for entry in stage
    )


def _randomize_bn(bn, seed):
    rng = np.random.default_rng(seed)
    channels = bn.gamma.shape[0]
    bn.gamma.assign(rng.uniform(0.5, 1.5, channels).astype("float32"))
    bn.beta.assign(rng.normal(size=channels).astype("float32"))
    bn.moving_mean.assign(rng.normal(size=channels).astype("float32"))
    bn.moving_variance.assign(rng.uniform(0.5, 2.0, channels).astype("float32"))


def _numpy_downsample(x, depthwise_kernel, bn, pointwise):
    """Depthwise 3x3 stride 2 (TF SAME), BN (inference), SiLU, 1x1 conv, in numpy."""
    batch, height, width, channels = x.shape
    out_h, out_w = -(-height // 2), -(-width // 2)
    pad_h = max((out_h - 1) * 2 + 3 - height, 0)
    pad_w = max((out_w - 1) * 2 + 3 - width, 0)
    padded = np.pad(
        x, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
    )
    out = np.zeros((batch, out_h, out_w, channels), dtype="float64")
    for i in range(out_h):
        for j in range(out_w):
            window = padded[:, 2 * i:2 * i + 3, 2 * j:2 * j + 3, :]
            out[:, i, j, :] = np.einsum("bhwc,hwc->bc", window, depthwise_kernel[:, :, :, 0])
    gamma, beta = bn.gamma.numpy(), bn.beta.numpy()
    mean, var = bn.moving_mean.numpy(), bn.moving_variance.numpy()
    out = (out - mean) / np.sqrt(var + bn.epsilon) * gamma + beta
    out = out * (1.0 / (1.0 + np.exp(-out)))
    kernel = pointwise.kernel.numpy()[0, 0]
    out = np.einsum("bhwc,cd->bhwd", out, kernel)
    if pointwise.bias is not None:
        out = out + pointwise.bias.numpy()
    return out


def test_the_downsampler_is_depthwise_stride2_bn_silu_pointwise():
    model = _tiny()
    for index, entry in enumerate(model.downsamplers):
        _randomize_bn(entry["bn"], seed=index)
        entry["pointwise"].bias.assign(
            np.random.default_rng(10 + index).normal(size=entry["pointwise"].bias.shape[0]).astype("float32")
        )
    x = np.random.default_rng(3).normal(size=(2, 16, 16, 8)).astype("float32")
    entry = model.downsamplers[0]
    want = _numpy_downsample(
        x, entry["depthwise"].kernel.numpy(), entry["bn"], entry["pointwise"]
    )
    # On the CPU: a GPU convolution may run in TF32, whose rounding (about 1e-4)
    # would force a tolerance too loose to catch a missing SiLU or BN.
    with tf.device("/CPU:0"):
        got = keras.ops.convert_to_numpy(model._downsample(0, x, training=False))
    assert got.shape == (2, 8, 8, 16)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_equal_widths_need_no_pointwise_conv():
    model = HierarchicalCliffordNet(
        num_classes=10, stage_depths=[1, 1], stage_dims=[8, 8], shifts=[1, 2]
    )
    model.build((None, 16, 16, 3))
    assert model.downsamplers[0]["pointwise"] is None
    shapes = [tuple(f.shape) for f in model.extract_stage_features(_images(), training=False)]
    assert shapes == [(2, 16, 16, 8), (2, 8, 8, 8)]


def test_the_default_widths_double_from_32():
    model = HierarchicalCliffordNet(num_classes=100)
    assert model.stage_depths == [3, 4, 5]
    assert model.stage_dims == [32, 64, 128]
    assert model.patch_size == 1


# ---------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------


def test_the_output_is_float32_logits_of_the_right_shape():
    model = _tiny()
    out = model(_images(), training=False)
    assert tuple(out.shape) == (2, 10)
    assert keras.backend.standardize_dtype(out.dtype) == "float32"


def test_training_differs_from_inference():
    # layer_scale_init=1 so the blocks' updates are not squashed to 1e-5, which
    # would hide both the batch-norm and the drop-path difference.
    model = _tiny(stochastic_depth_rate=0.5, layer_scale_init=1.0)
    x = _images(batch=8)
    infer = keras.ops.convert_to_numpy(model(x, training=False))
    train = keras.ops.convert_to_numpy(model(x, training=True))
    assert not np.allclose(infer, train, atol=1e-6)


def test_gradients_reach_every_trainable_weight():
    import tensorflow as tf

    model = _tiny(stochastic_depth_rate=0.0)
    x = _images(batch=4)
    y = np.array([0, 1, 2, 3])
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        loss = keras.ops.mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    missing = [v.path for v, g in zip(model.trainable_variables, grads) if g is None]
    assert not missing, f"no gradient for: {missing}"
    dead = [
        v.path for v, g in zip(model.trainable_variables, grads)
        if not np.all(np.isfinite(keras.ops.convert_to_numpy(g)))
    ]
    assert not dead, f"non-finite gradient for: {dead}"


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_get_config_round_trips():
    model = _tiny()
    config = model.get_config()
    clone = HierarchicalCliffordNet.from_config(config)
    assert clone.get_config() == config


def test_a_keras_archive_reloads_with_identical_logits(tmp_path):
    model = _tiny()
    x = _images()
    want = keras.ops.convert_to_numpy(model(x, training=False))
    path = str(tmp_path / "hier.keras")
    model.save(path)
    reloaded = keras.models.load_model(path)
    assert isinstance(reloaded, HierarchicalCliffordNet)
    got = keras.ops.convert_to_numpy(reloaded(x, training=False))
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_the_registration_key_has_the_family_directory_stripped():
    assert (
        keras.saving.get_registered_name(HierarchicalCliffordNet)
        == "dl_techniques.models.cliffordnet.hierarchical>HierarchicalCliffordNet"
    )


# ---------------------------------------------------------------------
# Validation, variants, factory
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        dict(num_classes=0),
        dict(stage_depths=[]),
        dict(stage_depths=[1, 0, 1]),
        dict(stage_dims=[8, 16]),
        dict(stage_dims=[8, 0, 32]),
        dict(patch_size=3),
        dict(patch_size=0),
        dict(cli_mode="bogus"),
    ],
)
def test_invalid_arguments_raise(overrides):
    with pytest.raises(ValueError):
        HierarchicalCliffordNet(**{**TINY, **overrides})


def test_pretrained_true_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        create_hierarchical_cliffordnet("hier_345", num_classes=100, pretrained=True)


def test_an_unknown_variant_raises():
    with pytest.raises(ValueError):
        HierarchicalCliffordNet.from_variant("nope", num_classes=10)


def test_the_preset_table_is_the_authors_defaults():
    assert set(HierarchicalCliffordNet.MODEL_VARIANTS) == {"hier_345"}
    config = HierarchicalCliffordNet.MODEL_VARIANTS["hier_345"]
    assert config["stage_depths"] == [3, 4, 5]
    assert config["stage_dims"] == [32, 64, 128]
    assert config["shifts"] == [1, 2]
    assert config["patch_size"] == 1
    assert config["cli_mode"] == "full"
    assert config["ctx_mode"] == "diff"


def _plain(table):
    """The list, int, float and str entries of every preset (an initializer object
    compares by identity, so it is left out)."""
    return {
        name: copy.deepcopy({k: v for k, v in cfg.items() if isinstance(v, (list, int, float, str))})
        for name, cfg in table.items()
    }


def test_from_variant_does_not_poison_the_table():
    before = _plain(HierarchicalCliffordNet.MODEL_VARIANTS)
    model = HierarchicalCliffordNet.from_variant("hier_345", num_classes=10)
    model.stage_depths.append(99)
    model.shifts.append(99)
    assert _plain(HierarchicalCliffordNet.MODEL_VARIANTS) == before


def test_the_factory_returns_a_preset_that_builds():
    model = create_hierarchical_cliffordnet("hier_345", num_classes=100)
    assert isinstance(model, HierarchicalCliffordNet)
    model.build((None, 32, 32, 3))
    assert model.built


def test_the_preset_has_its_pinned_parameter_count():
    model = HierarchicalCliffordNet.from_variant("hier_345", num_classes=100)
    model.build((None, 32, 32, 3))
    # Regression pin of the count measured here; no published figure exists.
    assert model.count_params() == PINNED_HIER_345_PARAMS


PINNED_HIER_345_PARAMS = 767_476

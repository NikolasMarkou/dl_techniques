"""Weight-decay exclusions of the LightGlue trainer's optimizer (D-019), per variable."""

import keras
import numpy as np
import pytest

import train.lightglue.train_lightglue as trainer
from train.lightglue.train_lightglue import LightGlueTrainConfig, build_optimizer

from .conftest import make_lightglue

_LEAVES_EXCLUDED = ("bias", "gamma", "beta")


def _config(weight_decay: float = 0.5) -> LightGlueTrainConfig:
    return LightGlueTrainConfig(
        superpoint_checkpoint="unused.keras", weight_decay=weight_decay, clip_norm=0.0)


@pytest.fixture()
def built():
    model = make_lightglue()
    model.build({"keypoints0": (None, 8, 2), "keypoints1": (None, 8, 2),
                 "descriptors0": (None, 8, 32), "descriptors1": (None, 8, 32)})
    return model


def _decays(optimizer, variable) -> bool:
    return optimizer._use_weight_decay(variable)


def _expected_decay(variable) -> bool:
    if variable.path.endswith("posenc/kernel"):
        return False
    return variable.name not in _LEAVES_EXCLUDED


def test_every_variable_has_the_intended_decay_status(built):
    optimizer = build_optimizer(_config(), 10, lightglue=built)
    variables = built.trainable_variables
    assert variables
    for v in variables:
        assert _decays(optimizer, v) == _expected_decay(v), v.path
    decayed = [v for v in variables if _decays(optimizer, v)]
    assert decayed and all(v.name == "kernel" for v in decayed)
    # the attention and FFN kernels do decay, only the positional encoding does not
    assert any("Wqkv/kernel" in v.path for v in decayed)
    assert not any("posenc" in v.path for v in decayed)


def test_posenc_kernel_is_excluded_and_dense_kernel_is_not(built):
    optimizer = build_optimizer(_config(), 10, lightglue=built)
    assert not _decays(optimizer, built.posenc.kernel)
    dense = next(v for v in built.trainable_variables if "Wqkv/kernel" in v.path)
    assert _decays(optimizer, dense)


def test_one_zero_gradient_step_shrinks_a_dense_kernel_only(built):
    optimizer = build_optimizer(_config(0.5), 10, lightglue=built)
    posenc = built.posenc.kernel
    dense = next(v for v in built.trainable_variables if "Wqkv/kernel" in v.path)
    bias = next(v for v in built.trainable_variables if v.name == "bias")
    variables = [posenc, dense, bias]
    before = [np.array(v.numpy()) for v in (posenc, dense, bias)]
    optimizer.build(variables)
    optimizer.iterations.assign(optimizer.iterations + 100)  # past step 0 of the warmup (lr 0)
    optimizer.apply([keras.ops.zeros_like(v) for v in variables], variables)
    assert np.array_equal(posenc.numpy(), before[0])
    assert np.array_equal(bias.numpy(), before[2])
    assert not np.array_equal(dense.numpy(), before[1])
    assert np.all(np.abs(dense.numpy()) <= np.abs(before[1]))


def test_guard_is_red_with_the_old_list(built):
    """The pre-D-019 behaviour (no var_list exemption) decays the positional encoding."""
    optimizer = build_optimizer(_config(), 10, lightglue=None)  # old call shape
    assert _decays(optimizer, built.posenc.kernel)  # the defect this step removed


def test_unbuilt_lightglue_is_refused():
    with pytest.raises(ValueError, match="built"):
        build_optimizer(_config(), 10, lightglue=make_lightglue())


def test_the_name_patterns_are_the_documented_ones():
    assert trainer.NO_DECAY_NAME_PATTERNS == list(_LEAVES_EXCLUDED)

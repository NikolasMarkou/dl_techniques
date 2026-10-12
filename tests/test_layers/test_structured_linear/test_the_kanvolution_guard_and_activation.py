"""KANvolution: the strides/dilation contract is validated, and the activation
follows the shared helper like every sibling layer.

`strides > 1` with `dilation_rate > 1` built fine and died inside TF at
`call()`; the docstring already declared them incompatible. The layer also
used the raw `utils/activation_serialization` pair -- the one import path
`layers/AGENTS.md` names as unsafe -- with a dead `is not None` guard.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.structured_linear.bitlinear_layer import BitLinear
from dl_techniques.layers.structured_linear.kanvolution import KANvolution
from dl_techniques.layers.structured_linear.orthoblock import OrthoBlock
from dl_techniques.layers.structured_linear.random_fourier_features import (
    RFFKernelLayer,
)


def test_strides_and_dilation_together_raise_at_construction():
    with pytest.raises(ValueError, match="strides > 1.*dilation_rate > 1"):
        KANvolution(filters=3, kernel_size=3, strides=(2, 2), dilation_rate=(2, 2))
    with pytest.raises(ValueError, match="strides > 1.*dilation_rate > 1"):
        KANvolution(filters=3, kernel_size=3, strides=2, dilation_rate=2)


def test_either_alone_still_builds():
    KANvolution(filters=3, kernel_size=3, strides=(2, 2)).build((None, 8, 8, 2))
    KANvolution(filters=3, kernel_size=3, dilation_rate=(2, 2)).build((None, 8, 8, 2))


def test_a_layer_activation_dict_round_trips():
    layer = KANvolution(filters=2, kernel_size=3, activation="gelu")
    assert layer.get_config()["activation"] == "gelu"


def test_default_and_linear_activations_agree():
    # Same weights, so the only possible difference is the activation path.
    # (Two fresh builds cannot be compared: seedless initializer draws are
    # not pinned by `keras.utils.set_random_seed`.)
    x = np.random.RandomState(0).randn(1, 6, 6, 2).astype(np.float32)
    default = KANvolution(filters=2, kernel_size=3)
    np.asarray(default(x))
    linear = KANvolution(filters=2, kernel_size=3, activation="linear")
    linear.build((None, 6, 6, 2))
    linear.set_weights(default.get_weights())
    np.testing.assert_allclose(
        np.asarray(default(x)), np.asarray(linear(x)), rtol=0, atol=1e-6
    )


def test_the_identity_flag_exists_where_the_guard_was_dead():
    assert KANvolution(filters=2, kernel_size=3)._is_identity_activation is True
    assert BitLinear(units=4)._is_identity_activation is True
    assert RFFKernelLayer(input_dim=8)._is_identity_activation is True
    assert BitLinear(units=4, activation="relu")._is_identity_activation is False


def test_orthoblock_rejects_a_bool_unit_count():
    with pytest.raises(ValueError, match="positive integer"):
        OrthoBlock(units=True)

"""``sLSTMBlock`` builds its FFN from its own arguments.

Measured at HEAD before the fix: the FFN was created from ``ffn_type``,
``output_dim``, ``ffn_expansion_factor`` and ``dropout_rate`` only.

* ``kernel_initializer``, ``bias_initializer``, ``kernel_regularizer`` and
  ``bias_regularizer`` never reached it: ``kernel_regularizer=L2(1.0)`` charged
  1 loss (the sLSTM kernel) where the three SwiGLU kernels make it 4.
* Every ``ffn_type`` the docstring advertised except ``'swiglu'`` raised
  ``ValueError``, because the block never passed the ``hidden_dim`` they require.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.xlstm_blocks import sLSTMBlock

ADVERTISED_FFN_TYPES = ["mlp", "swiglu", "geglu", "glu", "differential", "residual", "swin_mlp"]


def _inputs():
    return np.random.default_rng(0).normal(size=(2, 8, 4)).astype("float32")


def _ffn_kernels(block: sLSTMBlock):
    return [v for v in block.ffn.trainable_weights if v.path.endswith("kernel")]


def test_the_kernel_regularizer_is_charged_on_the_ffn() -> None:
    block = sLSTMBlock(units=4, kernel_regularizer=keras.regularizers.L2(1.0))
    block(_inputs())
    assert len(_ffn_kernels(block)) == 3
    assert len(block.losses) == 1 + len(_ffn_kernels(block))


def test_the_kernel_initializer_reaches_the_ffn() -> None:
    block = sLSTMBlock(units=4, kernel_initializer="ones")
    block(_inputs())
    for kernel in _ffn_kernels(block):
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(kernel), 1.0)


@pytest.mark.parametrize("ffn_type", ADVERTISED_FFN_TYPES)
def test_every_advertised_ffn_type_builds(ffn_type: str) -> None:
    block = sLSTMBlock(units=4, ffn_type=ffn_type, ffn_expansion_factor=2)
    assert tuple(block(_inputs()).shape) == (2, 8, 4)

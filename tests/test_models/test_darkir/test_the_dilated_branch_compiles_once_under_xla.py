"""DarkIR's training step compiles once under XLA: no weight becomes a constant.

Guard for plan-2026-10-14T062616-81938e81 step 13 (D-096). Measured at 561e02165
on GPU 1: ``train_darkir.py`` (Keras' default ``jit_compile="auto"``, i.e. XLA)
ran at 46 s per step at 32x32, batch 64, and a stock ``fit`` of the medium
model at 25 s per step against 0.25 s with ``jit_compile=False``. An XLA dump
showed one ``one_step_on_data`` module compiled per step, and the compiler's
argument log listed the input batch and every model variable as
``kind=constant``: the step was recompiled with the current weights baked in.

The cause is ``DilatedBranch.conv``, a ``Conv2D(groups=channels,
dilation_rate=d)``. For ``d > 1`` the TensorFlow-to-XLA bridge folds that
convolution's kernel into a compile-time constant, so every weight update
invalidates the compiled step (``Conv2D(groups=C)`` at ``d = 1``, a dense
dilated ``Conv2D`` and a dilated ``DepthwiseConv2D`` do not). The branch is a
depthwise convolution with multiplier 1, so it is now a ``DepthwiseConv2D``:
the same weight paths, the same fixed-seed draw in the same element order, and
the same output up to float32 rounding.

The test reads the compiled HLO, which is device-independent: a step whose
weights stay arguments has one entry parameter per input and per variable.
"""

import re

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.image_restoration.darkir.model import (
    DarkIRDecoderBlock,
    DilatedBranch,
)
from tests.numerics import reassociation_atol, true_float32_matmul


CHANNELS = 8
SHAPE = (2, 16, 16, CHANNELS)


def _entry_parameters(layer: keras.layers.Layer) -> int:
    """Entry parameters of the XLA-compiled forward-plus-gradient step."""
    x = tf.constant(np.random.default_rng(0).random(SHAPE, dtype=np.float32))
    layer(x)

    @tf.function(jit_compile=True)
    def step(inputs):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(layer(inputs) ** 2)
        return tape.gradient(loss, layer.trainable_weights)

    hlo = step.experimental_get_compiler_ir(x)(stage="hlo")
    entry = hlo[hlo.index("ENTRY"):]
    return len(re.findall(r"parameter\(\d+\)", entry))


@pytest.mark.parametrize("dilation", [1, 4, 9])
def test_a_dilated_branch_keeps_its_weights_as_arguments(dilation):
    branch = DilatedBranch(CHANNELS, expansion=1, dilation=dilation)
    assert _entry_parameters(branch) == 1 + len(branch.weights)


def test_a_decoder_block_keeps_its_weights_as_arguments():
    block = DarkIRDecoderBlock(
        CHANNELS, dilations=[1, 4, 9], extra_depth_wise=True
    )
    assert _entry_parameters(block) == 1 + len(block.weights)


@pytest.mark.parametrize("dilation", [1, 4])
def test_the_branch_is_the_grouped_convolution_it_replaced(dilation):
    keras.utils.set_random_seed(3)
    branch = DilatedBranch(CHANNELS, expansion=1, dilation=dilation)
    branch.build(SHAPE)
    keras.utils.set_random_seed(3)
    reference = keras.layers.Conv2D(
        CHANNELS, 3, padding="same", dilation_rate=dilation,
        groups=CHANNELS, name=f"dilated_conv_d{dilation}",
    )
    reference.build(SHAPE)
    # Same paths, and the same fixed-seed draw in the same element order.
    assert [w.path.split("/", 1)[1] for w in branch.weights] == [
        w.path for w in reference.weights
    ]
    np.testing.assert_array_equal(
        np.asarray(branch.conv.kernel).reshape(-1),
        np.asarray(reference.kernel).reshape(-1),
    )
    x = np.random.default_rng(1).standard_normal(SHAPE).astype("float32")
    # The bound below is float32 rounding, so TF32 convolutions are off.
    with true_float32_matmul():
        got = np.asarray(branch(x))
        want = np.asarray(reference(x))
    # A 3x3 per-channel contraction (9 terms), one evaluation per side.
    atol = reassociation_atol([9], 1, float(np.max(np.abs(want))))
    np.testing.assert_allclose(got, want, rtol=0.0, atol=atol)

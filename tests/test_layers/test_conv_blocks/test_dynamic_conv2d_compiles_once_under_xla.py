"""A grouped, dilated ``DynamicConv2D`` keeps its weights as XLA arguments.

Guard for plan-2026-10-14T062616-81938e81 step 13.5 (D-107), the D-096 trap in
a second layer. Measured at ffa4c2870 with the D-096 instrument (the HLO of an
XLA-compiled forward-plus-gradient step, which is device-independent): every
expert of a ``DynamicConv2D`` is a ``Conv2D(groups=g, dilation_rate=d)``, and for
``g > 1`` and ``d > 1`` the compiled step had one entry parameter fewer than it
has inputs plus variables, at ``g = 2`` and at ``g = C`` alike, for ``d`` 2 and 4:
a kernel was folded into a compile-time constant, so each weight update
recompiles the step. ``g = 1`` and ``d = 1`` were unaffected.

Each expert's grouped dilated convolution is now computed from the expert's own
kernel and bias with operations the bridge compiles with the weights as
arguments: one depthwise convolution when every group is one channel, else one
dense dilated convolution per group. The weights, their paths and their
fixed-seed draw are unchanged; the output equals the grouped ``Conv2D`` up to
float32 rounding.
"""

import re

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.conv_blocks.dynamic_conv2d import DynamicConv2D
from tests.numerics import reassociation_atol

CHANNELS = 8
SHAPE = (2, 16, 16, CHANNELS)

# (groups, dilation, filters): depthwise with multiplier 1 and 2, a 2-group
# convolution, and the unaffected dense and undilated cases.
CASES = [
    (CHANNELS, 2, CHANNELS),
    (CHANNELS, 4, 2 * CHANNELS),
    (2, 2, CHANNELS),
    (2, 4, 2 * CHANNELS),
    (1, 4, CHANNELS),
    (2, 1, CHANNELS),
]


def _layer(groups, dilation, filters):
    return DynamicConv2D(
        filters=filters, kernel_size=3, padding="same", groups=groups,
        dilation_rate=dilation, num_kernels=2,
    )


def _entry_parameters(layer):
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


@pytest.mark.parametrize("groups, dilation, filters", CASES)
def test_the_compiled_step_keeps_every_weight_as_an_argument(groups, dilation, filters):
    layer = _layer(groups, dilation, filters)
    assert _entry_parameters(layer) == 1 + len(layer.trainable_weights)


@pytest.mark.parametrize("groups, dilation, filters", CASES)
def test_each_expert_equals_its_grouped_convolution(groups, dilation, filters):
    layer = _layer(groups, dilation, filters)
    layer.build(SHAPE)
    x = np.random.default_rng(1).standard_normal(SHAPE).astype("float32")
    tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    try:
        for expert in layer.conv_layers:
            got = np.asarray(layer._expert_output(expert, x))
            want = np.asarray(expert(x))
            # A 3x3 x (C / groups) contraction, one evaluation per side.
            terms = 9 * (CHANNELS // groups)
            atol = reassociation_atol([terms], 1, float(np.max(np.abs(want))))
            np.testing.assert_allclose(got, want, rtol=0.0, atol=atol)
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(tf32)


def test_the_weights_are_still_the_grouped_conv_kernels():
    layer = _layer(2, 4, CHANNELS)
    layer.build(SHAPE)
    shapes = {w.path.split("/", 1)[1]: tuple(w.shape) for w in layer.weights}
    assert shapes["conv_expert_0/kernel"] == (3, 3, CHANNELS // 2, CHANNELS)
    assert shapes["conv_expert_1/bias"] == (CHANNELS,)

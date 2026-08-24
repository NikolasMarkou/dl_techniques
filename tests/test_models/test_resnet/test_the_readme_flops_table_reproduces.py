"""README section 17's parameter and FLOPs tables are re-derived here.

F-15. The table used to read `1.8 / 3.6 / 4.1 / 7.8 / 11.6 GFLOPs` with no
provenance and no "not measured here" disclaimer -- the only numeric claim left
in that README without one after the prior plan's cleanup.

This plan pre-registered a decision rule BEFORE any number was looked at:
attempt a real measurement, and trust it ONLY if (i) a calibration arm agrees
with a hand-computed analytic FLOP count within 5%, and (ii) the instrument can
be shown to descend into custom subclassed layers. Otherwise publish nothing
and label the table as uncited literature. The rule exists because this repo
has a recorded case of a layer-tree MAC walk reaching only 2 convolutions and
reporting 527.7 MMAC against 26,199 real -- a ~50x undercount.

BOTH ARMS PASSED, so the table was published, and both arms live on here as
permanent tests rather than as a one-off measurement:

    calibration:  Conv2D(64, 7, s=2, no bias) on 224x224x3
                  analytic MACs             = 118,013,952
                  profiler total_float_ops  = 236,027,904
                  ratio                     = exactly 2.0, 0.0000% error
    descent:      a 2-conv keras.layers.Layer SUBCLASS profiles to exactly the
                  same total as the same 2 convs written flat in a Functional
                  model, and both equal the analytic value.

The calibration arm is also what ESTABLISHED the unit rather than assuming it:
this profiler counts the multiply and the add separately, so its "FLOPs" are
2x the multiply-accumulate count the literature quotes. Comparing the two
without noticing that is a silent 2x error.

MEASURED, 224x224x3, num_classes=1000, include_top=True, stem_type='imagenet':

    variant     trainable    non-train    count_params()      MACs      FLOPs
    resnet18   11,689,512        9,600      11,699,112     1.818 G    3.636 G
    resnet34   21,797,672       17,024      21,814,696     3.669 G    7.338 G
    resnet50   25,557,032       53,120      25,610,152     4.104 G    8.208 G
    resnet101  44,549,160      105,344      44,654,504     7.823 G   15.646 G
    resnet152  60,192,808      151,424      60,344,232    11.544 G   23.088 G

The trainable column matches the widely quoted torchvision counts for all five
variants EXACTLY. The README previously printed those figures as "parameters",
which is not what `count_params()` returns here -- Keras counts BatchNorm's
`moving_mean`/`moving_variance` and PyTorch does not, and that difference is
precisely the non-trainable column.

RED PROOFS -- three named injections, ACTUAL observed text.

Injection A, stem ``kernel_size`` 7 -> 5 (a real FLOP change that leaves the
parameter counts close) -> **7 failed, 2 passed**:

  - "AssertionError: resnet18 trainable params 11,684,904 != README 11,689,512"
    (and the same for all five variants)
  - "AssertionError: resnet18: 3,520,122,472 total_float_ops != README
     3,635,727,976 (3.520 GFLOPs vs 3.636 GFLOPs)"
  - "AssertionError: resnet50: 8,092,577,384 total_float_ops != README
     8,208,182,888 (8.093 GFLOPs vs 8.208 GFLOPs)"

  Calibration and descent correctly stayed GREEN -- they do not touch ResNet.

Injection B, replace ``profile_flops`` with the DEFECTIVE instrument this
plan's rule exists to reject: a walk over the model's direct ``model.layers``
children only -> **3 failed, 6 passed**:

  - "AssertionError: the profiler did NOT descend into the subclassed layer:
     0 vs 115,605,504. No model FLOP number may be published under this plan's
     pre-registered rule."
  - both ResNet FLOPs arms: "AttributeError: The layer stem_conv has never been
     called and thus has no defined input."

  The CALIBRATION ARM PASSED under this injection. That is the whole reason
  both arms exist: a single flat ``Conv2D`` is a direct child, so calibration
  is structurally blind to a descent failure and cannot substitute for it.

Injection C, ``return result.total_float_ops // 2`` (i.e. the instrument
silently reports MACs where the table says FLOPs -- the exact 2x unit confusion
this table used to invite) -> **4 failed, 5 passed**:

  - "AssertionError: calibration FAILED: profiler 118,013,952 vs analytic
     2*MACs 236,027,904 (50.00% error). Under this plan's pre-registered rule,
     no model FLOP number may be published."
  - "AssertionError: resnet18: 1,817,863,988 total_float_ops != README
     3,635,727,976 (1.818 GFLOPs vs 3.636 GFLOPs)"

  Note that 1.818 GMACs is EXACTLY the number the old table printed as
  "1.8 GFLOPs". The mislabelled table and this injection are the same error.
"""

import numpy as np
import pytest
import tensorflow as tf
import keras
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

from dl_techniques.models.resnet import ResNet

INPUT_SHAPE = (224, 224, 3)

# Trainable-parameter counts, which are also the published torchvision figures.
EXPECTED_TRAINABLE = {
    "resnet18": 11_689_512,
    "resnet34": 21_797_672,
    "resnet50": 25_557_032,
    "resnet101": 44_549_160,
    "resnet152": 60_192_808,
}
EXPECTED_NON_TRAINABLE = {
    "resnet18": 9_600,
    "resnet34": 17_024,
    "resnet50": 53_120,
    "resnet101": 105_344,
    "resnet152": 151_424,
}
# total_float_ops (2 x MACs), as printed in README section 17.
EXPECTED_FLOPS = {
    "resnet18": 3_635_727_976,
    "resnet50": 8_208_182_888,
}


def _frozen_graph_def(model: keras.Model, input_shape) -> "tf.compat.v1.GraphDef":
    fn = tf.function(lambda x: model(x, training=False))
    concrete = fn.get_concrete_function(tf.TensorSpec([1, *input_shape], tf.float32))
    return convert_variables_to_constants_v2(concrete).graph.as_graph_def()


def profile_flops(model: keras.Model, input_shape) -> int:
    """Total float ops of one forward pass, counted on the frozen graph.

    Counting after tracing is what makes this valid for this package: a
    layer-tree walk stops at a custom subclassed layer, a frozen graph has
    already dissolved ``BasicBlock``/``BottleneckBlock`` into primitive ops.
    """
    graph_def = _frozen_graph_def(model, input_shape)
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        result = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=tf.compat.v1.RunMetadata(),
            cmd="scope",
            options=opts,
        )
    return result.total_float_ops


def test_the_profiler_agrees_with_a_hand_computed_conv_within_5_percent() -> None:
    """CALIBRATION ARM. Run before any model number may be trusted.

    A single convolution whose FLOP count is computable by hand:
    3 -> 64 channels, 7x7 kernel, stride 2, no bias, 224x224 input, so a
    112x112 output. That is 112*112*64*3*7*7 multiply-accumulates.
    """
    analytic_macs = 112 * 112 * 64 * 3 * 7 * 7
    assert analytic_macs == 118_013_952, "the hand computation itself changed"

    inputs = keras.Input(shape=INPUT_SHAPE)
    outputs = keras.layers.Conv2D(
        64, 7, strides=2, padding="same", use_bias=False
    )(inputs)
    measured = profile_flops(keras.Model(inputs, outputs), INPUT_SHAPE)

    # The instrument counts the multiply and the add separately.
    error = abs(measured - 2 * analytic_macs) / (2 * analytic_macs)
    assert error <= 0.05, (
        f"calibration FAILED: profiler {measured:,} vs analytic 2*MACs "
        f"{2 * analytic_macs:,} ({error:.2%} error). Under this plan's "
        f"pre-registered rule, no model FLOP number may be published."
    )

    # Pin the unit itself: a silent 1x-vs-2x mixup is a 2x documentation error.
    assert measured == 2 * analytic_macs, (
        f"the profiler's unit changed: {measured:,} is not 2x {analytic_macs:,}"
    )


def test_the_profiler_descends_into_a_custom_subclassed_layer() -> None:
    """DESCENT ARM. A layer-tree walk would stop at ``CustomBlock``."""

    class CustomBlock(keras.layers.Layer):
        """Deliberately opaque: no Functional graph inside, like ResNet's blocks."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.conv_1 = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)
            self.conv_2 = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)

        def call(self, inputs):
            return self.conv_2(self.conv_1(inputs))

    shape = (56, 56, 32)
    analytic = 2 * 2 * (56 * 56 * 32 * 32 * 3 * 3)

    flat_in = keras.Input(shape=shape)
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(flat_in)
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    flat_flops = profile_flops(keras.Model(flat_in, x), shape)

    nested_in = keras.Input(shape=shape)
    nested_model = keras.Model(nested_in, CustomBlock()(nested_in))
    nested_flops = profile_flops(nested_model, shape)

    assert flat_flops == analytic, f"flat model {flat_flops:,} != {analytic:,}"
    assert nested_flops == flat_flops, (
        f"the profiler did NOT descend into the subclassed layer: "
        f"{nested_flops:,} vs {flat_flops:,}. No model FLOP number may be "
        f"published under this plan's pre-registered rule."
    )


@pytest.mark.parametrize("variant", sorted(EXPECTED_TRAINABLE))
def test_the_readme_parameter_columns_reproduce(variant: str) -> None:
    """Both parameter columns of README section 17, re-derived."""
    model = ResNet.from_variant(variant, num_classes=1000, input_shape=INPUT_SHAPE)
    model.build((1, *INPUT_SHAPE))

    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)

    assert trainable == EXPECTED_TRAINABLE[variant], (
        f"{variant} trainable params {trainable:,} != README "
        f"{EXPECTED_TRAINABLE[variant]:,}"
    )
    assert non_trainable == EXPECTED_NON_TRAINABLE[variant], (
        f"{variant} non-trainable params {non_trainable:,} != README "
        f"{EXPECTED_NON_TRAINABLE[variant]:,}"
    )
    assert model.count_params() == trainable + non_trainable


@pytest.mark.parametrize("variant", sorted(EXPECTED_FLOPS))
def test_the_readme_flops_column_reproduces(variant: str) -> None:
    """README section 17's FLOPs column, re-derived on the frozen graph."""
    model = ResNet.from_variant(variant, num_classes=1000, input_shape=INPUT_SHAPE)
    measured = profile_flops(model, INPUT_SHAPE)
    expected = EXPECTED_FLOPS[variant]

    assert measured == expected, (
        f"{variant}: {measured:,} total_float_ops != README {expected:,} "
        f"({measured / 1e9:.3f} GFLOPs vs {expected / 1e9:.3f} GFLOPs)"
    )

    # And the MACs column, which is the one comparable to published figures.
    published_macs = {"resnet18": 1.8, "resnet50": 4.1}[variant]
    measured_macs = measured / 2e9
    assert abs(measured_macs - published_macs) / published_macs < 0.02, (
        f"{variant}: {measured_macs:.3f} GMACs is more than 2% from the "
        f"commonly published {published_macs} GMACs"
    )

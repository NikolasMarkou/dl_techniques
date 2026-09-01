"""Direct guards on ``yolo12_blocks.yolo12_conv_block`` -- the helper all 20 migrated sites use.

Why this file exists
--------------------
Plan `plan-2026-09-01T055648-e6d380a5` deleted ``yolo12_blocks.ConvBlock`` and routed every
one of its 20 construction sites through the module-level factory
:func:`~dl_techniques.layers.yolo12_blocks.yolo12_conv_block`, which is also the sole consumer
of ``YOLO12_NORM_KWARGS`` (the D-005 / D-067 single home for the BatchNorm pair). After that
migration the helper is a genuine choke point: **six** independent facts about every
convolution in the YOLOv12 tree are stated once, here, and nowhere else.

An iteration-1 adversarial review found the helper had ZERO direct tests. The gates the plan
named could not see a regression that flipped ``use_bias`` to ``True``, moved
``kernel_initializer`` off ``he_normal``, or dropped ``momentum`` from the norm kwargs --
because the whole-model equivalence check transfers weights by ordered ``set_weights`` (a
bias would change the shape sequence, but only *if* someone re-ran it) and every numeric
comparison in this plan runs ``training=False``, the one regime in which BatchNorm
``momentum`` has no effect on the output at all.

What is asserted here is therefore the CONFIGURATION, statically, with no forward pass:
a wrong value is caught at construction rather than waiting for a training run to diverge
silently. See ``decisions.md`` D-005 for why the norm pair is threaded as data, and D-067
(``plan-2026-08-19T163559-499b6f0e``) for why the two numbers are what they are -- in
particular that ``momentum=0.97`` is the Keras transcription of Ultralytics' ``0.03``,
because the two frameworks define momentum with OPPOSITE senses.

What this file deliberately does NOT do
---------------------------------------
It does not re-measure equivalence against the pre-move class; that is
``tests/test_layers/test_the_yolo12_relocation_is_equivalent.py``. It does not census the
assembled model; that is ``tests/test_models/test_yolo12/test_the_relocation_is_a_model_level_noop.py``.
This file is the unit-level contract only.
"""

import keras
import pytest

from dl_techniques.layers import standard_blocks
from dl_techniques.layers.yolo12_blocks import YOLO12_NORM_KWARGS, yolo12_conv_block


class TestTheHelperReturnsTheSharedConvBlock:
    """The whole point of the migration: one ConvBlock class, not two."""

    def test_it_returns_a_standard_blocks_conv_block(self) -> None:
        block = yolo12_conv_block(filters=8)
        assert type(block) is standard_blocks.ConvBlock, (
            f"yolo12_conv_block returned {type(block)!r}. The plan's entire premise is "
            "that yolo12 has NO ConvBlock of its own -- a second class reappearing here "
            "re-creates the duplication that was removed."
        )

    def test_the_returned_block_is_unbuilt(self) -> None:
        """Callers build it into their own graph; a pre-built block would carry weights."""
        assert not yolo12_conv_block(filters=8).built


class TestTheSixPinnedDefaults:
    """Every convolution in the YOLOv12 tree is bias-free, He-initialised, BN'd, SiLU'd."""

    def test_it_is_bias_free(self) -> None:
        block = yolo12_conv_block(filters=8)
        assert block.use_bias is False, (
            "yolo12 convolutions are bias-free because the BatchNorm that follows carries "
            "its own beta; a bias here is a redundant parameter that also breaks ordered "
            "set_weights transfer against any pre-existing checkpoint."
        )
        assert block.conv.use_bias is False, (
            "the block records use_bias=False but its Conv2D was built with a bias -- the "
            "flag is not reaching the layer it describes"
        )

    def test_the_kernel_initializer_is_he_normal(self) -> None:
        block = yolo12_conv_block(filters=8)
        assert isinstance(block.kernel_initializer, keras.initializers.HeNormal), (
            f"kernel_initializer resolved to {block.kernel_initializer!r}, not HeNormal. "
            "ConvBlock's OWN default is 'glorot_uniform'; the yolo12 helper exists partly "
            "to override it, so a regression here is silent and inherits the wrong default."
        )

    def test_the_normalization_is_batch_norm(self) -> None:
        block = yolo12_conv_block(filters=8)
        assert block.normalization_type == "batch_norm"
        assert isinstance(block.norm, keras.layers.BatchNormalization)

    def test_the_norm_carries_both_halves_of_the_d067_pair(self) -> None:
        """Epsilon AND momentum. The census oracle buckets epsilon only, so momentum has
        no other guard anywhere in the tree at the model level."""
        block = yolo12_conv_block(filters=8)
        assert block.normalization_kwargs == {"epsilon": 1e-3, "momentum": 0.97}, (
            f"norm kwargs are {block.normalization_kwargs}, not the D-067 pair. "
            "create_normalization_layer SILENTLY falls back to Keras' 1e-6/0.99 when a "
            "key is missing -- no raise, no shape change, only different inference and a "
            "different moving-average time constant."
        )
        assert block.norm.epsilon == pytest.approx(1e-3, rel=0, abs=0.0)
        assert block.norm.momentum == pytest.approx(0.97, rel=0, abs=0.0)

    def test_the_helper_reads_the_single_home_rather_than_repeating_the_literals(self) -> None:
        """D-005: ``YOLO12_NORM_KWARGS`` is the ONE home; the helper must not fork it."""
        assert YOLO12_NORM_KWARGS == {"epsilon": 1e-3, "momentum": 0.97}
        assert yolo12_conv_block(filters=8).normalization_kwargs == YOLO12_NORM_KWARGS

    def test_the_helper_copies_the_norm_kwargs_rather_than_aliasing_them(self) -> None:
        """Two blocks must not share one mutable dict with the module-level constant."""
        a, b = yolo12_conv_block(filters=8), yolo12_conv_block(filters=8)
        assert a.normalization_kwargs is not YOLO12_NORM_KWARGS
        assert a.normalization_kwargs is not b.normalization_kwargs


class TestTheActivationRouting:
    """``activation=True/False`` -> silu / linear. The False path has one production site."""

    def test_true_gives_silu(self) -> None:
        block = yolo12_conv_block(filters=8, activation=True)
        assert block.activation_type == "silu"

    def test_the_default_is_silu(self) -> None:
        assert yolo12_conv_block(filters=8).activation_type == "silu"

    def test_false_gives_linear_not_a_dropped_activation(self) -> None:
        block = yolo12_conv_block(filters=8, activation=False)
        assert block.activation_type == "linear", (
            f"activation=False produced activation_type={block.activation_type!r}. "
            "'linear' is a WEIGHTLESS EXACT IDENTITY, which is what the pre-move class's "
            "activation-off path did; anything else (relu, None, a dropped attribute) is a "
            "behaviour change at the one production site that passes activation=False."
        )

    def test_the_linear_path_is_weightless(self) -> None:
        """An identity that carried weights would change the ordered set_weights sequence."""
        block = yolo12_conv_block(filters=4, activation=False)
        block.build((None, 8, 8, 4))
        assert block.activation.weights == [], (
            f"the activation-off path built {len(block.activation.weights)} weight(s); it "
            "must be a weightless identity"
        )

    def test_the_two_activation_routes_actually_differ_at_the_output(self) -> None:
        """Anti-vacuity: if silu and linear produced the same tensor, every assertion
        above would be pinning a label with no consequence."""
        x = keras.ops.convert_to_tensor(
            [[[[-2.0, -1.0, 1.0, 2.0]]]], dtype="float32")
        outs = []
        for flag in (True, False):
            keras.utils.set_random_seed(0)
            block = yolo12_conv_block(filters=4, activation=flag)
            block.build((None, 1, 1, 4))
            outs.append(keras.ops.convert_to_numpy(block(x, training=False)))
        assert not keras.ops.all(keras.ops.isclose(outs[0], outs[1])), (
            "silu and linear produced indistinguishable outputs -- the activation routing "
            "assertions above are vacuous"
        )


class TestTheForwardedArguments:
    """``groups`` and the rest reach the Conv2D rather than being swallowed by **kwargs."""

    @pytest.mark.parametrize("groups", [1, 2, 4])
    def test_groups_forwards_to_the_convolution(self, groups: int) -> None:
        block = yolo12_conv_block(filters=8, groups=groups)
        assert block.groups == groups
        assert block.conv.groups == groups, (
            f"groups={groups} was recorded on the block but the Conv2D has "
            f"groups={block.conv.groups}. Four yolo12 sites pass groups; a swallowed "
            "value silently converts a grouped convolution into a dense one -- same "
            "output shape, ~groups x the parameters, different model."
        )

    def test_a_depthwise_group_count_changes_the_parameter_count(self) -> None:
        """Anti-vacuity for ``groups``: the flag must have a measurable consequence."""
        dense = yolo12_conv_block(filters=8, groups=1, kernel_size=3)
        deep = yolo12_conv_block(filters=8, groups=8, kernel_size=3)
        dense.build((None, 8, 8, 8))
        deep.build((None, 8, 8, 8))
        assert dense.conv.kernel.shape[2] == 8
        assert deep.conv.kernel.shape[2] == 1, (
            "groups=8 on an 8-channel input did not produce a depthwise kernel"
        )

    @pytest.mark.parametrize(
        "kwarg,value,attr",
        [
            ("kernel_size", 1, "kernel_size"),
            ("strides", 2, "strides"),
            ("padding", "valid", "padding"),
            ("filters", 13, "filters"),
        ],
    )
    def test_the_plain_geometry_arguments_forward(self, kwarg, value, attr) -> None:
        block = yolo12_conv_block(**{"filters": 8, kwarg: value})
        assert getattr(block, attr) == value

    def test_use_bias_and_kernel_initializer_are_overridable(self) -> None:
        """The defaults are pinned above; they must still be arguments, not constants."""
        block = yolo12_conv_block(filters=8, use_bias=True,
                                  kernel_initializer="glorot_uniform")
        assert block.use_bias is True
        assert isinstance(block.kernel_initializer, keras.initializers.GlorotUniform)

    def test_the_kernel_regularizer_forwards(self) -> None:
        reg = keras.regularizers.L2(1e-4)
        block = yolo12_conv_block(filters=8, kernel_regularizer=reg)
        assert block.kernel_regularizer is not None
        assert block.conv.kernel_regularizer is not None

    def test_layer_kwargs_such_as_name_reach_the_base_class(self) -> None:
        assert yolo12_conv_block(filters=8, name="probe_block").name == "probe_block"

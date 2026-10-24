"""The depth-scaled residual-projection init reaches every feed-forward type the block accepts.

plan-2026-10-23T190557-972a9ac5 step 13.1 (review pass 6, concern 1). The block hands
``output_kernel_initializer`` (``initializer_range / sqrt(2 * num_layers)``) to the
shared FFN helper, whose registry intersection keeps the key for ``'mlp'`` only. For
every other type it was silently dropped, so the output projection started at the
plain range (0.0176 instead of 0.0036 at depth 12), and two types (``gelu_tanh``,
``orthoglu``) did not read ``initializer_range`` at all, while the docstring and the
README promised the scaling for any registry type. ``TransformerLayer`` refuses this
exact combination by name (transformer.py, ``_reject_unsupported_residual_output_init``);
the block now does the same.

Ruling (step 13.1): a type that cannot take the scaled output init is REFUSED at
construction, naming the type and the supported ones. Scaling a foreign layer's
variables after the build would need a hand-kept table of "which kernel is the output
projection" for ten layouts, which is exactly the lockstep invariant this repository
avoids; ``TransformerLayer`` is the block to use for those types.

The expected deviations come from the truncated-normal distribution
(``scipy.stats.truncnorm``) and the formula ``range / sqrt(2 * depth)``; nothing is read
back from the code under test except the weights it built.

Run on CPU (``CUDA_VISIBLE_DEVICES=``).
"""

import math
from typing import Dict, List

import pytest
import scipy.stats

from dl_techniques.layers.ffn.factory import FFN_REGISTRY
from dl_techniques.models.language.topolm import TopoLM, TopoLMBlock, create_topolm
from dl_techniques.models.language.topolm.components import accepted_ffn_types

from .test_topolm_ffn import make_block, weights_by_suffix

DEPTH = 12
RANGE = 0.02
BIG_H, BIG_I, BIG_HEADS = 256, 1024, 4
TRUNCATION_FACTOR = float(scipy.stats.truncnorm(-2.0, 2.0).std())

#: For each type the block may build: the suffix of its OUTPUT projection kernel in
#: ``weights_by_suffix``. Written by hand from the layer's own structure, once. A type
#: that becomes constructible without a row here fails ``test_every_accepted_type_has_a_measured_row``.
OUTPUT_PROJECTION: Dict[str, str] = {"mlp": "fc2/kernel"}

#: Every registry type whose output width is ``output_dim`` (the ones that fit a
#: residual branch) -- read from the registry, not from the block's own list.
WIDTH_COMPATIBLE: List[str] = sorted(
    name for name, info in FFN_REGISTRY.items()
    if info.get("output_dim_param") == "output_dim"
)

#: The types the review measured as silently unscaled (swiglu, geglu, glu, reglu,
#: bilinear, differential, residual, lowrank, squared_relu, swin_mlp) or ignoring
#: ``initializer_range`` (gelu_tanh, orthoglu), plus the two that need arguments.
REFUSED_TYPES = [
    "bilinear", "counting", "differential", "geglu", "gelu_tanh", "glu", "logic",
    "lowrank", "monarch", "orthoglu", "reglu", "residual", "squared_relu",
    "swiglu", "swin_mlp",
]


def expected_residual_std(depth: int = DEPTH, init_range: float = RANGE) -> float:
    return TRUNCATION_FACTOR * init_range / math.sqrt(2.0 * depth)


def close(measured: float, expected: float, tolerance: float = 0.03) -> bool:
    return abs(measured / expected - 1.0) < tolerance


class TestTheRegistrySurfaceIsWhatTheReviewMeasured:
    def test_the_refused_list_is_the_width_compatible_registry_minus_the_accepted(self):
        """The literal list above equals the registry's, so a new type is noticed."""
        assert sorted(set(WIDTH_COMPATIBLE) - set(accepted_ffn_types())) == REFUSED_TYPES

    def test_only_a_type_that_declares_the_output_initializer_is_accepted(self):
        declaring = sorted(
            name for name in WIDTH_COMPATIBLE
            if "output_kernel_initializer" in FFN_REGISTRY[name]["optional_params"]
        )
        assert accepted_ffn_types() == declaring == ["mlp"]

    def test_every_accepted_type_has_a_measured_row(self):
        assert sorted(OUTPUT_PROJECTION) == accepted_ffn_types()


class TestAnAcceptedTypeStartsItsOutputProjectionAtTheDepthRule:
    @pytest.mark.parametrize("ffn_type", sorted(OUTPUT_PROJECTION))
    @pytest.mark.parametrize("depth", [1, 2, 12, 24])
    def test_the_output_projection_std_follows_one_over_root_two_depth(self, ffn_type, depth):
        block = make_block(hidden=BIG_H, inner=BIG_I, heads=BIG_HEADS,
                           ffn_type=ffn_type, num_layers=depth)
        measured = float(weights_by_suffix(block.ffn)[OUTPUT_PROJECTION[ffn_type]].std())
        assert close(measured, expected_residual_std(depth)), (
            ffn_type, depth, measured, expected_residual_std(depth))

    @pytest.mark.parametrize("ffn_type", sorted(OUTPUT_PROJECTION))
    def test_the_plain_depth_one_value_fails_the_depth_twelve_oracle(self, ffn_type):
        """The instrument can fail: an unscaled std is 3.46x the depth-12 target."""
        block = make_block(hidden=BIG_H, inner=BIG_I, heads=BIG_HEADS,
                           ffn_type=ffn_type, num_layers=1)
        measured = float(weights_by_suffix(block.ffn)[OUTPUT_PROJECTION[ffn_type]].std())
        assert not close(measured, expected_residual_std(DEPTH))

    @pytest.mark.parametrize("ffn_type", sorted(OUTPUT_PROJECTION))
    def test_the_model_hands_the_depth_to_the_accepted_type(self, ffn_type):
        model = TopoLM(vocab_size=64, embed_dim=BIG_H, depth=DEPTH, num_heads=BIG_HEADS,
                       max_seq_len=16, radius=1, ffn_type=ffn_type, name="topolm")
        model.build((None, 8))
        for block in model.blocks:
            measured = float(weights_by_suffix(block.ffn)[OUTPUT_PROJECTION[ffn_type]].std())
            assert close(measured, expected_residual_std()), block.name


class TestATypeThatCannotTakeTheScalingIsRefused:
    @pytest.mark.parametrize("ffn_type", REFUSED_TYPES)
    def test_the_block_refuses_it_by_name_and_lists_the_supported_types(self, ffn_type):
        with pytest.raises(ValueError) as excinfo:
            make_block(build=False, ffn_type=ffn_type)
        message = str(excinfo.value)
        assert repr(ffn_type) in message
        assert "residual" in message
        assert "output_kernel_initializer" in message
        assert "['mlp']" in message

    @pytest.mark.parametrize("ffn_type", ["swiglu", "geglu", "gelu_tanh", "orthoglu"])
    def test_the_model_refuses_it_too(self, ffn_type):
        with pytest.raises(ValueError, match=rf"ffn_type {ffn_type!r}.*\['mlp'\]"):
            TopoLM(vocab_size=50, embed_dim=64, depth=1, num_heads=4,
                   max_seq_len=16, radius=1, ffn_type=ffn_type)

    def test_from_variant_and_the_factory_refuse_it_too(self):
        for build in (TopoLM.from_variant, create_topolm):
            with pytest.raises(ValueError, match=r"ffn_type 'swiglu'"):
                build("tiny", vocab_size=50, depth=1, max_seq_len=16, ffn_type="swiglu")

    def test_a_row_of_the_variant_table_with_such_a_type_is_refused(self, monkeypatch):
        row = {"embed_dim": 64, "depth": 1, "num_heads": 4, "ffn_intermediate_size": 128,
               "max_seq_len": 16, "radius": 1, "ffn_type": "geglu"}
        monkeypatch.setitem(TopoLM.MODEL_VARIANTS, "with_geglu", row)
        with pytest.raises(ValueError, match=r"ffn_type 'geglu'"):
            TopoLM.from_variant("with_geglu", vocab_size=50)

    def test_an_unregistered_type_keeps_its_own_message(self):
        with pytest.raises(ValueError, match=r"ffn_type must be one of .*'not_an_ffn'"):
            make_block(build=False, ffn_type="not_an_ffn")

    def test_the_refusal_names_the_alternative(self):
        with pytest.raises(ValueError, match="TransformerLayer"):
            make_block(build=False, ffn_type="swiglu")

    def test_the_accepted_type_still_builds(self):
        assert isinstance(make_block(ffn_type="mlp"), TopoLMBlock)

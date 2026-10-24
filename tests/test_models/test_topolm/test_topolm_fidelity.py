"""TopoLM's init and activation follow the released architecture, checked independently.

plan-2026-10-23T190557-972a9ac5 step 11 (D-031): the fidelity audit compared this
repository's TopoLM with the authors' released code and found

* the residual-path output projections initialised at ``std / sqrt(2)`` with no depth
  factor, where the released ``model.py`` uses ``std / sqrt(2 * n_layer)`` (3.5x too
  large at depth 12), under a comment that claimed GPT-2 parity;
* the feed-forward activation defaulting to the tanh GELU at the model level while the
  block itself defaulted to the exact one (two defaults for one thing); the released
  code uses ``nn.GELU()``, the exact erf form.

Nothing here reads a number back from the code under test. The expected standard
deviation is computed from the truncated-normal distribution itself
(``scipy.stats.truncnorm``) and the formula ``range / sqrt(2 * depth)``; the activation
values come from numpy/scipy applied to the block's own weights
(``test_topolm_ffn.assert_ffn_forward``).

Run on CPU (``CUDA_VISIBLE_DEVICES=``).
"""

import math
from typing import Any, Dict

import keras
import numpy as np
import pytest
import scipy.stats

from dl_techniques.layers.activations import gelu_tanh
from dl_techniques.layers.activations.exact_activations import EXACT_ACTIVATIONS
from dl_techniques.models.language.topolm import TopoLM, TopoLMBlock

from .test_topolm_ffn import (
    BATCH,
    SEQ,
    assert_ffn_forward,
    make_block,
    make_model,
    randomize,
    run_with_spies,
    weights_by_suffix,
)

# A layer large enough that the sample std of a kernel is within a fraction of a percent
# of the distribution's std: 512 x 512 = 262,144 draws for the attention projection and
# 2048 x 512 = 1,048,576 for the feed-forward ones (relative noise 1 / sqrt(2 N) is
# 0.14% and 0.07%), so the 3% tolerance below is a bound on the formula, not on luck.
BIG_H, BIG_I, BIG_HEADS, DEPTH = 512, 2048, 8, 12
RANGE = 0.02
TOLERANCE = 0.03

#: Keras' TruncatedNormal truncates at two standard deviations; the std of the
#: truncated sample is this multiple of ``stddev``. Computed from the distribution,
#: not read back from the initializer.
TRUNCATION_FACTOR = float(scipy.stats.truncnorm(-2.0, 2.0).std())


def expected_residual_std(depth: int, init_range: float = RANGE) -> float:
    """The released code's rule: ``range / sqrt(2 * n_layer)``, truncated at 2 sigma."""
    return TRUNCATION_FACTOR * init_range / math.sqrt(2.0 * depth)


def expected_plain_std(init_range: float = RANGE) -> float:
    return TRUNCATION_FACTOR * init_range


def kernel_std(weights: Dict[str, np.ndarray], key: str) -> float:
    return float(weights[key].std())


def big_block(**kwargs: Any) -> TopoLMBlock:
    return make_block(hidden=BIG_H, inner=BIG_I, heads=BIG_HEADS, **kwargs)


def close(measured: float, expected: float) -> bool:
    return abs(measured / expected - 1.0) < TOLERANCE


# ---------------------------------------------------------------------
# (a) the residual-projection init carries the depth factor
# ---------------------------------------------------------------------


class TestTheResidualProjectionsAreScaledByDepth:
    def test_the_oracle_distinguishes_the_old_value_from_the_new_one(self):
        """The instrument can fail: the old std is 3.46x the right one at depth 12."""
        old = TRUNCATION_FACTOR * RANGE / math.sqrt(2.0)
        assert not close(old, expected_residual_std(DEPTH))
        assert close(expected_residual_std(1), old)
        # and the factor itself is the distribution's, not 1
        assert 0.87 < TRUNCATION_FACTOR < 0.89

    def test_the_attention_output_and_the_feed_forward_output_follow_the_depth_rule(self):
        block = big_block(num_layers=DEPTH)
        attention = weights_by_suffix(block.attention)
        ffn = weights_by_suffix(block.ffn)
        expected = expected_residual_std(DEPTH)
        assert close(kernel_std(attention, "proj/kernel"), expected), (
            kernel_std(attention, "proj/kernel"), expected)
        assert close(kernel_std(ffn, "fc2/kernel"), expected), (
            kernel_std(ffn, "fc2/kernel"), expected)

    def test_the_projections_that_are_not_on_the_residual_path_keep_the_range(self):
        block = big_block(num_layers=DEPTH)
        expected = expected_plain_std()
        assert close(kernel_std(weights_by_suffix(block.attention), "qkv/kernel"), expected)
        assert close(kernel_std(weights_by_suffix(block.ffn), "fc1/kernel"), expected)

    @pytest.mark.parametrize("depth", [1, 2, 6, 24])
    def test_the_std_scales_as_one_over_root_two_depth(self, depth):
        block = big_block(num_layers=depth)
        expected = expected_residual_std(depth)
        assert close(kernel_std(weights_by_suffix(block.ffn), "fc2/kernel"), expected)
        assert close(kernel_std(weights_by_suffix(block.attention), "proj/kernel"), expected)

    def test_the_old_value_fails_the_depth_twelve_oracle(self):
        """A depth-1 block (std / sqrt(2), the old value) must NOT pass at depth 12."""
        block = big_block(num_layers=1)
        expected = expected_residual_std(DEPTH)
        assert not close(kernel_std(weights_by_suffix(block.ffn), "fc2/kernel"), expected)
        assert not close(kernel_std(weights_by_suffix(block.attention), "proj/kernel"),
                         expected)

    def test_the_range_enters_the_rule(self):
        block = big_block(num_layers=DEPTH, initializer_range=0.05)
        expected = expected_residual_std(DEPTH, 0.05)
        assert close(kernel_std(weights_by_suffix(block.ffn), "fc2/kernel"), expected)

    def test_the_model_hands_its_depth_to_every_block(self):
        model = TopoLM(vocab_size=64, embed_dim=BIG_H, depth=DEPTH, num_heads=BIG_HEADS,
                       max_seq_len=16, radius=1, name="topolm")
        model.build((None, 8))
        expected = expected_residual_std(DEPTH)
        assert [block.num_layers for block in model.blocks] == [DEPTH] * DEPTH
        for block in model.blocks:
            ffn = weights_by_suffix(block.ffn)
            attention = weights_by_suffix(block.attention)
            assert close(kernel_std(ffn, "fc2/kernel"), expected), block.name
            assert close(kernel_std(attention, "proj/kernel"), expected), block.name
            assert close(kernel_std(ffn, "fc1/kernel"), expected_plain_std()), block.name

    def test_the_embeddings_keep_the_initializer_range(self):
        model = TopoLM(vocab_size=4096, embed_dim=BIG_H, depth=DEPTH, num_heads=BIG_HEADS,
                       max_seq_len=1024, radius=1, name="topolm")
        model.build((None, 8))
        expected = expected_plain_std()
        for layer in (model.word_embeddings, model.positional_embeddings):
            std = float(keras.ops.convert_to_numpy(layer.embeddings).std())
            assert close(std, expected), (layer.name, std, expected)

    def test_a_model_of_another_depth_gets_its_own_factor(self):
        shallow = TopoLM(vocab_size=64, embed_dim=BIG_H, depth=2, num_heads=BIG_HEADS,
                         max_seq_len=16, radius=1, name="topolm")
        shallow.build((None, 8))
        ffn = weights_by_suffix(shallow.blocks[0].ffn)
        assert close(kernel_std(ffn, "fc2/kernel"), expected_residual_std(2))
        assert not close(kernel_std(ffn, "fc2/kernel"), expected_residual_std(DEPTH))

    @pytest.mark.parametrize("bad", [0, -3, 1.5, True, "12", None])
    def test_a_bad_num_layers_is_refused_by_name(self, bad):
        with pytest.raises(ValueError, match=r"num_layers must be a positive integer"):
            make_block(build=False, num_layers=bad)

    def test_num_layers_is_serialized(self):
        block = make_block(build=False, num_layers=7)
        config = block.get_config()
        assert config["num_layers"] == 7
        assert TopoLMBlock.from_config(config).num_layers == 7
        model = make_model(depth=2)
        assert [b.get_config()["num_layers"] for b in model.blocks] == [2, 2]

    def test_a_standalone_block_is_a_one_layer_stack(self):
        assert make_block(build=False).num_layers == 1


# ---------------------------------------------------------------------
# (b) one activation for the whole feed-forward, exact GELU by default
# ---------------------------------------------------------------------


class TestTheDefaultActivationIsTheExactGelu:
    def test_the_default_block_matches_the_erf_oracle(self):
        assert_ffn_forward(make_block(), "gelu")

    def test_the_default_model_matches_the_erf_oracle(self):
        model = make_model()
        assert_ffn_forward(model.blocks[0], "gelu", eps=model.layer_norm_eps)

    def test_the_default_model_does_not_match_the_tanh_oracle(self):
        """The oracle can tell the two apart: the tanh form is measurably different."""
        model = make_model()
        with pytest.raises(AssertionError):
            assert_ffn_forward(model.blocks[0], "gelu_tanh", eps=model.layer_norm_eps)

    def test_the_model_and_the_block_share_one_default(self):
        assert make_model().get_config()["activation"] == "gelu"
        assert make_block(build=False).get_config()["activation"] == "gelu"

    def test_the_whole_feed_forward_uses_one_resolved_activation(self):
        model = make_model(depth=3)
        resolved = {id(block.ffn.activation_fn) for block in model.blocks}
        assert len(resolved) == 1
        assert model.blocks[0].ffn.activation_fn is EXACT_ACTIVATIONS["gelu"]

    @pytest.mark.parametrize("spec", ["gelu_tanh", gelu_tanh],
                             ids=["string", "callable"])
    def test_the_tanh_form_is_available_and_matches_its_oracle(self, spec):
        block = make_block(activation=spec)
        assert_ffn_forward(block, "gelu_tanh")
        with pytest.raises(AssertionError):
            assert_ffn_forward(make_block(activation=spec), "gelu")

    def test_the_model_accepts_the_tanh_form_by_name(self):
        model = make_model(activation="gelu_tanh")
        assert_ffn_forward(model.blocks[0], "gelu_tanh", eps=model.layer_norm_eps)
        assert model.get_config()["activation"] == "gelu_tanh"

    def test_the_two_forms_differ_measurably_on_the_same_weights(self):
        x = np.random.default_rng(4).standard_normal((BATCH, SEQ, 64)).astype("float32")
        outs = {}
        for name in ("gelu", "gelu_tanh"):
            block = make_block(activation=name)
            randomize(block, seed=5)
            outs[name] = run_with_spies(block, x)[2]
        gap = float(np.abs(outs["gelu"] - outs["gelu_tanh"]).max())
        assert gap > 1e-5, gap
        assert gap < 1e-1, gap  # the same function up to its approximation error

    def test_a_non_default_activation_still_round_trips(self, tmp_path):
        model = make_model(activation="gelu_tanh")
        ids = np.random.default_rng(0).integers(0, 50, size=(2, 8)).astype("int32")
        before = keras.ops.convert_to_numpy(model(ids, training=False)["logits"])
        path = str(tmp_path / "m.keras")
        model.save(path)
        restored = keras.models.load_model(path)
        assert restored.get_config()["activation"] == "gelu_tanh"
        after = keras.ops.convert_to_numpy(restored(ids, training=False)["logits"])
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)

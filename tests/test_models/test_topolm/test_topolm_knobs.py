"""TopoLM's bias, norm-centre and tap-placement knobs, checked against independent forms.

plan-2026-10-23T190557-972a9ac5 step 11.1 (D-031 c, d). The released TopoLM code has
no feed-forward bias and no LayerNorm beta, so the defaults here are the upstream's:

* ``use_bias=False`` (feed-forward projections; the attention never has one),
* ``norm_center=False`` (LayerNorm ``center``; RMS norm has no beta whatever it says),
* ``tap_placement='branch'``: the paper text applies the loss to the branch output
  "prior to normalization and addition into the residual stream". ``'residual'``
  taps the post-add tensor, which is what the released training yaml's
  ``with_resid: True`` does.

Nothing here reads a count or a tensor back from the code under test. The parameter
counts come from the closed form below; the tapped tensors come from calling the
block's own sub-layers one by one and adding by hand, and are compared with what the
real :class:`SpatialSmoothness` tap and the ``taps=`` capture were handed.

Run on CPU (``CUDA_VISIBLE_DEVICES=``).
"""

from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from dl_techniques.models.language.topolm import (
    MODEL_VARIANTS,
    TopoLM,
    TopoLMBlock,
)
from dl_techniques.layers.norms.rms_norm import RMSNorm

from .test_topolm_ffn import (
    BATCH,
    EVERY_OPTION,
    H,
    HEADS,
    I,
    SEQ,
    Spy,
    VARIANT_NAMES,
    attention_params,
    make_block,
    make_model,
    mlp_params,
    randomize,
    trainable_count,
    _ids,
)

#: Every knob of this step, set away from its default.
KNOBS_ON: Dict[str, Any] = dict(
    use_bias=True, norm_center=True, tap_placement="residual",
)


# ---------------------------------------------------------------------
# independent closed forms
# ---------------------------------------------------------------------


def norm_count(hidden: int, center: bool, kind: str = "layer_norm") -> int:
    """gamma, plus a beta only when centred; RMS norm is one scale whatever."""
    if kind == "rms_norm":
        return hidden
    return hidden * (2 if center else 1)


def block_count(hidden: int, inner: int, bias: bool, center: bool,
                kind: str = "layer_norm") -> int:
    """attention (no bias) + two norms + one MLP."""
    return (
        attention_params(hidden)
        + 2 * norm_count(hidden, center, kind)
        + mlp_params(hidden, inner, bias)
    )


def model_count(vocab: int, max_len: int, hidden: int, inner: int, depth: int,
                bias: bool, center: bool, kind: str = "layer_norm") -> int:
    """Tied embeddings + positions + ``depth`` blocks + the final norm."""
    return (
        vocab * hidden + max_len * hidden
        + depth * block_count(hidden, inner, bias, center, kind)
        + norm_count(hidden, center, kind)
    )


def paths_ending(layer: keras.layers.Layer, *suffixes: str) -> List[str]:
    return [w.path for w in layer.weights if w.path.endswith(suffixes)]


# ---------------------------------------------------------------------
# the defaults are the upstream's
# ---------------------------------------------------------------------


class TestTheDefaultsAreTheReleasedArchitectures:
    def test_the_documented_defaults(self):
        for config in (make_block(build=False).get_config(),
                       make_model().get_config()):
            assert config["use_bias"] is False
            assert config["norm_center"] is False
            assert config["tap_placement"] == "branch"

    def test_the_default_model_has_no_bias_and_no_beta_anywhere(self):
        model = make_model()
        assert paths_ending(model, "/bias", "/beta") == []
        assert any(p.endswith("/gamma") for p in [w.path for w in model.weights])

    def test_the_default_block_count_is_the_closed_form(self):
        block = make_block()
        assert trainable_count(block) == block_count(H, I, bias=False, center=False)

    def test_the_default_model_count_is_the_closed_form(self):
        model = make_model(depth=2)
        assert trainable_count(model) == model_count(
            50, 16, H, I, 2, bias=False, center=False)

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_every_variant_block_is_attention_norms_and_one_bare_mlp(self, variant):
        row = MODEL_VARIANTS[variant]
        h, i = row["embed_dim"], row["ffn_intermediate_size"]
        block = TopoLMBlock(
            hidden_size=h, num_heads=row["num_heads"], intermediate_size=i,
            radius=row.get("radius", 5), grid_dim=row.get("grid_dim", 2),
            name="block")
        block.build((None, 4, h))
        assert trainable_count(block) == block_count(h, i, False, False)

    def test_the_paper_variant_at_the_padded_gpt2_vocabulary_is_128_8_million(self):
        """0.5M of the 50,304 x 784 table aside, the released 12 x 784 model."""
        row = MODEL_VARIANTS["paper"]
        expected = model_count(50304, row["max_seq_len"], row["embed_dim"],
                               row["ffn_intermediate_size"], row["depth"],
                               bias=False, center=False)
        assert round(expected / 1e6, 1) == 128.8
        model = TopoLM.from_variant("paper", vocab_size=50304)
        model.build((None, 8))
        assert trainable_count(model) == expected


# ---------------------------------------------------------------------
# one weight family per knob
# ---------------------------------------------------------------------


class TestEachKnobOwnsExactlyItsWeights:
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("center", [False, True])
    def test_the_block_count_follows_both_knobs(self, bias, center):
        block = make_block(use_bias=bias, norm_center=center)
        assert trainable_count(block) == block_count(H, I, bias, center)

    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("center", [False, True])
    def test_the_model_count_follows_both_knobs(self, bias, center):
        model = make_model(depth=2, use_bias=bias, norm_center=center)
        assert trainable_count(model) == model_count(
            50, 16, H, I, 2, bias=bias, center=center)

    def test_use_bias_adds_only_the_two_feed_forward_biases_per_block(self):
        model = make_model(depth=2, use_bias=True)
        assert sorted(paths_ending(model, "/bias")) == sorted(
            f"topolm/block_{b}/ffn/fc{n}/bias" for b in range(2) for n in (1, 2))
        assert paths_ending(model, "/beta") == []

    def test_norm_center_adds_only_the_betas_of_every_norm_in_the_stack(self):
        model = make_model(depth=2, norm_center=True)
        expected = sorted(
            [f"topolm/block_{b}/{n}/beta" for b in range(2)
             for n in ("attention_norm", "ffn_norm")] + ["topolm/final_norm/beta"])
        assert sorted(paths_ending(model, "/beta")) == expected
        assert paths_ending(model, "/bias") == []

    def test_the_layer_norm_is_built_with_the_centre_flag(self):
        for center in (False, True):
            model = make_model(norm_center=center)
            norms = [model.final_norm, model.blocks[0].attention_norm,
                     model.blocks[0].ffn_norm]
            assert [n.center for n in norms] == [center] * 3

    def test_rms_norm_has_no_beta_whatever_the_flag_says(self):
        for center in (False, True):
            model = make_model(normalization_type="rms_norm", norm_center=center)
            assert isinstance(model.final_norm, RMSNorm)
            assert paths_ending(model, "/beta") == []
            assert trainable_count(model) == model_count(
                50, 16, H, I, 1, bias=False, center=center, kind="rms_norm")

    def test_a_norms_own_args_override_the_flag_for_that_norm_alone(self):
        block = make_block(attention_norm_args={"center": True})
        assert [w.path.split("/")[-1] for w in block.attention_norm.weights] == [
            "gamma", "beta"]
        assert [w.path.split("/")[-1] for w in block.ffn_norm.weights] == ["gamma"]

    def test_attention_never_has_a_bias_whatever_use_bias_says(self):
        assert paths_ending(make_block(use_bias=True).attention, "/bias") == []

    def test_a_bias_free_block_still_computes_dense_activation_dense(self):
        """The oracle of step 10 with the biases absent: ``gelu(h W1) W2``."""
        import math
        import scipy.special
        from .test_topolm_ffn import run_with_spies, np_layer_norm, weights_by_suffix

        block = make_block()
        randomize(block)
        x = np.random.default_rng(2).standard_normal((BATCH, SEQ, H)).astype("float32")
        _, attn_out, mlp_out = run_with_spies(block, x)
        hidden = x.astype("float64") + attn_out
        gamma = keras.ops.convert_to_numpy(block.ffn_norm.gamma)
        normed = np_layer_norm(hidden, gamma, 0.0, 1e-5)
        w = weights_by_suffix(block.ffn)
        pre = normed @ w["fc1/kernel"]
        expected = (0.5 * pre * (1.0 + scipy.special.erf(pre / math.sqrt(2.0)))) \
            @ w["fc2/kernel"]
        np.testing.assert_allclose(mlp_out, expected, rtol=0, atol=2e-5)


# ---------------------------------------------------------------------
# tap placement
# ---------------------------------------------------------------------


class RecordingTap(keras.layers.Layer):
    """Wraps a real tap and records the tensor the spatial loss is computed on."""

    def __init__(self, inner: keras.layers.Layer, **kwargs: Any) -> None:
        super().__init__(name=inner.name, **kwargs)
        self.inner = inner
        self.seen: List[np.ndarray] = []

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        self.seen.append(keras.ops.convert_to_numpy(x))
        return self.inner(x, training=training)


def recorded_block(**kwargs: Any) -> TopoLMBlock:
    """A block whose two taps record their inputs; built after the swap."""
    block = make_block(build=False, **kwargs)
    block.attention_tap = RecordingTap(block.attention_tap)
    block.mlp_tap = RecordingTap(block.mlp_tap)
    block.build((None, SEQ, H))
    randomize(block)
    return block


def independent_branches(block: TopoLMBlock, x: np.ndarray, position: str):
    """``(attn branch, hidden, mlp branch)`` from the block's own sub-layers."""
    run = lambda layer, v: keras.ops.convert_to_numpy(layer(v, training=False))
    if position == "pre":
        attn = run(block.attention, run(block.attention_norm, x))
        hidden = x + attn
        mlp = run(block.ffn, run(block.ffn_norm, hidden))
    else:
        attn = run(block.attention, x)
        hidden = run(block.attention_norm, x + attn)
        mlp = run(block.ffn, hidden)
    return attn, hidden, mlp


class TestTheTapSeesTheBranchOrTheSum:
    @pytest.mark.parametrize("position", ["pre", "post"])
    @pytest.mark.parametrize("placement", ["branch", "residual"])
    def test_the_tapped_tensor_is_the_branch_or_the_post_add_sum(self, placement, position):
        block = recorded_block(tap_placement=placement, normalization_position=position)
        x = np.random.default_rng(3).standard_normal((BATCH, SEQ, H)).astype("float32")
        spies = (Spy(name="spy_attn"), Spy(name="spy_mlp"))
        out = keras.ops.convert_to_numpy(block(x, taps=spies, training=False))

        attn, hidden, mlp = independent_branches(block, x, position)
        if placement == "branch":
            expected_attn, expected_mlp = attn, mlp
        else:
            expected_attn, expected_mlp = x + attn, hidden + mlp

        # what the real tap (and so the loss) was handed, and what the capture saw
        for seen, expected in ((block.attention_tap.seen[0], expected_attn),
                               (block.mlp_tap.seen[0], expected_mlp)):
            np.testing.assert_allclose(seen, expected, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(spies[0].seen[0], block.attention_tap.seen[0])
        np.testing.assert_array_equal(spies[1].seen[0], block.mlp_tap.seen[0])
        assert len(block.attention_tap.seen) == len(block.mlp_tap.seen) == 1

        # the forward is the identity either way: the block output is the same sum
        final = hidden + mlp
        if position == "post":
            norm = block.ffn_norm
            final = keras.ops.convert_to_numpy(norm(final, training=False))
        np.testing.assert_allclose(out, final, rtol=0, atol=1e-5)

    def test_the_two_placements_hand_the_taps_different_tensors(self):
        x = np.random.default_rng(3).standard_normal((BATCH, SEQ, H)).astype("float32")
        seen = {}
        for placement in ("branch", "residual"):
            block = recorded_block(tap_placement=placement)
            block(x, training=False)
            seen[placement] = block.attention_tap.seen[0]
        assert np.abs(seen["branch"] - seen["residual"]).max() > 0.1

    def test_the_forward_does_not_depend_on_the_placement(self):
        x = np.random.default_rng(3).standard_normal((BATCH, SEQ, H)).astype("float32")
        branch, residual = make_block(), make_block(tap_placement="residual")
        residual.set_weights(branch.get_weights())
        a = keras.ops.convert_to_numpy(branch(x, training=False))
        b = keras.ops.convert_to_numpy(residual(x, training=False))
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

    def test_the_spatial_loss_is_computed_on_the_chosen_tensor(self):
        """Same weights, same input: the penalty differs, so the placement is real."""
        x = np.random.default_rng(3).standard_normal((BATCH, SEQ, H)).astype("float32")
        losses = {}
        for placement in ("branch", "residual"):
            block = make_block(alpha=2.5, tap_placement=placement)
            randomize(block)
            block(x, training=True)
            losses[placement] = [float(keras.ops.convert_to_numpy(v))
                                 for v in block.losses]
        assert len(losses["branch"]) == len(losses["residual"]) == 2
        assert all(v > 0 for v in losses["branch"] + losses["residual"])
        assert all(abs(a - b) > 1e-4
                   for a, b in zip(losses["branch"], losses["residual"])), losses

    def test_a_single_site_block_taps_only_that_site_in_either_placement(self):
        x = np.random.default_rng(3).standard_normal((BATCH, SEQ, H)).astype("float32")
        for placement in ("branch", "residual"):
            block = recorded_block(tap_placement=placement, tap_sites=("mlp",))
            spy = Spy(name="spy")
            block(x, taps=(spy,), training=False)
            assert block.attention_tap.seen == []
            assert len(block.mlp_tap.seen) == 1
            np.testing.assert_array_equal(spy.seen[0], block.mlp_tap.seen[0])

    def test_a_block_without_taps_runs_in_both_placements(self):
        x = np.random.default_rng(3).standard_normal((BATCH, SEQ, H)).astype("float32")
        outs = []
        for placement in ("branch", "residual"):
            block = make_block(tap_placement=placement, tap_sites=())
            outs.append(keras.ops.convert_to_numpy(block(x, training=False)))
            assert block.losses == []
        np.testing.assert_allclose(outs[0].shape, (BATCH, SEQ, H))

    def test_the_model_hands_the_placement_to_every_block(self):
        model = make_model(depth=3, tap_placement="residual")
        assert [b.tap_placement for b in model.blocks] == ["residual"] * 3
        assert make_model(depth=3).blocks[2].tap_placement == "branch"

    def test_the_captures_of_a_model_see_the_post_add_tensor(self):
        """Through the model: capture 1 of block 0 is ``embeddings + attention``."""
        model = make_model(depth=1, tap_placement="residual")
        ids = _ids()
        captures = [Spy(name=f"c{i}") for i in range(2)]
        model(ids, taps=tuple(captures), training=False)

        block = model.blocks[0]
        embedded = keras.ops.convert_to_numpy(
            model.word_embeddings(ids)
            + model.positional_embeddings(np.arange(SEQ)))
        from dl_techniques.utils.masking import create_causal_attend_mask
        mask = create_causal_attend_mask(embedded, None)
        attn = keras.ops.convert_to_numpy(block.attention(
            block.attention_norm(embedded, training=False),
            attention_mask=mask, training=False))
        np.testing.assert_allclose(captures[0].seen[0], embedded + attn,
                                   rtol=0, atol=1e-5)


# ---------------------------------------------------------------------
# validation and serialization
# ---------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("bad", ["middle", "Branch", "", None, 1])
    def test_a_bad_tap_placement_is_refused_by_name(self, bad):
        with pytest.raises(ValueError, match=r"tap_placement must be one of "
                                             r"\['branch', 'residual'\], got"):
            make_block(build=False, tap_placement=bad)

    def test_the_model_refuses_it_too(self):
        with pytest.raises(ValueError, match=r"tap_placement must be one of .*'middle'"):
            TopoLM(vocab_size=50, embed_dim=H, depth=1, num_heads=HEADS,
                   max_seq_len=16, radius=1, tap_placement="middle")

    @pytest.mark.parametrize("bad", ["False", 0, 1, None])
    def test_norm_center_must_be_a_bool(self, bad):
        with pytest.raises(ValueError, match=r"norm_center must be a bool, got"):
            make_block(build=False, norm_center=bad)

    def test_the_model_refuses_a_bad_norm_center_too(self):
        with pytest.raises(ValueError, match=r"norm_center must be a bool"):
            TopoLM(vocab_size=50, embed_dim=H, depth=1, num_heads=HEADS,
                   max_seq_len=16, radius=1, norm_center="yes")


ALL_OPTIONS = dict(EVERY_OPTION, **KNOBS_ON)


class TestSerialization:
    def test_the_block_config_round_trips_with_every_option(self):
        block = make_block(build=False, **ALL_OPTIONS)
        config = block.get_config()
        for key, value in ALL_OPTIONS.items():
            assert config[key] == value, key
        assert TopoLMBlock.from_config(config).get_config() == config

    def test_the_model_config_round_trips_with_every_option(self):
        model = make_model(**ALL_OPTIONS)
        config = model.get_config()
        for key, value in ALL_OPTIONS.items():
            assert config[key] == value, key
        assert TopoLM.from_config(config).get_config() == config

    def test_a_keras_archive_restores_every_option_and_every_value(self, tmp_path):
        model = make_model(**dict(KNOBS_ON, normalization_type="layer_norm"))
        before = keras.ops.convert_to_numpy(model(_ids(), training=False)["logits"])
        path = str(tmp_path / "topolm.keras")
        model.save(path)
        restored = keras.models.load_model(path)
        assert restored.get_config() == model.get_config()
        assert [w.path for w in restored.weights] == [w.path for w in model.weights]
        assert restored.blocks[0].tap_placement == "residual"
        after = keras.ops.convert_to_numpy(restored(_ids(), training=False)["logits"])
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)

    def test_from_variant_takes_the_knobs_as_keywords(self):
        model = TopoLM.from_variant(
            "tiny", vocab_size=50, depth=1, max_seq_len=16, tap_placement="residual",
            norm_center=True, use_bias=True)
        config = model.get_config()
        assert (config["tap_placement"], config["norm_center"], config["use_bias"]) == (
            "residual", True, True)

    def test_a_variant_row_may_carry_the_knobs(self, monkeypatch):
        row = {"embed_dim": H, "depth": 1, "num_heads": HEADS,
               "ffn_intermediate_size": I, "max_seq_len": 16, "radius": 1,
               "tap_placement": "residual", "norm_center": True}
        monkeypatch.setitem(TopoLM.MODEL_VARIANTS, "with_knobs", row)
        model = TopoLM.from_variant("with_knobs", vocab_size=50)
        assert model.blocks[0].tap_placement == "residual"
        assert model.final_norm.center is True

    def test_the_defaults_survive_a_config_missing_the_keys(self):
        config = make_model().get_config()
        for key in ("tap_placement", "norm_center", "use_bias"):
            del config[key]
        again = TopoLM.from_config(config).get_config()
        assert (again["tap_placement"], again["norm_center"], again["use_bias"]) == (
            "branch", False, False)


class TestTheControlArmKeepsTheWeightSetUnderTheKnobs:
    @pytest.mark.parametrize("kwargs", [
        KNOBS_ON, {"tap_placement": "residual"}, {"norm_center": True},
        {"use_bias": True},
    ], ids=["all", "residual", "centre", "bias"])
    def test_alpha_zero_has_the_identical_weights(self, kwargs):
        trained = make_model(alpha=2.5, **kwargs)
        control = make_model(alpha=0.0, **kwargs)
        assert [(w.path, tuple(w.shape)) for w in trained.weights] == [
            (w.path, tuple(w.shape)) for w in control.weights]

    def test_the_taps_exist_unconditionally_in_the_residual_placement(self):
        model = make_model(alpha=0.0, tap_placement="residual")
        assert len(model.tap_layers) == 2
        assert all(tap is not None for tap in model.tap_layers)

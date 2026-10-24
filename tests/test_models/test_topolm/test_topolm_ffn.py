"""The TopoLM block's feed-forward is ONE configurable MLP, checked against an oracle.

Until plan-2026-10-23T190557-972a9ac5 step 10 (D-030) the block chained two
``create_ffn_layer("mlp")`` calls: four ``Dense`` layers and two activations, 3.12x
the parameters of a GPT-2 MLP, while the tests compared every structure with itself.
Nothing in this file reads a structure back from the code under test:

* the parameter counts come from the closed forms below (``2 * H * I + H + I`` for
  one MLP; the per-type forms from the factory docs for ``swiglu`` / ``geglu``),
* the forward values come from numpy applied to the block's OWN weights, taken from
  the branch just before the residual add through the ``taps=`` capture hook,
* the guards are proven able to fail: ``_ChainedMLPs`` rebuilds the old four-layer
  structure and both oracles must reject it.

Run on CPU (``CUDA_VISIBLE_DEVICES=``): the tolerances are derived for float32 without
TF32.
"""

import math
from typing import Any, Dict, List

import keras
import numpy as np
import pytest
import scipy.special

from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.ffn.mlp import MLPBlock
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.models.language.topolm import (
    MODEL_VARIANTS,
    TopoLM,
    TopoLMBlock,
)
from tests.numerics import reassociation_atol

#: The five shipped shapes: (hidden, intermediate, heads, radius, grid_dim).
VARIANT_NAMES = ("tiny", "small", "paper", "tiny3d", "paper3d")

H, I, HEADS = 64, 128, 4
BATCH, SEQ = 2, 8


# ---------------------------------------------------------------------
# independent closed forms
# ---------------------------------------------------------------------


def mlp_params(hidden: int, inner: int, bias: bool = True) -> int:
    """One GPT-2 MLP: ``Linear(H->I)``, activation, ``Linear(I->H)``."""
    return 2 * hidden * inner + ((hidden + inner) if bias else 0)


def swiglu_width(hidden: int, factor: int = 4, multiple: int = 256) -> int:
    """The 2/3 rule of the swiglu docs: ``int(H * factor * 2/3)`` up to a multiple."""
    raw = int(hidden * factor * 2 / 3)
    return multiple * math.ceil(raw / multiple)


def swiglu_params(hidden: int, **rule: int) -> int:
    """gate, up and down projections, no biases (the factory default)."""
    return 3 * hidden * swiglu_width(hidden, **rule)


def geglu_params(hidden: int, inner: int) -> int:
    """``input_proj`` H -> 2I (gate and value), ``output_proj`` I -> H, both biased."""
    return hidden * 2 * inner + 2 * inner + inner * hidden + hidden


def attention_params(hidden: int) -> int:
    """Fused qkv (H x 3H) and output projection (H x H); the block builds no biases."""
    return 3 * hidden * hidden + hidden * hidden


def norm_params(hidden: int, kind: str = "layer_norm") -> int:
    """gamma and beta for layer norm, one scale for RMS norm."""
    return 2 * hidden if kind == "layer_norm" else hidden


def trainable_count(layer: keras.layers.Layer) -> int:
    return sum(int(np.prod(w.shape)) for w in layer.trainable_weights)


def ffn_count(layer: keras.layers.Layer) -> int:
    return sum(int(np.prod(w.shape)) for w in layer.ffn.weights)


# ---------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------


def make_block(hidden: int = H, inner: int = I, heads: int = HEADS,
               build: bool = True, **kwargs: Any) -> TopoLMBlock:
    keras.utils.set_random_seed(0)
    config: Dict[str, Any] = dict(
        hidden_size=hidden, num_heads=heads, intermediate_size=inner,
        radius=1, name="block",
    )
    config.update(kwargs)
    block = TopoLMBlock(**config)
    if build:
        block.build((None, SEQ, hidden))
    return block


def make_variant_block(name: str) -> TopoLMBlock:
    row = MODEL_VARIANTS[name]
    block = TopoLMBlock(
        hidden_size=row["embed_dim"], num_heads=row["num_heads"],
        intermediate_size=row["ffn_intermediate_size"],
        radius=row.get("radius", 5), grid_dim=row.get("grid_dim", 2),
        name="block",
    )
    block.build((None, 4, row["embed_dim"]))
    return block


def make_model(**kwargs: Any) -> TopoLM:
    keras.utils.set_random_seed(0)
    config: Dict[str, Any] = dict(
        vocab_size=50, embed_dim=H, depth=1, num_heads=HEADS,
        ffn_intermediate_size=I, max_seq_len=16, radius=1, name="topolm",
    )
    config.update(kwargs)
    model = TopoLM(**config)
    model(_ids(), training=False)
    return model


def _ids() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 50, size=(BATCH, SEQ)).astype("int32")


def randomize(block: TopoLMBlock, seed: int = 1) -> None:
    """Replace every trainable weight by a random one (norm scales near 1).

    Initial gammas are 1 and betas 0: a block left like that cannot tell a bias, a
    scale or a wrong activation order from the right ones.
    """
    rng = np.random.default_rng(seed)
    for w in block.trainable_weights:
        shape = tuple(w.shape)
        if "norm" in w.path:
            w.assign(1.0 + 0.3 * rng.standard_normal(shape))
        else:
            w.assign(0.5 * rng.standard_normal(shape) / math.sqrt(shape[0]))


class Spy(keras.layers.Layer):
    """Identity layer that records what it is handed."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seen: List[np.ndarray] = []

    def call(self, x, training=None):
        self.seen.append(keras.ops.convert_to_numpy(x))
        return x


def run_with_spies(block: TopoLMBlock, x: np.ndarray, training: bool = False):
    """``(output, attention branch output, feed-forward branch output)``."""
    spies = (Spy(name="spy_attn"), Spy(name="spy_mlp"))
    out = block(x, taps=spies, training=training)
    return (keras.ops.convert_to_numpy(out), spies[0].seen[0], spies[1].seen[0])


# ---------------------------------------------------------------------
# numpy reference
# ---------------------------------------------------------------------

ACTIVATIONS = {
    # The exact (erf) GELU: what the string 'gelu' resolves to in this repo.
    "gelu": lambda v: 0.5 * v * (1.0 + scipy.special.erf(v / math.sqrt(2.0))),
    # GPT-2's tanh approximation: what ``activation='gelu_tanh'`` computes.
    "gelu_tanh": lambda v: 0.5 * v * (1.0 + np.tanh(
        math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3))),
    "relu": lambda v: np.maximum(v, 0.0),
}


def np_layer_norm(v: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                  eps: float) -> np.ndarray:
    mu = v.mean(-1, keepdims=True)
    var = v.var(-1, keepdims=True)
    return (v - mu) / np.sqrt(var + eps) * gamma + beta


def weights_by_suffix(layer: keras.layers.Layer) -> Dict[str, np.ndarray]:
    """``{'fc1/kernel': array, ...}``; two weights under one suffix is a failure."""
    found: Dict[str, np.ndarray] = {}
    for w in layer.weights:
        key = "/".join(w.path.split("/")[-2:])
        assert key not in found, f"two weights end in {key!r}: {[v.path for v in layer.weights]}"
        found[key] = keras.ops.convert_to_numpy(w)
    return found


def assert_ffn_param_count(block: TopoLMBlock, hidden: int, inner: int,
                           bias: bool = True) -> None:
    """The oracle behind the count: one MLP is ``2HI + H + I`` parameters."""
    expected = mlp_params(hidden, inner, bias)
    got = ffn_count(block)
    assert got == expected, (
        f"feed-forward holds {got} parameters, one MLP at H={hidden}, I={inner} "
        f"holds {expected}: {[(w.path, tuple(w.shape)) for w in block.ffn.weights]}"
    )
    assert len(block.ffn.weights) == (4 if bias else 2)


def assert_ffn_forward(block: TopoLMBlock, activation: str = "gelu",
                       eps: float = 1e-5) -> None:
    """Dense -> activation -> Dense of the block's own weights equals the branch."""
    randomize(block)
    x = np.random.default_rng(2).standard_normal((BATCH, SEQ, block.hidden_size))
    x = x.astype("float32")
    out, attn_out, mlp_out = run_with_spies(block, x)

    w = weights_by_suffix(block.ffn)
    norm = {k: keras.ops.convert_to_numpy(v)
            for k, v in zip(("gamma", "beta"), block.ffn_norm.weights)}
    hidden = x.astype("float64") + attn_out.astype("float64")
    normed = np_layer_norm(hidden, norm["gamma"], norm["beta"], eps)
    inner = ACTIVATIONS[activation](normed @ w["fc1/kernel"] + w["fc1/bias"])
    expected = inner @ w["fc2/kernel"] + w["fc2/bias"]

    atol = reassociation_atol(
        [block.hidden_size, block.hidden_size, block.intermediate_size], 1,
        float(np.abs(expected).max()))
    np.testing.assert_allclose(mlp_out, expected, rtol=0, atol=atol)
    # the residual add comes AFTER the branch, on the stream that entered the FFN
    np.testing.assert_allclose(out, mlp_out + hidden, rtol=0, atol=atol)


class _ChainedMLPs(keras.layers.Layer):
    """The structure this plan removed: two MLPs chained (four Dense, two GELU)."""

    def __init__(self, hidden: int, inner: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden, self.inner = hidden, inner
        self.first = create_ffn_layer("mlp", name="first", hidden_dim=hidden,
                                      output_dim=inner)
        self.second = create_ffn_layer("mlp", name="second", hidden_dim=inner,
                                       output_dim=hidden)

    def build(self, input_shape):
        self.first.build(input_shape)
        self.second.build(tuple(input_shape[:-1]) + (self.inner,))
        super().build(input_shape)

    def call(self, x, training=None):
        return self.second(self.first(x, training=training), training=training)


def make_chained_block() -> TopoLMBlock:
    block = make_block(build=False)
    block.ffn = _ChainedMLPs(H, I, name="ffn")
    block.build((None, SEQ, H))
    return block


# ---------------------------------------------------------------------
# (a) the structure: one MLP, counted
# ---------------------------------------------------------------------


class TestTheFeedForwardIsOneMLP:
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_the_feed_forward_holds_the_closed_form_count(self, variant):
        row = MODEL_VARIANTS[variant]
        block = make_variant_block(variant)
        assert_ffn_param_count(block, row["embed_dim"], row["ffn_intermediate_size"])

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_everything_but_the_feed_forward_is_unchanged(self, variant):
        h = MODEL_VARIANTS[variant]["embed_dim"]
        block = make_variant_block(variant)
        rest = trainable_count(block) - ffn_count(block)
        assert rest == attention_params(h) + 2 * norm_params(h)

    def test_the_tiny_block_is_the_gpt2_block(self):
        """525,568 parameters at 256 / 1024; the four-Dense structure held 1,640,960."""
        block = make_variant_block("tiny")
        assert ffn_count(block) == 525_568 == 2 * 256 * 1024 + 256 + 1024

    def test_the_feed_forward_is_one_layer_with_two_projections(self):
        block = make_block()
        assert isinstance(block.ffn, MLPBlock)
        assert sorted(weights_by_suffix(block.ffn)) == [
            "fc1/bias", "fc1/kernel", "fc2/bias", "fc2/kernel"]
        assert not hasattr(block, "ffn_dense_1")
        assert not hasattr(block, "ffn_dense_2")
        assert tuple(block.ffn.fc1.kernel.shape) == (H, I)
        assert tuple(block.ffn.fc2.kernel.shape) == (I, H)

    def test_the_weight_paths_are_those_of_the_shared_block(self):
        block = make_block()
        assert [w.path for w in block.ffn.weights] == [
            "block/ffn/fc1/kernel", "block/ffn/fc1/bias",
            "block/ffn/fc2/kernel", "block/ffn/fc2/bias"]


# ---------------------------------------------------------------------
# (b) the forward, against numpy
# ---------------------------------------------------------------------


class TestTheForwardIsDenseActivationDense:
    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    def test_the_branch_equals_the_numpy_oracle(self, activation):
        assert_ffn_forward(make_block(activation=activation), activation)

    def test_the_default_model_activation_is_the_exact_gelu(self):
        # step 11 (D-031): the model default was the tanh form while the block's
        # was the exact one; both are the exact erf GELU now.
        model = make_model()
        block = model.blocks[0]
        assert_ffn_forward(block, "gelu", eps=model.layer_norm_eps)

    def test_the_post_norm_branch_runs_the_same_two_projections(self):
        block = make_block(normalization_position="post")
        randomize(block)
        x = np.random.default_rng(3).standard_normal((BATCH, SEQ, H)).astype("float32")
        out, attn_out, mlp_out = run_with_spies(block, x)
        gamma = {k: keras.ops.convert_to_numpy(v) for k, v in zip(
            ("g", "b"), block.attention_norm.weights)}
        ffn_gamma = {k: keras.ops.convert_to_numpy(v) for k, v in zip(
            ("g", "b"), block.ffn_norm.weights)}
        hidden = np_layer_norm(x.astype("float64") + attn_out, gamma["g"], gamma["b"], 1e-5)
        w = weights_by_suffix(block.ffn)
        expected = ACTIVATIONS["gelu"](hidden @ w["fc1/kernel"] + w["fc1/bias"]) \
            @ w["fc2/kernel"] + w["fc2/bias"]
        atol = reassociation_atol([H, H, I], 1, float(np.abs(expected).max()))
        np.testing.assert_allclose(mlp_out, expected, rtol=0, atol=atol)
        final = np_layer_norm(mlp_out + hidden, ffn_gamma["g"], ffn_gamma["b"], 1e-5)
        np.testing.assert_allclose(out, final, rtol=0, atol=10 * atol)


# ---------------------------------------------------------------------
# (c) the oracles can fail
# ---------------------------------------------------------------------


class TestTheOraclesRejectTheOldStructure:
    def test_the_chained_block_fails_the_count_oracle(self):
        block = make_chained_block()
        assert ffn_count(block) != mlp_params(H, I)
        with pytest.raises(AssertionError):
            assert_ffn_param_count(block, H, I)

    def test_the_chained_block_fails_the_forward_oracle(self):
        with pytest.raises(AssertionError):
            assert_ffn_forward(make_chained_block())

    def test_the_real_block_passes_the_same_two_oracles(self):
        block = make_block()
        assert_ffn_param_count(block, H, I)
        assert_ffn_forward(block)


# ---------------------------------------------------------------------
# (d) the options
# ---------------------------------------------------------------------


class TestEachOptionChangesTheBlockAsDocumented:
    def test_swiglu_builds_runs_and_counts_by_its_own_rule(self):
        block = make_block(ffn_type="swiglu")
        assert ffn_count(block) == swiglu_params(H) == 3 * H * 256
        assert sorted(weights_by_suffix(block.ffn)) == [
            "down_proj/kernel", "gate_proj/kernel", "up_proj/kernel"]
        out = block(np.zeros((BATCH, SEQ, H), "float32"), training=False)
        assert tuple(out.shape) == (BATCH, SEQ, H)

    def test_swiglu_sizes_itself_and_ffn_args_resize_it(self):
        block = make_block(ffn_type="swiglu", ffn_args={"ffn_multiple_of": 32})
        assert swiglu_width(H, multiple=32) == 192
        assert ffn_count(block) == swiglu_params(H, multiple=32) == 3 * H * 192

    def test_geglu_builds_runs_and_counts_by_its_own_rule(self):
        block = make_block(ffn_type="geglu")
        assert ffn_count(block) == geglu_params(H, I)
        out = block(np.zeros((BATCH, SEQ, H), "float32"), training=False)
        assert tuple(out.shape) == (BATCH, SEQ, H)

    def test_the_type_changes_the_weight_set(self):
        sets = {
            t: sorted(w.path for w in make_block(ffn_type=t).ffn.weights)
            for t in ("mlp", "swiglu", "geglu")
        }
        assert len({tuple(v) for v in sets.values()}) == 3

    def test_the_activation_changes_the_output_on_a_fixed_input(self):
        x = np.random.default_rng(4).standard_normal((BATCH, SEQ, H)).astype("float32")
        outs = {}
        for name in ("gelu", "relu"):
            block = make_block(activation=name)
            randomize(block, seed=5)
            outs[name] = keras.ops.convert_to_numpy(block(x, training=False))
        assert np.abs(outs["gelu"] - outs["relu"]).max() > 1e-3

    def test_use_bias_false_removes_exactly_the_feed_forward_biases(self):
        biased, bare = make_block(), make_block(use_bias=False)
        assert_ffn_param_count(bare, H, I, bias=False)
        assert ffn_count(biased) - ffn_count(bare) == H + I
        assert trainable_count(biased) - trainable_count(bare) == H + I
        assert sorted(weights_by_suffix(bare.ffn)) == ["fc1/kernel", "fc2/kernel"]

    def test_rms_norm_has_no_beta_and_both_norms_follow(self):
        layer_norm, rms = make_block(), make_block(normalization_type="rms_norm")
        assert isinstance(rms.attention_norm, RMSNorm)
        assert isinstance(rms.ffn_norm, RMSNorm)
        assert trainable_count(layer_norm) - trainable_count(rms) == 2 * H
        rest = trainable_count(rms) - ffn_count(rms)
        assert rest == attention_params(H) + 2 * norm_params(H, "rms_norm")
        assert not [w.path for w in rms.weights if w.path.endswith("/beta")]

    def test_the_models_final_norm_follows_the_option(self):
        plain = make_model()
        rms = make_model(normalization_type="rms_norm")
        assert isinstance(rms.final_norm, RMSNorm)
        assert not isinstance(plain.final_norm, RMSNorm)
        assert [w.path for w in rms.final_norm.weights] == ["topolm/final_norm/scale"]
        assert [w.path for w in plain.final_norm.weights] == [
            "topolm/final_norm/gamma", "topolm/final_norm/beta"]

    def test_norm_args_reach_their_own_norm_only(self):
        block = make_block(attention_norm_args={"center": False},
                           ffn_norm_args={"epsilon": 1e-3})
        assert [w.path.split("/")[-1] for w in block.attention_norm.weights] == ["gamma"]
        assert [w.path.split("/")[-1] for w in block.ffn_norm.weights] == ["gamma", "beta"]
        assert block.ffn_norm.epsilon == 1e-3
        assert block.attention_norm.epsilon == block.layer_norm_eps == 1e-5

    def test_the_output_projection_is_scaled_for_the_residual_path(self):
        """GPT-2 divides the residual projections by sqrt(2 * n_layer).

        A block used alone is a one-layer stack (``num_layers=1``), so it scales
        them by sqrt(2) (each block adds two branches); fc1 keeps
        ``initializer_range``. The depth factor of a model is pinned in
        ``test_topolm_fidelity.py``. A ``TruncatedNormal`` sample's std is a fixed
        multiple of its ``stddev``, so the ratio of the two kernels' sample stds
        is 1/sqrt(2) whatever that multiple is (8192 draws each: about 1.5% noise).
        """
        block = make_block(hidden=64, inner=128)
        fc1 = keras.ops.convert_to_numpy(block.ffn.fc1.kernel)
        fc2 = keras.ops.convert_to_numpy(block.ffn.fc2.kernel)
        ratio = float(fc2.std() / fc1.std())
        assert 0.66 < ratio < 0.75, ratio
        assert not np.array_equal(fc1.ravel()[:100], fc2.ravel()[:100])
        bias = [keras.ops.convert_to_numpy(w) for w in block.ffn.weights
                if w.path.endswith("/bias")]
        assert len(bias) == 2 and all(not b.any() for b in bias)

    def test_the_branch_dropout_is_applied_exactly_once(self):
        """GPT-2's ``resid_pdrop`` after the output projection and nowhere else."""
        for ffn_type in ("mlp", "swiglu", "geglu"):
            block = make_block(ffn_type=ffn_type, dropout_rate=0.5)
            inside = [l.rate for l in block.ffn._flatten_layers(include_self=False)
                      if isinstance(l, keras.layers.Dropout)]
            assert inside in ([], [0.0]), (ffn_type, inside)
            assert block.ffn_dropout.rate == 0.5
        block = make_block(dropout_rate=0.5, build=False)
        block.build((None, 32, H))
        x = np.random.default_rng(6).standard_normal((BATCH, 32, H)).astype("float32")
        _, _, mlp_out = run_with_spies(block, x, training=True)
        zero_fraction = float((mlp_out == 0.0).mean())
        assert abs(zero_fraction - 0.5) < 0.06, zero_fraction

    @pytest.mark.parametrize("kwargs, needle", [
        ({"ffn_type": "not_an_ffn"}, r"ffn_type must be one of .*'not_an_ffn'"),
        ({"ffn_type": "kan"}, r"ffn_type must be one of .*'kan'"),
        ({"normalization_type": "not_a_norm"},
         r"normalization_type must be one of .*'not_a_norm'"),
        ({"ffn_args": {"dropout_rate": 0.1}}, r"ffn_args must not set dropout_rate"),
        ({"ffn_args": [("a", 1)]}, r"ffn_args must be a dict"),
        ({"attention_norm_args": "x"}, r"attention_norm_args must be a dict"),
        ({"ffn_norm_args": 3}, r"ffn_norm_args must be a dict"),
        ({"ffn_args": {"not_a_param": 1}}, r"Failed to create .*not_a_param"),
        ({"normalization_position": "middle"},
         r"normalization_position must be 'pre' or 'post', got 'middle'"),
    ])
    def test_a_bad_value_is_refused_by_name(self, kwargs, needle):
        with pytest.raises(ValueError, match=needle):
            make_block(build=False, **kwargs)

    def test_the_ffn_type_error_lists_the_accepted_types(self):
        with pytest.raises(ValueError) as excinfo:
            make_block(build=False, ffn_type="not_an_ffn")
        for accepted in ("mlp", "swiglu", "geglu"):
            assert accepted in str(excinfo.value)

    def test_the_norm_type_error_lists_the_accepted_types(self):
        with pytest.raises(ValueError) as excinfo:
            make_block(build=False, normalization_type="not_a_norm")
        for accepted in ("layer_norm", "rms_norm"):
            assert accepted in str(excinfo.value)

    @pytest.mark.parametrize("kwargs, needle", [
        ({"ffn_type": "not_an_ffn"}, r"ffn_type must be one of .*'not_an_ffn'"),
        ({"normalization_type": "not_a_norm"},
         r"normalization_type must be one of .*'not_a_norm'"),
    ])
    def test_the_model_refuses_them_too(self, kwargs, needle):
        with pytest.raises(ValueError, match=needle):
            TopoLM(vocab_size=50, embed_dim=H, depth=1, num_heads=HEADS,
                   max_seq_len=16, radius=1, **kwargs)

    def test_the_model_counts_match_a_closed_form_under_every_option(self):
        cases = [
            ({}, mlp_params(H, I), "layer_norm"),
            ({"ffn_type": "swiglu"}, swiglu_params(H), "layer_norm"),
            ({"ffn_type": "geglu"}, geglu_params(H, I), "layer_norm"),
            ({"use_bias": False}, mlp_params(H, I, False), "layer_norm"),
            ({"normalization_type": "rms_norm"}, mlp_params(H, I), "rms_norm"),
        ]
        for kwargs, ffn, norm in cases:
            model = make_model(**kwargs)
            expected = (
                50 * H + 16 * H
                + 1 * (attention_params(H) + 2 * norm_params(H, norm) + ffn)
                + norm_params(H, norm)
            )
            assert trainable_count(model) == expected, kwargs


# ---------------------------------------------------------------------
# (e) serialization with every option set
# ---------------------------------------------------------------------

EVERY_OPTION: Dict[str, Any] = dict(
    ffn_type="swiglu", ffn_args={"ffn_multiple_of": 32}, activation="relu",
    use_bias=False, normalization_type="rms_norm",
    attention_norm_args={"epsilon": 1e-4}, ffn_norm_args={"epsilon": 1e-3},
)


class TestSerialization:
    def test_the_block_config_round_trips_with_every_option(self):
        block = make_block(build=False, **EVERY_OPTION)
        config = block.get_config()
        for key, value in EVERY_OPTION.items():
            assert config[key] == value, key
        clone = TopoLMBlock.from_config(config)
        assert clone.get_config() == config

    def test_the_defaults_are_the_documented_ones(self):
        config = make_block(build=False).get_config()
        assert (config["ffn_type"], config["ffn_args"], config["use_bias"]) == (
            "mlp", {}, True)
        assert (config["normalization_type"], config["attention_norm_args"],
                config["ffn_norm_args"]) == ("layer_norm", {}, {})

    def test_the_model_config_round_trips_with_every_option(self):
        model = make_model(**EVERY_OPTION)
        config = model.get_config()
        for key, value in EVERY_OPTION.items():
            assert config[key] == value, key
        assert TopoLM.from_config(config).get_config() == config

    def test_a_keras_archive_restores_every_option_and_every_value(self, tmp_path):
        model = make_model(**EVERY_OPTION)
        before = keras.ops.convert_to_numpy(model(_ids(), training=False)["logits"])
        path = str(tmp_path / "topolm.keras")
        model.save(path)
        restored = keras.models.load_model(path)
        assert restored.get_config() == model.get_config()
        assert [w.path for w in restored.weights] == [w.path for w in model.weights]
        after = keras.ops.convert_to_numpy(restored(_ids(), training=False)["logits"])
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)

    def test_a_variant_row_may_carry_the_options(self, monkeypatch):
        row = {"embed_dim": H, "depth": 1, "num_heads": HEADS,
               "ffn_intermediate_size": I, "max_seq_len": 16, "radius": 1,
               "ffn_type": "geglu", "normalization_type": "rms_norm"}
        monkeypatch.setitem(TopoLM.MODEL_VARIANTS, "with_options", row)
        model = TopoLM.from_variant("with_options", vocab_size=50)
        assert model.blocks[0].ffn_type == "geglu"
        assert model.get_config()["normalization_type"] == "rms_norm"

    def test_a_from_variant_keyword_sets_an_option(self):
        model = TopoLM.from_variant(
            "tiny", vocab_size=50, depth=1, max_seq_len=16, ffn_type="swiglu",
            use_bias=False)
        assert model.blocks[0].ffn_type == "swiglu"
        assert model.get_config()["use_bias"] is False


# ---------------------------------------------------------------------
# (f) the taps sit on the branch outputs
# ---------------------------------------------------------------------


class TestTheTapsSitOnTheBranchOutputs:
    @pytest.mark.parametrize("kwargs", [
        {}, {"ffn_type": "swiglu"}, {"normalization_type": "rms_norm"},
    ], ids=["mlp", "swiglu", "rms_norm"])
    def test_the_ffn_tap_is_handed_the_feed_forward_output_and_the_add_follows(self, kwargs):
        block = make_block(**kwargs)
        randomize(block)
        x = np.random.default_rng(7).standard_normal((BATCH, SEQ, H)).astype("float32")
        out, attn_out, mlp_out = run_with_spies(block, x)
        hidden = x + attn_out
        branch = keras.ops.convert_to_numpy(block.ffn(block.ffn_norm(hidden), training=False))
        np.testing.assert_allclose(mlp_out, branch, rtol=0, atol=1e-6)
        np.testing.assert_allclose(out, mlp_out + hidden, rtol=0, atol=1e-6)

    def test_the_attention_tap_is_handed_the_attention_output_before_its_add(self):
        block = make_block()
        randomize(block)
        x = np.random.default_rng(8).standard_normal((BATCH, SEQ, H)).astype("float32")
        _, attn_out, _ = run_with_spies(block, x)
        branch = keras.ops.convert_to_numpy(
            block.attention(block.attention_norm(x), training=False))
        np.testing.assert_allclose(attn_out, branch, rtol=0, atol=1e-6)

    def test_the_real_taps_see_the_feed_forward_branch_during_training(self):
        block = make_block(alpha=2.5)
        randomize(block)
        x = np.random.default_rng(9).standard_normal((BATCH, SEQ, H)).astype("float32")
        block(x, training=True)
        assert len(block.losses) == 2
        assert all(float(keras.ops.convert_to_numpy(l)) > 0 for l in block.losses)


# ---------------------------------------------------------------------
# (g) the alpha = 0 control keeps the weight set
# ---------------------------------------------------------------------


class TestTheControlArmKeepsTheWeightSet:
    @pytest.mark.parametrize("kwargs", [
        {}, EVERY_OPTION, {"ffn_type": "geglu", "normalization_type": "rms_norm"},
    ], ids=["default", "every_option", "geglu_rms"])
    def test_alpha_zero_has_the_identical_weights(self, kwargs):
        trained = make_block(alpha=2.5, **kwargs)
        control = make_block(alpha=0.0, **kwargs)
        def signature(b):
            return [(w.path, tuple(w.shape)) for w in b.weights]

        assert signature(trained) == signature(control)

    def test_the_model_control_has_the_identical_weights_under_options(self):
        trained = make_model(alpha=2.5, **EVERY_OPTION)
        control = make_model(alpha=0.0, **EVERY_OPTION)
        assert [(w.path, tuple(w.shape)) for w in trained.weights] == [
            (w.path, tuple(w.shape)) for w in control.weights]

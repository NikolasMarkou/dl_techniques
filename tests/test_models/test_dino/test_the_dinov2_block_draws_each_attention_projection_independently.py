"""A DINOv2 block's attention projections each start from their own random draw.

``MultiHeadAttention`` hands ONE initializer instance to its ``qkv`` and ``proj`` Dense layers, and
``DINOv2Block`` passed it nothing else: the smaller draw (``proj``) was a rescaled prefix of the larger
(``qkv``), one replayed kernel pair per block (12 of 3,403 pairs in the real Depth Anything encoder, 12
of 1,326 in a bare ViT-S). The block now builds its attention with the keywords that separate the
projections (:data:`SUPPORTED_ATTENTION_TYPES`), and only for the attention types whose layer declares
them. ``anchor``, ``perceiver``, ``performer``, ``rpc`` and ``wave_field`` replayed too and had no keyword;
their layers now clone an initializer per projection themselves (D-056), so they joined the set.
``mean_shift`` clones one per projection from its first version and joined with them.
Every other type is refused at construction because the block cannot build it from its arguments or run
it on its tokens, and a fixed layer does not change that.

This file pins what the runtime contract's replay rows cannot see by themselves:

* the supported set is exactly this one, named by the refusal and by the docstring;
* each separating initializer is the attention layer's own default for ``qkv``, in class;
* the seeded draws: ``qkv`` takes the first default seed of the block and ``proj`` the second, so the
  block's first kernel is unmoved and every initializer created after ``proj`` sits one seed further
  down the stream (carve-out C-7, decision D-054);
* no weight name changed, and weights keyed by path still transplant;
* every refused type is a ``pytest.raises`` row carrying the exception class and the message it fails with
  when the table is patched to let it through, so the reason is pinned, not described: a block that learns
  to pass the missing argument turns its row red and the type can be measured and added.
"""

import re
from typing import Dict, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.factory import ATTENTION_REGISTRY
from dl_techniques.models.vision.dino import dino_v2
from dl_techniques.models.vision.dino.dino_v2 import (
    DINOv2,
    DINOv2Block,
    DINOv2VisionTransformer,
)
from tests.test_models.initializer_replay_oracle import replayed_pairs

SUPPORTED = ["multi_head", "multi_head_cross", "gated", "linear", "ring",
             "anchor", "perceiver", "performer", "rpc", "wave_field", "mean_shift"]
#: The types that draw independently because their LAYER clones an initializer per projection.
LAYER_CLONES = ["gated", "linear", "ring", "anchor", "perceiver", "performer", "rpc", "wave_field",
                "mean_shift"]
DIM, HEADS, TOKENS = 64, 4, 17


def _tokens() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(2, TOKENS, DIM)).astype("float32")


def _block(attention_type: str = "multi_head", **changes) -> DINOv2Block:
    arguments = dict(dim=DIM, num_heads=HEADS, attention_type=attention_type, init_values=1e-5,
                     name="blk")
    arguments.update(changes)
    block = DINOv2Block(**arguments)
    block.build((None, TOKENS, DIM))
    return block


def _backbone(attention_type: str = "multi_head", **changes) -> DINOv2VisionTransformer:
    arguments = dict(image_size=28, patch_size=14, embed_dim=DIM, depth=3, num_heads=HEADS,
                     init_values=1e-5, attention_type=attention_type)
    arguments.update(changes)
    return DINOv2VisionTransformer(**arguments)


def _forward_backbone(model: DINOv2VisionTransformer) -> Dict[str, np.ndarray]:
    images = np.random.default_rng(0).uniform(size=(2, 28, 28, 3)).astype("float32")
    masks = np.zeros((2, model.num_patches), dtype="bool")
    return model([images, masks], training=False)


def _engine(block: DINOv2Block):
    """The layer that owns the Dense projections: `MultiHeadAttention` wraps it, the cross layer is it."""
    return getattr(block.attention, "cross_attention", block.attention)


class TestEachProjectionDrawsItself:

    def test_a_default_backbone_replays_no_draw(self) -> None:
        keras.utils.set_random_seed(1234)
        model = _backbone()
        _forward_backbone(model)
        pairs, compared = replayed_pairs(model)
        assert compared > 0, "the backbone exposes fewer than two kernels: nothing was compared"
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"

    def test_the_wrapper_with_a_classifier_replays_no_draw(self) -> None:
        keras.utils.set_random_seed(1234)
        model = DINOv2(image_size=28, patch_size=14, num_classes=5, embed_dim=DIM, depth=2,
                       num_heads=HEADS, init_values=1e-5)
        pairs, compared = replayed_pairs(model)
        assert compared > 0
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"

    @pytest.mark.parametrize("attention_type", SUPPORTED)
    def test_every_supported_attention_type_runs_and_replays_no_draw(
            self, attention_type: str) -> None:
        keras.utils.set_random_seed(1234)
        model = _backbone(attention_type)
        out = _forward_backbone(model)
        assert np.isfinite(np.asarray(out["x_norm_clstoken"])).all()
        pairs, compared = replayed_pairs(model)
        assert compared > 0, "the backbone exposes fewer than two kernels: nothing was compared"
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"

    def test_two_blocks_do_not_repeat_each_others_projection_either(self) -> None:
        keras.utils.set_random_seed(1234)
        model = _backbone(depth=4)
        kernels = [np.asarray(_engine(block).proj_dense.kernel).ravel()
                   for block in model.transformer_blocks]
        for i, a in enumerate(kernels):
            for j in range(i + 1, len(kernels)):
                r = float(np.corrcoef(a[:256], kernels[j][:256])[0, 1])
                assert abs(r) < 0.5, f"blocks {i} and {j} start `proj` from one draw (r={r:.4f})"


class TestTheSeparatingInitializers:

    @pytest.mark.parametrize("attention_type", SUPPORTED)
    def test_each_one_is_the_layers_own_kernel_default(self, attention_type: str) -> None:
        declared = ATTENTION_REGISTRY[attention_type]["optional_params"]
        own = declared["kernel_initializer"]
        extra = dino_v2.SUPPORTED_ATTENTION_TYPES[attention_type] or {}
        for keyword, value in extra.items():
            assert keyword in declared, f"{attention_type} does not declare {keyword}"
            assert value == own, (
                f"{attention_type}: {keyword}={value!r} is not the layer's kernel default {own!r}")

    def test_only_the_types_that_declare_the_keywords_are_given_them(self) -> None:
        for attention_type, extra in dino_v2.SUPPORTED_ATTENTION_TYPES.items():
            declared = ATTENTION_REGISTRY[attention_type]["optional_params"]
            has_output = "output_kernel_initializer" in declared
            assert bool(extra) == has_output, (
                f"{attention_type}: declares output_kernel_initializer={has_output}, "
                f"gets {extra!r}")

    def test_proj_is_a_new_instance_of_the_class_qkv_has(self) -> None:
        engine = _engine(_block())
        qkv, proj = engine.qkv_dense.kernel_initializer, engine.proj_dense.kernel_initializer
        assert proj is not qkv
        assert type(proj) is type(qkv) is keras.initializers.HeNormal

    def test_the_cross_attention_projections_are_three_instances_of_one_class(self) -> None:
        engine = _engine(_block("multi_head_cross"))
        inits = [engine.q_dense.kernel_initializer, engine.kv_dense.kernel_initializer,
                 engine.proj_dense.kernel_initializer]
        assert len({id(i) for i in inits}) == 3
        assert {type(i) for i in inits} == {keras.initializers.GlorotUniform}

    @pytest.mark.parametrize("attention_type", LAYER_CLONES)
    def test_the_types_that_separate_their_projections_themselves_get_no_extra_keyword(
            self, attention_type: str) -> None:
        assert dino_v2.SUPPORTED_ATTENTION_TYPES[attention_type] is None


#: Why each type outside the supported set is outside it: the exception class and a fragment of the message
#: it fails with when ``SUPPORTED_ATTENTION_TYPES`` is patched to let it through, on a block of ``dim=64``,
#: ``num_heads=4`` over 17 tokens (``step19_3/refused_reasons_dino.txt``). ``build`` rows fail in the
#: block's constructor, ``call`` rows on the first forward pass.
_NO_DIM_DROPOUT_HEADS = "unsupported parameter(s) ['dim', 'dropout_rate', 'num_heads']"
_REFUSED: Dict[str, Tuple[type, str, str]] = {
    "area": (ValueError, "call", "Kernel shape must have the same length as input"),
    "beit": (ValueError, "build", "unsupported parameter(s) ['dropout_rate']"),
    "capsule_routing": (ValueError, "build", "unsupported parameter(s) ['dim']"),
    "cbam": (ValueError, "build", _NO_DIM_DROPOUT_HEADS),
    "channel": (ValueError, "build", _NO_DIM_DROPOUT_HEADS),
    "differential": (ValueError, "build", "Required parameters for 'differential' are missing: ['head_dim']"),
    "energy": (ValueError, "build", "unsupported parameter(s) ['dropout_rate']"),
    "fnet": (ValueError, "build", _NO_DIM_DROPOUT_HEADS),
    "group_query": (ValueError, "build", "Required parameters for 'group_query' are missing: ['num_kv_heads']"),
    "hopfield": (ValueError, "build", "unsupported parameter(s) ['dim']"),
    "lighthouse": (ValueError, "call", "must be divisible by pooling_factor ** (num_levels - 1) = 16"),
    "mobile_mqa": (ValueError, "build", "unsupported parameter(s) ['dropout_rate']"),
    "multi_dconv_head_transposed": (ValueError, "build", "unsupported parameter(s) ['dropout_rate']"),
    "multi_head_latent": (
        ValueError, "build", "Required parameters for 'multi_head_latent' are missing: ['kv_latent_dim']"),
    "non_local": (ValueError, "build", "unsupported parameter(s) ['dim', 'num_heads']"),
    "shared_weights_cross": (TypeError, "call", "missing a required argument: 'split_sizes'"),
    "single_window": (
        ValueError, "build", "Required parameters for 'single_window' are missing: ['window_size']"),
    "spatial": (ValueError, "build", _NO_DIM_DROPOUT_HEADS),
    "tripse1": (ValueError, "build", _NO_DIM_DROPOUT_HEADS),
    "tripse2": (ValueError, "build", _NO_DIM_DROPOUT_HEADS),
    "tripse3": (ValueError, "build", _NO_DIM_DROPOUT_HEADS),
    "tripse4": (ValueError, "build", _NO_DIM_DROPOUT_HEADS),
    "window": (ValueError, "build", "Required parameters for 'window' are missing: ['window_size']"),
    "window_band": (ValueError, "build", "Required parameters for 'window_band' are missing: ['window_size']"),
    "window_zigzag": (
        ValueError, "build", "Required parameters for 'window_zigzag' are missing: ['window_size']"),
}


def test_the_refused_types_and_the_supported_ones_are_every_registry_type() -> None:
    """A type added to the attention registry is either supported here or named with its reason."""
    known = set(SUPPORTED) | set(_REFUSED)
    assert set(ATTENTION_REGISTRY) - known == set(), sorted(set(ATTENTION_REGISTRY) - known)
    assert not set(SUPPORTED) & set(_REFUSED)
    assert len(_REFUSED) == 25


@pytest.mark.parametrize("attention_type", sorted(_REFUSED))
def test_a_refused_type_fails_for_the_reason_the_table_gives(
        attention_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal is not a layer defect: patched in, the type still cannot be built or run here."""
    exception, stage, fragment = _REFUSED[attention_type]
    monkeypatch.setitem(dino_v2.SUPPORTED_ATTENTION_TYPES, attention_type, None)
    keras.utils.set_random_seed(1234)

    def build() -> DINOv2Block:
        return DINOv2Block(dim=DIM, num_heads=HEADS, attention_type=attention_type, name="blk")

    if stage == "build":
        with pytest.raises(exception) as raised:
            build()
    else:
        block = build()
        with pytest.raises(exception) as raised:
            block(_tokens(), training=False)
    assert fragment in str(raised.value), str(raised.value)[:300]


class TestTheSupportedSet:

    def test_the_supported_types_are_exactly_these(self) -> None:
        assert sorted(dino_v2.SUPPORTED_ATTENTION_TYPES) == sorted(SUPPORTED)
        assert all(f"``{name}``" in DINOv2Block.__doc__ for name in SUPPORTED), (
            "the attention_type docstring of the block must name the supported set")

    @pytest.mark.parametrize("attention_type", [*sorted(_REFUSED), "multi_head_attention",
                                                "no_such_attention"])
    def test_a_block_refuses_an_unsupported_type_naming_the_supported_set(
            self, attention_type: str) -> None:
        with pytest.raises(ValueError, match=re.escape(
                f"attention_type must be one of {sorted(SUPPORTED)}, got {attention_type!r}")):
            DINOv2Block(dim=DIM, num_heads=HEADS, attention_type=attention_type)

    def test_the_backbone_refuses_it_at_construction_too(self) -> None:
        with pytest.raises(ValueError, match=re.escape(
                f"attention_type must be one of {sorted(SUPPORTED)}, got 'differential'")):
            _backbone("differential")

    def test_the_refusal_comes_before_any_layer_is_built(self) -> None:
        keras.utils.set_random_seed(7)
        seeds = [keras.initializers.HeNormal().seed for _ in range(2)]
        keras.utils.set_random_seed(7)
        with pytest.raises(ValueError):
            DINOv2Block(dim=DIM, num_heads=HEADS, attention_type="differential")
        assert keras.initializers.HeNormal().seed == seeds[0]


class TestTheSeededDraws:
    """Carve-out C-7: ``qkv`` keeps the first default seed of the block, ``proj`` takes the second."""

    #: Default seeds one default block (the ``mlp`` FFN draws three) takes from the stream.
    SEEDS_PER_BLOCK = 5

    def _stream(self, count: int) -> Tuple[int, ...]:
        keras.utils.set_random_seed(7)
        return tuple(keras.initializers.HeNormal().seed for _ in range(count))

    def test_qkv_keeps_the_first_seed_and_proj_takes_the_second(self) -> None:
        stream = self._stream(self.SEEDS_PER_BLOCK + 1)
        keras.utils.set_random_seed(7)
        engine = _engine(DINOv2Block(dim=DIM, num_heads=HEADS, name="blk"))
        assert engine.qkv_dense.kernel_initializer.seed == stream[0]
        assert engine.proj_dense.kernel_initializer.seed == stream[1]
        assert stream[0] != stream[1]

    def test_qkv_is_the_draw_a_plain_dense_makes_at_that_position(self) -> None:
        keras.utils.set_random_seed(1234)
        want = np.asarray(keras.initializers.get("he_normal")((DIM, 3 * DIM), dtype="float32"))
        keras.utils.set_random_seed(1234)
        got = np.asarray(_engine(_block()).qkv_dense.kernel)
        assert np.array_equal(got, want)

    def test_proj_is_the_draw_of_the_next_position(self) -> None:
        keras.utils.set_random_seed(1234)
        first = keras.initializers.get("he_normal")
        second = keras.initializers.get("he_normal")
        want = np.asarray(second((DIM, DIM), dtype="float32"))
        got_qkv_first = np.asarray(first((DIM, 3 * DIM), dtype="float32"))
        keras.utils.set_random_seed(1234)
        got = np.asarray(_engine(_block()).proj_dense.kernel)
        assert np.array_equal(got, want)
        assert not np.array_equal(got.ravel()[:256], got_qkv_first.ravel()[:256])

    def test_a_block_takes_exactly_one_more_default_seed_than_before(self) -> None:
        """Every default-seeded tensor created after a block's ``proj`` sits one seed further down
        the stream (the FFN of the block, the later blocks, the classifier of the wrapper)."""
        stream = self._stream(self.SEEDS_PER_BLOCK + 1)
        keras.utils.set_random_seed(7)
        DINOv2Block(dim=DIM, num_heads=HEADS, name="blk")
        assert keras.initializers.HeNormal().seed == stream[self.SEEDS_PER_BLOCK]


class TestNoWeightNameChanged:

    #: The weights of a default block named ``blk``, in creation order: the attention's names are
    #: what the official-weight loader and every archive address.
    PATHS = [
        "blk/norm1/gamma", "blk/norm1/beta", "blk/norm2/gamma", "blk/norm2/beta",
        "blk/attention/cross_attention/qkv/kernel", "blk/attention/cross_attention/qkv/bias",
        "blk/attention/cross_attention/proj/kernel", "blk/attention/cross_attention/proj/bias",
        "blk/ffn/fc1/kernel", "blk/ffn/fc1/bias", "blk/ffn/fc2/kernel", "blk/ffn/fc2/bias",
        "blk/ls1/gamma", "blk/ls2/gamma",
    ]

    def test_the_block_holds_the_same_weights_under_the_same_paths(self) -> None:
        assert [w.path for w in _block().weights] == self.PATHS

    def test_the_config_is_unchanged(self) -> None:
        config = _block().get_config()
        assert config["attention_type"] == "multi_head"
        assert not {k for k in config if "initializer" in k}, "no initializer enters the config"
        rebuilt = DINOv2Block.from_config(config)
        assert rebuilt.get_config() == config

    def test_weights_keyed_by_path_transplant_between_two_builds(self) -> None:
        keras.utils.set_random_seed(1)
        source = _backbone()
        keras.utils.set_random_seed(2)
        target = _backbone()
        by_path = {w.path: np.asarray(w) for w in source.weights}
        assert {w.path for w in target.weights} == set(by_path)
        for weight in target.weights:
            weight.assign(by_path[weight.path])
        want = _forward_backbone(source)["x_norm_clstoken"]
        got = _forward_backbone(target)["x_norm_clstoken"]
        assert np.array_equal(np.asarray(want), np.asarray(got))


class TestTheRuntimeContractReplaysDinoV2:
    """The shared contract's replay rows are the guard that fails when a block shares a draw again;
    these keep the rows themselves from being dropped or pointed at another model."""

    @staticmethod
    def _subjects() -> Dict[str, Tuple]:
        from tests.test_models.test_the_vision_classifiers_share_one_runtime_contract import (
            REPLAY_SUBJECTS,
        )
        return {key: REPLAY_SUBJECTS[key]
                for key in ("dinov2_vit", "dinov2_classifier",
                            *(f"dinov2_vit_{name}" for name in SUPPORTED if name != "multi_head"))}

    def test_a_row_exists_for_the_backbone_the_wrapper_and_every_other_supported_type(self) -> None:
        assert len(self._subjects()) == 2 + len(SUPPORTED) - 1

    def test_each_row_builds_the_model_and_the_attention_type_it_is_named_for(self) -> None:
        for key, (build, knob) in self._subjects().items():
            assert knob is False, f"{key}: the backbone takes no initializer to arm"
            model = build(None)
            backbone = getattr(model, "backbone", model)
            want = key.removeprefix("dinov2_vit_") if key.startswith("dinov2_vit_") else "multi_head"
            assert backbone.attention_type == want, key
            assert isinstance(model, DINOv2 if key == "dinov2_classifier" else DINOv2VisionTransformer)

    def test_the_real_depth_anything_row_is_no_longer_ledgered(self) -> None:
        from tests.test_models.test_the_vision_classifiers_share_one_runtime_contract import LEDGER
        assert ("no_replayed_draw::default", "depth_anything_real") not in LEDGER

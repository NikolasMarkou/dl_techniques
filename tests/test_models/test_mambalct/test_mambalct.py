"""Tests for the MambaLCT tracker: shapes, context sensitivity, serialization."""

import inspect
import io
import json
import os
import re
import tempfile
import zipfile

import h5py
import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.multi_head_cross_attention import MultiHeadCrossAttention
from dl_techniques.models.vision.mambalct import MambaLCT, create_mambalct
from dl_techniques.models.vision.mambalct.model import FUSION_ATTENTION_INITIALIZER
from dl_techniques.models.vision.mambalct.uca_encoder import UcaEncoder

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight
from ..initializer_replay_oracle import replayed_pairs


def _tiny_model() -> MambaLCT:
    return MambaLCT(
        template_size=32,
        search_size=64,
        context_len=1,
        stage_dims=[64, 128, 512],
        stage_depths=[1, 1, 1],
        num_heads=[2, 4, 8],
        fusion_heads=8,
        head_hidden_dim=32,
        name="mambalct_tiny",
    )


@pytest.fixture
def tiny_inputs() -> tuple:
    rng = np.random.default_rng(0)
    template = rng.uniform(0, 1, size=(1, 32, 32, 3)).astype(np.float32)
    search = rng.uniform(0, 1, size=(1, 2, 64, 64, 3)).astype(np.float32)
    return template, search


def test_forward_shapes_and_finite(tiny_inputs: tuple) -> None:
    model = _tiny_model()
    out = model(list(tiny_inputs), training=False)
    assert tuple(out["scores"].shape) == (1, 2, 1)
    assert tuple(out["boxes"].shape) == (1, 2, 4)
    assert tuple(out["updated_context"].shape) == (1, 1, 512)
    for value in out.values():
        assert np.all(np.isfinite(np.asarray(value)))


def test_boxes_in_unit_range(tiny_inputs: tuple) -> None:
    model = _tiny_model()
    out = model(list(tiny_inputs), training=False)
    boxes = np.asarray(out["boxes"])
    assert boxes.min() >= 0.0 and boxes.max() <= 1.0


def test_context_changes_output(tiny_inputs: tuple) -> None:
    """Incoming context reaches the frames, and the frames reach the context.

    The defects this guards are PATH defects: under a tail-append framing the
    causal scan gives the frames no path to the context, and a bridge that
    ignored the frames would return the context unchanged. Either makes the
    two compared outputs bit-identical. So the assertion is the house
    dead-component convention (``dead_component_oracle.component_response``,
    ``atol=0.0``): on a deterministic forward, any difference at all is a
    response.

    Until plan-2026-10-14T062616-81938e81 step 6 this compared against a pasted
    ``1e-4`` on an unseeded model. MEASURED over 20 init seeds on GPU 1
    (``scratchpad/step6/probes/b6.py``): box difference 5.6e-05 .. 4.3e-04
    (median 2.1e-04), BELOW 1e-4 for 3 of 20 seeds; context-history difference
    3.9e-03 .. 1.1e-02; the same input twice differed by exactly 0.0 in every
    output for all 20. The 1e-4 bar sat inside the effect's own spread, so the
    verdict was decided by the init draw (baseline B-6 red at 3.2e-05, green
    8/8 at step 1). The premise is now asserted, the model is seeded, and the
    bar is the bit-identity the defect produces.
    """
    keras.utils.set_random_seed(0)
    model = _tiny_model()
    template, search = tiny_inputs
    rng = np.random.default_rng(7)
    # Random (not constant) contexts: a per-token norm maps any constant
    # token to beta, so constant pairs can only differ by fp dust.
    ctx_a = rng.normal(0, 1, size=(1, 1, 512)).astype(np.float32)
    ctx_b = rng.normal(0, 1, size=(1, 1, 512)).astype(np.float32)
    out_a = model([template, search, ctx_a], training=False)
    out_a_again = model([template, search, ctx_a], training=False)
    out_b = model([template, search, ctx_b], training=False)

    # The premise of a zero bar: the forward is deterministic.
    for key in out_a:
        np.testing.assert_array_equal(
            np.asarray(out_a[key]), np.asarray(out_a_again[key]),
            err_msg=f"the same input gave two {key!r}; a difference below "
                    f"would be noise, not a response")

    # Incoming context must reach the frames: under a tail-append framing
    # the causal scan gives frames no path to later tokens, so this would be
    # bit-identical.
    box_diff = np.abs(np.asarray(out_a["boxes"]) - np.asarray(out_b["boxes"])).max()
    assert box_diff > 0.0, "the boxes do not depend on the incoming context"
    # History aggregation: different frames with the same context must move
    # the update (the bridge output is the history summary).
    rng2 = np.random.default_rng(9)
    other_search = rng2.uniform(0, 1, size=(1, 2, 64, 64, 3)).astype(np.float32)
    out_c = model([template, other_search, ctx_a], training=False)
    hist_diff = np.abs(
        np.asarray(out_a["updated_context"]) - np.asarray(out_c["updated_context"])
    ).max()
    assert hist_diff > 0.0, "the updated context does not depend on the frames"
    assert np.all(np.isfinite(np.asarray(out_c["updated_context"])))


def _relative_weights(model: keras.Model):  # type: ignore[no-untyped-def]
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


def test_explicit_build_matches_lazy_build() -> None:
    explicit = _tiny_model()
    explicit.build([(None, 32, 32, 3), (None, 2, 64, 64, 3), (None, 1, 512)])
    lazy = _tiny_model()
    rng = np.random.default_rng(8)
    lazy(
        [
            rng.uniform(0, 1, size=(1, 32, 32, 3)).astype(np.float32),
            rng.uniform(0, 1, size=(1, 2, 64, 64, 3)).astype(np.float32),
            np.zeros((1, 1, 512), dtype=np.float32),
        ]
    )
    assert _relative_weights(explicit) == _relative_weights(lazy)
    # Pair nest builds the identical population (context is zeros-filled).
    pair = _tiny_model()
    pair.build([(None, 32, 32, 3), (None, 2, 64, 64, 3)])
    assert _relative_weights(pair) == _relative_weights(lazy)


def test_symbolic_frames_build() -> None:
    # Regression: Python arithmetic on shape tensors broke build_from_config
    # with a dynamic frames axis (and with it, pretrained weight loading).
    model = _tiny_model()
    model.build([(None, 32, 32, 3), (None, None, 64, 64, 3), (None, 1, 512)])
    assert model.built
    lazy = _tiny_model()
    lazy(
        [
            np.zeros((1, 32, 32, 3), dtype=np.float32),
            np.zeros((1, 2, 64, 64, 3), dtype=np.float32),
            np.zeros((1, 1, 512), dtype=np.float32),
        ]
    )
    assert _relative_weights(model) == _relative_weights(lazy)


def test_subclass_save_load_restores_weights(tiny_inputs: tuple) -> None:
    template, search = tiny_inputs
    context = np.zeros((1, 1, 512), dtype=np.float32)
    model = _tiny_model()
    model.build([(None, 32, 32, 3), (None, 2, 64, 64, 3), (None, 1, 512)])
    saved = [np.asarray(w).copy() for w in model.weights]
    assert saved, "donor has no weights to compare"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mambalct_sub.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        # BEFORE any forward on `loaded`: afterwards a build gap would be
        # silently filled with fresh random weights at the same count.
        assert len(loaded.weights) == len(saved)
        for donor, restored in zip(saved, loaded.weights):
            np.testing.assert_allclose(
                donor, np.asarray(restored), atol=0.0,
                err_msg="subclass weights were not restored",
            )
        out_orig = model([template, search, context], training=False)
        out_rest = loaded([template, search, context], training=False)
    for key in ("scores", "boxes", "updated_context"):
        np.testing.assert_allclose(
            np.asarray(out_orig[key]), np.asarray(out_rest[key]),
            atol=1e-6, rtol=0,
        )


def test_variant_contract() -> None:
    model = create_mambalct("mambalct-256")
    assert model.template_size == 128
    assert model.search_size == 256
    with pytest.raises(ValueError, match="mambalct-256"):
        create_mambalct("nope")
    with pytest.raises(NotImplementedError):
        create_mambalct("mambalct-256", pretrained=True)


def test_serialization_round_trip(tiny_inputs: tuple) -> None:
    template, search = tiny_inputs
    context = np.zeros((1, 1, 512), dtype=np.float32)
    t_in = keras.Input(shape=(32, 32, 3))
    s_in = keras.Input(shape=(2, 64, 64, 3))
    c_in = keras.Input(shape=(1, 512))
    tiny = _tiny_model()
    out = tiny([t_in, s_in, c_in])
    model = keras.Model([t_in, s_in, c_in], out)
    orig = model([template, search, context], training=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mambalct.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        rest = loaded([template, search, context], training=False)
    for key in ("scores", "boxes", "updated_context"):
        np.testing.assert_allclose(
            np.asarray(orig[key]), np.asarray(rest[key]), atol=1e-6, rtol=0
        )


def _tiny_uca_encoder() -> UcaEncoder:
    return UcaEncoder(stage_dims=[16, 32], stage_depths=[1, 1], num_heads=[2, 4])


def _count_calls(layer: keras.layers.Layer) -> list:
    """Wrap ``layer.call`` so each execution appends to the returned list.

    Interface contract: returns the (initially empty) list the wrapper appends
    ``1`` to on every call of ``layer.call``; the layer is mutated in place and
    is never restored, so use it on a throwaway instance.
    """
    calls: list = []
    original_call = layer.call

    def counting_call(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original_call(*args, **kwargs)

    layer.call = counting_call
    return calls


def test_the_uca_encoder_applies_its_stem_norm_exactly_once() -> None:
    """The stem is conv -> norm ONCE (the pre-migration encoder).

    The migration onto ``ConvStem`` (which already owns the norm) left a second
    ``self.stem_norm(x)`` in ``call()``: the stem norm ran twice and the encoder
    computed a different function. Counting the calls of ``stem.norm1`` during
    one forward sees it. Positive controls: the counter sees the single call of
    ``final_norm``, and it reads 2 on a planted encoder that applies the stem
    norm twice, so a reading of 1 is a real count and not a blind counter.
    """
    x = np.random.default_rng(0).uniform(0, 1, (1, 16, 16, 3)).astype("float32")

    encoder = _tiny_uca_encoder()
    stem_norm_calls = _count_calls(encoder.stem.norm1)
    final_norm_calls = _count_calls(encoder.final_norm)
    encoder(x, training=False)
    assert len(final_norm_calls) == 1, "the counter did not see final_norm run"
    assert len(stem_norm_calls) == 1, (
        f"the stem norm ran {len(stem_norm_calls)} times in one forward; "
        f"the pre-migration encoder applied it once"
    )

    planted = _tiny_uca_encoder()
    planted_calls = _count_calls(planted.stem.norm1)
    stem_call = planted.stem.call

    def stem_then_norm_again(*args, **kwargs):  # type: ignore[no-untyped-def]
        out = stem_call(*args, **kwargs)
        return planted.stem.norm1(out, training=kwargs.get("training"))

    planted.stem.call = stem_then_norm_again
    planted(x, training=False)
    assert len(planted_calls) == 2, "the counter cannot see a planted double norm"


def test_the_uca_encoder_has_no_stem_aliases() -> None:
    """``stem`` is the only stem handle: no ``stem_conv`` / ``stem_norm`` alias."""
    encoder = _tiny_uca_encoder()
    assert not hasattr(encoder, "stem_conv")
    assert not hasattr(encoder, "stem_norm")


# ---------------------------------------------------------------------------
# UcaEncoder: arguments, configuration, and the draws of its attention projections
# ---------------------------------------------------------------------------


def _images(height: int = 16, width: int = 16) -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 1, (1, height, width, 3)).astype("float32")


def test_the_uca_encoder_config_is_json_safe_and_round_trips() -> None:
    encoder = UcaEncoder(
        stage_dims=[16, 32], stage_depths=[1, 2], num_heads=[2, 4], mlp_ratio=2.0,
        norm_epsilon=1e-3, dropout_rate=0.1, use_bias=False, name="uca")
    config = encoder.get_config()
    json.dumps(config)
    rebuilt = UcaEncoder.from_config(config)
    assert rebuilt.get_config() == config
    assert [b.name for b in rebuilt.blocks] == [b.name for b in encoder.blocks]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(stage_dims=[16, 32], stage_depths=[1], num_heads=[2, 4]),
         "stage_dims ([16, 32]), stage_depths ([1]) and num_heads ([2, 4]) must have equal length"),
        (dict(stage_dims=[16, 0], stage_depths=[1, 1], num_heads=[2, 4]),
         "stage_dims must be positive, got [16, 0]"),
        (dict(stage_dims=[16, 32], stage_depths=[1, 0], num_heads=[2, 4]),
         "stage_depths must be positive, got [1, 0]"),
        (dict(stage_dims=[16, 32], stage_depths=[1, 1], num_heads=[2, 0]),
         "num_heads must be positive, got [2, 0]"),
        (dict(stage_dims=[16, 30], stage_depths=[1, 1], num_heads=[2, 4]),
         "each stage_dim must be divisible by its num_heads, got [16, 30] / [2, 4]"),
        (dict(mlp_ratio=0.0), "mlp_ratio must be positive, got 0.0"),
        (dict(norm_epsilon=0.0), "norm_epsilon must be positive, got 0.0"),
        (dict(dropout_rate=1.0), "dropout_rate must be in [0, 1), got 1.0"),
    ],
)
def test_a_bad_uca_argument_raises_the_encoders_own_message_naming_the_value(kwargs, message) -> None:
    """The message starts with the encoder's own sentence: a bad ``dropout_rate`` would also be
    refused later by the transformer block, with that layer's wording, if the encoder let it by."""
    with pytest.raises(ValueError, match=f"^{re.escape(message)}"):
        UcaEncoder(**kwargs)


def test_the_uca_encoder_refuses_an_input_that_is_not_an_image_batch() -> None:
    with pytest.raises(ValueError, match="4D"):
        _tiny_uca_encoder().build((None, 16, 3))


def test_the_uca_encoder_reduces_the_stride_by_sixteen_at_every_input_size() -> None:
    encoder = UcaEncoder(stage_dims=[16, 32, 32], stage_depths=[1, 1, 1], num_heads=[2, 4, 4])
    for height, width in ((32, 32), (32, 48)):
        tokens = encoder(_images(height, width), training=False)
        assert tuple(tokens.shape) == (1, (height // 16) * (width // 16), 32)


def test_every_block_of_the_uca_encoder_draws_each_projection_independently() -> None:
    """No two kernels of a built encoder carry one random draw, in particular a block's
    ``qkv`` and ``proj`` (one initializer instance gave both the same stream) and the
    ``proj`` kernels of two blocks of one width."""
    keras.utils.set_random_seed(1234)
    encoder = UcaEncoder(stage_dims=[16, 32], stage_depths=[2, 2], num_heads=[2, 4])
    encoder(_images(), training=False)
    pairs, compared = replayed_pairs(encoder)
    assert compared > 0, "the encoder exposes fewer than two kernels: nothing was compared"
    assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"


SUPPORTED_ATTENTION_TYPES = ["multi_head", "group_query", "multi_head_latent"]


@pytest.mark.parametrize("attention_type", SUPPORTED_ATTENTION_TYPES)
def test_every_supported_attention_type_runs_and_draws_each_projection_independently(
        attention_type: str) -> None:
    """For every attention type the encoder accepts, no two kernels of a built encoder carry
    one random draw (two blocks per stage, so a draw replayed across blocks is seen too)."""
    keras.utils.set_random_seed(1234)
    encoder = UcaEncoder(
        stage_dims=[16, 32], stage_depths=[2, 2], num_heads=[2, 4], attention_type=attention_type)
    tokens = encoder(_images(), training=False)
    assert tuple(tokens.shape) == (1, 4, 32) and np.isfinite(np.asarray(tokens)).all()
    pairs, compared = replayed_pairs(encoder)
    assert compared > 0, "the encoder exposes fewer than two kernels: nothing was compared"
    assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"


def test_the_supported_attention_types_are_exactly_these() -> None:
    from dl_techniques.models.vision.mambalct import uca_encoder
    assert sorted(uca_encoder.SUPPORTED_ATTENTION_TYPES) == sorted(SUPPORTED_ATTENTION_TYPES)
    assert all(f'"{name}"' in UcaEncoder.__doc__ for name in SUPPORTED_ATTENTION_TYPES), (
        "the attention_type docstring must name the supported set")


@pytest.mark.parametrize("attention_type", [
    "differential", "anchor", "window", "window_zigzag", "window_band", "beit", "lighthouse",
    "fnet", "multi_head_cross", "no_such_attention"])
def test_an_unsupported_attention_type_is_refused_naming_the_supported_set(
        attention_type: str) -> None:
    with pytest.raises(ValueError, match=re.escape(
            "attention_type must be one of ['group_query', 'multi_head', 'multi_head_latent'], "
            f"got {attention_type!r}")):
        UcaEncoder(stage_dims=[16, 32], stage_depths=[1, 1], num_heads=[2, 4],
                   attention_type=attention_type)


#: Why each refused type is refused, as the encoder measured it: the types that replay a draw hand
#: ONE initializer instance to several projections and declare no argument to separate them; the
#: others cannot run on the encoder's token shape. The rows are strict expected failures: when a
#: library layer is fixed the row passes, the run goes red, and the type can join the supported set.
_REFUSED = {
    "differential": "60 of 630 kernel pairs replay one draw (q1 ~ k1 ~ q2 ...)",
    "anchor": "40 of 496 kernel pairs replay one draw (query_proj ~ key_proj ~ value_proj)",
    "window": "4 of 276 kernel pairs replay one draw (qkv ~ proj)",
    "window_band": "4 of 190 kernel pairs replay one draw (qkv ~ proj)",
    "window_zigzag": "ValueError: needs a fixed sequence length for XLA",
    "beit": "InvalidArgumentError in BeitAttention.call at the encoder's token shape",
    "lighthouse": "RuntimeError: needs a statically known sequence length",
    "fnet": "ValueError: sequence length and hidden dimension must be known at build time",
    "multi_head_cross": "ValueError: TransformerLayer has no such attention type",
}


@pytest.mark.parametrize("attention_type", [
    pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=f"measured: {why}"), id=name)
    for name, why in _REFUSED.items()])
def test_a_refused_attention_type_would_have_to_run_and_draw_independently(
        attention_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from dl_techniques.models.vision.mambalct import uca_encoder
    monkeypatch.setitem(uca_encoder.SUPPORTED_ATTENTION_TYPES, attention_type, None)
    keras.utils.set_random_seed(1234)
    encoder = UcaEncoder(
        stage_dims=[16, 32], stage_depths=[2, 2], num_heads=[2, 4], attention_type=attention_type)
    assert np.isfinite(np.asarray(encoder(_images(), training=False))).all()
    pairs, _ = replayed_pairs(encoder)
    assert not pairs, f"{len(pairs)} kernel pairs replay one draw: {pairs[:2]}"


@pytest.mark.parametrize("attention_type", ["group_query", "multi_head_latent"])
def test_the_types_that_clone_per_projection_themselves_are_given_no_extra_arguments(
        attention_type: str) -> None:
    encoder = UcaEncoder(
        stage_dims=[16, 32], stage_depths=[1, 1], num_heads=[2, 4], attention_type=attention_type)
    assert all(block.attention_args == {} for block in encoder.blocks)


def test_the_multi_head_blocks_are_given_the_output_initializer() -> None:
    encoder = _tiny_uca_encoder()
    assert all(block.attention_args == {"output_kernel_initializer": "glorot_uniform"}
               for block in encoder.blocks)


def test_a_saved_mambalct_stores_the_encoder_under_the_attribute_names_old_archives_use() -> None:
    """A ``.keras`` archive stores a layer under its ATTRIBUTE path, not its layer name, so
    renaming ``merges``, ``merge_norms``, ``cpe_convs`` or ``blocks`` makes every archive
    written before the rename refuse to load, while renaming a layer (``downsample_0_conv``
    for ``merge_1``) does not. The names below are the ones the archives of the previous
    revision carry."""
    model = _tiny_model()
    model.build([(None, 32, 32, 3), (None, 2, 64, 64, 3), (None, 1, 512)])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mambalct_keys.keras")
        model.save(path)
        with zipfile.ZipFile(path) as archive:
            store = h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r")
    assert set(store["encoder"]) == {
        "blocks", "cpe_convs", "final_norm", "merge_norms", "merges", "stem", "vars"}


def test_the_uca_encoder_layers_are_named_by_the_table_whatever_their_attribute() -> None:
    encoder = UcaEncoder(stage_dims=[16, 32, 32], stage_depths=[2, 1, 1], num_heads=[2, 4, 4])
    assert [layer.name for layer in encoder.merges] == ["downsample_0_conv", "downsample_1_conv"]
    assert [layer.name for layer in encoder.merge_norms] == ["downsample_0_norm", "downsample_1_norm"]
    assert [layer.name for layer in encoder.cpe_convs] == ["stage1_cpe", "stage2_cpe", "stage3_cpe"]
    assert [layer.name for layer in encoder.blocks] == [
        "stage1_block1", "stage1_block2", "stage2_block1", "stage3_block1"]


def test_the_attention_output_projection_starts_from_the_same_distribution_as_qkv() -> None:
    """The output projection is given ``ATTENTION_OUTPUT_INITIALIZER`` while ``qkv`` takes
    ``TransformerLayer``'s own default; the two must be the same kind of initializer, or the
    projection would start from a distribution the rest of the block does not use."""
    encoder = _tiny_uca_encoder()
    encoder(_images(), training=False)
    for block in encoder.blocks:
        engine = block.attention.cross_attention
        assert type(engine.proj_dense.kernel_initializer) is type(engine.qkv_dense.kernel_initializer)
        assert type(engine.proj_dense.kernel_initializer) is type(block.ffn_layer.fc1.kernel_initializer)


def test_every_block_of_the_uca_encoder_is_on_the_forward_path() -> None:
    """Each of the 48 trainable weights of a two-stage, three-block encoder receives a finite,
    non-zero gradient: a stage that ran fewer blocks than it built would leave the others dead."""
    encoder = UcaEncoder(stage_dims=[16, 32], stage_depths=[2, 1], num_heads=[2, 4])
    x = np.random.default_rng(0).uniform(0, 1, (2, 16, 16, 3)).astype("float32")
    encoder(x, training=False)
    report = assert_gradients_reach_every_trainable_weight(encoder, x, training=False)
    assert len(report) == len(encoder.trainable_weights) == 48


# ---------------------------------------------------------------------------
# MambaLCT: the draws of the fusion attention
# ---------------------------------------------------------------------------


def _built_tiny_model() -> MambaLCT:
    keras.utils.set_random_seed(1234)
    model = _tiny_model()
    model.build([(None, 32, 32, 3), (None, 2, 64, 64, 3), (None, 1, 512)])
    return model


def test_the_fusion_attention_projections_start_from_independent_draws() -> None:
    """`q`, `kv` and `proj` of the fusion attention: the layer hands one initializer instance to
    all three by default, and the smaller kernels are then rescaled prefixes of the largest."""
    fusion = _built_tiny_model().fusion_attn
    pairs, compared = replayed_pairs(fusion)
    assert compared == 3, "the three projection kernels were not all compared"
    assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs}"
    initializers = [fusion.q_dense.kernel_initializer, fusion.kv_dense.kernel_initializer,
                    fusion.proj_dense.kernel_initializer]
    assert len({id(i) for i in initializers}) == 3


def test_no_two_kernels_of_a_tiny_mambalct_carry_one_draw() -> None:
    pairs, compared = replayed_pairs(_built_tiny_model())
    assert compared > 100, "the tracker's kernels were not compared"
    assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"


def test_the_fusion_initializer_is_the_layers_own_default() -> None:
    """Naming the initializer must not change the distribution the projections start from."""
    default = inspect.signature(MultiHeadCrossAttention.__init__).parameters["kernel_initializer"].default
    assert type(keras.initializers.get(FUSION_ATTENTION_INITIALIZER)) is type(
        keras.initializers.get(default))

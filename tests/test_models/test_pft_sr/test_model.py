"""
Test suite for PFTSR (Progressive Focused Transformer super-resolution).

create_pft_sr(scale, variant) builds the model; the paper-sourced variants use
window_size=32 (repo_medium keeps 8), so input H/W are
kept divisible by 32. NHWC float32 input (B, H, W, 3); at scale=2 the output is
the upsampled image (B, 2H, 2W, 3). Covers a forward pass and the M2 full
.keras save -> load -> identical-output round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.vision.super_resolution.pft_sr.model import create_pft_sr, PFTSR

SIZE = 32
SCALE = 2


def _model():
    return create_pft_sr(scale=SCALE, variant="light")


def _images(batch=2):
    return np.random.default_rng(0).random((batch, SIZE, SIZE, 3)).astype("float32")


class TestPFTSR:

    def test_factory_construction(self):
        assert isinstance(_model(), PFTSR)

    def test_forward_upsamples(self):
        out = _model()(_images(), training=False)
        assert tuple(out.shape) == (2, SIZE * SCALE, SIZE * SCALE, 3)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            create_pft_sr(scale=2, variant="nonexistent")

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _images()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "pft_sr.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="PFTSR differs after .keras round-trip")

    def test_keras_round_trip_at_nonzero_drop_path_rate(self, tmp_path):
        """SC-8: round-trip the STOCHASTIC-DEPTH path, by value.

        ``create_pft_sr`` hard-codes ``drop_path_rate=0.0``, so
        ``test_keras_round_trip`` above exercises the DEFAULT configuration and
        cannot observe the ``drop_path_rate > 0`` branch at all -- the branch
        that was dead on arrival until iter-1/step-8 (a ``StochasticDepth``
        kwarg name plus an ``EagerTensor.item()`` call in the schedule). This is
        the same class of gap step 6 found for ``use_free_transformer``: a
        round-trip test that exists but never reaches the new path.

        ``training=False`` makes ``StochasticDepth`` the identity, so the
        comparison is deterministic; the schedule assertion below is what proves
        the reloaded model really carries the non-zero rates.
        """
        model = PFTSR(
            scale=SCALE, in_channels=3, embed_dim=16, num_blocks=[1, 1],
            num_heads=4, window_size=8, mlp_ratio=2.0, drop_path_rate=0.1,
        )
        x = np.random.default_rng(11).random((1, 16, 16, 3)).astype("float32")
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "pft_sr_dp.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        assert float(np.max(np.abs(before))) > 1e-5, \
            "round-trip compared all-zero values"
        np.testing.assert_allclose(
            before, after, atol=1e-4,
            err_msg="PFTSR differs after .keras round-trip at drop_path_rate>0")
        assert loaded.drop_path_rate == pytest.approx(0.1)
        assert max(loaded.dpr) == pytest.approx(0.1), \
            f"the reloaded model lost its stochastic-depth schedule: {loaded.dpr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestTheDropPathScheduleReachesTheLayers:
    """Moved from the shared transformer block's tests (their
    ``TestPftSrConsumerSmoke``) when PFT-SR stopped building that block
    (plan-2026-10-14T062616-81938e81 step 7, D-212; the block and its tests
    were deleted in step 7.1, D-216). The official PFT has no
    stochastic depth; this port keeps it as an opt-in, zero by default."""

    def test_a_nonzero_rate_builds_and_runs_in_training(self):
        model = PFTSR(scale=2, in_channels=3, embed_dim=16, num_blocks=[1, 1],
                      num_heads=4, window_size=8, mlp_ratio=2.0, drop_path_rate=0.1)
        x = np.random.default_rng(5).standard_normal((1, 16, 16, 3)).astype("float32")
        out = keras.ops.convert_to_numpy(model(x, training=True))
        assert out.shape == (1, 32, 32, 3)
        assert np.all(np.isfinite(out))

    def test_each_layer_gets_its_linear_rate(self):
        model = PFTSR(scale=2, in_channels=3, embed_dim=16, num_blocks=[2, 2],
                      num_heads=4, window_size=8, mlp_ratio=2.0, drop_path_rate=0.4)
        rates = [layer.drop_path_rate for layer in model.pft_layers]
        # linear_drop_path_rates returns float32-rounded floats.
        assert rates == pytest.approx([0.0, 0.4 / 3, 0.8 / 3, 0.4], abs=1e-6)
        assert model.pft_layers[0].drop_path is None
        assert all(layer.drop_path is not None for layer in model.pft_layers[1:])
        assert all(layer.drop_path is None
                   for layer in PFTSR(scale=2, embed_dim=16, num_blocks=[2, 2], num_heads=4,
                                      window_size=8).pft_layers)


def test_the_upsampling_scale_is_not_a_norm_knob():
    """``PFTSR.scale`` is the upsampling factor; the LayerNorm ``scale`` (a bool,
    the learned gamma) is a different thing that only shares the name, which is
    why ``test_package_api_contract._NORM_NAME_COLLISIONS`` waives it."""
    model = PFTSR(scale=4, embed_dim=16, num_blocks=[1], num_heads=2, window_size=8)
    assert model.scale == 4
    assert model.patch_norm.scale is True and model.norm.scale is True


class TestNearestUpsamplerScaleValidation:
    """``'nearest+conv'`` is the official real-world tail, and it is x4 only.

    The official ``PFT.__init__`` asserts ``self.upscale == 4`` for it: the
    tail has exactly two nearest 2x stages (``conv_up1``, ``conv_up2``). This
    port used to stack ``int(log2(scale))`` stages and accept any power of two
    (anchor ``plan-2026-08-14T233721-d4f9beb2/D-078``, retired at
    plan-2026-10-14T062616-81938e81 step 6.1), a generalization the official
    code does not have. The refusal now fires at construction.
    """

    @staticmethod
    def _build(scale):
        model = PFTSR(
            scale=scale,
            upsampler="nearest+conv",
            embed_dim=16,
            num_blocks=[1],
            num_heads=2,
            window_size=8,
        )
        model.build((None, 16, 16, 3))
        return model

    @pytest.mark.parametrize("scale", [2, 3, 5, 8])
    def test_any_scale_but_four_raises(self, scale):
        with pytest.raises(ValueError, match=r"nearest\+conv"):
            self._build(scale)

    def test_four_builds_and_realizes_four(self):
        """Anti-vacuity: the guard must not reject everything, and what built
        must upsample by 4."""
        model = self._build(4)
        x = np.random.default_rng(0).random((1, 16, 16, 3)).astype("float32")
        assert tuple(model(x, training=False).shape[1:3]) == (64, 64)

    @pytest.mark.parametrize("upsampler, scale", [
        ("pixelshuffle", 3), ("pixelshuffle", 8), ("pixelshuffledirect", 3),
    ])
    def test_the_other_tails_keep_their_official_scales(self, upsampler, scale):
        """The x4 rule is specific to ``nearest+conv``: the official
        ``Upsample`` takes powers of two and 3, ``UpsampleOneStep`` any scale."""
        model = PFTSR(scale=scale, upsampler=upsampler, embed_dim=16,
                      num_blocks=[1], num_heads=2, window_size=8)
        x = np.random.default_rng(0).random((1, 16, 16, 3)).astype("float32")
        assert tuple(model(x, training=False).shape[1:3]) == (16 * scale, 16 * scale)

    @pytest.mark.parametrize("upsampler, scale", [
        ("pixelshuffle", 5), ("pixelshuffle", 6), ("bicubic", 2),
    ])
    def test_a_tail_the_official_code_does_not_build_is_refused(self, upsampler, scale):
        with pytest.raises(ValueError, match="upsampler"):
            PFTSR(scale=scale, upsampler=upsampler, embed_dim=16,
                  num_blocks=[1], num_heads=2, window_size=8)


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 10)
# ---------------------------------------------------------------------

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


class TestPFTSRGradientFlow:
    """Every trainable weight must be on the backward graph.

    Hundreds of trainable weights across progressive-focused attention layers and the
    ``include_top`` upsampler tail. This is the size at which an AGGREGATE
    gradient-norm assertion stops meaning anything -- one dead attention
    projection among hundreds of live tensors moves a global norm by nothing
    measurable. The oracle asserts per weight, keyed by ``Variable.path``.

    Skipped from 2026-10-10 to 2026-10-14 as an "isolation issue" with six
    ``expect_zero`` waivers. The measured cause: the waived paths named the
    head's common norm and FFN, which its ``call`` never applied, under the
    head's auto-suffixed name, which depends on how many heads the process had
    built. The head is now named ``enhancement_head`` and owns neither
    (plan-2026-10-14T062616-81938e81 step 6, D-020): no waiver. Since step 6.1
    there is no head at all: the model applies the official upsampler tail,
    and every one of its weights is covered here too.
    """

    def test_gradients_reach_every_trainable_weight(self):
        model = _model()
        x = _images()
        model(x, training=False)  # a subclassed model is unbuilt until first call

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == len(model.trainable_weights)
        assert len(report) > 0
        assert max(v for v in report.values() if v is not None) > 0.0


# ---------------------------------------------------------------------------
# DECISION plan-2026-08-19T163559-499b6f0e/D-118
#
# `create_pft_sr(scale, variant)` took ZERO `**kwargs` and hard-coded nine
# `PFTSR` arguments in its `return`, making `window_size`, `drop_path_rate`,
# `upsampler`, `norm_type`, `use_lepe` and the three dropout knobs unreachable
# through the only public factory this package exposes. Each hard-coded literal
# was byte-identical to the constructor's own default, which is why the three
# variants' parameter counts are UNCHANGED by the repair (measured before and
# after: light 1_025_027, base 2_180_283, large 4_120_739).
#
# Those three numbers were superseded on 2026-08-23 (D-463): `light` and `base`
# were corrected to the paper's own two released configs and `large` was renamed
# `repo_medium`. The derivation of each new number, MEASURED at scale=4 on a
# (1, 32, 32, 3) input:
#   light  52-wide, [2,4,6,6,6], 4 heads, mlp 1.0, window 32 -> 636_691 trainable
#   base   240-wide, [4,4,4,6,6,6], 6 heads, mlp 2.0, window 32 -> 18_656_163 trainable
#   repo_medium  unchanged in every field from the old `large` -> 2_744_483 trainable
#          (the rename moved no number)
#
# At the time those three numbers were first pinned, `count_params()` also
# included a non-trainable SW-MSA `attention_mask` buffer -- 12 (light) / 15
# (base) shifted blocks x one (1, 1024, 1024) array at window_size=32 -- that
# was registered as a weight, inflating the totals to 13_219_603 / 34_384_803
# (repo_medium's window_size=8 mask is tiny: 4_120_739 total). A later fix to
# the shared SW-MSA attention layer PFT-SR then built (the mask crashing every
# enclosing `Model`; that layer was deleted in step 7.1, D-216) made the mask a
# plain numpy array cached at build time, converted
# to a tensor only inside `call()`, so `count_params()` no longer counts it.
# The three pins below are the TRAINABLE totals from the derivation above --
# re-measured 2026-09-04 after that fix, and now identical to the trainable
# figures the original derivation already recorded.
#
# 2026-10-09: Added EnhancementHead for super-resolution (include_top=True).
# The EnhancementHead adds additional weights. Re-measured 2026-10-09 at scale=4:
#   light  52-wide, [2,4,6,6,6], 4 heads, mlp 1.0, window 32 -> 537_059 trainable
#   base   240-wide, [4,4,4,6,6,6], 6 heads, mlp 2.0, window 32 -> 16_526_883 trainable
#   repo_medium  unchanged in every field from the old `large` -> 2_508_323 trainable
#
# 2026-10-13: every count dropped, and the drop is accounted for rather than
# re-baselined. Commit 20225bf71 made `EnhancementHead.build` call
# `keras.layers.Layer.build` directly instead of `super().build`, so the head's
# `norm` and `ffn` are constructed but never built. `EnhancementHead.call` runs
# only `enhance_blocks` then `upsample`/`output_conv` and never applies either,
# so six weight tensors that no gradient ever reached are gone. Confirmed by
# diffing the full weight inventory at 3b281907a against HEAD: the ONLY
# difference is those six, per variant. The parameter deltas match the head
# widths exactly:
#   light       537_059 - 515_063 =  21_996 = ffn(52->208->52) + norm(52)
#   base      16_526_883 - 16_064_403 = 462_480 = ffn(240->960->240) + norm(240)
#   repo_medium 2_508_323 - 2_456_563 =  51_760 = ffn(80->320->80) + norm(80)
#
# 2026-10-15 (plan-2026-10-14T062616-81938e81 step 6.1, D-022): the
# EnhancementHead is gone and each variant ends in its OFFICIAL tail (light:
# 'pixelshuffledirect', base and repo_medium: 'pixelshuffle'). The body is
# untouched: the headless counts are the step-6 ones, 440_180 / 14_500_560 /
# 2_280_880. Each new total is that body plus the official tail's hand sum at
# scale 4 (c(i, o) = 9*i*o + o for a biased 3x3 conv, F = 64):
#   light       c(52, 16*3)                         =  22_512 -> 462_692
#   base        c(240, F) + 2*c(F, 4F) + c(F, 3)    = 435_459 -> 14_936_019
#   repo_medium c(80, F)  + 2*c(F, 4F) + c(F, 3)    = 343_299 -> 2_624_179
# Measured by the command below and by summing np.prod over model.weights
# (scratchpad step61/probes/pft_counts.py), both agreeing.
#
# 2026-10-20 (plan-2026-10-14T062616-81938e81 step 7, D-212): the body is the
# official one, and every count is the official hand sum of
# test_the_parameter_count_is_the_official_hand_sum.py, measured equal:
# patch-embed and final LayerNorm, a 3x3 conv per stage, per layer a
# relative-position bias table, a kernel-7 LePE and a ConvFFN with its kernel-7
# depthwise conv. Bodies: light 769_904, base 19_335_780, repo_medium 3_107_130;
# the tails are unchanged (22_512 / 435_459 / 343_299).
# ---------------------------------------------------------------------------
_PFT_VARIANT_PARAMS = {
    "light": 792_416,
    "base": 19_771_239,
    "repo_medium": 3_450_429,
}


@pytest.mark.parametrize("variant, expected_params", sorted(_PFT_VARIANT_PARAMS.items()))
def test_the_variant_parameter_counts_did_not_move(variant, expected_params):
    keras.utils.set_random_seed(0)
    model = create_pft_sr(4, variant)
    model(keras.ops.zeros((1, 32, 32, 3)))
    assert model.count_params() == expected_params


@pytest.mark.parametrize("key, value, default", [
    # 32, not the constructor's 8: `base` quotes the paper's PFT config, which is a
    # 32x32-window model, so MODEL_VARIANTS["base"] supplies window_size (D-463).
    ("window_size", 16, 32),
    ("upsampler", "pixelshuffledirect", "pixelshuffle"),
    ("convffn_kernel_size", 5, 7),
    ("num_topk", None, [1024] * 4 + [256] * 4 + [128] * 4 + [64] * 6 + [32] * 6 + [16] * 6),
    ("img_range", 255.0, 1.0),
    ("qkv_bias", False, True),
    ("drop_path_rate", 0.2, 0.0),
])
def test_every_constructor_knob_is_reachable_through_the_factory(key, value, default):
    keras.utils.set_random_seed(0)
    assert getattr(create_pft_sr(4, "base"), key) == default
    overridden = create_pft_sr(4, "base", **{key: value})
    assert getattr(overridden, key) == value, (
        f"create_pft_sr ignored the documented override {key}={value!r}"
    )


def test_a_factory_override_does_not_write_back_into_the_variant_table():
    """`.copy()` guards the LOCAL dict `config.update(kwargs)` mutates.

    Note the model-side alias this looks like is NOT the hazard: Keras 3
    rewraps any list assigned to a Model attribute as a fresh `TrackedList`,
    so `m.num_blocks.append(99)` leaves `MODEL_VARIANTS` untouched even on the
    unfixed source (measured). The table entry itself is what needed guarding.
    """
    before = list(PFTSR.MODEL_VARIANTS["base"]["num_blocks"])
    topk_before = list(PFTSR.MODEL_VARIANTS["base"]["num_topk"])
    create_pft_sr(4, "base", num_blocks=[1, 1], num_topk=[1024, 256])
    assert PFTSR.MODEL_VARIANTS["base"]["num_blocks"] == before
    assert PFTSR.MODEL_VARIANTS["base"]["num_topk"] == topk_before
    assert PFTSR.MODEL_VARIANTS["base"]["embed_dim"] == 240


def test_a_depth_override_without_its_own_schedule_is_refused():
    """``num_topk`` holds one entry per layer: the released variant's 30-entry
    schedule does not describe a 2-layer body, so the override must bring its
    own (or ``None``) instead of the model guessing which entries to drop."""
    with pytest.raises(ValueError, match="one positive integer per layer"):
        create_pft_sr(4, "base", num_blocks=[1, 1])
    assert create_pft_sr(4, "base", num_blocks=[1, 1], num_topk=None).num_topk is None


# ---------------------------------------------------------------------
# The official reconstruction tail (plan-2026-10-14T062616-81938e81 step 6.1)
# ---------------------------------------------------------------------

def _conv_params(i, o):
    """A biased 3x3 conv, ``nn.Conv2d(i, o, 3, 1, 1)``."""
    return 9 * i * o + o


#: CVL-UESTC/PFT-SR basicsr/archs/pft_arch.py, transcribed: the parameters of
#: each ``PFT`` reconstruction branch, ``num_feat = 64``, ``num_out_ch = 3``.
def _official_tail_params(upsampler, embed_dim, scale, out=3, feat=64):
    if upsampler == "pixelshuffledirect":    # UpsampleOneStep
        return _conv_params(embed_dim, scale * scale * out)
    if upsampler == "pixelshuffle":          # conv_before_upsample, Upsample, conv_last
        stages = (_conv_params(feat, 9 * feat) if scale == 3
                  else (scale.bit_length() - 1) * _conv_params(feat, 4 * feat))
        return _conv_params(embed_dim, feat) + stages + _conv_params(feat, out)
    # nearest+conv: conv_before_upsample, conv_up1, conv_up2, conv_hr, conv_last
    return _conv_params(embed_dim, feat) + 3 * _conv_params(feat, feat) + _conv_params(feat, out)


def _count(model):
    return sum(int(np.prod(w.shape)) for w in model.weights)


_TINY = dict(embed_dim=16, num_blocks=[1], num_heads=2, window_size=8)


class TestTheOfficialTail:
    """The default output is the official network's tail, selected by
    ``upsampler``; the headless model builds no tail and returns the features
    the tail reads. Before step 6.1 an ``EnhancementHead`` (three ConvBlocks
    and a stride-``scale`` transposed conv) replaced the tail, and the
    ``upsample``/``conv_last`` layers were built but applied in neither
    mode."""

    @pytest.mark.parametrize("upsampler, scale", [
        ("pixelshuffle", 2), ("pixelshuffle", 3), ("pixelshuffle", 4),
        ("pixelshuffledirect", 2), ("pixelshuffledirect", 3),
        ("nearest+conv", 4),
    ])
    def test_the_tail_is_the_official_hand_sum(self, upsampler, scale):
        x = np.zeros((1, 16, 16, 3), dtype="float32")
        headed = PFTSR(scale=scale, upsampler=upsampler, **_TINY)
        headless = PFTSR(scale=scale, upsampler=upsampler, include_top=False, **_TINY)
        headed(x)
        headless(x)
        assert _count(headed) - _count(headless) == _official_tail_params(
            upsampler, _TINY["embed_dim"], scale)

    @pytest.mark.parametrize("upsampler, scale", [
        ("pixelshuffle", 4), ("pixelshuffledirect", 2), ("nearest+conv", 4),
    ])
    def test_the_output_is_the_official_branch_applied_to_the_features(
            self, upsampler, scale, golden_reference_device):
        """Recompute the official ``PFT.forward`` branch by hand, with the
        slopes written out (torch ``nn.LeakyReLU()`` is 0.01; the
        ``nearest+conv`` stages use 0.2), on the headless model's features and
        the headed model's own tail weights. Exact: same ops, same device."""
        rng = np.random.default_rng(3)
        x = rng.random((1, 16, 16, 3)).astype("float32")
        with keras.device(golden_reference_device):
            keras.utils.set_random_seed(0)
            headed = PFTSR(scale=scale, upsampler=upsampler, **_TINY)
            y = keras.ops.convert_to_numpy(headed(x, training=False))
            headless = PFTSR(scale=scale, upsampler=upsampler, include_top=False, **_TINY)
            headless(x)
            for w in headless.weights:
                w.assign(next(v for v in headed.weights if v.path.split("/", 1)[1]
                              == w.path.split("/", 1)[1]))
            f = headless(x, training=False)
            layers = {layer.name: layer for layer in headed.tail_layers}

            def conv(name, t):
                return keras.ops.conv(t, layers[name].kernel, padding="same") + layers[name].bias

            def shuffle(t, r):
                b, h, w, c = t.shape
                t = keras.ops.reshape(t, (b, h, w, r, r, c // (r * r)))
                t = keras.ops.transpose(t, (0, 1, 3, 2, 4, 5))
                return keras.ops.reshape(t, (b, h * r, w * r, c // (r * r)))

            def lrelu(t, slope):
                return keras.ops.where(t >= 0, t, slope * t)

            # D-212: the official forward adds the RGB mean back after the tail.
            mean = np.array([0.4488, 0.4371, 0.4040], dtype="float32")
            if upsampler == "pixelshuffledirect":
                ref = shuffle(conv("upsample_conv", f), scale)
            elif upsampler == "pixelshuffle":
                t = lrelu(conv("conv_before_upsample", f), 0.01)
                for i in range(scale.bit_length() - 1):
                    t = shuffle(conv(f"upsample_conv_{i}", t), 2)
                ref = conv("conv_last", t)
            else:
                t = lrelu(conv("conv_before_upsample", f), 0.01)
                for i in (1, 2):
                    t = keras.ops.repeat(keras.ops.repeat(t, 2, axis=1), 2, axis=2)
                    t = lrelu(conv(f"conv_up{i}", t), 0.2)
                ref = conv("conv_last", lrelu(conv("conv_hr", t), 0.2))
            ref = keras.ops.convert_to_numpy(ref) + mean
        assert y.shape == (1, 16 * scale, 16 * scale, 3)
        assert float(np.abs(y).max()) > 1e-3
        np.testing.assert_array_equal(y, ref)

    def test_each_variant_carries_its_official_upsampler(self):
        assert PFTSR.MODEL_VARIANTS["light"]["upsampler"] == "pixelshuffledirect"
        assert PFTSR.MODEL_VARIANTS["base"]["upsampler"] == "pixelshuffle"
        assert create_pft_sr(scale=SCALE, variant="light").upsampler == "pixelshuffledirect"

    def test_the_headless_model_builds_no_tail_and_returns_the_features(self):
        model = create_pft_sr(scale=SCALE, variant="light", include_top=False)
        out = model(_images(1))
        assert model.tail_layers == []
        assert tuple(out.shape) == (1, SIZE, SIZE, PFTSR.MODEL_VARIANTS["light"]["embed_dim"])

    def test_the_deleted_head_knobs_are_refused(self):
        for knob in ("head_config_overrides", "head_config_preset"):
            with pytest.raises(ValueError, match=knob):
                create_pft_sr(scale=SCALE, variant="light", **{knob: {}})

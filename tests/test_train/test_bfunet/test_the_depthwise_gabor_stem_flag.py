"""Guard: `--gabor-filters-per-channel` reaches the model, and is ConvUNeXt-only.

`# DECISION standalone-2026-09-07-depthwise-gabor-stem/D-002`. The model arm is pinned by
`tests/test_models/test_bias_free_denoisers/test_the_depthwise_gabor_stem_arm.py`; this
module pins the TRAINER wiring, which is where this repo's recorded defect class lives -- a
flag that parses, lands in `config.json`, and reaches no layer. Four things are pinned:

  1. the flag survives argv -> `TrainingConfig` -> `build_model` -> the built graph, in
     BOTH the `--smoke` and the full config blocks (two hand-maintained construction
     sites, so a value forwarded in one and forgotten in the other is the likely bug);
  2. it is MUTUALLY EXCLUSIVE with `--gabor-filters` at parse time -- the two are
     different quantities (Conv2D output count vs depthwise multiplier) and accepting
     both would leave whichever lost silently inert;
  3. it is ConvUNeXt-ONLY: the U-Net and BFCNN trainers must not grow the flag, because
     `create_bfunet_denoiser` / `create_bfcnn_denoiser` have no such kwarg and a shared
     flag there would be inert by construction; and
  4. `--freeze-gabor-stem` still freezes -- stem KIND and stem FREEZING are independent
     axes and must compose.

Run:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_bfunet/test_the_depthwise_gabor_stem_flag.py -q
"""

import keras
import numpy as np
import pytest

from train.bfunet.common import validate_gabor_stem_channels
import train.bfunet.train_bfcnn_denoiser as bfcnn_trainer
import train.bfunet.train_convunext_denoiser as convunext_trainer
import train.bfunet.train_unet_denoiser as unet_trainer


# A small but REAL topology. `initial_filters` is NOT channels*PER_CHANNEL, so the
# projection is load-bearing and the no-projection rule is exercised separately.
CHANNELS = 3
PER_CHANNEL = 4
STEM_WIDTH = CHANNELS * PER_CHANNEL  # 12
SMALL = dict(
    variant="base", patch_size=32, channels=CHANNELS, initial_filters=16,
    filter_multiplier=1.0, depth=2, blocks_per_level=1,
    block_normalization="batchnorm", gabor_kernel_size=5,
)


def _config(argv):
    """Parse `argv` through the ConvUNeXt trainer's own parser."""
    import sys
    saved = sys.argv
    sys.argv = ["train_convunext_denoiser"] + argv
    try:
        return convunext_trainer.parse_arguments()
    finally:
        sys.argv = saved


def _build(**overrides):
    return convunext_trainer.build_model(
        convunext_trainer.TrainingConfig(**{**SMALL, **overrides})
    )


class TestTheFlagReachesTheGraph:
    def test_argv_lands_on_the_config(self):
        args = _config(["--gabor-filters-per-channel", str(PER_CHANNEL)])
        assert args.gabor_filters_per_channel == PER_CHANNEL

    def test_the_default_is_none(self):
        assert _config([]).gabor_filters_per_channel is None
        assert convunext_trainer.TrainingConfig(**SMALL).gabor_filters_per_channel is None

    def test_build_model_forwards_it(self):
        stem = _build(gabor_filters_per_channel=PER_CHANNEL).get_layer("gabor_stem")
        assert type(stem) is keras.layers.DepthwiseConv2D
        assert stem.output.shape[-1] == STEM_WIDTH

    def test_the_default_config_still_builds_the_cross_channel_stem(self):
        stem = _build(gabor_filters=16).get_layer("gabor_stem")
        assert type(stem) is keras.layers.Conv2D
        assert stem.output.shape[-1] == 16

    @pytest.mark.parametrize("smoke", [True, False], ids=["smoke", "full"])
    def test_both_main_config_blocks_forward_it(self, smoke, monkeypatch):
        """`main()` builds a `TrainingConfig` at TWO sites (--smoke and full). A value
        forwarded at one and forgotten at the other is invisible until someone runs the
        other mode, so both are driven through `main()` with training stubbed out."""
        argv = ["--gabor-filters-per-channel", str(PER_CHANNEL)]
        if smoke:
            argv.append("--smoke")
        seen = {}

        monkeypatch.setattr(convunext_trainer, "setup_gpu", lambda gpu_id=None: None)
        monkeypatch.setattr(
            convunext_trainer, "train",
            lambda config: seen.setdefault("config", config),
        )
        import sys
        monkeypatch.setattr(sys, "argv", ["train_convunext_denoiser"] + argv)
        convunext_trainer.main()

        assert seen["config"].gabor_filters_per_channel == PER_CHANNEL, (
            f"the {'--smoke' if smoke else 'full'} config block dropped "
            "gabor_filters_per_channel"
        )


class TestTheTwoCountsAreMutuallyExclusive:
    def test_passing_both_is_a_parse_error(self):
        with pytest.raises(SystemExit):
            _config(["--gabor-filters-per-channel", "4", "--gabor-filters", "16"])

    def test_either_alone_is_accepted(self):
        assert _config(["--gabor-filters-per-channel", "4"]).gabor_filters_per_channel == 4
        assert _config(["--gabor-filters", "16"]).gabor_filters == 16

    def test_an_explicit_default_gabor_filters_is_not_a_conflict(self):
        """The guard asks "was a DIFFERENT value requested?", not "was the flag typed?".
        `--gabor-filters 32` requests the value already in place and changes nothing, so
        refusing it would be a false conflict."""
        args = _config(["--gabor-filters-per-channel", "4", "--gabor-filters", "32"])
        assert args.gabor_filters_per_channel == 4 and args.gabor_filters == 32

    def test_with_no_gabor_stem_is_a_parse_error(self):
        with pytest.raises(SystemExit):
            _config(["--gabor-filters-per-channel", "4", "--no-gabor-stem"])

    def test_the_config_refuses_the_stemless_combination_too(self):
        """Belt-and-suspenders for a programmatically built config, which never sees the
        parser."""
        with pytest.raises(ValueError, match="use_gabor_stem=False"):
            convunext_trainer.TrainingConfig(
                **SMALL, gabor_filters_per_channel=4, use_gabor_stem=False
            )

    @pytest.mark.parametrize("bad", [0, -3])
    def test_the_config_refuses_a_non_positive_multiplier(self, bad):
        with pytest.raises(ValueError, match="must be >= 1"):
            convunext_trainer.TrainingConfig(**SMALL, gabor_filters_per_channel=bad)


class TestTheNoProjectionWidthRule:
    """Two arms, two rules. The depthwise one is `channels * per_channel ==
    initial_filters` -- textually the pre-2026-09-06 rule, back for this arm ONLY."""

    def test_the_predicate_accepts_a_matching_depthwise_width(self):
        assert validate_gabor_stem_channels(
            use_gabor_stem=True, gabor_stem_projection=False,
            gabor_filters=8, initial_filters=STEM_WIDTH,
            gabor_filters_per_channel=PER_CHANNEL, channels=CHANNELS) is None

    def test_the_predicate_rejects_a_mismatching_depthwise_width(self):
        with pytest.raises(ValueError, match="gabor_filters_per_channel"):
            validate_gabor_stem_channels(
                use_gabor_stem=True, gabor_stem_projection=False,
                gabor_filters=8, initial_filters=STEM_WIDTH + 4,
                gabor_filters_per_channel=PER_CHANNEL, channels=CHANNELS)

    def test_the_depthwise_arm_ignores_gabor_filters(self):
        """`gabor_filters == initial_filters` satisfies the OTHER arm's rule. On the
        depthwise arm it must not rescue a wrong width."""
        with pytest.raises(ValueError, match="gabor_filters_per_channel"):
            validate_gabor_stem_channels(
                use_gabor_stem=True, gabor_stem_projection=False,
                gabor_filters=99, initial_filters=99,
                gabor_filters_per_channel=PER_CHANNEL, channels=CHANNELS)

    def test_channels_is_required_not_guessed(self):
        """A defaulted channel count would compute a width the model never builds, and
        the check would pass or fail for reasons unrelated to the configuration."""
        with pytest.raises(ValueError, match="channels is required"):
            validate_gabor_stem_channels(
                use_gabor_stem=True, gabor_stem_projection=False,
                gabor_filters=8, initial_filters=STEM_WIDTH,
                gabor_filters_per_channel=PER_CHANNEL)

    def test_the_cross_channel_arm_is_untouched(self):
        """The three-argument call is the whole pre-existing contract; the depthwise
        parameters default to None and must change nothing about it."""
        assert validate_gabor_stem_channels(
            use_gabor_stem=True, gabor_stem_projection=False,
            gabor_filters=32, initial_filters=32) is None
        with pytest.raises(ValueError, match="no longer applies"):
            validate_gabor_stem_channels(
                use_gabor_stem=True, gabor_stem_projection=False,
                gabor_filters=8, initial_filters=24)

    def test_the_trainer_routes_through_the_predicate(self):
        """`build_model` must reject the bad width BEFORE the factory does; both raise,
        so the message is what distinguishes them."""
        with pytest.raises(ValueError, match="--no-gabor-projection"):
            _build(initial_filters=STEM_WIDTH + 4,
                   gabor_filters_per_channel=PER_CHANNEL,
                   gabor_stem_projection=False)

    def test_a_matching_width_builds(self):
        model = _build(initial_filters=STEM_WIDTH,
                       gabor_filters_per_channel=PER_CHANNEL,
                       gabor_stem_projection=False)
        assert model.get_layer("gabor_stem").output.shape[-1] == STEM_WIDTH


class TestItIsConvUNeXtOnly:
    """`create_bfunet_denoiser` and `create_bfcnn_denoiser` have no such kwarg, so the
    flag must not appear on those trainers -- there it could only be inert."""

    @pytest.mark.parametrize("mod", [unet_trainer, bfcnn_trainer],
                             ids=["train_unet_denoiser", "train_bfcnn_denoiser"])
    def test_the_other_trainers_reject_the_flag(self, mod, monkeypatch):
        import sys
        monkeypatch.setattr(
            sys, "argv", [mod.__name__, "--gabor-filters-per-channel", "4"])
        with pytest.raises(SystemExit):
            mod.parse_arguments()

    @pytest.mark.parametrize("mod", [unet_trainer, bfcnn_trainer],
                             ids=["train_unet_denoiser", "train_bfcnn_denoiser"])
    def test_the_other_configs_have_no_such_field(self, mod):
        import dataclasses
        cfg_cls = getattr(mod, "TrainingConfig", None) or mod.BFCNNTrainingConfig
        names = {f.name for f in dataclasses.fields(cfg_cls)}
        assert "gabor_filters_per_channel" not in names, (
            f"{mod.__name__}'s config grew a field its model factory cannot consume"
        )

    def test_the_shared_cli_did_not_grow_it(self):
        """It must be registered by the ConvUNeXt parser, NOT by the shared
        `add_common_arguments` -- which is what would put it on all three trainers."""
        import argparse
        from train.bfunet.common import add_common_arguments
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        options = {a for action in parser._actions for a in action.option_strings}
        assert "--gabor-filters-per-channel" not in options
        assert "--gabor-filters" in options  # the shared one is still shared


class TestFreezingComposesWithTheStemKind:
    """Stem KIND and stem FREEZING are independent axes."""

    def test_freeze_gabor_stem_freezes_the_depthwise_bank(self):
        model = _build(gabor_filters_per_channel=PER_CHANNEL,
                       trainable_gabor_stem=False)
        stem = model.get_layer("gabor_stem")
        assert type(stem) is keras.layers.DepthwiseConv2D
        assert stem.trainable is False
        assert stem.kernel.path not in {w.path for w in model.trainable_weights}

    def test_the_depthwise_bank_is_trainable_by_default(self):
        """The factory overrides `create_gabor_depthwise_conv2d`'s own frozen default,
        so that `--freeze-gabor-stem` is the ONE knob that freezes a stem. Without this
        the flag would be a no-op on this arm and the stem frozen for a second,
        invisible reason."""
        model = _build(gabor_filters_per_channel=PER_CHANNEL)
        stem = model.get_layer("gabor_stem")
        assert stem.trainable is True
        assert stem.kernel.path in {w.path for w in model.trainable_weights}

    def test_freezing_leaves_homogeneity_intact(self):
        """Homogeneity comes from `use_bias=False`, not from frozen-ness or stem kind."""
        y = np.random.RandomState(0).rand(1, 32, 32, CHANNELS).astype("float32")
        for frozen in (True, False):
            model = _build(gabor_filters_per_channel=PER_CHANNEL,
                           trainable_gabor_stem=not frozen)
            base = np.asarray(model(y, training=False))
            got = np.asarray(model(np.float32(3.0) * y, training=False))
            want = np.float32(3.0) * base
            rel = float(np.max(np.abs(got - want))) / max(float(np.max(np.abs(want))), 1e-30)
            assert rel < 1e-5, f"frozen={frozen}: D(a*x) != a*D(x) ({rel:.3e})"

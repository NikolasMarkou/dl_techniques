"""Guard: `--freeze-gabor-stem` actually freezes, and can never become a silent no-op.

The model factories HARDCODE `trainable=True` on the Gabor stem
(`models/vision/convunext/model.py`, `.../bias_free_denoisers/bfunet.py`) because a
Gabor-initialized-then-refined stem is the current default architecture. There is no
factory kwarg to thread, so the only window in which a caller can freeze it is after the
graph is built and before it is compiled. That makes the flag structurally fragile in two
directions, and this module pins both:

  1. it must actually flip `trainable` and move the stem's parameters out of
     `trainable_weights` -- a flag that parses but does not reach the optimizer is the
     defect `variance_probe.py` already documents in its free-Gabor comment; and
  2. it must RAISE, never no-op, when there is no stem to freeze (`--no-gabor-stem`, or
     the BFCNN trainer whose config pins `use_gabor_stem=False`).

It also pins the two properties a reader is most likely to assume wrongly: freezing does
NOT restore the paper's depthwise bank (the layer stays a cross-channel `Conv2D`), and it
does NOT touch degree-1 homogeneity, which comes from the stem's `use_bias=False`.

Freezing and stem KIND are independent axes. A depthwise bank IS reachable, but through
ConvUNeXt's separate `--gabor-filters-per-channel`
(`standalone-2026-09-07-depthwise-gabor-stem/D-001`), never through this flag. Every
build below leaves that argument at its `None` default, so the assertions here describe
the default stem; their composition is pinned in
`tests/test_train/test_bfunet/test_the_depthwise_gabor_stem_flag.py`.

Run:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_bfunet/test_the_frozen_gabor_stem_flag.py -q
"""

import dataclasses

import keras
import numpy as np
import pytest

from train.bfunet.common import freeze_gabor_stem_if_requested, add_common_arguments
import train.bfunet.train_unet_denoiser as unet_trainer
import train.bfunet.train_convunext_denoiser as convunext_trainer
import train.bfunet.train_bfcnn_denoiser as bfcnn_trainer


# A small but REAL topology, shared by both trainers so the parametrization compares
# like with like. depth=3 is the floor: the U-Net baseline rejects depth < 3, so a
# smaller build would silently test only the ConvUNeXt arm.
SMALL = dict(
    variant="base", patch_size=64, initial_filters=16, filter_multiplier=1.0,
    depth=3, blocks_per_level=1, gabor_filters=16, block_normalization="batchnorm",
)

TRAINERS = [
    pytest.param(convunext_trainer, id="convunext"),
    pytest.param(unet_trainer, id="unet"),
]


def _build(trainer, **overrides):
    return trainer.build_model(trainer.TrainingConfig(**{**SMALL, **overrides}))


def _stem_params(model):
    return int(np.prod(model.get_layer("gabor_stem").kernel.shape))


def _trainable_total(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


class TestTheFlagReachesTheOptimizer:
    @pytest.mark.parametrize("trainer", TRAINERS)
    def test_freezing_moves_the_stem_out_of_trainable_weights(self, trainer):
        free = _build(trainer)
        frozen = _build(trainer, trainable_gabor_stem=False)

        assert free.get_layer("gabor_stem").trainable is True
        assert frozen.get_layer("gabor_stem").trainable is False

        stem = _stem_params(frozen)
        assert stem > 0, "a stem with no parameters would make this test vacuous"
        # The whole point: the optimizer never sees the kernel.
        assert not any(
            w is frozen.get_layer("gabor_stem").kernel for w in frozen.trainable_weights
        )
        # Total is conserved; the parameters MOVE, they are not dropped.
        assert frozen.count_params() == free.count_params()
        assert _trainable_total(free) - _trainable_total(frozen) == stem

    @pytest.mark.parametrize("trainer", TRAINERS)
    def test_the_default_is_unchanged(self, trainer):
        """The default path must be bit-identical, not merely 'also trainable'."""
        keras.utils.set_random_seed(42)
        implicit = _build(trainer)
        keras.utils.set_random_seed(42)
        explicit = _build(trainer, trainable_gabor_stem=True)

        assert len(implicit.weights) == len(explicit.weights)
        assert all(
            np.array_equal(np.array(a), np.array(b))
            for a, b in zip(implicit.weights, explicit.weights)
        )
        assert len(implicit.trainable_weights) == len(explicit.trainable_weights)

    @pytest.mark.parametrize("trainer", TRAINERS)
    def test_the_projection_stays_trainable(self, trainer):
        """Freezing the bank must leave the 1x1 projection as the learned degree of
        freedom on top of it -- freezing both would leave nothing to learn before L0."""
        frozen = _build(trainer, trainable_gabor_stem=False)
        proj = frozen.get_layer("gabor_stem_projection")
        assert proj.trainable is True
        assert any(w is proj.kernel for w in frozen.trainable_weights)


class TestItCannotBecomeASilentNoOp:
    @pytest.mark.parametrize("trainer", TRAINERS)
    def test_freezing_without_a_stem_raises(self, trainer):
        with pytest.raises(ValueError, match="no 'gabor_stem' layer"):
            _build(trainer, use_gabor_stem=False, trainable_gabor_stem=False)

    def test_bfcnn_rejects_the_flag_it_can_never_honour(self):
        """BFCNN pins use_gabor_stem=False, so the shared flag must be refused at config
        time rather than parsed and ignored."""
        with pytest.raises(ValueError, match="not applicable to the BFCNN trainer"):
            bfcnn_trainer.BFCNNTrainingConfig(trainable_gabor_stem=False)
        # The default must still construct, or the guard above is over-broad.
        assert bfcnn_trainer.BFCNNTrainingConfig().trainable_gabor_stem is True

    def test_the_helper_is_a_no_op_when_the_stem_should_stay_trainable(self):
        """Guards the early return: a model with no stem must pass through untouched
        when no freeze was asked for, so the raise above is about the FLAG, not the
        topology."""
        cfg = convunext_trainer.TrainingConfig(**SMALL, use_gabor_stem=False)
        model = _build(convunext_trainer, use_gabor_stem=False)
        assert freeze_gabor_stem_if_requested(model, cfg) is model


class TestWhatFreezingDoesAndDoesNotChange:
    def test_the_frozen_stem_is_still_the_cross_channel_conv_not_the_papers_bank(self):
        """A reader may assume 'frozen Gabor stem' means the paper's construction. It
        does not: the paper's was a DepthwiseConv2D with 22 filters PER INPUT CHANNEL."""
        frozen = _build(convunext_trainer, trainable_gabor_stem=False)
        stem = frozen.get_layer("gabor_stem")
        assert isinstance(stem, keras.layers.Conv2D)
        assert not isinstance(stem, keras.layers.DepthwiseConv2D)
        # Cross-channel: the kernel reads all input channels at once.
        kh, kw, in_ch, out_ch = stem.kernel.shape
        assert int(in_ch) == 3 and int(out_ch) == SMALL["gabor_filters"]

    def test_freezing_leaves_degree_one_homogeneity_intact(self):
        """Homogeneity comes from use_bias=False, never from frozen-ness. Both builds
        must satisfy D(a*y) == a*D(y) to float32 round-off."""
        y = np.random.RandomState(0).rand(
            1, SMALL["patch_size"], SMALL["patch_size"], 3
        ).astype("float32")
        for trainable in (True, False):
            keras.utils.set_random_seed(42)
            model = _build(convunext_trainer, trainable_gabor_stem=trainable)
            base = np.array(model(y, training=False))
            for alpha in (0.25, 2.0, 8.0):
                scaled = np.array(model(alpha * y, training=False))
                rel = np.linalg.norm(scaled - alpha * base) / (
                    np.linalg.norm(alpha * base) + 1e-12
                )
                assert rel < 1e-2, (
                    f"homogeneity broken at alpha={alpha} with "
                    f"trainable_gabor_stem={trainable}: rel error {rel:.3e}"
                )
        # The stem must be bias-free in BOTH states, which is the actual mechanism.
        for trainable in (True, False):
            stem = _build(convunext_trainer, trainable_gabor_stem=trainable).get_layer(
                "gabor_stem"
            )
            assert stem.use_bias is False


class TestTheFlagIsWiredEndToEnd:
    def test_the_cli_exposes_it_and_it_maps_to_the_config_field(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_common_arguments(parser)

        default = parser.parse_args([])
        assert default.freeze_gabor_stem is False
        assert parser.parse_args(["--freeze-gabor-stem"]).freeze_gabor_stem is True

    @pytest.mark.parametrize("trainer", TRAINERS)
    def test_the_config_field_exists_with_a_backwards_compatible_default(self, trainer):
        fields = {f.name: f for f in dataclasses.fields(trainer.TrainingConfig)}
        assert "trainable_gabor_stem" in fields
        assert fields["trainable_gabor_stem"].default is True

    @pytest.mark.parametrize("trainer", TRAINERS)
    def test_the_trainers_route_through_the_shared_helper(self, trainer):
        """A shared helper nobody calls looks like coverage and provides none."""
        import inspect

        src = inspect.getsource(trainer.build_model)
        assert "freeze_gabor_stem_if_requested" in src

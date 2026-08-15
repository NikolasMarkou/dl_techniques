"""RED proof for C-23: ``use_side_loss=True`` must be trainable.

Before this suite, ``create_darkir_model(..., use_side_loss=True)`` built a
two-output model whose second output lives at bottleneck resolution
(``H / 2**len(enc_blk_nums)``), while the only trainer in the repo hardcoded
the flag to ``False`` and compiled a single full-resolution loss. Turning the
flag on therefore produced a model that BUILT cleanly and died at the first
``fit()`` step. These tests pin both halves: the failure the old shape produced,
and the trainer wiring that now makes the flag mean something.

CPU-only by construction (tiny model, 2 samples, 1 step).
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.darkir.model import create_darkir_model
from dl_techniques.losses.image_restoration_loss import DarkIRCompositeLoss
from dl_techniques.metrics.psnr_metric import PsnrMetric

from train.darkir.train_darkir import (
    attach_side_targets,
    create_darkir_config,
    side_output_downsample_factor,
)

SPATIAL = 32
ENC = [1, 2]
DEC = [1, 1]


def _model(use_side_loss: bool):
    return create_darkir_model(
        img_channels=3, width=8, middle_blk_num_enc=1, middle_blk_num_dec=1,
        enc_blk_nums=ENC, dec_blk_nums=DEC, dilations=[1],
        use_side_loss=use_side_loss,
    )


def _pairs(n=2):
    rng = np.random.default_rng(0)
    x = rng.random((n, SPATIAL, SPATIAL, 3)).astype("float32")
    y = rng.random((n, SPATIAL, SPATIAL, 3)).astype("float32")
    return x, y


def _charbonnier_only():
    return DarkIRCompositeLoss(
        charbonnier_weight=1.0, ssim_weight=0.0, perceptual_weight=0.0,
    )


class TestTrainerDerivesTheSideTargetFromTheModelConfig:
    def test_downsample_factor_tracks_the_number_of_encoder_stages(self):
        assert side_output_downsample_factor({'enc_blk_nums': [1, 2, 3]}) == 8
        assert side_output_downsample_factor({'enc_blk_nums': ENC}) == 4

    def test_config_propagates_the_flag(self):
        assert create_darkir_config('medium')['use_side_loss'] is False
        assert create_darkir_config(
            'medium', use_side_loss=True)['use_side_loss'] is True

    def test_attached_side_target_matches_the_side_output_shape(self):
        x, y = _pairs()
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        factor = side_output_downsample_factor({'enc_blk_nums': ENC})
        ds = attach_side_targets(ds, factor)

        bx, (by, by_small) = next(iter(ds))
        model = _model(use_side_loss=True)
        main, side = model(bx, training=False)
        assert tuple(by.shape[1:]) == tuple(main.shape[1:])
        assert tuple(by_small.shape[1:]) == tuple(side.shape[1:])

        # Area downsampling is the box filter: the mean is preserved.
        np.testing.assert_allclose(
            np.asarray(by_small).mean(axis=(1, 2)),
            np.asarray(by).mean(axis=(1, 2)),
            atol=1e-5,
        )


class TestSideLossTrainability:
    def test_full_resolution_target_on_the_side_head_dies_at_step_one(self):
        """The pre-fix behaviour, pinned: builds fine, then fails to train.

        The target here is well-formed as a structure — one full-resolution
        image per output — so the only thing that can fail is the RESOLUTION
        of the side head. That distinction matters: a single `y` against two
        outputs also raises, but for a structure reason that would still raise
        if the side head were full-resolution."""
        x, y = _pairs()
        model = _model(use_side_loss=True)
        model.compile(
            optimizer="adam",
            loss=[_charbonnier_only(), _charbonnier_only()],
        )
        with pytest.raises(Exception):
            model.fit(x, (y, y), epochs=1, batch_size=2, verbose=0)

    def test_downsampled_side_target_trains(self):
        x, y = _pairs()
        factor = side_output_downsample_factor({'enc_blk_nums': ENC})
        ds = attach_side_targets(
            tf.data.Dataset.from_tensor_slices((x, y)).batch(2), factor,
        )

        model = _model(use_side_loss=True)
        model.compile(
            optimizer="adam",
            loss=[_charbonnier_only(), _charbonnier_only()],
            loss_weights=[1.0, 0.2],
        )
        before = [w.copy() for w in model.get_weights()]
        history = model.fit(ds, epochs=1, verbose=0)

        assert np.isfinite(history.history['loss'][0])
        after = model.get_weights()
        moved = max(
            float(np.max(np.abs(a - b))) for a, b in zip(after, before)
        )
        assert moved > 0.0, "no weight moved: the fit() step was a no-op"

    def test_the_monitored_metric_name_exists_under_two_outputs(self):
        """Keras prefixes per-output metrics with the output name, so the
        trainer's `val_psnr` monitor silently stops matching anything the
        moment a second output appears — EarlyStopping and ModelCheckpoint
        then track nothing. Pin the name the trainer derives."""
        x, y = _pairs()
        factor = side_output_downsample_factor({'enc_blk_nums': ENC})
        ds = attach_side_targets(
            tf.data.Dataset.from_tensor_slices((x, y)).batch(2), factor,
        )
        model = _model(use_side_loss=True)
        model.compile(
            optimizer="adam",
            loss=[_charbonnier_only(), _charbonnier_only()],
            loss_weights=[1.0, 0.2],
            metrics=[[PsnrMetric(max_val=1.0, name='psnr')], []],
        )
        history = model.fit(ds, validation_data=ds, epochs=1, verbose=0)
        monitor = f"val_{model.output_names[0]}_psnr"
        assert monitor in history.history, sorted(history.history)
        assert 'val_psnr' not in history.history

    def test_side_gradient_reaches_the_encoder(self):
        """Anti-vacuity for the step above: the SIDE loss alone must train the
        shared trunk, otherwise a green fit() would prove only that the main
        loss still works."""
        x, y = _pairs()
        factor = side_output_downsample_factor({'enc_blk_nums': ENC})
        ds = attach_side_targets(
            tf.data.Dataset.from_tensor_slices((x, y)).batch(2), factor,
        )
        model = _model(use_side_loss=True)
        model.compile(
            optimizer="sgd",
            loss=[_charbonnier_only(), _charbonnier_only()],
            loss_weights=[0.0, 1.0],
        )
        names = [v.path for v in model.trainable_variables]
        before = [np.array(v) for v in model.trainable_variables]
        model.fit(ds, epochs=1, verbose=0)
        after = [np.array(v) for v in model.trainable_variables]

        moved = {
            n for n, a, b in zip(names, after, before)
            if float(np.max(np.abs(a - b))) > 0.0
        }
        assert any("encoder" in n or "intro" in n for n in moved), (
            f"the side loss moved nothing in the encoder; moved={sorted(moved)}"
        )

"""The horizon loss must count frames from the BATCH, not from `cfg.num_frames`.

F-52 of the 2026-08-18 deep review, fixed under
``plan-2026-08-18T140459-7991552f/D-041``.

`VideoJEPA` is `T`-generic everywhere except its own loss arithmetic:
`encode_frames`, `predictor` and `stream_step` all accept any `T`, and
`VideoJEPAConfig.__post_init__` only constrains ``max(predict_horizons) <
num_frames`` -- it says nothing about the runtime `T`. The loss read
``cfg.num_frames`` in two places, which broke in two different ways.

MEASURED at ``img_size=16, patch_size=8, embed_dim=16, num_frames=8,
predict_horizons=(1, 4)``:

* ``T=2``, masking OFF -> ``pred[:, :-4]`` is empty, ``ops.mean`` of it is
  **nan**, and that nan went through ``add_loss`` into ``train_step``. Measured
  loss vector pre-fix: ``[1.577979, nan, 0.041659]``.
* ``T=2``, masking ON -> the SAME empty slice is quieter and no better:
  ``sum(empty * w) / denom == 0.0``, so the horizon contributed exactly nothing
  while looking like a healthy number. Measured pre-fix: ``[0.196255, 0.0,
  0.225782, 0.026291]``.
* ``T=4``, masking ON -> ``denom`` counted ``cfg.num_frames - h = 7`` positions
  for a tensor holding ``T - h = 3``, scaling the horizon loss by ``3/7``.

Post-fix the same three cases give finite losses, drop the unreachable horizon
entirely (rather than adding a 0.0), and normalize by ``T - h``.

The denominator test below does NOT compare against a monkeypatched legacy body:
it recomputes the expected masked MSE from the model's own ``pred``,
``z_target`` and mask with an independent NumPy reduction, and additionally
asserts that the ``cfg.num_frames``-based denominator would have produced a
DIFFERENT number. That is the RED proof.
"""

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.models.video_jepa.model import create_video_jepa


IMG, PATCH, DIM = 16, 8, 16
NUM_FRAMES, HORIZONS = 8, (1, 4)
BATCH = 2


def _model(**overrides):
    keras.utils.set_random_seed(0)
    return create_video_jepa(
        img_size=IMG,
        patch_size=PATCH,
        embed_dim=DIM,
        num_frames=NUM_FRAMES,
        history_size_k=NUM_FRAMES,
        predictor_depth=1,
        encoder_clifford_depth=1,
        sigreg_num_proj=4,
        predict_horizons=HORIZONS,
        **overrides,
    )


def _pixels(t, seed=0):
    return ops.convert_to_tensor(
        np.random.RandomState(seed).randn(BATCH, t, IMG, IMG, 3).astype("float32")
    )


def _losses(model, pixels):
    model({"pixels": pixels}, training=True)
    return [float(ops.convert_to_numpy(v)) for v in model.losses]


class TestEmptySliceMechanism:
    """The premise: `pred[:, :-h]` at `T <= h` really is empty, and empty
    reductions really do produce nan / 0.0."""

    def test_empty_slice_mean_is_nan(self):
        x = ops.convert_to_tensor(np.zeros((2, 2, 3), dtype="float32"))
        empty = x[:, :-4]
        assert tuple(empty.shape)[1] == 0
        assert np.isnan(float(ops.convert_to_numpy(ops.mean(empty))))

    def test_empty_slice_sum_is_zero(self):
        x = ops.convert_to_tensor(np.zeros((2, 2, 3), dtype="float32"))
        assert float(ops.convert_to_numpy(ops.sum(x[:, :-4]))) == 0.0


class TestShortClipsDoNotPoisonTheLoss:
    def test_no_nan_with_masking_off(self):
        model = _model(mask_prediction_enabled=False)
        for t in (NUM_FRAMES, 4, 2):
            values = _losses(model, _pixels(t))
            assert all(np.isfinite(v) for v in values), f"T={t}: {values}"

    def test_no_nan_with_masking_on(self):
        model = _model(mask_prediction_enabled=True)
        for t in (NUM_FRAMES, 4, 2):
            values = _losses(model, _pixels(t))
            assert all(np.isfinite(v) for v in values), f"T={t}: {values}"

    def test_unreachable_horizon_is_dropped_not_zeroed(self):
        """At ``T=2`` the ``h=4`` horizon has no causal pair at all.

        It must not appear as a 0.0 loss term (which is what the pre-fix
        masking-on path produced, and which trains the model on nothing while
        looking healthy).
        """
        model = _model(mask_prediction_enabled=False)
        long_losses = _losses(model, _pixels(NUM_FRAMES))
        short_losses = _losses(model, _pixels(2))

        assert len(long_losses) == len(short_losses) + 1, (
            f"expected exactly one fewer loss term at T=2, got "
            f"{long_losses} vs {short_losses}"
        )
        assert not any(v == 0.0 for v in short_losses)

    def test_single_frame_clip_produces_no_horizon_loss(self):
        model = _model(mask_prediction_enabled=False)
        values = _losses(model, _pixels(1))
        assert all(np.isfinite(v) for v in values)
        # Only SIGReg survives at T=1.
        assert len(values) == 1


class TestDenominatorCountsBatchFrames:
    """`denom` normalizes by ``T - h``, independently recomputed."""

    @staticmethod
    def _fixed_mask(model, batch):
        rng = np.random.RandomState(7)
        side = model.config.patches_per_side
        n_masked = model.mask_gen.num_masked
        mask = np.zeros((batch, side * side), dtype="float32")
        for b in range(batch):
            mask[b, rng.permutation(side * side)[:n_masked]] = 1.0
        mask = mask.reshape(batch, side, side)
        tensor = ops.convert_to_tensor(mask)
        model.mask_gen.call = lambda batch_size, training=None: tensor
        return mask

    def test_horizon_loss_matches_an_independent_reduction(self):
        t, h_idx, h = 4, 0, HORIZONS[0]
        model = _model(mask_prediction_enabled=True)
        pixels = _pixels(t)
        model({"pixels": pixels}, training=True)  # build
        mask = self._fixed_mask(model, BATCH)

        pred = ops.convert_to_numpy(model({"pixels": pixels}, training=True))
        h_loss = float(ops.convert_to_numpy(model.losses[h_idx]))
        h_loss /= model.config.lambda_next_frame

        z_target = ops.convert_to_numpy(model.encode_frames_target(pixels))
        pred_ctx = ops.convert_to_numpy(
            model.pred_heads[h_idx](ops.convert_to_tensor(pred[:, :-h]))
        )
        sq = (pred_ctx - z_target[:, h:]) ** 2
        w = (1.0 - mask)[:, None, :, :, None]

        cfg = model.config
        unmasked = cfg.num_patches - model.mask_gen.num_masked
        numer = float((sq * w).sum())

        denom_batch = float(max(1, unmasked * (t - h) * cfg.embed_dim))
        denom_config = float(max(1, unmasked * (cfg.num_frames - h) * cfg.embed_dim))

        expected_batch = numer / (BATCH * denom_batch)
        expected_config = numer / (BATCH * denom_config)

        # The two candidate denominators are NOT the same number -- if they
        # were, this test could not tell the fix from the defect.
        ratio = expected_batch / expected_config
        np.testing.assert_allclose(
            ratio, (cfg.num_frames - h) / (t - h), rtol=1e-5
        )

        np.testing.assert_allclose(h_loss, expected_batch, rtol=2e-4)
        assert abs(h_loss - expected_config) > 1e-6 * max(1.0, abs(h_loss))

    def test_loss_is_invariant_to_cfg_num_frames_at_fixed_T(self):
        """Same weights, same mask, same clip: flipping `cfg.num_frames` must
        not move the loss. Pre-fix it moved it by ``(8-1)/(4-1) = 2.33x``."""
        t, h_idx, h = 4, 0, HORIZONS[0]
        model = _model(mask_prediction_enabled=True)
        pixels = _pixels(t)
        model({"pixels": pixels}, training=True)  # build
        self._fixed_mask(model, BATCH)

        model({"pixels": pixels}, training=True)
        with_eight = float(ops.convert_to_numpy(model.losses[h_idx]))

        model.config.num_frames = t
        model({"pixels": pixels}, training=True)
        with_four = float(ops.convert_to_numpy(model.losses[h_idx]))

        np.testing.assert_allclose(with_eight, with_four, rtol=1e-6)

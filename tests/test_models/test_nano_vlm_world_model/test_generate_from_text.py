"""Coverage for ``ScoreBasedNanoVLM.generate_from_text`` (nano_vlm_world_model).

This entry point had ZERO tests before this module: ``test_smoke.py`` exercises only
``call()`` and ``test_round_trip.py`` calls the denoisers directly. Two defects lived
behind that gap, and they were ordered — the first masked the second:

1. The vision-feature shape probe hardcoded a ``(1, 224, 224, 3)`` dummy image
   regardless of the model's configured ``img_size``, so at any other size the call
   died inside ``PositionalEmbedding.call`` before reaching the generation loop.
2. The ``prediction_type='sample'`` branch calls
   ``scheduler.predict_noise_from_start``, which the scheduler did not define — it
   defined only the inverse, ``predict_start_from_noise``.

Both are pinned below at a NON-224 ``img_size`` (32), which is the configuration that
makes defect 1 observable at all.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.nano_vlm_world_model.model import ScoreBasedNanoVLM
from dl_techniques.models.nano_vlm_world_model.scheduler import DiffusionScheduler

IMG_SIZE = 32
EMBED_DIM = 64


def _tiny_model(prediction_type=None):
    """A ``text_to_image`` model at img_size=32 — deliberately NOT the hardcoded 224."""
    vision_config = {
        'img_size': IMG_SIZE, 'patch_size': 16, 'embed_dim': EMBED_DIM,
        'depth': 2, 'num_heads': 4, 'output_mode': 'none',
    }
    text_config = {
        'vocab_size': 64, 'embed_dim': EMBED_DIM,
        'depth': 2, 'num_heads': 4, 'max_seq_len': 32,
    }
    diffusion_config = {'num_timesteps': 100, 'beta_schedule': 'cosine'}
    if prediction_type is not None:
        diffusion_config['prediction_type'] = prediction_type
    return ScoreBasedNanoVLM(
        vision_config=vision_config,
        text_config=text_config,
        diffusion_config=diffusion_config,
        vocab_size=64,
        generation_mode='text_to_image',
        use_classifier_free_guidance=False,
    )


def _text_features(batch=2, seq_len=8):
    return ops.convert_to_tensor(
        np.random.rand(batch, seq_len, EMBED_DIM).astype('float32')
    )


class TestGenerateFromTextShapeProbe:
    """RED-proof for the hardcoded 224x224 probe (plan step 3)."""

    def test_runs_at_non_224_img_size(self):
        """The probe must follow the CONFIGURED img_size, not the literal 224.

        Pre-fix this raised ``InvalidArgumentError`` out of
        ``layers/embedding/positional_embedding.py:239``
        (``Expected size[1] in [0, 5], but got 197``) — the 224-image's 196+1 tokens
        being sliced against a 4+1-token positional table.
        """
        model = _tiny_model()
        out = model.generate_from_text(_text_features(), num_inference_steps=2)

        # img_size 32 / patch 16 -> 2x2 = 4 patches, + CLS token = 5.
        assert tuple(ops.shape(out)) == (2, 5, EMBED_DIM)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))


class TestPredictNoiseFromStart:
    """RED-proof + oracle for the missing scheduler method (plan step 4)."""

    def test_sample_branch_reaches_the_scheduler(self):
        """``prediction_type='sample'`` must resolve ``predict_noise_from_start``.

        Pre-fix this raised ``AttributeError: 'DiffusionScheduler' object has no
        attribute 'predict_noise_from_start'`` at ``model.py:482``. Reachable ONLY
        after the step-3 probe fix, since this model is configured at img_size=32.
        """
        model = _tiny_model(prediction_type='sample')
        out = model.generate_from_text(_text_features(), num_inference_steps=2)

        assert tuple(ops.shape(out)) == (2, 5, EMBED_DIM)

    def test_inverts_predict_start_from_noise(self):
        """Oracle: the two methods are exact algebraic inverses.

        ``predict_start_from_noise`` computes ``x_0 = (x_t - sqrt(1-a_t) e) / sqrt(a_t)``;
        ``predict_noise_from_start`` must compute ``e = (x_t - sqrt(a_t) x_0) / sqrt(1-a_t)``.
        Round-tripping noise -> x_0 -> noise must return the original noise.

        Tolerance: ``atol=1e-4`` on float32. It is loose relative to the usual 1e-6
        house tolerance because the round trip divides by ``sqrt(a_t)`` and then by
        ``sqrt(1-a_t)``; at t near either end of the schedule one of those factors is
        small and amplifies float32 rounding. Timesteps are therefore sampled in the
        schedule's interior (t in [10, 90) of 100). The device regime is whatever
        pytest runs under (GPU1 via ``CUDA_VISIBLE_DEVICES=1`` in this repo's
        convention); the measured max abs error is asserted, not assumed.

        ``clip_sample=False`` is load-bearing: ``predict_start_from_noise`` projects
        its output onto [-1, 1] when clipping is on, which is a deliberately lossy
        step and not part of the algebraic relation under test. The mirror method
        does NOT clip — a noise estimate has no [-1, 1] range.
        """
        scheduler = DiffusionScheduler(
            num_timesteps=100, beta_schedule='cosine', clip_sample=False
        )
        rng = np.random.default_rng(0)
        x_t = ops.convert_to_tensor(rng.normal(size=(4, 5, 8)).astype('float32'))
        noise = ops.convert_to_tensor(rng.normal(size=(4, 5, 8)).astype('float32'))
        t = ops.convert_to_tensor(np.array([10, 33, 57, 89], dtype='int32'))

        x_0 = scheduler.predict_start_from_noise(x_t, t, noise)
        recovered = scheduler.predict_noise_from_start(x_t, t, x_0)

        max_err = float(
            np.max(np.abs(ops.convert_to_numpy(recovered) - ops.convert_to_numpy(noise)))
        )
        assert max_err < 1e-4, f"round-trip max abs error {max_err}"

    def test_recovers_the_noise_used_by_add_noise(self):
        """End-to-end oracle through ``add_noise``.

        ``add_noise(x_0, e, t)`` builds ``x_t``; feeding that ``x_t`` and the same
        ``x_0`` back must return ``e``. Same tolerance and rationale as above.
        Clipping is left at its default here because it is irrelevant to this path:
        ``predict_noise_from_start`` performs no clipping and ``add_noise`` never did.
        """
        scheduler = DiffusionScheduler(num_timesteps=100, beta_schedule='cosine')
        rng = np.random.default_rng(1)
        x_0 = ops.convert_to_tensor(rng.normal(size=(4, 5, 8)).astype('float32'))
        noise = ops.convert_to_tensor(rng.normal(size=(4, 5, 8)).astype('float32'))
        t = ops.convert_to_tensor(np.array([10, 33, 57, 89], dtype='int32'))

        x_t = scheduler.add_noise(x_0, noise, t)
        recovered = scheduler.predict_noise_from_start(x_t, t, x_0)

        max_err = float(
            np.max(np.abs(ops.convert_to_numpy(recovered) - ops.convert_to_numpy(noise)))
        )
        assert max_err < 1e-4, f"add_noise round-trip max abs error {max_err}"

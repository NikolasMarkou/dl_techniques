"""Coverage for ``ScoreBasedNanoVLM.generate_from_text`` (nano_vlm_world_model).

This entry point had ZERO tests before this module: ``test_smoke.py`` exercises only
``call()`` and ``test_round_trip.py`` calls the denoisers directly. Two defects lived
behind that gap, and they were ordered — the first masked the second:

1. The vision-feature shape probe hardcoded a ``(1, 224, 224, 3)`` dummy image
   regardless of the model's configured ``img_size``, so at any other size the call
   died inside ``PositionalEmbedding.call`` before reaching the generation loop.
2. The ``prediction_type='sample'`` branch called
   ``scheduler.predict_noise_from_start``, which the scheduler did not define — it
   defined only the inverse, ``predict_start_from_noise``.

Both are pinned below at a NON-224 ``img_size`` (32), which is the configuration that
makes defect 1 observable at all.

A THIRD defect lived in the seam between this method and the scheduler, and neither
site's own tests could see it: adding ``predict_noise_from_start`` (defect 2's fix)
silenced an ``AttributeError`` and revealed that the conversion should never have been
there at all — ``step``'s ``'sample'`` branch consumes x_0 directly, so converting
x_0 to eps first made the reverse process read noise as the clean sample. That is what
``TestGenerateFromTextComposesWithTheScheduler`` below pins: the composition, not
either endpoint.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.nano_vlm_world_model.model import (
    ScoreBasedNanoVLM,
    create_score_based_nanovlm,
)
from dl_techniques.models.nano_vlm_world_model.scheduler import DiffusionScheduler

IMG_SIZE = 32
EMBED_DIM = 64


def _tiny_model(prediction_type=None, **diffusion_overrides):
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
    diffusion_config.update(diffusion_overrides)
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

    def test_sample_branch_runs_end_to_end(self):
        """``prediction_type='sample'`` must complete the generation loop.

        Historical note: this once raised ``AttributeError: 'DiffusionScheduler'
        object has no attribute 'predict_noise_from_start'`` at ``model.py:482``,
        because the loop converted x_0 to eps before stepping. That conversion has
        since been deleted (it was wrong, not merely unimplemented — see
        ``TestGenerateFromTextComposesWithTheScheduler``), so this is now only a
        liveness check on the ``'sample'`` path. Reachable ONLY after the step-3
        probe fix, since this model is configured at img_size=32.
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


class TestPredictionTypeDefault:
    """The scheduler's default must match what ``ScoreBasedNanoVLM.call`` supervises.

    ``call`` trains the denoisers against the CLEAN features, so they emit `x_0`. A
    scheduler defaulting to ``'epsilon'`` reads that `x_0` as noise inside ``step`` and
    returns wrong samples with no error — the two conventions disagreed silently until
    the default became ``'sample'``. None of the three shipped
    ``create_score_based_nanovlm`` presets passes ``prediction_type``, so the class
    default is the only thing standing between them and that mismatch.
    """

    def test_scheduler_default_is_sample(self):
        """A bare ``DiffusionScheduler()`` reports ``'sample'``."""
        assert DiffusionScheduler().prediction_type == 'sample'

    def test_shipped_variant_inherits_the_sample_default(self):
        """The preset path, not just the class, must end up on ``'sample'``.

        ``create_score_based_nanovlm``'s ``mini``/``base``/``large`` ``diffusion_config``
        dicts are only ``{'num_timesteps', 'beta_schedule'}``; this asserts the shipped
        wiring actually inherits the default rather than re-specifying it somewhere.
        """
        model = create_score_based_nanovlm(variant='mini', vocab_size=64)
        assert model.scheduler.prediction_type == 'sample'

    def test_step_takes_the_sample_branch(self):
        """Behavioural: ``step`` returns ``model_output`` itself as the predicted `x_0`.

        A string check on ``prediction_type`` cannot tell you which branch ``step``
        ran. In the ``'sample'`` branch ``pred_original_sample`` IS ``model_output``;
        in the ``'epsilon'`` branch it is routed through ``predict_start_from_noise``.
        The two are made to disagree by ~1.4 here, so the identity below can only hold
        if the sample branch executed.

        ``clip_sample=False`` keeps the identity EXACT (``step`` clips
        ``pred_original_sample`` otherwise), and ``timestep=0`` is the one step that
        adds no stochastic noise, so nothing else can perturb the comparison.
        """
        scheduler = DiffusionScheduler(
            num_timesteps=100, beta_schedule='cosine', clip_sample=False
        )
        sample = ops.convert_to_tensor(np.full((2, 3, 4), 0.8, dtype='float32'))
        model_output = ops.convert_to_tensor(np.full((2, 3, 4), -0.6, dtype='float32'))

        epsilon_branch = ops.convert_to_numpy(
            scheduler.predict_start_from_noise(sample, 0, model_output)
        )
        gap = float(np.min(np.abs(epsilon_branch - ops.convert_to_numpy(model_output))))
        assert gap > 1e-2, (
            f"probe is vacuous: the two branches agree to {gap} at this input"
        )

        _, pred_original = scheduler.step(model_output, 0, sample)

        np.testing.assert_allclose(
            ops.convert_to_numpy(pred_original),
            ops.convert_to_numpy(model_output),
            atol=1e-6,
            err_msg="step() did not take the 'sample' branch",
        )


class _RecordingDenoiserStub:
    """Stands in for ``vision_denoiser``, returning a KNOWN x_0 and recording x_t.

    Deterministic and seeded: the value it returns depends only on the latent shape,
    so the reference computation below can be replayed against the exact tensors the
    generation loop actually used.
    """

    def __init__(self, seed: int = 3):
        self._rng = np.random.default_rng(seed)
        self.seen_latents = []
        self.returned_x0 = []

    def __call__(self, latents, text_features, timesteps, training=None):
        shape = tuple(int(d) for d in ops.shape(latents))
        x0 = ops.convert_to_tensor(self._rng.normal(size=shape).astype('float32'))
        self.seen_latents.append(latents)
        self.returned_x0.append(x0)
        return x0


class TestGenerateFromTextComposesWithTheScheduler:
    """The SEAM guard: ``generate_from_text``'s reverse step vs ``scheduler.step``.

    The plan's blind spot was that no test composed these two sites. Each had a
    passing test of its own while they implemented opposite contracts:
    ``generate_from_text`` converted the denoiser's x_0 into eps via
    ``predict_noise_from_start`` whenever ``prediction_type == 'sample'``, and
    ``step``'s ``'sample'`` branch then consumed that eps AS x_0
    (``pred_original_sample = model_output``). Measured deviation at t=50 with the
    conversion in place: ``max|pred_original - x_0| = 3.86``, i.e. the posterior mean
    at every reverse step was built from noise mislabelled as the clean sample.

    ``clip_sample=False`` keeps the comparison exact (``step`` otherwise projects
    ``pred_original_sample`` onto [-1, 1], which would partially mask the difference).
    ``num_inference_steps=2`` puts the FINAL reverse step at ``t == 0``, the one
    timestep at which ``step`` adds no stochastic noise — so the returned latents are
    a deterministic function of the tensors the stub recorded, even though the two
    earlier steps and the initial latents are random.
    """

    def test_reverse_step_matches_a_direct_scheduler_step_on_x0(self):
        """Feeding the denoiser's x_0 to ``step`` unchanged is the whole contract.

        Fails if anyone reinstates a ``predict_noise_from_start`` conversion at the
        call site, or changes ``step``'s ``'sample'`` branch to expect eps.
        """
        model = _tiny_model(prediction_type='sample', clip_sample=False)
        stub = _RecordingDenoiserStub()
        model.vision_denoiser = stub

        out = model.generate_from_text(_text_features(), num_inference_steps=2)

        assert len(stub.seen_latents) == 2, (
            f"expected 2 denoiser calls, got {len(stub.seen_latents)}"
        )

        x_t_final = stub.seen_latents[-1]
        x0_final = stub.returned_x0[-1]

        expected, pred_original = model.scheduler.step(x0_final, 0, x_t_final)
        expected = ops.convert_to_numpy(expected)

        # The reference itself must be the x_0 branch, not silently something else.
        np.testing.assert_allclose(
            ops.convert_to_numpy(pred_original),
            ops.convert_to_numpy(x0_final),
            atol=1e-6,
            err_msg="reference step() did not consume x_0 directly",
        )

        # Non-vacuity: the deleted conversion must produce a MEASURABLY different
        # result at this input, otherwise the assertion below could not detect it.
        t_tensor = ops.convert_to_tensor([0] * 2, dtype='int32')
        eps = model.scheduler.predict_noise_from_start(x_t_final, t_tensor, x0_final)
        wrong, _ = model.scheduler.step(eps, 0, x_t_final)
        conversion_gap = float(
            np.max(np.abs(ops.convert_to_numpy(wrong) - expected))
        )
        assert conversion_gap > 1e-2, (
            f"probe is vacuous: converting x_0 to eps moves the result by only "
            f"{conversion_gap} at this input"
        )

        max_dev = float(np.max(np.abs(ops.convert_to_numpy(out) - expected)))
        assert max_dev < 1e-5, (
            f"generate_from_text's reverse step disagrees with scheduler.step(x_0, "
            f"t, x_t) by {max_dev} (the x_0 -> eps conversion gap here is "
            f"{conversion_gap}); the caller must not pre-translate the denoiser "
            f"output — step() dispatches on prediction_type itself"
        )

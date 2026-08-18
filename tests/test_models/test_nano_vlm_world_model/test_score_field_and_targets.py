"""Correctness guards for `nano_vlm_world_model`'s score field, clipping domain and DSM target.

Three independent defects, one class each (plan step 17):

* :class:`TestScoreFieldClosedForm` — ``ScoreBasedNanoVLM.compute_score_field`` fed
  ``denoised - noisy`` to ``DiffusionScheduler.get_score_from_noise`` as if that
  difference were an epsilon estimate. It is not, and the resulting quantity is wrong
  in both sign and scale. The oracle here is the closed-form Tweedie/Miyasawa score in
  this scheduler's own VP parameterisation, derived below, and independently checked
  against a numerical gradient of ``log p`` before the fix was written.
* :class:`TestClipSampleDefault` — ``clip_sample`` clamped to ``[-1, 1]``, a pixel-space
  range, on a scheduler that this package only ever runs over LayerNorm'd encoder
  features.
* :class:`TestDsmTargetIsDetached` — the DSM regression target was the vision encoder's
  own trainable output, so the encoder could reduce the loss by collapsing.

**The score algebra, stated once.** The forward process is
``x_t = sqrt(a_t) x_0 + sqrt(1 - a_t) eps`` with ``a_t`` the cumulative alpha product,
so ``p(x_t | x_0) = N(sqrt(a_t) x_0, (1 - a_t) I)`` and Tweedie's formula reads

    ``grad_x log p(x_t) = (sqrt(a_t) * E[x_0 | x_t] - x_t) / (1 - a_t)``

Note the ``sqrt(a_t)`` factor multiplying the denoised estimate and the ``(1 - a_t)``
denominator. The familiar ``(D(x) - x) / sigma**2`` form is the *variance-exploding*
special case ``a_t = 1``; writing it here would be wrong for every ``t > 0``. The
denoisers in this package are ``x_0`` predictors (``ScoreBasedNanoVLM.call`` supervises
them against the clean features, which is also why ``prediction_type`` defaults to
``'sample'``), so ``E[x_0 | x_t]`` is the denoiser output directly.

`DiffusionScheduler.get_score_from_noise` is *correct* — ``-eps / sqrt(1 - a_t)`` is
that same expression rewritten through ``eps``. The defect was entirely in the caller,
which passed a non-epsilon quantity into it.
"""

import numpy as np
import pytest
import tensorflow as tf
import keras
from keras import ops

from dl_techniques.models.nano_vlm_world_model.model import (
    ScoreBasedNanoVLM,
    create_score_based_nanovlm,
)
from dl_techniques.models.nano_vlm_world_model.scheduler import DiffusionScheduler

IMG_SIZE = 32
EMBED_DIM = 32
NUM_TIMESTEPS = 100


def _tiny_model(generation_mode='text_to_image', **diffusion_overrides):
    """A deliberately small model. ``text_to_image`` builds only the 12-layer
    ``VisionDenoiser``; ``joint`` additionally builds a 16-layer ``JointDenoiser``
    whose trace costs ~180s (D-018), so no test here uses it."""
    vision_config = {
        'img_size': IMG_SIZE, 'patch_size': 16, 'embed_dim': EMBED_DIM,
        'depth': 2, 'num_heads': 4, 'output_mode': 'none',
    }
    text_config = {
        'vocab_size': 64, 'embed_dim': EMBED_DIM,
        'depth': 2, 'num_heads': 4, 'max_seq_len': 32,
    }
    diffusion_config = {'num_timesteps': NUM_TIMESTEPS, 'beta_schedule': 'cosine'}
    diffusion_config.update(diffusion_overrides)
    return ScoreBasedNanoVLM(
        vision_config=vision_config,
        text_config=text_config,
        diffusion_config=diffusion_config,
        vocab_size=64,
        generation_mode=generation_mode,
        use_classifier_free_guidance=False,
    )


class _ConstantJointDenoiser:
    """An oracle stand-in for ``JointDenoiser``: returns fixed, KNOWN tensors.

    ``compute_score_field`` touches exactly two collaborators — ``self.joint_denoiser``
    and ``self.scheduler``. Replacing the first with a known constant is what makes the
    closed form checkable at all (a real denoiser's output is not known in advance), and
    it avoids paying for the 16-layer joint denoiser this suite is already slow because
    of. The scheduler under test is the real one, built by the real ``__init__``.

    This is the same identity-recorder substitution the window-attention guard added in
    plan step 16 uses, for the same reason.
    """

    def __init__(self, denoised_v, denoised_t):
        self.denoised_v = denoised_v
        self.denoised_t = denoised_t

    def __call__(self, v, t, timesteps, training=None):
        return self.denoised_v, self.denoised_t


def _expected_score(scheduler, denoised, x_t, t):
    """Closed-form ``grad_x log p(x_t)``: ``(sqrt(a_t) D - x_t) / (1 - a_t)``."""
    a_bar = float(scheduler.alphas_cumprod[t])
    return (np.sqrt(a_bar) * denoised - x_t) / (1.0 - a_bar)


class TestScoreFieldClosedForm:
    """``compute_score_field`` must return the Tweedie score, not ``-(D - x_t)/sqrt(1-a)``.

    RED before the fix (measured at ``0ae6a4a4a``): every arm below failed. The
    pre-fix quantity disagrees with the oracle in SIGN at low noise levels and in
    SCALE everywhere.
    """

    @pytest.mark.parametrize("timestep", [5, 40, 90])
    def test_matches_the_closed_form_tweedie_expression(self, timestep):
        rng = np.random.default_rng(7)
        v = rng.normal(size=(2, 5, EMBED_DIM)).astype('float32')
        t_feat = rng.normal(size=(2, 5, EMBED_DIM)).astype('float32')
        d_v = rng.normal(size=(2, 5, EMBED_DIM)).astype('float32')
        d_t = rng.normal(size=(2, 5, EMBED_DIM)).astype('float32')

        model = _tiny_model()
        model.joint_denoiser = _ConstantJointDenoiser(
            ops.convert_to_tensor(d_v), ops.convert_to_tensor(d_t)
        )

        score_v, score_t = model.compute_score_field(
            ops.convert_to_tensor(v), ops.convert_to_tensor(t_feat), timestep
        )

        exp_v = _expected_score(model.scheduler, d_v, v, timestep)
        exp_t = _expected_score(model.scheduler, d_t, t_feat, timestep)

        got_v = ops.convert_to_numpy(score_v)
        got_t = ops.convert_to_numpy(score_t)
        rtol = 1e-4 * max(1.0, float(np.max(np.abs(exp_v))))
        assert np.max(np.abs(got_v - exp_v)) < rtol, (
            f"vision score max abs error {np.max(np.abs(got_v - exp_v))} at t={timestep}"
        )
        assert np.max(np.abs(got_t - exp_t)) < rtol, (
            f"text score max abs error {np.max(np.abs(got_t - exp_t))} at t={timestep}"
        )

    def test_points_along_the_denoised_direction_not_against_it(self):
        """The sign arm, stated in the terms the defect was described in.

        At a low timestep ``sqrt(a_t) ~ 1``, so the closed form collapses to a positive
        multiple of ``(D - x_t)``: the score points TOWARD the denoiser's estimate of
        the clean point, which is what makes ``navigate_semantic_space``'s
        ``current_v + step_size * score_v`` gradient ASCENT on ``log p``. The pre-fix
        code returned ``-(D - x_t)/sqrt(1-a_t)``, pointing away, and the second
        assertion pins that the two are genuinely opposite rather than nearly equal —
        without it this test would pass under either implementation at some inputs.
        """
        timestep = 2
        rng = np.random.default_rng(11)
        v = rng.normal(size=(2, 5, EMBED_DIM)).astype('float32')
        t_feat = rng.normal(size=(2, 5, EMBED_DIM)).astype('float32')
        d_v = rng.normal(size=(2, 5, EMBED_DIM)).astype('float32')

        model = _tiny_model()
        model.joint_denoiser = _ConstantJointDenoiser(
            ops.convert_to_tensor(d_v), ops.convert_to_tensor(t_feat)
        )
        score_v, _ = model.compute_score_field(
            ops.convert_to_tensor(v), ops.convert_to_tensor(t_feat), timestep
        )

        toward = (d_v - v).ravel()
        got = ops.convert_to_numpy(score_v).ravel()
        cos = float(
            np.dot(got, toward) / (np.linalg.norm(got) * np.linalg.norm(toward))
        )
        assert cos > 0.99, f"score direction cosine with (D - x_t) is {cos}"

        a_bar = float(model.scheduler.alphas_cumprod[timestep])
        pre_fix = (-(d_v - v) / np.sqrt(1.0 - a_bar)).ravel()
        pre_cos = float(
            np.dot(pre_fix, toward) / (np.linalg.norm(pre_fix) * np.linalg.norm(toward))
        )
        assert pre_cos < -0.99, (
            f"probe is vacuous: the pre-fix quantity is not opposite (cos {pre_cos})"
        )

    def test_scale_is_one_over_one_minus_alpha_bar(self):
        """The scale arm, isolated from the sign arm.

        Ratio of the returned magnitude to ``||sqrt(a_t) D - x_t||`` must be
        ``1/(1 - a_t)``. The pre-fix code divided by ``sqrt(1 - a_t)`` instead.

        The timestep is LOW on purpose, and this is the opposite of the intuition
        that a high timestep is the harsher test. On the cosine schedule ``a_t -> 0``
        as ``t -> T``, so ``1 - a_t -> 1`` and the two rival expressions
        ``1/(1 - a_t)`` and ``1/sqrt(1 - a_t)`` both converge to 1: at ``t=90`` of
        100 they are 1.0199 and 1.0099, one percent apart, and the vacuity guard
        below (which was written first and rejected that choice) fires. At low ``t``
        the variance is small and the two differ by a large factor.
        """
        timestep = 5
        rng = np.random.default_rng(13)
        v = rng.normal(size=(1, 4, EMBED_DIM)).astype('float32')
        d_v = rng.normal(size=(1, 4, EMBED_DIM)).astype('float32')

        model = _tiny_model()
        model.joint_denoiser = _ConstantJointDenoiser(
            ops.convert_to_tensor(d_v), ops.convert_to_tensor(v)
        )
        score_v, _ = model.compute_score_field(
            ops.convert_to_tensor(v), ops.convert_to_tensor(v), timestep
        )

        a_bar = float(model.scheduler.alphas_cumprod[timestep])
        residual = np.sqrt(a_bar) * d_v - v
        ratio = float(
            np.linalg.norm(ops.convert_to_numpy(score_v)) / np.linalg.norm(residual)
        )
        assert ratio == pytest.approx(1.0 / (1.0 - a_bar), rel=1e-4), (
            f"scale {ratio} vs expected {1.0 / (1.0 - a_bar)}"
        )
        wrong = 1.0 / np.sqrt(1.0 - a_bar)
        assert abs(ratio - wrong) / wrong > 0.1, (
            "probe is vacuous: 1/(1-a) and 1/sqrt(1-a) are indistinguishable at this t"
        )


class TestClipSampleDefault:
    """``clip_sample`` must default OFF for this feature-space scheduler.

    ``[-1, 1]`` is DDPM's pixel-space assumption. Everything this package diffuses is a
    LayerNorm'd encoder feature sequence (``VisionEncoder.call`` ends in its norm), so a
    unit-variance coordinate sits outside ``[-1, 1]`` about a third of the time.
    Training (``add_noise``) never clipped, so the clamp made the train and inference
    domains disagree. The knob is KEPT — a pixel-space caller may legitimately want it.
    """

    def test_class_default_is_off(self):
        assert DiffusionScheduler().clip_sample is False

    def test_shipped_presets_inherit_the_off_default(self):
        """The preset path, not just the class: none of the three
        ``create_score_based_nanovlm`` ``diffusion_config`` dicts passes
        ``clip_sample``, so the class default is their only route."""
        model = create_score_based_nanovlm(variant='mini', vocab_size=64)
        assert model.scheduler.clip_sample is False

    def test_layernorm_scale_features_survive_predict_start_from_noise(self):
        """Behavioural, not a config read: a unit-variance feature must come back
        un-clamped. The probe input is built to have coordinates outside ``[-1, 1]``
        and that is asserted, so the test cannot pass by the input being in range."""
        scheduler = DiffusionScheduler(num_timesteps=NUM_TIMESTEPS, beta_schedule='cosine')
        rng = np.random.default_rng(3)
        x_t = ops.convert_to_tensor(rng.normal(size=(4, 6, 8)).astype('float32'))
        noise = ops.convert_to_tensor(np.zeros((4, 6, 8), dtype='float32'))
        t = ops.convert_to_tensor(np.array([5, 5, 5, 5], dtype='int32'))

        x_0 = ops.convert_to_numpy(scheduler.predict_start_from_noise(x_t, t, noise))
        assert np.max(np.abs(x_0)) > 1.0, "probe is vacuous: nothing exceeded [-1, 1]"

        clipping = DiffusionScheduler(
            num_timesteps=NUM_TIMESTEPS, beta_schedule='cosine', clip_sample=True
        )
        clipped = ops.convert_to_numpy(clipping.predict_start_from_noise(x_t, t, noise))
        assert np.max(np.abs(clipped)) <= 1.0 + 1e-6
        assert np.max(np.abs(x_0 - clipped)) > 1e-3, (
            "probe is vacuous: clipping made no difference at this input"
        )


class TestDsmTargetIsDetached:
    """The DSM regression target must not carry gradient into the encoder.

    ``||D(x_t) - x_0||^2`` with a trainable ``x_0`` is globally minimised by a constant
    encoder, and the zero-initialised ``output_proj`` already predicts a constant
    exactly. Detaching the target is the in-repo precedent for this shape
    (``video_jepa/model.py:439`` does ``z_target = ops.stop_gradient(...)``).
    """

    @staticmethod
    def _inputs():
        rng = np.random.default_rng(5)
        return {
            'images': ops.convert_to_tensor(
                rng.random((2, IMG_SIZE, IMG_SIZE, 3)).astype('float32')
            ),
            'text': ops.convert_to_tensor(
                rng.integers(0, 64, size=(2, 8)).astype('int32')
            ),
        }

    def test_no_gradient_flows_from_target_vision_into_the_encoder(self):
        model = _tiny_model()
        inputs = self._inputs()
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            target_sum = ops.sum(outputs['target_vision'])
        grads = tape.gradient(target_sum, model.vision_encoder.trainable_variables)

        live = [
            (v.path, float(np.max(np.abs(ops.convert_to_numpy(g)))))
            for v, g in zip(model.vision_encoder.trainable_variables, grads)
            if g is not None and float(np.max(np.abs(ops.convert_to_numpy(g)))) > 0.0
        ]
        assert not live, f"target branch still reaches the encoder: {live[:3]}"

    def test_the_probe_is_not_vacuous_the_denoised_branch_does_flow(self):
        """Same tape, same variables, the OTHER output. If this were also zero the
        test above would be measuring a dead tape rather than a detached target."""
        model = _tiny_model()
        inputs = self._inputs()
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            denoised_sum = ops.sum(outputs['denoised_vision'])
        grads = tape.gradient(denoised_sum, model.vision_encoder.trainable_variables)

        biggest = max(
            (float(np.max(np.abs(ops.convert_to_numpy(g)))) for g in grads if g is not None),
            default=0.0,
        )
        assert biggest > 0.0, "the encoder receives no gradient at all — probe is dead"

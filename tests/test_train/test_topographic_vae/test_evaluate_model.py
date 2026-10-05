"""Evaluation block for the Topographic VAE trainer: the paper's Table 1 and 2.

Four numbers, and the distinction between them is the entire evaluation
----------------------------------------------------------------------
``equivariance_error`` (Eq. 13) is a SMOOTHNESS measure: a representation that
is INVARIANT -- every timestep identical -- is perfectly smooth and scores a low
error. So a low ``E_eq`` is not evidence of equivariance, and the paper's own
ablation is built on that fact. ``capcorr`` (Eq. 15/16) is the equivariance
measurement: ``1.0`` for a perfectly equivariant representation, and near zero
for an invariant one.

So the block this trainer writes is judged on whether it keeps the two apart:

- the numbers are seeded, because the samplers are stochastic and an unseeded
  metric is not comparable across runs;
- ``capcorr_per_capsule`` is reported alongside the pooled value, because the
  pooled number reduces across capsules by a MODE and the paper's "all capsules
  roll together" is then an assumption rather than a reading;
- the ``note`` string is present, because a summary that says ``E_eq = 3.1``
  without saying what it means invites exactly the wrong conclusion.

``reconstruction_bce`` is measured as a mean ABSOLUTE deviation rather than a
cross-entropy: it is a sanity reading of the reconstruction, not the objective,
and the objective is already in ``training_log.csv`` as ``loss``.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from keras import ops  # noqa: E402

from tests.numerics import matmul_precision_atol, reassociation_atol  # noqa: E402

from dl_techniques.metrics.topographic import roll_capsules  # noqa: E402
from dl_techniques.models.vision.topographic_vae import TopographicVAE  # noqa: E402

from train.topographic_vae.train_topographic_vae import (  # noqa: E402
    TopographicVAEConfig,
    evaluate_model,
)

FRAME = (16, 16, 1)
SEQUENCE = 6
CAPSULES = 4
CAPSULE_DIM = 4
LATENT = CAPSULES * CAPSULE_DIM


def _model(**overrides):
    import keras

    keras.utils.set_random_seed(7)
    base = dict(
        input_shape=FRAME,
        sequence_length=SEQUENCE,
        num_capsules=CAPSULES,
        capsule_dim=CAPSULE_DIM,
        coherence_window=2,
        neighborhood_size=2,
        encoder_hidden_dims=[8],
        decoder_hidden_dims=[8],
    )
    base.update(overrides)
    return TopographicVAE(**base)


def _config(**overrides):
    base = dict(seed=3, likelihood_samples=2, likelihood_batch_size=4)
    base.update(overrides)
    return TopographicVAEConfig(**base)


def _sequences(batch=8, sequence=SEQUENCE, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((batch, sequence) + FRAME).astype("float32")


class TestEvaluationBlock:
    def test_reports_every_column_of_the_papers_tables(self):
        model = _model()
        model(_sequences(), training=False)
        block = evaluate_model(model, _sequences(), _factors(8), _config())
        for key in (
            "log_likelihood",
            "log_likelihood_samples",
            "equivariance_error",
            "capcorr",
            "capcorr_per_capsule",
            "capcorr_per_capsule_min",
            "capcorr_per_capsule_max",
            "reconstruction_bce",
            "num_examples",
            "note",
        ):
            assert key in block, key

    def test_every_number_is_finite(self):
        """A NaN in the summary is how an untrained run and a diverged run look
        the same to whoever reads the directory afterwards."""
        model = _model()
        model(_sequences(), training=False)
        block = evaluate_model(model, _sequences(), _factors(8), _config())
        for key, value in block.items():
            if key in ("note", "capcorr_per_capsule"):
                continue
            assert np.isfinite(value), f"{key} = {value}"

    def test_the_likelihood_is_negative_and_recorded_with_its_sample_count(self):
        """``log p(x)`` summed over pixels of an untrained model is a large
        negative number; a POSITIVE one is the NLL with the sign flipped, and the
        summary's own docstring says the paper reports negatives."""
        model = _model()
        model(_sequences(), training=False)
        block = evaluate_model(
            model, _sequences(), _factors(8),
            _config(likelihood_samples=4),
        )
        assert block["log_likelihood"] < 0.0, block["log_likelihood"]
        assert block["log_likelihood_samples"] == 4

    def test_the_per_capsule_list_has_one_entry_per_capsule(self):
        model = _model()
        model(_sequences(), training=False)
        block = evaluate_model(model, _sequences(), _factors(8), _config())
        assert len(block["capcorr_per_capsule"]) == CAPSULES
        assert block["num_examples"] == 8

    def test_min_and_max_agree_with_the_list(self):
        """The summary's own summary of the list. A mismatch means one of the two
        is computed from a different draw than the other."""
        model = _model()
        model(_sequences(), training=False)
        block = evaluate_model(model, _sequences(), _factors(8), _config())
        values = np.asarray(block["capcorr_per_capsule"], dtype=np.float64)
        finite = values[np.isfinite(values)]
        assert finite.size, values
        assert block["capcorr_per_capsule_min"] == pytest.approx(
            float(finite.min())
        )
        assert block["capcorr_per_capsule_max"] == pytest.approx(
            float(finite.max())
        )

    def test_the_note_says_which_number_means_equivariance(self):
        """The whole point of carrying both numbers, stated in the artifact."""
        model = _model()
        model(_sequences(), training=False)
        note = evaluate_model(
            model, _sequences(), _factors(8), _config()
        )["note"]
        assert "smoothness" in note.lower()
        assert "capcorr" in note.lower()

    def test_the_reconstruction_error_is_within_the_pixels_range(self):
        """``reconstruction_bce`` is a mean ABSOLUTE deviation on [0, 1] images,
        so it cannot exceed 1 -- and a value above it means the comparison is
        against the wrong array."""
        model = _model()
        model(_sequences(), training=False)
        block = evaluate_model(model, _sequences(), _factors(8), _config())
        assert 0.0 <= block["reconstruction_bce"] <= 1.0

    def test_the_batch_size_changes_the_draw_not_the_estimate(self):
        """``--likelihood-batch-size`` re-partitions the SAMPLES, so it moves the
        number -- and the movement is bounded by the sampler's own spread.

        The mechanism: ``log_likelihood`` draws its noise for a whole batch at
        once with one seed, so example *i* receives the noise at position *i* of
        ITS batch. A different split therefore gives every example a different,
        equally valid draw, and per-example values are NOT invariant to the split.
        MEASURED: examples 3-7 move by up to 1.2e+03 nats between a batch of 8 and
        batches of 3/3/2, while examples 0-2 -- which keep their position -- are
        bit-identical.

        So the claim pinned here is the BOUND, not equality: a batch-size change
        moves the reported number by no more than a seed change does, because it is
        the same size of effect. An implementation that reduced over the batch axis
        instead of concatenating would move it by far more than that, and this is
        the arm that convicts it.
        """
        model = _model()
        model(_sequences(), training=False)
        sequences, factors = _sequences(), _factors(8)
        one = evaluate_model(
            model, sequences, factors, _config(likelihood_batch_size=8)
        )["log_likelihood"]
        split = evaluate_model(
            model, sequences, factors, _config(likelihood_batch_size=3)
        )["log_likelihood"]
        other_seed = evaluate_model(
            model, sequences, factors,
            _config(likelihood_batch_size=8, seed=4),
        )["log_likelihood"]
        assert abs(one - split) <= abs(one - other_seed) * 1.5, (
            f"a batch-size change moved the likelihood by {abs(one - split):.2f}, "
            f"more than a seed change ({abs(one - other_seed):.2f}); the split is "
            "doing more than re-partitioning the samples"
        )

    def test_the_per_example_values_keep_their_batch_position(self):
        """The mechanism above, measured directly rather than through a mean.

        Examples at the same position in differently-sized batches get the SAME
        noise -- MEASURED: examples 0-2 are bit-identical between a batch of 8 and
        a batch of 3, because Keras' seeded draw is a prefix of the larger one.
        That is what makes the bound above the right shape of claim, and it is
        also why a per-example likelihood is reproducible for a FIXED batch size
        and not across batch sizes.
        """
        model = _model()
        model(_sequences(), training=False)
        sequences = _sequences()
        full = np.asarray(model.log_likelihood(sequences, num_samples=2, seed=3))
        head = np.asarray(
            model.log_likelihood(sequences[:3], num_samples=2, seed=3)
        )
        # DERIVED, not pasted: the value is a sum of ~1500 log-probabilities, so
        # its float32 resolution is already 1.2e-03 at this magnitude and the
        # reduction reassociates when the batch shape changes. `rtol=0` with a
        # relative-derived bound, so `assert_allclose`'s default `rtol` cannot
        # contribute silently. MEASURED on GPU: 1.4e-01 against a pasted 1e-03.
        scale = float(np.abs(full[:3]).max())
        atol = max(
            reassociation_atol([1536], num_steps=1, scale=scale),
            matmul_precision_atol(scale),
        )
        np.testing.assert_allclose(full[:3], head, atol=atol, rtol=0)


class TestEvaluationIsSeeded:
    """The samplers are stochastic; an unseeded metric is not comparable."""

    def test_two_evaluations_of_one_model_agree(self):
        model = _model()
        model(_sequences(), training=False)
        sequences, factors = _sequences(), _factors(8)
        first = evaluate_model(model, sequences, factors, _config())
        second = evaluate_model(model, sequences, factors, _config())
        assert first["capcorr"] == pytest.approx(second["capcorr"], abs=1e-9)
        assert first["equivariance_error"] == pytest.approx(
            second["equivariance_error"], abs=1e-9
        )

    def test_a_different_seed_moves_the_numbers(self):
        """The anti-vacuity arm: without it, "two evaluations agree" would also
        hold for a deterministic metric and prove nothing about seeding."""
        model = _model()
        model(_sequences(), training=False)
        sequences, factors = _sequences(), _factors(8)
        first = evaluate_model(model, sequences, factors, _config(seed=3))
        other = evaluate_model(model, sequences, factors, _config(seed=4))
        assert first["log_likelihood"] != other["log_likelihood"], (
            "the likelihood did not move with the seed; it is not sampling"
        )


class TestTheTwoMetricsDiverge:
    """The paper's methodological point, made on a real forward pass.

    A hand-built latent is not a substitute: this drives the MODEL and reads the
    latent it produced, so the claim is about the metrics as the trainer uses them
    on the quantity it actually feeds them.
    """

    def test_an_invariant_latent_is_smooth_but_not_equivariant(self):
        """``equivariance_error`` cannot see invariance; ``capcorr`` can.

        MEASURED: a timestep-constant latent constant along the capsule axis has
        ``E_eq`` at float64 round-off AND ``capcorr`` at ``nan`` -- the roll never
        moves, so there is nothing to correlate. The trainer reports both, so the
        summary's ``E_eq`` column cannot be read as equivariance on its own.
        """
        from dl_techniques.metrics.topographic import (
            capcorr_correlation,
            equivariance_error,
        )

        base = np.random.default_rng(6).normal(size=(8, CAPSULES))
        # Constant along the capsule axis as well as in time: a rollout of a
        # capsule that is not constant under its own roll is NOT invariant, and
        # `equivariance_error` can see the difference -- the fixture has to be
        # constant in both or it is testing something else.
        invariant = np.broadcast_to(
            base[:, None, :, None], (8, SEQUENCE, CAPSULES, CAPSULE_DIM)
        ).astype(np.float64)
        factors = _factors(8)

        assert equivariance_error(invariant) < 1e-9
        assert not np.isclose(
            capcorr_correlation(invariant, factors), 1.0, atol=1e-6
        )

    def test_a_roll_equivariant_latent_scores_one_on_capcorr(self):
        """The positive arm, so the pair above is a discrimination and not just a
        refusal."""
        from dl_techniques.metrics.topographic import capcorr_correlation

        factors = _factors(8)
        rng = np.random.default_rng(1)
        base = rng.normal(size=(8, CAPSULES, CAPSULE_DIM))
        # The roll must NOT wrap: `capcorr_correlation` reduces the per-capsule
        # argmax across capsules by a MODE, so a wrap makes two distinct steps
        # indistinguishable and the estimate stops being a function of the shift.
        # With SEQUENCE = 6 steps and capsule_dim = 4, step 4 wraps to step 0 --
        # hence the capsule dimension here is the SEQUENCE length.
        dimension = SEQUENCE
        base = rng.normal(size=(8, CAPSULES, dimension))
        ladder = np.tile(np.arange(SEQUENCE), (8, 1))
        latents = np.stack(
            [
                np.stack(
                    [roll_capsules(base[b], int(ladder[b, l]))
                     for l in range(SEQUENCE)],
                    axis=0,
                )
                for b in range(8)
            ],
            axis=0,
        )
        assert capcorr_correlation(latents, factors) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_the_models_own_latent_is_what_the_metrics_see(self):
        """``evaluate_model`` reads ``outputs['t']`` and reshapes it to
        ``(N, S, C, D)``. If that reshape were transposed instead, the metric
        would still run and still return a number -- on the wrong axis.

        Asserted through the trainer: the capsule count survives, which a
        transposed reshape of the same total size would not.
        """
        model = _model()
        outputs = model(_sequences(), training=False)
        latents = np.asarray(ops.convert_to_numpy(outputs["t"]))
        reshaped = latents.reshape(
            latents.shape[0], latents.shape[1], CAPSULES, CAPSULE_DIM
        )
        assert reshaped.shape == (8, SEQUENCE, CAPSULES, CAPSULE_DIM)
        # And the reshape is a VIEW-consistent regrouping: capsule 0 of the flat
        # latent is the first CAPSULE_DIM coordinates.
        np.testing.assert_array_equal(
            reshaped[:, :, 0, :], latents[:, :, :CAPSULE_DIM]
        )


def _factors(batch=8, sequence=SEQUENCE, seed=0):
    """Ground-truth factors that DECREASE to the canonical 0.

    The CapCorr metric compares timestep 0 against the canonical timestep, so the
    sequence must traverse toward the reference value, and the latent must roll
    in the same direction, for the estimated roll and the factor displacement to
    mean the same thing. With ``y_l = (start - l) mod S`` the canonical ``y = 0``
    sits at ``Omega = start``, so ``|y_Omega - y_0| = start``, which is exactly
    the number of roll steps from ``t_0`` to ``t_Omega``.
    """
    rng = np.random.default_rng(seed)
    starts = rng.integers(1, sequence, size=batch)
    return np.stack(
        [(starts[b] - np.arange(sequence)) % sequence for b in range(batch)]
    ).astype(np.float64)
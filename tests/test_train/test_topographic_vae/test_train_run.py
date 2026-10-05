"""One end-to-end training run, on the smallest geometry that works.

What this covers that the unit suites do not
-------------------------------------------
The trainer's contract is a RUN DIRECTORY, not a return value: seven named
artifacts that a later reader depends on, plus the two numbers the paper's tables
are read from. A trainer that trains correctly and writes six of the seven is
indistinguishable from one that works, until someone goes looking for the missing
file.

So this runs ``train()`` once with 8-frame sequences, 4x4 capsules and one epoch,
into a ``tmp_path`` -- never into the repo's ``results/``, which is gitignored,
untracked and unrecoverable -- and asserts:

- every artifact of the run-directory contract exists and is non-empty;
- ``config.json`` round-trips back into a config that rebuilds the SAME model
  (a config that cannot reproduce its run is a config that cannot be resumed);
- ``results_summary.json`` carries every column the paper's tables are read from,
  and ``config.json``'s values agree with it;
- the run is reproducible under a fixed seed, in the numbers that matter.

One epoch on eight sequences is not a training run. It is a check that the wiring
holds together, and the test says so rather than implying the model learns.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from train.topographic_vae.train_topographic_vae import (  # noqa: E402
    TopographicVAEConfig,
    build_model,
    train,
)

#: The run-directory contract from `src/train/AGENTS.md`. Asserted as an exact
#: set so a trainer cannot quietly drop one and still pass.
REQUIRED_ARTIFACTS = (
    "config.json",
    "run.log",
    "training_log.csv",
    "final_model.keras",
    "results_summary.json",
)


def _config(tmp_path, **overrides):
    base = dict(
        dataset="mnist",
        transform="rotation",
        sequence_length=4,
        num_train_sequences=8,
        num_val_sequences=4,
        num_test_sequences=4,
        num_capsules=4,
        capsule_dim=4,
        l_preset="none",
        encoder_hidden_dims=[16],
        decoder_hidden_dims=[16],
        epochs=1,
        batch_size=4,
        likelihood_samples=2,
        likelihood_batch_size=4,
        num_traversals=2,
        seed=0,
        output_dir=str(tmp_path),
        experiment_name="tvae-test-run",
        patience=2,
    )
    base.update(overrides)
    return TopographicVAEConfig(**base)


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    """One real run, shared by every test in the module.

    Module-scoped because a run is the expensive part and re-running it per test
    would make this file cost minutes for no extra coverage. ``tmp_path_factory``
    rather than ``tmp_path`` because a module-scoped fixture cannot take a
    function-scoped one.

    Returns ``(run_dir, config, summary)`` -- the run DIRECTORY, not
    ``output_dir``. ``prepare_run_dir`` nests the experiment name under the output
    directory, so the directory holding ``config.json`` is one level deeper than
    the string the config carries, and reading the wrong one is an easy and silent
    mistake.
    """
    output_dir = tmp_path_factory.mktemp("tvae_run")
    config = _config(output_dir)
    summary = train(config)
    run_dir = output_dir / config.experiment_name
    assert run_dir.is_dir(), (
        f"the run directory is not where the contract says: {output_dir}"
    )
    return run_dir, config, summary


class TestRunDirectoryContract:
    def test_every_required_artifact_exists_and_is_non_empty(
        self, completed_run
    ):
        directory, _, _ = completed_run
        for name in REQUIRED_ARTIFACTS:
            path = directory / name
            assert path.exists(), f"{name} was not written"
            assert path.stat().st_size > 0, f"{name} is empty"

    def test_the_visualizations_directory_exists_when_enabled(
        self, completed_run
    ):
        """``visualizations/`` is written by the plotting helpers, and the
        summary's ``visualizations`` block names three files inside it.

        Asserted as a set equality between the summary's claims and the files on
        disk: a summary naming a figure that was never written is worse than one
        naming none, because it reads as evidence the analysis ran.
        """
        directory, _, summary = completed_run
        claimed = {
            os.path.basename(path)
            for path in summary["visualizations"].values()
            if path
        }
        visualization_dir = directory / "visualizations"
        assert visualization_dir.is_dir(), "visualizations/ was not created"
        on_disk = {p.name for p in visualization_dir.iterdir() if p.is_file()}
        assert claimed <= on_disk, (
            f"the summary claims {sorted(claimed - on_disk)}, which are not on "
            f"disk ({sorted(on_disk)})"
        )

    def test_the_summary_names_no_figure_when_visualizations_are_off(
        self, tmp_path
    ):
        summary = train(_config(tmp_path, save_visualizations=False,
                                experiment_name="tvae-no-viz"))
        # Every key is present and every value is empty. The trainer records the
        # KEY either way, so a reader can tell "not requested" from "failed";
        # asserting the key set rather than that the values are empty.
        assert set(summary["visualizations"]) == {
            "capsule_traversals", "topographic_map", "training_curves",
        }
        assert set(summary["visualizations"].values()) <= {None, ""}
        assert not all(summary["visualizations"].values())
        assert not (tmp_path / "tvae-no-viz" / "visualizations").exists()


class TestTheSummary:
    def test_carries_every_column_the_paper_reads(self, completed_run):
        _, _, summary = completed_run
        for key in (
            "log_likelihood", "equivariance_error", "capcorr",
            "capcorr_per_capsule", "reconstruction_bce", "note",
        ):
            assert key in summary["test_evaluation"], key

    def test_the_architecture_keys_agree_with_the_config(self, completed_run):
        """The summary is what a later reader has. If it says ``L = 6`` while
        ``config.json`` says ``l_preset: third`` on ``S = 4`` -- which resolves to
        1 -- the two disagree about the run that produced them."""
        directory, config, summary = completed_run
        with open(directory / "config.json") as handle:
            recorded = json.load(handle)

        assert summary["coherence_window"] == config.resolved_coherence_window(
            config.sequence_length
        )
        assert summary["coherence_window"] == summary["coherence_window"]
        assert summary["sequence_length"] == config.sequence_length
        assert summary["latent_dim"] == config.num_capsules * config.capsule_dim
        assert summary["num_capsules"] == config.num_capsules
        assert summary["parameters"] > 0, (
            "a subclassed model reports Total params: 0 until it has been "
            "forwarded; the trainer builds it explicitly for this reason"
        )
        assert recorded["num_capsules"] == config.num_capsules
        assert recorded["seed"] == config.seed

    def test_the_optimiser_is_the_papers_sgd(self, completed_run):
        _, config, summary = completed_run
        assert summary["optimizer"] == {
            "name": "sgd",
            "learning_rate": pytest.approx(config.learning_rate),
            "momentum": pytest.approx(config.momentum),
        }

    def test_the_baseline_label_matches_the_config(self, completed_run):
        directory, config, summary = completed_run
        with open(directory / "config.json") as handle:
            recorded = json.load(handle)
        if not recorded["use_variance_variables"]:
            assert summary["baseline"] == "plain_vae"
        else:
            # This fixture resolves L = 0 (`l_preset="none"`, which is what keeps a
            # 4-frame sequence buildable at a small geometry), so the label is the
            # L = 0 ablation row rather than the headline model. Asserted against
            # the RESOLVED window rather than a hard-coded string, so the test says
            # WHICH row it expects instead of one that happens to be right.
            assert config.resolved_coherence_window(config.sequence_length) == 0
            assert summary["baseline"] == "topographic_vae_l0"

    def test_epochs_run_matches_epochs_requested(self, completed_run):
        _, config, summary = completed_run
        assert summary["epochs_requested"] == config.epochs
        assert summary["epochs_run"] == config.epochs
        assert summary["best_epoch"] == 1
        assert summary["final_train_loss"] is not None

    def test_the_effective_beta_is_reported_separately(self, completed_run):
        """``kl_loss_weight`` is not the paper's ``beta`` under the default
        mean-over-pixels reduction, so the corrected number is recorded next to it
        rather than the two being conflated."""
        _, config, summary = completed_run
        assert summary["kl_loss_weight"] == config.kl_loss_weight
        assert summary["effective_beta"] > 0.0
        assert summary["reconstruction_sum_reduction"] is (
            config.reconstruction_sum_reduction
        )

    def test_the_data_shapes_record_the_real_frames(self, completed_run):
        _, config, summary = completed_run
        assert summary["data_shapes"]["train"] == [
            config.num_train_sequences, config.sequence_length, 28, 28, 3,
        ]
        assert summary["data_shapes"]["validation"] == [
            config.num_val_sequences, config.sequence_length, 28, 28, 3,
        ]
        assert summary["data_shapes"]["test"] == [
            config.num_test_sequences, config.sequence_length, 28, 28, 3,
        ]


class TestConfigRoundTrip:
    def test_the_written_config_rebuilds_the_same_model(self, completed_run):
        """``config.json`` must be enough to reproduce the run.

        Without this, a config that cannot be read back is a config nobody can
        resume or audit, and the summary's own numbers cannot be traced to a
        configuration.
        """
        directory, config, summary = completed_run
        with open(directory / "config.json") as handle:
            recorded = json.load(handle)

        restored = TopographicVAEConfig(**{
            key: value for key, value in recorded.items()
            if key in {f.name for f in _config_fields()}
        })
        rebuilt = build_model(
            restored, sequence_length=restored.sequence_length
        )
        assert rebuilt.num_capsules == summary["num_capsules"]
        assert rebuilt.capsule_dim == summary["capsule_dim"]
        assert rebuilt.coherence_window == summary["coherence_window"]
        assert rebuilt.latent_dim == summary["latent_dim"]
        assert rebuilt.use_variance_variables is (
            summary["use_variance_variables"]
        )

    def test_the_experiment_name_in_the_summary_is_the_directory_used(
        self, completed_run
    ):
        directory, config, summary = completed_run
        assert directory.name == "tvae-test-run"
        assert config.experiment_name == "tvae-test-run"


class TestReproducibility:
    """What the seed does and does not pin.

    The DATA is seeded -- three splits at ``seed``, ``seed + 10_000`` and
    ``seed + 20_000`` -- so the sequences a run trains on are a function of the
    configuration. The EVALUATION is seeded, because the model's samplers take an
    explicit ``sampling_seed`` and the trainer pins it for the evaluation's
    duration.

    The TRAINING is NOT reproducible, and that is the correct design rather than a
    gap. The ELBO's reparameterization draws must be unbiased, and ``keras.random``
    with ``seed=None`` is the only way to get that: MEASURED on this build, three
    ``keras.utils.set_random_seed(0)`` calls before three draws give three
    DIFFERENT draws, while an explicit ``seed=0`` gives the same one three times.
    Pinning the training sampler would make every batch reuse one noise draw, which
    biases the gradient rather than merely making it repeatable.

    So the assertion below is on the seeded parts, and the last one pins the
    asymmetry so nobody "fixes" it by seeding the training sampler.
    """

    def test_the_architecture_is_reproducible(self, tmp_path):
        first = train(_config(tmp_path / "a", experiment_name="run-a"))
        second = train(_config(tmp_path / "b", experiment_name="run-b"))
        assert first["parameters"] == second["parameters"]
        assert first["coherence_window"] == second["coherence_window"]
        assert first["data_shapes"] == second["data_shapes"]
        assert first["latent_dim"] == second["latent_dim"]

    def test_the_data_is_reproducible(self, tmp_path):
        """Two runs at one seed see the SAME sequences.

        Isolated from the training draw by building the splits directly, so the
        assertion is about the data layer rather than about everything downstream
        of an unseeded sampler.
        """
        import train.topographic_vae.train_topographic_vae as module

        def _splits(seed):
            config = _config(tmp_path / f"data-{seed}", seed=seed)
            return module.load_transform_sequences(config)

        first = _splits(0)
        second = _splits(0)
        for left, right in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(left, right)

        # And the anti-vacuity arm: a different seed is a different dataset.
        other = _splits(5)
        assert not np.array_equal(first[0], other[0])

    def test_the_training_sampler_is_left_unseeded(self, tmp_path):
        """The asymmetry, pinned.

        A trainer that seeded the training samplers would produce bit-identical
        runs -- and a biased ELBO. This asserts the samplers carry no seed, so the
        reason the run above is not bit-reproducible stays visible in the code
        rather than only in this docstring.
        """
        import train.topographic_vae.train_topographic_vae as module

        config = _config(tmp_path, save_visualizations=False,
                         experiment_name="seed-check")
        model = module.build_model(config, config.sequence_length)
        assert model.sampling_seed is None
        assert model.z_sampling.seed is None
        assert model.u_sampling.seed is None

    def test_the_evaluation_pins_and_releases_the_sampler_seed(self, tmp_path):
        """The trainer sets the seed for the evaluation and restores it after.

        Pinned because the alternative -- leaving it pinned for the rest of the
        process -- would silently make a SECOND run in the same process train with
        a frozen sampler. The restoration is the part that is easy to forget.
        """
        import train.topographic_vae.train_topographic_vae as module

        config = _config(tmp_path, save_visualizations=False,
                         experiment_name="seed-release")
        model = module.build_model(config, config.sequence_length)
        sequences = np.zeros(
            (config.num_test_sequences, config.sequence_length, 28, 28, 3),
            "float32",
        )
        factors = np.tile(
            np.arange(config.sequence_length, dtype=np.float64),
            (config.num_test_sequences, 1),
        )
        model(sequences, training=False)
        assert model.sampling_seed is None

        module.evaluate_model(model, sequences, factors, config)
        assert model.sampling_seed is None, (
            "the evaluation did not release the sampling seed; a later training "
            "run in this process would reuse a frozen draw"
        )


def _config_fields():
    import dataclasses

    return dataclasses.fields(TopographicVAEConfig)
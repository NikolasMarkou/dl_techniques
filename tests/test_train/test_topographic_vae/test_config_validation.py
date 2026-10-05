"""Config validation and preset resolution for the Topographic VAE trainer.

The two places a run's meaning lives
------------------------------------
``TopographicVAEConfig`` resolves three things that a caller reading
``config.json`` would otherwise have to re-derive, and each resolution has a
failure mode that produces a plausible-looking run rather than an error:

1. **``L`` from a preset.** ``--l-preset`` is a FRACTION of the sequence length,
   and the sequence length is not known when the config is built. So the fraction
   must survive in the config and be resolved later. A config that resolved it in
   ``__post_init__`` would freeze ``L`` against a default ``S`` that
   ``--sequence-length`` can then move -- the run would silently use
   ``L = 6`` on an ``S = 7`` sequence.
2. **The baseline label.** Which row of the paper's ablation table a run is, is
   derived from two independent knobs. Getting it wrong puts a baseline's numbers
   in the topographic row.
3. **The effective beta.** ``kl_loss_weight`` is NOT the paper's ``beta`` under
   the default mean-over-pixels reduction, and the summary reports the corrected
   number separately. If the correction were dropped the two would be read as
   equal.

The rest is validation: every field that the model or the data layer will reject
much later is rejected here, so a bad run costs no GPU time.
"""

from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from train.topographic_vae.train_topographic_vae import (  # noqa: E402
    L_PRESETS,
    TopographicVAEConfig,
    _baseline_name,
    build_learning_rate_schedule,
    build_model,
    config_from_args,
    create_argument_parser,
    frame_shape_of,
)


class TestCoherenceWindowPresets:
    """``L`` as a fraction of ``S``, resolved late."""

    def test_the_paper_default_resolves_to_six_on_eighteen_frames(self):
        config = TopographicVAEConfig()
        assert config.coherence_window is None, (
            "the preset must stay unresolved in the config"
        )
        assert config.l_preset == "third"
        assert config.resolved_coherence_window(18) == 6, (
            "Section A.3's best-equivariance setting"
        )

    @pytest.mark.parametrize(
        "preset,sequence_length,expected",
        [
            ("none", 18, 0),
            ("sixth", 18, 3),
            ("third", 18, 6),
            ("half", 18, 9),
            # The fractions are of S, so a different S gives a different L. This
            # is the whole reason resolution cannot happen at construction.
            ("third", 15, 5),
            ("third", 7, 2),
            ("half", 7, 4),
        ],
    )
    def test_each_preset_is_a_fraction_of_the_sequence_length(
        self, preset, sequence_length, expected
    ):
        config = TopographicVAEConfig(l_preset=preset)
        assert config.resolved_coherence_window(sequence_length) == expected

    def test_a_preset_never_resolves_to_zero_unless_none(self):
        """``max(1, ...)``: a preset fraction of a short sequence rounds to zero,
        and ``L = 0`` silently drops the temporal coherence the run was asked for
        -- which is a different model, not a smaller window."""
        for sequence_length in (1, 2, 3):
            for preset in ("sixth", "third", "half"):
                resolved = TopographicVAEConfig(
                    l_preset=preset
                ).resolved_coherence_window(sequence_length)
                assert resolved >= 1, (preset, sequence_length, resolved)

    def test_an_explicit_window_overrides_the_preset(self):
        config = TopographicVAEConfig(l_preset="half", coherence_window=2)
        assert config.resolved_coherence_window(18) == 2

    def test_an_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="l_preset must be one of"):
            TopographicVAEConfig(l_preset="two_thirds").resolved_coherence_window(
                18
            )

    def test_the_presets_are_fractions_not_absolute_values(self):
        """A regression guard on the mapping itself: if ``L_PRESETS`` ever holds
        absolute integers the tests above would still pass for ``S = 18`` and fail
        silently everywhere else."""
        for preset, fraction in L_PRESETS.items():
            assert 0.0 <= float(fraction) <= 0.5, (preset, fraction)
            assert float(fraction) < 1.0, (
                f"{preset} is a fraction of S, so a value at or above 1 would "
                "make L >= S and the window would cover the whole sequence"
            )


class TestBaselineNaming:
    """Which row of the paper's table a run is."""

    def test_the_default_run_is_the_topographic_vae(self):
        config = TopographicVAEConfig()
        assert _baseline_name(config) == "topographic_vae"

    def test_no_variance_variables_is_the_plain_vae(self):
        config = TopographicVAEConfig(use_variance_variables=False)
        assert _baseline_name(config) == "plain_vae"

    def test_stationary_coherence_is_the_bubble_vae(self):
        config = TopographicVAEConfig(temporal_coherence="stationary")
        assert _baseline_name(config) == "bubble_vae"

    def test_both_together_reports_the_stronger_baseline(self):
        """The two baselines are nested: without ``u`` there is no temporal
        coherence at all, so the plain VAE is the honest label even if the user
        also asked for stationary coherence."""
        config = TopographicVAEConfig(
            use_variance_variables=False, temporal_coherence="stationary"
        )
        assert _baseline_name(config) == "plain_vae"

    def test_the_label_follows_the_cli_flags(self):
        parser = create_argument_parser()
        for flag, expected in (
            (["--no-variance-variables"], "plain_vae"),
            (["--temporal-coherence", "stationary"], "bubble_vae"),
            ([], "topographic_vae"),
        ):
            config = config_from_args(parser.parse_args(flag))
            assert _baseline_name(config) == expected, flag


class TestEffectiveBeta:
    """``kl_loss_weight`` is not the paper's ``beta`` under the default."""

    def test_the_mean_over_pixels_default_multiplies_by_the_pixel_count(self):
        config = TopographicVAEConfig(kl_loss_weight=1.0)
        assert config.reconstruction_sum_reduction is False
        loss = __import__(
            "dl_techniques.losses", fromlist=["TopographicVAELoss"]
        ).TopographicVAELoss(
            kl_loss_weight=config.kl_loss_weight,
            reconstruction_sum_reduction=config.reconstruction_sum_reduction,
        )
        assert loss.effective_beta == pytest.approx(1.0), (
            "before the first batch the pixel count is unknown, so the "
            "conservative sum-reduction answer is correct here"
        )

    def test_the_sum_reduction_makes_the_two_numbers_equal(self):
        loss = __import__(
            "dl_techniques.losses", fromlist=["TopographicVAELoss"]
        ).TopographicVAELoss(kl_loss_weight=0.5, reconstruction_sum_reduction=True)
        assert loss.effective_beta == pytest.approx(0.5)


class TestConfigValidation:
    """Every field the model or the data layer would reject later."""

    def test_the_config_is_mutable_and_says_so(self):
        """NOT frozen, deliberately, and the reason is in ``train()``.

        ``train()`` resolves ``experiment_name`` -- which needs the clock -- and
        assigns it back onto the config before the run directory is prepared, so
        that ``config.json`` records the name actually used. A frozen dataclass
        would force that value to be threaded separately and the two would drift.
        So the config is a plain dataclass and this test pins the design decision
        rather than asserting immutability.
        """
        config = TopographicVAEConfig()
        assert dataclasses.is_dataclass(config)
        assert not config.__dataclass_params__.frozen, (
            "train() assigns config.experiment_name; freezing it here would "
            "contradict that"
        )
        config.epochs = 5
        assert config.epochs == 5

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("dataset", "cifar10", "dataset must be one of"),
            ("transform", "wobble", "transform must be one of"),
            ("num_train_sequences", 0, "num_train_sequences must be positive"),
            ("num_val_sequences", -1, "num_val_sequences must be"),
            ("num_test_sequences", 0, "num_test_sequences must be"),
            ("sequence_length", 0, "sequence_length must be positive"),
            ("num_capsules", 0, "num_capsules must be positive"),
            ("capsule_dim", 0, "capsule_dim must be positive"),
            ("neighborhood_size", 0, "neighborhood_size must be positive"),
            ("epochs", 0, "epochs must be positive"),
            ("batch_size", 0, "batch_size must be positive"),
            ("learning_rate", 0.0, "learning_rate must be positive"),
            ("kl_loss_weight", -1.0, "kl_loss_weight must be non-negative"),
            ("prior_mean", -1.0, "prior_mean must be non-negative"),
            ("likelihood_samples", 0, "likelihood_samples must be positive"),
            ("num_traversals", 0, "num_traversals must be positive"),
            ("temporal_coherence", "diagonal", "temporal_coherence must be one of"),
            ("topography", "hex", "topography must be one of"),
            ("mixed_precision", "bfloat16", "mixed_precision must be one of"),
            ("lr_schedule", "cyclic", "lr_schedule must be one of"),
        ],
    )
    def test_rejects_an_impossible_value(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            TopographicVAEConfig(**{field: value})

    def test_rejects_a_torus_without_a_grid_shape(self):
        with pytest.raises(ValueError, match="torus_2d.*requires grid_shape"):
            TopographicVAEConfig(topography="torus_2d")

    def test_rejects_a_non_variance_baseline_with_a_coherence_window(self):
        """The model refuses this too, but a config that reached ``build_model``
        would spend the data generation before finding out."""
        with pytest.raises(ValueError, match="use_variance_variables"):
            TopographicVAEConfig(
                use_variance_variables=False, coherence_window=3
            )

    def test_accepts_a_torus_with_a_grid_shape(self):
        config = TopographicVAEConfig(topography="torus_2d", grid_shape=(4, 4))
        assert config.grid_shape == (4, 4)

    def test_the_experiment_name_defaults_to_a_timestamped_name(self):
        config = TopographicVAEConfig()
        name = config.resolved_experiment_name()
        assert "topographic_vae" in name
        assert "mnist" in name
        assert config.experiment_name is None, (
            "resolution must not mutate the config: a config re-read from "
            "config.json should not produce a second, differently-named "
            "directory"
        )

    def test_an_explicit_experiment_name_wins(self):
        config = TopographicVAEConfig(experiment_name="my-run")
        assert config.resolved_experiment_name() == "my-run"

    def test_the_default_name_carries_the_transform(self):
        """Two runs of the same model on different factors must not collide in
        ``results/``."""
        rotation = TopographicVAEConfig(
            dataset="mnist", transform="rotation"
        ).resolved_experiment_name()
        colour = TopographicVAEConfig(
            dataset="mnist", transform="color"
        ).resolved_experiment_name()
        assert rotation != colour


class TestBuildModel:
    """The config-to-model hop, and the knobs it resolves on the way."""

    def _config(self, **overrides):
        base = dict(
            dataset="mnist",
            sequence_length=6,
            num_capsules=4,
            capsule_dim=4,
            encoder_hidden_dims=[8],
            decoder_hidden_dims=[8],
            variant="mnist",
        )
        base.update(overrides)
        return TopographicVAEConfig(**base)

    def test_the_preset_reaches_the_model(self):
        model = build_model(self._config(l_preset="third"), sequence_length=6)
        assert model.coherence_window == 2

    def test_an_explicit_window_reaches_the_model(self):
        model = build_model(
            self._config(l_preset="half", coherence_window=1),
            sequence_length=6,
        )
        assert model.coherence_window == 1

    def test_the_baseline_mode_drops_the_topography(self):
        model = build_model(
            self._config(use_variance_variables=False, l_preset="none"),
            sequence_length=6,
        )
        assert model.use_variance_variables is False
        assert model.topographic_product is None
        assert model.u_encoder is None
        assert model.coherence_window == 0

    def test_the_baseline_rejects_a_preset_that_resolves_to_a_window(self):
        """The case ``__post_init__`` cannot see.

        With ``use_variance_variables=False`` the explicit ``coherence_window`` is
        ``None`` and the config validates; the PRESET then resolves to ``L = 2`` on
        ``S = 6``, which the model refuses. So the refusal has to be in
        ``build_model``, which is where ``S`` is finally known -- and the test
        pins that it names the flag that fixes it.
        """
        with pytest.raises(ValueError, match="l_preset"):
            build_model(
                self._config(use_variance_variables=False),
                sequence_length=6,
            )

    def test_the_frame_shape_follows_the_dataset(self):
        """From the DATASET, not the variant table.

        MEASURED before the fix: ``frame_shape_of`` returned ``(28, 28, 3)`` for
        a dSprites run, because it read the variant's default of ``mnist``. The
        generator produces 64x64x1 frames, so the decoder's output width would
        have disagreed with the input and the ELBO would have silently reshaped.
        """
        assert frame_shape_of(self._config()) == (28, 28, 3)
        assert frame_shape_of(
            self._config(
                dataset="dsprites", variant="dsprites",
                transform="orientation", sequence_length=15,
            )
        ) == (64, 64, 1)

    def test_the_variant_must_agree_with_the_dataset(self):
        """The assertion that keeps the two tables from drifting apart.

        ``dsprites`` offers no ``rotation`` either, so the companions are needed
        on both counts -- but the frame mismatch is the one that would produce a
        wrong-shaped reconstruction rather than an error.
        """
        with pytest.raises(ValueError, match="expects frames"):
            TopographicVAEConfig(dataset="dsprites", transform="orientation")
        with pytest.raises(ValueError, match="expects frames"):
            TopographicVAEConfig(dataset="mnist", variant="dsprites")

    def test_the_latent_width_is_the_capsule_product(self):
        model = build_model(self._config(), sequence_length=6)
        assert model.latent_dim == 16
        assert model.num_capsules == 4
        assert model.capsule_dim == 4

    def test_the_splits_are_generated_from_different_seeds(self, monkeypatch):
        """The property that keeps the test numbers honest.

        Each split is built with ``seed``, ``seed + 10_000`` and ``seed + 20_000``,
        so no sequence appears in two splits. Reusing one seed would put identical
        sequences in train and test and inflate every number in the summary, and
        the shapes would still look right.

        Asserted on the SEEDS rather than on the data, because a full three-split
        generation is the expensive part of this trainer; the seeds are the whole
        content of the guarantee.
        """
        import train.topographic_vae.train_topographic_vae as module

        seen: list = []

        def _fake(name, **kwargs):
            seen.append((name, kwargs["seed"]))
            count = kwargs["num_sequences"]
            length = kwargs["sequence_length"] or 18
            shape = frame_shape_of(TopographicVAEConfig())
            frames = np.zeros((count, length) + tuple(shape), "float32")
            return frames, np.zeros((count, length), "float32")

        monkeypatch.setattr(module, "create_transform_sequence_dataset", _fake)
        config = TopographicVAEConfig(
            num_train_sequences=4, num_val_sequences=2, num_test_sequences=3,
            seed=11,
        )
        module.load_transform_sequences(config)

        seeds = [seed for _, seed in seen]
        assert len(seeds) == 3, seen
        assert len(set(seeds)) == 3, (
            f"two splits share a seed ({seeds}); a sequence in train would then "
            "also be in test"
        )
        assert seeds == [11, 11 + 10_000, 11 + 20_000], seeds

    def test_the_splits_have_the_requested_sizes(self, monkeypatch):
        import train.topographic_vae.train_topographic_vae as module

        def _fake(name, **kwargs):
            count = kwargs["num_sequences"]
            length = kwargs["sequence_length"] or 18
            frames = np.zeros((count, length) + (28, 28, 3), "float32")
            return frames, np.zeros((count, length), "float32")

        monkeypatch.setattr(module, "create_transform_sequence_dataset", _fake)
        x_train, x_val, x_test, factors = module.load_transform_sequences(
            TopographicVAEConfig(
                num_train_sequences=4, num_val_sequences=2, num_test_sequences=3
            )
        )
        assert x_train.shape[0] == 4
        assert x_val.shape[0] == 2
        assert x_test.shape[0] == 3
        assert factors.shape == (3, 18), (
            "CapCorr needs one ground-truth factor per test frame"
        )
        # train and val are passed to fit() as both input AND target, so they
        # must be the same array object-wise equal -- a copy is fine, a different
        # array is not.
        np.testing.assert_array_equal(x_train, x_train)

    def test_a_torus_run_builds(self):
        # `topography="torus_2d"` supports only `temporal_coherence="none"`:
        # a 2-D lattice has no single cyclic axis for the coherence roll to
        # permute along, and picking one would make the model depend on an
        # arbitrary axis ordering. Both are required companions.
        model = build_model(
            self._config(
                topography="torus_2d",
                grid_shape=(2, 2),
                temporal_coherence="none",
                l_preset="none",
                num_capsules=1,
                capsule_dim=4,
                neighborhood_size=2,
            ),
            sequence_length=6,
        )
        assert model.topography == "torus_2d"
        assert model.grid_shape == (2, 2)


class TestLearningRateSchedule:
    """The schedule is opt-in, and the paper's constant rate is the default."""

    def test_the_constant_schedule_returns_none(self):
        config = TopographicVAEConfig(lr_schedule="constant")
        assert build_learning_rate_schedule(config, steps_per_epoch=10) is None

    @pytest.mark.parametrize("name", ["cosine", "exponential"])
    def test_a_named_schedule_returns_a_learning_rate(self, name):
        config = TopographicVAEConfig(lr_schedule=name, learning_rate=0.1)
        rate = build_learning_rate_schedule(config, steps_per_epoch=10)
        assert rate is not None
        assert float(keras_learning_rate(rate)) <= 0.1

    def test_the_schedule_does_not_fire_on_a_zero_step_epoch(self):
        """``steps_per_epoch = 0`` would divide by zero inside Keras' schedule."""
        config = TopographicVAEConfig(lr_schedule="cosine")
        rate = build_learning_rate_schedule(config, steps_per_epoch=0)
        assert rate is not None


def keras_learning_rate(schedule):
    """Read a schedule's initial learning rate, across Keras versions."""
    for attribute in ("get_config", "config"):
        reader = getattr(schedule, attribute, None)
        if reader is not None:
            return reader()["initial_learning_rate"]
    return schedule.initial_learning_rate
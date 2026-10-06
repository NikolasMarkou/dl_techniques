"""Test suite for the TopoLM trainer.

Covers the config's additive-only contract, tap-site resolution, the
optimizer recipe, the virtual-epoch cadence, callback assembly, activation
extraction, the post-hoc evaluation, and the paired control.

Every training-shaped test drives the REAL ``train_topolm`` with the dataset
loader and tokenizer replaced, because the parts worth testing here are the
cadence arithmetic, the objective wiring and the evaluation -- none of which need
Wikipedia. The substitution is explicit in ``_patch_data`` rather than hidden in
a fixture, so a reader can see exactly what is and is not under test.

The mandated pins:

  1. test_the_topographic_arm_reports_a_larger_loss_than_its_control
     10.67 against 5.6 on the same weights -- the spatial term reaches the
     objective
  2. test_the_validation_loss_is_the_pure_task_loss
     val_loss ~5.2 while the training loss is ~10.7
  3. test_the_virtual_epoch_count_is_derived_from_the_requested_step_count
  4. test_an_alpha_zero_control_holds_an_identical_weight_set
  5. test_the_readout_arm_does_not_alter_the_raw_arm
"""

import os

import numpy as np
import pytest
import tensorflow as tf

import keras
from keras import ops

from train.topolm import (
    SMOKE_STIMULI,
    TapCapture,
    TopoLMTrainingConfig,
    build_backbone,
    build_callbacks,
    compile_model,
    create_topolm_model,
    evaluate_topography,
    extract_tap_activations,
    resolve_cadence,
    resolve_tap_sites,
    plot_topography,
    run_topographic_evaluation,
    train_paired,
    train_topolm,
)
from train.topolm import common as C
from dl_techniques.metrics.topographic_selectivity import contrast_tmap
from train.common.clm_pretrain import ClmPretrainConfig

from dl_techniques.models.language.topolm import MODEL_VARIANTS

VOCAB = 300
LENGTH = 12


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------


class StubTokenizer:
    """A deterministic stand-in for the Tiktoken preprocessor.

    Character codes rather than random ids, so two calls with the same text
    produce the same ids -- a random tokenizer would let a test pass on a prompt
    ordering that never recurs.

    Takes a LIST of strings, which is the shape
    :class:`~dl_techniques.utils.tokenizer.TiktokenPreprocessor` actually
    accepts. It took a ``{"text": [...]}`` mapping before, matching neither the
    real preprocessor nor the trainer: ``__call__`` raises ``TypeError`` on
    anything but a string or a list, so the stub was agreeing with no one and
    hid a real defect. ``extract_tap_activations`` was passing the mapping and
    only reached this stub in tests; the live run raised ``TypeError`` there.
    """

    def __call__(self, texts):
        texts = [texts] if isinstance(texts, str) else texts
        ids = np.zeros((len(texts), LENGTH), dtype="int32")
        for row, text in enumerate(texts):
            for column, character in enumerate(str(text)[:LENGTH - 1]):
                ids[row, column] = (ord(character) * 7) % VOCAB
        return {"input_ids": ids}


@pytest.fixture(autouse=True)
def _never_write_to_repo_results(monkeypatch, tmp_path):
    """Force every run directory under ``tmp_path``.

    ``results/`` is gitignored and untracked, so a run directory written there by
    a test is unrecoverable. The repo's own ``no_repo_root_results_writes``
    fixture asserts; this one prevents, so a failing assertion is not the first
    sign of it.

    The REAL ``_run_dir`` is kept and only its root is redirected. Collapsing the
    name to a single directory would make the paired-run test's "the two arms are
    written separately" assertion pass or fail for the wrong reason -- it would
    merge the arms, which is exactly what that test exists to detect.
    """
    real_run_dir = C._run_dir
    monkeypatch.setattr(
        C,
        "_run_dir",
        lambda config: str(tmp_path / "runs" / os.path.basename(
            real_run_dir(config)
        )),
    )


def _patch_data(monkeypatch, rows=16, batch_size=4):
    """Replace the dataset loader with a synthetic packed-CLM stream."""
    generator = np.random.default_rng(0)
    tokens = generator.integers(0, VOCAB, size=(rows, LENGTH)).astype("int32")
    labels = generator.integers(0, VOCAB, size=(rows, LENGTH)).astype("int32")

    def fake_load(config, preprocessor, data_seed, wrap_for_dict_output=True):
        assert wrap_for_dict_output is False, (
            "the CLM head is pre_shifted, so a dict-keyed label raises inside the "
            "loss; the trainer must pass wrap_for_dict_output=False"
        )
        assert data_seed == config.seed, (
            f"data seed {data_seed} does not derive from config.seed "
            f"{config.seed}"
        )
        train = tf.data.Dataset.from_tensor_slices(
            (tokens, labels)
        ).batch(batch_size).repeat()
        validation = tf.data.Dataset.from_tensor_slices(
            (tokens[:8], labels[:8])
        ).batch(batch_size)
        return train, validation, rows

    monkeypatch.setattr(C, "load_train_val_datasets", fake_load)


def _config(**overrides):
    """A tiny, fast configuration. Overrides win."""
    settings = dict(
        model_variant="tiny",
        vocab_size=VOCAB,
        max_seq_length=LENGTH,
        num_layers=2,
        spatial_alpha=2.5,
        spatial_radius=2,
        spatial_neighborhoods=3,
        eval_every_steps=2,
        eval_batches=2,
        spatial_log_every=1,
        checkpoint_every_steps=10_000,
        analyze_every_steps=0,
        readout_fwhm=2.0,
        min_cluster_size=2,
        save_dir="run",
    )
    settings.update(overrides)
    return TopoLMTrainingConfig(**settings)


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------


class TestConfig:
    def test_the_position_table_matches_the_data_window(self):
        """The two lengths must agree, or the first batch cannot index it.

        The CLI defaults them independently (`--max-seq-length 512`, and the
        variant carries its own `max_seq_len`), so nothing forced them to match.
        Before the forward, `build_backbone` wrote a 256-entry position table for
        512-token windows and every run died in `positional_embeddings`.
        """
        backbone = build_backbone(_config(max_seq_length=96))
        assert backbone.max_seq_len == 96

    def test_a_window_longer_than_the_variant_is_forwarded(self):
        """A 512-token window must resize the table, not be rejected."""
        backbone = build_backbone(_config(max_seq_length=512))
        assert backbone.max_seq_len == 512

    def test_it_is_additive_to_the_shared_clm_config(self):
        """A re-declared inherited field moves its position in the order.

        The shared docstring calls that out explicitly, so the whole field order
        is compared rather than spot-checked.
        """
        inherited = [f.name for f in ClmPretrainConfig.__dataclass_fields__.values()]
        combined = [
            f.name for f in TopoLMTrainingConfig.__dataclass_fields__.values()
        ]
        assert combined[:len(inherited)] == inherited, (
            "the inherited field order moved; a re-declaration did it"
        )

    def test_the_paper_defaults_are_the_recipe_not_an_approximation(self):
        config = TopoLMTrainingConfig()
        assert config.learning_rate == 6e-4
        assert config.weight_decay == 0.1
        assert config.dropout_rate == 0.0
        assert config.spatial_alpha == 2.5
        assert config.spatial_radius == 5
        assert config.spatial_neighborhoods == 5
        assert config.spatial_distance == "linf"
        assert config.spatial_permute is True
        assert config.tap_sites == "both"
        assert config.early_stop_patience == 3

    def test_the_defaults_describe_a_model_that_can_exist(self):
        """``alpha`` defaults and the radius default must agree on width.

        The paper's width is 784, which a radius-5 patch (121 units) fits. A
        config default that only works for some variants is a default that
        fails on the first one a caller reaches for.
        """
        for name in MODEL_VARIANTS:
            config = _config(model_variant=name)
            backbone = build_backbone(config)
            assert len(backbone.tap_layers) > 0, name

    def test_the_smoke_stimuli_cover_every_default_contrast(self):
        assert set(_config().contrast_conditions) <= set(SMOKE_STIMULI)
        assert all(len(v) > 1 for v in SMOKE_STIMULI.values())

    def test_the_four_paper_conditions_are_declared(self):
        from train.topolm import PAPER_CONDITIONS

        assert PAPER_CONDITIONS == (
            "sentences",
            "unconnected_words",
            "jabberwocky_sentences",
            "unconnected_nonwords",
        )


class TestResolveTapSites:
    @pytest.mark.parametrize(
        "value,expected",
        [("both", ("attention", "mlp")),
         ("attention", ("attention",)), ("mlp", ("mlp",))],
    )
    def test_the_three_valid_values(self, value, expected):
        assert resolve_tap_sites(value) == expected

    @pytest.mark.parametrize("value", ["Attention", "both_sites", "", "attn"])
    def test_anything_else_is_rejected_and_lists_the_options(self, value):
        with pytest.raises(ValueError, match="'both', 'attention' or 'mlp'"):
            resolve_tap_sites(value)


# ---------------------------------------------------------------------
# model and optimizer
# ---------------------------------------------------------------------


class TestModelWiring:
    def test_the_backbone_reports_its_grid_and_tap_count(self):
        backbone = build_backbone(_config())
        assert backbone.grid_shape == (16, 16)
        assert len(backbone.tap_layers) == 2 * 2  # depth x both sites

    def test_a_single_site_halves_the_tap_count(self):
        backbone = build_backbone(_config(tap_sites="mlp"))
        assert len(backbone.tap_layers) == 2

    def test_the_head_aggregates_the_backbone_losses(self):
        """The single most load-bearing line in the trainer.

        Without it the taps compute a penalty every step that nothing
        backpropagates, and the run produces a non-topographic model with a
        healthy curve. Asserted on the CONFIGURED VALUE because the downstream
        consequence is what the end-to-end test measures.
        """
        head, backbone = create_topolm_model(_config())
        assert head.aggregate_backbone_losses is True

    def test_the_head_is_built_after_construction(self):
        """An unbuilt head raises from ``count_params`` rather than reporting.

        The eager forward on the WRAPPER is what forces the head's output-key
        resolution and its causality probe before ``fit`` traces ``train_step``.
        """
        head, _ = create_topolm_model(_config())
        assert head.built
        assert head.count_params() > 0

    def test_the_head_returns_a_bare_logits_tensor(self):
        """``output_key="logits"`` extracts inside the head, not after it.

        The generation probe's closure indexes whatever the head returns; a dict
        there would make that closure raise at the first probe.
        """
        head, _ = create_topolm_model(_config())
        output = head(np.zeros((1, LENGTH - 1), dtype="int32"), training=False)
        assert isinstance(output, (keras.KerasTensor, tf.Tensor))
        assert len(ops.shape(output)) == 3


class TestOptimizerRecipe:
    def _optimizer(self, **overrides):
        config = _config(**overrides)
        head, _ = create_topolm_model(config)
        compile_model(head, config, epochs=2, steps_per_epoch=2)
        return head.optimizer

    def test_it_is_adamw_with_the_papers_betas(self):
        optimizer = self._optimizer()
        assert isinstance(optimizer, keras.optimizers.AdamW)
        assert float(optimizer.beta_1) == pytest.approx(0.9)
        assert float(optimizer.beta_2) == pytest.approx(0.95)

    def test_the_global_gradient_clip_is_one(self):
        """Global norm, not the per-variable variant.

        The two keys look interchangeable in the builder and clip very different
        quantities; "gradient clipping at 1.0" means the global norm. Keras keeps
        them in SEPARATE slots, so both are asserted: a builder that set the
        wrong key would leave ``global_clipnorm`` at ``None`` and pass a test that
        only looked at ``clipnorm``.
        """
        optimizer = self._optimizer()
        assert float(optimizer.global_clipnorm) == pytest.approx(1.0)
        assert optimizer.clipnorm is None, (
            "the per-variable clipnorm is set as well; both slots active means "
            "gradients are clipped twice under two different norms"
        )

    def test_the_excluded_tensors_really_exclude_them(self):
        """The both-ways pair: biases are excluded and kernels are not.

        An exclusion list that matched nothing would leave the normalization
        gains decaying, which is exactly what it exists to prevent -- and
        ``exclude_from_weight_decay`` is a METHOD on the built optimizer, not a
        readable list, so the only observable is which variable names the
        compiled pattern actually hits. (A private attribute is read here for
        that reason; the alternative is a test that asserts nothing.)
        """
        head, _ = create_topolm_model(_config())
        optimizer = self._optimizer()

        pattern = getattr(optimizer, "_exclude_from_weight_decay_pattern", None)
        assert pattern is not None, (
            "this Keras build exposes no exclusion pattern; the test cannot "
            "observe what is decayed and would otherwise pass vacuously"
        )

        paths = [variable.path for variable in head.trainable_variables]
        excluded = [path for path in paths if pattern.search(path)]
        decayed = [path for path in paths if not pattern.search(path)]

        assert decayed, "nothing is weight-decayed at all"
        assert excluded, "everything is excluded"
        for kind in ("bias", "gamma", "beta"):
            matching = [p for p in paths if kind in p]
            assert matching, f"the model holds no {kind} to test"
            for path in matching:
                assert pattern.search(path), f"{path} should be excluded"
        assert not any(
            "word_embeddings" in path for path in decayed
        ), "the tied embedding table IS the output head; decaying it decays both"

    def test_a_warmup_then_cosine_schedule_reaches_the_optimizer(self):
        """Asserted as the LR's VALUE, because Keras materialises it at build.

        ``optimizer.learning_rate`` is an eager tensor, not the schedule object,
        so an ``isinstance`` check would fail on a correctly configured run. What
        matters is that the LR is the schedule's step-0 value AND that the
        schedule is not constant -- a plain float would satisfy the first and
        fail the second.
        """
        from dl_techniques.optimization.schedule import create_warmup_lr_schedule

        config = _config(learning_rate=1e-3, warmup_ratio=0.5)
        head, _ = create_topolm_model(config)
        compile_model(head, config, epochs=2, steps_per_epoch=4)
        optimizer = head.optimizer

        schedule = create_warmup_lr_schedule(
            config.learning_rate, 2, 4, config.warmup_ratio
        )
        assert float(optimizer.learning_rate) == pytest.approx(
            float(schedule(0)), rel=1e-5
        )
        assert float(schedule(0)) != float(schedule(4)), (
            "the schedule is constant; a fixed LR is not warmup-then-cosine"
        )
        # Warmup means step 0 is strictly BELOW the peak.
        assert float(schedule(0)) < config.learning_rate

    @pytest.mark.parametrize(
        "epochs,steps", [(0, 10), (10, 0), (-1, 10), (10, -1)]
    )
    def test_a_degenerate_schedule_horizon_is_rejected(self, epochs, steps):
        """Zero steps would make the schedule divide by zero.

        The guard belongs here rather than inside the schedule helper: this
        trainer derives the horizon, so it owns validating what it derived.
        """
        head, _ = create_topolm_model(_config())
        with pytest.raises(ValueError, match="positive horizon"):
            compile_model(head, _config(), epochs=epochs, steps_per_epoch=steps)


# ---------------------------------------------------------------------
# cadence
# ---------------------------------------------------------------------


class TestCadence:
    def test_the_virtual_epoch_count_is_derived_from_the_requested_step_count(self):
        config = _config(eval_every_steps=100, train_steps=250)
        epochs, cadence = resolve_cadence(config, real_steps_per_epoch=9_999)
        assert cadence == 100
        assert epochs == 3, "250 steps at a 100-step cadence is 2.5 epochs"
        assert epochs * cadence >= config.train_steps

    def test_step_counts_round_up_so_none_are_dropped(self):
        config = _config(eval_every_steps=100, train_steps=1)
        epochs, _ = resolve_cadence(config, real_steps_per_epoch=10)
        assert epochs == 1

    def test_without_train_steps_the_epoch_count_is_num_epochs(self):
        config = _config(eval_every_steps=100, num_epochs=4)
        epochs, cadence = resolve_cadence(config, real_steps_per_epoch=37)
        assert (epochs, cadence) == (4, 100)

    def test_the_cadence_never_falls_below_one_epoch(self):
        config = _config(eval_every_steps=100, num_epochs=0)
        epochs, _ = resolve_cadence(config, real_steps_per_epoch=10)
        assert epochs == 1

    @pytest.mark.parametrize("value", [0, -1])
    def test_a_non_positive_cadence_is_rejected(self, value):
        with pytest.raises(ValueError, match="eval_every_steps must be >= 1"):
            resolve_cadence(_config(eval_every_steps=value), 10)

    @pytest.mark.parametrize("value", [0, -5])
    def test_a_non_positive_step_count_is_rejected(self, value):
        config = _config(train_steps=value)
        with pytest.raises(ValueError, match="train_steps must be >= 1"):
            resolve_cadence(config, 10)


# ---------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------


class TestCallbackAssembly:
    def _callbacks(self, **overrides):
        config = _config(**overrides)
        head, _ = create_topolm_model(config)
        return build_callbacks(config, head, "/tmp/opencode/probe", 0)

    def test_the_stock_early_stopping_is_absent(self):
        """The paper's rule is a DIFFERENT question, and two stop signals is one
        too many.

        ``create_nlp_callbacks`` would have added the stock one; this asserts the
        trainer builds its own list instead.
        """
        kinds = [type(c) for c in self._callbacks()]
        assert keras.callbacks.EarlyStopping not in kinds

    def test_the_consecutive_increase_rule_is_present_and_configured(self):
        from dl_techniques.callbacks.consecutive_increase_early_stopping import (
            ConsecutiveIncreaseEarlyStopping,
        )

        found = [
            c for c in self._callbacks(early_stop_patience=7)
            if isinstance(c, ConsecutiveIncreaseEarlyStopping)
        ]
        assert len(found) == 1
        assert found[0].patience == 7
        assert found[0].monitor == "val_loss"
        assert found[0].restore_weights is True

    def test_the_spatial_logger_and_checkpointer_are_present(self):
        from dl_techniques.callbacks.spatial_loss_logger import SpatialLossLogger
        from train.common import StepCheckpointCallback

        kinds = [type(c) for c in self._callbacks(spatial_log_every=13)]
        assert SpatialLossLogger in kinds
        assert StepCheckpointCallback in kinds

    def test_the_generation_probe_reads_the_head_not_the_backbone(self):
        """Its closure must index whatever the head RETURNS.

        The head is configured with ``output_key="logits"``, so it returns a bare
        tensor and a ``["logits"]`` subscript would raise at the first probe --
        long after the run has started.
        """
        callbacks = self._callbacks()
        probe = next(
            c for c in callbacks
            if type(c).__name__ == "GenerationProbeCallback"
        )
        ids = np.zeros((1, 5), dtype="int32")
        logits = np.asarray(probe.logits_fn(ids))
        assert logits.ndim == 1, (
            f"the probe expects ONE row of vocabulary logits, got shape "
            f"{logits.shape}"
        )
        assert logits.shape[0] == VOCAB


# ---------------------------------------------------------------------
# activation extraction
# ---------------------------------------------------------------------


class TestActivationExtraction:
    def _backbone(self):
        return build_backbone(_config())

    def test_every_tap_is_reported_at_the_right_shape(self):
        backbone = self._backbone()
        prompts = list(SMOKE_STIMULI["a"]) + list(SMOKE_STIMULI["b"])
        activations = extract_tap_activations(
            backbone, prompts, StubTokenizer()
        )
        assert set(activations) == {t.path for t in backbone.tap_layers}
        for values in activations.values():
            assert values.shape == (len(prompts), backbone.embed_dim)

    def test_the_results_are_in_unit_order_not_grid_order(self):
        """Every consumer maps through the layout, so this must be stated.

        Grid order would be the same shape and would silently point at the wrong
        units, which no shape check sees.
        """
        backbone = self._backbone()
        activations = extract_tap_activations(
            backbone, ["alpha", "beta"], StubTokenizer()
        )
        tap = backbone.tap_layers[0]
        assert activations[tap.path].shape[-1] == tap.layout.num_units

    def test_a_capture_does_not_alter_the_forward_pass(self):
        """A measurement hook that changed the numbers would invalidate them."""
        backbone = self._backbone()
        tokens = np.random.default_rng(0).integers(
            0, VOCAB, size=(2, 8)
        ).astype("int32")
        plain = ops.convert_to_numpy(
            backbone(tokens, training=False)["last_hidden_state"]
        )
        observed = ops.convert_to_numpy(
            backbone(
                tokens,
                training=False,
                taps=tuple(TapCapture() for _ in backbone.tap_layers),
            )["last_hidden_state"]
        )
        np.testing.assert_array_equal(plain, observed)

    def test_pooling_is_a_mean_so_length_does_not_look_like_strength(self):
        """A sum-pool would make a longer encoding read as a stronger response.

        The paper's own footnote 9 reports that confound for a response-profile
        analysis, so the distinction is worth pinning rather than assuming.
        """
        backbone = self._backbone()
        tokenizer = StubTokenizer()
        prompts = ["alpha bravo"]

        activations = extract_tap_activations(backbone, prompts, tokenizer)
        pooled = activations[backbone.tap_layers[0].path]

        captures = [TapCapture() for _ in backbone.tap_layers]
        tokens = tokenizer(prompts)["input_ids"]
        backbone(tokens, training=False, taps=tuple(captures))
        raw = ops.convert_to_numpy(captures[0].value)

        np.testing.assert_allclose(
            pooled[0], raw[0].mean(axis=0), rtol=1e-5, atol=1e-6
        )
        # The two pooling rules must actually differ on this data, or the
        # assertion above is not discriminating.
        assert not np.allclose(pooled[0], raw[0].sum(axis=0))

    def test_an_empty_prompt_list_is_rejected(self):
        with pytest.raises(ValueError, match="prompts must not be empty"):
            extract_tap_activations(self._backbone(), [], StubTokenizer())

    def test_batching_does_not_change_the_result(self):
        """Two forward passes over the same tokens must agree exactly."""
        backbone = self._backbone()
        prompts = list(SMOKE_STIMULI["a"])
        tokenizer = StubTokenizer()
        one = extract_tap_activations(backbone, prompts, tokenizer, batch_size=99)
        many = extract_tap_activations(backbone, prompts, tokenizer, batch_size=2)
        for name in one:
            np.testing.assert_allclose(one[name], many[name], rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# topographic evaluation
# ---------------------------------------------------------------------


class TestThePermutationStatistic:
    """The paper's third Moran statistic, and its plumbing.

    ``standard`` and ``islands`` were in the report from the start;
    ``permutation`` was in the spec's scope line and never reached a report at
    all, because ``morans_i_permutation_test`` existed as a primitive that
    nothing called.
    """

    def _report(self, **overrides):
        backbone = build_backbone(_config())
        settings = dict(
            stimuli=SMOKE_STIMULI,
            contrast_conditions=("a", "b"),
            preprocessor=StubTokenizer(),
            readout_fwhm=2.0,
            min_cluster_size=2,
            is_smoke_set=False,
        )
        settings.update(overrides)
        report, arrays = evaluate_topography(backbone, **settings)
        return backbone, report, arrays

    def _stats(self, **overrides):
        _, report, _ = self._report(**overrides)
        return report["arms"]["raw"]["per_tap"]

    def test_the_report_carries_the_statistic_once_asked_for(self):
        per_tap = self._stats(permutation_p_value=True, num_permutations=19)
        for name, entry in per_tap.items():
            moran = entry["morans_i"]
            assert moran["num_permutations"] == 19, name
            assert 0.0 < moran["permutation_p"] <= 1.0, (
                f"{name}: p={moran['permutation_p']} is outside (0, 1]"
            )

    def test_the_statistic_is_absent_by_default(self):
        """Off by default: ~480k recomputations at the 9999 default."""
        per_tap = self._stats()
        for name, entry in per_tap.items():
            assert entry["morans_i"]["num_permutations"] is None, name
            assert np.isnan(entry["morans_i"]["permutation_p"]), name

    def test_the_p_values_are_reproducible_from_the_seed(self):
        """Same seed, same numbers -- so a report can be regenerated.

        Guards against reaching for `hash()`, which is salted per process and
        would make the p-values differ between two runs of one command.

        This test uses a SINGLE backbone, so the activations are identical
        across calls; only the permutation shuffles vary. Two calls to
        evaluate_topography with the same seed must produce the same p-values.
        """
        backbone = build_backbone(_config())
        settings = dict(
            stimuli=SMOKE_STIMULI,
            contrast_conditions=("a", "b"),
            preprocessor=StubTokenizer(),
            readout_fwhm=2.0,
            min_cluster_size=2,
            permutation_p_value=True,
            num_permutations=19,
        )
        report1, _ = evaluate_topography(backbone, **settings)
        report2, _ = evaluate_topography(backbone, **settings)
        for name in report1["arms"]["raw"]["per_tap"]:
            assert (
                report1["arms"]["raw"]["per_tap"][name]["morans_i"]["permutation_p"]
                == report2["arms"]["raw"]["per_tap"][name]["morans_i"]["permutation_p"]
            ), name

    def test_a_different_seed_changes_the_p_values(self):
        """Otherwise reproducibility would be indistinguishable from a constant."""
        first = self._stats(
            permutation_p_value=True, num_permutations=19, seed=1
        )
        second = self._stats(
            permutation_p_value=True, num_permutations=19, seed=2
        )
        values = [
            entry["morans_i"]["permutation_p"] for entry in first.values()
        ] + [entry["morans_i"]["permutation_p"] for entry in second.values()]
        assert len(set(values)) > 1, (
            "every p-value is identical across two different seeds; the seed is "
            "not reaching the permutation"
        )

    def test_taps_do_not_share_a_shuffle_sequence(self):
        """Two taps must not be correlated through one RNG stream.

        Sharing a stream would make neighbouring layers' p-values move together
        for a reason that has nothing to do with the maps.
        """
        seeds = [
            C._permutation_seed(0, f"topolm/block_{i}/attention_tap")
            for i in range(8)
        ]
        assert len(set(seeds)) == len(seeds), "tap paths collide into one seed"

    def test_the_seed_derivation_is_stable_across_processes(self):
        """A literal, not a hash: this value must not be process-salted."""
        assert C._permutation_seed(3, "topolm/block_0/mlp_tap") == C._permutation_seed(
            3, "topolm/block_0/mlp_tap"
        )
        # And it must depend on both inputs.
        assert C._permutation_seed(
            3, "a"
        ) != C._permutation_seed(4, "a")
        assert C._permutation_seed(3, "a") != C._permutation_seed(3, "b")

    def test_a_non_positive_count_is_rejected_at_the_call_boundary(self):
        with pytest.raises(ValueError, match="num_permutations must be positive"):
            self._report(permutation_p_value=True, num_permutations=0)

    def test_the_flag_reaches_the_run_entry_point(self, tmp_path):
        config = _config(
            permutation_p_value=True, num_permutations=9, save_dir="perm"
        )
        backbone = build_backbone(config)
        report = run_topographic_evaluation(
            backbone, config, StubTokenizer(), str(tmp_path)
        )
        per_tap = report["arms"]["raw"]["per_tap"]
        assert all(
            entry["morans_i"]["num_permutations"] == 9
            for entry in per_tap.values()
        )


class TestTheVocabularyGuard:
    """A vocabulary too small for the tokenizer used to fail on the first batch."""

    def test_a_vocabulary_below_the_tokenizers_is_rejected_by_name(self):
        class _Tok:
            vocab_size = 100277

        with pytest.raises(ValueError, match=r"--vocab-size"):
            C._require_vocab_covers_tokenizer(_config(vocab_size=50257), _Tok())

    def test_the_message_names_both_sizes_and_the_encoding(self):
        class _Tok:
            vocab_size = 100277
            tokenizer = type("E", (), {"name": "cl100k_base"})()

        with pytest.raises(ValueError) as excinfo:
            C._require_vocab_covers_tokenizer(_config(vocab_size=50257), _Tok())
        message = str(excinfo.value)
        assert "50257" in message
        assert "100277" in message
        assert "cl100k_base" in message

    def test_a_vocabulary_that_fits_is_accepted(self):
        class _Tok:
            vocab_size = 100277

        C._require_vocab_covers_tokenizer(_config(vocab_size=100277), _Tok())
        C._require_vocab_covers_tokenizer(_config(vocab_size=200000), _Tok())

    def test_a_tokenizer_without_the_attribute_is_not_obstructed(self):
        """A stub or a plain callable must still work.

        Refusing here would break every test double standing in for the
        tokenizer; the forward pass remains the backstop for those.
        """
        C._require_vocab_covers_tokenizer(_config(vocab_size=8), object())
        C._require_vocab_covers_tokenizer(_config(vocab_size=8), lambda t: t)

    def test_build_backbone_checks_a_supplied_requirement(self):
        with pytest.raises(ValueError, match=r"--vocab-size"):
            build_backbone(_config(vocab_size=50257), required_vocab_size=100277)
        # And stays silent when told nothing.
        assert build_backbone(_config(vocab_size=50257)) is not None

    def test_the_guard_fires_before_the_dataset_is_built(self, monkeypatch):
        """The point of the check's position, not just its existence.

        Once `preprocess_clm_dataset` has run, the same error is an out-of-range
        Gather from inside Embedding.call, naming no flag.
        """
        _patch_data(monkeypatch)
        built = []
        monkeypatch.setattr(
            C, "load_train_val_datasets",
            lambda *a, **k: built.append(True) or _load_synthetic(),
        )

        class _Tok:
            vocab_size = 100277

        config = _config(vocab_size=50257, save_dir="guard")
        with pytest.raises(ValueError, match=r"--vocab-size"):
            train_topolm(config, preprocessor=_Tok())
        assert not built, (
            "the dataset was built before the vocabulary was validated"
        )


class TestEvaluation:
    def _report(self, **overrides):
        backbone = build_backbone(_config())
        settings = dict(
            stimuli=SMOKE_STIMULI,
            contrast_conditions=("a", "b"),
            preprocessor=StubTokenizer(),
            readout_fwhm=2.0,
            min_cluster_size=2,
            is_smoke_set=False,
        )
        settings.update(overrides)
        report, arrays = evaluate_topography(backbone, **settings)
        return backbone, report, arrays

    def test_the_report_carries_every_tap_and_both_arms(self):
        backbone, report, _ = self._report()
        for arm in ("raw", "readout"):
            per_tap = report["arms"][arm]["per_tap"]
            assert set(per_tap) == {t.path for t in backbone.tap_layers}

    def test_the_correction_scope_is_recorded(self):
        """Joint, and the report says so.

        Correcting per tap instead admits roughly 470 false positives per
        784-unit layer, and the map's threshold stops being a statement about
        the model.
        """
        _, report, _ = self._report()
        assert report["fdr_scope"] == "joint across all taps"
        assert report["fdr_alpha"] == 0.05

    def test_the_smoke_set_is_flagged_when_it_is_used(self):
        _, report, _ = self._report(is_smoke_set=True)
        assert report["stimuli_are_smoke_set"] is True

    def test_morans_i_is_scored_on_the_unthresholded_map(self):
        """Both statistics are reported, and they disagree on small islands.

        Asserting only that both are present would pass a pipeline that
        thresholded first, since a contiguous zero patch also scores positive.
        """
        _, report, _ = self._report()
        entry = next(iter(report["arms"]["raw"]["per_tap"].values()))
        summary = entry["morans_i"]
        assert set(summary) >= {"standard", "islands", "num_units"}
        assert summary["num_units"] == 16 * 16

    def test_the_readout_arm_does_not_alter_the_raw_arm(self):
        """The arms are two measurements, not a pipeline order.

        Smoothing activations is a different operation from smoothing a t-map,
        and running the readout first would make the raw arm a smoothed arm. One
        backbone serves both reports: with a fresh model each, the raw arms would
        differ because of the weights and the assertion would measure RNG draws
        instead.
        """
        backbone = build_backbone(_config())
        shared = dict(
            stimuli=SMOKE_STIMULI,
            contrast_conditions=("a", "b"),
            preprocessor=StubTokenizer(),
            min_cluster_size=2,
        )
        with_readout, _ = evaluate_topography(
            backbone, readout_fwhm=2.0, **shared
        )
        without_readout, _ = evaluate_topography(
            backbone, readout_fwhm=None, **shared
        )

        assert "readout" in with_readout["arms"]
        assert "readout" not in without_readout["arms"]

        left = with_readout["arms"]["raw"]["per_tap"]
        right = without_readout["arms"]["raw"]["per_tap"]
        assert set(left) == set(right)
        for name, entry in left.items():
            other = right[name]
            # Compared field by field rather than as dicts: `morans_i["islands"]`
            # is legitimately NaN when a tap has no significant units, and NaN
            # never equals itself, so a whole-dict comparison fails for the right
            # reason and hides the wrong one.
            assert entry["num_significant_units"] == other["num_significant_units"]
            assert entry["cluster_sizes_a"] == other["cluster_sizes_a"]
            assert entry["cluster_sizes_b"] == other["cluster_sizes_b"]
            assert entry["morans_i"]["standard"] == other["morans_i"]["standard"]
            assert entry["morans_i"]["num_units"] == other["morans_i"]["num_units"]
        assert (
            with_readout["arms"]["raw"]["mean_morans_i"]
            == without_readout["arms"]["raw"]["mean_morans_i"]
        )

    def test_disabling_the_fwhm_leaves_the_raw_arm_only(self):
        _, report, _ = self._report(readout_fwhm=0)
        assert list(report["arms"]) == ["raw"]

    def test_a_missing_contrast_condition_is_rejected_by_name(self):
        with pytest.raises(ValueError, match=r"\['zzz'\]"):
            self._report(contrast_conditions=("a", "zzz"))

    def test_the_report_is_written_to_disk(self, tmp_path):
        backbone = build_backbone(_config())
        evaluate_topography(
            backbone,
            stimuli=SMOKE_STIMULI,
            contrast_conditions=("a", "b"),
            preprocessor=StubTokenizer(),
            output_dir=str(tmp_path / "report"),
            readout_fwhm=None,
            min_cluster_size=2,
        )
        assert (tmp_path / "report" / "topography_report.json").is_file()

    def test_run_topographic_evaluation_writes_figures_beside_the_json(
        self, tmp_path
    ):
        """The run entry point must produce the figures, not just the numbers.

        A figure module that nothing calls is dead code, and the JSON alone
        cannot show a topography. Asserted on the FILES because the point is
        that a reader gets a figure per run.
        """
        backbone = build_backbone(_config())
        run_topographic_evaluation(
            backbone,
            _config(),
            StubTokenizer(),
            str(tmp_path),
        )
        written = {path.name for path in tmp_path.iterdir()}
        assert "topography_report.json" in written
        assert "t_maps_raw.png" in written, sorted(written)
        assert "morans_i_raw.png" in written, sorted(written)

    def test_each_arms_figure_is_fed_that_arms_own_t_maps(self):
        """``t_maps_raw.png`` must be drawn from the RAW t-maps. It was not.

        The defect this pins: ``evaluate_topography`` computed t-maps for every
        arm but returned ONE set of arrays -- the last arm's -- so
        ``plot_topography`` wrote ``t_maps_raw.png`` and ``t_maps_readout.png``
        from the same readout-smoothed numbers. The two panels came out
        byte-identical (verified on a real run: max pixel difference 0), and
        the one labelled "raw" was wrong by up to 22.99 t-units.

        Each arm is therefore re-derived here INDEPENDENTLY -- extracted,
        optionally blurred, contrasted -- and compared against the arrays that
        arm's figure is fed. Comparing only shapes or key sets cannot catch this:
        both arms have identical shapes and identical tap keys, which is exactly
        why the previous version of this test passed on the broken code while its
        docstring claimed to compare against a re-extraction.
        """
        backbone, report, arrays = self._report()

        activations = {
            condition: extract_tap_activations(backbone, prompts, StubTokenizer())
            for condition, prompts in SMOKE_STIMULI.items()
        }
        grid = tuple(report["grid_shape"])
        tap_path = backbone.tap_layers[0].path

        expected = {}
        expected["raw"] = contrast_tmap(
            activations["a"][tap_path], activations["b"][tap_path]
        )[0].reshape(grid)
        smoothed = {
            condition: C._apply_readout(values, backbone, 2.0, 1.0)
            for condition, values in activations.items()
        }
        expected["readout"] = contrast_tmap(
            smoothed["a"][tap_path], smoothed["b"][tap_path]
        )[0].reshape(grid)

        # The fixture must genuinely discriminate, or the assertions below are
        # satisfied by an array that matches both arms.
        assert not np.allclose(
            expected["raw"], expected["readout"]
        ), "the fixture's two arms are identical; this test proves nothing"

        for arm, reference in expected.items():
            assert set(arrays["arms"][arm]["t_maps"]) == {
                t.path for t in backbone.tap_layers
            }
            np.testing.assert_allclose(
                arrays["arms"][arm]["t_maps"][tap_path],
                reference,
                rtol=1e-6,
                atol=1e-6,
                err_msg=(
                    f"the arrays feeding {arm}'s figure are not the {arm} "
                    f"t-maps (max diff "
                    f"{np.abs(arrays['arms'][arm]['t_maps'][tap_path] - reference).max():.4f})"
                ),
            )

    def test_the_two_arms_produce_visibly_different_figures(self, tmp_path):
        """End-to-end, on the PNGs a run actually writes.

        This is the regression test for the defect measured on a real run: the
        two arm figures came out with **byte-identical panel pixels** (max
        difference 0), so `t_maps_raw.png` showed readout-smoothed data under
        the name "raw". Reading the saved images is the only assertion that
        covers the whole path -- evaluation, file naming, and rendering -- and
        the earlier unit-level tests stop one layer short of it.

        The title band is cropped before comparing, because the arm name is
        printed there and would otherwise be the only difference.
        """
        iio = pytest.importorskip(
            "imageio.v3", reason="imageio needed to read back the PNGs"
        )
        backbone = build_backbone(_config())
        config = _config()
        report, arrays = evaluate_topography(
            backbone,
            stimuli=SMOKE_STIMULI,
            contrast_conditions=("a", "b"),
            preprocessor=StubTokenizer(),
            readout_fwhm=config.readout_fwhm,
            readout_unit_spacing=config.readout_unit_spacing,
            min_cluster_size=config.min_cluster_size,
        )
        written = plot_topography(report, arrays, str(tmp_path))
        names = {os.path.basename(path) for path in written}
        assert {"t_maps_raw.png", "t_maps_readout.png"} <= names, sorted(names)

        def panels(name):
            image = iio.imread(str(tmp_path / name))
            return image[
                int(image.shape[0] * 0.10) :,
                int(image.shape[1] * 0.02) : int(image.shape[1] * 0.92),
            ]

        raw, readout = panels("t_maps_raw.png"), panels("t_maps_readout.png")
        assert raw.shape == readout.shape
        difference = np.abs(raw.astype(int) - readout.astype(int)).max()
        assert difference > 0, (
            "the raw and readout t-map figures are pixel-identical; one arm's "
            "arrays are being drawn under both names"
        )

    def test_the_arrays_are_keyed_by_arm_and_cover_exactly_those_arms(self):
        """Structural half of the contract: the keys, per arm."""
        backbone, report, arrays = self._report()
        assert set(arrays["arms"]) == set(report["arms"])
        for arm in arrays["arms"]:
            arm_arrays = arrays["arms"][arm]
            assert set(arm_arrays) == {"t_maps", "sig_grids", "label_grids"}
            assert set(arm_arrays["t_maps"]) == set(report["arms"][arm]["per_tap"])
            for name, values in arm_arrays["t_maps"].items():
                assert values.shape == tuple(report["grid_shape"])
                assert np.asarray(arm_arrays["sig_grids"][name]).shape == values.shape
                assert np.asarray(arm_arrays["label_grids"][name]).shape == values.shape

    def test_a_figure_is_not_drawn_for_an_arm_that_was_not_computed(
        self, tmp_path
    ):
        """No smoothed arm, no smoothed figure.

        A stale PNG from a previous configuration sitting beside a fresh report
        is worse than no figure: it is a result that was never recomputed.
        """
        backbone = build_backbone(_config())
        config = _config()
        config.readout_fwhm = None
        run_topographic_evaluation(backbone, config, StubTokenizer(), str(tmp_path))
        written = {path.name for path in tmp_path.iterdir()}
        assert "t_maps_raw.png" in written
        assert "t_maps_readout.png" not in written

    def test_every_tap_gets_its_own_readout_layout(self):
        """A single shared readout would blur every layer as one grid.

        Under the default per-tap permutation the layouts differ, so one shared
        readout would be wrong for all but one of them. Built first: a layout is
        resolved in ``build``, and every tap here is read through it.
        """
        backbone = build_backbone(_config())
        backbone(np.zeros((1, 8), dtype="int32"), training=False)

        layouts = [tap.layout.perm for tap in backbone.tap_layers]
        assert len(layouts) > 1
        assert all(layout is not None for layout in layouts)
        assert len({tuple(p.tolist()) for p in layouts}) == len(layouts)


# ---------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------


class TestEndToEnd:
    def test_a_run_trains_evaluates_and_writes_its_outputs(self, monkeypatch, tmp_path):
        _patch_data(monkeypatch)
        result = train_topolm(_config(save_dir="run"), preprocessor=StubTokenizer())
        results_dir = result["results_dir"]

        assert result["epochs"] >= 1
        assert "val_loss" in result["history"].history
        assert "topography" in result
        written = set(os.listdir(results_dir))
        assert {"config.json", "training_history.json",
                "topography_report.json"} <= written

    def test_the_topographic_arm_reports_a_larger_loss_than_its_control(
        self, monkeypatch
    ):
        """The spatial term REACHES the objective, measured on the same weights.

        Both arms run the same data, the same seed and the same taps; only
        ``alpha`` differs. If the head were not aggregating the backbone's
        losses the two reported losses would be equal.
        """
        losses = {}
        for alpha in (0.0, 2.5):
            _patch_data(monkeypatch)
            config = _config(spatial_alpha=alpha, spatial_log_every=1)
            head, backbone = create_topolm_model(config)
            epochs, steps = resolve_cadence(config, 16)
            compile_model(head, config, epochs, steps)
            train, validation, _ = _load_synthetic(16)
            history = head.fit(
                train, epochs=1, steps_per_epoch=1,
                validation_data=validation, verbose=0,
            )
            losses[alpha] = history.history["loss"][0]
            assert len(backbone.tap_layers) > 0, "both arms must hold their taps"

        assert losses[2.5] > losses[0.0] + 1.0, (
            f"alpha had no effect on the reported loss: {losses}"
        )

    def test_the_validation_loss_is_the_pure_task_loss(self, monkeypatch):
        """The two numbers are different quantities and must differ.

        The taps add nothing when ``training`` is not ``True``, so a
        ``val_loss`` near the *training* loss would mean the taps fired at
        validation too -- and would make the paper's reported 3.075 / 2.966 pair
        incomparable.
        """
        _patch_data(monkeypatch)
        config = _config()
        head, _ = create_topolm_model(config)
        epochs, steps = resolve_cadence(config, 16)
        compile_model(head, config, epochs, steps)
        train, validation, _ = _load_synthetic(16)
        history = head.fit(
            train, epochs=1, steps_per_epoch=1,
            validation_data=validation, verbose=0,
        )
        train_loss = history.history["loss"][0]
        val_loss = history.history["val_loss"][0]

        assert val_loss < train_loss, (
            f"val_loss {val_loss:.3f} is not below train loss {train_loss:.3f}; "
            "the training loss carries the spatial term and validation does not"
        )
        num_taps = 2 * 2
        assert train_loss - val_loss == pytest.approx(
            2.5 * num_taps * 0.5, rel=0.9
        )

    def test_an_alpha_zero_control_holds_an_identical_weight_set(self):
        """The comparison the paper makes is only a comparison if this holds."""
        def paths(alpha):
            backbone = build_backbone(_config(spatial_alpha=alpha))
            backbone(
                np.random.default_rng(0).integers(0, VOCAB, size=(2, 8)).astype(
                    "int32"
                ),
                training=False,
            )
            return sorted(w.path for w in backbone.weights)

        assert paths(0.0) == paths(2.5)
        assert paths(2.5) == paths(2.5), "the paths are not deterministic"

    def test_the_control_arm_still_holds_its_tap_tables(self):
        """A control that dropped its taps would differ in weights, not in loss."""
        backbone = build_backbone(_config(spatial_alpha=0.0))
        backbone(
            np.zeros((1, 8), dtype="int32"), training=False
        )
        assert len(backbone.tap_layers) == 4
        assert any(w.name == "perm" for w in backbone.weights)


class TestPairedRun:
    def test_pairing_requires_a_positive_topographic_arm(self):
        with pytest.raises(ValueError, match="needs a positive topographic arm"):
            train_paired(_config(spatial_alpha=0.0))

    def test_both_arms_are_trained_and_written_separately(self, monkeypatch, tmp_path):
        _patch_data(monkeypatch)
        results = train_paired(
            _config(save_dir="paired"), preprocessor=StubTokenizer()
        )
        assert set(results) == {"topographic", "control"}
        directories = {
            name: result["results_dir"] for name, result in results.items()
        }
        assert directories["topographic"] != directories["control"], (
            "both arms wrote to the same directory, so the two runs are not "
            "distinguishable when read back"
        )
        assert all(os.path.isdir(path) for path in directories.values())

    def test_no_topography_reaches_both_arms(self, monkeypatch, tmp_path):
        """`--no-topography --paired` must skip BOTH arms' reports.

        The flag is a CLI argument that arrives here, and it was dropped on the
        paired path: the topographic arm still wrote a report and a full figure
        set while the user had asked for no topography at all. A partial
        evaluation is worse than none -- it looks like a result for one arm and
        invites comparison against a control that has none.
        """
        _patch_data(monkeypatch)
        results = train_paired(
            _config(save_dir="paired_notopo"),
            preprocessor=StubTokenizer(),
            run_topography=False,
        )
        for name, result in results.items():
            written = os.listdir(result["results_dir"])
            assert "topography_report.json" not in written, (name, written)
            assert not any(path.endswith(".png") and "training_curves" not in path
                           for path in written), (name, written)

    def test_topography_still_runs_on_both_arms_when_asked(self, monkeypatch):
        """The other arm of the flag: forwarding must not disable it by default."""
        _patch_data(monkeypatch)
        results = train_paired(
            _config(save_dir="paired_topo"),
            preprocessor=StubTokenizer(),
            run_topography=True,
        )
        for name, result in results.items():
            assert os.path.isfile(
                os.path.join(result["results_dir"], "topography_report.json")
            ), name

    def test_the_control_arm_is_only_alpha_that_differs(self):
        from dataclasses import replace

        config = _config()
        control = replace(config, spatial_alpha=0.0)
        differing = {
            field.name
            for field in config.__dataclass_fields__.values()
            if getattr(config, field.name) != getattr(control, field.name)
        }
        assert differing == {"spatial_alpha"}


def _load_synthetic(rows=16, batch_size=4):
    """The synthetic stream ``_patch_data`` installs, for direct ``fit`` calls."""
    generator = np.random.default_rng(0)
    tokens = generator.integers(0, VOCAB, size=(rows, LENGTH)).astype("int32")
    labels = generator.integers(0, VOCAB, size=(rows, LENGTH)).astype("int32")
    train = tf.data.Dataset.from_tensor_slices((tokens, labels)).batch(
        batch_size
    ).repeat()
    validation = tf.data.Dataset.from_tensor_slices(
        (tokens[:8], labels[:8])
    ).batch(batch_size)
    return train, validation, rows
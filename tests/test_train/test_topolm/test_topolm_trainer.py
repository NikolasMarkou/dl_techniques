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
from train.topolm import pretrain as pretrain_cli
from dl_techniques.metrics.topographic_selectivity import contrast_tmap
from train.common.clm_pretrain import ClmPretrainConfig

from dl_techniques.models.language.topolm import MODEL_VARIANTS
from tests.numerics import MATMUL_ULP_ALLOWANCE, matmul_unit_roundoff

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
        # Rewritten deliberately (plan 972a9ac5 step 5, D-006): the dataclass
        # default is now None ("the variant's own radius") so a 3D row's radius
        # can reach the model; the paper's 5 is what that resolves to for the
        # default 2D variant, which is the recipe this test exists to pin.
        assert config.spatial_radius is None
        assert C.resolve_spatial_radius(config) == 5
        assert config.spatial_neighborhoods == 5
        assert config.spatial_distance == "linf"
        assert config.spatial_permute is True
        assert config.tap_sites == "both"
        assert config.early_stop_patience == 3

    def test_the_defaults_describe_a_model_that_can_exist(self):
        """Every variant row builds at the radius its own row resolves to.

        Rewritten deliberately (plan 972a9ac5 step 5, D-006): it used to expect a
        ``ValueError`` for the 3D rows, which the trainer refused by name. They
        build now, so the loop covers EVERY ``MODEL_VARIANTS`` row, with the
        radius left unset (``None``): a 2D row gets the paper's 5, which its grid
        holds, and a 3D row gets its own, which the radius-5 default could not
        (5 needs every axis 11 across). A default that only works for some
        variants fails on the first one a caller reaches for.
        """
        for name, row in MODEL_VARIANTS.items():
            config = _config(model_variant=name, spatial_radius=None)
            backbone = build_backbone(config)
            assert len(backbone.tap_layers) > 0, name
            assert backbone.radius == row.get("radius", 5), name
            assert len(backbone.grid_shape) == row.get("grid_dim", 2), name

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
        assert len(keras.ops.shape(output)) == 3


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
        plain = keras.ops.convert_to_numpy(
            backbone(tokens, training=False)["last_hidden_state"]
        )
        observed = keras.ops.convert_to_numpy(
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
        raw = keras.ops.convert_to_numpy(captures[0].value)

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
        """Batching must not mix items: one batch and batches of two agree.

        The bound depends on the matmul precision in force, which is measured,
        not read off a flag (``tests/numerics.py`` D-030). With true float32
        matmuls (CPU, or a GPU with TF32 off) it is the exact ``1e-6``: measured
        0.0 on CPU and at most 1.2e-8 on GPU 1 with TF32 off. With TF32 on (the
        default on GPU 1, and what the trainer runs under) the two batch shapes
        pick different kernels, and the measured gap was at most 1.53 TF32 unit
        roundoffs of the tap's max |activation| over 30 seeds (7.1e-5 at a max
        |activation| of 0.14). The bound is ``MATMUL_ULP_ALLOWANCE`` (4) of those
        units, per tap, with no floor at 1: the taps are 0.01 to 0.14 in
        magnitude, so a floor would loosen the small taps about 90x. Swapping
        two items moves a tap by 5e-3 or more, at least 250x this bound.
        Proven on GPU 1 with TF32 on: an extraction that swaps the first two
        items of every batch fails here, and so does one that leaks 1% of the
        previous batch's last item into the next batch's first.
        """
        backbone = self._backbone()
        prompts = list(SMOKE_STIMULI["a"])
        tokenizer = StubTokenizer()
        one = extract_tap_activations(backbone, prompts, tokenizer, batch_size=99)
        many = extract_tap_activations(backbone, prompts, tokenizer, batch_size=2)
        unit_roundoff = matmul_unit_roundoff()
        reduced_precision = unit_roundoff > float(np.finfo(np.float32).eps) / 2.0
        for name in one:
            if reduced_precision:
                atol = MATMUL_ULP_ALLOWANCE * unit_roundoff * np.max(np.abs(one[name]))
                np.testing.assert_allclose(one[name], many[name], rtol=0, atol=atol)
            else:
                np.testing.assert_allclose(
                    one[name], many[name], rtol=1e-6, atol=1e-6
                )


# ---------------------------------------------------------------------
# activation extraction: what is pooled (D-020)
# ---------------------------------------------------------------------

CLS_ID, SEP_ID, PAD_ID = 7, 8, 9


class SpecialIdTokenizer:
    """A stand-in that wraps and pads like the Tiktoken preprocessor does.

    ``[CLS] text [SEP] [PAD] ...`` to ``max_length``, with the three special ids
    published as attributes (``cls_token_id`` and friends) exactly as
    :class:`~dl_techniques.utils.tokenizer.TiktokenPreprocessor` publishes them,
    and DIFFERENT from the real ones, so a hard-coded id fails here. Text ids sit
    in ``[10, VOCAB)``.
    """

    cls_token_id = CLS_ID
    sep_token_id = SEP_ID
    pad_token_id = PAD_ID

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, texts):
        texts = [texts] if isinstance(texts, str) else texts
        ids = np.full((len(texts), self.max_length), PAD_ID, dtype="int32")
        for row, text in enumerate(texts):
            body = [
                10 + (ord(c) * 7) % (VOCAB - 10)
                for c in str(text)[:self.max_length - 2]
            ]
            sequence = [CLS_ID] + body + [SEP_ID]
            ids[row, :len(sequence)] = sequence
        return {"input_ids": ids}


class _PoisonedBackbone:
    """A real backbone whose captures are overwritten at chosen positions.

    The function under test reads ``backbone.tap_layers``, ``backbone.built`` and
    calls ``backbone(tokens, training=False, taps=...)``; everything else is
    delegated. After the real forward pass it writes ``value`` into every capture
    at the positions ``select(tokens)`` marks, which is the position-wise oracle
    for "this position does not enter the pooled response".
    """

    def __init__(self, backbone, select, value=1.0e6):
        self._backbone = backbone
        self._select = select
        self._value = value

    def __getattr__(self, name):
        return getattr(self._backbone, name)

    def __call__(self, tokens, **kwargs):
        output = self._backbone(tokens, **kwargs)
        marked = self._select(np.asarray(tokens))
        for capture in kwargs["taps"]:
            raw = keras.ops.convert_to_numpy(capture.value).copy()
            raw[marked] = self._value
            capture.value = raw
        return output


def _raw_taps(backbone, ids):
    """``{tap_path: (batch, positions, units)}`` float64, one forward pass."""
    captures = [TapCapture() for _ in backbone.tap_layers]
    backbone(ids, training=False, taps=tuple(captures))
    return {
        tap.path: keras.ops.convert_to_numpy(capture.value).astype("float64")
        for tap, capture in zip(backbone.tap_layers, captures)
    }


def _text_only_loop(raw, ids, special):
    """The independent oracle: an explicit loop over sentences and positions."""
    pooled = np.zeros((raw.shape[0], raw.shape[2]), dtype="float64")
    for sentence in range(raw.shape[0]):
        total = np.zeros(raw.shape[2], dtype="float64")
        count = 0
        for position in range(raw.shape[1]):
            if int(ids[sentence, position]) in special:
                continue
            total += raw[sentence, position]
            count += 1
        pooled[sentence] = total / count
    return pooled


def _all_position_pooling(backbone, prompts, preprocessor, batch_size=32):
    """The pre-D-020 behaviour, kept as a reference: a mean over EVERY position.

    Used as the measuring instrument for the length controls ("what would the
    old code have reported on this model"), never as the code under test.
    """
    ids = np.asarray(preprocessor(list(prompts))["input_ids"], dtype="int32")
    pooled = {tap.path: [] for tap in backbone.tap_layers}
    for start in range(0, ids.shape[0], batch_size):
        for path, raw in _raw_taps(backbone, ids[start:start + batch_size]).items():
            pooled[path].append(raw.mean(axis=1))
    return {path: np.concatenate(rows) for path, rows in pooled.items()}


def _forward_close(actual, expected):
    """Compare two forward-pass results under the matmul regime in force.

    Same recipe as ``test_batching_does_not_change_the_result``: exact float32
    tolerance on CPU, ``MATMUL_ULP_ALLOWANCE`` TF32 roundoffs of the largest
    magnitude on a TF32 device.
    """
    unit_roundoff = matmul_unit_roundoff()
    if unit_roundoff > float(np.finfo(np.float32).eps) / 2.0:
        atol = MATMUL_ULP_ALLOWANCE * unit_roundoff * np.max(np.abs(expected))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def _tokenizer(kind, window):
    """``SpecialIdTokenizer`` ('stub') or the real gpt2 preprocessor ('gpt2')."""
    if kind == "stub":
        return SpecialIdTokenizer(window)
    return C.create_tokenizer("gpt2", window, 50257, 50258, 50259, 50260)


def _model(kind, window):
    """A tiny model whose table holds every id the tokenizer of ``kind`` emits."""
    vocab = VOCAB if kind == "stub" else 50261
    return build_backbone(_config(vocab_size=vocab, max_seq_length=window))


def _special_ids(preprocessor):
    return {
        int(preprocessor.cls_token_id),
        int(preprocessor.sep_token_id),
        int(preprocessor.pad_token_id),
    }


_KINDS = pytest.mark.parametrize("kind", ["stub", "gpt2"])


class TestPoolingIsOverTextTokensOnly:
    """D-020: a unit's response is its mean over the TEXT tokens of a sentence.

    The tokenizer wraps every stimulus ``[CLS] text [SEP]`` and pads it to the
    window (512 in the trainer), tokens the model never sees as inputs in
    training. A mean over all positions was ~98% padding, so a t-map between two
    conditions measured padding and sentence length (review pass 2, concern 1).

    The measured effect on the TINY test model (random weights, window 512, the
    gpt2 preprocessor, CPU, seeds 0 to 5), significant units summed over the four
    taps (1024 units) at FDR 0.05, for the reviewer's control "the five
    condition-a sentences against themselves with ' . . . .' appended":

        pooling over all positions (the old code)    109 to 211 of 1024
        pooling over text tokens (this code)           0 to 2 of 1024

    The a-versus-b smoke contrast gives 0 under both poolings on every one of
    those random models (an untrained network has no contrast to find), so the
    direction of the real effect is not asserted there: the trained 2D and 3D
    example checkpoints measured it (plans/.../findings/
    pooling-fix-measurements.md). The control is seeded below so the numbers
    reproduce, and its thresholds sit well inside the six measured pairs.
    """

    def test_it_equals_an_independent_loop_over_text_positions(self):
        """The fallback path (a preprocessor that can only be called): CLS, SEP and
        PAD ids are what separates text from the rest. The gpt2 framing has its
        own oracle in ``TestStimuliAreFramedLikeTheTrainingStream`` (D-024)."""
        kind = "stub"
        backbone, preprocessor = _model(kind, 64), _tokenizer(kind, 64)
        prompts = list(SMOKE_STIMULI["a"]) + list(SMOKE_STIMULI["b"])
        ids = np.asarray(preprocessor(prompts)["input_ids"], dtype="int32")
        special = _special_ids(preprocessor)
        assert (ids == int(preprocessor.pad_token_id)).any(), "no padding to test"

        pooled = extract_tap_activations(backbone, prompts, preprocessor)
        raw = _raw_taps(backbone, ids)
        for path, values in pooled.items():
            _forward_close(values, _text_only_loop(raw[path], ids, special))

    @_KINDS
    def test_the_window_length_does_not_change_the_response(self, kind):
        """THE discriminating control: the same text, more padding, same numbers.

        The model is causal, so the activation at a position cannot depend on a
        later one; pads after the text therefore cannot change a text position's
        activation, and a text-only pooling is exactly window-independent. The
        causal assumption is verified in the next test, not taken on trust.
        """
        backbone = _model(kind, 256)
        # 96, not 64: the stub cuts a sentence to ``window - 2`` characters, and the
        # longest smoke sentence is 64 (D-025 lengthened condition b), so a 64
        # window would truncate TEXT and the two windows would not see the same
        # sentences.
        long_window, short_window = _tokenizer(kind, 256), _tokenizer(kind, 96)
        prompts = list(SMOKE_STIMULI["a"]) + list(SMOKE_STIMULI["b"])
        short_ids = np.asarray(short_window(prompts)["input_ids"])
        assert (short_ids == int(short_window.pad_token_id)).any()

        long = extract_tap_activations(backbone, prompts, long_window)
        short = extract_tap_activations(backbone, prompts, short_window)
        for path in long:
            _forward_close(short[path], long[path])

        # The control has power: the all-position mean moves by orders of
        # magnitude more than the tolerance when the window changes.
        old_long = _all_position_pooling(backbone, prompts, long_window)
        old_short = _all_position_pooling(backbone, prompts, short_window)
        gap = max(
            float(np.max(np.abs(old_long[p] - old_short[p]))) for p in old_long
        )
        scale = max(float(np.max(np.abs(long[p]))) for p in long)
        assert gap > 0.05 * scale, (gap, scale)

    def test_the_model_is_causal_so_the_trailing_pads_cannot_matter(self):
        """The assumption behind the invariance test, executed.

        One backbone, the same prompts through a 40-token and a 256-token window:
        the first 40 positions of every captured activation are identical.
        """
        backbone = _model("gpt2", 256)
        prompts = list(SMOKE_STIMULI["a"])
        short, long = _tokenizer("gpt2", 40), _tokenizer("gpt2", 256)
        short_ids = np.asarray(short(prompts)["input_ids"], dtype="int32")
        long_ids = np.asarray(long(prompts)["input_ids"], dtype="int32")
        np.testing.assert_array_equal(short_ids, long_ids[:, :40])
        short_raw = _raw_taps(backbone, short_ids)
        long_raw = _raw_taps(backbone, long_ids)
        for path in short_raw:
            _forward_close(short_raw[path], long_raw[path][:, :40])

    @pytest.mark.parametrize(
        "excluded", ["cls", "sep", "pad"], ids=["cls", "sep", "pad"]
    )
    def test_a_special_token_position_does_not_enter_the_response(
        self, excluded
    ):
        """Set every CLS (or SEP, or PAD) activation to 1e6: nothing moves.

        The fallback path only: the gpt2 preprocessor's stimuli carry no CLS or SEP
        any more (D-024), so a poison keyed on those ids would mark nothing and
        pass vacuously; ``TestStimuliAreFramedLikeTheTrainingStream`` poisons the
        end-of-text prefix and the pads instead.

        Excluding only the pad ids would pass the loop oracle on a tokenizer
        whose CLS and SEP happened to be quiet; the huge constant cannot hide.
        The control below poisons one TEXT position and the response does move.
        """
        kind = "stub"
        backbone, preprocessor = _model(kind, 48), _tokenizer(kind, 48)
        prompts = list(SMOKE_STIMULI["a"])
        token = int(getattr(preprocessor, f"{excluded}_token_id"))
        clean = extract_tap_activations(backbone, prompts, preprocessor)

        poisoned = extract_tap_activations(
            _PoisonedBackbone(backbone, lambda ids: ids == token),
            prompts, preprocessor,
        )
        for path in clean:
            _forward_close(poisoned[path], clean[path])

        special = _special_ids(preprocessor)

        def first_text_position(ids):
            marked = np.zeros(ids.shape, dtype=bool)
            marked[:, 1] = ~np.isin(ids[:, 1], list(special))
            return marked

        moved = extract_tap_activations(
            _PoisonedBackbone(backbone, first_text_position),
            prompts, preprocessor,
        )
        for path in clean:
            assert np.max(np.abs(moved[path] - clean[path])) > 1.0, path

    def test_a_sentence_with_no_text_token_is_refused_by_name(self):
        backbone, preprocessor = _model("stub", 24), _tokenizer("stub", 24)
        with pytest.raises(ValueError) as excinfo:
            extract_tap_activations(
                backbone, ["alpha beta", "", "gamma"], preprocessor
            )
        message = str(excinfo.value)
        assert "index 1" in message and "''" in message, message
        assert "no text token" in message, message

    def test_the_refusal_comes_before_any_forward_pass(self):
        """Nothing is computed for a batch that cannot be pooled."""
        backbone, preprocessor = _model("stub", 24), _tokenizer("stub", 24)

        class _Unbuilt:
            built = True
            tap_layers = backbone.tap_layers

            def __call__(self, *args, **kwargs):
                raise AssertionError("the forward pass ran before the refusal")

        with pytest.raises(ValueError, match="no text token"):
            extract_tap_activations(_Unbuilt(), ["", "alpha"], preprocessor)

    def test_a_preprocessor_without_special_ids_excludes_nothing(self):
        """No attributes (or ``None``) means every position is text."""
        backbone = build_backbone(_config())
        prompts = ["alpha bravo", "charlie"]

        class _NoneIds(StubTokenizer):
            cls_token_id = None
            sep_token_id = None
            pad_token_id = None

        plain = extract_tap_activations(backbone, prompts, StubTokenizer())
        nones = extract_tap_activations(backbone, prompts, _NoneIds())
        ids = np.asarray(StubTokenizer()(prompts)["input_ids"], dtype="int32")
        raw = _raw_taps(backbone, ids)
        for path in plain:
            _forward_close(plain[path], raw[path].mean(axis=1))
            _forward_close(nones[path], plain[path])

    def test_a_condition_against_itself_has_no_contrast(self):
        backbone, preprocessor = _model("gpt2", 512), _tokenizer("gpt2", 512)
        report, arrays = evaluate_topography(
            backbone,
            {"x": SMOKE_STIMULI["a"], "y": SMOKE_STIMULI["a"]},
            ("x", "y"),
            preprocessor,
            min_cluster_size=2,
            batch_size=8,
        )
        for values in arrays["arms"]["raw"]["t_maps"].values():
            assert np.all(values == 0.0)
        for entry in report["arms"]["raw"]["per_tap"].values():
            assert entry["num_significant_units"] == 0

    def test_extra_neutral_tokens_do_not_create_a_contrast(
        self, monkeypatch
    ):
        """The reviewer's length control, end to end through the evaluation.

        The same five sentences against themselves with ' . . . .' appended.
        Under an all-position mean the length difference alone makes a large
        part of the units significant; over text tokens it does not. Both numbers
        are measured on the same model in the same test, so the control cannot
        pass because the model happens to be quiet.
        """
        keras.utils.set_random_seed(0)
        backbone, preprocessor = _model("gpt2", 512), _tokenizer("gpt2", 512)
        stimuli = {
            "plain": list(SMOKE_STIMULI["a"]),
            "dotted": [s + " . . . ." for s in SMOKE_STIMULI["a"]],
        }

        def significant():
            report, _ = evaluate_topography(
                backbone, stimuli, ("dotted", "plain"), preprocessor,
                min_cluster_size=2, batch_size=8,
            )
            return sum(
                entry["num_significant_units"]
                for entry in report["arms"]["raw"]["per_tap"].values()
            )

        fixed = significant()
        monkeypatch.setattr(C, "extract_tap_activations", _all_position_pooling)
        unfixed = significant()
        monkeypatch.undo()

        units = len(backbone.tap_layers) * backbone.embed_dim
        assert unfixed > 0.05 * units, (unfixed, units)
        assert fixed < 0.25 * unfixed, (fixed, unfixed)
        assert fixed < 0.02 * units, (fixed, units)


class _SpyBackbone:
    """A real backbone that records the ids it is given; everything else delegates."""

    def __init__(self, backbone):
        self._backbone = backbone
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._backbone, name)

    def __call__(self, tokens, **kwargs):
        self.calls.append(np.asarray(tokens).copy())
        return self._backbone(tokens, **kwargs)


class _RawEncoder:
    """The raw-encoder surface of a tiktoken ``Encoding``: ``encode`` and ``eot_token``.

    Character codes as ids in ``[10, VOCAB)`` and an end-of-text id (77) that is
    neither the real gpt2 one nor any CLS/SEP/PAD id, so an id typed into the code
    fails here.
    """

    eot_token = 77

    def encode(self, text):
        return [10 + (ord(c) * 7) % (VOCAB - 10) for c in str(text)]


class _RawEncoderPreprocessor(SpecialIdTokenizer):
    """``SpecialIdTokenizer`` that also publishes ``.tokenizer``, like the real one."""

    def __init__(self, max_length):
        super().__init__(max_length)
        self.tokenizer = _RawEncoder()


def _gpt2_encoding():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def _ids_the_model_sees(backbone, prompts, preprocessor, **kwargs):
    """``(ids the model was called with, pooled responses)``: one spy per call.

    The model is built first, so the one-row build call of an unbuilt backbone is
    not mistaken for a stimulus batch.
    """
    backbone(np.zeros((1, 4), dtype="int32"), training=False)
    spy = _SpyBackbone(backbone)
    pooled = extract_tap_activations(spy, prompts, preprocessor, **kwargs)
    return np.concatenate(spy.calls) if spy.calls else None, pooled


def _framed_oracle(backbone, encoding, prompts, keep=None, pad_to=None):
    """Independent oracle: each sentence ALONE, ``[eot] + text`` (+ optional pads).

    The mean over the text positions (1 .. n) of one forward pass of one sentence.
    No batching and, by default, no padding, so neither can be what the function
    under test is compared against.
    """
    pooled = {tap.path: [] for tap in backbone.tap_layers}
    for prompt in prompts:
        text = encoding.encode(prompt)[:keep]
        ids = [int(encoding.eot_token)] + text
        if pad_to is not None:
            ids = ids + [50259] * (pad_to - len(ids))
        raw = _raw_taps(backbone, np.asarray([ids], dtype="int32"))
        for path, values in raw.items():
            pooled[path].append(values[0, 1:1 + len(text)].mean(axis=0))
    return {path: np.stack(rows) for path, rows in pooled.items()}


class TestStimuliAreFramedLikeTheTrainingStream:
    """D-024: a stimulus is ``<eot> text pad...``, not ``[CLS] text [SEP] pad...``.

    The model is causal: a text token's activation depends on everything before
    it. The packed training stream never contains CLS; a document there starts
    after ``<|endoftext|>``. Framing the stimuli with CLS fed every text token a
    context the model never trained on, and the trained 2D example's selective
    "islands" (38.9 significant units per tap, 5/7 clusters) were largely a
    product of it: with ``<eot>`` instead, 0.0 and 0/0 (review pass 3, concern 1).

    Pooling is over the text tokens, so the prefix position is excluded exactly as
    CLS was (a constant position weighted 1/n would re-introduce a length
    dependence). The gpt2 tests run on the REAL preprocessor.
    """

    PROMPTS = list(SMOKE_STIMULI["a"]) + list(SMOKE_STIMULI["b"])

    def _ids(self, window=512):
        backbone, preprocessor = _model("gpt2", 64), _tokenizer("gpt2", window)
        return backbone, preprocessor, _ids_the_model_sees(
            backbone, self.PROMPTS, preprocessor
        )

    def test_every_stimulus_starts_with_the_end_of_text_id_and_carries_no_cls_or_sep(
        self,
    ):
        _, preprocessor, (ids, _) = self._ids()
        eot = int(_gpt2_encoding().eot_token)
        assert eot == int(preprocessor.tokenizer.eot_token) == 50256
        assert np.all(ids[:, 0] == eot), ids[:, 0]
        assert not np.isin(
            ids, [preprocessor.cls_token_id, preprocessor.sep_token_id]
        ).any(), "a CLS or SEP id reached the model"
        assert np.count_nonzero(ids == eot) == len(self.PROMPTS), (
            "the end-of-text id appears somewhere other than the prefix"
        )

    def test_the_text_ids_are_the_plain_tokenization_followed_by_pads_only(self):
        _, preprocessor, (ids, _) = self._ids()
        encoding = _gpt2_encoding()
        for row, prompt in enumerate(self.PROMPTS):
            expected = encoding.encode(prompt)
            np.testing.assert_array_equal(ids[row, 1:1 + len(expected)], expected)
            assert np.all(ids[row, 1 + len(expected):] == preprocessor.pad_token_id)

    def test_the_end_of_text_id_is_read_from_the_encoder_not_typed_in(self):
        backbone = build_backbone(_config())
        preprocessor = _RawEncoderPreprocessor(max_length=LENGTH)
        ids, _ = _ids_the_model_sees(backbone, ["ab", "abcd"], preprocessor)
        assert np.all(ids[:, 0] == _RawEncoder.eot_token), ids[:, 0]
        assert not np.isin(ids, [CLS_ID, SEP_ID]).any()

    def test_the_batch_is_padded_to_the_longest_stimulus_plus_the_prefix_only(self):
        _, _, (ids, _) = self._ids(window=512)
        encoding = _gpt2_encoding()
        longest = max(len(encoding.encode(p)) for p in self.PROMPTS)
        assert ids.shape == (len(self.PROMPTS), longest + 1), ids.shape

    def test_it_equals_an_independent_loop_one_sentence_at_a_time(self):
        backbone, preprocessor = _model("gpt2", 64), _tokenizer("gpt2", 512)
        pooled = extract_tap_activations(backbone, self.PROMPTS, preprocessor)
        expected = _framed_oracle(backbone, _gpt2_encoding(), self.PROMPTS)
        for path, values in pooled.items():
            _forward_close(values, expected[path])

    def test_more_padding_after_the_text_does_not_change_the_response(self):
        """The same sentences with 60 more pad ids behind them: same numbers.

        The model is causal (``test_the_model_is_causal_so_the_trailing_pads_cannot_matter``),
        so the batch padded to the longest stimulus and a window of 70 agree.
        """
        backbone, preprocessor = _model("gpt2", 80), _tokenizer("gpt2", 512)
        pooled = extract_tap_activations(backbone, self.PROMPTS, preprocessor)
        padded = _framed_oracle(
            backbone, _gpt2_encoding(), self.PROMPTS, pad_to=70
        )
        for path, values in pooled.items():
            _forward_close(values, padded[path])

    def test_a_stimulus_longer_than_the_position_table_is_cut_not_refused(self):
        """Window = min(preprocessor window, model ``max_seq_len``); prefix included."""
        backbone, preprocessor = _model("gpt2", 8), _tokenizer("gpt2", 512)
        ids, pooled = _ids_the_model_sees(backbone, self.PROMPTS, preprocessor)
        assert ids.shape[1] == 8, ids.shape
        expected = _framed_oracle(
            backbone, _gpt2_encoding(), self.PROMPTS, keep=7
        )
        for path, values in pooled.items():
            _forward_close(values, expected[path])

    @pytest.mark.parametrize("poisoned", ["prefix", "pad"])
    def test_a_non_text_position_does_not_enter_the_response(self, poisoned):
        """1e6 at every end-of-text prefix (or every pad) position: nothing moves.

        The text-position control below moves the response by more than 1, so the
        poison is visible when it lands on a pooled position, and the hit counter
        proves it landed on positions at all.
        """
        backbone, preprocessor = _model("gpt2", 64), _tokenizer("gpt2", 512)
        clean = extract_tap_activations(backbone, self.PROMPTS, preprocessor)
        hits = []

        def select(ids):
            marked = np.zeros(ids.shape, dtype=bool)
            if poisoned == "prefix":
                marked[:, 0] = True
            else:
                marked |= ids == int(preprocessor.pad_token_id)
            hits.append(int(marked.sum()))
            return marked

        poisoned_result = extract_tap_activations(
            _PoisonedBackbone(backbone, select), self.PROMPTS, preprocessor
        )
        assert hits and hits[0] > 0, "the poison marked no position"
        for path in clean:
            _forward_close(poisoned_result[path], clean[path])

        def first_text_position(ids):
            marked = np.zeros(ids.shape, dtype=bool)
            marked[:, 1] = True
            return marked

        moved = extract_tap_activations(
            _PoisonedBackbone(backbone, first_text_position),
            self.PROMPTS, preprocessor,
        )
        for path in clean:
            assert np.max(np.abs(moved[path] - clean[path])) > 1.0, path

    def test_a_sentence_with_no_text_token_is_refused_by_name_before_any_pass(self):
        backbone, preprocessor = _model("gpt2", 64), _tokenizer("gpt2", 512)
        spy = _SpyBackbone(backbone)
        with pytest.raises(ValueError) as excinfo:
            extract_tap_activations(spy, ["alpha beta", "", "gamma"], preprocessor)
        message = str(excinfo.value)
        assert "index 1" in message and "''" in message, message
        assert "no text token" in message, message
        assert spy.calls == []

    def test_the_prefix_is_a_real_input_so_it_changes_the_response(self):
        """The framing is a choice that matters: with CLS in front the numbers differ.

        Without this the tests above would pass for a framing that ignored the
        prefix altogether. The old CLS framing is rebuilt by hand and fed to the
        same backbone.
        """
        backbone, preprocessor = _model("gpt2", 64), _tokenizer("gpt2", 512)
        framed = extract_tap_activations(backbone, self.PROMPTS, preprocessor)
        encoding = _gpt2_encoding()
        with_cls = []
        for prompt in self.PROMPTS:
            text = encoding.encode(prompt)
            ids = np.asarray(
                [[int(preprocessor.cls_token_id)] + text], dtype="int32"
            )
            raw = _raw_taps(backbone, ids)
            with_cls.append(
                {path: values[0, 1:].mean(axis=0) for path, values in raw.items()}
            )
        for path in framed:
            cls_response = np.stack([row[path] for row in with_cls])
            assert np.max(np.abs(framed[path] - cls_response)) > 1e-3, path

    def test_a_call_only_preprocessor_still_gets_the_cls_sep_pad_exclusion(self):
        """No ``.tokenizer``: the previous path, the non-text ids left out."""
        preprocessor = _tokenizer("stub", 24)
        assert not hasattr(preprocessor, "tokenizer")
        tokens, text, _ = C._encode_stimuli(preprocessor, ["ab", "abcd"], None)
        assert tokens.shape == (2, 24)
        assert np.all(tokens[:, 0] == CLS_ID)
        assert text.sum(axis=1).tolist() == [2, 4]
        assert not text[:, 0].any()


def _trainer_tokenizer():
    """The preprocessor ``train_topolm`` builds when none is passed (D-025).

    Same call, same config fields, so the lengths a test counts are the lengths
    the trainer's stimuli have.
    """
    config = TopoLMTrainingConfig()
    return C.create_tokenizer(
        config.encoding_name,
        config.max_seq_length,
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
    )


def _text_token_counts(stimuli):
    """``{condition: [number of gpt2 text tokens of sentence i]}``."""
    encoder = _trainer_tokenizer().tokenizer
    return {
        condition: [len(encoder.encode(sentence)) for sentence in sentences]
        for condition, sentences in stimuli.items()
    }


class TestTheSmokeStimuliAreLengthMatched:
    """D-025: sentence i has the same number of gpt2 TEXT tokens in every condition.

    A t-map between two conditions of different length measures the length. The
    built-in smoke set used to differ by 8.0 against 5.8 text tokens (a against
    b) and 10.2 against 11.6 (c against d); length-only differences
    produced a large contrast on the trained 3D example (word lists from one
    vocabulary, 3 against 7 words: median 225 significant units per tap, against
    89 for the shipped a-versus-b), while equal lengths gave 0 in 12 draws.

    The counts are those of the trainer's own tokenizer after D-024's framing:
    the text ids are the plain ``encode`` of the sentence, so the count is
    ``len(encoder.encode(sentence))`` with no CLS and no SEP in it.
    """

    #: Text tokens of sentence i, the same in a, b, c and d.
    COUNTS = [9, 9, 7, 7, 8]

    def test_every_condition_has_the_same_token_count_at_every_index(self):
        counts = _text_token_counts(SMOKE_STIMULI)
        assert counts == {c: self.COUNTS for c in ("a", "b", "c", "d")}, counts

    def test_the_contrast_pairs_are_matched_whatever_the_other_conditions_do(self):
        """The two pairs the trainer contrasts (a/b) and the paper's (c/d)."""
        counts = _text_token_counts(SMOKE_STIMULI)
        assert counts["a"] == counts["b"]
        assert counts["c"] == counts["d"]

    def test_there_are_four_conditions_of_five_sentences(self):
        assert sorted(SMOKE_STIMULI) == ["a", "b", "c", "d"]
        assert {len(v) for v in SMOKE_STIMULI.values()} == {5}

    def test_no_sentence_is_repeated_inside_or_across_conditions(self):
        every = [s for sentences in SMOKE_STIMULI.values() for s in sentences]
        assert len(set(every)) == len(every)

    def test_each_sentence_is_short(self):
        assert max(max(v) for v in _text_token_counts(SMOKE_STIMULI).values()) <= 12

    def test_the_conditions_keep_their_meaning(self):
        """b and d are lists, not sentences: no function word from a's frames
        and, for c and d, the nonwords are not words of the real-word lists."""
        function_words = {"the", "a", "and", "by", "over", "through", "before"}
        for condition in ("b",):
            for sentence in SMOKE_STIMULI[condition]:
                assert not function_words & set(sentence.split()), sentence
        real_words = {w for s in SMOKE_STIMULI["b"] for w in s.split()}
        nonwords = {"wug", "blarnket", "mibbed", "slarn", "glarn", "prarn", "doark"}
        for condition in ("c", "d"):
            used = {w for s in SMOKE_STIMULI[condition] for w in s.split()}
            assert used & nonwords, condition
            assert not used & real_words, condition
        # c is a sentence frame (function words present), d is a list of
        # nonwords with no frame: the three first words of c's sentences are
        # function words or pronouns, d's are not required to be.
        assert all(
            s.split()[0] in {"the", "she", "they", "he", "we"}
            for s in SMOKE_STIMULI["c"]
        )


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


def _real_tokenizer(config):
    """The tokenizer ``train_topolm`` builds from ``config``, arguments verbatim."""
    return C.create_tokenizer(
        config.encoding_name,
        config.max_seq_length,
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
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

    def test_the_requirement_includes_the_special_ids_of_the_real_tokenizer(self):
        """The base size (50257) is not the requirement; the special ids are.

        The config assigns cls 50257, sep 50258, pad 50259 and mask 50260, all
        above the Tiktoken gpt2 base vocabulary, and padding is ``max_length`` so
        the pad id reaches every batch. A 50257-row table therefore dies on the
        first batch with ``indices[..] = 50259 is not in [0, 50257)``.
        """
        preprocessor = _real_tokenizer(_config())
        assert preprocessor.vocab_size == 50257
        assert C._required_vocab_size(preprocessor) == 50261

    @pytest.mark.parametrize("rows", [50257, 50260])
    def test_a_table_that_misses_a_special_id_is_rejected_by_name(self, rows):
        config = _config(vocab_size=rows)
        preprocessor = _real_tokenizer(config)
        with pytest.raises(ValueError, match=r"--vocab-size") as excinfo:
            C._require_vocab_covers_tokenizer(config, preprocessor)
        message = str(excinfo.value)
        assert f"vocab_size {rows}" in message
        assert "50261" in message
        assert "special" in message
        # And the model-side check, fed the value train_topolm feeds it.
        with pytest.raises(ValueError, match=r"--vocab-size"):
            build_backbone(
                config,
                required_vocab_size=C._required_vocab_size(preprocessor),
            )

    @pytest.mark.parametrize("rows", [50261, 100277])
    def test_a_table_that_holds_every_id_is_accepted(self, rows):
        config = _config(vocab_size=rows)
        preprocessor = _real_tokenizer(config)
        C._require_vocab_covers_tokenizer(config, preprocessor)
        # The model-side check accepts the same table at the true minimum.
        build_backbone(config, required_vocab_size=50261)

    def test_a_tokenizer_without_special_ids_keeps_its_base_size(self):
        class _Tok:
            vocab_size = 100277

        assert C._required_vocab_size(_Tok()) == 100277
        assert C._required_vocab_size(object()) is None

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


_VARIANTS_3D = sorted(
    name for name, row in MODEL_VARIANTS.items() if row.get("grid_dim", 2) == 3
)


class TestThreeDimensionalVariantsTrain:
    """A 3D row builds at its OWN radius and trains; nothing refuses it.

    Rewritten deliberately from ``TestThreeDimensionalVariantsAreRefused`` (plan
    972a9ac5 step 5, D-005 and D-006). The refusal existed because the trainer
    overrode the row's radius with ``spatial_radius=5`` (no 3D width holds that)
    and because evaluation, plots and the readout were 2D-only. The radius is
    now resolved from the row, the metrics and plots follow the grid's rank, and
    the readout arm is skipped. What stays pinned: the rank comes from the
    variant (no grid flag), an explicit radius the grid cannot hold is still an
    error from the model, and an unknown variant is still reported by the model.
    Removed with the refusal, because the behaviour they pinned is gone:
    ``test_the_refusal_lists_the_variants_that_do_work`` and
    ``test_the_rank_of_the_built_backbone_is_checked_too``.
    """

    def test_the_table_has_the_two_rows_this_class_is_about(self):
        assert _VARIANTS_3D == ["paper3d", "tiny3d"]

    @pytest.mark.parametrize(
        "name,grid,radius",
        [("paper3d", (10, 10, 10), 4), ("tiny3d", (7, 7, 7), 2)],
    )
    def test_build_backbone_builds_at_the_rows_radius(self, name, grid, radius):
        backbone = build_backbone(
            _config(model_variant=name, spatial_radius=None, num_layers=1)
        )
        assert tuple(backbone.grid_shape) == grid
        assert backbone.radius == radius
        assert backbone.grid_dim == 3

    def test_a_radius_the_grid_cannot_hold_is_still_an_error_from_the_model(self):
        """The trainer no longer pre-empts it, so the model's own message (which
        names the largest usable radius) is what the user reads."""
        with pytest.raises(ValueError, match=r"largest radius this grid can use is 3"):
            build_backbone(_config(model_variant="tiny3d", spatial_radius=5))

    def test_an_unknown_variant_is_still_reported_by_the_model(self):
        with pytest.raises(ValueError, match=r"Unknown variant 'nope'"):
            build_backbone(_config(model_variant="nope"))

    def test_train_topolm_writes_the_resolved_radius_and_a_3d_report(
        self, monkeypatch
    ):
        """Through the REAL ``train_topolm`` on ``tiny3d`` with ``spatial_radius``
        left unset: ``config.json`` records the integer the model was built with
        (the other half of D-006: resolving only inside ``build_backbone`` would
        write ``null``), the report's ``grid_shape`` is a 3-list and only the raw
        arm exists. ``paper3d`` trains in the GPU step; its build is pinned above.
        """
        import json

        _patch_data(monkeypatch)
        result = train_topolm(
            _config(model_variant="tiny3d", spatial_radius=None, num_layers=1),
            preprocessor=StubTokenizer(),
        )
        written = json.loads(
            open(os.path.join(result["results_dir"], "config.json")).read()
        )
        assert written["spatial_radius"] == 2
        assert result["backbone"].radius == 2
        report = result["topography"]
        assert report["grid_shape"] == [7, 7, 7]
        assert list(report["arms"]) == ["raw"]

    def test_train_paired_reaches_both_arms_for_a_3d_variant(self, monkeypatch):
        """Nothing in the paired driver refuses a 3D row now: both arms are
        handed to ``train_topolm`` (stubbed here; the arm itself is covered above)."""
        seen = []

        def _fake_train(config, preprocessor=None, run_topography=True):
            seen.append((config.model_variant, config.spatial_alpha))
            return {"summary": {}}

        monkeypatch.setattr(C, "train_topolm", _fake_train)
        train_paired(_config(model_variant="tiny3d", spatial_radius=None))
        assert seen == [("tiny3d", 2.5), ("tiny3d", 0.0)]

    @pytest.mark.parametrize("name", _VARIANTS_3D)
    def test_the_cli_entry_hands_the_trainer_the_resolved_3d_config(
        self, monkeypatch, tmp_path, name
    ):
        """Through the real ``main``: argument parsing and ``_config_from_args``,
        with the trainer itself replaced by a recorder (``setup_gpu`` too, which
        configures the process's devices)."""
        captured = {}

        def _record(config, run_topography=True):
            captured["config"] = config
            return {"results_dir": str(tmp_path)}

        monkeypatch.setattr(pretrain_cli, "setup_gpu", lambda gpu_id=None: None)
        monkeypatch.setattr(pretrain_cli, "train_topolm", _record)
        monkeypatch.setattr(
            "sys.argv",
            ["pretrain", "--variant", name, "--save-dir", str(tmp_path / "out")],
        )
        pretrain_cli.main()
        config = captured["config"]
        assert config.spatial_radius == MODEL_VARIANTS[name]["radius"]
        assert config.readout_fwhm is None

    def test_the_cli_adds_no_grid_flag_and_still_offers_every_variant(self):
        parser = pretrain_cli._build_parser()
        options = {o for a in parser._actions for o in a.option_strings}
        assert not any("grid" in o for o in options)
        variant = next(a for a in parser._actions if "--variant" in a.option_strings)
        assert set(variant.choices) == set(MODEL_VARIANTS)


class TestTheGridIsReportedAtItsOwnRank:
    """``height, width = backbone.grid_shape`` and ``[g[0], g[1]]`` were the two
    rank-2 lines in the trainer. They are now rank-agnostic, with the 2D output
    unchanged: the report is ``[h, w]`` and the log reads ``grid=HxW``."""

    @pytest.mark.parametrize(
        "grid,label",
        [((16, 16), "16x16"), ((16, 32), "16x32"), ((7, 7, 7), "7x7x7"),
         ((7, 8, 14), "7x8x14")],
    )
    def test_the_grid_label_joins_every_axis(self, grid, label):
        assert C._grid_label(grid) == label

    def test_the_backbone_log_line_still_reads_grid_HxW(self, monkeypatch):
        lines = []
        monkeypatch.setattr(C.logger, "info", lambda message, *a, **k: lines.append(message))
        build_backbone(_config())
        assert any("grid=16x16," in line for line in lines), lines

    @pytest.mark.parametrize("grid", [(16, 32), (2, 3, 4)], ids=["2D", "3D"])
    def test_the_report_carries_every_axis(self, monkeypatch, grid):
        """Isolated from the model: a stub backbone with no taps, and the two
        helpers that need taps replaced, so the report dict is all that is left.
        (Through a real 3D backbone ``grow_clusters`` would refuse first.)"""
        import types

        monkeypatch.setattr(C, "extract_tap_activations", lambda *a, **k: {})
        monkeypatch.setattr(C, "fdr_across_layers", lambda *a, **k: {})
        stub = types.SimpleNamespace(grid_shape=grid, tap_layers=[])
        report, _ = evaluate_topography(
            stub,
            stimuli=SMOKE_STIMULI,
            contrast_conditions=("a", "b"),
            preprocessor=None,
            readout_fwhm=None,
            min_cluster_size=2,
            is_smoke_set=False,
        )
        assert report["grid_shape"] == list(grid)


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

        Rewritten deliberately for D-001. The expected arrays used to be
        ``contrast_tmap(...)[0].reshape(grid)``, i.e. the test pinned the same
        unit-order reshape the code under test had, which is the wrong map under
        the trainer's default ``permute=True`` (unit ``i`` sits at cell
        ``perm[i]``). The expected arrays are now placed through
        ``layout.positions`` by ``_on_grid_by_positions``, a plain loop that
        shares neither ``to_grid`` nor ``cell_to_unit`` with the code under test.
        The arm-versus-arm discrimination this test exists for is unchanged.
        """
        backbone, report, arrays = self._report()

        activations = {
            condition: extract_tap_activations(backbone, prompts, StubTokenizer())
            for condition, prompts in SMOKE_STIMULI.items()
        }
        tap_layout = backbone.tap_layers[0].layout
        tap_path = backbone.tap_layers[0].path

        expected = {}
        expected["raw"] = _on_grid_by_positions(tap_layout, contrast_tmap(
            activations["a"][tap_path], activations["b"][tap_path]
        )[0])
        smoothed = {
            condition: C._apply_readout(values, backbone, 2.0, 1.0)
            for condition, values in activations.items()
        }
        expected["readout"] = _on_grid_by_positions(tap_layout, contrast_tmap(
            smoothed["a"][tap_path], smoothed["b"][tap_path]
        )[0])

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
        # The figures are saved with ``bbox_inches="tight"`` (plotting.py), so
        # their pixel extents follow the tick and colorbar labels, which follow
        # the data -- and the backbone is not seeded, so a few rows more or
        # fewer per run is normal (MEASURED: 493 vs 497 rows on 2 of 10 seeds;
        # an earlier `assert raw.shape == readout.shape` failed on those). The
        # claim here is that the two arms are DRAWN from different arrays, not
        # that their canvases are the same size, so compare on the common
        # extent. Identical arrays give identical layouts, hence identical
        # shapes and a zero difference: the assertion below still fails for the
        # defect this test exists for.
        height = min(raw.shape[0], readout.shape[0])
        width = min(raw.shape[1], readout.shape[1])
        raw, readout = raw[:height, :width], readout[:height, :width]
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
            assert set(arm_arrays) == {
                "t_maps", "sig_grids", "label_grids", "label_grids_b"
            }
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

# =====================================================================
# plan-2026-10-23T190557-972a9ac5, step 1: pins and RED tests
#
# ``TestTwoDimensionalUnchanged*`` record what the trainer produces for a 2-D
# variant at commit 9a1a446ec and stay green through every later step (the only
# permitted change is the additive ``viz_every_steps`` field). The remaining
# classes were RED when written, each for the reason in its docstring; the old
# tests that pin the behaviour being replaced are NOT edited here (step 3 and
# step 5 rewrite them, with the reason in each docstring).
# =====================================================================

import types

from scipy import ndimage

from dl_techniques.layers.regularization.spatial_smoothness import SpatialLayout
from dl_techniques.metrics.spatial_autocorrelation import morans_i_grid
from dl_techniques.metrics.topographic_selectivity import fdr_across_layers

from .test_cli_contract import BASELINE_2D_CONFIG

_PLANT_STIMULI = {
    "a": [f"a sentence {i}" for i in range(5)],
    "b": [f"b sentence {i}" for i in range(5)],
}
#: Zero-mean jitter, identical in every unit and condition ``a`` only. With
#: condition ``b`` held at zero, the Student t of a unit with effect ``e`` is
#: ``e / 0.0707`` (pooled variance 0.0125, five rows each), i.e. linear in ``e``
#: and exactly ``0`` for ``e = 0``.
_JITTER = 0.1 * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def _stub_backbone(
    num_units=256, grid_dim=2, permute=True, seeds=(11, 12, 13), grid_shape=None
):
    """A model-free backbone: what ``evaluate_topography`` reads and nothing else.

    Real :class:`SpatialLayout` objects, one per tap with its own seed, so each
    tap has its own unit order -- but no weights, no tokenizer and no forward, so
    the test controls the activations exactly (see :func:`_plant`).
    """
    layouts = [
        SpatialLayout(
            num_units, grid_shape=grid_shape, permute=permute, seed=seed,
            grid_dim=grid_dim,
        )
        for seed in seeds
    ]
    taps = [
        types.SimpleNamespace(path=f"blocks.{index}.ffn/attn_pre", layout=layout)
        for index, layout in enumerate(layouts)
    ]
    return types.SimpleNamespace(
        grid_shape=layouts[0].grid_shape, grid_dim=grid_dim, tap_layers=taps,
        built=True,
    )


def _plant(monkeypatch, backbone, effects):
    """Replace the forward pass with planted activations; return them per condition.

    Condition ``a`` is ``effect + jitter`` and condition ``b`` is zero, per tap
    and per unit, so the t-value of a unit is a known function of its effect and
    nothing else. ``effects`` is ``{tap_path: (num_units,)}`` in UNIT order.
    Returns ``{condition: {tap_path: (5, num_units)}}``, the arrays the patched
    ``extract_tap_activations`` hands out, so an oracle can start from them.
    """
    def activations_for(prompts):
        is_a = prompts[0].startswith("a")
        activations = {}
        for tap in backbone.tap_layers:
            effect = np.asarray(effects[tap.path], dtype="float64")
            rows = np.zeros((len(prompts), effect.size))
            if is_a:
                rows = rows + effect[None, :] + _JITTER[:, None]
            activations[tap.path] = rows
        return activations

    monkeypatch.setattr(
        C, "extract_tap_activations",
        lambda backbone_, prompts, preprocessor, batch_size=32: activations_for(
            list(prompts)
        ),
    )
    return {
        condition: activations_for(prompts)
        for condition, prompts in _PLANT_STIMULI.items()
    }


def _evaluate_planted(backbone, readout_fwhm=None, **overrides):
    settings = dict(
        stimuli=_PLANT_STIMULI, contrast_conditions=("a", "b"),
        preprocessor=None, readout_fwhm=readout_fwhm, min_cluster_size=2,
    )
    settings.update(overrides)
    return evaluate_topography(backbone, **settings)


def _block_effect(layout, lows, highs, value=1.0):
    """Unit-order effects: ``value`` for units whose POSITION lies in the box."""
    effect = np.zeros(layout.num_units)
    for unit in range(layout.num_units):
        cell = tuple(int(c) for c in layout.positions[unit])
        if all(lo <= c < hi for c, lo, hi in zip(cell, lows, highs)):
            effect[unit] = value
    return effect


def _on_grid_by_positions(layout, per_unit):
    """The expected map: a plain loop through ``layout.positions``.

    Shares nothing with the code under test (no ``to_grid``, no
    ``cell_to_unit``, no reshape), which is the whole point of an oracle.
    """
    grid = np.zeros(layout.grid_shape, dtype=np.asarray(per_unit).dtype)
    for unit in range(layout.num_units):
        grid[tuple(int(c) for c in layout.positions[unit])] = per_unit[unit]
    return grid


def _expected_arm(backbone, activations, arm_activations=None):
    """Per tap: the unit-order t and BH-FDR rejection, then placed by positions."""
    from dl_techniques.metrics.topographic_selectivity import contrast_tmap

    t_units, p_units = {}, {}
    for tap in backbone.tap_layers:
        source = (arm_activations or activations)
        t_units[tap.path], p_units[tap.path] = contrast_tmap(
            source["a"][tap.path], source["b"][tap.path]
        )
    rejected = fdr_across_layers(p_units, alpha=0.05)
    return {
        tap.path: (
            _on_grid_by_positions(tap.layout, t_units[tap.path]),
            _on_grid_by_positions(
                tap.layout, np.asarray(rejected[tap.path]["reject"], dtype=bool)
            ),
        )
        for tap in backbone.tap_layers
    }


class TestTheMapPlacementFollowsTheLayout:
    """D-001: unit ``i`` sits at cell ``layout.positions[i]``, so a unit-order
    t-vector must be SCATTERED onto the grid, not reshaped.

    RED on the baseline for the reason the plan names: ``evaluate_topography``
    reshapes (``t_maps[name].reshape(grid_shape)``), which is only the right
    placement when the layout is the identity. Every oracle here is a loop
    through ``layout.positions``.
    """

    def _blocks(self, backbone):
        """A different 5x5 block per tap, as (low, high) boxes on the 16x16 grid."""
        return {
            tap.path: ((3 + k, 4 + 2 * k), (8 + k, 9 + 2 * k))
            for k, tap in enumerate(backbone.tap_layers)
        }

    def test_the_raw_arm_puts_t_and_significance_on_the_cells_of_their_units(
        self, monkeypatch
    ):
        backbone = _stub_backbone()
        boxes = self._blocks(backbone)
        effects = {
            tap.path: _block_effect(tap.layout, *boxes[tap.path])
            for tap in backbone.tap_layers
        }
        activations = _plant(monkeypatch, backbone, effects)

        report, arrays = _evaluate_planted(backbone)

        expected = _expected_arm(backbone, activations)
        for tap in backbone.tap_layers:
            t_expected, sig_expected = expected[tap.path]
            t_drawn = arrays["arms"]["raw"]["t_maps"][tap.path]
            sig_drawn = arrays["arms"]["raw"]["sig_grids"][tap.path]
            np.testing.assert_allclose(t_drawn, t_expected, rtol=1e-9, atol=1e-9)
            np.testing.assert_array_equal(sig_drawn, sig_expected)
            # The planted block IS the significant set (25 cells), on the grid.
            lows, highs = boxes[tap.path]
            block = np.zeros(tap.layout.grid_shape, dtype=bool)
            block[lows[0]:highs[0], lows[1]:highs[1]] = True
            np.testing.assert_array_equal(sig_drawn, block)

    def test_the_planted_block_is_one_cluster_on_its_own_cells(self, monkeypatch):
        """Clusters grown on a scrambled map are scattered singletons; on the
        placed map the block is ONE cluster of 25 whose label grid is the block."""
        backbone = _stub_backbone()
        boxes = self._blocks(backbone)
        effects = {
            tap.path: _block_effect(tap.layout, *boxes[tap.path])
            for tap in backbone.tap_layers
        }
        _plant(monkeypatch, backbone, effects)

        report, arrays = _evaluate_planted(backbone)

        for tap in backbone.tap_layers:
            entry = report["arms"]["raw"]["per_tap"][tap.path]
            assert entry["cluster_sizes_a"] == [25], entry["cluster_sizes_a"]
            assert entry["cluster_sizes_b"] == []
            lows, highs = boxes[tap.path]
            labels = arrays["arms"]["raw"]["label_grids"][tap.path]
            assert {tuple(int(v) for v in c) for c in np.argwhere(labels > 0)} == {
                (r, c)
                for r in range(lows[0], highs[0])
                for c in range(lows[1], highs[1])
            }

    def test_the_readout_arm_is_placed_the_same_way(self, monkeypatch):
        """The readout returns activations in UNIT order (pinned below), so its
        t-map needs the same scatter; the expected map blurs the same planted
        activations with the same readout and places them by positions."""
        backbone = _stub_backbone()
        boxes = self._blocks(backbone)
        effects = {
            tap.path: _block_effect(tap.layout, *boxes[tap.path])
            for tap in backbone.tap_layers
        }
        activations = _plant(monkeypatch, backbone, effects)
        blurred = {
            condition: C._apply_readout(values, backbone, 2.0, 1.0)
            for condition, values in activations.items()
        }

        _, arrays = _evaluate_planted(backbone, readout_fwhm=2.0)

        expected = _expected_arm(backbone, activations, arm_activations=blurred)
        for tap in backbone.tap_layers:
            t_expected, sig_expected = expected[tap.path]
            np.testing.assert_allclose(
                arrays["arms"]["readout"]["t_maps"][tap.path], t_expected,
                rtol=1e-9, atol=1e-9,
            )
            np.testing.assert_array_equal(
                arrays["arms"]["readout"]["sig_grids"][tap.path], sig_expected
            )
        # The two arms must also differ, or the readout assertion above would be
        # satisfied by the raw map.
        tap = backbone.tap_layers[0].path
        assert not np.allclose(
            arrays["arms"]["raw"]["t_maps"][tap],
            arrays["arms"]["readout"]["t_maps"][tap],
        )

    def test_a_field_smooth_on_the_layout_scores_high_placement_permute_true(
        self, monkeypatch
    ):
        """The user-visible symptom: a map that is smooth on the layout grid has a
        high Moran's I only if it is placed on that grid. MEASURED on 28x28 by the
        explorer: 0.7927 through ``to_grid`` against 0.0005 through ``reshape``;
        here on 16x16: 0.7541 for the field itself against -0.0048 on the
        baseline, whose reshape scrambles it."""
        backbone = _stub_backbone()
        field = ndimage.gaussian_filter(
            np.random.default_rng(3).normal(size=(16, 16)), 2.0, mode="nearest"
        )
        effects = {
            tap.path: np.array(
                [field[tuple(int(c) for c in tap.layout.positions[u])]
                 for u in range(256)]
            )
            for tap in backbone.tap_layers
        }
        _plant(monkeypatch, backbone, effects)

        report, arrays = _evaluate_planted(backbone)

        # t is linear in the effect, so its Moran's I is the field's own.
        target = morans_i_grid(field)
        assert target > 0.5, "the fixture is not smooth enough to discriminate"
        for tap in backbone.tap_layers:
            standard = report["arms"]["raw"]["per_tap"][tap.path]["morans_i"]["standard"]
            assert standard == pytest.approx(target, abs=1e-9)
            drawn = arrays["arms"]["raw"]["t_maps"][tap.path]
            assert morans_i_grid(drawn) == pytest.approx(target, abs=1e-9)


class TestTwoDimensionalUnchangedPlacementWithoutPermutation:
    """The half of D-001 that must NOT move: with an identity layout the reshape
    and the scatter are the same map, and the correct code keeps returning it."""

    def test_permute_false_old_reshape_and_new_placement_agree(self, monkeypatch):
        backbone = _stub_backbone(permute=False)
        effects = {
            tap.path: _block_effect(tap.layout, (3 + k, 4 + k), (8 + k, 9 + k))
            for k, tap in enumerate(backbone.tap_layers)
        }
        activations = _plant(monkeypatch, backbone, effects)

        _, arrays = _evaluate_planted(backbone)

        from dl_techniques.metrics.topographic_selectivity import contrast_tmap

        for tap in backbone.tap_layers:
            t_units = contrast_tmap(
                activations["a"][tap.path], activations["b"][tap.path]
            )[0]
            drawn = arrays["arms"]["raw"]["t_maps"][tap.path]
            np.testing.assert_allclose(drawn, t_units.reshape(16, 16), atol=1e-12)
            np.testing.assert_allclose(
                drawn, _on_grid_by_positions(tap.layout, t_units), atol=1e-12
            )
            assert drawn.max() > 5.0, "the planted block was not detected"

    def test_the_readout_returns_unit_order_so_it_needs_the_same_scatter(self):
        """The assumption D-001 rests for the readout arm (plan Assumption 2):
        an impulse on unit ``u`` blurs into a bump around ``positions[u]`` on the
        grid, i.e. ``_apply_readout`` hands back UNIT order, not grid order. Seen
        through positions only; the same vector reshaped has its peak elsewhere."""
        layout = SpatialLayout(256, seed=7)
        backbone = types.SimpleNamespace(
            tap_layers=[types.SimpleNamespace(path="t0", layout=layout)]
        )
        impulse = np.zeros((1, 256))
        impulse[0, 37] = 1.0
        blurred = C._apply_readout({"t0": impulse}, backbone, 2.0, 1.0)["t0"][0]

        by_position = _on_grid_by_positions(layout, blurred)
        assert tuple(np.unravel_index(by_position.argmax(), by_position.shape)) == tuple(
            int(c) for c in layout.positions[37]
        )
        near = [
            u for u in range(256)
            if np.abs(layout.positions[u] - layout.positions[37]).max() <= 2
        ]
        assert blurred[near].sum() / blurred.sum() > 0.99
        assert tuple(np.unravel_index(blurred.argmax(), (16, 16))) != tuple(
            int(c) for c in layout.positions[37]
        ), "the permutation is the identity here, so this proves nothing"


# ---------------------------------------------------------------------
# 2-D pins that must survive the 3-D enablement
# ---------------------------------------------------------------------


class TestTwoDimensionalUnchangedReport:
    """The shape of ``topography_report.json`` and of the figure set for a 2-D run."""

    def _report(self):
        backbone = build_backbone(_config())
        return evaluate_topography(
            backbone, stimuli=SMOKE_STIMULI, contrast_conditions=("a", "b"),
            preprocessor=StubTokenizer(), readout_fwhm=2.0, min_cluster_size=2,
        )

    def test_the_report_has_exactly_these_keys_at_every_level(self):
        report, arrays = self._report()
        assert sorted(report) == [
            "arms", "conditions", "contrast", "fdr_alpha", "fdr_scope",
            "grid_shape", "num_taps", "stimuli_are_smoke_set",
        ]
        assert sorted(report["arms"]) == ["raw", "readout"]
        for arm in report["arms"].values():
            assert sorted(arm) == [
                "mean_morans_i", "mean_num_significant_units", "per_tap",
                "readout_fwhm", "total_clusters_a", "total_clusters_b",
            ]
            for tap in arm["per_tap"].values():
                assert sorted(tap) == [
                    "cluster_sizes_a", "cluster_sizes_b", "morans_i",
                    "num_significant_units",
                ]
                assert sorted(tap["morans_i"]) == [
                    "connectivity", "islands", "num_permutations", "num_units",
                    "permutation_p", "standard",
                ]
        assert report["grid_shape"] == [16, 16]
        assert report["fdr_scope"] == "joint across all taps"
        assert sorted(arrays) == ["arms"]
        assert sorted(arrays["arms"]["raw"]) == [
            "label_grids", "label_grids_b", "sig_grids", "t_maps",
        ]

    def test_the_raw_only_report_has_the_same_keys_without_a_readout_section(self):
        backbone = build_backbone(_config())
        report, _ = evaluate_topography(
            backbone, stimuli=SMOKE_STIMULI, contrast_conditions=("a", "b"),
            preprocessor=StubTokenizer(), readout_fwhm=None, min_cluster_size=2,
        )
        assert sorted(report["arms"]) == ["raw"]
        assert "readout_skipped" not in report

    @staticmethod
    def _synthetic(with_cluster):
        """A report and arrays for two arms, drawn without a model."""
        names = [f"blocks.{i}.ffn/attn_pre" for i in range(3)]
        grid = np.random.default_rng(5).normal(size=(8, 8))
        labels = np.zeros((8, 8), dtype=int)
        if with_cluster:
            labels[2:5, 2:5] = 1
        report = {
            "contrast": ["a", "b"],
            "arms": {
                arm: {"per_tap": {n: {"morans_i": {"standard": 0.1 * i}}
                                  for i, n in enumerate(names)}}
                for arm in ("raw", "readout")
            },
        }
        arrays = {
            "arms": {
                arm: {
                    "t_maps": {n: grid for n in names},
                    "sig_grids": {n: np.abs(grid) > 1.5 for n in names},
                    "label_grids": {n: labels for n in names},
                }
                for arm in ("raw", "readout")
            }
        }
        return report, arrays

    def test_the_figure_files_are_named_per_arm(self, tmp_path):
        report, arrays = self._synthetic(with_cluster=True)
        written = plot_topography(report, arrays, str(tmp_path))
        assert sorted(os.path.basename(p) for p in written) == [
            "clusters_raw.png", "clusters_readout.png", "morans_i_raw.png",
            "morans_i_readout.png", "t_maps_raw.png", "t_maps_readout.png",
        ]
        assert sorted(os.listdir(tmp_path)) == sorted(
            os.path.basename(p) for p in written
        )

    def test_no_cluster_means_no_cluster_figure(self, tmp_path):
        report, arrays = self._synthetic(with_cluster=False)
        written = plot_topography(report, arrays, str(tmp_path))
        assert sorted(os.path.basename(p) for p in written) == [
            "morans_i_raw.png", "morans_i_readout.png", "t_maps_raw.png",
            "t_maps_readout.png",
        ]


class TestTwoDimensionalUnchangedConfig:
    def test_config_json_keys_are_the_baseline_fields(self, tmp_path):
        """``config.json`` is the dataclass, field for field. The one permitted
        addition (later) is ``viz_every_steps``; any other new, renamed or
        removed key fails here."""
        import json

        config = pretrain_cli._config_from_args(
            pretrain_cli._build_parser().parse_args([])
        )
        C.prepare_run_dir(
            config, output_dir=str(tmp_path), config_filename="config.json"
        )
        written = json.loads((tmp_path / "config.json").read_text())
        assert sorted(set(written) - {"viz_every_steps"}) == sorted(BASELINE_2D_CONFIG)
        if "viz_every_steps" in written:
            assert written["viz_every_steps"] == 0

    def test_a_2d_variant_keeps_radius_five_and_readout_two(self):
        """The config-level defaults the resolver must not move for 2-D rows."""
        parser = pretrain_cli._build_parser()
        for name in ("small", "tiny", "paper"):
            config = pretrain_cli._config_from_args(
                parser.parse_args(["--variant", name])
            )
            assert config.spatial_radius == 5, name
            assert config.readout_fwhm == 2.0, name


# ---------------------------------------------------------------------
# D-006: the radius comes from the variant row
# ---------------------------------------------------------------------


class TestRadiusResolution:
    """RED on the baseline: ``resolve_spatial_radius`` does not exist, and
    ``build_backbone`` always passes ``config.spatial_radius`` (default 5), which
    overrides the row's own radius and fits no 3-D width in the table."""

    def test_the_resolver_is_importable(self):
        import train.topolm.common as module

        assert callable(module.resolve_spatial_radius)

    @pytest.mark.parametrize(
        "variant,expected",
        [("small", 5), ("tiny", 5), ("paper", 5), ("tiny3d", 2), ("paper3d", 4)],
    )
    def test_an_unset_radius_is_the_rows_own_or_the_default(self, variant, expected):
        from train.topolm.common import resolve_spatial_radius

        config = TopoLMTrainingConfig(model_variant=variant, spatial_radius=None)
        assert resolve_spatial_radius(config) == expected

    @pytest.mark.parametrize("variant", ["small", "tiny3d", "paper3d"])
    def test_an_explicit_radius_wins_over_the_row(self, variant):
        from train.topolm.common import resolve_spatial_radius

        config = TopoLMTrainingConfig(model_variant=variant, spatial_radius=3)
        assert resolve_spatial_radius(config) == 3

    def test_the_dataclass_default_means_unset(self):
        assert TopoLMTrainingConfig().spatial_radius is None

    def test_a_3d_backbone_is_built_at_the_rows_radius(self):
        """No refusal and no override: ``tiny3d`` at its own radius 2."""
        backbone = build_backbone(
            _config(model_variant="tiny3d", spatial_radius=None, num_layers=2)
        )
        assert tuple(backbone.grid_shape) == (7, 7, 7)
        assert backbone.radius == 2


class TestVizEveryStepsField:
    """RED on the baseline: the field does not exist."""

    def test_it_defaults_to_off(self):
        assert TopoLMTrainingConfig().viz_every_steps == 0

    def test_it_is_additive_to_the_inherited_fields(self):
        names = [f.name for f in TopoLMTrainingConfig.__dataclass_fields__.values()]
        assert "viz_every_steps" in names
        assert names.index("viz_every_steps") > names.index("spatial_radius")


# =====================================================================
# Step 5.3 (plan-2026-10-23T190557-972a9ac5, decision D-018): the variant row
# is checked against the resolved radius BEFORE any run work, and the vocab
# message names only flags that exist. Written RED-first against 363aaaaa5.
# =====================================================================


class _NoRunWork:
    """Spies on everything a refused run must not reach: the seeds, the save
    directory, the tokenizer, the dataset and the run directory."""

    def __init__(self, monkeypatch, tmp_path):
        self.calls = []
        for name in ("set_seeds", "create_tokenizer", "load_train_val_datasets",
                     "prepare_run_dir"):
            def spy(*args, _name=name, **kwargs):
                self.calls.append(_name)
                raise AssertionError(f"{_name} was reached by a refused run")
            monkeypatch.setattr(C, name, spy)
        self.save_dir = tmp_path / "save"
        self.runs = tmp_path / "runs"

    def assert_nothing_happened(self):
        assert self.calls == []
        assert not self.save_dir.exists(), "the save directory was created"
        assert not self.runs.exists(), "a run directory was created"


class TestTheVariantIsCheckedAgainstTheRadiusFirst:
    @pytest.mark.parametrize(
        "variant,grid,largest",
        [("tiny3d", "7x7x7", 3), ("paper3d", "10x10x10", 4), ("tiny", "16x16", 7)],
    )
    def test_a_radius_the_grid_cannot_hold_is_refused_before_any_run_work(
        self, monkeypatch, tmp_path, variant, grid, largest
    ):
        guard = _NoRunWork(monkeypatch, tmp_path)
        radius = 9 if variant == "tiny" else 5
        config = _config(
            model_variant=variant, spatial_radius=radius,
            save_dir=str(guard.save_dir),
        )
        with pytest.raises(ValueError) as caught:
            C.train_topolm(config, preprocessor=StubTokenizer(), run_topography=False)
        message = str(caught.value)
        assert variant in message
        assert f"--radius {radius}" in message and f"radius-{radius}" in message
        assert grid in message
        assert f"largest radius this grid can use is {largest}" in message
        guard.assert_nothing_happened()

    def test_a_paired_run_is_refused_the_same_way(self, monkeypatch, tmp_path):
        guard = _NoRunWork(monkeypatch, tmp_path)
        config = _config(
            model_variant="tiny3d", spatial_radius=5, save_dir=str(guard.save_dir)
        )
        with pytest.raises(ValueError, match="tiny3d"):
            C.train_paired(config, preprocessor=StubTokenizer(), run_topography=False)
        guard.assert_nothing_happened()

    def test_the_command_line_path_is_refused_by_name(self, monkeypatch, tmp_path):
        guard = _NoRunWork(monkeypatch, tmp_path)
        args = pretrain_cli._build_parser().parse_args(
            ["--variant", "tiny3d", "--radius", "5", "--save-dir", str(guard.save_dir)]
        )
        config = pretrain_cli._config_from_args(args)
        with pytest.raises(ValueError, match="tiny3d") as caught:
            C.train_topolm(config)
        assert "7x7x7" in str(caught.value)
        guard.assert_nothing_happened()

    def test_a_short_axis_is_named_when_the_unit_count_suffices(self, monkeypatch):
        """784 units clear the 3-D floor of radius 4 (729) yet resolve to 7x8x14,
        whose first axis cannot hold a 9-wide patch: the message names the axis."""
        row = dict(MODEL_VARIANTS["tiny3d"], embed_dim=784, num_heads=8)
        monkeypatch.setitem(MODEL_VARIANTS, "probe3d", row)
        with pytest.raises(ValueError) as caught:
            C.check_variant_fits_radius(
                _config(model_variant="probe3d", spatial_radius=4)
            )
        message = str(caught.value)
        assert "7x8x14" in message and "axis 0 has 7" in message
        assert "largest radius this grid can use is 3" in message

    @pytest.mark.parametrize("radius", [2, 3])
    def test_a_radius_the_grid_holds_is_accepted(self, radius):
        C.check_variant_fits_radius(
            _config(model_variant="tiny3d", spatial_radius=radius)
        )

    @pytest.mark.parametrize("variant", sorted(MODEL_VARIANTS))
    def test_every_shipped_variant_passes_at_its_own_resolved_radius(self, variant):
        C.check_variant_fits_radius(
            _config(model_variant=variant, spatial_radius=None)
        )

    def test_an_unknown_variant_is_left_to_the_model_to_name(self):
        C.check_variant_fits_radius(_config(model_variant="no_such_variant"))

    @pytest.mark.parametrize("variant", ["tiny", "tiny3d"])
    @pytest.mark.parametrize("heads", [None, 3])
    def test_the_verdict_equals_whether_the_model_builds(self, variant, heads):
        """Not stricter and not looser than the model: for every radius and the
        heads override, the pre-check raises exactly when ``build_backbone``
        does (this is what keeps a 2-D config that builds today accepted)."""
        for radius in range(1, 9):
            config = _config(
                model_variant=variant, spatial_radius=radius, num_layers=1,
                num_heads=heads,
            )
            try:
                C.check_variant_fits_radius(config)
                checked = True
            except ValueError:
                checked = False
            try:
                build_backbone(config)
                built = True
            except ValueError:
                built = False
            assert checked == built, (variant, radius, heads, checked, built)


class TestTheVocabularyMessageNamesOnlyRealFlags:
    @staticmethod
    def _flags_in(message):
        import re

        return set(re.findall(r"--[a-z][a-z-]*", message))

    @staticmethod
    def _real_flags():
        return {
            option
            for action in pretrain_cli._build_parser()._actions
            for option in action.option_strings
        }

    def test_the_preprocessor_guard_points_at_remedies_that_exist(self):
        class _Tok:
            vocab_size = 100277

        with pytest.raises(ValueError) as caught:
            C._require_vocab_covers_tokenizer(_config(vocab_size=50257), _Tok())
        message = str(caught.value)
        assert "--vocab-size" in message and "100277" in message
        assert self._flags_in(message) <= self._real_flags(), (
            self._flags_in(message) - self._real_flags()
        )
        assert "encoding-name" not in message

    def test_the_backbone_guard_points_at_remedies_that_exist(self):
        with pytest.raises(ValueError) as caught:
            build_backbone(_config(vocab_size=50257), required_vocab_size=100277)
        assert self._flags_in(str(caught.value)) <= self._real_flags()


# =====================================================================
# Step 5.4 (plan-2026-10-23T190557-972a9ac5, decision D-019, defect D7): the
# early radius refusal reaches the user as a parser error (one message on
# stderr, exit code 2), not as a traceback. Written RED-first against 2650cb6f9.
# =====================================================================


class _CliSpies(_NoRunWork):
    """:class:`_NoRunWork` plus the two trainer entry points and the GPU set-up,
    which a refused command line must not reach either."""

    def __init__(self, monkeypatch, tmp_path):
        super().__init__(monkeypatch, tmp_path)
        for owner, name in ((pretrain_cli, "train_topolm"), (pretrain_cli, "train_paired"),
                            (pretrain_cli, "setup_gpu")):
            def spy(*args, _name=name, **kwargs):
                self.calls.append(_name)
                raise AssertionError(f"{_name} was reached by a refused command line")
            monkeypatch.setattr(owner, name, spy)


def _refused_message(argv_tail):
    """The exact text ``check_variant_fits_radius`` raises for ``argv_tail``."""
    args = pretrain_cli._build_parser().parse_args(argv_tail)
    config = pretrain_cli._config_from_args(args)
    with pytest.raises(ValueError) as caught:
        C.check_variant_fits_radius(config)
    return str(caught.value)


class TestTheRadiusRefusalIsAParserError:
    @pytest.mark.parametrize("extra", [[], ["--paired"]], ids=["plain", "paired"])
    def test_tiny3d_with_radius_5_exits_2_with_the_message_and_no_run_work(
        self, monkeypatch, tmp_path, capsys, extra
    ):
        guard = _CliSpies(monkeypatch, tmp_path)
        argv = ["--variant", "tiny3d", "--radius", "5", "--save-dir",
                str(guard.save_dir), *extra]
        expected = _refused_message(argv)
        with pytest.raises(SystemExit) as caught:
            pretrain_cli.main(argv)
        error = capsys.readouterr().err
        assert caught.value.code == 2
        assert "usage:" in error and "error:" in error
        assert expected in error, "the parser error must carry the refusal's own text"
        assert "largest radius this grid can use is 3" in error
        assert "--radius 5" in error and "tiny3d" in error and "7x7x7" in error
        assert "Traceback" not in error
        guard.assert_nothing_happened()       # no seeds, tokenizer, dataset, save dir

    def test_a_head_count_that_does_not_divide_the_width_is_refused_the_same_way(
        self, monkeypatch, tmp_path, capsys
    ):
        guard = _CliSpies(monkeypatch, tmp_path)
        argv = ["--variant", "tiny", "--num-heads", "5", "--save-dir", str(guard.save_dir)]
        expected = _refused_message(argv)
        with pytest.raises(SystemExit) as caught:
            pretrain_cli.main(argv)
        error = capsys.readouterr().err
        assert caught.value.code == 2 and expected in error and "divisible" in error
        guard.assert_nothing_happened()

    def test_other_refusals_are_unchanged(self, monkeypatch, tmp_path):
        """A 3-D variant with a typed readout is refused by ``_config_from_args``
        with a ``ValueError``, before the radius check, as before."""
        guard = _CliSpies(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="--readout-fwhm"):
            pretrain_cli.main(["--variant", "tiny3d", "--readout-fwhm", "2.0",
                               "--save-dir", str(guard.save_dir)])
        guard.assert_nothing_happened()

    def test_a_radius_the_grid_holds_still_reaches_the_trainer(
        self, monkeypatch, tmp_path
    ):
        captured = {}

        def _record(config, run_topography=True):
            captured["config"] = config
            return {"results_dir": str(tmp_path)}

        monkeypatch.setattr(pretrain_cli, "setup_gpu", lambda gpu_id=None: None)
        monkeypatch.setattr(pretrain_cli, "train_topolm", _record)
        pretrain_cli.main(["--variant", "tiny3d", "--radius", "3",
                           "--save-dir", str(tmp_path / "out")])
        assert captured["config"].spatial_radius == 3

    def test_main_still_reads_sys_argv_without_an_argument(self, monkeypatch, tmp_path):
        captured = {}

        def _record(config, run_topography=True):
            captured["config"] = config
            return {"results_dir": str(tmp_path)}

        monkeypatch.setattr(pretrain_cli, "setup_gpu", lambda gpu_id=None: None)
        monkeypatch.setattr(pretrain_cli, "train_topolm", _record)
        monkeypatch.setattr(
            "sys.argv", ["pretrain", "--variant", "tiny3d", "--save-dir", str(tmp_path / "o")]
        )
        pretrain_cli.main()
        assert captured["config"].model_variant == "tiny3d"

    def test_the_refusal_happens_before_the_gpu_is_touched(self, monkeypatch, tmp_path):
        """A command that cannot run should not allocate a GPU context first."""
        order = []
        monkeypatch.setattr(pretrain_cli, "setup_gpu", lambda gpu_id=None: order.append("gpu"))
        monkeypatch.setattr(
            pretrain_cli, "train_topolm",
            lambda config, run_topography=True: order.append("train") or {"results_dir": ""},
        )
        with pytest.raises(SystemExit):
            pretrain_cli.main(["--variant", "tiny3d", "--radius", "5",
                               "--save-dir", str(tmp_path / "o")])
        assert order == []
        pretrain_cli.main(["--variant", "tiny3d", "--save-dir", str(tmp_path / "o")])
        assert order == ["gpu", "train"]


# =====================================================================
# Step 5.6 (plan-2026-10-23T190557-972a9ac5, decision D-022, review pass 2
# concern 5): `--radius 0` and `--radius -2` passed the early refusal and the
# command line, and the model then refused them ("radius must be >= 1") only
# after the GPU set-up, the seeds, the save directory, the tokenizer and the
# dataset load. Written RED-first against ca08dedd0.
# =====================================================================


class TestARadiusBelowOneIsRefusedUpFront:
    @pytest.mark.parametrize("radius", [0, -2], ids=["zero", "negative"])
    @pytest.mark.parametrize("extra", [[], ["--paired"]], ids=["plain", "paired"])
    def test_the_command_line_exits_2_naming_the_radius_and_does_no_work(
        self, monkeypatch, tmp_path, capsys, radius, extra
    ):
        guard = _CliSpies(monkeypatch, tmp_path)
        argv = ["--variant", "tiny", "--radius", str(radius),
                "--save-dir", str(guard.save_dir), *extra]
        with pytest.raises(SystemExit) as caught:
            pretrain_cli.main(argv)
        error = capsys.readouterr().err
        assert caught.value.code == 2
        assert "usage:" in error and "error:" in error
        assert f"--radius {radius}" in error and "at least 1" in error
        assert "Traceback" not in error
        guard.assert_nothing_happened()       # no GPU, seeds, tokenizer, dataset, save dir

    @pytest.mark.parametrize("radius", [0, -2], ids=["zero", "negative"])
    @pytest.mark.parametrize("variant", ["tiny", "tiny3d", "no_such_variant"])
    def test_the_check_refuses_it_for_every_variant(self, variant, radius):
        """A radius below 1 is wrong whatever the grid is, so the variant does not
        matter (an unknown variant is still named by the model otherwise)."""
        with pytest.raises(ValueError) as caught:
            C.check_variant_fits_radius(
                _config(model_variant=variant, spatial_radius=radius)
            )
        message = str(caught.value)
        assert f"--radius {radius}" in message and "at least 1" in message
        assert "Nothing was started" in message

    @pytest.mark.parametrize("radius", [0, -2], ids=["zero", "negative"])
    def test_a_programmatic_run_is_refused_before_any_run_work(
        self, monkeypatch, tmp_path, radius
    ):
        guard = _NoRunWork(monkeypatch, tmp_path)
        config = _config(spatial_radius=radius, save_dir=str(guard.save_dir))
        with pytest.raises(ValueError, match=f"--radius {radius}"):
            C.train_topolm(config, preprocessor=StubTokenizer(), run_topography=False)
        with pytest.raises(ValueError, match=f"--radius {radius}"):
            C.train_paired(config, preprocessor=StubTokenizer(), run_topography=False)
        guard.assert_nothing_happened()

    @pytest.mark.parametrize("variant", ["tiny", "tiny3d"])
    @pytest.mark.parametrize("radius", [0, -1, -2])
    def test_the_verdict_still_equals_whether_the_model_builds(self, variant, radius):
        """Extends the 1..8 sweep below the floor: the check is exactly as strict
        as the model, which refuses these in its constructor."""
        config = _config(model_variant=variant, spatial_radius=radius, num_layers=1)
        with pytest.raises(ValueError):
            build_backbone(config)
        with pytest.raises(ValueError):
            C.check_variant_fits_radius(config)

    @pytest.mark.parametrize("variant", sorted(MODEL_VARIANTS))
    def test_a_radius_of_one_is_accepted_by_every_variant(self, variant):
        C.check_variant_fits_radius(
            _config(model_variant=variant, spatial_radius=1)
        )

    def test_a_radius_of_one_reaches_the_trainer_on_the_command_line(
        self, monkeypatch, tmp_path
    ):
        seen = {}

        def _record(config, run_topography=True):
            seen["radius"] = config.spatial_radius
            return {"results_dir": str(tmp_path)}

        monkeypatch.setattr(pretrain_cli, "setup_gpu", lambda gpu_id=None: None)
        monkeypatch.setattr(pretrain_cli, "train_topolm", _record)
        for variant in sorted(MODEL_VARIANTS):
            pretrain_cli.main(["--variant", variant, "--radius", "1",
                               "--save-dir", str(tmp_path / variant)])
            assert seen["radius"] == 1, variant

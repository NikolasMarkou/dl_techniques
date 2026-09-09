"""Guards for the H-Net trainer core (``src/train/hnet/common.py``).

What each group of tests exists to catch, and why the twin arms are there:

* **The ratio loss must reach the optimizer.** ``HNet.call`` contributes it
  through ``add_loss`` and stock ``fit()`` sums ``model.losses``. A test that
  only asserts ``model.losses`` is non-empty passes against a model that adds a
  CONSTANT ZERO, so the value is asserted non-zero AND an ``alpha = 0.0`` twin
  is asserted to read exactly zero. Neither arm passes alone.
* **Every CLI flag must reach the field it names.** A flag ``config_from_args``
  forgets to forward is a knob that silently does nothing -- the recurring
  defect class ``tests/test_train/test_config_fields_are_live.py`` was written
  for. The wiring is checked field by field from the dataclass's own field
  list, so a field added later without a flag fails rather than going unseen,
  and every probe value is asserted DIFFERENT from the default so no assertion
  can pass vacuously.
* **The monitor direction has ONE producer.** ``create_callbacks`` resolves it
  through ``resolve_monitor_mode``; ``train`` must not hand a mode in. The spy
  asserts the ABSENCE of ``monitor_mode`` at the call site as well as the
  resolved direction on the callbacks that came back.
* **``optimizer_builder`` renames the clipping keys.** A literal ``"clipnorm"``
  key is dropped silently, so the guard reads ``optimizer.clipnorm`` off the
  constructed object rather than trusting the config dict, and has a
  clip-disabled twin.

The corpus is always a handful of in-memory strings. Nothing here reads the
20 GB Wikipedia cache and nothing writes to repo-root ``results/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
from typing import Any, Dict, List

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.byte_lm import BYTE_VOCAB_SIZE
from dl_techniques.models.language.hnet.config import MODEL_VARIANTS
from dl_techniques.models.language.hnet.model import RATIO_LOSS_ALPHA, HNet
from train.common import resolve_monitor_mode
from train.hnet import common as hnet_common
from train.hnet.common import (
    ARCH_VARIANTS,
    DEV_VARIANT,
    RESULTS_DIR_PREFIX,
    TRAIN_MONITOR,
    HNetTrainingConfig,
    add_common_arguments,
    build_byte_datasets,
    build_model,
    build_optimizer,
    config_from_args,
    get_arch_config,
)

# ---------------------------------------------------------------------
# Tiny fixtures
# ---------------------------------------------------------------------

CORPUS: List[str] = [
    "H-Net reads raw bytes, and the boundaries are learned rather than tokenized. ",
    "Multi-byte text keeps the packer honest: éèê 你好 \U0001f9ea. ",
    "The quick brown fox jumps over the lazy dog, repeatedly and at length. ",
]


def tiny_config(**overrides: Any) -> HNetTrainingConfig:
    """A dev-scale config small enough for a real optimizer step on CPU."""
    kwargs: Dict[str, Any] = dict(
        arch_variant=DEV_VARIANT,
        seq_len=32,
        batch_size=2,
        epochs=1,
        steps_per_epoch=2,
        validation_steps=1,
        headdim=16,
        max_chunks=[8],
        shuffle_buffer=8,
    )
    kwargs.update(overrides)
    return HNetTrainingConfig(**kwargs)


def text_dataset(repeats: int = 40) -> tf.data.Dataset:
    """An in-memory corpus, long enough to fill several packed windows."""
    return tf.data.Dataset.from_tensor_slices(CORPUS * repeats)


def one_batch(config: HNetTrainingConfig):
    train_ds, _ = build_byte_datasets(text_dataset(), text_dataset(), config)
    return next(iter(train_ds.take(1)))


# ---------------------------------------------------------------------
# The byte pipeline
# ---------------------------------------------------------------------


class TestBytePipeline:
    """Shapes, dtypes and the causal shift the model consumes."""

    def test_the_pipeline_yields_the_shape_and_dtype_the_model_consumes(self):
        config = tiny_config()
        inputs, labels = one_batch(config)

        assert inputs.shape == (config.batch_size, config.seq_len)
        assert labels.shape == (config.batch_size, config.seq_len)
        assert inputs.dtype == tf.int32
        assert labels.dtype == tf.int32

        # The model's own contract: rank-2 integer ids in [0, vocab_size).
        model = build_model(config, steps_per_epoch=2)
        logits = model(inputs, training=False)
        assert tuple(logits.shape) == (
            config.batch_size, config.seq_len, BYTE_VOCAB_SIZE,
        )

    def test_every_id_is_a_byte(self):
        inputs, labels = one_batch(tiny_config())
        for tensor in (inputs, labels):
            values = np.asarray(tensor)
            assert values.min() >= 0
            assert values.max() < BYTE_VOCAB_SIZE

    def test_the_packed_window_is_one_byte_longer_than_the_model_sees(self):
        """The +1 the causal shift spends is the packer's, not the model's.

        Getting this backwards is silent: the batch would simply be one byte
        short and every shape assertion downstream would still hold.
        """
        config = tiny_config(seq_len=32)
        assert config.packed_window == 33
        inputs, _ = one_batch(config)
        assert inputs.shape[1] == 32

    def test_labels_are_the_inputs_shifted_by_one(self):
        """``labels[i] == inputs[i + 1]`` within a window -- the causal shift.

        Asserted on an UNSHUFFLED single-window pipeline so the two tensors of
        one batch are genuinely adjacent slices of the same byte stream.
        """
        config = tiny_config(batch_size=1, shuffle_buffer=1)
        inputs, labels = one_batch(config)
        i = np.asarray(inputs)[0]
        l = np.asarray(labels)[0]
        np.testing.assert_array_equal(i[1:], l[:-1])

    def test_a_different_corpus_gives_different_bytes(self):
        """Anti-vacuity twin: the pipeline reads its input, it does not
        fabricate a constant batch that would satisfy every assertion above."""
        config = tiny_config(batch_size=1, shuffle_buffer=1)
        a, _ = one_batch(config)

        other = tf.data.Dataset.from_tensor_slices(["zzzz" * 200] * 40)
        train_ds, _ = build_byte_datasets(other, other, config)
        b, _ = next(iter(train_ds.take(1)))

        assert not np.array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------
# The ratio loss
# ---------------------------------------------------------------------


class TestRatioLossReachesTheOptimizer:
    """``model.losses`` must be non-empty AND non-zero after a real step."""

    def test_the_ratio_loss_is_present_and_non_zero_after_one_real_step(self):
        config = tiny_config()
        model = build_model(config, steps_per_epoch=2)
        train_ds, val_ds = build_byte_datasets(
            text_dataset(), text_dataset(), config
        )
        history = model.fit(
            train_ds,
            epochs=1,
            steps_per_epoch=2,
            validation_data=val_ds,
            validation_steps=1,
            verbose=0,
        )
        assert np.isfinite(history.history["loss"][0])

        # `model.losses` after `fit` holds the SYMBOLIC tensor of the compiled
        # step, so the value is read from a real eager call on real batch data
        # -- the same code path `fit` traced.
        inputs, _ = next(iter(train_ds.take(1)))
        model(inputs, training=True)
        losses = [float(value) for value in model.losses]

        assert losses, "add_loss contributed nothing: the ratio loss is dead"
        assert len(losses) == 1
        assert losses[0] > 0.0, (
            f"the ratio loss reached model.losses but is {losses[0]} -- a "
            "constant-zero auxiliary loss satisfies a non-emptiness assertion "
            "while changing no gradient"
        )

    def test_the_alpha_zero_twin_reads_exactly_zero(self):
        """The 'these differ' twin: alpha scales the term it is supposed to."""
        config = tiny_config(ratio_loss_alpha=0.0)
        model = build_model(config, steps_per_epoch=2)
        inputs, _ = one_batch(config)
        model(inputs, training=True)
        losses = [float(value) for value in model.losses]

        assert losses, "the add_loss site vanished entirely at alpha=0"
        assert losses[0] == 0.0

    def test_a_larger_alpha_scales_the_term(self):
        """A second differ-twin, so the alpha wiring is not merely on/off."""
        inputs, _ = one_batch(tiny_config())

        def ratio_at(alpha: float) -> float:
            keras.utils.set_random_seed(17)
            model = build_model(
                tiny_config(ratio_loss_alpha=alpha), steps_per_epoch=2
            )
            model(inputs, training=True)
            return float(model.losses[0])

        small = ratio_at(RATIO_LOSS_ALPHA)
        large = ratio_at(10.0 * RATIO_LOSS_ALPHA)
        assert large == pytest.approx(10.0 * small, rel=1e-4)

    def test_the_trainer_default_alpha_IS_the_model_packages_constant(self):
        """The trainer's default must be the pinned constant, not a re-typed copy.

        `HNetTrainingConfig.ratio_loss_alpha` defaults to `RATIO_LOSS_ALPHA`, so a
        change to the constant changes every run launched from this CLI. Until
        2026-09-09 nothing here observed that: `RATIO_LOSS_ALPHA 0.03 -> 0.30` was
        MEASURED green across all 239 tests of this suite, because every alpha guard
        above passed a literal. `test_models/test_hnet/test_model.py` owns the value
        and effect pins; this arm owns the WIRING -- that the trainer reads the
        constant rather than carrying a second copy that can drift from it.
        """
        assert tiny_config().ratio_loss_alpha == RATIO_LOSS_ALPHA
        assert HNetTrainingConfig().ratio_loss_alpha == RATIO_LOSS_ALPHA
        assert RATIO_LOSS_ALPHA == 0.03

    def test_the_trainer_never_defines_a_custom_train_step(self):
        """A HARD repo invariant, asserted on the objects rather than the text."""
        assert "train_step" not in vars(HNet)
        assert not [
            name for name, _ in inspect.getmembers(hnet_common, inspect.isfunction)
            if name == "train_step"
        ]


# ---------------------------------------------------------------------
# args -> config wiring
# ---------------------------------------------------------------------

# One NON-DEFAULT probe value per config field. Every entry is asserted to
# differ from the field's default, so a forwarding bug cannot hide behind a
# value that happens to equal the default.
PROBE_VALUES: Dict[str, Any] = {
    "arch_variant": "hnet_1stage_L",
    "dataset_root": "/tmp/hnet-probe-root",
    "wikipedia_config": "20231101.simple",
    "seq_len": 64,
    "batch_size": 3,
    "epochs": 7,
    "steps_per_epoch": 11,
    "validation_steps": 5,
    "learning_rate": 1e-3,
    "final_learning_rate": 1e-4,
    "weight_decay": 0.07,
    "warmup_ratio": 0.11,
    "gradient_clip_norm": 2.5,
    "headdim": 32,
    "ratio_loss_alpha": 0.09,
    "target_ratio": 4.5,
    "max_chunks": [17],
    "min_article_length": 250,
    "max_train_samples": 9,
    "max_val_samples": 13,
    "val_fraction": 0.05,
    "shuffle_shards": 2,
    "shuffle_buffer": 64,
    "seed": 7,
    "patience": 3,
    "output_dir": "/tmp/hnet-probe-out",
}


def _flag(field_name: str) -> str:
    return "--" + field_name.replace("_", "-")


def _probe_argv() -> List[str]:
    argv: List[str] = []
    for name, value in PROBE_VALUES.items():
        argv.append(_flag(name))
        if isinstance(value, list):
            argv.extend(str(item) for item in value)
        else:
            argv.append(str(value))
    return argv


def _config_field_names() -> List[str]:
    return [f.name for f in dataclasses.fields(HNetTrainingConfig)]


class TestArgsReachTheConfig:
    """Every declared flag must land on the field it is named for."""

    def test_the_probe_table_covers_every_config_field(self):
        """A field added later without a probe fails HERE, loudly."""
        assert sorted(PROBE_VALUES) == sorted(_config_field_names())

    def test_every_probe_value_differs_from_the_default(self):
        """Anti-vacuity: an equal-to-default probe asserts nothing."""
        defaults = HNetTrainingConfig()
        same = [
            name for name, value in PROBE_VALUES.items()
            if getattr(defaults, name) == value
        ]
        assert same == []

    def test_every_config_field_has_a_cli_flag(self):
        parser = add_common_arguments(argparse.ArgumentParser())
        declared = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        missing = [
            _flag(name) for name in _config_field_names()
            if _flag(name) not in declared
        ]
        assert missing == []

    @pytest.mark.parametrize("field_name", _config_field_names())
    def test_the_flag_reaches_its_field(self, field_name: str):
        parser = add_common_arguments(argparse.ArgumentParser())
        config = config_from_args(parser.parse_args(_probe_argv()))
        assert getattr(config, field_name) == PROBE_VALUES[field_name]

    def test_omitting_every_flag_reproduces_the_dataclass_defaults(self):
        parser = add_common_arguments(argparse.ArgumentParser())
        assert config_from_args(parser.parse_args([])) == HNetTrainingConfig()

    def test_an_unknown_arch_variant_is_rejected_by_the_parser(self):
        parser = add_common_arguments(argparse.ArgumentParser())
        with pytest.raises(SystemExit):
            parser.parse_args(["--arch-variant", "not-a-variant"])


# ---------------------------------------------------------------------
# Architecture resolution
# ---------------------------------------------------------------------


class TestArchVariants:
    def test_every_advertised_choice_resolves_to_an_architecture(self):
        for name in ARCH_VARIANTS:
            assert get_arch_config(name).num_stages >= 2

    def test_the_dev_layout_is_not_smuggled_into_the_cited_variant_table(self):
        """``MODEL_VARIANTS`` holds transcribed reference configs only."""
        assert DEV_VARIANT not in MODEL_VARIANTS
        assert set(ARCH_VARIANTS) == {DEV_VARIANT} | set(MODEL_VARIANTS)

    def test_an_unknown_name_raises_and_lists_the_legal_ones(self):
        with pytest.raises(ValueError, match="unknown arch_variant"):
            get_arch_config("hnet_3stage_XXL")

    def test_the_dev_layout_is_the_only_untied_small_one(self):
        """The dev head is UNTIED, like all six reference variants.

        Tying it against the unit-standard-deviation embedding table starts the
        model at a MEASURED cross-entropy of 40.5 instead of ln(256) = 5.55,
        which would make a smoke run's loss trend meaningless.
        """
        assert get_arch_config(DEV_VARIANT).tie_embeddings is False
        model = build_model(tiny_config(), steps_per_epoch=2)
        inputs, labels = one_batch(tiny_config())
        logits = model(inputs, training=False)
        ce = float(
            keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
                labels, logits
            )
        )
        assert ce == pytest.approx(np.log(BYTE_VOCAB_SIZE), abs=0.5)


# ---------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------


class TestOptimizer:
    """``optimizer_builder`` renames the clipping keys; a literal 'clipnorm'
    key is dropped with no error at all."""

    def test_gradient_clipping_survives_the_key_rename(self):
        optimizer = build_optimizer(
            tiny_config(gradient_clip_norm=1.25), steps_per_epoch=10
        )
        assert optimizer.clipnorm == pytest.approx(1.25)

    def test_a_zero_clip_norm_disables_clipping(self):
        """The differ-twin: the field is read, not hard-coded to a constant."""
        optimizer = build_optimizer(
            tiny_config(gradient_clip_norm=0.0), steps_per_epoch=10
        )
        assert optimizer.clipnorm is None

    def test_the_optimizer_is_adamw_with_the_builder_s_variable_scope(self):
        """``optimizer_builder`` hard-codes ``name="AdamW"`` -- the optimizer's
        variable scope, and therefore a checkpoint-compatibility fact."""
        optimizer = build_optimizer(tiny_config(), steps_per_epoch=10)
        assert isinstance(optimizer, keras.optimizers.AdamW)
        assert optimizer.name == "AdamW"

    def test_weight_decay_reaches_the_optimizer_and_nothing_else(self):
        """AdamW decays; no layer carries a kernel regularizer as well.

        Both halves matter: decaying twice inflates the loss AND decays the
        parameter again (``src/train/CLAUDE.md``).
        """
        config = tiny_config(weight_decay=0.123)
        optimizer = build_optimizer(config, steps_per_epoch=10)
        assert optimizer.weight_decay == pytest.approx(0.123)

        model = build_model(config, steps_per_epoch=10)
        model(one_batch(config)[0], training=False)
        regularized = [
            layer for layer in model._flatten_layers()
            if getattr(layer, "kernel_regularizer", None) is not None
        ]
        assert regularized == []

    @staticmethod
    def _schedule_of(optimizer):
        """The schedule object the optimizer will actually consult.

        Keras 3's public ``optimizer.learning_rate`` EVALUATES the schedule at
        the current iteration rather than returning it
        (``BaseOptimizer._get_current_learning_rate``), so the object itself is
        only reachable through ``_learning_rate``. The assertion below pins the
        two together, so this is a reading of the live schedule and not of some
        detached second copy.
        """
        schedule = optimizer._learning_rate
        assert isinstance(
            schedule, keras.optimizers.schedules.LearningRateSchedule
        )
        assert float(optimizer.learning_rate) == pytest.approx(
            float(schedule(0)), rel=1e-6
        )
        return schedule

    def test_the_warmup_and_floor_reach_the_schedule(self):
        config = tiny_config(
            learning_rate=1e-3,
            final_learning_rate=1e-4,
            warmup_ratio=0.5,
            epochs=1,
        )
        schedule = self._schedule_of(build_optimizer(config, steps_per_epoch=100))

        start = float(schedule(0))
        peak = float(schedule(50))
        end = float(schedule(99))

        assert start < peak, "warmup did not raise the learning rate"
        assert peak == pytest.approx(1e-3, rel=1e-3)
        assert end == pytest.approx(1e-4, rel=0.05)

    def test_a_zero_warmup_ratio_starts_at_the_peak(self):
        """Differ-twin for the warmup arm above."""
        config = tiny_config(learning_rate=1e-3, warmup_ratio=0.0, epochs=1)
        schedule = self._schedule_of(build_optimizer(config, steps_per_epoch=100))
        assert float(schedule(0)) == pytest.approx(1e-3, rel=1e-3)


# ---------------------------------------------------------------------
# Compile + callbacks
# ---------------------------------------------------------------------


class TestCompileAndCallbacks:
    def test_the_compiled_loss_is_a_from_logits_sparse_crossentropy(self):
        model = build_model(tiny_config(), steps_per_epoch=2)
        loss = model.loss
        assert isinstance(loss, keras.losses.SparseCategoricalCrossentropy)
        assert loss.get_config()["from_logits"] is True

    def test_resolve_monitor_mode_maps_val_loss_to_min(self):
        assert resolve_monitor_mode(TRAIN_MONITOR) == "min"
        assert TRAIN_MONITOR == "val_loss"

    def test_resolve_monitor_mode_maps_an_accuracy_to_max(self):
        """The differ-twin: the resolver reads the NAME, it is not a constant
        'min' that would agree with the assertion above by accident."""
        assert resolve_monitor_mode("val_accuracy") == "max"

    def test_train_uses_the_hnet_prefix_and_lets_the_resolver_choose(
        self, tmp_path, monkeypatch
    ):
        """``train`` end to end on an in-memory corpus, into ``tmp_path``.

        Wikipedia is never touched (``build_datasets`` is replaced) and nothing
        is written under repo-root ``results/`` (``output_dir`` is ``tmp_path``).
        The spy asserts the ABSENCE of a ``monitor_mode`` argument as well as
        the direction that came back on the callbacks: passing an explicit mode
        would defeat ``resolve_monitor_mode`` while still producing 'min' here.
        """
        config = tiny_config(output_dir=str(tmp_path))
        captured: Dict[str, Any] = {}
        real_create_callbacks = hnet_common.create_callbacks

        def fake_build_datasets(cfg):
            train_ds, val_ds = build_byte_datasets(
                text_dataset(), text_dataset(), cfg
            )
            return train_ds, val_ds, 2

        def spy_create_callbacks(**kwargs):
            captured.update(kwargs)
            # The epoch ModelAnalyzer is a per-epoch weight/spectral dump; it is
            # off HERE only, to keep the test to seconds. Every other argument
            # is forwarded verbatim.
            return real_create_callbacks(include_analyzer=False, **kwargs)

        monkeypatch.setattr(hnet_common, "build_datasets", fake_build_datasets)
        monkeypatch.setattr(
            hnet_common, "create_callbacks", spy_create_callbacks
        )

        model, history, results_dir = hnet_common.train(config)

        assert captured["results_dir_prefix"] == RESULTS_DIR_PREFIX == "hnet"
        assert captured["monitor"] == TRAIN_MONITOR
        assert "monitor_mode" not in captured, (
            "train passed an explicit monitor_mode; resolve_monitor_mode is the "
            "ONE producer of a checkpoint-selection direction"
        )
        assert captured["output_root"] == str(tmp_path)
        assert captured["patience"] == config.patience

        assert results_dir.startswith(str(tmp_path))
        assert (tmp_path / "..").exists()
        assert "loss" in history.history
        assert isinstance(model, HNet)

        run_dir = tmp_path / results_dir.split("/")[-1]
        assert (run_dir / "config.json").is_file()

    def test_the_callbacks_select_on_a_minimized_monitor(self, tmp_path):
        """The direction actually reaches the callbacks, not just the kwargs."""
        callbacks, _ = hnet_common.create_callbacks(
            model_name="dev",
            results_dir_prefix=RESULTS_DIR_PREFIX,
            output_root=str(tmp_path),
            monitor=TRAIN_MONITOR,
            patience=3,
            include_analyzer=False,
        )
        checkpoints = [
            cb for cb in callbacks
            if isinstance(cb, keras.callbacks.ModelCheckpoint)
        ]
        assert checkpoints
        assert checkpoints[0].monitor_op is np.less


# ---------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------


class TestConfigValidation:
    def test_there_is_no_chunk_size_knob(self):
        """H17: the reference's Mamba-2 ``chunk_size`` is not a knob this port
        has, and a field nothing consumes is worse than no field at all."""
        assert "chunk_size" not in _config_field_names()

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"arch_variant": "nope"}, "unknown arch_variant"),
            ({"seq_len": 1}, "seq_len must be >= 2"),
            ({"batch_size": 0}, "batch_size must be positive"),
            ({"steps_per_epoch": 0}, "steps_per_epoch must be positive"),
            ({"learning_rate": 0.0}, "learning_rate must be positive"),
            (
                {"learning_rate": 1e-4, "final_learning_rate": 1e-3},
                "final_learning_rate must be positive",
            ),
            ({"weight_decay": -0.1}, "weight_decay must be non-negative"),
            ({"warmup_ratio": 1.0}, "warmup_ratio must be in"),
            ({"gradient_clip_norm": -1.0}, "gradient_clip_norm must be"),
            ({"ratio_loss_alpha": -1.0}, "ratio_loss_alpha must be"),
            ({"target_ratio": 1.0}, "target_ratio must exceed"),
            ({"min_article_length": -1}, "min_article_length must be"),
            ({"val_fraction": 0.0}, "val_fraction must be in"),
            ({"max_train_samples": 0}, "max_train_samples must be positive"),
            ({"max_chunks": [4, 4]}, "one entry per chunking level"),
            ({"max_chunks": [0]}, "max_chunks entry must be >= 1"),
        ],
    )
    def test_a_bad_knob_is_refused_at_construction(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            tiny_config(**overrides)

    def test_the_default_config_is_valid(self):
        """Anti-vacuity twin for the table above: the validator is not simply
        rejecting everything."""
        assert HNetTrainingConfig().arch_variant == DEV_VARIANT

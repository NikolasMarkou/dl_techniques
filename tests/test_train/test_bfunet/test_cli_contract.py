"""CLI contract guard for ``src/train/bfunet/train_convunext_denoiser.py``.

The defect class
----------------
A trainer declares a flag, ``--help`` advertises it, and ``config_from_args`` never
forwards it (or forwards it into the wrong field). The flag then silently does nothing
and ``config.json`` records the DEFAULT while the user believes they overrode it. The
bfunet trainers had no such guard: their ``main()`` used to build two hand-written
``TrainingConfig`` blocks, and 23 flags were dropped on the smoke branch for exactly
this reason (``test_smoke_and_explicit_flags.py``). This file pins the repaired surface
with the shared ``tests/test_train/_cli_contract.py`` driver:

- every flag reaches its ``TrainingConfig`` field with a probe value that is NOT the
  default (trap 1) and that no other row uses (trap 3);
- the row table equals the flags the REAL parser declares (trap 2), so a flag added
  without a row turns the completeness test RED;
- ``--gpu``, ``--dashboard``, ``--smoke`` and ``--deep-supervision`` are not config
  fields and are ``namespace_dest`` rows; the ``main()`` hops behind the first two and
  the config-time refusal behind the last are their own tests;
- the smoke mode drives EVERY row with ``--smoke`` in front, so a typed value that a
  smoke preset would overwrite fails on the row that types it. Each preset dest must
  own a row (``test_every_smoke_preset_dest_has_a_row_in_the_smoke_mode``).

Only the ConvUNeXt trainer is in scope (D-013): the unet and bfcnn mains are untouched
and their flags are not this parser's. Nothing here trains, loads a dataset or touches
a GPU; every ``--output-dir`` is a string that is never created, and the only ``main()``
runs stub ``setup_gpu``, ``train`` and ``build_dashboard_from_dir``.

Every expected literal is typed here, never imported from the trainer.
"""

from __future__ import annotations

import os
import sys
from dataclasses import fields
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

import train.bfunet.train_convunext_denoiser as trainer  # noqa: E402

from .._cli_contract import (  # noqa: E402,TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)


def _rows() -> Tuple[Row, ...]:
    """One row per declared flag (88 rows for 90 option strings: ``--test-eval`` and ``--model-analysis`` have two negations).

    Probes differ from the parser defaults, from every ``SMOKE_PRESET`` value (so the same
    table can also be driven behind ``--smoke``) and from each other. A row whose flag is
    only admissible with a companion flag carries the companion in its argv; the companion
    is not the row's subject and its own row has a different probe value.
    """
    return (
        # -- data / patches ---------------------------------------------------------
        Row(("--epochs",), ("--epochs", "7"), "epochs", 7),
        Row(("--batch-size",), ("--batch-size", "24"), "batch_size", 24),
        Row(("--patch-size",), ("--patch-size", "96"), "patch_size", 96),
        Row(("--channels",), ("--channels", "1"), "channels", 1),
        Row(("--patches-per-image",), ("--patches-per-image", "6"), "patches_per_image", 6),
        Row(("--max-train-files",), ("--max-train-files", "321"), "max_train_files", 321),
        Row(("--max-val-files",), ("--max-val-files", "123"), "max_val_files", 123),
        Row(("--steps-per-epoch",), ("--steps-per-epoch", "77"), "steps_per_epoch", 77),
        Row(("--validation-steps",), ("--validation-steps", "33"), "validation_steps", 33),
        Row(("--seed",), ("--seed", "1234"), "seed", 1234),
        Row(("--dataset-shuffle-buffer",), ("--dataset-shuffle-buffer", "512"),
            "dataset_shuffle_buffer", 512),
        Row(("--patch-shuffle-buffer",), ("--patch-shuffle-buffer", "256"),
            "patch_shuffle_buffer", 256),
        Row(("--no-augment",), ("--no-augment",), "augment_data", False),
        # -- optimisation -----------------------------------------------------------
        Row(("--mixed-precision",), ("--mixed-precision",), "mixed_precision", True),
        Row(("--learning-rate",), ("--learning-rate", "2.5e-4"), "learning_rate", 2.5e-4),
        Row(("--weight-decay",), ("--weight-decay", "0.02"), "weight_decay", 0.02),
        Row(("--warmup-epochs",), ("--warmup-epochs", "2"), "warmup_epochs", 2),
        Row(("--optimizer-type",), ("--optimizer-type", "sgd"), "optimizer_type", "sgd"),
        # The config refuses every schedule but cosine_decay, but compares
        # case-insensitively and stores the RAW string, so a differently-cased spelling is
        # the one admissible probe that is not the default.
        Row(("--lr-schedule-type",), ("--lr-schedule-type", "Cosine_Decay"),
            "lr_schedule_type", "Cosine_Decay"),
        Row(("--gradient-clipping",), ("--gradient-clipping", "0.5"), "gradient_clipping", 0.5),
        Row(("--early-stopping-patience",), ("--early-stopping-patience", "4"),
            "early_stopping_patience", 4),
        # -- noise and curriculum ---------------------------------------------------
        Row(("--no-clip",), ("--no-clip",), "clip_noise", False),
        Row(("--noise-sigma-min",), ("--noise-sigma-min", "0.01"), "noise_sigma_min", 0.01),
        Row(("--sigma-max-start",), ("--sigma-max-start", "0.03"), "sigma_max_start", 0.03),
        Row(("--sigma-max-end",), ("--sigma-max-end", "0.35"), "sigma_max_end", 0.35),
        Row(("--curriculum-schedule",), ("--curriculum-schedule", "cosine"),
            "curriculum_schedule", "cosine"),
        Row(("--curriculum-epochs",), ("--curriculum-epochs", "13"), "curriculum_epochs", 13),
        Row(("--multiplicative-noise",), ("--multiplicative-noise",), "noise_type",
            "multiplicative"),
        Row(("--composite-noise",), ("--composite-noise",), "noise_type", "composite"),
        Row(("--composite-additive-ratio",), ("--composite-additive-ratio", "0.7"),
            "composite_additive_ratio", 0.7),
        Row(("--symmetry-weight",), ("--symmetry-weight", "0.05"), "symmetry_weight", 0.05),
        Row(("--symmetry-probes",), ("--symmetry-probes", "3"), "symmetry_probes", 3),
        # -- model topology ---------------------------------------------------------
        Row(("--variant",), ("--variant", "small"), "variant", "small"),
        Row(("--convnext-version",), ("--convnext-version", "v2"), "convnext_version", "v2"),
        Row(("--initial-filters",), ("--initial-filters", "80"), "initial_filters", 80),
        Row(("--filter-multiplier",), ("--filter-multiplier", "1.5"), "filter_multiplier", 1.5),
        Row(("--depth",), ("--depth", "11"), "depth", 11),
        Row(("--blocks-per-level",), ("--blocks-per-level", "10"), "blocks_per_level", 10),
        # -1 = one group per output channel (3), which must divide initial_filters, so the
        # companion sets 48. The value is stored as typed; the resolution is the builder's.
        Row(("--final-projection-groups",),
            ("--initial-filters", "48", "--final-projection-groups", "-1"),
            "final_projection_groups", -1),
        Row(("--laplacian-pyramid",), ("--laplacian-pyramid",), "use_laplacian_pyramid", True),
        Row(("--high-freq-blocks",), ("--high-freq-blocks", "22"), "high_freq_blocks", 22),
        Row(("--zero-pad-channels",), ("--zero-pad-channels",), "zero_pad_channels", True),
        Row(("--mean-pooling",), ("--mean-pooling",), "downsample_pool_type", "average"),
        Row(("--block-normalization",), ("--block-normalization", "layernorm"),
            "block_normalization", "layernorm"),
        Row(("--block-activation",), ("--block-activation", "relu"), "block_activation", "relu"),
        Row(("--block-activation-alpha",), ("--block-activation-alpha", "0.2"),
            "block_activation_alpha", 0.2),
        Row(("--expose-bottleneck",), ("--expose-bottleneck",), "expose_bottleneck", True),
        Row(("--extra-zero-output-channels",), ("--extra-zero-output-channels",),
            "extra_zero_output_channels", True),
        Row(("--depthwise-initializer",), ("--depthwise-initializer", "orthonormal"),
            "depthwise_initializer", "orthonormal"),
        Row(("--depthwise-l2",), ("--depthwise-l2", "0.001"), "depthwise_l2", 0.001),
        Row(("--dropout",), ("--dropout", "0.12"), "dropout_rate", 0.12),
        Row(("--bottleneck-attention-blocks",), ("--bottleneck-attention-blocks", "18"),
            "bottleneck_attention_blocks", 18),
        Row(("--bottleneck-attention-heads",), ("--bottleneck-attention-heads", "16"),
            "bottleneck_attention_heads", 16),
        # -- Gabor stem -------------------------------------------------------------
        Row(("--no-gabor-stem",), ("--no-gabor-stem",), "use_gabor_stem", False),
        Row(("--freeze-gabor-stem",), ("--freeze-gabor-stem",), "trainable_gabor_stem", False),
        Row(("--gabor-filters",), ("--gabor-filters", "20"), "gabor_filters", 20),
        Row(("--gabor-kernel-size",), ("--gabor-kernel-size", "15"), "gabor_kernel_size", 15),
        Row(("--gabor-activation",), ("--gabor-activation", "linear"),
            "gabor_activation", "linear"),
        Row(("--gabor-filters-per-channel",), ("--gabor-filters-per-channel", "14"),
            "gabor_filters_per_channel", 14),
        # Without the 1x1 projection the stem width must equal initial_filters (a
        # config-time refusal), so both counts are typed as the companion, equal.
        Row(("--no-gabor-projection",),
            ("--gabor-filters", "40", "--initial-filters", "40", "--no-gabor-projection"),
            "gabor_stem_projection", False),
        # -- self-iterate and WW-PGD ------------------------------------------------
        Row(("--self-iterate",), ("--self-iterate",), "self_iterate", True),
        Row(("--self-iterate-pool-size",), ("--self-iterate-pool-size", "300"),
            "self_iterate_pool_size", 300),
        Row(("--self-iterate-regen-freq",), ("--self-iterate-regen-freq", "17"),
            "self_iterate_regen_freq", 17),
        Row(("--self-iterate-mix-ratio",), ("--self-iterate-mix-ratio", "0.3"),
            "self_iterate_mix_ratio", 0.3),
        Row(("--ww-pgd",), ("--ww-pgd",), "ww_pgd", True),
        Row(("--ww-pgd-log-alpha",), ("--ww-pgd-log-alpha",), "ww_pgd_log_alpha", True),
        Row(("--ww-pgd-warmup-epochs",), ("--ww-pgd-warmup-epochs", "19"),
            "ww_pgd_warmup_epochs", 19),
        Row(("--ww-pgd-ramp-epochs",), ("--ww-pgd-ramp-epochs", "21"),
            "ww_pgd_ramp_epochs", 21),
        Row(("--ww-pgd-apply-every-epochs",), ("--ww-pgd-apply-every-epochs", "23"),
            "ww_pgd_apply_every_epochs", 23),
        Row(("--ww-pgd-q",), ("--ww-pgd-q", "2.5"), "ww_pgd_q", 2.5),
        Row(("--ww-pgd-blend-eta",), ("--ww-pgd-blend-eta", "0.6"), "ww_pgd_blend_eta", 0.6),
        Row(("--ww-pgd-cayley-eta",), ("--ww-pgd-cayley-eta", "0.15"),
            "ww_pgd_cayley_eta", 0.15),
        Row(("--ww-pgd-min-tail",), ("--ww-pgd-min-tail", "29"), "ww_pgd_min_tail", 29),
        Row(("--init-from",), ("--init-from", "/probe/ckpt.keras"), "init_from",
            "/probe/ckpt.keras"),
        # -- analysis, visualisation, evaluation, output ---------------------------
        Row(("--analyzer",), ("--analyzer",), "enable_analyzer", True),
        Row(("--analyzer-freq",), ("--analyzer-freq", "9"), "analyzer_freq", 9),
        Row(("--analyzer-start-epoch",), ("--analyzer-start-epoch", "5"),
            "analyzer_start_epoch", 5),
        Row(("--viz-freq",), ("--viz-freq", "8"), "viz_freq", 8),
        Row(("--viz-samples",), ("--viz-samples", "12"), "viz_samples", 12),
        # BooleanOptionalAction, default True: the probe is the non-default spelling.
        Row(("--test-eval", "--no-test-eval"), ("--no-test-eval",), "test_eval", False),
        Row(("--test-num-samples",), ("--test-num-samples", "37"), "test_num_samples", 37),
        # BooleanOptionalAction, default True: the probe is the non-default spelling.
        Row(("--model-analysis", "--no-model-analysis"), ("--no-model-analysis",),
            "model_analysis", False),
        Row(("--output-dir",), ("--output-dir", "/probe/bfunet-out"), "output_dir",
            "/probe/bfunet-out"),
        Row(("--experiment-name",), ("--experiment-name", "probe-experiment-name"),
            "experiment_name", "probe-experiment-name"),
        # -- not config fields ------------------------------------------------------
        # Consumed once by setup_gpu in main(); the hop is test_gpu_flag_reaches_setup_gpu.
        Row(("--gpu",), ("--gpu", "1"), namespace_dest="gpu", expected=1),
        # Rebuilds the dashboard from a run directory and returns; the hop is
        # test_dashboard_flag_reaches_build_dashboard_from_dir.
        Row(("--dashboard",), ("--dashboard", "/probe/dashboard-run"),
            namespace_dest="dashboard", expected="/probe/dashboard-run"),
        # Selects the preset inside config_from_args (the smoke mode below drives it).
        Row(("--smoke",), ("--smoke",), namespace_dest="smoke", expected=True),
        # Refused at config time (enable_deep_supervision is not wired), so it can only
        # be observed on the namespace; test_deep_supervision_is_refused_at_config pins
        # the refusal.
        Row(("--deep-supervision",), ("--deep-supervision",),
            namespace_dest="deep_supervision", expected=True),
    )


ROWS: Tuple[Row, ...] = _rows()

CONTRACT = Contract(
    name="train.bfunet.train_convunext_denoiser",
    build_parser=lambda monkeypatch: trainer._build_parser(),
    # The real parser (sentinel-explicit detection) and the real config builder; sys.argv
    # is set by the driver. No main(): nothing is trained, no GPU is set up.
    build_config=lambda monkeypatch: trainer.config_from_args(
        *trainer.parse_arguments_with_explicit(None)
    ),
    modes=(
        Mode(id="plain", required_argv=(), rows=ROWS),
        # Every row again, behind --smoke: a typed value must beat the preset, so a row
        # whose dest is in SMOKE_PRESET fails here if the preset overwrites it. The
        # baseline is smoke alone, so trap 1 measures against the PRESET value.
        Mode(
            id="smoke",
            required_argv=("--smoke",),
            rows=tuple(row for row in ROWS if row.flags != ("--smoke",)),
        ),
    ),
)
_CASES, _IDS = cases([CONTRACT])


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    """A flag that parses and is never forwarded, or forwarded to the wrong field, fails."""
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    """Trap 1: without this every row above could pass vacuously."""
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


def test_every_declared_flag_has_a_contract_row(monkeypatch) -> None:
    """A new flag added without a row fails HERE."""
    declared = declared_option_strings(CONTRACT.build_parser(monkeypatch))
    covered = CONTRACT.covered_flags
    assert declared == covered, (
        f"flags with no contract row {sorted(declared - covered)}; rows for flags "
        f"the parser no longer declares {sorted(covered - declared)}"
    )


# Fields a row legitimately moves BESIDES its own: values derived from it in the config
# (``epochs`` seeds ``curriculum_epochs`` and the 10 percent ``warmup_epochs``), the field an
# option documents as implied, and the companion flags a row must type to be admissible.
ALSO_MOVES: Dict[str, frozenset] = {
    "--epochs": frozenset({"curriculum_epochs", "warmup_epochs"}),
    "--ww-pgd-log-alpha": frozenset({"ww_pgd"}),
    "--final-projection-groups": frozenset({"initial_filters"}),
    "--no-gabor-projection": frozenset({"gabor_filters", "initial_filters"}),
}


def _config_for(monkeypatch, argv: Tuple[str, ...]):
    monkeypatch.setattr(sys, "argv", [CONTRACT.name, *argv])
    return CONTRACT.build_config(monkeypatch)


_FIELD_CASES = [
    (mode, row) for mode in CONTRACT.modes for row in mode.rows if row.field is not None
]


@pytest.mark.parametrize(
    "mode,row", _FIELD_CASES,
    ids=[f"[{mode.id}]{row.id}" for mode, row in _FIELD_CASES],
)
def test_a_row_moves_only_its_own_field(monkeypatch, mode, row) -> None:
    """A forward hijacked by ANOTHER flag's value must be seen even when the values agree.

    ``test_flag_reaches_its_config_field`` compares one field with one literal, so a
    forward rewired to another flag whose DEFAULT happens to equal the probe (``--channels
    1`` against ``symmetry_probes``' default 1, ``--viz-freq 8`` against ``viz_samples``'
    default 8) still reads green. Every field is compared with the baseline instead: the
    row must move its own field and nothing else beyond the documented ``ALSO_MOVES``.
    """
    base = _config_for(monkeypatch, mode.defaults_argv)
    moved = _config_for(monkeypatch, (*mode.required_argv, *row.argv))
    changed = {
        f.name for f in fields(trainer.TrainingConfig)
        # The default name is timestamped (and carries the variant), so it is compared only
        # for the row that types it.
        if (f.name != "experiment_name" or row.field == "experiment_name")
        and getattr(base, f.name) != getattr(moved, f.name)
    }
    allowed = {row.field} | ALSO_MOVES.get(row.flags[0], frozenset())
    assert row.field in changed, f"{row.id} did not move {row.field}"
    assert changed <= allowed, (
        f"{row.id} moved {sorted(changed - allowed)} besides {row.field}: a forward is "
        "reading another flag's value"
    )


def test_the_table_is_90_option_strings(monkeypatch) -> None:
    """Pins the size of the surface so a silent drop of half the rows cannot stay green."""
    assert len(declared_option_strings(CONTRACT.build_parser(monkeypatch))) == 90
    assert len(ROWS) == 88


def test_no_flag_has_two_rows(monkeypatch) -> None:
    """A flag listed twice would let one row's argv stand in for the other's."""
    flags = [flag for row in ROWS for flag in row.flags]
    assert len(flags) == len(set(flags))


@pytest.mark.parametrize("mode", CONTRACT.modes, ids=[m.id for m in CONTRACT.modes])
def test_probe_values_are_mutually_distinct(mode) -> None:
    """Trap 3: two rows sharing a value could hide a cross-wired forward.

    Booleans are exempt (there are only two, and a cross-wire between two flags that both
    probe ``True`` is caught because each row is driven alone against default argv and the
    OTHER field stays at its default), and so are the ``namespace_dest`` rows (they never
    reach a config field, so no forward can be crossed with them).
    """
    values = [
        row.expected for row in mode.rows
        if row.field is not None and not isinstance(row.expected, bool)
    ]
    seen: Dict[str, int] = {}
    for value in values:
        seen[repr(value)] = seen.get(repr(value), 0) + 1
    assert {k: n for k, n in seen.items() if n > 1} == {}


def test_the_smoke_mode_is_the_plain_mode_minus_the_smoke_flag_itself() -> None:
    plain, smoke = CONTRACT.modes
    assert [r.flags for r in smoke.rows] == [r.flags for r in plain.rows if r.flags != ("--smoke",)]
    assert smoke.required_argv == ("--smoke",)


def test_every_smoke_preset_dest_has_a_row_in_the_smoke_mode(monkeypatch) -> None:
    """Every preset dest names a real flag and that flag is driven behind ``--smoke``.

    Maps every ``SMOKE_PRESET`` dest to its option string through the real parser. A dest
    that names no flag is a dead preset entry (a typo the preset would silently never
    apply); one whose flag has no smoke-mode row is a preset value nothing proves a typed
    value can beat.
    """
    parser = CONTRACT.build_parser(monkeypatch)
    flag_of: Dict[str, str] = {
        a.dest: a.option_strings[0] for a in parser._actions if a.option_strings
    }
    unknown = sorted(dest for dest in trainer.SMOKE_PRESET if dest not in flag_of)
    assert unknown == [], f"SMOKE_PRESET dests that are not parser dests: {unknown}"
    smoke_rows = {flag for row in CONTRACT.modes[1].rows for flag in row.flags}
    missing = sorted(
        dest for dest in trainer.SMOKE_PRESET if flag_of[dest] not in smoke_rows
    )
    assert missing == [], f"SMOKE_PRESET dests with no smoke-mode row: {missing}"


def test_the_config_fields_no_flag_can_set_are_exactly_the_data_locations() -> None:
    """A config field no flag reaches is a knob only code can turn; the list is pinned.

    The five left are the corpus locations and layout, which are constants of this
    machine's data, not per-run options. A NEW field without a flag lands here and fails
    until it is either given a flag and a row or consciously added to this literal.
    """
    row_fields = {row.field for row in ROWS if row.field}
    # deep supervision is a namespace row whose field exists but is refused, so its field
    # is not in row_fields by design; it is checked next to the refusal test.
    config_fields = {f.name for f in fields(trainer.TrainingConfig)}
    assert config_fields - row_fields == {
        "train_image_dirs",
        "val_image_dirs",
        "dataset_weights",
        "data_range",
        "image_extensions",
        "enable_deep_supervision",
    }
    assert row_fields - config_fields == set()


def test_unset_flags_leave_every_config_default_untouched() -> None:
    """Parsing nothing must reproduce ``TrainingConfig()`` field for field.

    The parser and the config each carry a default; this is the drift guard between the
    two, so retuning one alone turns it RED. ``experiment_name`` is a timestamped
    derivative and is compared by prefix.
    """
    built = trainer.config_from_args(*trainer.parse_arguments_with_explicit([]))
    default = trainer.TrainingConfig()
    for f in fields(trainer.TrainingConfig):
        if f.name == "experiment_name":
            continue
        assert getattr(built, f.name) == getattr(default, f.name), f.name
    assert built.experiment_name.startswith("convunext_denoiser_base_")


# ---------------------------------------------------------------------
# rows that are not a plain flag -> field hop
# ---------------------------------------------------------------------


def test_deep_supervision_is_refused_at_config() -> None:
    """``--deep-supervision`` parses and is refused with its own message; it never trains."""
    args, explicit = trainer.parse_arguments_with_explicit(["--deep-supervision"])
    assert args.deep_supervision is True
    with pytest.raises(ValueError, match="not wired in this trainer"):
        trainer.config_from_args(args, explicit)


def test_ww_pgd_log_alpha_alone_turns_ww_pgd_on() -> None:
    """``--ww-pgd-log-alpha`` is documented as implying ``--ww-pgd``; a second field moves."""
    config = trainer.config_from_args(*trainer.parse_arguments_with_explicit(["--ww-pgd-log-alpha"]))
    assert (config.ww_pgd_log_alpha, config.ww_pgd) == (True, True)
    plain = trainer.config_from_args(*trainer.parse_arguments_with_explicit([]))
    assert (plain.ww_pgd_log_alpha, plain.ww_pgd) == (False, False)


def test_noise_flags_are_mutually_exclusive_by_precedence_not_by_last_wins() -> None:
    """With both noise flags typed the composite noise wins (the documented precedence)."""
    config = trainer.config_from_args(
        *trainer.parse_arguments_with_explicit(["--multiplicative-noise", "--composite-noise"])
    )
    assert config.noise_type == "composite"
    config = trainer.config_from_args(
        *trainer.parse_arguments_with_explicit(["--composite-noise", "--multiplicative-noise"])
    )
    assert config.noise_type == "composite"


# ---------------------------------------------------------------------
# --gpu and --dashboard: the hop after the parse, through the real main()
# ---------------------------------------------------------------------


def _stub_main_targets(monkeypatch) -> Dict[str, List[Tuple[tuple, dict]]]:
    """Replace every expensive call ``main()`` can reach with a recorder.

    ``setup_gpu``, ``train`` and ``build_dashboard_from_dir`` are bound by name into the
    trainer module, so they are patched there. Each records ``(args, kwargs)``.
    """
    calls: Dict[str, List[Tuple[tuple, dict]]] = {
        "setup_gpu": [], "train": [], "build_dashboard_from_dir": [],
    }

    def _recorder(name):
        def _record(*args, **kwargs):
            calls[name].append((args, kwargs))
        return _record

    for name in calls:
        monkeypatch.setattr(trainer, name, _recorder(name))
    return calls


def test_gpu_flag_reaches_setup_gpu(monkeypatch) -> None:
    """``--gpu 1`` must arrive as ``setup_gpu(gpu_id=1)`` inside ``main()``."""
    calls = _stub_main_targets(monkeypatch)
    trainer.main(["--gpu", "1", "--output-dir", "/probe/bfunet-out"])
    assert calls["setup_gpu"] == [((), {"gpu_id": 1})], (
        f"--gpu 1 did not reach setup_gpu(gpu_id=1): {calls['setup_gpu']!r}"
    )
    assert len(calls["train"]) == 1


def test_no_gpu_flag_reaches_setup_gpu_as_none(monkeypatch) -> None:
    """Untyped ``--gpu`` is ``None`` (memory growth on all devices), not a number."""
    calls = _stub_main_targets(monkeypatch)
    trainer.main(["--output-dir", "/probe/bfunet-out"])
    assert calls["setup_gpu"] == [((), {"gpu_id": None})]


def test_dashboard_flag_reaches_build_dashboard_from_dir(monkeypatch) -> None:
    """``--dashboard DIR`` rebuilds from DIR and returns before any GPU setup or training."""
    calls = _stub_main_targets(monkeypatch)
    trainer.main(["--dashboard", "/probe/dashboard-run"])
    assert calls["build_dashboard_from_dir"] == [(("/probe/dashboard-run",), {})]
    assert calls["setup_gpu"] == []
    assert calls["train"] == []


def test_main_hands_the_config_built_from_the_flags_to_train(monkeypatch) -> None:
    """The last hop: ``train`` receives the object ``config_from_args`` built."""
    calls = _stub_main_targets(monkeypatch)
    trainer.main(["--epochs", "7", "--output-dir", "/probe/bfunet-out", "--experiment-name", "probe"])
    ((args, kwargs),) = calls["train"]
    (config,) = args
    assert kwargs == {}
    assert (config.epochs, config.output_dir, config.experiment_name) == (7, "/probe/bfunet-out", "probe")

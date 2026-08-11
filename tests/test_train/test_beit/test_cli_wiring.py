"""SC-11 guard: every CLI flag of every BEiT trainer reaches its ``TrainingConfig``.

**The trap.** ``config_from_args()`` builds ``TrainingConfig(...)`` field-by-field from
``args.*``. A flag that ``parse_arguments()`` DEFINES but ``config_from_args()`` never READS
becomes a SILENT NO-OP: the user passes ``--num-embeddings 1024``, argparse accepts it
without a murmur, the run trains at the dataclass default, and the resulting curve is then
attributed to the model. This has already bitten this repository (bfunet's
``high_freq_blocks`` and ``filter_multiplier`` were both silent no-ops for real runs).

**Why this test is STRUCTURAL, not a checklist.** A hand-written list of asserts guards only
the flags that existed the day it was written; flag #25, added next month, sails straight
through -- which is the exact failure class we are guarding. So the test introspects instead:

* the flag surface comes from ``vars(parse_arguments([]))``, i.e. every argparse ``dest``
  of every non-``help`` action (``help`` exits the process, so it has no ``dest`` in the
  namespace; nothing here uses ``default=SUPPRESS``);
* the config surface comes from ``dataclasses.fields(TrainingConfig)``;
* a flag whose ``dest`` maps to no config field FAILS unless it is in an explicit,
  commented exclusion allow-list, and a config field fed by no flag FAILS the same way.

That makes the test FAIL-CLOSED: adding a flag without wiring it is RED by default.

It also covers the entry-point liveness question the unit suite cannot answer: each trainer
is invoked as a SUBPROCESS with ``--help`` and must exit 0. A ``%`` in an argparse help
string is a latent ``--help`` crash that a 100%-green in-process suite never sees, because
``ArgumentDefaultsHelpFormatter`` only interpolates when help is actually rendered.

Covers success criterion SC-11 and invariant 10 (``output_dir`` is repo-root ``results/``).
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import pytest

from train.beit import train_tokenizer as tok_trainer

# ---------------------------------------------------------------------------
# The dest -> config-field map
# ---------------------------------------------------------------------------

# argparse dests DELIBERATELY named differently from the config field they feed.
# Anything not listed here must map to a field of the SAME name.
DEST_RENAMES: Dict[str, str] = {
    "optimizer": "optimizer_type",
    "lr_schedule": "lr_schedule_type",
}

# ---------------------------------------------------------------------------
# Exclusion allow-lists -- the ONLY escape hatches. Both are EMPTY, and that is the point:
# an unlisted flag that reaches no config field FAILS. Adding an entry here is a deliberate,
# reviewable act, not something a new flag gets for free.
# ---------------------------------------------------------------------------

EXCLUDED_DESTS: Dict[str, str] = {}  # dest -> why it feeds no config field
EXCLUDED_FIELDS: Dict[str, str] = {}  # field -> why no flag feeds it


# ---------------------------------------------------------------------------
# Non-default CLI values. EVERY dest of EVERY trainer must appear in its spec, or the test
# fails -- that is what makes a newly-added flag RED instead of silently uncovered.
#
# Each value is chosen to DIFFER from the dataclass default; `test_probe_values_are_non_default`
# proves that, so a value that accidentally equals the default cannot make the wiring asserts
# pass vacuously.
# ---------------------------------------------------------------------------

# Shared by every BEiT trainer (same names, same defaults).
COMMON_SPEC: Dict[str, Tuple[str, Any]] = {
    "dataset": ("--dataset", "cifar10"),           # default: imagenette
    "batch_size": ("--batch-size", 8),             # default: 32
    "augment_data": ("--no-augmentation", False),  # default: True (store_false)
    "learning_rate": ("--learning-rate", 1e-5),    # default: 5e-4
    "optimizer": ("--optimizer", "sgd"),           # default: adamw
    "lr_schedule": ("--lr-schedule", "constant"),  # default: cosine_decay
    "warmup_epochs": ("--warmup-epochs", 1),       # default: 2
    "weight_decay": ("--weight-decay", 0.01),      # default: 0.05
    "gradient_clipping": ("--gradient-clipping", 0.5),            # default: 1.0
    "early_stopping_patience": ("--early-stopping-patience", 4),  # default: 15
    "max_steps": ("--max-steps", 7),               # default: None
    "experiment_name": ("--experiment-name", "beit_cli_wiring_probe"),  # default: None
    "seed": ("--seed", 123),                       # default: 42
    "gpu": ("--gpu", 1),                           # default: None
    # `output_dir` is filled in per-test from tmp_path (default: "results").
}

TOKENIZER_ONLY_SPEC: Dict[str, Tuple[str, Any]] = {
    # 64 is divisible by the probe downsample_factor 32, so __post_init__ stays happy.
    "image_size": ("--image-size", 64),            # default: 224
    "downsample_factor": ("--downsample-factor", 32),  # default: 16
    "num_embeddings": ("--num-embeddings", 1024),  # default: 8192
    "embedding_dim": ("--embedding-dim", 16),      # default: 32
    "hidden_channels": ("--hidden-channels", 64),  # default: 128
    "num_res_blocks": ("--num-res-blocks", 1),     # default: 2
    "commitment_cost": ("--commitment-cost", 0.5),  # default: 0.25
    "use_ema": ("--use-ema", True),                # default: False (store_true)
    "epochs": ("--epochs", 3),                     # default: 50
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TrainerCase:
    """One trainer under test: its CLI, its config, and its non-default probe values."""

    name: str
    module: str  # importable module path, for the --help subprocess probe
    parse_arguments: Callable[[Optional[list]], Any]
    config_from_args: Callable[[Any], Any]
    config_cls: type
    spec: Dict[str, Tuple[str, Any]]


def _write_dummy_checkpoint(tmp_path, name: str = "dummy.keras") -> str:
    """A real, loadable `.keras` file. Checkpoint-valued flags are validated for EXISTENCE
    at config time, so a bare string path would raise before the wiring assert.
    """
    model = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Dense(2)])
    path = tmp_path / name
    model.save(path)
    return str(path)


def _tokenizer_case(tmp_path) -> TrainerCase:
    return TrainerCase(
        name="train_tokenizer",
        module="train.beit.train_tokenizer",
        parse_arguments=tok_trainer.parse_arguments,
        config_from_args=tok_trainer.config_from_args,
        config_cls=tok_trainer.TrainingConfig,
        spec={
            **COMMON_SPEC,
            **TOKENIZER_ONLY_SPEC,
            "output_dir": ("--output-dir", str(tmp_path / "results")),
        },
    )


CASE_BUILDERS: Dict[str, Callable[[Any], TrainerCase]] = {
    "tokenizer": _tokenizer_case,
}


@pytest.fixture(params=sorted(CASE_BUILDERS))
def case(request, tmp_path) -> TrainerCase:
    """Every BEiT trainer, driven through the identical structural checks."""
    return CASE_BUILDERS[request.param](tmp_path)


def _cli_dests(case: TrainerCase) -> set:
    """Every argparse `dest` the trainer defines, read off a defaults-only parse."""
    return set(vars(case.parse_arguments([])).keys())


def _config_fields(case: TrainerCase) -> Dict[str, dataclasses.Field]:
    return {f.name: f for f in dataclasses.fields(case.config_cls)}


def _field_for(dest: str) -> str:
    return DEST_RENAMES.get(dest, dest)


def _build_argv(case: TrainerCase) -> list:
    argv: list = []
    for _dest, (flag, value) in case.spec.items():
        if isinstance(value, bool):
            # store_true / store_false: the bare flag IS the non-default value.
            argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


# ---------------------------------------------------------------------------
# 1. Surface coverage -- the fail-closed core
# ---------------------------------------------------------------------------

def test_every_cli_flag_maps_to_a_config_field(case: TrainerCase) -> None:
    """A flag that maps to NO config field is a silent no-op by construction."""
    fields = _config_fields(case)
    unmapped = sorted(
        dest for dest in _cli_dests(case)
        if dest not in EXCLUDED_DESTS and _field_for(dest) not in fields
    )
    assert not unmapped, (
        f"{case.name}: CLI flag(s) {unmapped} map to no TrainingConfig field. Each is a "
        f"SILENT NO-OP: argparse accepts the value and the run uses the default. Wire it "
        f"into config_from_args(), add a DEST_RENAMES entry, or justify it in EXCLUDED_DESTS."
    )


def test_every_config_field_is_fed_by_a_cli_flag(case: TrainerCase) -> None:
    """The mirror image: a config field no flag can reach is dead config."""
    reachable = {_field_for(dest) for dest in _cli_dests(case)}
    unreachable = sorted(
        name for name in _config_fields(case)
        if name not in reachable and name not in EXCLUDED_FIELDS
    )
    assert not unreachable, (
        f"{case.name}: TrainingConfig field(s) {unreachable} are settable from no CLI flag. "
        f"Add the flag, or justify the omission in EXCLUDED_FIELDS."
    )


def test_every_cli_flag_has_a_non_default_probe_value(case: TrainerCase) -> None:
    """Fail-closed on the TEST side too: a new flag with no probe value is RED, not skipped."""
    uncovered = sorted(
        dest for dest in _cli_dests(case)
        if dest not in case.spec and dest not in EXCLUDED_DESTS
    )
    assert not uncovered, (
        f"{case.name}: CLI flag(s) {uncovered} have no non-default probe value in this "
        f"test's spec, so their wiring is UNVERIFIED. Add them to the trainer's spec."
    )


# ---------------------------------------------------------------------------
# 2. Self-check -- the probe values must actually be non-default
# ---------------------------------------------------------------------------

def test_probe_values_are_non_default(case: TrainerCase) -> None:
    """If a "non-default" probe value equals the dataclass default, the wiring assert below
    passes VACUOUSLY -- it would pass just as happily with the wiring line deleted. This
    test is the guard on the guard.
    """
    fields = _config_fields(case)
    vacuous = []
    for dest, (_flag, value) in case.spec.items():
        field = fields[_field_for(dest)]
        if field.default is not dataclasses.MISSING and field.default == value:
            vacuous.append(f"{dest}={value!r} == default")
    assert not vacuous, (
        f"{case.name}: probe value(s) equal their dataclass default: {vacuous}. The wiring "
        f"asserts would pass even with the wiring line DELETED. Pick different values."
    )


# ---------------------------------------------------------------------------
# 3. The wiring assert itself
# ---------------------------------------------------------------------------

def test_every_cli_value_reaches_the_config(case: TrainerCase) -> None:
    """THE guard: parse a fully non-default argv, and demand every field carry the CLI value."""
    args = case.parse_arguments(_build_argv(case))
    config = case.config_from_args(args)

    dropped = []
    fields = _config_fields(case)
    for dest, (flag, expected) in case.spec.items():
        field_name = _field_for(dest)
        actual = getattr(config, field_name)
        if actual != expected:
            default = fields[field_name].default
            at_default = " (still at the DATACLASS DEFAULT -- the flag is a SILENT NO-OP)" \
                if actual == default else ""
            dropped.append(
                f"{flag} -> TrainingConfig.{field_name}: expected {expected!r}, "
                f"got {actual!r}{at_default}"
            )

    assert not dropped, (
        f"{case.name}: {len(dropped)} CLI flag(s) did not reach TrainingConfig:\n  "
        + "\n  ".join(dropped)
    )


def test_output_dir_default_is_repo_root_results(case: TrainerCase) -> None:
    """H-13 / invariant 10: training output goes to repo-root ``results/``, never
    ``src/results/`` and never an absolute path baked into the dataclass.
    """
    field = _config_fields(case)["output_dir"]
    assert field.default == "results", (
        f"{case.name}: TrainingConfig.output_dir default is {field.default!r}, must be "
        f"'results' (repo-root)."
    )
    config = case.config_from_args(case.parse_arguments([]))
    assert config.output_dir == "results"


def test_defaults_only_parse_still_builds_a_valid_config(case: TrainerCase) -> None:
    """Sanity floor: the no-flags path must still produce a usable config."""
    config = case.config_from_args(case.parse_arguments([]))
    assert config.experiment_name  # __post_init__ generates one
    assert config.output_dir == "results"


# ---------------------------------------------------------------------------
# 4. Entry-point liveness -- a green unit suite does not prove the script RUNS
# ---------------------------------------------------------------------------

def test_help_runs_as_a_subprocess(case: TrainerCase) -> None:
    """``python -m <trainer> --help`` must exit 0.

    ``ArgumentDefaultsHelpFormatter`` interpolates ``%(default)s`` only when help is
    rendered, so a stray bare ``%`` in any help string is a crash that is invisible to
    every in-process test in this file.
    """
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = ""  # --help must not need (or grab) a GPU
    result = subprocess.run(
        [sys.executable, "-m", case.module, "--help"],
        cwd=str(repo_root / "src"),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{case.name}: `python -m {case.module} --help` exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "--output-dir" in result.stdout


# ---------------------------------------------------------------------------
# 5. Per-trainer __post_init__ validation -- loud raises BEFORE any data/model work
# ---------------------------------------------------------------------------

class TestTokenizerConfigValidation:
    """``train_tokenizer.TrainingConfig.__post_init__`` must refuse bad geometry."""

    def test_non_divisible_geometry_raises_and_names_the_grid(self) -> None:
        with pytest.raises(ValueError, match=r"non-integral code grid"):
            # 100 / 8 = 12.5 -- the factor IS a power of 2, so this reaches the
            # divisibility check rather than short-circuiting on the one above it.
            tok_trainer.TrainingConfig(image_size=100, downsample_factor=8)

    def test_non_power_of_two_downsample_factor_raises(self) -> None:
        with pytest.raises(ValueError, match=r"power of 2"):
            tok_trainer.TrainingConfig(image_size=224, downsample_factor=14)

    @pytest.mark.parametrize("kwargs,pattern", [
        ({"image_size": 0}, "image_size must be positive"),
        ({"downsample_factor": 0}, "downsample_factor must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"epochs": 0}, "epochs must be positive"),
        ({"num_embeddings": 0}, "num_embeddings must be positive"),
        ({"embedding_dim": 0}, "embedding_dim must be positive"),
        ({"hidden_channels": 0}, "hidden_channels must be positive"),
        ({"num_res_blocks": -1}, "num_res_blocks must be non-negative"),
        ({"commitment_cost": -0.1}, "commitment_cost must be non-negative"),
        ({"max_steps": 0}, "max_steps must be positive"),
        ({"dataset": "mnist"}, "Unsupported dataset"),
    ])
    def test_invalid_field_raises(self, kwargs: Dict[str, Any], pattern: str) -> None:
        with pytest.raises(ValueError, match=pattern):
            tok_trainer.TrainingConfig(**kwargs)

    def test_code_grid_is_derived_from_the_validated_geometry(self) -> None:
        assert tok_trainer.TrainingConfig(
            image_size=224, downsample_factor=16).code_grid == (14, 14)
        assert tok_trainer.TrainingConfig(
            image_size=112, downsample_factor=8).code_grid == (14, 14)
        assert tok_trainer.TrainingConfig(
            image_size=32, downsample_factor=8).code_grid == (4, 4)

    def test_dataset_default_geometry_is_dataset_dependent(self) -> None:
        """``--image-size``/``--downsample-factor`` default to None and are TRANSFORMED,
        not copied. That transform is the one place a wrong value hides behind a plausible
        number, so both datasets are pinned explicitly.
        """
        cifar = tok_trainer.config_from_args(
            tok_trainer.parse_arguments(["--dataset", "cifar10"]))
        assert (cifar.image_size, cifar.downsample_factor) == (32, 8)
        assert cifar.code_grid == (4, 4)

        nette = tok_trainer.config_from_args(
            tok_trainer.parse_arguments(["--dataset", "imagenette"]))
        assert (nette.image_size, nette.downsample_factor) == (224, 16)
        assert nette.code_grid == (14, 14)


# ---------------------------------------------------------------------------
# 6. The measured geometry contract (step-9 STOP-IF-3 falsification probe)
# ---------------------------------------------------------------------------

class TestTokenizerCodeGridIsReal:
    """The config's ``code_grid`` is arithmetic; this asserts the MODEL agrees with it.

    Measured, not derived: ``encode_to_indices`` is actually run. A control at a geometry
    that must NOT produce the same grid keeps the assertion from passing vacuously.
    """

    @pytest.mark.parametrize("image_size,factor,expected", [
        (224, 16, (14, 14)),   # D-004's single-resolution scheme (the defaults)
        (112, 8, (14, 14)),    # BEiT's own two-resolution scheme (H-11), the fallback
        (64, 16, (4, 4)),      # control: must NOT be 14x14
    ])
    def test_encode_to_indices_matches_the_config_grid(
            self, image_size: int, factor: int, expected: Tuple[int, int]) -> None:
        import numpy as np

        config = tok_trainer.TrainingConfig(
            dataset="cifar10",
            image_size=image_size,
            downsample_factor=factor,
            num_embeddings=64,
            embedding_dim=8,
            hidden_channels=16,
            num_res_blocks=1,
        )
        assert config.code_grid == expected

        model = tok_trainer.build_tokenizer(config)
        ids = model.encode_to_indices(
            np.zeros((2, image_size, image_size, 3), dtype="float32"))
        assert tuple(int(v) for v in ids.shape) == (2,) + expected

    def test_build_tokenizer_raises_when_the_grid_disagrees(self) -> None:
        """The RuntimeError guard in ``build_tokenizer`` must be reachable, not decorative.

        There is no legal config that trips it (``__post_init__`` already refuses
        non-divisible geometry), so the ARITHMETIC is made to lie while the model stays
        honest -- which is exactly the shape of a future edit that changes the encoder's
        stride compound without changing the derived grid (or vice versa).
        """
        class _LyingConfig(tok_trainer.TrainingConfig):
            @property
            def code_grid(self) -> Tuple[int, int]:
                return (8, 8)

        config = _LyingConfig(
            dataset="cifar10",
            image_size=32,
            downsample_factor=8,   # really yields (4, 4)
            num_embeddings=64,
            embedding_dim=8,
            hidden_channels=16,
            num_res_blocks=1,
        )
        assert config.code_grid == (8, 8)
        with pytest.raises(RuntimeError, match=r"code grid is \(4, 4\), expected \(8, 8\)"):
            tok_trainer.build_tokenizer(config)

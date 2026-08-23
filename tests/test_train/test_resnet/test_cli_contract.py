"""CLI contract guard for ``src/train/resnet/train_resnet.py``.

``src/train/resnet/`` had ZERO tests of any kind before this plan (F-12), which
is the causal explanation for its two CRITICAL defects shipping unnoticed. The
sibling ``tests/test_train/test_bert_wikipedia/test_cli_contract.py`` is the
shape copied here.

What each guard pins, and why it is not satisfied-by-construction:

``test_every_parser_dest_is_accounted_for``
    The *completeness* half. The dest set is read off the real
    ``argparse.Namespace`` that ``parse_arguments()`` returns -- not off an AST
    walk and not off ``parser._actions`` -- and is compared against an explicit
    dest-to-``TrainingConfig``-field map plus the one deliberately unmapped
    dest. ``gpu`` is unmapped ON PURPOSE: it is not a ``TrainingConfig`` field
    at all, it is forwarded as ``train_resnet_imagenet(config, gpu_id=args.gpu)``
    (``train_resnet.py:390``), and ``test_gpu_reaches_the_trainer_not_the_config``
    is what proves that claim rather than asserting it. Adding a new
    ``add_argument`` without wiring it fails HERE, before anyone has to notice a
    knob that silently does nothing.

``test_every_cli_value_reaches_the_config``
    The repo's documented silent-no-op bug class: a flag that parses and is then
    never forwarded. Every pinned value DIFFERS from the ``TrainingConfig``
    dataclass default (asserted mechanically by
    ``test_the_pinned_values_all_differ_from_the_dataclass_defaults``), so the
    test cannot pass by observing a fresh default config, and it additionally
    asserts that fields with no CLI flag at all still hold their defaults.

``test_help_exits_zero_without_reaching_anything_expensive``
    ``--help`` must exit having called NOTHING expensive. Sentinels are
    installed over ``setup_gpu``, ``create_resnet``, ``make_imagenet_filesystem_dataset``
    and ``train_resnet_imagenet``; each raises on contact and each is asserted to
    have been called ZERO times. Moving ``parse_arguments()`` below any of them
    fails by the named ``--help reached ...`` message, not by the exit code -- so
    the guard distinguishes "parsed first" from "parsed at all".

RED PROOFS -- one injection per assertion, with the ACTUAL observed text
(this repo's predicted RED line is wrong roughly half the time). Every
injection was reverted immediately afterwards.

1. ``test_every_cli_value_reaches_the_config``. Injection: delete
   ``label_smoothing=args.label_smoothing`` from the ``TrainingConfig(...)``
   construction at ``train_resnet.py:370-383``. Observed (1 failed, 7 passed)::

       AssertionError: --label-smoothing never reached
       TrainingConfig.label_smoothing: argv asked for 0.037, config holds 0.1.
       A flag that parses and is then not forwarded onto the config is a knob
       that silently does nothing.
       assert False

2. ``test_every_parser_dest_is_accounted_for``. Injection: add
   ``parser.add_argument('--fake-knob', type=int, default=1)``. Observed
   (1 failed, 7 passed)::

       AssertionError: train_resnet.py grew CLI flag(s) ['fake_knob'] that this
       contract does not account for. ...
       assert not ['fake_knob']

3. ``test_help_exits_zero_without_reaching_anything_expensive``. Injection: put
   ``setup_gpu(None)`` above ``args = parse_arguments()`` in ``main()``.
   Observed (6 failed, 2 passed) -- the --help arm failed by its OWN named
   message, not by the exit code::

       Failed: --help reached 'setup_gpu' before argparse could exit.
       `args = parse_arguments()` must be the FIRST statement of main(), above
       GPU setup, dataset construction and model construction.

No training run, no GPU allocation, no dataset read: ``train_resnet_imagenet``
is sentinelled off in every test here, and every path config routes through
pytest's ``tmp_path``. Nothing is written into repo-root ``results/``.
"""

import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

import train.resnet.train_resnet as train_resnet
from train.resnet.train_resnet import TrainingConfig


# Every ``add_argument`` dest, mapped onto the ``TrainingConfig`` field it must
# reach. The one deliberate omission is ``gpu`` -- see the module docstring and
# ``test_gpu_reaches_the_trainer_not_the_config``.
DEST_TO_FIELD: Dict[str, str] = {
    "train_data_dir": "train_data_dir",
    "val_data_dir": "val_data_dir",
    "image_size": "image_size",
    "variant": "model_variant",
    "num_classes": "num_classes",
    "pretrained": "pretrained",
    "enable_deep_supervision": "enable_deep_supervision",
    "deep_supervision_schedule": "deep_supervision_schedule_type",
    "epochs": "epochs",
    "batch_size": "batch_size",
    "learning_rate": "learning_rate",
    "optimizer": "optimizer_type",
    "lr_schedule": "lr_schedule_type",
    "warmup_epochs": "warmup_epochs",
    "weight_decay": "weight_decay",
    "augment_data": "augment_data",
    "label_smoothing": "label_smoothing",
    "output_dir": "output_dir",
    "experiment_name": "experiment_name",
    "monitor_every": "monitor_every_n_epochs",
    "early_stopping_patience": "early_stopping_patience",
}

# Dests that are NOT ``TrainingConfig`` fields, each with the reason it is not.
UNMAPPED_DESTS: Dict[str, str] = {
    "gpu": "forwarded as train_resnet_imagenet(config, gpu_id=args.gpu)",
}

# ``TrainingConfig`` fields with no CLI flag at all. Asserted to still hold
# their dataclass defaults, so the value assertions cannot pass by looking at a
# config object that is not the one ``main()`` built.
FIELDS_WITH_NO_FLAG: Tuple[str, ...] = (
    "gradient_clipping",
    "momentum",
    "validation_steps",
    "cache_dataset",
)

# Every value differs from the dataclass default -- mechanically asserted below.
EXPECTED: Dict[str, Any] = {
    "image_size": 96,
    "model_variant": "resnet18",
    "num_classes": 7,
    "pretrained": True,
    "enable_deep_supervision": True,
    "deep_supervision_schedule_type": "curriculum",
    "epochs": 3,
    "batch_size": 5,
    "learning_rate": 0.0123,
    "optimizer_type": "adamw",
    "lr_schedule_type": "exponential_decay",
    "warmup_epochs": 2,
    "weight_decay": 0.0009,
    "augment_data": False,
    "label_smoothing": 0.037,
    "experiment_name": "cli_contract_probe",
    "monitor_every_n_epochs": 11,
    "early_stopping_patience": 4,
}

GPU_INDEX = 3


class _Abort(BaseException):
    """Raised by an active sentinel on contact.

    Derives from ``BaseException``, NOT ``Exception``, on purpose:
    ``train_resnet.main()`` wraps its trainer call in ``except Exception``, so
    an ``Exception``-derived sentinel would be logged as a training failure and
    re-raised as itself only by accident of the bare ``raise``. Using
    ``BaseException`` keeps the abort unambiguous.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class _Sentinel:
    """Callable that records contact, and aborts unless it is passive.

    Contract: ``calls`` counts invocations and ``args``/``kwargs`` hold the last
    call's arguments. An active sentinel raises :class:`_Abort` carrying ``name``
    and never returns; a passive one records and returns ``None``, which is what
    lets a probe run PAST a call it only wants to neutralise.
    """

    def __init__(self, name: str, active: bool = True) -> None:
        self.name = name
        self.active = active
        self.calls = 0
        self.args: Tuple[Any, ...] = ()
        self.kwargs: Dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.args = args
        self.kwargs = kwargs
        if self.active:
            raise _Abort(self.name)
        return None


# Everything ``main()`` can reach that costs a GPU, a dataset read or a model.
EXPENSIVE = (
    "setup_gpu",
    "create_resnet",
    "make_imagenet_filesystem_dataset",
    "train_resnet_imagenet",
)


def _install_sentinels(
    monkeypatch: pytest.MonkeyPatch, passive: Tuple[str, ...] = ()
) -> Dict[str, _Sentinel]:
    """Install a recording sentinel over every expensive name ``main()`` reaches.

    Contract:
      - Returns ``{name: sentinel}`` covering every entry of :data:`EXPENSIVE`;
        each counts its calls, and each NOT named in ``passive`` raises
        :class:`_Abort` on contact.
      - Every name is patched on the ``train_resnet`` MODULE, because all four
        are imported into it by name at import time -- patching the defining
        module would not be seen by ``main()``.
      - ``monkeypatch`` restores every attribute at teardown.
    """
    sentinels = {
        name: _Sentinel(name, active=name not in passive) for name in EXPENSIVE
    }
    for name, sentinel in sentinels.items():
        monkeypatch.setattr(train_resnet, name, sentinel)
    return sentinels


def _argv(tmp_path: Path) -> List[str]:
    """A full argv exercising every flag, with real directories under ``tmp_path``.

    ``TrainingConfig.__post_init__`` raises unless both data directories exist,
    so they are created here. Nothing outside ``tmp_path`` is touched.
    """
    train_dir = tmp_path / "imagenet_train"
    val_dir = tmp_path / "imagenet_val"
    train_dir.mkdir()
    val_dir.mkdir()
    return [
        "train_resnet.py",
        "--train-data-dir", str(train_dir),
        "--val-data-dir", str(val_dir),
        "--image-size", str(EXPECTED["image_size"]),
        "--variant", EXPECTED["model_variant"],
        "--num-classes", str(EXPECTED["num_classes"]),
        "--pretrained",
        "--enable-deep-supervision",
        "--deep-supervision-schedule", EXPECTED["deep_supervision_schedule_type"],
        "--epochs", str(EXPECTED["epochs"]),
        "--batch-size", str(EXPECTED["batch_size"]),
        "--learning-rate", str(EXPECTED["learning_rate"]),
        "--optimizer", EXPECTED["optimizer_type"],
        "--lr-schedule", EXPECTED["lr_schedule_type"],
        "--warmup-epochs", str(EXPECTED["warmup_epochs"]),
        "--weight-decay", str(EXPECTED["weight_decay"]),
        "--no-augmentation",
        "--label-smoothing", str(EXPECTED["label_smoothing"]),
        "--output-dir", str(tmp_path / "run_output"),
        "--experiment-name", EXPECTED["experiment_name"],
        "--monitor-every", str(EXPECTED["monitor_every_n_epochs"]),
        "--early-stopping-patience", str(EXPECTED["early_stopping_patience"]),
        "--gpu", str(GPU_INDEX),
    ]


def _run_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Tuple[TrainingConfig, Dict[str, _Sentinel]]:
    """Drive the real ``main()`` on a full argv and return the config it built.

    Contract:
      - ``train_resnet_imagenet`` is PASSIVE, so ``main()`` runs to completion
        and the config is captured from the sentinel's recorded call arguments
        -- the same object ``main()`` would have trained with, not a
        reconstruction.
      - All three other expensive names stay ACTIVE, so reaching any of them is
        a failure rather than a silent cost.
      - ``sys.argv`` is patched because ``parse_arguments()`` takes no argv
        parameter; it calls ``parser.parse_args()`` with no arguments.
    """
    argv = _argv(tmp_path)
    sentinels = _install_sentinels(monkeypatch, passive=("train_resnet_imagenet",))
    monkeypatch.setattr(sys, "argv", argv)

    train_resnet.main()

    trainer = sentinels["train_resnet_imagenet"]
    assert trainer.calls == 1, (
        f"main() called train_resnet_imagenet {trainer.calls} times, expected 1 "
        f"-- the probe did not reach the point it claims to measure"
    )
    assert len(trainer.args) == 1, (
        f"train_resnet_imagenet received positional args {trainer.args!r}; this "
        f"guard assumes the config is the single positional argument"
    )
    return trainer.args[0], sentinels


def _default(field_name: str) -> Any:
    """The ``TrainingConfig`` dataclass default for ``field_name``."""
    for field in dataclass_fields(TrainingConfig):
        if field.name == field_name:
            return field.default
    raise AssertionError(f"TrainingConfig has no field named {field_name!r}")


# ---------------------------------------------------------------------
# completeness: every dest is either mapped or explicitly accounted for
# ---------------------------------------------------------------------


def test_every_parser_dest_is_accounted_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    args = train_resnet.parse_arguments()

    dests = set(vars(args))
    accounted = set(DEST_TO_FIELD) | set(UNMAPPED_DESTS)

    unaccounted = sorted(dests - accounted)
    assert not unaccounted, (
        f"train_resnet.py grew CLI flag(s) {unaccounted} that this contract does "
        f"not account for. Add each to DEST_TO_FIELD (and wire it into the "
        f"TrainingConfig(...) construction), or to UNMAPPED_DESTS with the "
        f"reason it is not a config field."
    )

    stale = sorted(accounted - dests)
    assert not stale, (
        f"this contract names dest(s) {stale} that parse_arguments() no longer "
        f"produces; the guard is looking at flags that do not exist"
    )

    missing_fields = sorted(
        field for field in DEST_TO_FIELD.values()
        if field not in {f.name for f in dataclass_fields(TrainingConfig)}
    )
    assert not missing_fields, (
        f"DEST_TO_FIELD names TrainingConfig field(s) {missing_fields} that the "
        f"dataclass does not have"
    )


def test_the_pinned_values_all_differ_from_the_dataclass_defaults() -> None:
    """Anti-vacuity: a pinned value equal to its default proves nothing."""
    same = {
        field: value
        for field, value in EXPECTED.items()
        if _default(field) == value
    }
    assert not same, (
        f"pinned expectation(s) {sorted(same)} equal the TrainingConfig default, "
        f"so test_every_cli_value_reaches_the_config would pass even if the "
        f"assignment were deleted. Pick a different value."
    )


# ---------------------------------------------------------------------
# argv -> config: the silent-no-op bug class
# ---------------------------------------------------------------------


def test_every_cli_value_reaches_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _ = _run_main(monkeypatch, tmp_path)

    flag_for = {field: dest for dest, field in DEST_TO_FIELD.items()}
    for field, want in EXPECTED.items():
        got = getattr(config, field)
        flag = "--" + flag_for[field].replace("_", "-")
        if isinstance(want, float):
            matched = got == pytest.approx(want)
        else:
            matched = got == want and type(got) is type(want)
        assert matched, (
            f"{flag} never reached TrainingConfig.{field}: argv asked for "
            f"{want!r}, config holds {got!r}. A flag that parses and is then not "
            f"forwarded onto the config is a knob that silently does nothing."
        )


def test_the_data_directory_flags_reach_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The two required path flags, checked against the tmp paths argv used."""
    config, _ = _run_main(monkeypatch, tmp_path)

    assert config.train_data_dir == str(tmp_path / "imagenet_train")
    assert config.val_data_dir == str(tmp_path / "imagenet_val")
    assert config.output_dir == str(tmp_path / "run_output"), (
        "--output-dir never reached TrainingConfig.output_dir; a run would write "
        "into the default 'results' tree instead of where it was told to"
    )


def test_fields_with_no_flag_keep_their_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity twin: prove the probe is looking at the config main() built.

    If every field came back at its default this suite would be reading a fresh
    ``TrainingConfig()``. These four have no CLI flag and MUST be at their
    defaults, while everything in ``EXPECTED`` must not be.
    """
    config, _ = _run_main(monkeypatch, tmp_path)
    for field in FIELDS_WITH_NO_FLAG:
        assert getattr(config, field) == _default(field), (
            f"{field} has no CLI flag but is not at its default "
            f"{_default(field)!r}; this test is not looking at the config it "
            f"thinks it is"
        )


def test_gpu_reaches_the_trainer_not_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``gpu`` is the one dest with no config field -- prove where it does go."""
    config, sentinels = _run_main(monkeypatch, tmp_path)

    assert not hasattr(config, "gpu"), (
        "TrainingConfig grew a `gpu` field; UNMAPPED_DESTS' reason for exempting "
        "it from the config contract is now stale"
    )
    assert sentinels["train_resnet_imagenet"].kwargs == {"gpu_id": GPU_INDEX}, (
        f"--gpu did not reach train_resnet_imagenet(config, gpu_id=...); the "
        f"call received kwargs {sentinels['train_resnet_imagenet'].kwargs!r}"
    )


# ---------------------------------------------------------------------
# --help is cheap
# ---------------------------------------------------------------------


def test_help_exits_zero_without_reaching_anything_expensive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    sentinels = _install_sentinels(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train_resnet.py", "--help"])

    try:
        train_resnet.main()
    except SystemExit as exit_exc:
        code = exit_exc.code
    except _Abort as abort:
        pytest.fail(
            f"--help reached {abort.name!r} before argparse could exit. "
            f"`args = parse_arguments()` must be the FIRST statement of main(), "
            f"above GPU setup, dataset construction and model construction."
        )
    else:
        pytest.fail("main() returned without raising SystemExit on --help")

    assert code == 0, f"--help exited {code!r}, expected 0"

    captured = capsys.readouterr()
    assert "usage:" in captured.out, (
        f"--help printed no usage line; stdout was {captured.out[:200]!r}"
    )

    reached = {name: s.calls for name, s in sentinels.items() if s.calls}
    assert reached == {}, (
        f"--help performed side effects: {reached}. A --help that allocates a "
        f"GPU, builds a dataset or constructs a model is the original defect."
    )


def test_missing_required_flags_exit_two_without_side_effects(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--train-data-dir``/``--val-data-dir`` are required; argparse must say so."""
    sentinels = _install_sentinels(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train_resnet.py"])

    with pytest.raises(SystemExit) as excinfo:
        train_resnet.main()

    assert excinfo.value.code == 2, (
        f"a bare argv exited {excinfo.value.code!r}; argparse reports a missing "
        f"required argument with exit code 2"
    )
    reached = {name: s.calls for name, s in sentinels.items() if s.calls}
    assert reached == {}, f"a rejected argv still performed side effects: {reached}"

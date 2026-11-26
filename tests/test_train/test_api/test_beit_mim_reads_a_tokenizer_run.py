"""``beit_mim`` reads a tokenizer run: the check of the tokenizer's own ``task.json`` before the claim, and the messages that name its command.

plan-2026-11-25T124419-fa94f59c Step 8 (D-013, D-052, D-053). A ``beit_mim`` run takes ``model_args.tokenizer_checkpoint=<a .keras file>``. When that
file sits beside a ``task.json`` that says it is a ``beit_tokenizer`` run's, the run holds the record to its own patch, image size and input
statistics BEFORE the claim (the file is never opened: the stand-ins here are EMPTY ``.keras`` files, so a check that read one would fail), and
refuses a difference naming both values and the tokenizer run. A file with no sibling record (a legacy tokenizer) is checked as it always was, a
sibling that is no tokenizer record (another task's, not JSON, not an object) is not treated as one, and a tokenizer record that fails its own
check is refused. The stand-in record is the one the real task writes (``BeitTokenizerTrainer.task_record``). The refusals of a missing, mis-named
or absent tokenizer name ``python -m train beit_tokenizer``, the command that makes one, and never the legacy script.

Tests write only under ``tmp_path``; no data is read (a claim is a spy).
"""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import numpy as np
import pytest

from train.api import cli, trainer as trainer_module
from train.api.errors import ConfigError
from train.api.tasks import pretrain
from train.api.tasks.pretrain import PretrainConfig, PretrainTrainer
from train.common import pretrain_run
from train.common.image_streams import stream_statistics

from ._tokenizer_world import prepared as tokenizer_prepared, world  # noqa: F401  (``world`` is a fixture)
from .test_init_from_runs import classification_record, pretrain_record
from .test_pretrain_task import BEIT_MODEL, beit_config, make_config

LEGACY, COMMAND = "train.beit.train_tokenizer", "python -m train beit_tokenizer"


class Claimed(Exception):
    """Raised by the spy that replaces the claim: the request passed every refusal and reached ``open_run``."""


def stand_in(root: Path, tweak: Optional[Callable[[Dict[str, Any]], None]] = None, **changes: Any) -> str:
    """A tokenizer run directory: the record the real task writes at the tiny request ``changes`` makes, and an EMPTY ``best_model.keras``."""
    root.mkdir(parents=True)
    trainer, resolved, context = tokenizer_prepared(root / "unclaimed", **changes)
    side = resolved.transform_args.image_size
    record = trainer.task_record(resolved, context, SimpleNamespace(input_shape=(side, side, 3)))
    if tweak is not None:
        tweak(record)
    (root / "task.json").write_text(json.dumps(record))
    (root / "best_model.keras").write_bytes(b"")
    return str(root / "best_model.keras")


def reaches_the_claim(monkeypatch, out: Path, path: str) -> None:
    def spy(run_dir, resume):
        raise Claimed(str(run_dir))

    monkeypatch.setattr(trainer_module, "open_run", spy)
    with pytest.raises(Claimed):
        PretrainTrainer(beit_config(out, path)).run()


def refusal(out: Path, path: str) -> str:
    """The message of a ``ConfigError`` the ``beit_mim`` run at 32 px and patch 4 raises before the claim."""
    with pytest.raises(ConfigError) as caught:
        PretrainTrainer(beit_config(out, path)).run()
    assert not out.exists(), "the refusal came after the claim"
    return str(caught.value)


def nudged(key: str) -> Callable[[Dict[str, Any]], None]:
    def tweak(record: Dict[str, Any]) -> None:
        record[key] = [value + 1e-3 for value in record[key]]

    return tweak


# ---------------------------------------------------------------------
# A tokenizer run beside the file
# ---------------------------------------------------------------------


def test_a_tokenizer_run_of_this_runs_geometry_passes_the_check_and_reaches_the_claim_without_opening_the_file(world, tmp_path, monkeypatch):
    path = stand_in(tmp_path / "tok")
    assert Path(path).stat().st_size == 0, "the control: the file is empty, so only a check that reads JSON can pass"
    reaches_the_claim(monkeypatch, tmp_path / "run", path)


#: What a CIFAR-10 stream standardizes with, as the record holds it (Python floats of the float32 values).
MEAN, STD = ([float(value) for value in values] for values in stream_statistics("cifar10"))
#: ``name: (how the tokenizer run differs, the words the refusal holds)``. This run is a ``beit_mim`` at 32 px and patch 4 on CIFAR-10.
MISMATCHES = {
    "patch": ({"data.patch_size": 8}, ("data.patch_size 8", "data.patch_size 4")),
    "size": ({"transform_args.image_size": 64}, ("transform_args.image_size [64, 64]", "transform_args.image_size [32, 32]")),
    "mean": ({"tweak": nudged("mean")}, ("the input mean", str([value + 1e-3 for value in MEAN]), str(MEAN))),
    "std": ({"tweak": nudged("std")}, ("the input std", str([value + 1e-3 for value in STD]), str(STD))),
}


@pytest.mark.parametrize("case", sorted(MISMATCHES))
def test_a_tokenizer_run_of_another_geometry_or_statistics_is_refused_before_the_claim_naming_both_values_and_the_run(world, tmp_path, case):
    changes, words = MISMATCHES[case]
    path = stand_in(tmp_path / "tok", **changes)
    message = refusal(tmp_path / "run", path)
    assert message.startswith("Invalid value for 'model_args.tokenizer_checkpoint': ") and COMMAND in message
    assert path in message and str(Path(path).parent) in message, "the file and the tokenizer run are named"
    for word in words:
        assert word in message, (word, message)


def test_every_difference_is_named_at_once(world, tmp_path):
    path = stand_in(tmp_path / "tok", tweak=nudged("std"), **{"data.patch_size": 8, "transform_args.image_size": 64})
    message = refusal(tmp_path / "run", path)
    for word in ("data.patch_size 8", "data.patch_size 4", "transform_args.image_size [64, 64]", "transform_args.image_size [32, 32]",
                 "the input std"):
        assert word in message, (word, message)
    assert "the input mean" not in message, "only what differs is named"


def test_a_file_named_through_a_symlink_is_checked_against_the_record_of_the_run_it_points_into(world, tmp_path, monkeypatch):
    # Review pass 1, N2: the sibling was read from the directory of the path AS GIVEN, so a symlink in another directory found no record and the
    # statistics of the tokenizer were checked nowhere. The sibling is read from the RESOLVED path. The copy has no record to find: no check.
    real = stand_in(tmp_path / "tok", **{"data.patch_size": 8})
    (tmp_path / "elsewhere").mkdir()
    link = tmp_path / "elsewhere" / "link.keras"
    link.symlink_to(real)
    message = refusal(tmp_path / "run", str(link))
    assert str(link) in message and str(Path(real).parent) in message, "the file as given and the tokenizer run it points into are named"
    assert "data.patch_size 8" in message and "data.patch_size 4" in message
    named_directly = refusal(tmp_path / "direct", real)
    assert "data.patch_size 8" in named_directly, "the control: the same file named directly is refused"
    copy = tmp_path / "elsewhere" / "copy.keras"
    copy.write_bytes(Path(real).read_bytes())
    reaches_the_claim(monkeypatch, tmp_path / "copied", str(copy))  # a copy has no sibling: the README says no sibling means no check


def test_the_command_line_exits_2_and_claims_nothing(world, tmp_path, capsys):
    path = stand_in(tmp_path / "tok", **{"data.patch_size": 8})
    out = tmp_path / "run"
    code = cli.main(["pretrain", f"out={out}", "model=beit_mim/beit_tiny", "data.dataset=cifar10", f"model_args.tokenizer_checkpoint={path}",
                     "epochs=1", "batch_size=4"])
    assert code == 2 and "data.patch_size 8" in capsys.readouterr().err and not out.exists()


def test_a_tokenizer_record_that_fails_its_own_check_is_refused_naming_the_problem(world, tmp_path):
    path = stand_in(tmp_path / "tok", tweak=lambda record: record.pop("patch_size"))
    message = refusal(tmp_path / "run", path)
    assert message.startswith("Invalid value for 'model_args.tokenizer_checkpoint': ") and "task.json" in message
    assert "patch_size" in message and "beit_tokenizer" in message


# ---------------------------------------------------------------------
# No record, or a record that is not a tokenizer's: the checks it always had
# ---------------------------------------------------------------------


def test_a_keras_file_with_no_sibling_record_passes_this_check_and_fails_only_where_it_always_failed(world, tmp_path, monkeypatch):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "best_model.keras").write_bytes(b"")
    reaches_the_claim(monkeypatch, tmp_path / "run", str(legacy / "best_model.keras"))
    assert "does not exist" in refusal(tmp_path / "other", str(legacy / "gone.keras"))
    assert "does not end in .keras" in refusal(tmp_path / "third", str(legacy / "model.h5"))


OTHERS = {
    "a classifier's record": json.dumps(classification_record()),
    "a pretrain record": json.dumps(pretrain_record("beit_mim/beit_tiny")),
    "not JSON": "this is not json {",
    "a JSON list": "[1, 2, 3]",
    "a record of no task": json.dumps({"schema_version": 2}),
}


@pytest.mark.parametrize("case", sorted(OTHERS))
def test_a_sibling_task_json_that_is_no_tokenizer_record_is_not_treated_as_one(world, tmp_path, monkeypatch, case):
    folder = tmp_path / "other"
    folder.mkdir()
    (folder / "best_model.keras").write_bytes(b"")
    (folder / "task.json").write_text(OTHERS[case])
    reaches_the_claim(monkeypatch, tmp_path / "run", str(folder / "best_model.keras"))


# ---------------------------------------------------------------------
# The messages name the command that makes a tokenizer
# ---------------------------------------------------------------------


def test_the_messages_name_the_command_that_trains_a_tokenizer_and_never_the_legacy_script(world, tmp_path):
    out, texts = tmp_path / "run", {}
    with pytest.raises(ConfigError) as caught:
        PretrainTrainer(make_config(out, model=BEIT_MODEL, model_args={})).run()
    texts["no tokenizer given"] = str(caught.value)
    texts["not a .keras file"] = refusal(tmp_path / "second", str(tmp_path / "tokenizer.h5"))
    texts["no such file"] = refusal(tmp_path / "third", str(tmp_path / "nowhere.keras"))
    (help_text,) = [f.metadata["help"] for f in dataclasses.fields(PretrainConfig) if f.name == "model_args"]
    texts["the help of model_args"] = help_text
    with pytest.raises(ValueError) as narrow:
        pretrain_run.checked_tokenizer_usage(np.arange(96) % 60, path="t.keras")
    texts["a codebook used too narrowly"] = str(narrow.value)
    for name, text in texts.items():
        assert COMMAND in text and LEGACY not in text, (name, text)
    # Nor does either module name it anywhere else (a comment or a docstring would be the next message).
    for module in (pretrain, pretrain_run):
        assert LEGACY not in Path(module.__file__).read_text(), module.__name__

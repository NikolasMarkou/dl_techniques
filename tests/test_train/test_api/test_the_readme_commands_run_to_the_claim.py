"""Every ``python -m train time_series_forecasting ...``, ``language_model_pretraining ...`` and ``text_classification ...`` command a
document prints is a command the CLI accepts (D-046; the text commands are plan 7806a290 steps 7 and 13: before them this guard read the
forecasting commands by name, and a misspelled key in a language command passed).

The first two commands of the forecasting section of ``src/train/README.md`` exited 2 with ``Missing required key 'data'`` and no test
ran them. This test reads the documents, takes every such command line (a backslash continuation joined, a shell comment dropped,
the quoting as a shell reads it) and runs it through ``train.api.cli.main`` in this process up to the run directory: the one place
the pre-claim path ends is ``train.api.trainer.open_run``, which is replaced by a function that raises a sentinel, so a command that
PASSES validation reaches the sentinel, a command the CLI refuses returns 2, and nothing is ever claimed or trained.

Pass means: the command returned 0 (the bare command is the help) or reached the sentinel. A command a document broke (a missing
``data.format=builtin``, a key that does not exist) returns 2 and fails by name. The csv the csv command names is written into
``tmp_path`` (the working directory of the test) with enough rows for the recipe's batch.

A command with ``init=<dir>`` names the ``out`` of an earlier documented command (the README trains the source first). Its
source is a stand-in run written in ``tmp_path`` from that earlier command: the forecast record of its model at its lookback and
horizon (the recipe's when it names none), or the language or classification record of its model at its ``data.seq_len`` and ``data.tokenizer``, its
``config.json`` and a placeholder archive, which the pre-claim path hashes and never loads; the ``init`` value is replaced by that
directory. An ``init`` that names no documented ``out`` fails by name: the example ``init=results/tirex_168`` named a run no command
trained and exited 2 on every checkout (the 4g README).

A text command needs two things a fresh checkout lacks: the TFDS reviews and the pinned tokenizer files. The test stands in for both
(see the anchor in the claim test), so it passes on any machine and still names the tokenizer the check was asked for.

A ``beit_tokenizer`` command and a ``pretrain`` command that reads a tokenizer run (``model_args.tokenizer_checkpoint=``) are run the same way
(plan fa94f59c step 9). Each tokenizer run a command names, as ``init=<out>`` or as ``tokenizer_checkpoint=<out>/best_model.keras``, is the ``out`` of a
documented ``beit_tokenizer`` command, whose stand-in is the record the task writes for that very command and an EMPTY archive: both checks
before the claim read the record only. A path no documented command trains, or a file other than ``best_model.keras`` (the file the verdict of the
run describes), fails by name.
"""

import json
import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import pytest

from train.api import cli, data_args, predictor, registry, trainer
from train.api import text_models
from train.api.config import validate
from train.api.tasks.beit_tokenizer import BeitTokenizerConfig, BeitTokenizerTrainer
from train.api.tasks.dino import DinoConfig, DinoTrainer
from train.api.text_args import CLASS_NAMES
from train.api.forecast_models import HORIZON, LOOKBACK
from train.api.text_tokenizer import tokenizer_spec

from ._forecast_fixtures import world, write_series_csv  # noqa: F401  (``world`` is a fixture)

ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "src" / "train" / "README.md", ROOT / "src" / "train" / "AGENTS.md"]
FORECASTING, LANGUAGE, CLASSIFICATION = "time_series_forecasting", "language_model_pretraining", "text_classification"
TOKENIZER, PRETRAIN = "beit_tokenizer", "pretrain"
DINO, CLASSIFIER = "dino", "image_classification"
#: The recipe's 128 batch needs 21,600 rows of a csv at the default window; the csv the README names is written at this length.
CSV_ROWS = 25_000
OUT, INIT = re.compile(r"(?:^|\s)out=(\S+)"), re.compile(r"(?:^|\s)init=(\S+)")
CHECKPOINT = re.compile(r"(?:^|\s)model_args\.tokenizer_checkpoint=(\S+)")


class Claimed(Exception):
    """Raised where the run directory would be claimed: validation is over."""


def commands(text: str, task: str = FORECASTING) -> List[str]:
    """The argument text after ``<task>`` of every command line in ``text``."""
    joined = re.sub(r"\\\n\s*", " ", text)
    pattern = re.compile(rf"python -m train {task}\b([^`\n]*)")
    return [m.group(1).split(" #")[0].strip() for line in joined.splitlines() for m in pattern.finditer(line)]


def documented(task: str = FORECASTING) -> List[Tuple[str, str]]:
    return [(document.name if document.parent == ROOT else f"{document.parent.name}/{document.name}", tail)
            for document in DOCUMENTS if document.is_file() for tail in commands(document.read_text(), task)]


FOUND = documented()
FOUND_LANGUAGE = documented(LANGUAGE)
FOUND_CLASSIFICATION = documented(CLASSIFICATION)
FOUND_TEXT = [(LANGUAGE, source, tail) for source, tail in FOUND_LANGUAGE] + [(CLASSIFICATION, source, tail) for source, tail in FOUND_CLASSIFICATION]
FOUND_TOKENIZER = documented(TOKENIZER)
#: The ``pretrain`` commands that read a tokenizer run (the others, and the sentences that say a pretrain request is refused, are not run here).
FOUND_PRETRAIN = [(source, tail) for source, tail in documented(PRETRAIN) if CHECKPOINT.search(tail)]
FOUND_TOKENIZER_COMMANDS = [(TOKENIZER, source, tail) for source, tail in FOUND_TOKENIZER] + [(PRETRAIN, source, tail) for source, tail in FOUND_PRETRAIN]
FOUND_DINO = documented(DINO)
DINO_OUTS = {out.group(1) for _, tail in FOUND_DINO if (out := OUT.search(tail))}
#: The ``vit`` pair: the classification commands whose ``init=`` is the ``out`` of a documented ``dino`` command.
FOUND_VIT_PAIR = [(source, tail) for source, tail in documented(CLASSIFIER) if (init := INIT.search(tail)) and init.group(1) in DINO_OUTS]
FOUND_DINO_COMMANDS = [(DINO, source, tail) for source, tail in FOUND_DINO] + [(CLASSIFIER, source, tail) for source, tail in FOUND_VIT_PAIR]


def a_documented_source(name: str, root: Path) -> Path:
    """The stand-in of the run the documented command with ``out=name`` trains, written under ``root``."""
    trained = [tail for _, tail in FOUND if (out := OUT.search(tail)) and out.group(1) == name]
    assert trained, f"init={name} names a run no documented command trains (no command has out={name})"
    args = dict(arg.split("=", 1) for arg in shlex.split(trained[0]) if "=" in arg)
    model = args["model"]
    family, variant = model.split("/")
    record = predictor.build_forecast_record(
        model=model, family=family, variant=variant, lookback=int(args.get("data.lookback", LOOKBACK)),
        horizon=int(args.get("data.horizon", HORIZON)), channels=1, value_cols=None, time_col=None, scale_floor=1e-6,
        decode=family, quantile_levels=None, point_index=None, data_format="builtin", data_sha256=None)
    return stand_in(name, root, model, record)


def stand_in(name: str, root: Path, model: str, record: dict) -> Path:
    """A run directory under ``root`` that holds ``record``, a ``config.json`` and a placeholder archive."""
    source = root / "sources" / Path(name).name
    source.mkdir(parents=True)
    (source / "task.json").write_text(json.dumps(record))
    (source / "config.json").write_text(json.dumps({"model": model, "used": {"model_args": {}}}))
    (source / "best_model.keras").write_bytes(b"a stand-in archive: hashed before the claim, never loaded")
    return source


def a_documented_text_source(name: str, root: Path) -> Path:
    """The stand-in of the run the documented language or classification command with ``out=name`` trains, written under ``root``."""
    trained = [(task, tail) for task, _, tail in FOUND_TEXT if (out := OUT.search(tail)) and out.group(1) == name]
    assert trained, f"init={name} names a run no documented text command trains (no command has out={name})"
    task, tail = trained[0]
    args = dict(arg.split("=", 1) for arg in shlex.split(tail) if "=" in arg)
    family, variant = args["model"].split("/")
    common = dict(model=args["model"], family=family, variant=variant, seq_len=int(args.get("data.seq_len", 256)),
                  tokenizer=tokenizer_spec(args.get("data.tokenizer", "gpt2")).record())
    record = (predictor.build_language_model_record(**common) if task == LANGUAGE
              else predictor.build_text_classification_record(**common, class_names=CLASS_NAMES))
    return stand_in(name, root, args["model"], record)


def a_documented_tokenizer_run(name: str, root: Path) -> Path:
    """The stand-in of the run the documented ``beit_tokenizer`` command with ``out=name`` trains, written under ``root``.

    The record is the one the task writes for THAT command (parsed by the real command line, resolved by the real trainer), and the archive is empty:
    ``init`` reads the record and the file's sha256, and ``beit_mim`` reads the record only, before the claim.
    """
    # DECISION plan-2026-11-25T124419-fa94f59c/D-056
    # The record is written by the real task for the documented command (parsed and resolved by the real code), never built by hand from the
    # geometry and codebook tables: a hand copy would agree with itself, and the pre-claim checks of `init` and of `beit_mim` (the tokenizer run's
    # own patch, size and statistics) would then judge the README against a record the task never writes. Do NOT stand in a placeholder that
    # satisfies the path check alone, and do NOT name a run no documented command trains: that fails here, by name.
    trained = [tail for _, tail in FOUND_TOKENIZER if (out := OUT.search(tail)) and out.group(1) == name]
    assert trained, f"{name} names a tokenizer run no documented command trains (no `python -m train beit_tokenizer` command has out={name})"
    config_path, raw = cli.parse_task_arguments(BeitTokenizerConfig, shlex.split(trained[0]))
    config = validate(BeitTokenizerConfig, cli.layer_command_line(BeitTokenizerConfig, config_path, raw))
    run = BeitTokenizerTrainer(config)
    family, variant = registry.parse_model_name(config.model, task=run.task)
    resolved, context = run._resolve(run.describe_data(config), family, variant)
    side = resolved.transform_args.image_size
    source = root / "tokenizers" / Path(name).name
    source.mkdir(parents=True)
    (source / "task.json").write_text(json.dumps(run.task_record(resolved, context, SimpleNamespace(input_shape=(side, side, 3)))))
    (source / "config.json").write_text(json.dumps({"model": config.model, "used": {"model_args": {}}}))
    (source / "best_model.keras").write_bytes(b"")
    return source


#: Any task's command line, for the init check below.
ANY_COMMAND = re.compile(r"python -m train [a-z_]+\b([^`\n]*)")


def test_every_documented_init_names_a_documented_out():
    """An ``init=results/<x>`` example of ANY task names the ``out`` of a documented command (the reader can run the source).

    The forecasting, BEiT and depth examples of the ``init`` section each named a source no command trained (D-057).
    """
    tails = [m.group(1).split(" #")[0] for document in DOCUMENTS if document.is_file()
             for m in ANY_COMMAND.finditer(re.sub(r"\\\n\s*", " ", document.read_text()))]
    outs = {out.group(1) for tail in tails if (out := OUT.search(tail))}
    inits = [init.group(1) for tail in tails if (init := INIT.search(tail)) and init.group(1).startswith("results/")]
    assert len(inits) >= 5, f"the init section's examples were not found: {inits}"
    assert [name for name in inits if name not in outs] == []


def test_a_documented_init_names_a_documented_source():
    """Vacuity guard for the stand-in: the README holds a forecasting init command, and its source is documented."""
    names = [init.group(1) for _, tail in FOUND if (init := INIT.search(tail))]
    assert names and all(any((out := OUT.search(tail)) and out.group(1) == name for _, tail in FOUND) for name in names)


def test_the_documents_hold_a_builtin_and_a_csv_command_to_run():
    """Vacuity guard: a reader that found nothing would pass every case below."""
    tails = [tail for _, tail in FOUND]
    assert any("data.format=csv" in tail for tail in tails) and any("model=" in tail and "data.path" not in tail for tail in tails)


@pytest.mark.parametrize("source, tail", FOUND, ids=[f"{source}:{tail[:60] or 'bare'}" for source, tail in FOUND])
def test_a_documented_command_passes_the_cli_up_to_the_claim(source, tail, world, tmp_path, monkeypatch, capsys):
    write_series_csv(tmp_path / "series.csv", rows=CSV_ROWS, columns=["load"], time_col="ts")
    claimed: List[Path] = []

    def open_run(run_dir, resume):
        claimed.append(run_dir)
        raise Claimed(str(run_dir))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trainer, "open_run", open_run)
    if init := INIT.search(tail):
        tail = tail.replace(f"init={init.group(1)}", f"init={a_documented_source(init.group(1), tmp_path)}")
    try:
        code = cli.main(["time_series_forecasting", *shlex.split(tail)])
    except Claimed:
        code = None
    error = capsys.readouterr().err
    assert code in (0, None), f"{source}: `python -m train time_series_forecasting {tail}` exited {code}:\n{error[-1500:]}"
    assert code == 0 or len(claimed) == 1, "a command that is not help reaches the claim exactly once, and the claim is never made"


def test_the_documents_hold_the_text_commands_to_run():
    """Vacuity guard: per text task the bare command, a synthetic and an IMDB run; an ``init`` whose source a documented command trains."""
    for found in (FOUND_LANGUAGE, FOUND_CLASSIFICATION):
        tails = [tail for _, tail in found]
        assert "" in tails
        assert any("data.dataset=synthetic" in tail and "data.tokenizer=ascii95" in tail for tail in tails)
        assert any("data.dataset=imdb_reviews" in tail and "data.tokenizer=gpt2" in tail for tail in tails)
    tails = [tail for _, _, tail in FOUND_TEXT]
    inits = [init.group(1) for tail in tails if (init := INIT.search(tail))]
    assert inits and all(any((out := OUT.search(tail)) and out.group(1) == name for tail in tails) for name in inits)
    assert any("model=bert/" in tail for _, tail in FOUND_CLASSIFICATION) and any(INIT.search(tail) for _, tail in FOUND_CLASSIFICATION)


@pytest.mark.parametrize("task, source, tail", FOUND_TEXT, ids=[f"{task}:{source}:{tail[:60] or 'bare'}" for task, source, tail in FOUND_TEXT])
def test_a_documented_text_command_passes_the_cli_up_to_the_claim(task, source, tail, world, tmp_path, monkeypatch, capsys):
    # DECISION plan-2026-11-24T092348-7806a290/D-045
    # The two things a fresh checkout lacks are stood in for: a TFDS directory that holds the file the pre-claim check looks for, and a
    # recorder in place of the pinned-files hash check (test_language_model_pretraining_task.py runs that check against real and wrong
    # directories). Do NOT read the real pinned directory or the real TFDS cache here (the guard would fail on every machine but the
    # author's), and do NOT drop the assertion on the recorder (a command that names a tokenizer must reach the check with that name).
    reviews = tmp_path / "tfds" / "imdb_reviews" / "plain_text" / "1.0.0"
    reviews.mkdir(parents=True)
    (reviews / "dataset_info.json").write_text("{}")
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path / "tfds"))
    asked: List[Tuple[str, object]] = []
    # the check both text tasks make before the claim (`text_models.check_text_window`) reads the name in `text_models`
    monkeypatch.setattr(text_models, "check_tokenizer_dir", lambda name, directory=None: asked.append((name, directory)))
    claimed: List[Path] = []

    def open_run(run_dir, resume):
        claimed.append(run_dir)
        raise Claimed(str(run_dir))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trainer, "open_run", open_run)
    if init := INIT.search(tail):
        tail = tail.replace(f"init={init.group(1)}", f"init={a_documented_text_source(init.group(1), tmp_path)}")
    try:
        code = cli.main([task, *shlex.split(tail)])
    except Claimed:
        code = None
    error = capsys.readouterr().err
    assert code in (0, None), f"{source}: `python -m train {task} {tail}` exited {code}:\n{error[-1500:]}"
    assert code == 0 or len(claimed) == 1, "a command that is not help reaches the claim exactly once, and the claim is never made"
    if code is None:
        named = dict(arg.split("=", 1) for arg in shlex.split(tail) if "=" in arg).get("data.tokenizer", "gpt2")
        assert [name for name, _ in asked] == [named], "the pinned-files check ran once, for the tokenizer the command names"


def test_the_documents_hold_the_tokenizer_commands_to_run():
    """Vacuity guard: the bare command, a cifar10 run, an ``init`` whose source a documented tokenizer command trains, and every ``beit_mim``
    command that reads a tokenizer run names the ``best_model.keras`` of a documented one; no command line of the legacy trainer is printed."""
    tails = [tail for _, tail in FOUND_TOKENIZER]
    assert "" in tails and any("data.dataset=cifar10" in tail and "epochs=5" in tail for tail in tails)
    outs = {out.group(1) for tail in tails if (out := OUT.search(tail))}
    inits = [init.group(1) for tail in tails if (init := INIT.search(tail))]
    assert inits and [name for name in inits if name not in outs] == [], "a tokenizer `init` names a run no documented command trains"
    named = [checkpoint.group(1) for _, tail in FOUND_PRETRAIN if (checkpoint := CHECKPOINT.search(tail))]
    assert len(named) >= 2, f"the pretrain section's example and the init section's name a tokenizer run: {named}"
    wrong = [path for path in named if not path.endswith("/best_model.keras") or path[: -len("/best_model.keras")] not in outs]
    assert wrong == [], f"a tokenizer_checkpoint must be the best_model.keras of a documented beit_tokenizer out (those are {sorted(outs)}): {wrong}"
    assert not any("python -m train.beit.train_tokenizer" in document.read_text() for document in DOCUMENTS if document.is_file()), (
        "a document prints a command line of the legacy tokenizer trainer: the command that makes a tokenizer is `python -m train beit_tokenizer`")


@pytest.mark.parametrize("task, source, tail", FOUND_TOKENIZER_COMMANDS, ids=[f"{task}:{source}:{tail[:60] or 'bare'}" for task, source, tail in FOUND_TOKENIZER_COMMANDS])
def test_a_documented_tokenizer_command_passes_the_cli_up_to_the_claim(task, source, tail, world, tmp_path, monkeypatch, capsys):
    claimed: List[Path] = []

    def open_run(run_dir, resume):
        claimed.append(run_dir)
        raise Claimed(str(run_dir))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trainer, "open_run", open_run)
    if init := INIT.search(tail):
        tail = tail.replace(f"init={init.group(1)}", f"init={a_documented_tokenizer_run(init.group(1), tmp_path)}")
    if checkpoint := CHECKPOINT.search(tail):
        named = checkpoint.group(1)
        assert named.endswith("/best_model.keras"), f"{source}: tokenizer_checkpoint={named}: a tokenizer run is read by its best_model.keras, the file its verdict describes"
        run = a_documented_tokenizer_run(named[: -len("/best_model.keras")], tmp_path)
        tail = tail.replace(f"tokenizer_checkpoint={named}", f"tokenizer_checkpoint={run / 'best_model.keras'}")
    try:
        code = cli.main([task, *shlex.split(tail)])
    except Claimed:
        code = None
    error = capsys.readouterr().err
    assert code in (0, None), f"{source}: `python -m train {task} {tail}` exited {code}:\n{error[-1500:]}"
    assert code == 0 or len(claimed) == 1, "a command that is not help reaches the claim exactly once, and the claim is never made"


def a_documented_dino_run(name: str, root: Path) -> Path:
    """The stand-in of the run the documented ``dino`` command with ``out=name`` trains, written under ``root``.

    The record is the one the task writes for THAT command (parsed by the real command line, resolved by the real trainer): the classifier's
    pre-claim read of a dino source holds its own patch, its size and its statistics to it, and the archive is hashed there, never loaded.
    """
    # DECISION plan-2026-11-26T224507-e3381378/D-075
    # Same rule as the tokenizer's stand-in (plan fa94f59c D-056): the record is written by the real task for the documented command, never built
    # by hand from the geometry table, so the `init` pair is judged against the record a run writes. Do NOT stand in a placeholder that satisfies
    # the path check alone, and do NOT name a run no documented command trains: that fails here, by name.
    trained = [tail for _, tail in FOUND_DINO if (out := OUT.search(tail)) and out.group(1) == name]
    assert trained, f"{name} names a dino run no documented command trains (no `python -m train dino` command has out={name})"
    config_path, raw = cli.parse_task_arguments(DinoConfig, shlex.split(trained[0]))
    config = validate(DinoConfig, cli.layer_command_line(DinoConfig, config_path, raw))
    run = DinoTrainer(config)
    family, variant = registry.parse_model_name(config.model, task=run.task)
    resolved, context = run._resolve(run.describe_data(config), family, variant)
    side = resolved.transform_args.image_size
    source = root / "dino" / Path(name).name
    source.mkdir(parents=True)
    (source / "task.json").write_text(json.dumps(run.task_record(resolved, context, SimpleNamespace(input_shape=(side, side, 3)))))
    (source / "config.json").write_text(json.dumps({"model": config.model, "used": {"model_args": {}}}))
    (source / "best_model.keras").write_bytes(b"a stand-in archive: hashed before the claim, never loaded")
    return source


def test_the_documents_hold_the_dino_commands_to_run():
    """Vacuity guard: the bare command, a cifar10 smoke run, an Imagenette run, and a ``vit`` pair command whose ``init`` is a documented dino ``out``;
    no command line of the legacy trainer is printed (the command that trains DINO is ``python -m train dino``)."""
    tails = [tail for _, tail in FOUND_DINO]
    assert "" in tails
    assert any("data.dataset=cifar10" in tail and "debug_args.smoke=true" in tail for tail in tails)
    assert any("data.dataset=imagenette" in tail and "model=dino_v1/tiny" in tail for tail in tails)
    assert DINO_OUTS and FOUND_VIT_PAIR, "the init section's pair: a vit classification command whose init is the out of a documented dino command"
    assert all("model=vit/" in tail for _, tail in FOUND_VIT_PAIR)
    assert not any("python -m train.dino.train_dino" in document.read_text() for document in DOCUMENTS if document.is_file()), (
        "a document prints a command line of the legacy DINO trainer: the command that trains DINO is `python -m train dino`")


@pytest.mark.parametrize("task, source, tail", FOUND_DINO_COMMANDS, ids=[f"{task}:{source}:{tail[:60] or 'bare'}" for task, source, tail in FOUND_DINO_COMMANDS])
def test_a_documented_dino_command_passes_the_cli_up_to_the_claim(task, source, tail, world, tmp_path, monkeypatch, capsys):
    claimed: List[Path] = []
    records: List[str] = []

    def open_run(run_dir, resume):
        claimed.append(run_dir)
        raise Claimed(str(run_dir))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trainer, "open_run", open_run)
    # A fresh checkout has no TFDS directory: the check is stood in for, and an Imagenette command must reach it (the dino and the classifier read it alike).
    monkeypatch.setattr(data_args, "_check_imagenette_records", lambda: records.append("imagenette"))
    if init := INIT.search(tail):
        tail = tail.replace(f"init={init.group(1)}", f"init={a_documented_dino_run(init.group(1), tmp_path)}")
    try:
        code = cli.main([task, *shlex.split(tail)])
    except Claimed:
        code = None
    error = capsys.readouterr().err
    assert code in (0, None), f"{source}: `python -m train {task} {tail}` exited {code}:\n{error[-1500:]}"
    assert code == 0 or len(claimed) == 1, "a command that is not help reaches the claim exactly once, and the claim is never made"
    if code is None:
        assert bool(records) == ("data.dataset=imagenette" in tail), "an Imagenette command reaches the records check and no other does"

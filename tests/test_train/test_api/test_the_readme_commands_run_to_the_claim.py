"""Every ``python -m train time_series_forecasting ...`` command a document prints is a command the CLI accepts (D-046).

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
horizon (the recipe's when it names none), its ``config.json`` and a placeholder archive, which the pre-claim path hashes and
never loads; the ``init`` value is replaced by that directory. An ``init`` that names no documented ``out`` fails by name: the
example ``init=results/tirex_168`` named a run no command trained and exited 2 on every checkout (the 4g README).
"""

import json
import re
import shlex
from pathlib import Path
from typing import List, Tuple

import pytest

from train.api import cli, predictor, trainer
from train.api.forecast_models import HORIZON, LOOKBACK

from ._forecast_fixtures import world, write_series_csv  # noqa: F401  (``world`` is a fixture)

ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "src" / "train" / "README.md", ROOT / "src" / "train" / "AGENTS.md"]
COMMAND = re.compile(r"python -m train time_series_forecasting\b([^`\n]*)")
#: The recipe's 128 batch needs 21,600 rows of a csv at the default window; the csv the README names is written at this length.
CSV_ROWS = 25_000
OUT, INIT = re.compile(r"(?:^|\s)out=(\S+)"), re.compile(r"(?:^|\s)init=(\S+)")


class Claimed(Exception):
    """Raised where the run directory would be claimed: validation is over."""


def commands(text: str) -> List[str]:
    """The argument text after ``time_series_forecasting`` of every command line in ``text``."""
    joined = re.sub(r"\\\n\s*", " ", text)
    return [m.group(1).split(" #")[0].strip() for line in joined.splitlines() for m in COMMAND.finditer(line)]


def documented() -> List[Tuple[str, str]]:
    return [(document.name if document.parent == ROOT else f"{document.parent.name}/{document.name}", tail)
            for document in DOCUMENTS if document.is_file() for tail in commands(document.read_text())]


FOUND = documented()


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
    source = root / "sources" / Path(name).name
    source.mkdir(parents=True)
    (source / "task.json").write_text(json.dumps(record))
    (source / "config.json").write_text(json.dumps({"model": model, "used": {"model_args": {}}}))
    (source / "best_model.keras").write_bytes(b"a stand-in archive: hashed before the claim, never loaded")
    return source


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

"""The numbers the forecasting documentation states are the numbers of the code (plan-2026-11-13T105600-3a8319ef, step 18).

``test_the_knob_docs_name_real_keys.py`` reads the keys, defaults and paths of the knob region. The prose around it also states
numbers that the code defines: how many families the registry holds and how they split over the tasks, how many tasks there are,
how many cells of the forecasting export matrix are refused, the width of the strict bar's tie band. A number typed into a
document and never read again rots silently (step 17 planted a wrong count in each of these places and nothing in the repository
failed; only the derivation script of that step did). Each claim below is DERIVED from the code and must appear in the document
word for word, and every ``src/...`` or ``tests/...`` path the two documents cite must exist.
"""

import re
from pathlib import Path

import pytest

from train.api import exporting, recipes, registry
from train.api import forecast_predictor as fp
from train.api.tasks import BUILTIN_TASKS, register_builtin_tasks
from train.common import forecast_run

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "src" / "train" / "README.md"
AGENTS = ROOT / "src" / "train" / "AGENTS.md"
FORECASTING = "time_series_forecasting"


@pytest.fixture(scope="module")
def world():
    """An empty registry holding every built-in family and task."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def families_per_task():
    counts = {}
    for spec in registry.MODELS.values():
        for task in spec.tasks:
            counts[task] = counts.get(task, 0) + 1
    return counts


def export_cells():
    """``(cells, refused)`` of the forecasting export matrix, by asking ``check_export`` of every row and format."""
    record = {"lookback": 168, "channels": 1, "horizon": 24}
    cells = refused = 0
    for row in forecast_run.DECODE:
        for fmt in exporting.FORMATS:
            cells += 1
            try:
                fp.check_export({**record, "decode": row}, fmt, (168, 1))
            except Exception:
                refused += 1
    return cells, refused


def test_the_family_count_and_its_split_over_the_tasks_are_the_registrys(world):
    counts = families_per_task()
    order = [spec.name for spec in BUILTIN_TASKS]
    split = " + ".join(str(counts[task]) for task in order)
    assert sum(counts.values()) == len(registry.MODELS), "a family is trained by exactly one task"
    assert f"meets {len(registry.MODELS)} families ({split}" in AGENTS.read_text()


def test_the_number_of_tasks_is_the_registrys_in_both_documents(world):
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    sentence = f"It has {words[len(registry.TASKS)]} tasks"
    assert FORECASTING in registry.TASKS
    assert sentence in README.read_text() and sentence in AGENTS.read_text()


def test_the_export_matrix_counts_are_the_cells_check_export_refuses(world):
    cells, refused = export_cells()
    words = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    text = README.read_text()
    assert f"({cells} cells, {cells - refused} export and {refused} are refused" in text
    assert f"refuses {words[refused]} of its {cells} cells" in text


def test_the_width_of_the_bars_tie_band_is_the_codes():
    assert f"more than {forecast_run.TIE_ULPS} float32 steps (`forecast_run.TIE_ULPS`)" in README.read_text()


CITED = re.compile(r"`((?:tests|src)/[A-Za-z0-9_./\-]+\.(?:py|md|json|sh|yaml|yml))`")


@pytest.mark.parametrize("document", [README, AGENTS], ids=["README.md", "AGENTS.md"])
def test_every_source_and_test_path_the_document_cites_exists(document):
    cited = sorted(set(CITED.findall(document.read_text())))
    assert cited, "vacuity guard: the document cites paths"
    missing = [path for path in cited if not (ROOT / path).exists()]
    assert not missing, f"{document.name} cites paths that do not exist: {missing}"

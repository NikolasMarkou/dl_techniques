"""The numbers the forecasting documentation states are the numbers of the code.

``test_the_knob_docs_name_real_keys.py`` reads the keys, defaults and paths of the knob region. The prose around it also states
numbers that the code defines: how many families the registry holds and how they split over the tasks, how many tasks there are,
how many cells of the forecasting export matrix are refused, the width of the strict bar's tie band. A number typed into a
document and never read again rots silently (a wrong count planted in each of these places failed nothing in the repository
until this test existed). Each claim below is DERIVED from the code and must appear in the document word for word, and every ``src/...`` or ``tests/...`` path the two documents cite must exist.
"""

import dataclasses
import math
import re
from pathlib import Path

import pytest

from train.api import exporting, predictor, registry
from train.api import forecast_models as fm
from train.api import forecast_predictor as fp
from train.api.config import PRECISIONS, validate
from train.api.errors import ConfigValidationError
from train.api.tasks import BUILTIN_TASKS
from train.api.forecast_args import ForecastTransformArgs, TimeSeriesData
from train.api.forecast_series import FLAT_SHARE, FLOAT32_RANGE
from train.api.tasks.time_series_forecasting import TimeSeriesForecastingConfig
from train.common import forecast_run

from ._forecast_fixtures import world  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "src" / "train" / "README.md"
AGENTS = ROOT / "src" / "train" / "AGENTS.md"
FORECASTING = "time_series_forecasting"


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


def test_every_other_n_tasks_phrase_counts_the_tasks_of_the_registry_but_one(world):
    """A task's section says "the other four tasks refuse these keys the same way": the number is the registry's, minus the task itself.

    The denoising section kept "the other three tasks" after the pretrain and forecasting tasks were added; no test read it.
    """
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    pattern = re.compile(r"\bthe other (one|two|three|four|five|six|seven) tasks?\b")
    found = pattern.findall(README.read_text()) + pattern.findall(AGENTS.read_text())
    assert len(found) >= 3, "the control: the README names the other tasks in the denoising, pretrain and forecasting sections"
    assert set(found) == {words[len(registry.TASKS) - 1]}, found


def test_the_export_matrix_counts_are_the_cells_check_export_refuses(world):
    cells, refused = export_cells()
    words = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    text = README.read_text()
    assert f"({cells} cells, {cells - refused} export and {refused} are refused" in text
    assert f"refuses {words[refused]} of its {cells} cells" in text


def test_the_width_of_the_bars_tie_band_is_the_codes():
    """D-043: ``TIE_ULPS * max(1, ceil(sqrt(n_batches / 16)))``; the README states the formula and the widths at the batch counts it names."""
    text = README.read_text()
    assert "`TIE_ULPS * max(1, ceil(sqrt(n_batches / 16)))` float32 steps" in text
    widths = [forecast_run.tie_band(n) for n in (16, 17, 2048)]
    assert widths == [16, 32, 192]
    assert f"({widths[0]} up to 16 select batches, {widths[1]} at the 17 of the defaults, {widths[2]} at 2,048;" in text
    assert forecast_run.TIE_ULPS == widths[0]


def test_the_precisions_the_readme_refuses_are_the_ones_the_code_refuses_before_the_claim(world, tmp_path):
    """D-041: ``precision`` other than ``float32`` is refused by name; the README said "not refused and NOT verified" before."""
    refused = [name for name in PRECISIONS if name != "float32"]
    assert len(refused) >= 1, "vacuity guard: there is a precision to refuse"
    text = " ".join(README.read_text().split())
    names = " and ".join(f"`{name}`" for name in refused)
    assert f"`precision` must be `float32`: {names} are REFUSED by name before the run directory is claimed" in text
    assert "not refused and NOT verified" not in text
    for name in refused:
        asked = {"out": str(tmp_path / "run"), "model": "nbeats/tiny", "data": {"lookback": 32, "horizon": 8}, "precision": name}
        with pytest.raises(ConfigValidationError, match=rf"Invalid value for 'precision': '{name}' is not accepted for {FORECASTING}"):
            validate(TimeSeriesForecastingConfig, asked)
    assert not (tmp_path / "run").exists()


def test_the_readme_lists_the_keys_of_the_forecasting_record_in_the_codes_order():
    """The record paragraph names every key a forecasting ``task.json`` holds beyond the five that identify the run, in the code's order."""
    identity = ("schema_version", "task", "model", "family", "variant")
    keys = [key for key in predictor.RECORD_KEYS[FORECASTING] if key not in identity]
    assert "time_col" in keys, "vacuity guard: the key this step added is in the table"
    sentence = "a forecasting record has " + ", ".join(f"`{key}`" for key in keys[:-2]) + f", `{keys[-2]}` and `{keys[-1]}`"
    assert sentence in " ".join(README.read_text().split())
    agents = " ".join(AGENTS.read_text().split())
    start = agents.index("a forecasting record (the fourth kind) has")
    assert all(f"`{key}`" in agents[start:start + 600] for key in keys), "AGENTS.md names the same keys"


def test_the_minimums_the_readme_states_are_the_family_rows():
    """D-042: the lookback and horizon minimums of the README are ``FamilyRow.min_lookback``, ``min_lookback_over`` and ``min_horizon``."""
    text = " ".join(README.read_text().split())
    prism = ", ".join(f"{variant} {floor}" for variant, floor in fm.FAMILIES["prism"].min_lookback.items())
    assert f"`prism` {prism};" in text, prism
    (variant, (key, extra)), = fm.FAMILIES["ets"].min_lookback_over.items()
    assert f"`ets/{variant}` needs `model_args.{key}` plus {extra};" in text
    assert [family for family, row in fm.FAMILIES.items() if row.min_lookback] == ["prism"], "the README names the families with a row"
    assert [family for family, row in fm.FAMILIES.items() if row.min_horizon > 1] == ["nbeats"]
    assert f"`data.horizon` below {fm.FAMILIES['nbeats'].min_horizon} for `nbeats`" in text
    assert f"builds at {fm.MIN_LOOKBACK}, the floor of every family" in text


def test_the_readme_says_a_hand_edited_scale_floor_cannot_be_detected():
    text = " ".join(README.read_text().split())
    assert "`scale_floor` is not checked against the model" in text and "cannot be detected" in text


def test_the_required_keys_the_documents_state_are_the_configs_fields_without_a_default():
    """Review W2: the README said `out` and `model` were required, the module said `out`, `model` and `data`, and the code wanted all three."""
    config = TimeSeriesForecastingConfig
    required = [f.name for f in dataclasses.fields(config) if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    assert required == ["out", "model", "data"], "vacuity guard: the fields that have no default"
    words = ", ".join(f"`{name}`" for name in required[:-1]) + f" and `{required[-1]}`"
    text = " ".join(README.read_text().split())
    assert f"{words} are required" in text and f"Required: {words};" in text
    module = " ".join((ROOT / "src" / "train" / "api" / "tasks" / "time_series_forecasting.py").read_text().split())
    assert ", ".join(f"``{name}``" for name in required[:-1]) + f" and ``{required[-1]}`` are required" in module
    assert "the group `data` itself must be named" in text, "the README says what naming it takes (`data.format=builtin`)"


def test_the_lookback_help_states_the_minimum_the_code_refuses_below():
    help_text = {f.name: f for f in dataclasses.fields(TimeSeriesData)}["lookback"].metadata["help"]
    assert fm.MIN_LOOKBACK == 2, "vacuity guard: the floor of every family"
    assert f"(at least {fm.MIN_LOOKBACK};" in help_text


def test_the_readme_states_the_shift_bound_the_flat_rule_has_and_the_flat_flag_and_counts():
    """D-045: shift invariance holds while a window's std is above ``scale_floor`` times its mean |x|; a flat window is in raw units."""
    floor = ForecastTransformArgs().scale_floor
    bound = f"1e{round(math.log10(1 / floor))}"
    text = " ".join(README.read_text().split())
    assert f"shift invariance holds while a window's std is above `transform_args.scale_floor` ({floor:g}) times its mean |x|, that is while the level over the std is below {bound}" in text
    assert "a flat window is trained and scored in the series' raw units" in text
    assert "`predict` returns `flat`, one bool per window" in text and "`flat_windows`" in text
    assert f"more than {FLAT_SHARE:.0%} of the windows of a segment" in text and "at stride `data.horizon`" in text
    span = ", ".join(f"{x:.0e}".replace("e+", "e") for x in FLOAT32_RANGE)
    assert f"outside [{span}]" in text and "float32 range" in text


CITED = re.compile(r"`((?:tests|src)/[A-Za-z0-9_./\-]+\.(?:py|md|json|sh|yaml|yml))`")


@pytest.mark.parametrize("document", [README, AGENTS], ids=["README.md", "AGENTS.md"])
def test_every_source_and_test_path_the_document_cites_exists(document):
    cited = sorted(set(CITED.findall(document.read_text())))
    assert cited, "vacuity guard: the document cites paths"
    missing = [path for path in cited if not (ROOT / path).exists()]
    assert not missing, f"{document.name} cites paths that do not exist: {missing}"

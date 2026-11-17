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
from train.api.forecast_series import FLAT_SHARE, FLOAT32_RANGE, FLOAT32_TINY
from train.api.tasks.time_series_forecasting import TimeSeriesForecastingConfig
from train.common import forecast_run

from ._forecast_fixtures import SHIFT_BOUND, world  # noqa: F401  (fixture)

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
    assert "std <= max(1.18e-38, transform_args.scale_floor * mean|context|)" in text, "the flat rule: float32's smallest normal number is its one absolute term"
    assert FLOAT32_TINY == 1.1754943508222875e-38 and "There is no larger absolute guard" in text and "`span * std(deviation / span)`" in text
    assert "The two earlier guards, 1e-15 and 1e-18," in text and "max(1e-18" not in text and "max(1e-15" not in text, "the old guards are named once, as the ones that were wrong"
    assert "are resolution-flat" in text and "an EXACTLY constant window" in text and "never refused" in text, "the two kinds of flat window"
    # review pass 4 W1: a difference removes a constant level exactly, so the text names it as a remedy and does not say it does not help
    assert "or give the model the first or the seasonal difference of the column" in text and "the model then forecasts differences, not values" in text
    assert "differencing does not help" not in text and "as shifted" not in text
    assert "subtract the level" in text and "multiply the column by a power of ten" in text and "which lowers the RELATIVE term only" in text
    assert "The one override is `transform_args.scale_floor`" in text and "so the override is stated and not advised" in text
    assert "about one window std at a level of 1e7 over a std of 1" in text, "the cost of the override (review pass 4 NOTE 6)"
    assert "naive-last MAE of 0.012 and 1.202" in text and "depend on the unit the csv is written in" in text, "review pass 4 NOTE 4"
    assert SHIFT_BOUND == 1e-7 and f"holds the shift to {SHIFT_BOUND:.0e}".replace("e-0", "e-") + " of a std for every decode row" in text
    assert "all nine decode rows" in text, "the shift bound says on which rows it was measured"
    # D-049 (review pass 5): ONE decision for the flag, the count, the scale, the training function and the file, and the measured band it leaves
    assert "read ONE decision (D-049)" in text and "counts exactly the windows that were scored in the series' own unit" in text
    assert "3 counted against 0 scored on a csv with 5 windows 0.1 percent above the threshold" in text, "review pass 5 W2"
    assert "disagree on at most 1 window of 4,000 at any of twelve distances from 1e-7 to 3e-5 of the threshold" in text, "the measured band (r4_rel_fine.log)"
    assert "`standardize_tf` agrees with it on 3,600 of 3,600 windows at three levels" in text
    assert "within 4e-6 of a window std of `predict` at window spreads from 1e-19 to 1e-30" in text and "0.09 to 0.23 window std from `predict` on float64 windows" in text
    assert "7e15 window stds apart" in text and "forecast about 2.6e8 times its own level" in text
    # D-050 (review pass 6 W1): a float32 denormal reads 0 in the decision of every implementation but onnxruntime's; the TensorFlow files also flush the cells
    assert "the decision reads a denormal as 0, as TensorFlow does (D-050)" in text and "both read 0 for a denormal now" in text, "review pass 6 W1"
    assert "the summary said 0 flat and the scoring treated 38 as flat" in text and "1,153 of 2,000 windows at 1.0001 times it at lookback 168" in text, "s11b and t2"
    assert "8.7 window stds at a spread of 0.9 times 1.18e-38, 5.9 at 1.0001 times, 3.9 at 1.5 times, 1.8 at 3.5 times, 0.32 at 10 times, 0.028 at 100 times, 0.047 at 1e-36 and 3e-6 at 1e-33" in text
    assert "flush a float32 value below 1.18e-38 to 0 and" not in text and "297 of 2,000 windows at 0.9 times it" not in text, "the pass-5 sentence, measured false in pass 6"
    assert "after the UTF-8 byte-order mark of a spreadsheet's \"CSV UTF-8\", if the file has one" in text, "review pass 6 N5"
    assert "A flat window has no scale" in text and "forecast about 2.6e8 times its own level" in text, "review pass 5 NOTE 8"
    assert "which means little where the series' unit is far from 1" in text
    for gone in ("within two float32 steps", "0 disagreements at every distance", "squares deviations in float32"):
        assert gone not in text, f"{gone!r}: a claim of the squares-guard design, measured false in pass 5 (W1)"
    span = ", ".join(f"{x:.0e}".replace("e+", "e") for x in FLOAT32_RANGE)
    assert f"outside [{span}]" in text and "the range this task trains and serves in, and no longer a float32 limit" in text
    assert f"`predict` refuses a window with a cell above {FLOAT32_RANGE[1]:.0e}".replace("e+", "e") + " in magnitude by name, the same bound as a training column" in text, "review pass 5 NOTE 6"
    assert "a non-finite cell, a cell above 1e15 in magnitude, an empty input" in text


def test_the_readme_states_what_a_blank_line_does_in_a_csv():
    """Review pass 4 NOTE 7 and pass 5 NOTE 1 and NOTE 2: a blank line inside the series is refused with its row, one before the header or
    after the last row shifts nothing and is ignored, and a text cell is refused with its row and the cell as written."""
    text = " ".join(README.read_text().split())
    assert "a blank line INSIDE the series is a missing cell refused with its row" in text
    assert "blank lines before the header (after the UTF-8 byte-order mark of a spreadsheet's \"CSV UTF-8\", if the file has one) and after the last row shift nothing and are ignored" in text
    assert "a text cell, a whitespace-only line included, is refused with its row and the cell as written" in text


def test_the_readme_gate_section_names_the_code_head_it_ran_on_and_says_the_first_gate_is_superseded():
    """The final gate (plan 3a8319ef, Step 21) ran on one code HEAD; the README is a docs-only commit after it, and says so."""
    text = " ".join(README.read_text().split())
    start = text.index("**Gate numbers** (plan 3a8319ef")
    gate = text[start:start + 12000]
    assert "run on HEAD `78e3295df` of the branch `train-api/phase-4e`" in gate
    assert "the code HEAD `78e3295df56d9df62609d9afd28af038be383980`" in gate
    assert "the commit that holds this README is a docs-only commit after that HEAD" in gate
    assert "the first gate, run on HEAD `1904f07a8`" in gate and "is superseded and every number below replaces its numbers" in gate
    assert "PENDING" not in text and "NOT YET MEASURED" not in text
    assert gate.count("| G1 |") == 1 and gate.count("| G2 |") == 1 and gate.count("| G3 |") == 1 and gate.count("| G4 |") == 1


CITED = re.compile(r"`((?:tests|src)/[A-Za-z0-9_./\-]+\.(?:py|md|json|sh|yaml|yml))`")


@pytest.mark.parametrize("document", [README, AGENTS], ids=["README.md", "AGENTS.md"])
def test_every_source_and_test_path_the_document_cites_exists(document):
    cited = sorted(set(CITED.findall(document.read_text())))
    assert cited, "vacuity guard: the document cites paths"
    missing = [path for path in cited if not (ROOT / path).exists()]
    assert not missing, f"{document.name} cites paths that do not exist: {missing}"

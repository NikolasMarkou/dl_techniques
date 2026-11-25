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
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    sentence = f"It has {words[len(registry.TASKS)]} tasks"
    assert FORECASTING in registry.TASKS
    assert sentence in README.read_text() and sentence in AGENTS.read_text()
    # two more sentences count the tasks: the file form of the data group is every task's, and every task writes a record
    count = words[len(registry.TASKS)]
    assert f"holds for all {count} tasks" in " ".join(README.read_text().split())
    assert f"All {count} tasks write a record" in " ".join(AGENTS.read_text().split())


def test_every_other_n_tasks_phrase_counts_the_tasks_of_the_registry_but_one(world):
    """A task's section says "the other four tasks refuse these keys the same way": the number is the registry's, minus the task itself.

    The denoising section kept "the other three tasks" after the pretrain and forecasting tasks were added; no test read it.
    """
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    pattern = re.compile(r"\bthe other (one|two|three|four|five|six|seven|eight) tasks?\b")
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
    """The README said `out` and `model` were required, the module said `out`, `model` and `data`, and the code wanted all three."""
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
    # a difference removes a constant level exactly, so the text names it as a remedy and does not say it does not help
    assert "or give the model the first or the seasonal difference of the column" in text and "the model then forecasts differences, not values" in text
    assert "differencing does not help" not in text and "as shifted" not in text
    assert "subtract the level" in text and "multiply the column by a power of ten" in text and "which lowers the RELATIVE term only" in text
    assert "The one override is `transform_args.scale_floor`" in text and "so the override is stated and not advised" in text
    assert "about one window std at a level of 1e7 over a std of 1" in text, "the cost of the override"
    assert "naive-last MAE of 0.012 and 1.202" in text and "depend on the unit the csv is written in" in text, "a flat window is scored in the column's own unit"
    assert SHIFT_BOUND == 1e-7 and f"holds the shift to {SHIFT_BOUND:.0e}".replace("e-0", "e-") + " of a std for every decode row" in text
    assert "all nine decode rows" in text, "the shift bound says on which rows it was measured"
    # D-049: ONE decision for the flag, the count, the scale, the training function and the file, and the measured band it leaves
    assert "read ONE decision (D-049)" in text and "counts exactly the windows that were scored in the series' own unit" in text
    assert "3 counted against 0 scored on a csv with 5 windows 0.1 percent above the threshold" in text, "the count and the scoring were two decisions"
    assert "disagree only within about 1e-5 of the threshold, as a relative distance of the window's std measured on the float32 values" in text, "the measured band (t3_band_distance.log)"
    assert "the largest distance of a disagreement in any cell was 1.0e-5, at 3 channels and lookback 1,024" in text and "1 window of 4,000" not in text, "the band is a distance, not a rate"
    assert "`standardize_tf` agrees with it on 3,600 of 3,600 windows at three levels" in text
    # maxima once measured on 16 windows were exceeded on 64; re-measured on 128 windows per cell and the sample size is in the text
    assert "within 6e-6 of a window std of `predict` at window spreads from 1e-33 to 1e-19 and from 1e16 to 1e35 (128 windows per cell" in text
    assert "0.14 to 0.35 window std from `predict` on float64 windows (the largest gap over the 512 windows of each of 45 cells" in text and "0.09 to 0.23" not in text and "0.08 to 0.31" not in text, "the gap is the largest seen over 512 windows per cell"
    assert "so these are largest values seen and not bounds" in text and "a second sample of 512 windows per cell, two seeds of 256, gave 4.8e-6 and 5.8e-6, still within 6e-6" in text, "the maxima are largest values seen, with the second sample"
    assert "to about 3e-7 of its forecast" in text and "within 3.2e-7" not in text and "the largest seen was 2.6e-7 in 512 windows per cell over two seeds, and 3.8e-7 once" in text, "flat windows: an order of magnitude, not a bound"
    assert "within about 5e-6 of a window std of `predict` at every spread from 1.0001 times 1.18e-38 up, for the windows that `predict` standardizes" in text and "within 4.3e-6 of a window std" not in text, "onnx: for the windows predict standardizes"
    assert "differs from `predict` on those by up to 0.0073 of the forecast" in text, "onnx: the windows predict flags flat"
    assert "within 4e-6 of a window std of `predict` at window spreads from 1e-19 to 1e-30" not in text and "(r5_files.log" not in text, "the 16-window numbers are gone"
    assert "7e15 window stds apart" in text and "forecast about 2.6e8 times its own level" in text
    # D-050: a float32 denormal reads 0 in the decision of every implementation but onnxruntime's; the TensorFlow files also flush the cells
    assert "the decision reads a denormal as 0, as TensorFlow does (D-050)" in text and "both read 0 for a denormal now" in text, "a denormal reads 0 in the decision"
    assert "the summary said 0 flat and the scoring treated 38 as flat" in text and "1,153 of 2,000 windows at 1.0001 times it at lookback 168" in text, "s11b and t2"
    # the single-seed list read as a smooth curve; it is not monotone, so the text gives ranges and says it varies with the seed
    assert "up to about 9 window stds at spreads from 0.9 to 1.5 times 1.18e-38 (4 to 9 in every sample measured, with no steady fall over that range" in text, "the tiny-edge gap is a range"
    assert "about 2 at 3 to 3.5 times, 0.3 to 0.6 at 10 times, 0.01 to 0.03 at 100 times, 0.01 to 0.05 at 1e-36 and under 4e-6 at 1e-33" in text, "the tiny-edge gaps as ranges"
    assert "8.7 window stds at a spread of 0.9 times" not in text and "3.9 at 1.5 times" not in text, "the single-seed digits are gone"
    assert "flush a float32 value below 1.18e-38 to 0 and" not in text and "297 of 2,000 windows at 0.9 times it" not in text, "a sentence measured false"
    assert "after the UTF-8 byte-order mark of a spreadsheet's \"CSV UTF-8\", if the file has one" in text, "a byte-order mark before a blank line"
    assert "A flat window has no scale" in text and "forecast about 2.6e8 times its own level" in text, "what a flat window costs"
    assert "which means little where the series' unit is far from 1" in text
    for gone in ("within two float32 steps", "0 disagreements at every distance", "squares deviations in float32"):
        assert gone not in text, f"{gone!r}: a claim of the squares-guard design, measured false"
    span = ", ".join(f"{x:.0e}".replace("e+", "e") for x in FLOAT32_RANGE)
    assert f"outside [{span}]" in text and "the range this task trains and serves in, and no longer a float32 limit" in text
    assert f"`predict` refuses a window with a cell above {FLOAT32_RANGE[1]:.0e}".replace("e+", "e") + " in magnitude by name, the same bound as a training column" in text, "predict's bound is the training range"
    assert "a non-finite cell, a cell above 1e15 in magnitude, an empty input" in text


def test_the_readme_states_what_a_blank_line_does_in_a_csv():
    """A blank line inside the series is refused with its row, one before the header or
    after the last row shifts nothing and is ignored, and a text cell is refused with its row and the cell as written."""
    text = " ".join(README.read_text().split())
    assert "a blank line INSIDE the series is a missing cell refused with its row" in text
    assert "blank lines before the header (after the UTF-8 byte-order mark of a spreadsheet's \"CSV UTF-8\", if the file has one) and after the last row shift nothing and are ignored" in text
    assert "a text cell, a whitespace-only line included, is refused with its row and the cell as written" in text


def test_the_readme_gate_section_names_the_code_head_it_ran_on_and_says_the_first_gate_is_superseded():
    """The final gate (plan 3a8319ef) ran on one code HEAD; the README says what the commits after it change (no behaviour), D-053."""
    text = " ".join(README.read_text().split())
    start = text.index("**Gate numbers** (plan 3a8319ef")
    gate = text[start:start + 12000]
    assert "run on HEAD `c4cffe8fa` of the branch `train-api/phase-4e`" in gate
    assert "the code HEAD `c4cffe8fabccc0539169051cc949f070bc4073a0`" in gate
    assert ("the commits after that HEAD change this README, comments, docstrings, formatting, one equivalent check (`is_count`) "
            "and where one help paragraph lives, and no behaviour (checked by targeted tests, D-053)") in gate
    assert "docs-only" not in gate, "the commits after the gate HEAD are not docs-only (D-053)"
    assert "the first gate, run on HEAD `1904f07a8`" in gate and "is superseded and every number below replaces its numbers" in gate
    assert "PENDING" not in text and "NOT YET MEASURED" not in text
    assert gate.count("| G1 |") == 1 and gate.count("| G2 |") == 1 and gate.count("| G3 |") == 1 and gate.count("| G4 |") == 1


def test_the_language_section_states_the_numbers_of_the_code():
    """Plan 7806a290 step 7: the variants, the recipe, the vocabularies, the id limit and the pinned files of the language section are the code's."""
    import inspect

    from dl_techniques.models.language.gpt2.gpt2 import GPT2
    from train.api import text_args, text_models, text_predictor, text_tokenizer as tt
    from train.api.exporting import TFLITE_CHECK_RTOL
    from train.api.trainer import SMOKE_MAX_SAMPLES

    text = " ".join(README.read_text().split())
    table = GPT2.MODEL_VARIANTS
    tiny, bigger = table["tiny"], sorted((name for name in table if name != "tiny"), key=lambda name: table[name]["embed_dim"])
    rows = [f"`{name}` ({table[name]['embed_dim']:,}, {table[name]['depth']}, {table[name]['num_heads']})" for name in bigger]
    assert len({table[name]["max_seq_len"] for name in bigger}) == 1, "the README says the last four share one position table"
    assert (f"`tiny` ({tiny['embed_dim']} wide, {tiny['depth']} blocks, {tiny['num_heads']} heads, a position table of {tiny['max_seq_len']} rows), "
            f"{', '.join(rows[:-1])} and {rows[-1]}, the last four with {table[bigger[0]]['max_seq_len']:,} rows") in text
    recipe, optim = text_models.LM_RECIPE, text_models.LM_RECIPE["optim_args"]
    assert (optim["optimizer"], optim["decay_over"]) == ("adamw", "total")
    assert (f"AdamW at {optim['lr']:.0e}".replace("e-0", "e-") + f", weight decay {optim['weight_decay']:g}, {optim['warmup_epochs']} warmup epoch, "
            f"cosine decay over the whole run, a global gradient clip of {optim['global_clipnorm']:.1f}, batch {recipe['batch_size']}, "
            f"{recipe['epochs']} epochs, patience {recipe['checkpoint_args']['early_stopping_patience']}.") in text
    ascii95, gpt2 = tt.tokenizer_spec(tt.ASCII95), tt.tokenizer_spec(tt.GPT2)
    assert (gpt2.cls_id, gpt2.sep_id, gpt2.pad_id, gpt2.mask_id) == tuple(range(gpt2.n_vocab, gpt2.n_vocab + 4))
    assert f"a character tokenizer: {ascii95.vocab_size} ids, no file" in text
    assert (f"{gpt2.n_vocab:,} ids, the end-of-text id {gpt2.eot_id:,} among them, then `cls`, `sep`, `pad` and `mask` at {gpt2.cls_id:,} to "
            f"{gpt2.mask_id:,}, so `vocab_size` is {gpt2.vocab_size:,}") in text
    assert f"the library's default of {inspect.signature(GPT2.__init__).parameters['vocab_size'].default:,}" in text
    assert f"exact below 2^24 = {text_predictor.FLOAT_ID_LIMIT:,}" in text and text_predictor.FLOAT_ID_LIMIT == 2 ** 24
    assert text_predictor.TOP_K == 5 and "the five most likely ids" in text
    assert f"export check and its tests allow {TFLITE_CHECK_RTOL:.0e}".replace("e-0", "e-") in text
    assert "64 float32 steps at those 183 batches" in text and forecast_run.tie_band(183) == 64
    assert f"synthetic: {text_args.SYNTHETIC_TRAIN_DOCUMENTS} train and {text_args.SYNTHETIC_TEST_DOCUMENTS} test when null" in text
    assert f"`debug_args.smoke` sets {SMOKE_MAX_SAMPLES} when null" in text and f"`data.max_samples` {SMOKE_MAX_SAMPLES} when it is null" in text
    assert all(name in text for name, _ in tt.GPT2_FILES) and tt.DEFAULT_TOKENIZER_DIR in text
    # review pass 1, N4: tiktoken's loader deletes a cache file of another hash and the pinned directory is writable (D-015)
    assert "Do not point `TIKTOKEN_CACHE_DIR` at the pinned directory" in text and "or set `TIKTOKEN_CACHE_DIR` to the pinned directory" not in text
    # the stream bullet says what the four shared sentences say for this task (D-050; D-049 named the defect they were): the clause it
    # quotes is the code's, so a change of the wording moves the document with it, and the other tasks' sentences still say a fresh pass
    from train.api.args import CheckpointArgs
    from train.api.config import BACKUP_HELP, BACKUP_HELP_EXACT
    from train.api.tasks.language_model_pretraining import LanguageModelPretrainingTrainer

    clause = "from the stream at the step that backup holds (the batches the unkilled run reads next)"
    assert LanguageModelPretrainingTrainer.resume_stream_exact is True and "`TaskTrainer.resume_stream_exact`" in text
    assert clause in text and clause in BACKUP_HELP_EXACT and "open defect" not in text and "wrong for this task" not in text
    assert "so no example repeats and none is missed" in text and '"from a fresh pass of the data"' in text
    assert all("from a fresh pass of the data" in sentence for sentence in (
        BACKUP_HELP, {f.name: f for f in dataclasses.fields(CheckpointArgs)}["backup_every_steps"].metadata["help"]))


def test_the_agents_counts_about_the_tasks_their_hooks_and_the_language_record_are_the_codes(world):
    """Plan 7806a290 step 7: five sentences that counted something no test read (a plant of each was green before this test)."""
    import importlib
    import inspect

    from train.api.trainer import TaskTrainer

    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    agents, readme = " ".join(AGENTS.read_text().split()), " ".join(README.read_text().split())
    trainers = []
    for spec in BUILTIN_TASKS:
        module = importlib.import_module(spec.config_class.partition(":")[0])
        trainers.append(next(v for v in vars(module).values() if inspect.isclass(v) and issubclass(v, TaskTrainer) and v.__module__ == module.__name__))
    users = [trainer for trainer in trainers if {"fit_kwargs", "resume_stream"} <= set(vars(trainer))]
    assert f"Both hooks have {words[len(users)]} users today" in agents
    staged = [path for path in (ROOT / "tests/test_train/test_api").glob("test_*_task.py") if "def test_the_task_does_not_name_a_stage" in path.read_text()]
    assert f"pinned for {words[len(staged)]} tasks only" in agents
    assert f"`ExportRule({', '.join(predictor.ExportRule._fields)})`" in agents
    ordinal = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")[list(predictor.RECORD_KEYS).index("language_model_pretraining")]
    start = agents.index(f"a language-model record (the {ordinal} kind")
    keys = [key for key in predictor.RECORD_KEYS["language_model_pretraining"] if key not in ("schema_version", "task", "model", "family", "variant")]
    assert all(f"`{key}`" in agents[start:start + 700] for key in keys) and f"the {ordinal} record kind" in readme
    exact = [trainer for trainer in trainers if trainer.resume_stream_exact]
    sets = f"{words[len(exact)]} task sets it today" if len(exact) == 1 else f"{words[len(exact)]} tasks set it today"
    assert "`TaskTrainer.resume_stream_exact`" in agents and sets in agents
    compared = predictor.DATA_KEYS["language_model_pretraining"]
    resume = "a resume compares " + ", ".join(f"`{key}`" for key in compared[:-1]) + f" and `{compared[-1]}`"
    assert resume in agents[start:start + 900] and resume in readme, "the keys a resume compares"


def test_the_eighth_record_kind_is_described_as_the_code_defines_it():
    """Plan 7806a290 step 12: the text-classification record's ordinal, own keys and values, and resume keys (a wrong word stayed green)."""
    task, language = "text_classification", "language_model_pretraining"
    kind = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")[list(predictor.RECORD_KEYS).index(task)]
    agents = " ".join(AGENTS.read_text().split())
    sentence = agents[agents.index(f"a text-classification record (the {kind} kind"):][:520]
    own = [key for key in predictor.RECORD_KEYS[task] if key not in predictor.RECORD_KEYS[language] or key == "output"]
    assert own and all(f"`{key}`" in sentence for key in own), own
    assert f"`{predictor.TEXT_LAYOUTS[0]}`" in sentence and f"`{predictor.TEXT_CLASSIFICATION_OUTPUTS[0]}`" in sentence
    assert predictor.DATA_KEYS[task] == (*predictor.DATA_KEYS[language], "class_names")
    shared = ("zero", "one", "two", "three", "four", "five")[len(predictor.DATA_KEYS[language])]
    assert f"a resume compares the same {shared} and `class_names`" in sentence
    assert f"`{task}` the {kind} (the same" in agents, "the tables bullet names the kind"
    readme = " ".join(README.read_text().split())  # step 13: the README says it too
    sentence = readme[readme.index(f"the {kind} record kind"):][:700]
    assert all(f"`{key}`" in sentence for key in own) and f"`{predictor.TEXT_LAYOUTS[0]}`" in sentence
    compared = predictor.DATA_KEYS[task]
    assert "a resume compares " + ", ".join(f"`{key}`" for key in compared[:-1]) + f" and `{compared[-1]}`" in sentence


def test_the_classification_section_and_the_init_pairs_state_the_numbers_of_the_code(world):
    """Plan 7806a290 step 13: the BERT variants, the classifier recipe, the constants the two text sections share and the count of declared pairs."""
    import inspect

    from dl_techniques.models.language.bert.model import BERT
    from train.api import text_args, text_models
    from train.api.trainer import SMOKE_MAX_SAMPLES
    from train.common import text_data, text_run

    text = " ".join(README.read_text().split())
    assert f"plus {({3: 'three'})[int(text_run.BAR_STANDARD_ERRORS)]} binomial standard errors of that share on those rows" in text, "the bar's margin"
    assert "max(1, max_samples // 4)" in inspect.getsource(text_data.corpus_documents) and "else N and max(1, N // 4)" in text, "the test split's size"
    table, positions = BERT.MODEL_VARIANTS, inspect.signature(BERT.__init__).parameters["max_position_embeddings"].default
    order = sorted(table, key=lambda name: table[name]["hidden_size"])
    rows = [f"`{name}` ({table[name]['hidden_size']:,}, {table[name]['num_layers']}, {table[name]['num_heads']})" for name in order[1:]]
    assert order[0] == "tiny" and all("max_position_embeddings" not in variant for variant in table.values()), "every variant has the library's table"
    tiny = table["tiny"]
    assert (f"`tiny` ({tiny['hidden_size']} wide, {tiny['num_layers']} blocks, {tiny['num_heads']} heads), {', '.join(rows[:-1])} and {rows[-1]}, "
            f"each with a position table of {positions} rows") in text
    recipe, optim = text_models.CLASSIFIER_RECIPE, text_models.CLASSIFIER_RECIPE["optim_args"]
    assert (f"AdamW at {optim['lr']:.0e}".replace("e-0", "e-") + f", weight decay {optim['weight_decay']:g}, {optim['warmup_epochs']} warmup epoch, "
            f"cosine decay over the whole run, a global gradient clip of {optim['global_clipnorm']:.1f}, batch {recipe['batch_size']}, "
            f"{recipe['epochs']} epochs, patience {recipe['checkpoint_args']['early_stopping_patience']}.") in text
    keys = text[text.index("<!-- keys-of: text_classification -->"):][:2500]  # the section's own table and its next paragraphs
    assert f"`debug_args.smoke` sets {SMOKE_MAX_SAMPLES} when null" in keys and text_args.CLASS_NAMES == ("neg", "pos")
    pairs = sum(len(spec.transfer.sources) for spec in registry.MODELS.values() if spec.transfer)
    words = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    assert f"are the {words[pairs]} declared cross-task pairs" in text, "the init section counts the pairs the families declare"
    # review pass 2, concern 6: the recipe sentence says what the notice says the bar is, the guesser percentages are the exact binomial's
    # (IMDB: 1,256 of 2,498 selection rows, the bar and the bare share) in the README and in text_run, and both keys rows name the fallback
    from scipy.stats import binom

    rows, share = 2498, 1256 / 2498

    def guesser(bar):   # the share of 8-epoch runs of a label-independent guesser whose best epoch clears the float32 compare
        needed = next(count for count in range(rows + 1) if text_run.strictly_above(count / rows, bar))
        return 100 * (1 - binom.cdf(needed - 1, rows, 0.5) ** 8)

    said = f"clears in {guesser(share + text_run.accuracy_margin(share, rows)):.2f} percent of 8-epoch runs (the bare share let {guesser(share):.1f} percent through)"
    assert text_models._CLASSIFIER_BAR in text and "the bar is a floor" not in text and said in text
    assert all(f"{guesser(share):.1f} percent" in source and "98.3" not in source for source in (inspect.getdoc(text_run), inspect.getsource(text_run.accuracy_margin)))
    keys_rows = [line for line in README.read_text().splitlines() if line.startswith("| `data.tokenizer_dir` |")]
    assert len(keys_rows) == 2 and all("or the pinned copy when that directory is gone" in line for line in keys_rows)
    assert "a rounding band and not a noise margin" in text, "under init the loaded weights' term is the shared one, named for what it is"


CITED = re.compile(r"`((?:tests|src)/[A-Za-z0-9_./\-]+\.(?:py|md|json|sh|yaml|yml))`")


@pytest.mark.parametrize("document", [README, AGENTS], ids=["README.md", "AGENTS.md"])
def test_every_source_and_test_path_the_document_cites_exists(document):
    cited = sorted(set(CITED.findall(document.read_text())))
    assert cited, "vacuity guard: the document cites paths"
    missing = [path for path in cited if not (ROOT / path).exists()]
    assert not missing, f"{document.name} cites paths that do not exist: {missing}"

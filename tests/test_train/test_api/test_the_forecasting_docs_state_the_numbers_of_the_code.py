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
VQ_README = ROOT / "src" / "dl_techniques" / "models" / "vision" / "vq_vae_rotation" / "README.md"
TOKENIZER_DATA = ROOT / "src" / "train" / "common" / "tokenizer_data.py"
FORECASTING = "time_series_forecasting"


def defined_tests() -> str:
    """The text of every ``test_*.py`` of the API tests: a test is looked up BY NAME in it (``def name(``), never by importing its module, which would register that
    module's fixtures and state in this process."""
    return "\n".join(path.read_text() for path in sorted((ROOT / "tests" / "test_train" / "test_api").glob("test_*.py")))


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
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
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
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    pattern = re.compile(r"\bthe other (one|two|three|four|five|six|seven|eight|nine|ten) tasks?\b")
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

    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
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
    words = {1: "one", 2: "two", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    assert f"are the {words[pairs]} declared cross-task pairs" in text, "the init section counts the pairs the families declare"
    # Review pass 1, N1: "Those four, the two pretrain pairs below and that language-model pair" counts the declared pairs by the task of their SOURCE, derived here;
    # the tokenizer takes no cross-task pair (an earlier run of its own task only), so its sentence comes after the count and is no part of it.
    by_source = [task for spec in registry.MODELS.values() if spec.transfer for (task, _family) in spec.transfer.sources]
    earlier = [task for task in by_source if task not in ("pretrain", "language_model_pretraining")]
    assert f"Those {words[len(earlier)]}, the {words[by_source.count('pretrain')]} pretrain pairs below and that language-model pair are the {words[pairs]} declared" in text
    assert by_source.count("language_model_pretraining") == 1 and len(earlier) + by_source.count("pretrain") + 1 == pairs
    assert registry.MODELS["vq_vae_rotation"].transfer.sources == {}, "a tokenizer run takes no cross-task source"
    assert text.index(f"are the {words[pairs]} declared cross-task pairs") < text.index("which `init` refuses by policy (see `beit_tokenizer`)")
    assert "at its own size and at 64 px" not in text, "the 64 px copy is the engine's census with the refusal bypassed, not something `init` does"
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


def tokenizer_refusals(side, patch, codes, floor):
    """The messages of the first seven rows of the README's refusal table, as the code raises them, in the table's order."""
    from train.api import tokenizer_args, tokenizer_models as tm
    from train.api.errors import ConfigError

    messages = []
    for call in (lambda: registry.parse_model_name("vit/vit_pico", task=tm.TASK),
                 lambda: tokenizer_args.check_geometry(3, side, codes, floor=floor, smoke=False),
                 lambda: tokenizer_args.check_geometry(1, side, codes, floor=floor, smoke=False),
                 lambda: tokenizer_args.check_geometry(patch, side, 32, floor=floor, smoke=False)):
        with pytest.raises(ConfigError) as caught:
            call()
        messages.append(str(caught.value))
    return messages + [tm.problems_of({"kmeans_init": True})[0], tm.problems_of({"hidden_channels": 0})[0], tm.problems_of({"use_ema": "False"})[0]]


def test_the_tokenizer_section_states_the_numbers_of_the_code(world):
    """Plan fa94f59c step 9: the variant, the model keys, the recipe, the geometry, the floor, the selection split and four refusals of the ``beit_tokenizer`` section."""
    from dl_techniques.datasets.vision.classification import get_spec
    from train.api import recipes, tokenizer_models as tm
    from train.api.errors import ConfigError
    from train.api.pretrain_args import GEOMETRY
    from train.common import pretrain_run, tokenizer_data as td
    from train.common.classification_data import DEFAULT_VALIDATION_SPLIT

    from ._tokenizer_world import cell_index_codes, with_image_wide_bits

    text = " ".join(README.read_text().split())
    start = text.index("**`beit_tokenizer`** (plan fa94f59c")
    section = text[start:text.index("<!-- tokenizer-gate:end -->", start)]
    words = {4: "four", 5: "five", 6: "six"}
    small, (side, patch), codes = tm.VARIANTS["small"], GEOMETRY["cifar10"], tm.CODEBOOK["cifar10"]
    model = recipes.build_model(registry.MODELS[tm.FAMILY], "small", None, (side, side, 3), {"num_embeddings": codes, "patch_size": patch})
    assert f"{model.count_params():,} parameters in {len(model.weights)} weights at {side} px with {codes} codes" in section
    assert f"{small['hidden_channels']} hidden channels, {small['num_res_blocks']} residual blocks and an embedding of {small['embedding_dim']}" in section
    offered = ", ".join(f"`{key}`" for key in tm.ALLOWED[:-1]) + f", `{tm.ALLOWED[-1]}`"
    assert f"{words[len(tm.ALLOWED)].capitalize()} `model_args` are offered ({offered}), {len(tm.SUPPLIED)} are given by the task and {len(tm.REFUSED)} more are refused" in section
    optim = tm.RECIPE["optim_args"]
    assert (f"AdamW at {optim['lr']:.0e}".replace("e-0", "e-") + f", weight decay {optim['weight_decay']:g}, {optim['warmup_epochs']} warmup epochs, cosine decay over "
            f"the whole run to {optim['min_lr_ratio']:g} of the peak, a global gradient clip of {optim['global_clipnorm']:.1f}, batch {tm.BATCH_SIZE}, {tm.EPOCHS} epochs, "
            f"patience {tm.PATIENCE}") in section
    assert " and ".join(f"`{name}` is {size} px at patch {cell} (grid {size // cell} x {size // cell}, {tm.CODEBOOK[name]:,} codes)"
                        for name, (size, cell) in GEOMETRY.items()) in section
    floor, cells = pretrain_run.MIN_TOKENIZER_PERPLEXITY, (side // patch) ** 2
    selection = round(DEFAULT_VALIDATION_SPLIT * get_spec("cifar10").train_size)
    batches = -(-selection // tm.BATCH_SIZE)
    assert f"is at least the floor of {floor:g} effective codes" in section
    # Review pass 1, W3: the second floor and the three controls that show why the pooled one is not enough (derived from the function, never typed)
    alone, one, two = (td.per_cell_usage(codes).perplexity for codes in (cell_index_codes(64), with_image_wide_bits(1), with_image_wide_bits(2)))
    assert f"is at least {td.MIN_PER_CELL_PERPLEXITY:g} (`tokenizer_data.MIN_PER_CELL_PERPLEXITY`)" in section
    assert f"worth {alone:g} per cell, the cell index plus one image-wide bit {one:g} and two bits {two:g}" in section
    assert f"a code that is the cell index and nothing else, whatever the image, has a pooled perplexity of exactly {cells} and meets it" in section
    assert f"({forecast_run.tie_band(batches)} at the {batches} selection batches of CIFAR-10 at the defaults)" in section
    assert f"({selection:,} rows of seed 42)" in section and f"({selection:,} images x {cells} cells = {selection * cells:,} positions)" in section
    refusals = tokenizer_refusals(side, patch, codes, floor)
    assert len(refusals) == 7 and all(f"`{message}`" in section for message in refusals), refusals
    # Review pass 1, W2: the `init` refusal of another image size is the row's own sentence (the weights would copy), quoted whole
    from train.api import init_from

    source = {"channels": 3, "input_size": [side, side], "patch_size": patch, "num_embeddings": codes}
    with pytest.raises(ConfigError) as caught:
        init_from._check_identity(tm.FAMILY, tm.TASK, registry.MODELS[tm.FAMILY].transfer, source, {**source, "input_size": [2 * side, 2 * side]})
    assert f"`{caught.value}`" in section and "sizes its weights by transform_args.image_size" not in section
    # the file `beit_mim` takes is best_model.keras: the checkpoint is selected on the reconstruction alone, so it holds the epoch the verdict judged (D-006)
    assert "**`best_model.keras` is the file `beit_mim` takes.**" in section and "so give it `best_model.keras`" in section and "the last epoch's" in section
    # Review pass 2. W1: every offered key is held to its declared type, and `use_ema` is quoted (the library reads the string "False" as true). N1: what the floors do not certify,
    # and the test that pins it. N3: the size of a selection split the per-cell number reads true from, and the rows it means at the default split. N4: the two entries that lift
    # the size refusal of `init` (read off the row: one in `fixed`, one in `policy`).
    assert f"`{tm.problems_of({'use_ema': 'False'})[0]}`" in section and "a bool is no int and no float" in section
    locality = "test_an_image_wide_code_repeated_at_every_cell_meets_both_floors_so_the_floors_do_not_test_locality"
    assert f"def {locality}(" in defined_tests() and f"`{locality}`" in section
    assert "**The bar does not test locality:** the two floors count how the codes vary across images and cells, not whether a code describes its own patch" in section
    assert "so the reconstruction part, which asks only to be strictly below the constant image, is the only evidence of local content" in section
    assert f"one of fewer than {td.MIN_PER_CELL_IMAGES} images (`tokenizer_data.MIN_PER_CELL_IMAGES`" in section and td.MIN_PER_CELL_IMAGES == 16
    assert f"`data.max_samples` of {round(td.MIN_PER_CELL_IMAGES / DEFAULT_VALIDATION_SPLIT)} at the default split of {DEFAULT_VALIDATION_SPLIT:g}" in section
    row = registry.MODELS[tm.FAMILY].transfer
    assert "input_size" in row.fixed and set(row.policy) == {"input_size"}, "the refusal is two entries of the row: one in `fixed`, one in `policy`"
    assert "lifting the refusal is two entries of the family row (the `fixed` tuple and the `policy` entry) and its census test" in section
    assert "neither floor tests locality" in " ".join(AGENTS.read_text().split())


def test_the_tokenizer_record_and_the_init_keys_the_documents_name_are_the_codes():
    """Plan fa94f59c step 9: the ninth record kind's keys and the keys a resume compares (README and AGENTS), and the geometry keys of ``init`` (AGENTS)."""
    from train.api import init_from

    task, text = "beit_tokenizer", " ".join(README.read_text().split())
    agents = " ".join(AGENTS.read_text().split())
    assert list(predictor.RECORD_KEYS).index(task) == 8, "the ninth kind"
    keys = [key for key in predictor.RECORD_KEYS[task] if key not in ("schema_version", "task", "model", "family", "variant")]
    own = ", ".join(f"`{key}`" for key in keys[:-1]) + f" and `{keys[-1]}`"
    assert f"writes `task.json` (schema version 2, the ninth record kind) before its first epoch: {own} (`{predictor.TOKENIZER_OUTPUTS[0]}`)" in text
    compared = predictor.DATA_KEYS[task]
    assert f"a resume compares the {('zero', 'one', 'two', 'three', 'four', 'five', 'six')[len(compared)]} from `{compared[0]}` to `{compared[-1]}`" in text
    assert "a tokenizer record (the ninth kind" in agents and "a resume compares `" + "`, `".join(compared[:-1]) + f"` and `{compared[-1]}`" in agents
    assert "`fixed` (" + ", ".join(f"`{key}`" for key in init_from.GEOMETRY_NAMES) + ": refused before the claim)" in agents
    assert all(f"`{key}`" in agents[agents.index("returns the record keys this run will write"):][:400] for key in (*init_from.GEOMETRY_NAMES, "normalize", "data_range"))
    assert tuple(init_from.HEAD_KEYS[task]) == () and "neither has a tokenizer" in text, "a tokenizer has no head to reset"


PLACEHOLDERS = re.compile(r"<run>|<out>|\.\.\.")


def refusal_rows():
    """``[(case, message cell)]``: the rows of the README's refusal table of the ``beit_tokenizer`` section, in its order."""
    text = README.read_text()
    start = text.index("* **What is refused, and the message.**", text.index("**`beit_tokenizer`** (plan fa94f59c"))
    block = text[start:text.index("* **Limits.**", start)]
    rows = [line.strip()[2:-2].split(" | ", 1) for line in block.splitlines() if line.startswith("| ") and not line.startswith(("| Case", "|---"))]
    return [(case.strip(), message.strip()) for case, message in rows]


def is_the_message(cell, live):
    """Whether the backticked message of a table cell is ``live``: equal, or, when the cell is abbreviated (``<run>``, ``<out>``, ``...``), every piece between the
    placeholders is in ``live``, in the cell's order."""
    assert cell.startswith("`") and cell.endswith("`") and cell.count("`") == 2, cell
    quoted = cell[1:-1]
    if not PLACEHOLDERS.search(quoted):
        return quoted == live
    at = 0
    for piece in (piece.strip() for piece in PLACEHOLDERS.split(quoted)):
        found = live.find(piece, at) if piece else at
        if found < 0:
            return False
        at = found + len(piece)
    return True


def test_every_row_of_the_refusal_table_is_the_message_the_code_raises(world, tmp_path):
    """Hygiene finding 13 (Step 12): the table is read whole. Seven rows were compared with the code and four were written by hand (an ``init`` from another patch, size or
    codebook, an ``init`` from a ``beit_mim`` run, a ``beit_mim`` run beside the tokenizer of another patch, an ``init`` after the claim): each cell is now the message the code
    raises, equal or, where the cell is abbreviated (``<run>``, ``<out>``, ``...``), with every piece between the marks in the message, in order."""
    import json
    from types import SimpleNamespace

    from train.api import init_from, run_sequence, tokenizer_models as tm
    from train.api.errors import ConfigError
    from train.api.pretrain_args import GEOMETRY
    from train.api.tasks.beit_tokenizer import train_beit_tokenizer
    from train.api.tasks.pretrain import PretrainTrainer
    from train.common import pretrain_run
    from train.common.image_streams import stream_statistics

    from ._tokenizer_world import request

    # the matcher can fail: an abbreviated cell is read in order, an exact one is read whole
    assert is_the_message("`a <run> b ... c`", "a /x/y b and c") and not is_the_message("`b <run> a`", "a /x b")
    assert is_the_message("`a b`", "a b") and not is_the_message("`a b`", "a b c") and not is_the_message("`a <out> d`", "a b c")

    (side, patch), codes = GEOMETRY["cifar10"], tm.CODEBOOK["cifar10"]
    mean, std = (list(map(float, values)) for values in stream_statistics("cifar10"))
    rows = refusal_rows()
    cells = [cell for _, cell in rows]
    assert len(cells) == 11, [case for case, _ in rows]
    # rows 1 to 7: one quoted message each
    for number, (cell, live) in enumerate(zip(cells, tokenizer_refusals(side, patch, codes, pretrain_run.MIN_TOKENIZER_PERPLEXITY)), 1):
        assert is_the_message(cell, live), f"row {number}: the table quotes {cell}, the code raises {live}"

    # row 8, `init` from another patch, size or codebook: three messages of the one check (the codebook one quoted after "...")
    spec = registry.MODELS[tm.FAMILY].transfer
    source = {"channels": 3, "input_size": [side, side], "patch_size": patch, "num_embeddings": codes}

    def identity(**changed):
        with pytest.raises(ConfigError) as caught:
            init_from._check_identity(tm.FAMILY, tm.TASK, spec, source, {**source, **changed})
        return str(caught.value)

    by_patch, by_codebook, by_size = identity(patch_size=2 * patch), identity(num_embeddings=codes // 2), identity(input_size=[2 * side, 2 * side])
    assert f"`{by_patch}`" in cells[7] and f"`... {by_codebook.split('sizes its weights ', 1)[1]}`" in cells[7] and f"`{by_size}`" in cells[7], (cells[7], by_patch, by_codebook)

    # row 9, `init` from a `beit_mim` run: a stand-in run directory the real task refuses before the claim
    beit_mim = tmp_path / "beit_mim_run"
    beit_mim.mkdir()
    record = predictor.build_pretrain_record(model="beit_mim/beit_tiny", family="beit_mim", variant="beit_tiny", input_size=(side, side), channels=3, mean=mean, std=std,
                                             patch_size=patch, tokenizer="0" * 64)
    (beit_mim / "task.json").write_text(json.dumps(record))
    (beit_mim / "config.json").write_text(json.dumps({"model": "beit_mim/beit_tiny", "used": {"model_args": {}}}))
    (beit_mim / "best_model.keras").write_bytes(b"not an archive: nothing before the claim opens it")
    with pytest.raises(ConfigError) as caught:
        train_beit_tokenizer(**request(tmp_path / "run", init=str(beit_mim)))
    assert is_the_message(cells[8], str(caught.value)), f"row 9: the table quotes {cells[8]}, the code raises {caught.value}"

    # row 10, a `beit_mim` run at another patch than its tokenizer run's: the pre-claim check of the file's sibling record
    run = tmp_path / "tokenizer_run"
    run.mkdir()
    (run / "task.json").write_text(json.dumps(predictor.build_tokenizer_record(
        model=f"{tm.FAMILY}/small", family=tm.FAMILY, variant="small", input_size=(side, side), channels=3, mean=mean, std=std, patch_size=patch, num_embeddings=codes)))
    (run / "best_model.keras").write_bytes(b"x")
    this_run = SimpleNamespace(transform_args=SimpleNamespace(image_size=side), data=SimpleNamespace(dataset="cifar10", patch_size=2 * patch))
    with pytest.raises(ConfigError) as caught:
        PretrainTrainer._check_tokenizer_run(run / "best_model.keras", this_run)
    assert is_the_message(cells[9], str(caught.value)), f"row 10: the table quotes {cells[9]}, the code raises {caught.value}"

    # row 11, after the claim, `init` from a source of another width: two real models of the default size through the engine and the handler of the sequence
    def built(**model_args):
        return tm.create_vq_tokenizer(variant="small", input_shape=(side, side, 3), patch_size=patch, num_embeddings=codes, **model_args)

    plan = init_from.plan_transfer(built(), built(hidden_channels=64), spec, name=f"{tm.FAMILY}/small")
    with pytest.raises(ConfigError) as caught:
        with run_sequence._bad_value_is_a_config_error(tmp_path / "out", "the init transfer"):
            init_from.check(plan)
    assert is_the_message(cells[10], str(caught.value)), f"row 11: the table quotes {cells[10]}, the code raises {caught.value}"
    # the table says how to read an abbreviated cell, so a reader does not take a piece for the whole message
    assert "a message quoted with `<run>`, `<out>` or `...` is abbreviated there, and each piece between those marks is the code's own, in that order" in " ".join(README.read_text().split())


def test_the_measured_numbers_the_documents_repeat_are_the_same_digits():
    """Hygiene finding 13 (Step 12): the numbers of the probes that chose the linear decoder and the per-cell floor stand in two to four places (the section of the README, the
    README of the model, the comments of ``tokenizer_data.py``) and only 0.5339 had a reader. Read by pattern, no run: where a quantity is stated twice, its digits are equal."""
    def text_of(path, comments=False):
        text = path.read_text()
        return " ".join((re.sub(r"(?m)^\s*#:? ?", "", text) if comments else text).split())

    docs = {"README": text_of(README), "model README": text_of(VQ_README), "source": text_of(TOKENIZER_DATA, comments=True)}
    num = r"(\d+\.\d+)"
    quantities = {
        "the share of the pixels below 0 and above 1": [("README", rf"{num} percent of their pixels lie below 0 and {num} percent above 1", (0, 1)),
                                                        ("model README", rf"{num} percent of the pixels lie below 0 and {num} percent above 1", (0, 1))],
        "the best output confined to [0, 1]": [("README", rf"scores {num} on the selection split", (0,)), ("model README", rf"scores a mean squared error of {num}", (0,))],
        "the sigmoid decoder's loss at 750 steps": [("README", rf"{num} and {num} \(two seeds\) against {num} and {num}", (2, 3)),
                                                    ("model README", rf"the sigmoid decoder's `val_reconstruction_loss` was {num} and {num}", (0, 1))],
        "the linear decoder's loss at 750 steps": [("README", rf"{num} and {num} \(two seeds\) against", (0, 1)), ("model README", rf"the linear one's {num} and {num}", (0, 1)),
                                                   ("source", rf"750-step arms reached {num} and {num}", (0, 1))],
        "the pooled perplexity of the two linear arms": [("README", rf"{num} and {num} per cell \({num} and {num} pooled", (2, 3)),
                                                         ("model README", rf"against {num} and {num} \(linear\)", (0, 1)), ("source", rf"of their pooled {num} and {num}\)", (0, 1))],
        "the per-cell perplexity of the two linear arms": [("README", rf"were worth {num} and {num} per cell", (0, 1)), ("source", rf"selection images\) {num} and {num} per cell", (0, 1))],
        "the lowest per-cell value the floor is cut from": [("README", rf"mean of 2 and {num}:", (0,)), ("source", rf"and {num} \(the lowest trained value\)", (0,))],
    }
    for quantity, places in quantities.items():
        seen = {}
        for document, pattern, groups in places:
            found = re.findall(pattern, docs[document])
            assert len(found) == 1, f"{quantity}: {document} states it {len(found)} times by {pattern!r}"
            seen[document] = tuple(found[0][i] if isinstance(found[0], tuple) else found[0] for i in groups)
        assert len(set(seen.values())) == 1, f"{quantity}: the documents state different digits: {seen}"


def test_the_export_figures_no_test_repeats_are_named_recorded_measurements():
    """Hygiene finding 13 (Step 12): 720,896 positions, 23,819 parameters and 12.4 percent are the export probe's own figures; no test re-measures them, so the README says so
    in the sentence that states them (a figure with no reader is a recorded measurement, and is named one)."""
    text = " ".join(README.read_text().split())
    start = text.index("* **The record, `predict` and `export`.**", text.index("**`beit_tokenizer`** (plan fa94f59c"))
    bullet = text[start:text.index("* **`init`**", start)]
    sentence = "720,896 positions, 23,819 parameters and 12.4 percent are recorded measurements of this phase's export probe, which no test repeats"
    before, found, _ = bullet.partition(sentence)
    assert found and all(figure in before for figure in ("720,896 positions", "23,819", "12.4 percent")), "the figures are stated, then named recorded measurements"


def test_every_test_the_agents_bullet_of_the_tokenizer_task_names_exists():
    """Plan fa94f59c step 9: a rule line of the ``beit_tokenizer`` bullet names the test that holds it; a name that is no test is a rule held by nothing."""
    agents = AGENTS.read_text()
    start = agents.index("- **The `beit_tokenizer` task**")
    bullet = agents[start:agents.index("\n- **", start + 10)]
    names = re.findall(r"`(test_[a-z0-9_]+)(\.py)?`", bullet)
    assert len(names) >= 8, names
    directory = ROOT / "tests" / "test_train" / "test_api"
    source = defined_tests()
    missing = [name + suffix for name, suffix in names if not ((directory / f"{name}.py").is_file() if suffix else f"def {name}(" in source)]
    assert missing == [], f"the AGENTS bullet names tests that do not exist: {missing}"


def test_the_lost_tokenizers_sha256_is_quoted_in_full_once():
    """Plan fa94f59c step 9: the 4h gate's tokenizer file is lost; its sha256 survives in two runs' ``task.json`` (read from them by ``scratch_docs/derive_docs_numbers.py``)."""
    text = README.read_text()
    shas = re.findall(r"cdb578c7[0-9a-f]*", text)
    assert shas == ["cdb578c78471b4198b502401c9c0db153f613df8816dc96b20ab66bb1bdfd3f8"], "the full 64 digits, once, and no shortened form"
    assert "lost" in text[text.index(shas[0]) - 600:text.index(shas[0]) + 600] and "are NOT reproduced" in text


def test_the_repo_map_and_the_roadmaps_count_the_tasks_the_registry_has(world):
    """Plan fa94f59c step 9: three documents no other test reads say how many tasks there are and which is the newest (a stale count stayed in the roadmap)."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    names = [spec.name for spec in BUILTIN_TASKS]
    newest = names[-1]
    ordinal = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")[len(names) - 1]
    assert newest == "dino" and names[-2] == "beit_tokenizer" and len(names) == len(registry.TASKS)
    repo_map = " ".join((ROOT / "REPO_MAP.md").read_text().split())
    scripts = " ".join((ROOT / "research" / "roadmap" / "2026_train_scripts.md").read_text().split())
    init = " ".join((ROOT / "research" / "roadmap" / "20261117_train_api_init_from.md").read_text().split())
    assert "The ninth task, `beit_tokenizer`, trains the BEiT tokenizer that a `beit_mim` run reads" in repo_map
    assert f"The {ordinal} task, `{newest}`, trains a DINOv1 self-supervised ViT" in repo_map
    assert f"`python -m train <task> help` of all {words[len(names)]};" in scripts and f"`beit_tokenizer` (Phase 4j) and `{newest}` (Phase 4k), in that order" in scripts
    assert f"`beit_tokenizer` is the ninth task, one family `{registry.MODELS['vq_vae_rotation'].family}`" in scripts and "Phase 4j: the BEiT tokenizer. Code done, GPU gate run**" in scripts
    assert f"`{newest}` is the {ordinal} task, one family `{registry.MODELS['dino_v1'].family}`" in scripts and "Phase 4k: DINO." in scripts
    assert ("Phase 4j (plan fa94f59c, branch `train-api/phase-4j`: code done, its GPU gate run, numbers in `src/train/README.md` under "
            "\"Measured: the tokenizer gate\")") in init and "`beit_tokenizer` in Phase 4j" in init and f"`{newest}` in Phase 4k" in init
    # Review pass 2, N4: both roadmaps state what the code does: the bar's two floors, and the two entries of the family row that lift the size refusal of `init`
    from train.common import pretrain_run, tokenizer_data as td

    assert (f"a codebook perplexity over every selection position of at least {pretrain_run.MIN_TOKENIZER_PERPLEXITY:g} pooled and at least "
            f"{td.MIN_PER_CELL_PERPLEXITY:g} per cell") in scripts
    assert "lifting the refusal is two entries of the family row (the `fixed` tuple and the `policy` entry of `train/api/tokenizer_models.py`) and its census test" in init


def test_no_document_says_the_tokenizer_gate_has_not_run():
    """Plan fa94f59c step 14: the gate ran, so the five documents that said it had not (the README twice and its block, the AGENTS bullet, both roadmaps) say it has, and
    the block that ends the section is the gate's own, not the placeholder that stood there before the night."""
    texts = {"README": " ".join(README.read_text().split()), "AGENTS": " ".join(AGENTS.read_text().split()),
             "scripts": " ".join((ROOT / "research" / "roadmap" / "2026_train_scripts.md").read_text().split()),
             "init": " ".join((ROOT / "research" / "roadmap" / "20261117_train_api_init_from.md").read_text().split()),
             "repo map": " ".join((ROOT / "REPO_MAP.md").read_text().split())}
    stale = ["its GPU gate has not run", "GPU gate NOT run", "gate block is empty", "is empty until it has", "has not been run, so this block holds no measurement",
             "Measured: the tokenizer gate.** Not run", "Not done: the GPU gate", "the block that ends this section is empty"]
    found = [(name, phrase) for name, text in texts.items() for phrase in stale if phrase in text]
    assert found == [], f"a document still says the tokenizer gate has not run: {found}"
    readme = texts["README"]
    block = readme[readme.index("<!-- tokenizer-gate:begin -->"):readme.index("<!-- tokenizer-gate:end -->")]
    assert block.startswith("<!-- tokenizer-gate:begin --> **Measured: the tokenizer gate** (plan fa94f59c:") and "Run on the code HEAD" in block
    assert "`beit_tokenizer`, the ninth task, trains the tokenizer that `beit_mim` reads: it is in the code and its GPU gate has run (Phase 4j" in readme
    assert "and **its GPU gate has run** (its numbers are in the block that ends this section)" in readme[readme.index("**`beit_tokenizer`** (plan fa94f59c"):]
    agents = texts["AGENTS"]
    assert "[`README.md`](README.md), which holds the numbers of its GPU gate in its own block." in agents[agents.index("**The `beit_tokenizer` task**"):]


# ---------------------------------------------------------------------
# The dino documents (plan e3381378, step 9)
# ---------------------------------------------------------------------

DINO_HEAD = "**`dino`** (plan e3381378, Phase 4k)"
DINO_GATE_BEGIN, DINO_GATE_END = "<!-- dino-gate:begin -->", "<!-- dino-gate:end -->"


def dino_section() -> str:
    """The README's ``dino`` section, whitespace-normalised, from its heading to the end of its gate block."""
    text = " ".join(README.read_text().split())
    start = text.index(DINO_HEAD)
    return text[start:text.index(DINO_GATE_END, start)]


def dino_refusal_cases():
    """``[live message]``: what the code raises for each row of the README's refusal table of the ``dino`` section, in the table's order."""
    from train.api import dino_args as da, dino_models as dm, init_from
    from train.api.errors import ConfigError
    from train.api.tasks import dino as dino_task

    def raised(call):
        with pytest.raises(ConfigError) as caught:
            call()
        return str(caught.value)

    sources = init_from._declared_sources("image_classification", "resnet", registry.MODELS["resnet"].transfer)
    return [
        raised(lambda: registry.parse_model_name("vit/vit_tiny", task="dino")),
        raised(lambda: da.check_geometry(3, 96)),
        dm.problems_of({"student_temp": 0.2})[0],
        dm.problems_of({"teacher_temp": 0.04, "teacher_temp_final": 0.03})[0],
        dm.warmup_problems(60, 60)[0],
        dm.precision_problems("mixed_float16")[0],
        dm.schedule_problems("constant")[0],
        dm.backup_problems(100)[0],
        f"Invalid value for 'init': {dino_task._INIT_REFUSED}",
        f"Invalid value for 'init': dino_v1 (dino) cannot seed resnet (image_classification); resnet takes weights from {sources}",
    ]


def test_the_dino_section_states_the_numbers_of_the_code(world):
    """Plan e3381378 step 9: the variants, the model keys, the recipe, the geometry, the margin and where it comes from, the probe, the record file and the four
    refusals of the ``dino`` section are the code's, derived here and never typed."""
    from train.api import dino_args as da, dino_models as dm, recipes
    from train.api.tasks import dino as dino_task
    from train.dino.knn_eval import DEFAULT_KS

    section = dino_section()
    tiny, small = dm.VARIANTS["tiny"], dm.VARIANTS["small"]
    size, patch = da.GEOMETRY["imagenette"]
    crops, out = dm.RECIPE["data"]["n_local_crops"], dm.RECIPE["model_args"]["dino_out_dim"]
    model = recipes.build_model(registry.MODELS[da.FAMILY], "tiny", None, (size, size, 3), {"patch_size": patch, "n_local_crops": crops, "dino_out_dim": out})
    assert (f"{model.count_params():,} parameters in {len(model.weights)} weights, the student's {len(model.student.weights)} and the teacher's "
            f"{len(model.teacher.weights)}") in section
    assert (f"`tiny` ({tiny['embed_dim']} wide, {tiny['depth']} blocks, {tiny['num_heads']} heads) and `small` "
            f"({small['embed_dim']} wide, {small['depth']} blocks, {small['num_heads']} heads)") in section
    offered = ", ".join(f"`{key}`" for key in dm.ALLOWED[:-1]) + f", `{dm.ALLOWED[-1]}`"
    assert f"Four `model_args` are offered ({offered}), {len(dm.SUPPLIED)} are given by the task and {len(dm.REFUSED)} more are refused" in section and len(dm.ALLOWED) == 4
    optim, values = dm.RECIPE["optim_args"], dm.RECIPE["model_args"]
    assert (f"AdamW at {optim['lr']:.0e}".replace("e-0", "e-") + f", weight decay {optim['weight_decay']:g}, {optim['warmup_epochs']} warmup epochs, cosine decay over the "
            f"whole run to {optim['min_lr_ratio']:g} of the peak, a global gradient clip of {optim['global_clipnorm']:.1f}, batch {dm.BATCH_SIZE}, {dm.EPOCHS} epochs, "
            f"patience {dm.PATIENCE}, a {out:,}-wide head, {crops} local crops, the teacher temperature held at {values['teacher_temp']:g} and the teacher's EMA "
            f"warmed up over {values['ema_warmup_epochs']:g} epoch") in section
    assert " and ".join(f"`{name}` is {side} px at patch {cell} (grid {side // cell} x {side // cell})" for name, (side, cell) in da.GEOMETRY.items()) in section
    # the bar: both margins, and the two quantities each is the larger of (derived by `margin_of`, so the sentence cannot say another)
    for name in ("imagenette", "cifar10"):
        sd, null = dino_task.CONTROL_SD[name], dino_task.NULL_P99[name]
        assert dino_task.MARGIN[name] == dino_task.margin_of(sd, null)
        assert f"{dino_task.SD_MULTIPLE * 2 ** 0.5 * sd:.4f}" in section and f"{sd:.4f}" in section and f"{null:.4f}" in section, name
    assert f"{dino_task.MARGIN['imagenette']:.3f} for Imagenette and {dino_task.MARGIN['cifar10']:.3f} for cifar10" in section
    assert "ceil_to_0.005" not in section and "`margin_of`" in section
    # the probe: its estimator, its keys, its two values of k, the file the centre rides in
    assert (f"{dm.FIXED['knn_bank_batches']} bank batches and up to {dm.FIXED['knn_query_batches']} query batches" in section
            and f"k = {' and '.join(str(k) for k in DEFAULT_KS)}" in section)
    assert all(f"`{key}`" in section for key in dino_task.PROBE_KEYS) and len(dino_task.PROBE_KEYS) == 7
    assert f"`{dino_task.CENTRE_FILE}`" in section and "`best_model.keras`" in section and "`final_model.keras`" in section
    # the file `predict` and `export` read is the one the verdict judged (the loader's own path), and the section names that file as the one they derive the feature model from
    from train.common.callbacks import best_checkpoint_path

    assert Path(best_checkpoint_path("/run")).name == "best_model.keras"
    assert "**`best_model.keras` is the whole training model.**" in section and "`predict` and `export` DERIVE the feature model from it at every load" in section
    assert "`final_model.keras` is the last epoch's" in section and "from `final_model.keras`" not in section
    assert "`checkpoint_args.backup_every_steps` is refused by name" in section and "`resume_stream_exact`" in section
    # the table of refusals: every cell is the message the code raises (equal, or abbreviated at `<run>`, `<out>` or `...`)
    rows = dino_refusal_rows()
    live = dino_refusal_cases()
    assert len(rows) == len(live), [case for case, _ in rows]
    for number, ((case, cell), message) in enumerate(zip(rows, live), 1):
        # a message that holds a backtick is quoted in a double-backtick span; both sides are compared with the backtick read as an apostrophe
        quoted = cell[2:-2].strip() if cell.startswith("``") else cell[1:-1]
        assert is_the_message("`" + quoted.replace("`", "'") + "`", message.replace("`", "'")), f"row {number} ({case}): the table quotes {cell}, the code raises {message}"
    assert "a message quoted with `<run>`, `<out>` or `...` is abbreviated there" in section
    # fix round 11.1 (review-iter-1): what the corrections say, each sentence read against the code or the test that holds it
    from ._dino_world import CLAIM_LEAVES

    assert dm.warmup_problems(1.0, 1) and dm.warmup_problems(1.0, 1, smoke=True) == [], "the refusal and its smoke exemption are the code's"
    assert ("`model_args.ema_warmup_epochs` that is not below the run's `epochs`" in section and "except under `debug_args.smoke`" in section
            and "for a fresh run and a resume alike" in section)
    assert "the bank is the first fit rows in the split's seeded order and not a shuffle" in section and "does not reach it" in section
    assert dino_task.non_finite_terms(0.1, float("nan")) == ["epoch zero nan"] and dino_task.non_finite_terms(float("inf"), 0.1) == ["baseline inf"]
    assert ("An `epoch zero` term that is not a finite number (it is read from `fit_state.json`" in section and "a `baseline` term that is not (it is computed from the real labels" in section
            and "the report is refused before any scoring" in section and "a miss of part 1" in section and "damaged `fit_state.json`) or a `baseline`" not in section)
    # fix round 11.2: the warmup that rounds to every step, the smoke exemption in both documents and the Imagenette flag AGENTS says is refused with `augment`
    assert dm.warmup_steps_problems(59.999, 60, 266) and dm.warmup_steps_problems(59.999, 60, 266, smoke=True) == [] and "is 15,960 warmup steps of 15,960" in section
    agents = " ".join(AGENTS.read_text().split())
    assert "one that rounds to every step of the run after the data loads and before the model is built" in agents and "a `debug_args.smoke` run exempt from both" in agents
    from dl_techniques.datasets.vision.imagenette import ImagenetteConfig, imagenette_examples

    with pytest.raises(ValueError, match="shuffle=False needs augment=False"):
        imagenette_examples("train", ImagenetteConfig(shuffle=False, augment=True))
    assert "refused with `augment`; `ImagenetteConfig.shuffle` likewise" in agents
    assert "the 60-draw null was not repeated on it" in section and "measured on the probe bank's first construction" in section
    assert "A diverged student is caught by its loss, not by the probe" in section and "ends the run as `diverged`, exit 1, with no bar applied" in section
    assert "The centre file carries no epoch stamp" in section and "A resume that only reports (training had finished) restores the centre from the file before it scores the test split" in section
    assert ", ".join(f"`{name}`" for name in CLAIM_LEAVES[:-1]) + f" and `{CLAIM_LEAVES[-1]}` only" in section and "`vit/vit_small` after `dino_v1/tiny`" in section


def dino_refusal_rows():
    """``[(case, message cell)]``: the rows of the README's refusal table of the ``dino`` section, in its order."""
    text = README.read_text()
    start = text.index("* **What is refused, and the message.**", text.index(DINO_HEAD))
    block = text[start:text.index("* **Limits.**", start)]
    rows = [line.strip()[2:-2].split(" | ", 1) for line in block.splitlines() if line.startswith("| ") and not line.startswith(("| Case", "|---"))]
    return [(case.strip(), message.strip()) for case, message in rows]


def test_the_dino_record_the_init_pair_and_the_pair_counts_are_the_codes(world):
    """Plan e3381378 step 9: the tenth record kind's keys and the keys a resume compares (README and AGENTS); the counts of the ``dino`` into ``vit`` pair at the geometry
    the section documents, built and planned here; the number of declared cross-task pairs; and the refusal sentence that now names the second source of ``vit``."""
    from train.api import dino_args as da, dino_models as dm, init_from, recipes

    text, agents = dino_section(), " ".join(AGENTS.read_text().split())
    assert list(predictor.RECORD_KEYS).index("dino") == 9, "the tenth kind"
    keys = [key for key in predictor.RECORD_KEYS["dino"] if key not in ("schema_version", "task", "model", "family", "variant")]
    own = ", ".join(f"`{key}`" for key in keys[:-1]) + f" and `{keys[-1]}`"
    assert f"writes `task.json` (schema version 2, the tenth record kind) before its first epoch: {own} (`{predictor.DINO_OUTPUTS[0]}`)" in text
    compared = predictor.DATA_KEYS["dino"]
    assert f"a resume compares the {('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight')[len(compared)]} from `{compared[0]}` to `{compared[-1]}`" in text
    assert "a dino record (the tenth kind" in agents and "`" + "`, `".join(compared[:-1]) + f"` and `{compared[-1]}`" in agents[agents.index("a dino record (the tenth kind"):]
    # the pair, at the documented geometry
    size, patch = da.GEOMETRY["imagenette"]
    dino = recipes.build_model(registry.MODELS[da.FAMILY], "tiny", None, (size, size, 3), {"patch_size": patch, "n_local_crops": 4, "dino_out_dim": dm.RECIPE["model_args"]["dino_out_dim"]})
    vit = recipes.build_model(registry.MODELS["vit"], "vit_tiny", 10, (size, size, 3), {"patch_size": patch})
    if not vit.built:
        vit.build((None, size, size, 3))
    relation = recipes.VIT_TRANSFER.sources[("dino", "dino_v1")]
    plan = init_from.plan_transfer(init_from.source_view(dino, relation), vit, recipes.VIT_TRANSFER, relation=relation, name="vit/vit_tiny")
    init_from.check(plan)
    assert (relation.view, relation.drop, relation.rename) == ("student", ("dino_projection_head",), ((r"transformer_block_(\d+)", r"transformer_layer_\1"),))
    assert (f"`vit/vit_tiny` after `dino_v1/tiny` at {size} px and patch {patch} copies {len(plan.copied)}, resets {len(plan.reset)} (the classification head) and drops "
            f"{len(plan.dropped)} (the projection head)") in text
    assert "`transformer_block_N` renamed `transformer_layer_N`" in text
    readme = " ".join(README.read_text().split())
    assert (f"`dino_v1/tiny` into `vit/vit_tiny` at {size} px and patch {patch} copies {len(plan.copied)}, resets the {len(plan.reset)} weights of `classification_head` and drops "
            f"the {len(plan.dropped)} of `dino_projection_head`") in readme, "the init section's count of the pair (the count of declared pairs is the classification test's)"
    declared = init_from._declared_sources("image_classification", "vit", registry.MODELS["vit"].transfer)
    assert declared == "vit (image_classification) or dino_v1 (dino)"
    assert (f"`Invalid value for 'init': gpt2_lm (language_model_pretraining) cannot seed vit (image_classification); vit takes weights from {declared}`") in readme
    # AGENTS: both fields of `CrossTask` are named where the row's fields are described
    spec_bullet = agents[agents.index("**`TransferSpec` per family.**"):][:2600]
    assert all(f"`{field.name}`" in spec_bullet for field in dataclasses.fields(registry.CrossTask)), [f.name for f in dataclasses.fields(registry.CrossTask)]


def test_the_dino_gate_block_the_notices_name_exists_once_and_holds_the_gates_numbers(world):
    """Plan e3381378 steps 9 and 14 (the Step 5 note): the recipe notices of ``dino_models`` point at a README block; it exists, once, with the heading the notice names, and
    since the night has run it holds the gate's own numbers and the HEAD they were measured on (the numbers themselves are read by the harness, as in the 4j plan)."""
    from train.api import dino_models as dm

    readme = README.read_text()
    assert readme.count(DINO_GATE_BEGIN) == 1 and readme.count(DINO_GATE_END) == 1 and readme.index(DINO_GATE_BEGIN) < readme.index(DINO_GATE_END)
    assert "Measured: the dino gate" in dm.GATE_BLOCK and "dino-gate" in dm.GATE_BLOCK and "src/train/README.md" in dm.GATE_BLOCK
    block = readme.split(DINO_GATE_BEGIN)[1].split(DINO_GATE_END)[0]
    assert "**Measured: the dino gate" in block, "the heading the notice names"
    assert "Not run." not in block and re.search(r"\d", block) and "Run on the code HEAD" in block, "the block is the gate's own: its numbers, the HEAD it ran on"
    assert readme.index(DINO_HEAD) < readme.index(DINO_GATE_BEGIN) < readme.index("**Keys**: `key=value` tokens"), "the block ends the section"
    section = dino_section()
    assert "**its GPU gate has run**" in section and "its GPU gate is not yet run" not in section


def test_every_test_the_agents_bullet_of_the_dino_task_names_exists():
    """Plan e3381378 step 9: a rule line of the ``dino`` bullet names the test that holds it; a name that is no test is a rule held by nothing."""
    agents = AGENTS.read_text()
    start = agents.index("- **The `dino` task**")
    bullet = agents[start:agents.index("\n- **", start + 10)]
    names = re.findall(r"`(test_[a-z0-9_]+)(\.py)?`", bullet)
    assert len(names) >= 8, names
    directory = ROOT / "tests" / "test_train" / "test_api"
    source = defined_tests()
    missing = [name + suffix for name, suffix in names if not ((directory / f"{name}.py").is_file() if suffix else f"def {name}(" in source)]
    assert missing == [], f"the AGENTS bullet names tests that do not exist: {missing}"


def test_the_dino_status_lines_of_the_repo_map_and_the_roadmaps_say_the_code_is_done_and_the_gate_has_run(world):
    """Plan e3381378 steps 9 and 14: the three documents no other test reads state the tenth task and its status: the code is done and its GPU gate has run."""
    scripts = " ".join((ROOT / "research" / "roadmap" / "2026_train_scripts.md").read_text().split())
    init = " ".join((ROOT / "research" / "roadmap" / "20261117_train_api_init_from.md").read_text().split())
    repo_map = " ".join((ROOT / "REPO_MAP.md").read_text().split())
    assert "**Phase 4k: DINO. Code done, GPU gate run**" in scripts and "its section of `src/train/README.md` holds the numbers of the Phase 4k GPU gate" in repo_map
    assert ("Phase 4k (plan e3381378, branch `train-api/phase-4k`: code done, its GPU gate has run, numbers in `src/train/README.md` under \"Measured: the dino gate\") adds the DINO "
            "student as a source of a `vit` classifier") in init
    assert "**Phase 4k (plan e3381378; D-NNN below are that plan's).**" in init and "A dino run is no `init` target" in init
    assert "src/train/README.md`, the `dino` section" in scripts and "(written after the gate)" not in scripts
    readme, agents = " ".join(README.read_text().split()), " ".join(AGENTS.read_text().split())
    sites = {"README init paragraph": readme[readme.index("`dino`, the tenth task, trains"):readme.index("`load_model`, `export`, `list_models`")], "scripts": scripts, "init": init,
             "AGENTS bullet": agents[agents.index("- **The `dino` task**"):agents.index("- **Per-task tables")], "repo map": repo_map[repo_map.index("The tenth task, `dino`"):]}
    assert [name for name, text in sites.items() if "its GPU gate has run" not in text] == [], "every status sentence uses the one phrase (D-115)"


def test_no_document_says_the_dino_gate_has_not_run():
    """Plan e3381378 step 14: the gate ran, so the documents that said it had not (the README's dino section and its block, the AGENTS bullet, both roadmaps, the repo map) say it has."""
    texts = {"README": " ".join(README.read_text().split()), "AGENTS": " ".join(AGENTS.read_text().split()),
             "scripts": " ".join((ROOT / "research" / "roadmap" / "2026_train_scripts.md").read_text().split()),
             "init": " ".join((ROOT / "research" / "roadmap" / "20261117_train_api_init_from.md").read_text().split()),
             "repo map": " ".join((ROOT / "REPO_MAP.md").read_text().split())}
    stale = ["its GPU gate is not yet run", "GPU gate not yet run", "**Measured: the dino gate.** Not run."]
    found = [(name, phrase) for name, text in texts.items() for phrase in stale if phrase in text]
    assert found == [], f"a document still says the dino gate has not run: {found}"
    readme = texts["README"]
    block = readme[readme.index(DINO_GATE_BEGIN):readme.index(DINO_GATE_END)]
    assert block.startswith(f"{DINO_GATE_BEGIN} **Measured: the dino gate** (plan e3381378:") and "Run on the code HEAD" in block
    assert "**its GPU gate has run**" in readme[readme.index(DINO_HEAD):] and "holds the numbers of its GPU gate in its own block" in texts["AGENTS"][texts["AGENTS"].index("- **The `dino` task**"):]


CITED = re.compile(r"`((?:tests|src)/[A-Za-z0-9_./\-]+\.(?:py|md|json|sh|yaml|yml))`")


@pytest.mark.parametrize("document", [README, AGENTS], ids=["README.md", "AGENTS.md"])
def test_every_source_and_test_path_the_document_cites_exists(document):
    cited = sorted(set(CITED.findall(document.read_text())))
    assert cited, "vacuity guard: the document cites paths"
    missing = [path for path in cited if not (ROOT / path).exists()]
    assert not missing, f"{document.name} cites paths that do not exist: {missing}"

"""A third task is one row per table: its record is read, loaded and predicted by its own rows.

``train.api.predictor`` held two-task branches (``if task == CLASSIFICATION ... else <segmentation>``,
identity and statistics checks run for every task, ``load_model`` and ``input_shape_of`` choosing
segmentation by name) and ``cli._predict`` one more. Plan-2026-11-10T095307-d483ffab D-006 replaced
them with tables keyed by task (``predictor.TASK_TABLES`` and ``cli.PREDICT_WRITERS``). A task with
no class count (the denoising task of that plan) has no ``class_names``, ``normalize``, ``mean`` or
``std``; the identity and statistics checks run only for a task whose ``RECORD_KEYS`` hold those keys.

This file registers a FAKE third task (``fake_denoising``: a record of ``frame_size`` and ``channels``
and a ``domain`` text, no class names, no statistics) in every table, in the test only, and shows:

* every table of ``TASK_TABLES`` is walked by the fake (a new table without a fake row fails here);
* its record is checked by its own rows and by nothing of another task (no ``KeyError`` on
  ``class_names``), and each of its checks fires;
* ``load_model`` and ``predict`` reach the fake's rows: output check, constructor, input shape and
  key, writer;
* removing the fake's row from any ONE table refuses the record, naming the tasks that table knows,
  and the same for the ``predict`` writer table;
* the two real tasks have a row in every table (the tables agree on their keys);
* a task whose data has no classes (D-005): ``DataFacts`` and ``AutoContext`` take ``num_classes=None``
  and the run summary has no ``num_classes`` key, while a classifier's still has it.

Everything writes under ``tmp_path``; the module-level tables are patched with ``monkeypatch`` and
restored.
"""

import json
import re
import types
from pathlib import Path

import keras
import pytest

from train.api import cli, predictor, recipes, registry
from train.api.args import AutoContext, MetricArgs, OptimArgs
from train.api.data_args import DataFacts
from train.api.errors import ConfigError
from train.api.predictor import (
    CLASSIFICATION,
    DENOISING,
    PREDICTORS,
    SCHEMA_VERSION,
    SEGMENTATION,
    TASK_TABLES,
    check_record,
    input_shape_of,
    load_model,
    record_differences,
)

FAKE = "fake_denoising"
FAMILY = "fake_family"
FAKE_KEYS = ("schema_version", "task", "model", "family", "variant", "frame_size", "channels", "domain")

#: Rows the fake's checks were called with, in order: ``(table name, record task)``.
CALLS = []


class FakePredictor:
    """What the fake task's ``PREDICTORS`` row returns."""

    def __init__(self, model, record):
        self.model, self.record = model, record


def fake_geometry(record):
    CALLS.append(("GEOMETRY_CHECKS", record["task"]))
    size, channels = record["frame_size"], record["channels"]
    if not (isinstance(size, list) and len(size) == 2 and all(isinstance(s, int) and s > 0 for s in size)):
        return [f"'frame_size' must be [height, width], got {size!r}"], None
    return [], channels


def fake_tail(record):
    CALLS.append(("TAIL_CHECKS", record["task"]))
    return [] if record["domain"] == "[0,1]" else [f"'domain' must be '[0,1]', got {record['domain']!r}"]


def fake_input_shape(record):
    return (*record["frame_size"], record["channels"])


def fake_output_check(output, record):
    CALLS.append(("OUTPUT_CHECKS", record["task"]))
    if tuple(output.shape)[-1] != record["channels"]:
        raise ConfigError(f"The model outputs {tuple(output.shape)[1:]} but the record names {record['channels']} channels")


#: The fake's row in each table of ``TASK_TABLES``. A table added to ``TASK_TABLES`` without a row
#: here fails ``test_the_fake_has_a_row_for_every_table``.
FAKE_ROWS = {
    "RECORD_KEYS": FAKE_KEYS,
    "DATA_KEYS": ("frame_size", "channels", "domain"),
    "GEOMETRY_CHECKS": fake_geometry,
    "TAIL_CHECKS": fake_tail,
    "INPUT_SHAPE_OF": fake_input_shape,
    "INPUT_KEY": "frame_size",
    "OUTPUT_CHECKS": fake_output_check,
    "PREDICTORS": FakePredictor,
    "EXPORT_RULES": predictor.ExportRule(
        shape_of=lambda record: (None, None, record["channels"]),
        check=lambda record, format, shape: CALLS.append(("EXPORT_RULES.check", record["task"], format, shape)),
        note=lambda record: "the fake export note",
    ),
}

# The family the fake task trains: a module-level table and a factory this module defines, named by
# ``module:attr`` as every spec does.
FAKE_VARIANTS = {"tiny": {"filters": 1}}


def build_fake(*, variant, num_classes, input_shape, **kwargs):
    return keras.Sequential([keras.Input(input_shape), keras.layers.Conv2D(1, 3, padding="same")])


def fake_record(**changes):
    record = {
        "schema_version": SCHEMA_VERSION, "task": FAKE, "model": f"{FAMILY}/tiny", "family": FAMILY,
        "variant": "tiny", "frame_size": [8, 8], "channels": 1, "domain": "[0,1]",
    }
    record.update(changes)
    return record


def write_fake_predict(model_path, record, files, out):
    answer = {"predictor": type(load_model(model_path)).__name__, "files": [str(file) for file in files]}
    Path(out).write_text(json.dumps(answer))
    return 0


@pytest.fixture()
def fake_task(monkeypatch):
    """The fake registered in every table, its family and its ``predict`` writer; all restored."""
    CALLS.clear()
    for name, table in TASK_TABLES.items():
        monkeypatch.setitem(table, FAKE, FAKE_ROWS[name])
    monkeypatch.setattr(registry, "MODELS", {})
    recipes.register_builtin_models()
    registry.register_model(
        registry.ModelSpec(
            family=FAMILY,
            factory=f"{__name__}:build_fake",
            variants_attr=f"{__name__}:FAKE_VARIANTS",
            tasks=(FAKE,),
        )
    )
    monkeypatch.setitem(cli.PREDICT_WRITERS, FAKE, write_fake_predict)


@pytest.fixture()
def run_dir(fake_task, tmp_path):
    """A run directory holding a real saved model and the fake's ``task.json``."""
    directory = tmp_path / "run"
    directory.mkdir()
    build_fake(variant="tiny", num_classes=None, input_shape=(8, 8, 1)).save(directory / "best_model.keras")
    (directory / "task.json").write_text(json.dumps(fake_record()))
    return directory


# ---------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------


def test_the_fake_has_a_row_for_every_table():
    assert set(FAKE_ROWS) == set(TASK_TABLES)


REAL_TASKS = [CLASSIFICATION, SEGMENTATION, DENOISING]


@pytest.mark.parametrize("name", sorted(TASK_TABLES))
def test_the_real_tasks_have_a_row_in_every_table(name):
    assert sorted(TASK_TABLES[name]) == sorted(REAL_TASKS), name


def test_the_predict_writers_are_the_three_tasks_that_can_predict():
    # Denoising's writer arrived with its predictor (plan d483ffab step 12): one row per task that can be predicted.
    assert sorted(cli.PREDICT_WRITERS) == sorted(REAL_TASKS)


# ---------------------------------------------------------------------
# A third task's record is read by its own rows
# ---------------------------------------------------------------------


def test_a_third_task_record_with_no_class_names_and_no_statistics_is_valid(fake_task):
    """It has no ``class_names``, ``normalize``, ``mean`` or ``std``: those checks do not run for it."""
    record = fake_record()
    assert not {"class_names", "normalize", "mean", "std"} & set(record)
    assert predictor.record_problems(record) == []
    check_record(record, "task.json")
    assert CALLS == [("GEOMETRY_CHECKS", FAKE), ("TAIL_CHECKS", FAKE)] * 2


def test_a_third_task_record_is_checked_by_its_own_geometry_and_tail_rows(fake_task):
    with pytest.raises(ConfigError, match=r"'frame_size' must be \[height, width\], got \[8\]"):
        check_record(fake_record(frame_size=[8]), "task.json")
    with pytest.raises(ConfigError, match=r"'domain' must be '\[0,1\]', got 'unit'"):
        check_record(fake_record(domain="unit"), "task.json")


def test_a_third_task_record_still_has_its_identity_checked(fake_task):
    with pytest.raises(ConfigError, match=r"'model' is 'x/y' but 'family' and 'variant' give 'fake_family/tiny'"):
        check_record(fake_record(model="x/y"), "task.json")
    with pytest.raises(ConfigError, match="'family' must be a non-empty text"):
        check_record(fake_record(family=""), "task.json")


def test_a_third_task_record_does_not_take_another_tasks_keys(fake_task):
    with pytest.raises(ConfigError, match=r"unknown key\(s\) 'class_names'"):
        check_record(fake_record(class_names=["a"]), "task.json")
    with pytest.raises(ConfigError, match=r"missing key\(s\) 'domain'"):
        record = fake_record()
        del record["domain"]
        check_record(record, "task.json")


def test_a_real_task_record_is_not_checked_by_the_fakes_rows(fake_task):
    from tests.test_train.test_api.test_task_json_schema_v2 import classification, segmentation

    CALLS.clear()
    check_record(classification(), "task.json")
    check_record(segmentation(), "task.json")
    assert CALLS == []


def test_a_resume_compares_the_third_tasks_data_keys(fake_task):
    earlier = fake_record()
    assert record_differences(earlier, fake_record()) == []
    assert record_differences(earlier, fake_record(frame_size=[16, 16])) == ["frame_size: [8, 8] -> [16, 16]"]
    assert record_differences(earlier, fake_record(domain="x")) == ["domain: \"[0,1]\" -> \"x\""]


def test_the_input_shape_comes_from_the_tasks_row(fake_task):
    assert input_shape_of(fake_record(frame_size=[4, 6], channels=2)) == (4, 6, 2)
    assert input_shape_of({"task": CLASSIFICATION, "input_shape": [8, 8, 3]}) == (8, 8, 3)
    assert input_shape_of({"task": SEGMENTATION, "input_size": [5, 7], "channels": 3}) == (5, 7, 3)


# ---------------------------------------------------------------------
# load_model and predict reach the fake's rows
# ---------------------------------------------------------------------


def test_load_model_checks_the_output_and_builds_the_predictor_from_the_tasks_rows(run_dir):
    loaded = load_model(run_dir)
    assert type(loaded) is FakePredictor and loaded.record == fake_record()
    assert ("OUTPUT_CHECKS", FAKE) in CALLS


def test_load_model_names_the_tasks_own_input_key_when_the_model_does_not_take_it(run_dir):
    (run_dir / "task.json").write_text(json.dumps(fake_record(channels=3)))
    with pytest.raises(ConfigError, match=r"does not take the frame_size \[8, 8\] it names"):
        load_model(run_dir)


def test_load_model_refuses_an_output_the_tasks_check_does_not_accept(run_dir, monkeypatch):
    monkeypatch.setitem(TASK_TABLES["OUTPUT_CHECKS"], FAKE, lambda output, record: (_ for _ in ()).throw(
        ConfigError("the fake output check ran")))
    with pytest.raises(ConfigError, match="the fake output check ran"):
        load_model(run_dir)


def test_predict_writes_through_the_tasks_writer(run_dir, tmp_path, capsys):
    image = tmp_path / "a.png"
    image.write_bytes(b"not read by the fake writer")
    out = tmp_path / "answer.json"
    assert cli._predict([f"model={run_dir}", f"data={image}", f"out={out}"]) == 0
    assert json.loads(out.read_text()) == {"predictor": "FakePredictor", "files": [str(image)]}
    assert capsys.readouterr().out == ""


def test_the_real_tasks_keep_their_predictors():
    assert PREDICTORS[CLASSIFICATION].__name__ == "_classification_predictor"
    assert PREDICTORS[SEGMENTATION].__name__ == "_segmentation_predictor"
    assert PREDICTORS[DENOISING].__name__ == "_denoising_predictor"


def test_export_reaches_the_tasks_rule_for_its_shape_its_refusals_and_its_note(run_dir, tmp_path, monkeypatch):
    """``export`` is not a classification function with a segmentation branch: it asks the task's ``EXPORT_RULES`` row."""
    from train.api.exporting import export

    record = fake_record()
    model = keras.models.load_model(run_dir / "best_model.keras", compile=False)
    shape = predictor.EXPORT_RULES[FAKE].shape_of(record)
    assert shape == (None, None, 1)
    CALLS.clear()
    export(model=model, out=tmp_path / "m_saved_model", format="saved_model", input_shape=(8, 8, 1), record=record)
    assert [call for call in CALLS if call[0] == "EXPORT_RULES.check"] == [("EXPORT_RULES.check", FAKE, "saved_model", (8, 8, 1))]
    # A row that refuses stops the export before anything is written.
    refusing = predictor.ExportRule(fake_input_shape, lambda *_: (_ for _ in ()).throw(ConfigError("the fake refuses")),
                                    lambda record: "x")
    monkeypatch.setitem(predictor.EXPORT_RULES, FAKE, refusing)
    with pytest.raises(ConfigError, match="the fake refuses"):
        export(model=model, out=tmp_path / "n_saved_model", format="saved_model", input_shape=(8, 8, 1), record=record)
    assert not (tmp_path / "n_saved_model").exists() and not (tmp_path / "n_saved_model.task.json").exists()


# ---------------------------------------------------------------------
# Removing a row refuses, naming the tasks the table knows
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TASK_TABLES))
def test_a_task_missing_from_a_table_is_refused_naming_the_known_tasks(fake_task, monkeypatch, name):
    monkeypatch.delitem(TASK_TABLES[name], FAKE)
    knows = re.escape(", ".join(TASK_TABLES[name]))
    assert sorted(TASK_TABLES[name]) == sorted(REAL_TASKS)
    with pytest.raises(ConfigError, match=rf"'task' is '{FAKE}'; this reader knows {knows}$"):
        check_record(fake_record(), "task.json")


def test_a_task_missing_from_the_predict_writers_is_refused_naming_the_known_tasks(run_dir, tmp_path, monkeypatch):
    monkeypatch.delitem(cli.PREDICT_WRITERS, FAKE)
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    knows = re.escape(", ".join(cli.PREDICT_WRITERS))
    with pytest.raises(ConfigError, match=rf"'task' is '{FAKE}'; predict knows {knows}$"):
        cli._predict([f"model={run_dir}", f"data={image}", f"out={tmp_path / 'answer.json'}"])
    assert not (tmp_path / "answer.json").exists()


def test_a_task_in_no_table_is_refused_with_the_record_tasks(fake_task, monkeypatch):
    monkeypatch.delitem(TASK_TABLES["RECORD_KEYS"], FAKE)
    knows = re.escape(", ".join(TASK_TABLES["RECORD_KEYS"]))
    with pytest.raises(ConfigError, match=rf"this reader knows {knows}$"):
        check_record(fake_record(), "task.json")


# ---------------------------------------------------------------------
# A task with no classes (D-005)
# ---------------------------------------------------------------------


def summary_head_of(num_classes):
    """The keys ``TaskTrainer._summary_head`` writes for data holding ``num_classes``."""
    from train.api.trainer import TaskTrainer

    data = types.SimpleNamespace(
        test_ds=None, input_shape=(8, 8, 1), num_classes=num_classes, steps_per_epoch=1,
        selection_split="val", test_split=None, n_train=1, n_select=1, n_test=0,
    )
    config = types.SimpleNamespace(
        data=types.SimpleNamespace(dataset="folder", format="folder"), model="m/v", batch_size=1,
        seed=0, precision="float32", epochs=1, metric_args=MetricArgs(), optim_args=OptimArgs(),
    )
    me = types.SimpleNamespace(
        task="t", summary_fields=lambda *_: {}, task_metadata=lambda *_: {},
    )
    return TaskTrainer._summary_head(
        me, config, "f", "v", Path("run"), data, 1, {"loss": 1.0}, 1.0, None,
        {"gpu_name": "none", "tf_visible_devices": [], "cuda_visible_devices": ""},
    )


def test_facts_and_context_take_no_class_count():
    facts = DataFacts(dataset="folder", num_classes=None, image_size=None)
    context = AutoContext(
        family="f", variant="v", dataset=facts.dataset, num_classes=facts.num_classes, image_size=None, defaults={},
    )
    assert facts.num_classes is None and context.num_classes is None


def test_the_summary_has_no_num_classes_for_a_task_without_classes_and_keeps_it_for_a_classifier():
    assert "num_classes" not in summary_head_of(None)
    head = summary_head_of(10)
    assert head["num_classes"] == 10
    # the key keeps its place between input_shape and optim_args
    keys = list(head)
    assert keys[keys.index("input_shape") + 1] == "num_classes"
    assert keys[keys.index("num_classes") + 1] == "optim_args"


def test_a_missing_class_count_still_refuses_the_classifiers_success_threshold():
    """``None`` is allowed to exist; the classification ``MetricArgs`` still refuses it with its own sentence."""
    context = AutoContext(
        family="f", variant="v", dataset="d", num_classes=None, image_size=None, defaults={},
    )
    with pytest.raises(ConfigError, match=r"The data reports None classes; success_threshold needs at least 1"):
        MetricArgs().resolve_auto(context)

"""A run whose ``fit`` raises does not look like a success (invariant 6, D-009).

``keras.Model.fit`` is replaced by a function that raises, so everything before
``fit`` really runs (the claim, the request, the resolved config, the model, the
callbacks) and nothing after it does.

* **Subprocess.** ``python -m train image_classification ...`` with the patched
  ``fit`` exits non-zero but not 2 (2 is the code for a configuration the user can
  fix), the traceback names the failure, ``config.json`` is present, and neither
  ``final_model.keras`` nor ``best_model.keras`` exists. A summary that exists
  must not claim success.
* **In process.** The exception reaches the caller of the function, and also the
  caller of ``cli.main``: the CLI catches ``DLTrainError`` and nothing else.

The control for both is the real training run of ``test_the_cli_exit_codes.py``
(exit 0 and a ``final_model.keras``): the same driver without the patch.
"""

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import keras
import pytest

import train.api.recipes as recipes
from train.api import cli, registry
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import train_image_classification

from tests.classification_fakes import use_tiny_raw

REPO_ROOT = Path(__file__).resolve().parents[3]

MESSAGE = "fit was patched to fail"

#: The fake loader and the failing ``fit`` are installed in the child before
#: ``train.__main__`` runs, so the real task and ``sys.exit(main(...))`` run.
DRIVER = f'''
import runpy
import sys

sys.path.insert(0, {str(REPO_ROOT)!r})
import keras
import train.common.classification_data as classification_data
from tests.classification_fakes import tiny_raw

classification_data._load_raw = tiny_raw(n_train=40, n_test=8)


def failing_fit(self, *args, **kwargs):
    raise RuntimeError({MESSAGE!r})


keras.Model.fit = failing_fit
runpy.run_module("train", run_name="__main__")
'''

TOKENS = (
    "model=vit/vit_pico",
    "data.dataset=cifar10",
    "epochs=1",
    "batch_size=4",
    "analysis_args.model_analysis=false",
)


def run_failing(out: Path, cwd: Path, extra: Sequence[str] = ()):
    return subprocess.run(
        [sys.executable, "-c", DRIVER, "image_classification", f"out={out}", *TOKENS, *extra],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture(scope="module")
def child(tmp_path_factory):
    base = tmp_path_factory.mktemp("failing")
    (base / "cwd").mkdir()
    out = base / "run"
    return run_failing(out, base / "cwd"), out


def test_the_child_exits_non_zero_and_not_with_the_configuration_code(child):
    result, _ = child
    assert result.returncode != 0, result.stdout[-1000:]
    assert result.returncode != 2, result.stderr[-3000:]


def test_the_traceback_names_the_failure_so_it_was_the_patched_fit_that_ran(child):
    result, _ = child
    assert "Traceback" in result.stderr and MESSAGE in result.stderr
    # Not a configuration refusal reported as an `error:` line.
    assert not [line for line in result.stderr.splitlines() if line.startswith("error:")]


def test_the_resolved_config_is_written_and_no_model_is(child):
    _, out = child
    assert json.loads((out / "config.json").read_text())["model"] == "vit/vit_pico"
    assert not (out / "final_model.keras").exists()
    assert not (out / "best_model.keras").exists()


def test_a_summary_that_exists_does_not_claim_success(child):
    _, out = child
    summary = out / "results_summary.json"
    if summary.exists():
        assert json.loads(summary.read_text()).get("success") is not True


# ---------------------------------------------------------------------
# In process
# ---------------------------------------------------------------------


@pytest.fixture
def failing(monkeypatch):
    use_tiny_raw(monkeypatch, n_train=40, n_test=8)
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "TASKS", {})
    recipes.register_builtin_models()
    register_builtin_tasks()

    def failing_fit(self, *args, **kwargs):
        raise RuntimeError(MESSAGE)

    monkeypatch.setattr(keras.Model, "fit", failing_fit)


def test_the_exception_reaches_the_caller_of_the_function(failing, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match=MESSAGE):
        train_image_classification(
            out=str(out),
            model="vit/vit_pico",
            data={"dataset": "cifar10"},
            epochs=1,
            batch_size=4,
            analysis_args={"model_analysis": False},
        )
    assert (out / "config.json").is_file() and not (out / "final_model.keras").exists()


def test_the_exception_reaches_the_caller_of_the_cli_main(failing, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match=MESSAGE):
        cli.main(["image_classification", f"out={out}", *TOKENS])
    assert (out / "config.json").is_file() and not (out / "final_model.keras").exists()


# ---------------------------------------------------------------------
# The forecasting task: the same contract for a failure at each end of the run
# ---------------------------------------------------------------------
#
# Five places a forecasting run can fail, each stubbed where the real code reaches it: ``fit`` itself, the save of the best
# checkpoint (inside ``fit``, through the checkpoint callback), the save of the final model, the write of the history (both
# in ``finalize_last_epoch``, after the best weights were evaluated), and the evaluation of the test windows (the SECOND and
# only the second call of ``classification_run.evaluate_dict``: the first is the untrained model's, which must still run, and
# a failure that hit every later call too would survive a handler that swallowed one of them). None may exit 0 or 2.

FORECAST_MESSAGE = "forecasting failure was stubbed"
FORECAST_KINDS = ("fit", "best_save", "final_save", "history_write", "evaluation")

FORECAST_TOKENS = (
    "model=nbeats/tiny",
    "data.lookback=32",
    "data.horizon=8",
    "data.realizations=1",
    "data.series_length=600",
    'data.patterns=["ar_process","complex_seasonal","cubic_trend","arma_process"]',
    "data.n_select=16",
    "data.n_test=16",
    "data.steps_per_epoch=4",
    "epochs=1",
    "batch_size=8",
    "analysis_args.model_analysis=false",
    "analysis_args.figures=false",
)


def install_forecast_failure(kind: str, keras_module, classification_run, patch):
    """Install the failure ``kind`` (one of :data:`FORECAST_KINDS`) with ``patch`` (a monkeypatch) or, for ``None``, for good."""
    def set_attribute(owner, name, value):
        if patch is None:
            setattr(owner, name, value)
        else:
            patch.setattr(owner, name, value)

    def fail(*args, **kwargs):
        raise RuntimeError(FORECAST_MESSAGE)

    real_save, real_evaluate = keras_module.Model.save, classification_run.evaluate_dict
    calls = {"evaluate": 0}

    def save_unless(name):
        def save(self, filepath, *args, **kwargs):
            if name in str(filepath):
                raise OSError(FORECAST_MESSAGE)
            return real_save(self, filepath, *args, **kwargs)

        return save

    def evaluate(model, dataset):
        calls["evaluate"] += 1
        if calls["evaluate"] == 2:
            raise RuntimeError(FORECAST_MESSAGE)
        return real_evaluate(model, dataset)

    if kind == "fit":
        set_attribute(keras_module.Model, "fit", fail)
    elif kind == "best_save":
        set_attribute(keras_module.Model, "save", save_unless("best_model"))
    elif kind == "final_save":
        set_attribute(keras_module.Model, "save", save_unless("final_model"))
    elif kind == "history_write":
        set_attribute(classification_run, "save_training_history_json", fail)
    elif kind == "evaluation":
        set_attribute(classification_run, "evaluate_dict", evaluate)
    else:
        raise AssertionError(kind)


def forecast_driver(setup: Sequence[str] = ()) -> str:
    """The child's program: the ``setup`` lines, the failure ``install_forecast_failure`` installs, then ``train.__main__``
    (the real task and ``sys.exit``)."""
    return "\n".join([
        "import os", "import runpy", "import sys", f"sys.path.insert(0, {str(REPO_ROOT)!r})", "import keras",
        "import train.common.classification_run as classification_run", f"FORECAST_MESSAGE = {FORECAST_MESSAGE!r}",
        *setup, "",
        inspect.getsource(install_forecast_failure),
        'install_forecast_failure(os.environ["FORECAST_FAILURE"], keras, classification_run, None)',
        'runpy.run_module("train", run_name="__main__")',
    ])


@pytest.fixture(scope="module", params=FORECAST_KINDS)
def forecast_child(request, tmp_path_factory):
    base = tmp_path_factory.mktemp("forecast_failing_" + request.param)
    (base / "cwd").mkdir()
    out = base / "run"
    env = {**os.environ, "FORECAST_FAILURE": request.param, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "-c", forecast_driver(), "time_series_forecasting", f"out={out}", *FORECAST_TOKENS],
        cwd=base / "cwd", capture_output=True, text=True, timeout=900, env=env,
    )
    return request.param, result, out


def test_a_failed_forecasting_run_exits_non_zero_and_not_with_the_configuration_code(forecast_child):
    kind, result, _ = forecast_child
    assert result.returncode != 0, (kind, result.stdout[-1000:])
    assert result.returncode != 2, (kind, result.stderr[-3000:])


def test_the_forecasting_traceback_names_the_stubbed_failure(forecast_child):
    kind, result, _ = forecast_child
    assert "Traceback" in result.stderr and FORECAST_MESSAGE in result.stderr, (kind, result.stderr[-3000:])
    assert not [line for line in result.stderr.splitlines() if line.startswith("error:")]


def test_the_forecasting_files_are_the_ones_written_before_the_failure_and_no_summary_claims_success(forecast_child):
    kind, _, out = forecast_child
    assert (out / "config.json").is_file()
    # What exists is what the code reached: fit and the best save fail inside `fit`, so neither model is there; a final save
    # that fails leaves the best; a failed history write comes after both models; a failed evaluation comes after the best.
    expected = {
        "fit": (False, False), "best_save": (False, False), "final_save": (True, False),
        "history_write": (True, True), "evaluation": (True, False),
    }[kind]
    assert ((out / "best_model.keras").exists(), (out / "final_model.keras").exists()) == expected, kind
    assert not (out / "training_history.json").exists() or kind != "fit"
    summary = out / "results_summary.json"
    if summary.exists():
        assert json.loads(summary.read_text()).get("success") is not True


@pytest.fixture
def forecasting_world(monkeypatch):
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "TASKS", {})
    recipes.register_builtin_models()
    register_builtin_tasks()


def forecast_request(out: Path):
    from train.api.tasks.time_series_forecasting import train_time_series_forecasting

    return dict(
        function=train_time_series_forecasting,
        keywords=dict(
            out=str(out), model="nbeats/tiny", epochs=1, batch_size=8,
            data={"lookback": 32, "horizon": 8, "realizations": 1, "series_length": 600,
                  "patterns": ["ar_process", "complex_seasonal", "cubic_trend", "arma_process"],
                  "n_select": 16, "n_test": 16, "steps_per_epoch": 4},
            analysis_args={"model_analysis": False, "figures": False}),
    )


@pytest.mark.parametrize("kind", FORECAST_KINDS)
def test_the_stubbed_forecasting_failure_reaches_the_caller_of_the_function(forecasting_world, monkeypatch, tmp_path, kind):
    import train.common.classification_run as classification_run

    install_forecast_failure(kind, keras, classification_run, monkeypatch)
    request = forecast_request(tmp_path / "run")
    with pytest.raises((RuntimeError, OSError), match=FORECAST_MESSAGE):
        request["function"](**request["keywords"])
    assert (tmp_path / "run" / "config.json").is_file()


@pytest.mark.parametrize("kind", FORECAST_KINDS)
def test_the_stubbed_forecasting_failure_reaches_the_caller_of_the_cli_main(forecasting_world, monkeypatch, tmp_path, kind):
    import train.common.classification_run as classification_run

    install_forecast_failure(kind, keras, classification_run, monkeypatch)
    out = tmp_path / "run"
    tokens = [token for token in FORECAST_TOKENS]
    with pytest.raises((RuntimeError, OSError), match=FORECAST_MESSAGE):
        cli.main(["time_series_forecasting", f"out={out}", *tokens])
    assert (out / "config.json").is_file()


# ---------------------------------------------------------------------
# The depth task: the same five failures, stubbed where the real code reaches them
# ---------------------------------------------------------------------
#
# The same stubs as forecasting (``install_forecast_failure``) in a depth run on the synthetic MegaDepth tree. The model is
# ``test_depth_task.create_constant_depth``, a savable constant-output stand-in installed as the family's adapter: what fails
# here is the sequence and the depth task's own reporting (the second ``evaluate_dict`` is the test crops' score in
# ``evaluate_and_report``, the first the untrained model's in ``initial_loss_check``), and a vit_s would add a graph trace of
# about 40 s on the CPU to every one of the fifteen runs.

DEPTH_STAND_IN = (
    "import train.api.depth_models as depth_models",
    "from tests.test_train.test_api.test_depth_task import create_constant_depth",
    "depth_models.create_depth_anything = create_constant_depth",
)


def depth_tokens(out: Path, tree: Path) -> list:
    return [
        "depth_estimation", f"out={out}", "model=depth_anything/vit_s", f"data.root={tree}", "data.train_split=0.6",
        "data.n_select=8", "data.n_test=8", "data.layout_samples=16", "data.steps_per_epoch=3", "data.workers=0",
        "transform_args.image_size=28", "epochs=1", "batch_size=4", 'model_args={"pretrained":false}',
        "analysis_args.model_analysis=false", "analysis_args.figures=false",
    ]


@pytest.fixture(scope="module")
def depth_tree(tmp_path_factory):
    from tests.test_train.test_common_depth_data import write_megadepth_tree

    return write_megadepth_tree(tmp_path_factory.mktemp("depth_failing_tree"))


@pytest.fixture(scope="module", params=FORECAST_KINDS)
def depth_child(request, depth_tree, tmp_path_factory):
    base = tmp_path_factory.mktemp("depth_failing_" + request.param)
    (base / "cwd").mkdir()
    out = base / "run"
    env = {**os.environ, "FORECAST_FAILURE": request.param, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "-c", forecast_driver(DEPTH_STAND_IN), *depth_tokens(out, depth_tree)],
        cwd=base / "cwd", capture_output=True, text=True, timeout=900, env=env,
    )
    return request.param, result, out


def test_a_failed_depth_run_exits_non_zero_and_not_with_the_configuration_code(depth_child):
    kind, result, _ = depth_child
    assert result.returncode != 0, (kind, result.stdout[-1000:])
    assert result.returncode != 2, (kind, result.stderr[-3000:])


def test_the_depth_traceback_names_the_stubbed_failure(depth_child):
    kind, result, _ = depth_child
    assert "Traceback" in result.stderr and FORECAST_MESSAGE in result.stderr, (kind, result.stderr[-3000:])
    assert not [line for line in result.stderr.splitlines() if line.startswith("error:")]


def test_the_depth_files_are_the_ones_written_before_the_failure_and_no_summary_claims_success(depth_child):
    kind, _, out = depth_child
    assert (out / "config.json").is_file()
    # What exists is what the code reached, the order of forecasting: fit and the best save fail inside `fit`; a final save
    # that fails leaves the best; the history is written after both models; the test crops are scored before the final save.
    expected = {
        "fit": (False, False), "best_save": (False, False), "final_save": (True, False),
        "history_write": (True, True), "evaluation": (True, False),
    }[kind]
    assert ((out / "best_model.keras").exists(), (out / "final_model.keras").exists()) == expected, kind
    assert not (out / "training_history.json").exists() or kind != "fit"
    summary = out / "results_summary.json"
    if summary.exists():
        assert json.loads(summary.read_text()).get("success") is not True


@pytest.fixture
def depth_world(monkeypatch):
    from tests.test_train.test_api.test_depth_task import create_constant_depth
    import train.api.depth_models as depth_models

    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "TASKS", {})
    recipes.register_builtin_models()
    register_builtin_tasks()
    monkeypatch.setattr(depth_models, "create_depth_anything", create_constant_depth)


@pytest.mark.parametrize("kind", FORECAST_KINDS)
def test_the_stubbed_depth_failure_reaches_the_caller_of_the_function(depth_world, depth_tree, monkeypatch, tmp_path, kind):
    import train.common.classification_run as classification_run
    from train.api.tasks.depth_estimation import train_depth_estimation

    install_forecast_failure(kind, keras, classification_run, monkeypatch)
    out = tmp_path / "run"
    with pytest.raises((RuntimeError, OSError), match=FORECAST_MESSAGE):
        train_depth_estimation(
            out=str(out), model="depth_anything/vit_s", epochs=1, batch_size=4, model_args={"pretrained": False},
            data={"root": str(depth_tree), "train_split": 0.6, "n_select": 8, "n_test": 8, "layout_samples": 16,
                  "steps_per_epoch": 3, "workers": 0},
            transform_args={"image_size": 28}, analysis_args={"model_analysis": False, "figures": False})
    assert (out / "config.json").is_file()


@pytest.mark.parametrize("kind", FORECAST_KINDS)
def test_the_stubbed_depth_failure_reaches_the_caller_of_the_cli_main(depth_world, depth_tree, monkeypatch, tmp_path, kind):
    import train.common.classification_run as classification_run

    install_forecast_failure(kind, keras, classification_run, monkeypatch)
    out = tmp_path / "run"
    with pytest.raises((RuntimeError, OSError), match=FORECAST_MESSAGE):
        cli.main(depth_tokens(out, depth_tree))
    assert (out / "config.json").is_file()

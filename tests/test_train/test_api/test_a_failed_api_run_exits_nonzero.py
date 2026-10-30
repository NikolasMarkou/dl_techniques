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

import json
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

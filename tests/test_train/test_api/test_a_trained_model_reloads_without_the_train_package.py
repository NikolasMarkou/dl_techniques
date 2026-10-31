"""A model trained through the API loads by default wherever ``dl_techniques`` is importable.

Review iter 2, concern 8. ``pyproject.toml`` builds the wheel from
``dl_techniques*`` only, so an installed library has no ``train`` package. A
float32 model once depended on ``train.common.classification_run.TopKAccuracy``
and could not be loaded there: ``keras.models.load_model`` raised
``Could not deserialize class 'TopKAccuracy'``.

Now a float32 run compiles the STOCK top-k metric and a half-precision run
compiles ``dl_techniques.metrics.half_precision_top_k.HalfPrecisionTopKAccuracy``.
Both archives are produced by the real function, then loaded in a fresh
interpreter where ``sys.modules['train'] = None`` makes any import of ``train``
fail, with default ``load_model`` arguments. The metric names stay
``accuracy`` and ``top{k}_accuracy`` in both cases, so a ``watch_metric`` holds
either way. The tests write only under ``tmp_path``.
"""

import json
import os
import subprocess
import sys
import textwrap
import zipfile

import pytest

import train.api.recipes as recipes
from train.api import registry
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import train_image_classification

from tests.classification_fakes import use_tiny_raw

#: Run in the child: load with defaults, evaluate once, print what was seen as JSON.
CHILD = textwrap.dedent(
    """
    import json, sys
    sys.modules["train"] = None
    import numpy as np
    import keras
    import dl_techniques  # the library must be importable; the trainer package is not

    model = keras.models.load_model(sys.argv[1])
    scores = model.evaluate(
        np.zeros((4, 32, 32, 3), "float32"), np.zeros(4, "int32"), verbose=0, return_dict=True
    )
    print(json.dumps({
        "scores": sorted(scores),
        "policy": model.dtype_policy.name,
        "train_blocked": sys.modules["train"] is None,
    }))
    """
)


def compile_modules(archive):
    """Return the ``module`` of every metric the archive's compile config names."""
    with zipfile.ZipFile(archive) as zf:
        config = json.loads(zf.read("config.json"))
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "module" in node and "class_name" in node:
                found.append((node["module"], node["class_name"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(config["compile_config"]["metrics"])
    return found


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Train two tiny models through the real function: float32 and mixed_float16."""
    base = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        for precision in ("float32", "mixed_float16"):
            train_image_classification(
                out=str(base / precision),
                model="vit/vit_pico",
                data={"dataset": "cifar10"},
                epochs=1,
                batch_size=4,
                precision=precision,
                analysis_args={"model_analysis": False},
            )
    return base


def load_without_train(archive):
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(archive)],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "MPLBACKEND": "Agg"},
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("precision, policy", [("float32", "float32"), ("mixed_float16", "mixed_float16")])
@pytest.mark.parametrize("archive", ["best_model.keras", "final_model.keras"])
def test_both_models_reload_with_train_blocked_and_report_the_same_metric_names(
    runs, precision, policy, archive
):
    seen = load_without_train(runs / precision / archive)
    assert seen["train_blocked"] is True
    assert seen["policy"] == policy
    assert seen["scores"] == ["accuracy", "loss", "top5_accuracy"]


def test_a_float32_archive_names_no_class_of_this_repository(runs):
    modules = compile_modules(runs / "float32" / "final_model.keras")
    top_k = [pair for pair in modules if pair[1] != "SparseCategoricalAccuracy"]
    assert top_k == [("keras.metrics", "SparseTopKCategoricalAccuracy")]
    assert not [m for m, _ in modules if m.split(".")[0] in ("train", "dl_techniques")]


def test_a_half_precision_archive_names_the_library_class_and_never_train(runs):
    modules = compile_modules(runs / "mixed_float16" / "final_model.keras")
    assert ("dl_techniques.metrics.half_precision_top_k", "HalfPrecisionTopKAccuracy") in modules
    assert not [m for m, _ in modules if m.split(".")[0] == "train"]


def test_the_control_blocking_train_does_stop_an_import_of_it():
    done = subprocess.run(
        [sys.executable, "-c", "import sys; sys.modules['train'] = None; import train.common"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode != 0 and "ModuleNotFoundError" in done.stderr

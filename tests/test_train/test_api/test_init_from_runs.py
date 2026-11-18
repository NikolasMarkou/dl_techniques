"""``init``: a run that starts from an earlier run's weights, through the task functions and the command line.

plan-2026-11-18T072904-0fd54b91 (D-002, D-003, D-004, D-008, D-011, D-021, D-033). The ``preclaim`` tests: the
source run is read before the run directory is claimed (``init_from.read_source``, called by
``TaskTrainer._resolve``), and every refusal there exits 2, names what is wrong and claims nothing; ``checkpoint``,
the key ``init`` replaced, is refused by name on the command line and in YAML and is no Python keyword any more;
the ``model_args.pretrained`` doors (a classifier takes ``false`` only; a depth run under ``init`` reads no official
weights and keeps the warm-start rate and the head's 10x).

A ``preclaim`` source is a FAKE run directory: a ``task.json`` written by the record builders of
:mod:`train.api.predictor` (so it passes the real record check), a ``config.json`` holding the resolved
``used.model_args``, and a placeholder ``best_model.keras``. Nothing before the claim opens the archive, so no
training is needed.

The ``wire`` and ``resume`` tests (D-010, D-014, D-016, D-018) train REAL tiny CPU sources (module-scope fixtures)
and start targets from them: the weights ``fit`` starts from are read by a callback and compared, path by path
relative to the model, with the source archive's; the ``task.json`` ``init`` block is compared with what was
copied, resampled, reset, grown and dropped; a resume reads its pins from its own record and opens no file of the
source. ``keras.backend.clear_session()`` runs before every run, as a fresh process would start.

The ``split`` tests (D-012, D-037): the source's training data against this run's selection and test data, counted and
never refused. Each count is compared with an oracle written here (numpy set intersections of the seeded cuts, the
forecasting segment arithmetic, the scene sets); the task hooks are called on fake data where a run would add nothing,
and two real tiny runs check the record, the summary flags, the success line and that the run finishes.

The ``bar`` tests (D-013): under ``init`` a classification, segmentation or denoising run must also beat the loaded
weights' epoch-zero success value, read from the fit sidecar (also on a resume); a stub sidecar drives the real bars,
and runs at a 1e-30 learning rate (no float32 weight moves) miss the bar their absolute threshold alone would pass.

The ``record`` tests (D-014): the summary of a fine-tuned run holds its ``task.json`` block, the flags and the loaded
weights' value its bar compared; it predicts and exports with its source gone; each record kind takes the block.
"""

import builtins
import contextlib
import dataclasses
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional

import keras
import numpy as np
import pytest
from PIL import Image

from train.api import cli, init_from, predictor, recipes, registry, resume_decisions
from train.api.args import AutoContext
from train.api.config import validate
from train.api.depth_args import PRETRAINED_LR
from train.api.errors import ConfigError, ConfigValidationError
from train.api.pretrain_models import NO_TRANSFER
from train.api.tasks.depth_estimation import DepthEstimationConfig, DepthEstimationTrainer, train_depth_estimation
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
    train_image_classification,
)
from train.api.tasks.image_denoising import ImageDenoisingConfig, ImageDenoisingTrainer, train_image_denoising
from train.api.tasks.image_segmentation import ImageSegmentationTrainer, train_image_segmentation
from train.api.tasks.pretrain import train_pretrain
from train.api.tasks.time_series_forecasting import (
    TimeSeriesForecastingConfig,
    TimeSeriesForecastingTrainer,
    train_time_series_forecasting,
)
from train.api.trainer import TaskTrainer

from ._forecast_fixtures import world  # noqa: F401  (a module-scope fixture: the one registry world)

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_ENV = "DL_TECHNIQUES_DINOV2_WEIGHTS_DIR"
#: What a placeholder archive holds: nothing before the claim reads it, only its hash.
PLACEHOLDER = b"not a keras archive: nothing before the claim opens it"

#: The classification target every function case asks for (built-in cifar10: no data is loaded before the claim).
CLASSIFIER = {"model": "vit/vit_pico", "data": {"dataset": "cifar10"}, "epochs": 1, "batch_size": 8}
#: What a vit_pico source resolved at 32 px on cifar10 records under ``used.model_args``.
VIT_USED = {"patch_size": 4, "kernel_regularizer": None}


def classification_record(model: str = "vit/vit_pico", channels: int = 3, normalize: str = "standardize",
                          side: int = 32) -> Dict[str, Any]:
    family, variant = model.split("/")
    unit = normalize == "unit"
    return predictor.build_task_record(
        task="image_classification", model=model, family=family, variant=variant,
        class_names=[f"c{index}" for index in range(10)], input_shape=[side, side, channels], normalize=normalize,
        mean=[0.0 if unit else 0.5] * channels, std=[1.0 if unit else 0.25] * channels, data_kind="cifar10",
        source_channels=channels)


def segmentation_record(model: str = "convunext/tiny") -> Dict[str, Any]:
    family, variant = model.split("/")
    return predictor.build_segmentation_record(
        model=model, family=family, variant=variant, class_names=["a", "b", "c"], input_size=[64, 64], channels=3,
        normalize="unit", mean=[0.0] * 3, std=[1.0] * 3)


def forecast_record(model: str = "nbeats/tiny", lookback: int = 168, horizon: int = 24) -> Dict[str, Any]:
    family, variant = model.split("/")
    return predictor.build_forecast_record(
        model=model, family=family, variant=variant, lookback=lookback, horizon=horizon, channels=1, value_cols=None,
        time_col=None, scale_floor=1e-6, decode=family, quantile_levels=None, point_index=None,
        data_format="builtin", data_sha256=None)


def depth_record(side: int = 28) -> Dict[str, Any]:
    return predictor.build_depth_record(
        model="depth_anything/vit_s", family="depth_anything", variant="vit_s", input_size=[side, side], channels=3)


def write_source(root: Path, record: Optional[Dict[str, Any]], used: Optional[Dict[str, Any]] = None, *,
                 config: bool = True, best: bool = True) -> Path:
    """A fake run directory: ``task.json`` (``record``; ``None`` writes none), ``config.json`` and a placeholder archive."""
    root.mkdir(parents=True)
    if record is not None:
        (root / "task.json").write_text(json.dumps(record))
    if config:
        model = record["model"] if record is not None else "vit/vit_pico"
        (root / "config.json").write_text(json.dumps({"model": model, "used": {"model_args": dict(used or {})}}))
    if best:
        (root / "best_model.keras").write_bytes(PLACEHOLDER)
    return root


def old_record() -> Dict[str, Any]:
    return {**classification_record(), "schema_version": 1}


def as_file(root: Path) -> Path:
    write_source(root, classification_record(), VIT_USED)
    return root / "best_model.keras"


#: ``name: (make the source under a tmp directory, the extra request, the message pattern)``: one row per pre-claim
#: refusal of a classification target (vit_pico on cifar10 at 32 px, standardized).
CLASSIFIER_CASES = {
    "missing": (lambda root: root, {}, r"Invalid value for 'init': .*nowhere does not exist; give the directory of an "
                                       r"earlier python -m train run"),
    "a_file": (as_file, {}, r"Invalid value for 'init': .*best_model\.keras is a file; give the run directory that "
                            r"holds it \(init reads task\.json, config\.json and best_model\.keras\)"),
    "no_record": (lambda root: write_source(root, None, VIT_USED), {},
                  r"task\.json does not exist; a source is a python -m train run of a task that writes a record "
                  r"\(pretrain runs and legacy trainer runs write none\)"),
    "no_config": (lambda root: write_source(root, classification_record(), config=False), {},
                  r"config\.json does not exist; a source is a python -m train run directory"),
    "no_best_model": (lambda root: write_source(root, classification_record(), VIT_USED, best=False), {},
                      r"holds no best_model\.keras \(the run ended before its first epoch\)"),
    "old_record": (lambda root: write_source(root, old_record(), VIT_USED), {}, r"'init': the source's record cannot be used: .*'schema_version' is 1, an older layout"),
    "undeclared_pair": (lambda root: write_source(root, segmentation_record()), {},
                        r"convunext \(image_segmentation\) cannot seed vit \(image_classification\); vit takes weights "
                        r"from vit \(image_classification\)"),
    "channels": (lambda root: write_source(root, classification_record(channels=1), VIT_USED), {},
                 r"vit sizes its weights by the input channels; the source trained at 1, this run asks for 3"),
    "input_scaling": (lambda root: write_source(root, classification_record(normalize="unit"), VIT_USED), {},
                      r"the source takes its input in \[0, 1\] and this run standardizes its input "
                      r"\(transform_args\.normalize\)"),
    "no_pin": (lambda root: write_source(root, classification_record(), {}), {},
               r"config\.json records no used\.model_args\.patch_size, which init keeps from the source"),
    "pin_changed": (lambda root: write_source(root, classification_record(), VIT_USED),
                    {"model_args": {"patch_size": 8}},
                    r"Invalid value for 'model_args\.patch_size': the source was built with patch_size=4 and init "
                    r"keeps the source's architecture"),
}


@pytest.mark.parametrize("case", sorted(CLASSIFIER_CASES))
def test_preclaim_a_classifier_source_is_refused_by_name_and_nothing_is_claimed(world, tmp_path, case):
    make, extra, pattern = CLASSIFIER_CASES[case]
    source = make(tmp_path / "nowhere")
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=pattern):
        train_image_classification(out=out, init=source, **{**CLASSIFIER, **extra})
    assert not out.exists(), "a pre-claim refusal claimed the run directory"


def prepared(config: Any, trainer_class: Any) -> Any:
    """Resolve ``config`` as ``TaskTrainer.run`` does, up to the claim; return ``(trainer, resolved, context)``."""
    trainer = trainer_class(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    return trainer, resolved, context


def classifier_config(out: Path, **changes: Any) -> ImageClassificationConfig:
    return validate(ImageClassificationConfig, {"out": str(out), **CLASSIFIER, **changes})


def test_preclaim_a_well_formed_source_passes_and_its_pins_and_hash_are_taken(world, tmp_path):
    source = write_source(tmp_path / "source", classification_record(), VIT_USED)
    for asked in ({}, {"model_args": {"patch_size": 4}}):  # the source's own value, typed, is accepted
        trainer, resolved, _ = prepared(classifier_config(tmp_path / "run", init=str(source), **asked),
                                        ImageClassificationTrainer)
        taken = trainer._init_source
        assert taken.path == source and taken.pins == {"patch_size": 4} and taken.relation is None
        assert taken.sha256 == hashlib.sha256(PLACEHOLDER).hexdigest()
        assert taken.record == classification_record()
    assert not (tmp_path / "run").exists()


def test_preclaim_a_relative_init_resolves_at_the_repo_root_as_out_does(world, tmp_path):
    relative = "results/p4g_test_no_such_source_run"
    assert not (REPO_ROOT / relative).exists()
    with pytest.raises(ConfigError, match=f"'init': {REPO_ROOT / relative} does not exist"):
        prepared(classifier_config(tmp_path / "run", init=relative), ImageClassificationTrainer)


def test_preclaim_a_cross_task_source_is_read_with_its_declared_relation(world, tmp_path):
    from train.api.tasks.image_segmentation import ImageSegmentationConfig, ImageSegmentationTrainer
    from train.api.segmentation_models import CONVNEXT_SEG_TRANSFER

    # A segmenter's input is never standardized, so its classifier source left its input in [0, 1] too.
    source = write_source(tmp_path / "source", classification_record("convnext_v1/cifar10", normalize="unit"),
                          {"strides": 2})
    images, masks = segmentation_folders(tmp_path / "data")
    config = validate(ImageSegmentationConfig, {
        "out": str(tmp_path / "run"), "model": "convnext_seg/cifar10", "init": str(source),
        "data": {"format": "folder", "train_images": str(images), "train_masks": str(masks), "num_classes": 3},
        "transform_args": {"image_size": [64, 64]}})
    trainer, _, _ = prepared(config, ImageSegmentationTrainer)
    assert trainer._init_source.relation == CONVNEXT_SEG_TRANSFER.sources[("image_classification", "convnext_v1")]
    assert trainer._init_source.pins == {}


def segmentation_folders(root: Path):
    from ._segmentation_fixtures import make_folder

    return make_folder(root, count=4, size=(64, 64), classes=3)


def test_preclaim_a_lookback_the_family_sizes_its_weights_by_is_refused(world, tmp_path):
    source = write_source(tmp_path / "source", forecast_record("nbeats/tiny", lookback=168))
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=(
            r"nbeats sizes its weights by data\.lookback; the source trained at 168, this run asks for 336 "
            r"\(families of time_series_forecasting that keep every weight across a data\.lookback change: ets, "
            r"timestral, tirex, xlstm\)")):
        train_time_series_forecasting(
            out=out, model="nbeats/tiny", init=source, batch_size=8,
            data={"lookback": 336, "horizon": 24, "realizations": 1, "series_length": 5000,
                  "patterns": ["ar_process"], "n_select": 16, "n_test": 16})
    assert not out.exists()


def test_preclaim_pretrain_refuses_init_by_its_rows_reason(world, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="Invalid value for 'init': " + NO_TRANSFER.replace("(", r"\(").replace(")", r"\)")):
        train_pretrain(out=out, model="energy_transformer/et_tiny", data={"dataset": "cifar10"}, batch_size=4,
                       model_args={"num_steps": 2}, init=tmp_path / "any_source")
    assert not out.exists()


def test_preclaim_a_row_with_no_transfer_is_refused_by_name(world, tmp_path, monkeypatch):
    spec = registry.MODELS["vit"]
    monkeypatch.setitem(registry.MODELS, "vit", dataclasses.replace(spec, transfer=None, transfer_reason=""))
    with pytest.raises(ConfigError, match=r"'init': the family 'vit' declares no transfer"):
        prepared(classifier_config(tmp_path / "run", init=str(tmp_path / "src")), ImageClassificationTrainer)


def test_preclaim_the_default_init_identity_refuses_naming_the_task():
    with pytest.raises(ConfigError, match=r"Invalid value for 'init': the task 'some_task' does not start from "
                                          r"earlier weights"):
        TaskTrainer.init_identity(SimpleNamespace(task="some_task"), None, None)


def test_preclaim_the_classifier_channels_are_the_loaders(world, tmp_path):
    """The identity's channel count of a built-in dataset is the dataset table's, unless the recipe's loader repeats."""
    from dl_techniques.datasets.vision.classification import get_spec

    from train.api.data_args import DATASETS

    for dataset in DATASETS:
        for model, loader_channels in (("vit/vit_pico", None), ("convnext_v1/cifar10", 3)):
            trainer = ImageClassificationTrainer(classifier_config(tmp_path / "run", model=model))
            spec = registry.MODELS[model.split("/")[0]]
            context = SimpleNamespace(dataset=dataset, defaults=spec.recipe(dataset, model.split("/")[1]))
            expected = context.defaults.get("loader", {}).get("channels") or get_spec(dataset).input_shape[-1]
            identity = trainer.init_identity(trainer.config, context)
            assert identity["channels"] == expected, (dataset, model)
            assert identity["normalize"] == "standardize"


# ---------------------------------------------------------------------
# The command line, in a child: exit 2, nothing claimed, no device made; the reader itself loads no TensorFlow
# ---------------------------------------------------------------------

#: ``argv``: the command; prints what the child saw as one JSON line (the recorder of the builtin-sources test).
DRIVER = textwrap.dedent(
    """
    import json
    import sys

    from tensorflow.python.eager import context as eager

    import train.common.gpu as gpu
    from train.api import cli

    reached = []


    class Stop(Exception):
        pass


    def recorder(*args, **kwargs):
        reached.append("setup_gpu")
        raise Stop


    gpu.setup_gpu = recorder
    try:
        code = cli.main(sys.argv[1:])
    except Stop:
        code = "stopped at setup_gpu"
    print(json.dumps({"code": code, "setup_gpu_reached": bool(reached),
                      "devices_created": eager.context()._context_handle is not None}))
    """
)


@pytest.mark.parametrize("case", ["missing", "no_record", "undeclared_pair", "pin_changed"])
def test_preclaim_the_command_line_exits_2_and_claims_nothing(tmp_path, case):
    make, extra, pattern = CLASSIFIER_CASES[case]
    source = make(tmp_path / "nowhere")
    out = tmp_path / "run"
    argv = ["image_classification", f"out={out}", "model=vit/vit_pico", "data.dataset=cifar10", f"init={source}"]
    argv += [f"model_args.{key}={value}" for key, value in extra.get("model_args", {}).items()]
    done = subprocess.run([sys.executable, "-c", DRIVER, *argv], capture_output=True, text=True, timeout=300)
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen == {"code": 2, "setup_gpu_reached": False, "devices_created": False}, done.stderr[-2000:]
    assert re.search(pattern, done.stderr), done.stderr[-2000:]
    assert not out.exists()


#: Reads a passing source and a refused one through ``read_source`` alone, then reports the heavy modules loaded.
UNIT_DRIVER = textwrap.dedent(
    """
    import json
    import sys
    from types import SimpleNamespace

    from train.api import init_from, recipes, registry
    from train.api.errors import ConfigError

    recipes.register_builtin_models()
    trainer = SimpleNamespace(task="image_classification",
                              init_identity=lambda config, context: {"channels": 3, "normalize": "standardize"})
    config = SimpleNamespace(model_args={})
    context = SimpleNamespace(family="vit")
    taken = init_from.read_source(sys.argv[1], trainer, config, context)
    try:
        init_from.read_source(sys.argv[2], trainer, config, context)
        refused = False
    except ConfigError:
        refused = True
    print(json.dumps({"pins": taken.pins, "refused": refused,
                      "loaded": sorted(name for name in ("tensorflow", "keras", "numpy") if name in sys.modules)}))
    """
)


def test_preclaim_the_source_reader_loads_no_tensorflow_keras_or_numpy(tmp_path):
    source = write_source(tmp_path / "source", classification_record(), VIT_USED)
    done = subprocess.run([sys.executable, "-c", UNIT_DRIVER, str(source), str(tmp_path / "nowhere")],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"pins": {"patch_size": 4}, "refused": True,
                                                                 "loaded": []}


# ---------------------------------------------------------------------
# checkpoint: removed, refused by name; no Python keyword
# ---------------------------------------------------------------------

REMOVED = r"Unknown key 'checkpoint': it was replaced by 'init', which takes the directory of an earlier run"


def test_preclaim_checkpoint_on_the_command_line_is_refused_naming_init(world, tmp_path, capsys):
    out = tmp_path / "run"
    assert cli.main(["image_classification", f"out={out}", "model=vit/vit_pico", "data.dataset=cifar10",
                     "checkpoint=weights.keras"]) == 2
    err = capsys.readouterr().err
    assert re.search(REMOVED, err) and "checkpoint_args" not in err
    assert not out.exists()


def test_preclaim_checkpoint_in_a_config_file_is_refused_naming_init(world, tmp_path, capsys):
    out = tmp_path / "run"
    request = tmp_path / "request.yaml"
    request.write_text(f"out: {out}\nmodel: vit/vit_pico\ndata:\n  dataset: cifar10\ncheckpoint: weights.keras\n")
    assert cli.main(["image_classification", f"config={request}"]) == 2
    err = capsys.readouterr().err
    assert re.search(REMOVED, err) and "checkpoint_args" not in err
    assert not out.exists()


def test_preclaim_checkpoint_in_a_mapping_is_refused_naming_init(tmp_path):
    with pytest.raises(ConfigValidationError, match=REMOVED) as caught:
        validate(ImageClassificationConfig, {"out": str(tmp_path / "run"), **CLASSIFIER, "checkpoint": "w.keras"})
    assert "checkpoint_args" not in str(caught.value)


TASK_FUNCTIONS = (train_image_classification, train_image_segmentation, train_image_denoising,
                  train_time_series_forecasting, train_depth_estimation, train_pretrain)


@pytest.mark.parametrize("function", TASK_FUNCTIONS, ids=[f.__name__ for f in TASK_FUNCTIONS])
def test_preclaim_checkpoint_is_no_keyword_of_a_task_function(function, tmp_path):
    parameters = inspect.signature(function).parameters
    assert "init" in parameters and "checkpoint" not in parameters
    with pytest.raises(TypeError, match="checkpoint"):
        function(out=tmp_path / "run", model="vit/vit_pico", data={}, checkpoint="w.keras")
    assert not (tmp_path / "run").exists()


# ---------------------------------------------------------------------
# model_args.pretrained
# ---------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, "weights/vit.keras"])
def test_preclaim_a_classifier_takes_pretrained_false_only(world, tmp_path, value):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"Invalid value for 'model_args\.pretrained': .*init takes a run directory"):
        train_image_classification(out=out, model_args={"pretrained": value}, **CLASSIFIER)
    assert not out.exists()


def depth_config(out: Path, root: Path, **changes: Any) -> DepthEstimationConfig:
    return validate(DepthEstimationConfig, {
        "out": str(out), "model": "depth_anything/vit_s", "batch_size": 4, "epochs": 1,
        "data": {"root": str(root), "train_split": 0.6, "n_select": 4, "n_test": 4, "layout_samples": 4,
                 "steps_per_epoch": 1, "workers": 0},
        "transform_args": {"image_size": 28}, **changes})


@pytest.fixture(scope="module")
def megadepth_tree(tmp_path_factory):
    from tests.test_train.test_common_depth_data import write_megadepth_tree

    return write_megadepth_tree(tmp_path_factory.mktemp("megadepth"))


def test_preclaim_depth_under_init_reads_no_official_weights_and_keeps_the_warm_start(
        world, megadepth_tree, tmp_path, monkeypatch):
    """The source recorded ``pretrained: true``; the target under ``init`` asks the factory for no official weights
    (the weights directory is empty, so a read would be refused), and keeps 5e-6 with the head at 10x (D-004)."""
    from train.api import depth_models

    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "no_weights"))
    source = write_source(tmp_path / "source", depth_record(), {"pretrained": True})
    trainer, resolved, context = prepared(depth_config(tmp_path / "run", megadepth_tree, init=str(source)),
                                          DepthEstimationTrainer)
    kwargs = trainer._model_kwargs(resolved, context)
    assert kwargs["pretrained"] is False
    assert resolved.optim_args.lr == PRETRAINED_LR
    model = depth_models.create_depth_anything(variant="vit_s", input_shape=(28, 28, 3), **kwargs)
    trainer._built_kwargs = kwargs
    table = trainer.optimizer_options(resolved, model)["lr_multipliers"]
    assert table and set(table.values()) == {10.0}
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("value", [True, False])
def test_preclaim_depth_under_init_refuses_an_explicit_pretrained(world, megadepth_tree, tmp_path, value):
    source = write_source(tmp_path / "source", depth_record(), {"pretrained": True})
    config = depth_config(tmp_path / "run", megadepth_tree, init=str(source), model_args={"pretrained": value})
    with pytest.raises(ConfigError, match=r"Invalid value for 'model_args\.pretrained': init replaces every weight"):
        prepared(config, DepthEstimationTrainer)
    assert not (tmp_path / "run").exists()


def test_preclaim_depth_under_init_needs_adamw_for_the_heads_10x(world, megadepth_tree, tmp_path):
    source = write_source(tmp_path / "source", depth_record(), {"pretrained": True})
    config = depth_config(tmp_path / "run", megadepth_tree, init=str(source),
                          optim_args={"optimizer": "adam", "lr": 1e-5, "weight_decay": None})
    with pytest.raises(ConfigError, match=r"'optim_args\.optimizer'.*adamw"):
        prepared(config, DepthEstimationTrainer)


# ---------------------------------------------------------------------
# wire: real tiny CPU sources, and targets started from them
# ---------------------------------------------------------------------

#: What every classification run of this section asks besides its model, data and side.
QUIET = {"epochs": 1, "batch_size": 8, "analysis_args": {"model_analysis": False, "figures": False},
         "metric_args": {"success_threshold": 0.0}}
TEN = [f"k{index}" for index in range(10)]
RENAMED = [f"r{index}" for index in range(10)]
HUNDRED = [f"h{index:02d}" for index in range(100)]


def write_tree(root: Path, names, per_class: int = 3, side: int = 16, seed: int = 0) -> Path:
    """``root/<class>/<i>.png``: noise images, a tint per class."""
    rng = np.random.default_rng(seed)
    for k, name in enumerate(names):
        for index in range(per_class):
            path = root / name / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((rng.integers(0, 120, (side, side, 3)) + (k % 8) * 16).astype(np.uint8)).save(path)
    return root


def relative(model: Any) -> Dict[str, np.ndarray]:
    """``{path relative to the model: value}``: the model's own leading name dropped, as a fresh process would not have it."""
    root = f"{model.name}/"
    return {(v.path[len(root):] if v.path.startswith(root) else v.path): np.array(v) for v in model.weights}


def archive_weights(run_dir: Path) -> Dict[str, np.ndarray]:
    return relative(keras.models.load_model(str(run_dir / "best_model.keras"), compile=False))


def record_of(run_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(run_dir) / "task.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def started_weights(trainer_class: Any) -> Iterator[Dict[str, np.ndarray]]:
    """Patch ``trainer_class.extra_callbacks`` so the weights ``fit`` starts from are kept (relative paths)."""
    seen: Dict[str, np.ndarray] = {}
    original = trainer_class.extra_callbacks

    class Started(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            seen.update(relative(self.model))

    def extra(self, config, data):
        return [*original(self, config, data), Started()]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer_class, "extra_callbacks", extra)
        yield seen


def started(trainer_class: Any, function: Any, **request: Any) -> Dict[str, np.ndarray]:
    """Run ``function(**request)`` in a fresh Keras session and return the weights its ``fit`` started from."""
    keras.backend.clear_session()
    with started_weights(trainer_class) as seen:
        function(**request)
    return dict(seen)


def classify(out: Any, tree: Path, side: int, model: str = "vit/vit_pico", **changes: Any) -> Dict[str, Any]:
    return {"out": out, "model": model, "data": {"format": "folder", "train_dir": str(tree)},
            "transform_args": {"image_size": side}, **QUIET, **changes}


def check_copied(block: Dict[str, Any], start: Dict[str, np.ndarray], source: Dict[str, np.ndarray],
                 prefix: str = "") -> None:
    """Every target weight the block does not name as reset, grown or resampled equals the source's at its path."""
    transfer = block["transfer"]
    others = set(transfer["reset"]) | set(transfer["grown"]) | {row["path"] for row in transfer["resampled"]}
    copied = [path for path in start if path not in others]
    assert transfer["copied"] == len(copied)
    for path in copied:
        assert path.startswith(prefix), path
        np.testing.assert_array_equal(start[path], source[path[len(prefix):]], err_msg=path)


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    root = tmp_path_factory.mktemp("trees")
    return {"ten": write_tree(root / "ten", TEN), "renamed": write_tree(root / "renamed", RENAMED, seed=1),
            "hundred": write_tree(root / "hundred", HUNDRED, per_class=2, seed=2)}


@pytest.fixture(scope="module")
def vit_source(world, trees, tmp_path_factory):
    """vit_pico on 10 classes at 32 px (patch 4, a 65-row position table), 1 epoch."""
    out = tmp_path_factory.mktemp("vit_source") / "run"
    keras.backend.clear_session()
    train_image_classification(**classify(str(out), trees["ten"], 32))
    return out


@pytest.fixture(scope="module")
def vit_64(world, trees, vit_source, tmp_path_factory):
    """vit_pico on 100 classes at 64 px, from a copy of ``vit_source`` (a resume test renames the copy away)."""
    root = tmp_path_factory.mktemp("vit_64")
    source = root / "source"
    shutil.copytree(vit_source, source)
    out = root / "run"
    request = classify(str(out), trees["hundred"], 64, init=str(source), batch_size=16)
    start = started(ImageClassificationTrainer, train_image_classification, **request)
    return SimpleNamespace(source=source, out=out, request=request, start=start)


def test_wire_a_vit_at_32_px_seeds_one_at_64_px_keeping_its_patch_and_resampling_its_table(vit_source, vit_64):
    from tests.test_models.test_dino.dinov2_numpy_reference import torch_bicubic_2d

    used = json.loads((vit_64.out / "config.json").read_text())["used"]["model_args"]
    assert used["patch_size"] == 4  # the recipe derives 16 at 64 px; the source's 4 is kept
    source, start = archive_weights(vit_source), vit_64.start
    block = record_of(vit_64.out)["init"]
    assert (block["source"], block["file"], block["sha256"]) == (
        str(vit_64.source), "best_model.keras", sha(vit_64.source / "best_model.keras"))
    assert (block["task"], block["family"], block["variant"], block["relation"]) == (
        "image_classification", "vit", "vit_pico", "same_task")
    assert block["pinned"] == {"patch_size": 4}
    assert block["geometry"] == {key: record_of(vit_source)[key] for key in ("input_shape", "channels")}
    table, grid = "pos_embed/pos_embedding", (32 // 4, 64 // 4)
    assert block["transfer"]["resampled"] == [{"path": table, "rule": "vit_pos_bicubic",
                                               "from": [1, 1 + grid[0] ** 2, 192], "to": [1, 1 + grid[1] ** 2, 192]}]
    heads = sorted(path for path in start if path.startswith("classification_head/"))
    assert heads and sorted(block["transfer"]["reset"]) == heads  # 10 classes to 100: the head is fresh
    assert sorted(block["transfer"]["dropped"]) == sorted(p for p in source if p.startswith("classification_head/"))
    assert block["transfer"]["grown"] == []
    check_copied(block, start, source)
    assert start[table].shape == (1, 1 + grid[1] ** 2, 192)
    np.testing.assert_array_equal(start[table][:, :1], source[table][:, :1])  # the class row is kept
    image = source[table][0, 1:].astype(np.float64).reshape(grid[0], grid[0], 192).transpose(2, 0, 1)
    expected = torch_bicubic_2d(image, grid[1], grid[1]).transpose(1, 2, 0).reshape(grid[1] ** 2, 192)
    np.testing.assert_allclose(start[table][0, 1:], expected, atol=1e-5)


def test_wire_the_block_passes_the_record_check_and_a_block_without_a_key_is_refused(vit_64):
    record = record_of(vit_64.out)
    assert predictor.record_problems(record) == []
    assert list(record)[-1] == "init" and list(record)[:-1] == list(predictor.RECORD_KEYS["image_classification"])
    for key in record["init"]:
        broken = {**record, "init": {name: value for name, value in record["init"].items() if name != key}}
        assert any(repr(key) in line for line in predictor.record_problems(broken)), key
    unknown = {**record, "init": {**record["init"], "extra": 1}}
    assert any("'extra'" in line for line in predictor.record_problems(unknown))


@pytest.fixture(scope="module")
def vit_tiny(world, trees, vit_source, tmp_path_factory):
    """vit_tiny (deeper, same width) from ``vit_source`` on the same classes; ``out`` and ``init`` given as ``Path``."""
    out = tmp_path_factory.mktemp("vit_tiny") / "run"
    start = started(ImageClassificationTrainer, train_image_classification,
                    **classify(out, trees["ten"], 32, model="vit/vit_tiny", init=vit_source))
    return out, start


def test_wire_a_deeper_vit_grows_its_new_blocks_and_copies_every_other_weight_with_the_head(vit_source, vit_tiny):
    out, start = vit_tiny
    source = archive_weights(vit_source)
    block = record_of(out)["init"]
    depth = 1 + max(int(re.match(r"transformer_layer_(\d+)/", p).group(1)) for p in source if p.startswith("transformer_layer_"))
    grown = sorted(p for p in start if re.match(r"transformer_layer_(\d+)/", p)
                   and int(re.match(r"transformer_layer_(\d+)/", p).group(1)) >= depth)
    assert grown and sorted(block["transfer"]["grown"]) == grown
    assert block["transfer"]["reset"] == [] and block["transfer"]["dropped"] == []  # the same 10 class names
    assert block["transfer"]["resampled"] == []
    check_copied(block, start, source)
    assert block["transfer"]["copied"] == len(source)


def test_wire_a_path_init_and_out_are_recorded_as_text_once(vit_source, vit_tiny):
    out, _ = vit_tiny
    text = (out / "config.json").read_text()
    config = json.loads(text)
    assert (config["out"], config["init"]) == (str(out), str(vit_source))
    assert text.count(f'"init": "{vit_source}"') == 1 and text.count(str(out)) == 1


def test_wire_the_head_is_reset_when_the_class_names_differ_at_an_equal_count(world, trees, vit_source, tmp_path):
    out = tmp_path / "run"
    start = started(ImageClassificationTrainer, train_image_classification,
                    **classify(str(out), trees["renamed"], 32, init=str(vit_source)))
    source = archive_weights(vit_source)
    block = record_of(out)["init"]
    heads = sorted(p for p in start if p.startswith("classification_head/"))
    assert sorted(block["transfer"]["reset"]) == heads and sorted(block["transfer"]["dropped"]) == heads
    assert all(start[p].shape == source[p].shape for p in heads)  # same shapes: only the names say it is not the same head
    assert not all(np.array_equal(start[p], source[p]) for p in heads)
    check_copied(block, start, source)


def test_wire_another_architecture_is_refused_after_the_claim_and_nothing_is_trained(world, trees, vit_source, tmp_path):
    out = tmp_path / "run"
    keras.backend.clear_session()
    with pytest.raises(ConfigError) as caught:
        train_image_classification(**classify(str(out), trees["ten"], 32, init=str(vit_source),
                                              model_args={"use_layer_scale": True}))
    text = str(caught.value)
    assert re.search(r"'init': \d+ of \d+ weights of vit/vit_pico have no source weight of their shape and no "
                     r"declared rule: .*nothing was copied", text), text
    assert f"The run directory {out} was claimed" in text
    assert (out / "config.json").exists() and not (out / "training_log.csv").exists()
    assert not (out / "best_model.keras").exists() and not (out / "task.json").exists()


def test_wire_an_archive_that_changes_after_the_claim_stops_the_run_naming_both_hashes(
        world, trees, vit_source, tmp_path, monkeypatch):
    source = tmp_path / "source"
    shutil.copytree(vit_source, source)
    before = sha(source / "best_model.keras")
    real = predictor.load_model

    def load_then_rewrite(path):
        loaded = real(path)
        with open(Path(path) / "best_model.keras", "ab") as handle:
            handle.write(b"rewritten while the run read it")
        return loaded

    monkeypatch.setattr(predictor, "load_model", load_then_rewrite)
    out = tmp_path / "run"
    keras.backend.clear_session()
    with pytest.raises(ConfigError) as caught:
        train_image_classification(**classify(str(out), trees["ten"], 32, init=str(source)))
    text, after = str(caught.value), sha(source / "best_model.keras")
    assert before != after and before in text and after in text and f"The run directory {out} was claimed" in text
    assert not (out / "training_log.csv").exists() and not (out / "best_model.keras").exists()


def test_wire_a_pin_is_laid_over_the_computed_value_and_another_user_value_is_refused():
    context = AutoContext(family="vit", variant="vit_pico", dataset="folder", num_classes=3, image_size=64,
                                  defaults={}, pinned={"patch_size": 4})
    optim = SimpleNamespace(optimizer="adamw")
    assert recipes.merge_model_args(context, optim, {}, {"patch_size": 16})["patch_size"] == 4
    assert recipes.merge_model_args(context, optim, {"patch_size": 4}, {"patch_size": 16})["patch_size"] == 4
    with pytest.raises(ConfigError, match=r"Invalid value for 'model_args\.patch_size': the source was built with "
                                          r"patch_size=4 and init keeps the source's architecture"):
        recipes.merge_model_args(context, optim, {"patch_size": 8}, {"patch_size": 16})
    unpinned = dataclasses.replace(context, pinned={})
    assert AutoContext(family="vit", variant="vit_pico", dataset="folder", num_classes=3, image_size=64,
                               defaults={}).pinned == {}
    assert recipes.merge_model_args(unpinned, optim, {}, {"patch_size": 16})["patch_size"] == 16


@pytest.fixture(scope="module")
def segmentation_data(tmp_path_factory):
    from ._segmentation_fixtures import make_folder

    images, masks = make_folder(tmp_path_factory.mktemp("segmentation"), count=12, size=(64, 64), classes=3)
    return {"format": "folder", "train_images": str(images), "train_masks": str(masks), "num_classes": 3}


def test_wire_a_convnext_classifier_seeds_convnext_seg_under_backbone(world, segmentation_data, tmp_path, monkeypatch):
    """The gate's pair (D-035): a cifar10 classifier with its recipe's strides 2, trained on [0, 1] inputs."""
    from tests.classification_fakes import use_tiny_raw

    use_tiny_raw(monkeypatch, n_train=40, n_test=8)
    source_dir, out = tmp_path / "source", tmp_path / "run"
    keras.backend.clear_session()
    train_image_classification(out=str(source_dir), model="convnext_v1/cifar10", data={"dataset": "cifar10"},
                               transform_args={"normalize": "unit"}, **QUIET)
    start = started(ImageSegmentationTrainer, train_image_segmentation,
                    out=str(out), model="convnext_seg/cifar10", data=segmentation_data, init=str(source_dir),
                    epochs=1, batch_size=4, transform_args={"image_size": [32, 32]},
                    analysis_args={"model_analysis": False, "figures": False})
    source = archive_weights(source_dir)
    block = record_of(out)["init"]
    assert block["relation"] == "cross_task" and block["pinned"] == {}
    kept = [p for p in source if not p.startswith("classification_head/")]
    assert sorted(block["transfer"]["dropped"]) == sorted(set(source) - set(kept))
    assert block["transfer"]["copied"] == len(kept)
    assert set(block["transfer"]["reset"]) == {p for p in start if not p.startswith("backbone/")}
    check_copied(block, start, source, prefix="backbone/")


def test_wire_a_convunext_segmenter_seeds_the_biased_denoiser_with_a_fresh_output(world, segmentation_data, tmp_path):
    from .test_image_denoising_task import mapping, write_images

    source_dir, out = tmp_path / "source", tmp_path / "run"
    keras.backend.clear_session()
    train_image_segmentation(out=str(source_dir), model="convunext/tiny", data=segmentation_data, epochs=1,
                             batch_size=4, transform_args={"image_size": [32, 32]},
                             analysis_args={"model_analysis": False, "figures": False})
    folders = write_images(tmp_path / "train", 6, 0), write_images(tmp_path / "val", 3, 1)
    start = started(ImageDenoisingTrainer, train_image_denoising,
                    **mapping(out, folders, "convunext_biased", epochs=1, init=str(source_dir),
                              transform_args={"patch_size": 32, "patches_per_image": 2}))
    source = archive_weights(source_dir)
    block = record_of(out)["init"]
    assert block["relation"] == "cross_task"
    assert sorted(block["transfer"]["reset"]) == sorted(p for p in start if p.startswith("final_output/"))
    assert sorted(block["transfer"]["dropped"]) == sorted(p for p in source if p.startswith("final_output/"))
    assert block["transfer"]["copied"] == len(start) - len(block["transfer"]["reset"])
    check_copied(block, start, source)


def test_wire_a_denoiser_warm_start_scores_the_sources_last_validation_loss_at_epoch_zero(world, tmp_path):
    from .test_image_denoising_task import mapping, write_images

    folders = write_images(tmp_path / "train", 6, 0), write_images(tmp_path / "val", 3, 1)
    keras.backend.clear_session()
    first = train_image_denoising(**mapping(tmp_path / "first", folders, "bfcnn", epochs=1))
    start = started(ImageDenoisingTrainer, train_image_denoising,
                    **mapping(tmp_path / "warm", folders, "bfcnn", epochs=1, init=str(first.run_dir)))
    keras.backend.clear_session()
    cold = train_image_denoising(**mapping(tmp_path / "cold", folders, "bfcnn", epochs=1))
    summaries = [json.loads((tmp_path / name / "results_summary.json").read_text()) for name in ("first", "warm", "cold")]
    # The source's one epoch scored these weights on this very validation set: the warm start's epoch 0 is that score.
    last = json.loads((Path(first.run_dir) / "training_history.json").read_text())["val_loss"][-1]
    assert summaries[1]["initial_loss_sanity_eval"]["loss"] == pytest.approx(last, rel=1e-5)
    assert summaries[2]["initial_loss_sanity_eval"]["loss"] != pytest.approx(last, rel=1e-3)
    assert summaries[0]["input_psnr"] == summaries[1]["input_psnr"] == summaries[2]["input_psnr"]
    block = record_of(tmp_path / "warm")["init"]
    assert block["transfer"]["reset"] == [] and block["transfer"]["copied"] == len(start)
    check_copied(block, start, archive_weights(Path(first.run_dir)))


def test_wire_tirex_at_a_longer_lookback_copies_every_weight(world, tmp_path):
    data = {"lookback": 32, "horizon": 8, "realizations": 1, "series_length": 600, "patterns": ["ar_process"],
            "n_select": 16, "n_test": 16, "steps_per_epoch": 2}
    request = {"model": "tirex/tiny", "epochs": 1, "batch_size": 8,
               "analysis_args": {"model_analysis": False, "figures": False}}
    keras.backend.clear_session()
    train_time_series_forecasting(out=str(tmp_path / "source"), data=data, **request)
    start = started(TimeSeriesForecastingTrainer, train_time_series_forecasting, out=str(tmp_path / "run"),
                    data={**data, "lookback": 64}, init=str(tmp_path / "source"), **request)
    source = archive_weights(tmp_path / "source")
    block = record_of(tmp_path / "run")["init"]
    assert block["geometry"]["lookback"] == 32 and record_of(tmp_path / "run")["lookback"] == 64
    summary = summary_of(tmp_path / "run")  # epoch_zero_of once, from the head, not again from the task's extras
    assert (summary["epoch_zero_of"], summary["init"]) == ("init", block) and "loaded_success_value" not in summary
    assert block["transfer"]["copied"] == len(source) == len(start)
    assert all(not block["transfer"][key] for key in ("reset", "grown", "dropped", "resampled"))
    check_copied(block, start, source)


@pytest.fixture(scope="module")
def wide_megadepth_tree(tmp_path_factory):
    """The depth tests' MegaDepth-shaped tree with 48 px images, so a 42 px crop fits."""
    import tests.test_train.test_common_depth_data as depth_data

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(depth_data, "SIDE", 48)
        return depth_data.write_megadepth_tree(tmp_path_factory.mktemp("wide_megadepth"))


def test_wire_depth_at_a_larger_side_copies_every_weight_and_reads_no_official_file(
        world, wide_megadepth_tree, tmp_path, monkeypatch):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "no_weights"))
    source_dir, out = tmp_path / "source", tmp_path / "run"
    keras.backend.clear_session()
    train_depth_estimation(**depth_request(source_dir, wide_megadepth_tree, 28, model_args={"pretrained": False}))
    start = started(DepthEstimationTrainer, train_depth_estimation, **depth_request(out, wide_megadepth_tree, 42, init=str(source_dir)))
    source = archive_weights(source_dir)
    block = record_of(out)["init"]
    assert block["geometry"]["input_size"] == [28, 28] and record_of(out)["input_size"] == [42, 42]
    assert (summary_of(out)["epoch_zero_of"], summary_of(out)["init"]) == ("init", block)
    assert block["transfer"]["copied"] == len(source) == len(start)
    assert all(not block["transfer"][key] for key in ("reset", "grown", "dropped", "resampled"))
    check_copied(block, start, source)


def depth_request(out: Path, root: Path, side: int, **changes: Any) -> Dict[str, Any]:
    return {"out": str(out), "model": "depth_anything/vit_s", "batch_size": 4, "epochs": 1,
            "data": {"root": str(root), "train_split": 0.6, "n_select": 4, "n_test": 4, "layout_samples": 4,
                     "steps_per_epoch": 1, "workers": 0},
            "transform_args": {"image_size": side}, "analysis_args": {"model_analysis": False, "figures": False},
            **changes}


# ---------------------------------------------------------------------
# resume: the pins come from the run's own record; the source is never opened
# ---------------------------------------------------------------------


def test_resume_an_init_run_with_its_source_renamed_away_rebuilds_its_architecture_and_opens_no_source_file(
        vit_64, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    shutil.copytree(vit_64.out, run_dir)
    (run_dir / "results_summary.json").unlink()  # a run whose reporting died: resumed to report
    moved = vit_64.source.with_name("moved_away")
    vit_64.source.rename(moved)
    opened, rebuilt = [], {}
    gone = str(vit_64.source)

    def spied(real):
        def spy(file, *arguments, **keywords):
            if isinstance(file, (str, os.PathLike)) and os.fspath(file).startswith(gone):
                opened.append(os.fspath(file))
            return real(file, *arguments, **keywords)
        return spy

    real_restore = resume_decisions.restore_finished_fit

    def restore(model, *arguments):
        rebuilt.update({path: value.shape for path, value in relative(model).items()})
        return real_restore(model, *arguments)

    try:
        monkeypatch.setattr(builtins, "open", spied(builtins.open))
        monkeypatch.setattr(io, "open", spied(io.open))
        monkeypatch.setattr(os, "open", spied(os.open))
        monkeypatch.setattr(resume_decisions, "restore_finished_fit", restore)
        keras.backend.clear_session()
        train_image_classification(**{**vit_64.request, "out": str(run_dir), "resume_interrupted": True})
    finally:
        monkeypatch.undo()
        moved.rename(vit_64.source)
    summary = json.loads((run_dir / "results_summary.json").read_text())
    assert summary["reporting_resumed"] is True
    assert opened == []
    first = {path: value.shape for path, value in archive_weights(vit_64.out).items()}
    kernel = next(p for p in rebuilt if p.startswith("patch_embed/") and p.endswith("kernel"))
    assert rebuilt == first and rebuilt[kernel][:2] == (4, 4)  # patch 4 at 64 px: the record's pin


def test_resume_a_record_without_an_init_block_is_refused_before_anything_is_written(vit_64, tmp_path):
    run_dir = tmp_path / "run"
    shutil.copytree(vit_64.out, run_dir)
    record = record_of(run_dir)
    del record["init"]
    (run_dir / "task.json").write_text(json.dumps(record))
    before = sorted(p.name for p in run_dir.iterdir())
    with pytest.raises(ConfigError, match=r"'init': .*task\.json holds no init block"):
        train_image_classification(**{**vit_64.request, "out": str(run_dir), "resume_interrupted": True})
    assert sorted(p.name for p in run_dir.iterdir()) == before


def test_resume_a_user_value_against_the_records_pin_is_refused_before_the_claim(vit_64, tmp_path):
    run_dir = tmp_path / "run"
    shutil.copytree(vit_64.out, run_dir)
    (run_dir / "results_summary.json").unlink()
    with pytest.raises(ConfigError, match=r"'model_args\.patch_size': the source was built with patch_size=4"):
        train_image_classification(**{**vit_64.request, "out": str(run_dir), "resume_interrupted": True,
                                      "model_args": {"patch_size": 16}})


# ---------------------------------------------------------------------
# split: the source's training data against this run's held-out data; counted and marked, never refused
# ---------------------------------------------------------------------


def a_source(config: Dict[str, Any], record: Optional[Dict[str, Any]] = None, path: Path = Path("/nowhere")) -> Any:
    """A source as ``read_source`` returns it, from its ``config.json`` (only what the hooks read)."""
    return init_from.Source(path, record or classification_record(), config, "0" * 64, None, {})


def cut(n: int, seed: int, fraction: float = 0.1, cap: Optional[int] = None):
    """The ORACLE's seeded cut, written here: ``n`` rows permuted by ``seed``, capped, the first ``fraction`` held out.

    :return: ``(trained rows, selection rows)`` as sets.
    """
    order = np.random.default_rng(seed).permutation(n)[:cap]
    held = int(len(order) * fraction)
    return set(order[held:].tolist()), set(order[:held].tolist())


#: ``name: (the source's seed, data, the target's seed)``, CIFAR-10-shaped (50,000 train rows, 10,000 test rows).
CIFAR_CASES = {
    "seed_43": (42, {}, 43),
    "same_keys": (42, {}, 42),
    "capped_source": (42, {"max_samples": 20000}, 42),
    "other_dataset": (42, {"dataset": "cifar100"}, 42),
}


@pytest.mark.parametrize("case", sorted(CIFAR_CASES))
def test_split_a_cifar10_target_counts_the_sources_rows_as_the_oracle(world, tmp_path, case):
    seed, extra, target_seed = CIFAR_CASES[case]
    data = {"format": "builtin", "dataset": "cifar10", "validation_split": None, "max_samples": None, **extra}
    trainer = ImageClassificationTrainer(classifier_config(tmp_path / "run", seed=target_seed))
    fake = SimpleNamespace(n_train=45000, n_select=5000, n_test=10000, fingerprint=None)
    split = trainer.init_overlap(a_source({"seed": seed, "data": data}), trainer.config, fake)
    assert predictor._split_problems(split) == []
    flags = init_from.bias_of({"split": split})
    if case == "other_dataset":
        assert (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == ("different_data", 0, 0)
        assert not any(flags.values())
        return
    trained = cut(50000, seed, cap=extra.get("max_samples"))[0]
    expected = len(trained & cut(50000, target_seed)[1])
    assert split["n_select_overlap"] == expected and split["n_test_overlap"] == 0
    if case == "same_keys":
        assert expected == 0 and split["relation"] == "same_split" and not any(flags.values())
    else:
        assert expected > 0 and split["relation"] == "overlap" and split["unit"] == "rows"
        assert flags == {"selection_biased": True, "test_biased": False, "overlap_unknown": False}
        assert "seed=42" in split["reason"] and f"of this run's 5000 selection rows" in split["reason"]


def cifar_request(out: Path, **changes: Any) -> Dict[str, Any]:
    return {"out": str(out), "model": "vit/vit_pico", "data": {"dataset": "cifar10"}, **QUIET, **changes}


@pytest.fixture(scope="module")
def cifar_source(world, tmp_path_factory):
    """vit_pico on tiny CIFAR-10 stand-in arrays (40 train rows), seed 42, 1 epoch."""
    from tests.classification_fakes import use_tiny_raw

    out = tmp_path_factory.mktemp("cifar_source") / "run"
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        keras.backend.clear_session()
        train_image_classification(**cifar_request(out, seed=42))
    return out


def summary_of(run_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(run_dir) / "results_summary.json").read_text())


def test_split_end_to_end_an_overlap_and_an_unknown_split_are_marked_and_the_runs_finish(
        world, cifar_source, trees, tmp_path, monkeypatch):
    from tests.classification_fakes import use_tiny_raw

    use_tiny_raw(monkeypatch, n_train=40, n_test=8)
    out = tmp_path / "seed43"
    keras.backend.clear_session()
    # The verdict is the bar's (D-013: a fine-tune must beat its loaded weights), never the overlap's: the run finishes.
    assert train_image_classification(**cifar_request(out, seed=43, init=str(cifar_source))).success is summary_of(
        out)["success"]
    record = record_of(out)
    split = record["init"]["split"]
    expected = len(cut(40, 42)[0] & cut(40, 43)[1])
    assert expected > 0 and (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == (
        "overlap", expected, 0)
    flags = {key: summary_of(out)[key] for key in ("selection_biased", "test_biased", "overlap_unknown")}
    assert flags == {"selection_biased": True, "test_biased": False, "overlap_unknown": False}
    log = (out / "run.log").read_text()
    assert re.search(r"(Reached|Missed) success_threshold \S+ on \S+: best \S+ \(selection-biased\)", log), log[-2000:]
    assert "seed=42, data.validation_split=0.1, data.max_samples=null as the source's give 0" in log
    assert predictor.record_problems(record) == []
    for broken in ({**split, "n_select_overlap": None}, {**split, "relation": "leaked"}, {**split, "extra": 1}):
        assert predictor.record_problems({**record, "init": {**record["init"], "split": broken}}) != []
    # A folder target of a built-in source: no rule matches its files to the source's rows.
    out = tmp_path / "folder"
    keras.backend.clear_session()
    assert train_image_classification(**classify(str(out), trees["ten"], 32, init=str(cifar_source))).success is (
        summary_of(out)["success"])
    split = record_of(out)["init"]["split"]
    assert (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == ("unknown", None, None)
    assert summary_of(out)["overlap_unknown"] is True and summary_of(out)["selection_biased"] is False
    assert "(split overlap unknown)" in (out / "run.log").read_text()


def test_split_the_default_hook_is_unknown_with_its_reason():
    split = TaskTrainer.init_overlap(SimpleNamespace(), None, None, None)
    assert split == init_from.split_block("unknown", "this task does not rebuild a source's rows")
    assert predictor._split_problems(split) == []
    assert init_from.bias_words({"split": split}) == " (split overlap unknown)" and init_from.bias_words(None) == ""


FORECAST_DATA = {"format": "builtin", "dataset": "synthetic", "lookback": 32, "horizon": 8, "realizations": 1,
                 "series_length": 600, "noise_level": 0.1, "patterns": ["ar_process"], "select_fraction": 0.15,
                 "test_fraction": 0.15}


def test_split_forecasting_counts_windows_in_the_sources_train_segment(world, tmp_path):
    def trainer(seed: int = 42, **data: Any) -> Any:
        return TimeSeriesForecastingTrainer(validate(TimeSeriesForecastingConfig, {
            "out": str(tmp_path / "run"), "model": "tirex/tiny", "seed": seed, "data": {**FORECAST_DATA, **data}}))

    source = a_source({"seed": 42, "data": dict(FORECAST_DATA)}, forecast_record("tirex/tiny", 32, 8))
    # This run's select segment begins at 600 - 180 - 90 = 330; the source trained on rows [0, 600 - 90 - 90 = 420).
    starts = np.arange(330, 500, 7)
    fake = SimpleNamespace(bank=np.zeros((1, 600, 1)), select_windows=np.stack([0 * starts, starts], 1),
                           test_windows=np.array([[0, 520], [0, 550]]), csv_sha256=None, value_cols=None)
    longer = trainer(select_fraction=0.3)
    split = longer.init_overlap(source, longer.config, fake)
    assert (split["relation"], split["unit"]) == ("overlap", "windows")
    assert 0 < (starts < 420).sum() < len(starts)
    assert (split["n_select_overlap"], split["n_test_overlap"]) == (int((starts < 420).sum()), 0)
    same = trainer(lookback=64)
    assert same.init_overlap(source, same.config, fake)["relation"] == "same_split"
    other = trainer(seed=43)
    assert other.init_overlap(source, other.config, fake)["relation"] == "different_data"
    shared = trainer(realizations=2)
    assert shared.init_overlap(source, shared.config, fake)["relation"] == "unknown"


def test_split_depth_counts_the_sources_train_scenes_among_this_runs(world, megadepth_tree, tmp_path):
    from train.common.depth_data import deal_megadepth_scenes
    from train.common.megadepth import split_megadepth_scenes

    def trainer(split: float) -> Any:
        return DepthEstimationTrainer(validate(DepthEstimationConfig, {
            "out": str(tmp_path / "run"), "model": "depth_anything/vit_s",
            "data": {"root": str(megadepth_tree), "train_split": split}}))

    deal = deal_megadepth_scenes(str(megadepth_tree), 0.4)
    fake = SimpleNamespace(select_scenes=[name for name, _, _ in deal.select],
                           test_scenes=[name for name, _, _ in deal.test])
    source = a_source({"data": {"root": str(megadepth_tree), "train_split": 0.6}}, depth_record())
    trained = {name for name, _, _ in split_megadepth_scenes(str(megadepth_tree), 0.6)[0]}
    expected = len(trained & set(fake.select_scenes)), len(trained & set(fake.test_scenes))
    target = trainer(0.4)
    split = target.init_overlap(source, target.config, fake)
    assert sum(expected) > 0 and split["relation"] == "overlap"
    assert (split["n_select_overlap"], split["n_test_overlap"]) == expected
    same = trainer(0.6)
    assert same.init_overlap(source, same.config, fake)["relation"] == "same_split"
    moved = a_source({"data": {"root": str(tmp_path / "moved"), "train_split": 0.6}}, depth_record())
    assert target.init_overlap(moved, target.config, fake)["relation"] == "unknown"


def test_split_denoising_counts_the_sources_train_files_among_this_runs_held_out_files(world, tmp_path):
    from .test_image_denoising_task import mapping, write_images

    train, val, held = (write_images(tmp_path / name, 3, seed) for seed, name in enumerate(("train", "val", "held")))
    target = ImageDenoisingTrainer(validate(ImageDenoisingConfig, mapping(tmp_path / "run", (train, val))))
    target._held_out_sets = {"held": sorted(str(path) for path in held.iterdir())}
    fake = SimpleNamespace(val_files=tuple(sorted(str(path) for path in val.iterdir())))
    # The source trained on this run's validation and held-out directories.
    source = a_source({"data": {"train_dirs": [str(val), str(held)], "val_dirs": [str(train)]}})
    split = target.init_overlap(source, target.config, fake)
    assert (split["relation"], split["unit"], split["n_select_overlap"], split["n_test_overlap"]) == (
        "overlap", "files", 3, 3)
    assert init_from.bias_of({"split": split}) == {"selection_biased": True, "test_biased": True,
                                                  "overlap_unknown": False}
    apart = a_source({"data": {"train_dirs": [str(train)], "val_dirs": [str(val)]}})
    assert target.init_overlap(apart, target.config, fake)["n_select_overlap"] == 0
    gone = a_source({"data": {"train_dirs": [str(tmp_path / "gone")], "val_dirs": [str(val)]}})
    assert target.init_overlap(gone, target.config, fake)["relation"] == "unknown"


# ---------------------------------------------------------------------
# bar: under init a fine-tune must beat the loaded weights' epoch zero (D-013)
# ---------------------------------------------------------------------

#: ``task: (trainer, initial.metrics key, the loaded weights' score, the identity's PSNR the success value is net of,
#: the absolute bar)``; every one of the three is higher-is-better, and the loaded weights already clear the bar.
BAR_TASKS = {
    "classification": (ImageClassificationTrainer, "accuracy", 0.6, 0.0, 0.2),
    "segmentation": (ImageSegmentationTrainer, "miou", 0.6, 0.0, 0.2),
    "denoising": (ImageDenoisingTrainer, "psnr", 26.0, 20.0, 1.0),
}


#: A rate that moves no float32 weight: lr 1e-30 and no warmup (whose start rate, 1e-8, would train).
STILL = {"lr": 1e-30, "warmup_epochs": 0}


def write_sidecar(run_dir: Path, metrics: Dict[str, float]) -> Path:
    from train.common.resume_state import update_fit_state

    path = Path(run_dir) / "checkpoints" / "fit_state.json"
    update_fit_state(path, initial={"metrics": metrics, "ratio": None, "warned": None})
    return path


@pytest.mark.parametrize("task", sorted(BAR_TASKS))
def test_bar_under_init_a_fine_tune_must_beat_the_loaded_weights_epoch_zero_read_from_the_sidecar(task, tmp_path):
    trainer_class, key, score, net_of, threshold = BAR_TASKS[task]
    write_sidecar(tmp_path, {"loss": 1.0, key: score})
    config = SimpleNamespace(init="/the/source", metric_args=SimpleNamespace(success_threshold=None))
    trainer = object.__new__(trainer_class)
    held = score - net_of  # what the fine-tune reaches when it only holds the loaded weights' score
    assert trainer.bar_is_met(config, held, threshold) is True  # the absolute bar alone: today's verdict
    trainer._loaded = init_from.loaded_value(config, tmp_path, key, net_of)
    assert trainer._loaded == held
    assert trainer.bar_is_met(config, held, threshold) is False
    low = np.float32(held)
    step = np.spacing(low)
    assert trainer.bar_is_met(config, float(low + np.float32(16) * step), threshold) is False  # inside the tie band
    assert trainer.bar_is_met(config, float(low + np.float32(17) * step), threshold) is True
    assert trainer.bar_is_met(config, held - 0.1, threshold) is False
    # Without init nothing is read (not even a missing file) and the verdict is today's.
    assert init_from.loaded_value(SimpleNamespace(init=None), tmp_path / "missing", key) is None
    trainer._loaded = None
    assert trainer.bar_is_met(config, held, threshold) is True
    # A sidecar without the value under init is named, not read as zero.
    write_sidecar(tmp_path / "bare", {"loss": 1.0})
    with pytest.raises(ValueError, match=rf"initial\.metrics\.{key}"):
        init_from.loaded_value(config, tmp_path / "bare", key)


def test_bar_a_classifier_that_does_not_improve_on_its_loaded_weights_misses_and_its_resume_reads_the_sidecar(
        world, trees, vit_source, tmp_path):
    from .test_image_denoising_task import flatten

    out = tmp_path / "run"
    # lr 1e-30 with no warmup (whose start rate is 1e-8) moves no float32 weight, so the fine-tune scores what the
    # loaded weights scored at epoch zero; QUIET's success_threshold 0.0 is met by any score.
    request = classify(str(out), trees["ten"], 32, init=str(vit_source), optim_args=STILL)
    keras.backend.clear_session()
    assert train_image_classification(**request).success is False
    sidecar_path = out / "checkpoints" / "fit_state.json"
    sidecar, summary = json.loads(sidecar_path.read_text()), summary_of(out)
    loaded = sidecar["initial"]["metrics"]["accuracy"]
    assert summary["success_value"] == pytest.approx(loaded, abs=1e-6) and loaded >= summary["success_threshold"]
    log = (out / "run.log").read_text()
    assert "Evaluate of the loaded weights before fit: " in log and "Untrained evaluate BEFORE fit" not in log
    assert re.search(r"Under init the best val_accuracy \S+ does not beat the loaded weights' epoch-zero \S+", log)
    assert (summary["initial_loss_ratio"], summary["init_scale_warning"], sidecar["initial"]["ratio"]) == (None,) * 3
    assert init_from.NOT_JUDGED_UNDER_INIT in summary["notes"]
    # A reporting-only resume reads epoch zero from the sidecar, not from the restored weights: the same number is the
    # same miss (exit 1 on the command line), a lower number there is a met bar.
    for score, code in ((loaded, 1), (-1.0, 0)):
        sidecar["initial"]["metrics"]["accuracy"] = score
        sidecar_path.write_text(json.dumps(sidecar))
        (out / "results_summary.json").unlink()
        keras.backend.clear_session()
        assert cli.main(["image_classification", *flatten("", {**request, "resume_interrupted": True}, [])]) == code
        assert summary_of(out)["reporting_resumed"] is True and summary_of(out)["success"] is (code == 0)


@pytest.fixture(scope="module")
def segmenter(world, segmentation_data, tmp_path_factory):
    """convunext/tiny segmentation at 32 px, 1 epoch (layernorm only: a 1e-30 rate changes no score)."""
    out = tmp_path_factory.mktemp("segmenter") / "run"
    keras.backend.clear_session()
    train_image_segmentation(out=str(out), model="convunext/tiny", data=segmentation_data, epochs=1, batch_size=4,
                             transform_args={"image_size": [32, 32]},
                             analysis_args={"model_analysis": False, "figures": False})
    return out


def test_bar_a_segmenter_that_does_not_improve_on_its_loaded_weights_misses(world, segmentation_data, segmenter,
                                                                             tmp_path):
    out = tmp_path / "run"
    keras.backend.clear_session()
    result = train_image_segmentation(out=str(out), model="convunext/tiny", data=segmentation_data, epochs=1,
                                      batch_size=4, transform_args={"image_size": [32, 32]}, init=str(segmenter),
                                      optim_args=STILL, metric_args={"success_threshold": 0.0},
                                      analysis_args={"model_analysis": False, "figures": False})
    loaded = json.loads((out / "checkpoints" / "fit_state.json").read_text())["initial"]["metrics"]["miou"]
    summary = summary_of(out)
    assert result.success is False and summary["success"] is False
    assert summary["success_value"] == pytest.approx(loaded, abs=1e-6) and loaded >= summary["success_threshold"]


# ---------------------------------------------------------------------
# record: the summary of a fine-tuned run, and its predict and export with the source gone
# ---------------------------------------------------------------------

#: The summary keys an ``init`` run adds; a run without ``init`` has none of them.
INIT_SUMMARY_KEYS = {"init", "epoch_zero_of", "selection_biased", "test_biased", "overlap_unknown",
                     "loaded_success_value"}


def test_record_the_summary_holds_the_init_block_and_what_the_bar_compared(vit_source, vit_64):
    summary, block = summary_of(vit_64.out), record_of(vit_64.out)["init"]
    assert summary["init"] == block and block["transfer"]["copied"] > 0  # the counts are the task.json block's
    assert summary["epoch_zero_of"] == "init"
    assert {key: summary[key] for key in init_from.bias_of(block)} == init_from.bias_of(block)
    sidecar = json.loads((vit_64.out / "checkpoints" / "fit_state.json").read_text())
    assert summary["loaded_success_value"] == sidecar["initial"]["metrics"]["accuracy"]
    assert summary["initial_loss_ratio"] is None and init_from.NOT_JUDGED_UNDER_INIT in summary["notes"]
    assert not INIT_SUMMARY_KEYS & set(summary_of(vit_source))


def test_record_a_fine_tuned_run_loads_predicts_and_exports_at_its_size_with_its_source_renamed_away(vit_64, tmp_path):
    from train.api.exporting import export

    image = tmp_path / "image.png"
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (64, 64, 3)).astype(np.uint8)).save(image)
    moved = vit_64.source.with_name("moved_for_predict")
    vit_64.source.rename(moved)
    try:
        loaded = predictor.load_model(vit_64.out)
        answer = loaded.predict(str(image))
        target = export(model=loaded.model, out=tmp_path / "exported.keras", format="keras",
                        input_shape=loaded.record["input_shape"], record=loaded.record)
        beside = json.loads(Path(f"{target}.task.json").read_text())
        again = predictor.load_model(target).predict(str(image))
    finally:
        moved.rename(vit_64.source)
    assert loaded.record["input_shape"] == [64, 64, 3] and beside["init"] == record_of(vit_64.out)["init"]
    assert sorted(answer["labels"]) == HUNDRED and again["labels"] == answer["labels"]
    np.testing.assert_allclose(again["scores"], answer["scores"], atol=1e-5)


#: One record of each of the five kinds, built by the predictor's own builders.
RECORD_KINDS = {
    "image_classification": classification_record,
    "image_segmentation": segmentation_record,
    "image_denoising": lambda: predictor.build_denoising_record(
        model="bfunet/tiny", family="bfunet", variant="tiny", channels=3, size_multiple=8),
    "time_series_forecasting": forecast_record,
    "depth_estimation": depth_record,
}


@pytest.mark.parametrize("kind", sorted(RECORD_KINDS))
def test_record_every_kind_takes_a_well_formed_init_block_and_refuses_an_unknown_key_in_it(kind):
    block = {"source": "/runs/a", "file": "best_model.keras", "sha256": "0" * 64, "task": kind, "family": "f",
             "variant": "v", "geometry": {"input_shape": [32, 32, 3]}, "relation": "same_task", "pinned": {},
             "transfer": {"copied": 3, "resampled": [{"path": "t", "rule": "vit_pos_bicubic", "from": [1, 5],
                                                     "to": [1, 17]}], "reset": ["h/kernel"], "grown": [],
                          "dropped": []},
             "split": init_from.split_block("unknown", init_from.NOT_REBUILT)}
    record = {**RECORD_KINDS[kind](), "init": block}
    assert record["task"] == kind and predictor.record_problems(record) == []
    for where in ("init", "init.transfer", "init.split"):
        broken = json.loads(json.dumps(record))
        part = broken
        for name in where.split("."):
            part = part[name]
        part["extra"] = 1
        assert any(f"{where!r} has unknown key(s) 'extra'" in line for line in predictor.record_problems(broken)), where

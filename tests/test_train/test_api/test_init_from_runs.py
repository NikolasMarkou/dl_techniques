"""``init``: a run that starts from an earlier run's weights, through the task functions and the command line.

plan-2026-11-18T072904-0fd54b91 (D-002, D-003, D-004, D-008, D-011, D-021, D-033). The ``preclaim`` tests: the
source run is read before the run directory is claimed (``init_from.read_source``, called by
``TaskTrainer._resolve``), and every refusal there exits 2, names what is wrong and claims nothing; ``checkpoint``,
the key ``init`` replaced, is refused by name on the command line and in YAML and is no Python keyword any more;
the ``model_args.pretrained`` doors (a classifier takes ``false`` only; a depth run under ``init`` reads no official
weights and keeps the warm-start rate and the head's 10x).

A source here is a FAKE run directory: a ``task.json`` written by the record builders of
:mod:`train.api.predictor` (so it passes the real record check), a ``config.json`` holding the resolved
``used.model_args``, and a placeholder ``best_model.keras``. Nothing before the claim opens the archive, so no
training is needed.
"""

import dataclasses
import hashlib
import inspect
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from train.api import cli, predictor, registry
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
from train.api.tasks.image_denoising import train_image_denoising
from train.api.tasks.image_segmentation import train_image_segmentation
from train.api.tasks.pretrain import train_pretrain
from train.api.tasks.time_series_forecasting import train_time_series_forecasting
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

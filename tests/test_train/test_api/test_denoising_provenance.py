"""``task.json`` and the provenance stamps of an ``image_denoising`` run (plan d483ffab step 11, D-011, D-027).

A run of the API must be usable by the four readers that predate it, and by the API's own ``load_model``. Real fits on
the CPU of the five families (``bfcnn``, ``bfunet``, ``bfconvunext``, ``convunext_biased``, ``scunet``, all ``tiny``)
on synthetic folders (64 x 64 images, 32 px patches, two epochs, a noise range that is NOT the legacy default so a
stamp that is dropped shows), then:

* ``task.json`` is the record of the task (:data:`train.api.predictor.RECORD_KEYS`), passes its own gate, and
  ``load_model`` reads it, checks the model against it (a wrong variant and a wrong channel count are refused) and
  returns the denoising predictor (step 12 replaced the refusing stub of step 11);
* ``config.json`` holds the stamps beside the config's own fields, each equal to the value the config holds, and
  ``convnext_version`` for ``bfconvunext`` alone;
* the four legacy load paths, called for real on every run: ``eval_psnr_vs_noise.load_denoiser`` (and a PSNR from
  it), ``eval_per_pixel_uncertainty._load_eval_config`` (the run's own ``patch_size``, ``channels``,
  ``val_image_dirs``; its log line of the noise) and its ``_load_denoiser``, ``--init-from`` of ``bfunet/common.py``
  (the provenance gate and ``load_weights_from_checkpoint`` into the legacy build of the matching family: only
  ``bfcnn``, ``bfunet`` and ``bfconvunext`` have a legacy builder, so ``convunext_biased`` and ``scunet`` have no
  ``--init-from`` target), and ``DenoiserPrior.from_pretrained`` (``bfconvunext`` at ``dynamic``; the other four are
  refused there and load at ``fixed256``, which for an API model is a plain load: its input is not locked to 256);
* an unstamped copy, and a ``.keras`` file copied away from its ``config.json``, are refused by all four gates;
* removing each stamp in turn shows the path it serves fail: the gate for ``data_range``, the reader's own values
  for ``patch_size``, ``channels`` and ``val_image_dirs`` (which falls back to the legacy default directories WITHOUT
  an error, the silent one), its log line for the noise stamps, and the dynamic loader for ``convnext_version``;
* a resume writes no new stamp (``config.json`` and ``task.json`` byte-equal), and a held-out file added between a
  kill and the resume is counted by the cells (D-027: the held-out sets are not a resume key).

The tests write only under ``tmp_path``.
"""

import ast
import dataclasses
import json
import re
import shutil
import types
from pathlib import Path

import keras
import numpy as np
import pytest

import train.api.recipes as recipes
from applications.bias_free_denoiser.denoiser_prior import DenoiserPrior
from dl_techniques.utils.denoiser_provenance import require_unit_domain_checkpoint
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from train.api import registry
from train.api.config import validate
from train.api.errors import ConfigError
from train.api.predictor import (
    DATA_KEYS,
    DENOISING,
    OUTPUT_CHECKS,
    RECORD_KEYS,
    build_denoising_record,
    check_record,
    input_shape_of,
    load_model,
    read_record,
    record_differences,
)
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_denoising import ImageDenoisingConfig, ImageDenoisingTrainer, train_image_denoising
from train.bfunet import eval_per_pixel_uncertainty as pu
from train.bfunet import eval_psnr_vs_noise as ev
from train.bfunet.common import BFUnetTrainingConfig

from .test_image_denoising_task import (
    BIAS_FREE,
    FAMILIES,
    SIZE,
    TRAIN_FILES,
    VAL_FILES,
    mapping,
    read_json,
    write_images,
    write_sized,
)

#: The keys every run stamps at the top level of ``config.json`` (and ``convnext_version`` for the bias-free ConvUNext).
STAMPS = ("data_range", "patch_size", "channels", "val_image_dirs", "noise_type", "noise_sigma_min", "sigma_max_end")

#: A noise range that is not the legacy default (0 to 0.25), so that a dropped sigma stamp changes a log line.
SIGMA_MIN, SIGMA_MAX = 0.01, 0.2

#: What each family's run (the request of ``test_image_denoising_task.CASES``) must stamp and record, written by hand.
EXPECTED = {
    "bfcnn": {"noise_type": "additive", "channels": 3, "size_multiple": 1},
    "bfunet": {"noise_type": "multiplicative", "channels": 3, "size_multiple": 8},
    "bfconvunext": {"noise_type": "additive", "channels": 3, "size_multiple": 8},
    "convunext_biased": {"noise_type": "additive", "channels": 1, "size_multiple": 8},
    "scunet": {"noise_type": "composite", "channels": 3, "size_multiple": 1},
}

#: The ConvUNext recipe builds ``v1`` where the variant's table says ``v2``: the record names the difference.
OVERRIDES = {"bfconvunext": {"convnext_version": "v1"}}


# ---------------------------------------------------------------------
# Real runs, one per family, shared by the module
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    root = tmp_path_factory.mktemp("provenance_images")
    return write_images(root / "train", TRAIN_FILES, 0), write_images(root / "val", VAL_FILES, 1)


@dataclasses.dataclass(frozen=True)
class Run:
    family: str
    asked: dict
    dir: Path
    val_dir: Path

    @property
    def keras_file(self) -> Path:
        return self.dir / "final_model.keras"

    @property
    def config(self) -> dict:
        return read_json(self.dir / "config.json")

    @property
    def expected(self) -> dict:
        return EXPECTED[self.family]


def asked_run(out, folders, family, **changes):
    asked = mapping(out, folders, family, **changes)
    asked["data"].update(noise_sigma_min=SIGMA_MIN, noise_sigma_max=SIGMA_MAX, noise_sigma_max_start=0.05)
    asked["analysis_args"] = {"figures": False, "model_analysis": False}
    return asked


@pytest.fixture(scope="module")
def runs(world, folders, tmp_path_factory):
    """``runs(family)``: the real two-epoch run of that family, made on first use and kept for the module."""
    done = {}

    def of(family):
        if family not in done:
            out = tmp_path_factory.mktemp("provenance_" + family) / "run"
            asked = asked_run(out, folders, family)
            result = train_image_denoising(**asked)
            done[family] = Run(family, asked, Path(result.run_dir), folders[1])
        return done[family]

    return of


@pytest.fixture(scope="module", params=FAMILIES)
def finished(request, runs):
    return runs(request.param)


@pytest.fixture(scope="module", params=BIAS_FREE)
def bias_free(request, runs):
    return runs(request.param)


def copy_without(run: Run, root: Path, *dropped, config=True) -> Path:
    """Copy the final model into a new directory beside a ``config.json`` lacking ``dropped`` (or none at all)."""
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy(run.keras_file, root / "final_model.keras")
    if config:
        payload = {key: value for key, value in run.config.items() if key not in dropped}
        (root / "config.json").write_text(json.dumps(payload))
    return root / "final_model.keras"


# ---------------------------------------------------------------------
# task.json
# ---------------------------------------------------------------------


def test_the_task_json_is_the_denoising_record_of_the_run(finished):
    record = read_json(finished.dir / "task.json")
    overrides = OVERRIDES.get(finished.family)
    assert list(record) == [*RECORD_KEYS[DENOISING], *(["variant_overrides"] if overrides else [])]
    assert record["schema_version"] == 2 and record["task"] == DENOISING and record["output"] == "images"
    assert record["model"] == f"{finished.family}/tiny" and record["family"] == finished.family
    assert record["variant"] == "tiny" and record["data_range"] == "[0,1]"
    assert record["channels"] == finished.expected["channels"]
    assert record["size_multiple"] == finished.expected["size_multiple"]
    assert record.get("variant_overrides") == overrides
    assert not {"class_names", "normalize", "mean", "std", "input_size", "input_shape"} & set(record)
    check_record(record, "task.json")
    assert read_record(finished.dir / "task.json") == record


def test_load_model_reads_the_record_and_returns_the_denoising_predictor(finished):
    # Step 12 replaced the refusing stub: the record passed its gate, the model is the family and variant it names,
    # takes its channels and answers with one image tensor, and `load_model` hands back the predictor of the task.
    from train.api.denoising_predictor import DenoisingPredictor

    loaded = load_model(finished.dir)
    assert type(loaded) is DenoisingPredictor and loaded.record == read_record(finished.dir / "task.json")


def test_load_model_refuses_a_record_that_is_not_the_model(finished, tmp_path):
    record = read_json(finished.dir / "task.json")
    wrong_channels = 1 if finished.expected["channels"] == 3 else 3
    cases = {
        "channels": ({**record, "channels": wrong_channels}, rf"does not take the channels {wrong_channels} it names"),
        "variant": ({**record, "variant": "small", "model": f"{finished.family}/small"}, "is not that variant"),
    }
    for name, (changed, message) in cases.items():
        copy = tmp_path / name
        copy.mkdir()
        shutil.copy(finished.dir / "best_model.keras", copy / "best_model.keras")
        (copy / "task.json").write_text(json.dumps(changed))
        with pytest.raises(ConfigError, match=message):
            load_model(copy)


# ---------------------------------------------------------------------
# The record's own checks (no run)
# ---------------------------------------------------------------------


def record(**changes):
    value = build_denoising_record(model="bfunet/tiny", family="bfunet", variant="tiny", channels=3, size_multiple=8)
    value.update(changes)
    return value


def test_a_built_record_passes_and_each_bad_value_is_named():
    check_record(record(), "task.json")
    cases = {
        "channels": [2, 0, True, "3", None],
        "size_multiple": [0, -8, True, "8", 2.0],
        "data_range": ["[-0.5,0.5]", "[0,255]", None],
        "output": ["masks", "logits", ""],
    }
    for key, values in cases.items():
        for value in values:
            with pytest.raises(ConfigError, match=re.escape(f"'{key}' must")):
                check_record(record(**{key: value}), "task.json")


def test_a_denoising_record_takes_no_other_tasks_keys_and_misses_none_of_its_own():
    with pytest.raises(ConfigError, match=r"unknown key\(s\) 'class_names'"):
        check_record(record(class_names=["a"]), "task.json")
    with pytest.raises(ConfigError, match=r"unknown key\(s\) 'input_size'"):
        check_record(record(input_size=[8, 8]), "task.json")
    for key in RECORD_KEYS[DENOISING][2:]:  # without the version and the task word, which are read before the keys
        broken = record()
        del broken[key]
        with pytest.raises(ConfigError, match=rf"missing key\(s\) '{key}'"):
            check_record(broken, "task.json")


def test_the_zero_image_is_a_multiple_of_the_stride_and_the_output_check_wants_one_image_tensor():
    assert input_shape_of(record(size_multiple=1)) == (32, 32, 3)
    assert input_shape_of(record(size_multiple=8)) == (32, 32, 3)
    assert input_shape_of(record(size_multiple=24)) == (48, 48, 3)
    assert input_shape_of(record(size_multiple=64, channels=1)) == (64, 64, 1)
    check = OUTPUT_CHECKS[DENOISING]
    check(types.SimpleNamespace(shape=(1, 32, 32, 3)), record())
    for bad in (types.SimpleNamespace(shape=(1, 32, 32, 1)), types.SimpleNamespace(shape=(1, 16, 32, 3)),
                types.SimpleNamespace(shape=(1, 3)), [types.SimpleNamespace(shape=(1, 32, 32, 3))] * 2):
        with pytest.raises(ConfigError, match="the file and the model are not from one run"):
            check(bad, record())


def test_a_resume_compares_the_channels_and_the_domain_and_not_the_held_out_sets():
    assert DATA_KEYS[DENOISING] == ("channels", "data_range")  # D-027
    assert set(DATA_KEYS[DENOISING]) <= set(RECORD_KEYS[DENOISING])
    assert not any("test" in key or "held" in key for key in RECORD_KEYS[DENOISING])
    assert record_differences(record(), record()) == []
    assert record_differences(record(), record(channels=1)) == ["channels: 3 -> 1"]
    assert record_differences(record(), record(data_range="x")) == ['data_range: "[0,1]" -> "x"']
    # Not data: how the model is cut and how the predictor answers.
    assert record_differences(record(), record(size_multiple=16, output="other", variant="small")) == []


# ---------------------------------------------------------------------
# The stamps of config.json
# ---------------------------------------------------------------------


def test_the_stamps_equal_the_config_and_sit_at_the_top_level(finished):
    config = finished.config
    data, transform = config["data"], config["transform_args"]
    assert config["data_range"] == "[0,1]"
    assert config["patch_size"] == transform["patch_size"] == SIZE
    assert config["channels"] == transform["channels"] == finished.expected["channels"]
    assert config["val_image_dirs"] == data["val_dirs"] == [str(finished.val_dir)]
    assert config["noise_type"] == data["noise_kind"] == finished.expected["noise_type"]
    assert config["noise_sigma_min"] == data["noise_sigma_min"] == SIGMA_MIN
    assert config["sigma_max_end"] == data["noise_sigma_max"] == SIGMA_MAX
    if finished.family == "bfconvunext":
        assert config["convnext_version"] == config["used"]["model_args"]["convnext_version"] == "v1"
    else:
        assert "convnext_version" not in config
    # The config's own fields are all still there, unchanged by the stamps.
    assert {"out", "model", "data", "epochs", "transform_args", "optim_args", "used"} <= set(config)
    assert config["model"] == f"{finished.family}/tiny" and config["epochs"] == 2


def stamps_of(family, **model_args):
    """What ``config_stamps`` returns for a validated request of ``family``, with no run."""
    request = asked_run("unused", (Path("train"), Path("val")), family)
    request["model_args"] = model_args
    config = validate(ImageDenoisingConfig, request)
    return ImageDenoisingTrainer(config).config_stamps(config, registry.MODELS[family]), config


def test_the_stamp_keys_are_the_same_everywhere_and_never_collide_with_a_config_field(world):
    for family in FAMILIES:
        stamps, config = stamps_of(family)
        extra = {"convnext_version"} if family == "bfconvunext" else set()
        assert set(stamps) == set(STAMPS) | extra, family
        assert not set(stamps) & {field.name for field in dataclasses.fields(config)}, family


def test_the_convnext_version_stamped_is_the_resolved_one(world):
    assert stamps_of("bfconvunext")[0]["convnext_version"] == "v1"  # the recipe's
    assert stamps_of("bfconvunext", convnext_version="v2")[0]["convnext_version"] == "v2"  # the caller's wins
    for family in ("bfcnn", "bfunet", "convunext_biased", "scunet"):
        assert "convnext_version" not in stamps_of(family)[0], family


# ---------------------------------------------------------------------
# The four legacy readers, for real
# ---------------------------------------------------------------------


class Said:
    """Stands in for the module logger of a script and keeps what it said, formatted."""

    def __init__(self):
        self.lines = []

    def _say(self, level, message, *arguments):
        self.lines.append((level, message % arguments if arguments else message))

    def info(self, message, *arguments, **keywords):
        self._say("info", message, *arguments)

    def warning(self, message, *arguments, **keywords):
        self._say("warning", message, *arguments)

    def text(self, level=None):
        return "\n".join(line for tag, line in self.lines if level in (None, tag))


def eval_config_of(keras_file, monkeypatch):
    said = Said()
    monkeypatch.setattr(pu, "logger", said)
    return pu._load_eval_config(str(keras_file)), said


def test_eval_psnr_vs_noise_loads_the_run_and_gives_a_psnr(finished):
    model = ev.load_denoiser(str(finished.keras_file))
    assert model.input_shape == (None, None, None, finished.expected["channels"])
    cfg = ev.EvalConfig(
        models={"run": str(finished.keras_file)}, datasets={"val": [str(finished.val_dir)]}, sigmas_255=[25.0],
        num_samples=VAL_FILES, patch_size=SIZE, channels=finished.expected["channels"], batch_size=VAL_FILES,
        clip_noise=False,
    )
    paths = sorted(str(path) for path in finished.val_dir.glob("*.png"))
    rows = ev.evaluate_dataset({"run": model}, cfg, "val", paths, np.random.RandomState(0))
    assert len(rows) == 1 and rows[0]["n"] == VAL_FILES
    assert np.isfinite(rows[0]["psnr_mean"]) and np.isfinite(rows[0]["input_psnr_mean"])


def test_eval_per_pixel_uncertainty_reads_the_runs_own_patch_channels_and_validation_directories(finished, monkeypatch):
    legacy_default = BFUnetTrainingConfig()
    cfg, said = eval_config_of(finished.keras_file, monkeypatch)
    assert cfg.patch_size == SIZE != legacy_default.patch_size
    assert cfg.channels == finished.expected["channels"]
    assert cfg.val_image_dirs == [str(finished.val_dir)] != legacy_default.val_image_dirs
    # The noise stamps are read for the log line, and a non-additive run is warned about.
    assert f"sigma range {SIGMA_MIN:.3f}..{SIGMA_MAX:.3f}" in said.text("info")
    assert ("is not 'additive'" in said.text("warning")) == (finished.expected["noise_type"] != "additive")
    model = pu._load_denoiser(str(finished.keras_file))
    assert model.input_shape[-1] == finished.expected["channels"]


def legacy_build(family):
    if family == "bfcnn":
        from train.bfunet import train_bfcnn_denoiser as legacy

        config = legacy.BFCNNTrainingConfig(variant="tiny", patch_size=SIZE, channels=3)
    elif family == "bfunet":
        from train.bfunet import train_unet_denoiser as legacy

        config = legacy.TrainingConfig(variant="tiny", patch_size=SIZE, channels=3, final_projection_groups=1)
    else:
        from train.bfunet import train_convunext_denoiser as legacy

        config = legacy.TrainingConfig(variant="tiny", patch_size=SIZE, channels=3, final_projection_groups=1)
    return legacy.build_model(config)


def test_init_from_loads_every_layer_of_the_matching_legacy_build(bias_free):
    # `bfunet/common.train` does exactly these two calls for `--init-from` (the gate, then a by-name transfer of every
    # layer, `skip_prefixes=()`), and refuses 0 loaded layers and any shape mismatch; the stronger arms are here.
    model = legacy_build(bias_free.family)
    require_unit_domain_checkpoint(str(bias_free.keras_file))
    report = load_weights_from_checkpoint(model, ckpt_path=str(bias_free.keras_file), skip_prefixes=())
    weighted = [layer for layer in model.layers if layer.weights]
    assert len(weighted) > 3 and len(report.loaded) == len(weighted)
    assert report.missing_in_source == [] and report.shape_mismatch == [] and report.unused_in_source == []
    source = {layer.name: layer for layer in keras.models.load_model(str(bias_free.keras_file), compile=False).layers}
    for layer in weighted:
        assert all(np.array_equal(a, b) for a, b in zip(layer.get_weights(), source[layer.name].get_weights())), layer.name


def forward(prior, height, width, channels):
    x = np.random.default_rng(0).uniform(0, 1, (1, height, width, channels)).astype("float32")
    out = np.asarray(prior.denoise(x))
    assert out.shape == x.shape and np.all(np.isfinite(out))


def test_denoiser_prior_loads_bfconvunext_dynamic_and_every_family_fixed256(finished):
    channels = finished.expected["channels"]
    fixed = DenoiserPrior.from_pretrained(str(finished.keras_file), resolution="fixed256")
    forward(fixed, 32, 32, channels)
    from_directory = DenoiserPrior.from_pretrained(str(finished.dir), resolution="fixed256")  # best_model.keras
    forward(from_directory, 32, 32, channels)
    if finished.family == "bfconvunext":
        dynamic = DenoiserPrior.from_pretrained(str(finished.keras_file), resolution="dynamic")
        assert dynamic.model.input_shape == (None, None, None, channels)
        forward(dynamic, 32, 32, channels)
        forward(dynamic, 40, 48, channels)  # a size the training patch never had
    else:
        with pytest.raises(ValueError, match="resolution='dynamic' is not available"):
            DenoiserPrior.from_pretrained(str(finished.keras_file), resolution="dynamic")


# ---------------------------------------------------------------------
# An unstamped copy is refused; removing a stamp breaks the path it serves
# ---------------------------------------------------------------------

GATES = {
    "eval_psnr_vs_noise.load_denoiser": lambda path: ev.load_denoiser(path),
    "eval_per_pixel_uncertainty._load_denoiser": lambda path: pu._load_denoiser(path),
    "DenoiserPrior.from_pretrained": lambda path: DenoiserPrior.from_pretrained(path, resolution="fixed256"),
    "--init-from": lambda path: require_unit_domain_checkpoint(path),
}


def test_a_stamped_copy_passes_every_gate_the_control(finished, tmp_path):
    path = str(copy_without(finished, tmp_path / "stamped"))
    for name, gate in GATES.items():
        gate(path)  # no exception


def test_an_unstamped_copy_and_a_keras_file_without_its_config_are_refused_by_every_gate(finished, tmp_path):
    unstamped = str(copy_without(finished, tmp_path / "unstamped", *STAMPS, "convnext_version"))
    alone = str(copy_without(finished, tmp_path / "alone", config=False))
    for name, gate in GATES.items():
        with pytest.raises(ValueError, match="records NO data_range key"):
            gate(unstamped)
        with pytest.raises(ValueError, match="config.json is missing or unreadable"):
            gate(alone)


def test_dropping_data_range_alone_is_refused_by_every_gate(finished, tmp_path):
    path = str(copy_without(finished, tmp_path / "no_range", "data_range"))
    for name, gate in GATES.items():
        with pytest.raises(ValueError, match="records NO data_range key"):
            gate(path)


@pytest.mark.parametrize("key", ["patch_size", "channels", "val_image_dirs"])
def test_dropping_a_geometry_stamp_makes_the_reader_fall_back_to_its_own_defaults_without_an_error(
    finished, tmp_path, monkeypatch, key
):
    own = {"patch_size": SIZE, "channels": finished.expected["channels"], "val_image_dirs": [str(finished.val_dir)]}[key]
    default = getattr(BFUnetTrainingConfig(), key)
    stamped, _ = eval_config_of(finished.keras_file, monkeypatch)
    assert getattr(stamped, key) == own
    path = copy_without(finished, tmp_path / f"no_{key}", key)
    dropped, said = eval_config_of(path, monkeypatch)  # no error: the failure is silent
    assert getattr(dropped, key) == default
    if own != default:  # the arm is live where the run's value is not the legacy one (channels: convunext_biased)
        assert getattr(dropped, key) != own
    assert "using config defaults" not in said.text()  # it read a config.json, it just found no such key


@pytest.mark.parametrize("key", ["noise_sigma_min", "sigma_max_end"])
def test_dropping_a_sigma_stamp_changes_the_range_the_reader_reports(finished, tmp_path, monkeypatch, key):
    _, said = eval_config_of(finished.keras_file, monkeypatch)
    assert f"sigma range {SIGMA_MIN:.3f}..{SIGMA_MAX:.3f}" in said.text("info")
    _, dropped = eval_config_of(copy_without(finished, tmp_path / f"no_{key}", key), monkeypatch)
    assert f"sigma range {SIGMA_MIN:.3f}..{SIGMA_MAX:.3f}" not in dropped.text("info")
    assert ("sigma range 0.000..0.200" if key == "noise_sigma_min" else f"sigma range {SIGMA_MIN:.3f}..0.250") \
        in dropped.text("info")


def test_dropping_noise_type_silences_the_warning_for_a_run_that_is_not_additive(finished, tmp_path, monkeypatch):
    warned = "is not 'additive'" in eval_config_of(finished.keras_file, monkeypatch)[1].text("warning")
    assert warned == (finished.expected["noise_type"] != "additive")
    dropped = eval_config_of(copy_without(finished, tmp_path / "no_noise_type", "noise_type"), monkeypatch)[1]
    assert "is not 'additive'" not in dropped.text("warning")


def test_dropping_convnext_version_makes_denoiser_prior_refuse_the_dynamic_loader(runs, tmp_path):
    run = runs("bfconvunext")
    DenoiserPrior.from_pretrained(str(copy_without(run, tmp_path / "kept")), resolution="dynamic")
    path = str(copy_without(run, tmp_path / "no_version", "convnext_version"))
    with pytest.raises(ValueError, match="resolution='dynamic' is not available for a 'cliffordunet' checkpoint"):
        DenoiserPrior.from_pretrained(path, resolution="dynamic")
    DenoiserPrior.from_pretrained(path, resolution="fixed256")  # the saved graph still loads


def test_eval_per_pixel_uncertainty_clips_its_noise_always_a_stated_limit_on_unclipped_runs():
    # `data.noise_clip` defaults to false here; this script has no such switch. If it gains one, the README's and the
    # task docstring's statement of the limit must change with this test.
    calls = [
        node for node in ast.walk(ast.parse(Path(pu.__file__).read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "add_awgn"
    ]
    clips = [[keyword.value.value for keyword in call.keywords if keyword.arg == "clip"] for call in calls]
    assert len(calls) >= 2 and all(clip == [True] for clip in clips), clips


# ---------------------------------------------------------------------
# A resume writes no new stamp, and the held-out sets are not a resume key
# ---------------------------------------------------------------------


class Kill(keras.callbacks.Callback):
    """Raise at the start of epoch 1: a stand-in for a kill after the first epoch (as ``test_resume_interrupted``)."""

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 1:
            raise RuntimeError("killed")


class KillAtEpoch1(ImageDenoisingTrainer):
    """The real task with the killer after its own callbacks."""

    def extra_callbacks(self, config, data):
        return [*super().extra_callbacks(config, data), Kill()]


def test_a_resume_writes_no_new_stamp_and_counts_a_held_out_file_added_after_the_kill(world, folders, tmp_path):
    held = write_sized(tmp_path / "held", [(40, 40), (48, 40)], 7)
    out = tmp_path / "resumed" / "run"
    asked = asked_run(out, folders, "bfcnn", epochs=3)
    asked["data"].update(noise_kind="additive", noise_clip=False, test_dirs=[f"alpha={held}"])
    asked["data"].pop("noise_composite_ratio", None)
    with pytest.raises(RuntimeError, match="killed"):
        KillAtEpoch1(validate(ImageDenoisingConfig, asked)).run()
    names = ("config.json", "task.json", "config_requested.json")
    files = {name: (out / name).read_bytes() for name in names}
    written = {name: (out / name).stat().st_mtime_ns for name in names}
    assert json.loads(files["config.json"])["data_range"] == "[0,1]" and not (out / "results_summary.json").exists()

    write_sized(held, [(40, 40), (48, 40), (56, 48)], 7)  # the two files again (same bytes) and one more
    result = train_image_denoising(**{**asked, "resume_interrupted": True})

    assert {name: (out / name).read_bytes() for name in names} == files  # no stamp rewritten, no record rewritten
    # Equal bytes alone would pass a resume that rewrites the same config: the files were not even touched.
    assert {name: (out / name).stat().st_mtime_ns for name in names} == written
    summary = read_json(result.summary_path)
    assert set(summary["held_out"]["alpha"]) == {"15", "25", "50"}
    assert all(cell["n"] == 3 for cell in summary["held_out"]["alpha"].values())
    assert read_record(out / "task.json")["task"] == DENOISING

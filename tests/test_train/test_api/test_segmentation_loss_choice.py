"""``loss_args`` of the ``image_segmentation`` task: the choice of loss, its knobs and what a run says (D-014).

* ``loss_args.name`` selects ``ce`` (the default, Keras' own), ``dice``, ``ce_dice`` or ``focal``; the
  function and ``python -m train`` write the same ``config.json`` for each, and the compiled loss is the
  class that name stands for, built with the void value of the data and the knobs given;
* a knob the chosen loss does not read is REFUSED by its dotted key before the run directory is claimed
  (the function raises, the command exits 2), and so is a knob out of range or an unknown name;
* a real fit with each of the three new losses on a void fixture finishes, the summary's
  ``init_scale_warning`` is ``null`` with a note that NAMES THE LOSS (and does not claim unset BatchNorm
  statistics, which ConvUNext does not have), while ``ce`` keeps its ln(C) verdict;
* ``LOSS_KNOBS`` (the config's table) and the builders of ``segmentation_run`` (the loss's) name the same
  four losses and the same knobs;
* the AccUNet recipe still trains ``ce`` and its notice names ``loss_args.name=ce_dice`` and what the
  legacy ``combo`` is.

The tests write only under ``tmp_path``.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import train.api.recipes as recipes
from train.api import cli, registry
from train.api.errors import ConfigError, ConfigValidationError
from train.api.segmentation_args import LOSS_KNOBS, SegLossArgs
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_segmentation import ImageSegmentationTrainer, train_image_segmentation
from train.common import segmentation_run

from . import _segmentation_fixtures as fx
from .test_the_cli_exit_codes import run_command

VOID = 255
SIZE = 32


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
    """Pairs whose masks are 14 columns of class 0, 10 of class 1, 4 of class 2 and 4 void columns."""
    root = tmp_path_factory.mktemp("loss_choice")
    for prefix, count in (("train_", 24), ("test_", 6)):
        for index in range(count):
            mask = np.zeros((SIZE, SIZE), np.uint8)
            mask[:, 14:24] = 1
            mask[:, 24:28] = 2
            mask[:, 28:] = VOID
            fx.write_pair(root / f"{prefix}images", root / f"{prefix}masks", f"p{index:03d}", mask)
    return root


def data_of(folders):
    return {
        "format": "folder",
        "train_images": str(folders / "train_images"),
        "train_masks": str(folders / "train_masks"),
        "test_images": str(folders / "test_images"),
        "test_masks": str(folders / "test_masks"),
        "num_classes": 3,
        "ignore_index": VOID,
        "validation_split": 0.25,
    }


def read_json(path):
    return json.loads(Path(path).read_text())


NEW_LOSSES = ["dice", "ce_dice", "focal"]


# ---------------------------------------------------------------------
# The function and the command write one config; the loss is the one named
# ---------------------------------------------------------------------

class StopAtBuild(Exception):
    """Raised in place of ``build_model``: ``config.json`` is already on disk by then."""


@pytest.mark.parametrize(
    "name, extra",
    [("ce", {}), ("dice", {"dice_smooth": 1e-3}), ("ce_dice", {}), ("focal", {"focal_gamma": 1.5})],
)
def test_the_function_and_the_cli_write_the_same_config_bytes_for_each_loss(name, extra, world, folders, tmp_path):
    def stop(self, config, context, data):
        raise StopAtBuild

    data = data_of(folders)
    tokens = [f"data.{key}={value}" for key, value in data.items()]
    loss_tokens = [f"loss_args.name={name}", *[f"loss_args.{key}={value}" for key, value in extra.items()]]
    out = tmp_path / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        patch.setattr(ImageSegmentationTrainer, "build_model", stop)
        with pytest.raises(StopAtBuild):
            train_image_segmentation(
                out=str(out), model="convunext/tiny", data=data, epochs=1, batch_size=8, seed=7,
                transform_args={"image_size": SIZE}, loss_args={"name": name, **extra},
                analysis_args={"model_analysis": False, "figures": False},
            )
        out.rename(tmp_path / "function")
        with pytest.raises(StopAtBuild):
            cli.main([
                "image_segmentation", f"out={out}", "model=convunext/tiny", *tokens, "epochs=1",
                "batch_size=8", "seed=7", f"transform_args.image_size={SIZE}", *loss_tokens,
                "analysis_args.model_analysis=false", "analysis_args.figures=false",
            ])
        out.rename(tmp_path / "cli")
    for file in ("config.json", "config_requested.json"):
        assert (tmp_path / "function" / file).read_bytes() == (tmp_path / "cli" / file).read_bytes(), file
    assert read_json(tmp_path / "cli" / "config.json")["loss_args"] == {
        "name": name, "dice_smooth": extra.get("dice_smooth"), "focal_gamma": extra.get("focal_gamma")}


@pytest.mark.parametrize(
    "name, knobs, class_name, attributes",
    [
        ("ce", {}, "SparseCategoricalCrossentropy", {"ignore_class": VOID}),
        ("dice", {}, "SparseDiceLoss", {"ignore_index": VOID, "smooth": 1e-5}),
        ("dice", {"dice_smooth": 0.5}, "SparseDiceLoss", {"ignore_index": VOID, "smooth": 0.5}),
        ("ce_dice", {"dice_smooth": 1e-4}, "SparseCeDiceLoss", {"ignore_index": VOID, "smooth": 1e-4}),
        ("focal", {}, "SparseFocalLoss", {"ignore_index": VOID, "gamma": 2.0}),
        ("focal", {"focal_gamma": 0.0}, "SparseFocalLoss", {"ignore_index": VOID, "gamma": 0.0}),
    ],
)
def test_the_compiled_loss_is_the_class_the_name_stands_for_with_the_void_value_and_the_knobs(
    name, knobs, class_name, attributes, world, folders
):
    config = _config(folders, loss_args={"name": name, **knobs})
    loss, _ = ImageSegmentationTrainer(config).build_loss_and_metrics(
        config, type("Data", (), {"ignore_index": VOID, "num_classes": 3})()
    )
    assert type(loss).__name__ == class_name
    for key, value in attributes.items():
        assert getattr(loss, key) == value, key


def _config(folders, **changes):
    from train.api.config import validate
    from train.api.tasks.image_segmentation import ImageSegmentationConfig

    mapping = {
        "out": "never", "model": "convunext/tiny", "data": data_of(folders), "epochs": 1,
        "transform_args": {"image_size": SIZE},
    }
    mapping.update(changes)
    return validate(ImageSegmentationConfig, mapping)


# ---------------------------------------------------------------------
# What is refused, and how
# ---------------------------------------------------------------------

UNREAD = [
    ("ce", "dice_smooth", 1e-3), ("ce", "focal_gamma", 2.0),
    ("dice", "focal_gamma", 2.0), ("ce_dice", "focal_gamma", 2.0),
    ("focal", "dice_smooth", 1e-3),
]


@pytest.mark.parametrize("name, knob, value", UNREAD)
def test_a_knob_the_loss_does_not_read_is_refused_by_its_key_before_the_run_is_claimed(
    name, knob, value, world, folders, tmp_path
):
    out = tmp_path / "never"
    with pytest.raises(ConfigValidationError, match=rf"loss_args\.{knob}.*does not read it"):
        train_image_segmentation(
            out=str(out), model="convunext/tiny", data=data_of(folders), epochs=1,
            transform_args={"image_size": SIZE}, loss_args={"name": name, knob: value},
        )
    assert not out.exists()


@pytest.mark.parametrize("name, knob, value", UNREAD)
def test_the_command_exits_two_with_one_line_for_an_unread_knob_and_writes_nothing(
    name, knob, value, world, folders, tmp_path, capsys
):
    out = tmp_path / "never"
    data = data_of(folders)
    code, _, err = run_command(
        ["image_segmentation", f"out={out}", "model=convunext/tiny", *[f"data.{k}={v}" for k, v in data.items()],
         "epochs=1", f"transform_args.image_size={SIZE}", f"loss_args.name={name}", f"loss_args.{knob}={value}"],
        capsys,
    )
    assert code == 2
    assert f"loss_args.{knob}" in err and "Traceback" not in err
    assert not out.exists()


@pytest.mark.parametrize(
    "loss_args, match",
    [
        ({"name": "huber"}, r"loss_args\.name.*ce"),
        ({"name": "dice", "dice_smooth": 0.0}, r"loss_args\.dice_smooth.*positive"),
        ({"name": "ce_dice", "dice_smooth": -1.0}, r"loss_args\.dice_smooth.*positive"),
        ({"name": "focal", "focal_gamma": -0.1}, r"loss_args\.focal_gamma.*at least 0"),
    ],
)
def test_an_unknown_name_and_a_knob_out_of_range_are_refused(loss_args, match, world, folders, tmp_path):
    out = tmp_path / "never"
    with pytest.raises((ConfigError, ConfigValidationError), match=match):
        train_image_segmentation(
            out=str(out), model="convunext/tiny", data=data_of(folders), epochs=1,
            transform_args={"image_size": SIZE}, loss_args=loss_args,
        )
    assert not out.exists()


def test_the_default_is_the_cross_entropy_with_no_knob():
    default = SegLossArgs()
    assert (default.name, default.dice_smooth, default.focal_gamma) == ("ce", None, None)


# ---------------------------------------------------------------------
# The table of the config and the builders of the loss say the same thing
# ---------------------------------------------------------------------

def test_the_knob_table_and_the_loss_builders_name_the_same_losses_and_knobs():
    builders = segmentation_run._LOSS_BUILDERS
    assert set(builders) == set(LOSS_KNOBS)
    for name, builder in builders.items():
        taken = tuple(p for p in inspect.signature(builder).parameters if p != "ignore_index")
        assert taken == LOSS_KNOBS[name], name
    fields = {"dice_smooth", "focal_gamma"}
    assert {knob for knobs in LOSS_KNOBS.values() for knob in knobs} == fields
    assert fields <= {f for f in SegLossArgs.__dataclass_fields__}


def test_the_function_refuses_an_unknown_loss_and_an_unread_knob_in_python_too():
    with pytest.raises(ValueError, match="unknown segmentation loss 'huber'"):
        segmentation_run.segmentation_loss("huber", None)
    with pytest.raises(TypeError, match="focal_gamma"):
        segmentation_run.segmentation_loss("dice", None, focal_gamma=2.0)


# ---------------------------------------------------------------------
# A real fit: what the summary says per loss
# ---------------------------------------------------------------------

@pytest.fixture(scope="module", params=["ce", *NEW_LOSSES])
def finished(world, folders, request, tmp_path_factory):
    out = tmp_path_factory.mktemp("fit_" + request.param) / "run"
    result = train_image_segmentation(
        out=str(out), model="convunext/tiny", data=data_of(folders), epochs=2, batch_size=8,
        transform_args={"image_size": SIZE}, loss_args={"name": request.param},
        analysis_args={"model_analysis": False, "figures": False},
    )
    return request.param, result, read_json(result.summary_path), read_json(Path(result.run_dir) / "training_history.json")


def test_a_fit_with_each_loss_finishes_with_finite_numbers_and_a_summary_that_names_the_loss(finished):
    name, result, summary, history = finished
    assert np.isfinite(history["loss"]).all() and np.isfinite(history["val_loss"]).all()
    assert read_json(Path(result.run_dir) / "config.json")["loss_args"]["name"] == name
    notes = " ".join(summary["notes"])
    if name == "ce":
        assert summary["init_scale_warning"] is False
        assert "ln(C)" in notes and "cross-entropy per valid pixel" in notes
        assert "is the `" not in notes
    else:
        assert summary["init_scale_warning"] is None
        assert f"`{name}`" in notes
        assert "not compared with ln(C)" in notes
        # ConvUNext has no BatchNorm: the base reason must NOT be given for a null that has another one.
        assert "BatchNorm" not in notes
        assert summary["initial_loss_ratio"] == pytest.approx(
            summary["initial_loss_sanity_eval"]["loss"] / np.log(3), abs=1e-9)


def test_a_non_finite_initial_loss_is_still_refused_for_a_new_loss(world, folders, tmp_path, monkeypatch):
    import train.common.classification_run as classification_run

    monkeypatch.setattr(classification_run, "evaluate_dict", lambda model, dataset: {"loss": float("nan")})
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Initial validation loss is nan"):
        train_image_segmentation(
            out=str(out), model="convunext/tiny", data=data_of(folders), epochs=1, batch_size=8,
            transform_args={"image_size": SIZE}, loss_args={"name": "dice"},
            analysis_args={"model_analysis": False, "figures": False},
        )


def test_the_accunet_recipe_trains_ce_and_its_notice_names_ce_dice_and_what_combo_is(world):
    from train.api.segmentation_models import ACCUNET

    assert "loss_args" not in ACCUNET
    stub = object.__new__(ImageSegmentationTrainer)
    notices = stub.recipe_notices(type("F", (), {"dataset": "oxford_pet"})(), "accunet")
    text = " ".join(text for _, text in notices)
    assert "loss_args.name=ce_dice" in text and "combo" in text and "0.5 dice" in text
    assert SegLossArgs().name == "ce"

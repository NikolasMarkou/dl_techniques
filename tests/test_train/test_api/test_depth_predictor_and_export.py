"""The ``depth_estimation`` record, ``DepthPredictor``, its rows in the predictor tables and ``predict``.

First part (Step 9 of plan-2026-11-17T110935-9c7b9096): a depth run writes the fifth record kind (``input_size``,
``channels``, ``normalize`` ``"imagenet"``, ``output`` ``"relative_disparity"``) and has a row in every table of
``predictor.TASK_TABLES`` and in ``cli.DATA_LISTERS`` and ``cli.PREDICT_WRITERS``; ``load_model`` of a trained tiny run
returns a ``DepthPredictor``; an image of another size, a grey and an RGBA image give a ``(H, W)`` float32 disparity at
the image's own size; the model is fed what training feeds it (a numpy oracle of the ImageNet standardization, and the
loader's own crop of a file at the trained size); a moved ``.keras``, a record of another family or variant and a
record whose size is not the model's are named ``ConfigError``s; ``python -m train predict`` writes one ``.npy`` per
image and ``index.json`` last, and refuses an existing ``out``. Every export cell is refused until its cells are
measured (the second part, Step 10).

One vit_s is trained here, once, in a module fixture: from random weights at 28 px (a 2 x 2 patch grid), one epoch of
one batch on the synthetic MegaDepth tree, TensorBoard, figures, the analysis, the final model and the warm-up off
(none of them is what this file checks). The predictor is ever only asked to resize to 28 px and back: the model is
never rebuilt at another size (D-009, D-018, D-020).
"""

import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import dl_techniques
import train
from train.api import cli, predictor, recipes, registry
from train.api.errors import ConfigError
from train.api.predictor import (
    RECORD_KEYS,
    TASK_TABLES,
    check_record,
    input_shape_of,
    load_model,
    record_differences,
    record_problems,
)

TASK = "depth_estimation"
SIZE = 28
TINY = {"train_split": 0.6, "n_select": 4, "n_test": 4, "layout_samples": 4, "steps_per_epoch": 1, "workers": 0}
#: The ImageNet statistics typed here, as the oracle's own (the library's constants are held equal to them below).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WT_SRC = str(Path(train.__file__).resolve().parents[1])


def dp():
    """The module under test, imported per test so each test fails on its own until it exists."""
    return importlib.import_module("train.api.depth_predictor")


@pytest.fixture(scope="module")
def world():
    """An empty registry holding every built-in family and task."""
    from train.api.tasks import register_builtin_tasks

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


@pytest.fixture(scope="module")
def megadepth_tree(tmp_path_factory):
    from tests.test_train.test_common_depth_data import write_megadepth_tree

    return write_megadepth_tree(tmp_path_factory.mktemp("megadepth"))


@pytest.fixture(scope="module")
def trained(world, megadepth_tree, tmp_path_factory):
    """The run directory of one trained tiny vit_s (see the module docstring)."""
    from train.api.tasks.depth_estimation import train_depth_estimation

    run = tmp_path_factory.mktemp("depth_predict") / "run"
    train_depth_estimation(
        out=str(run), model="depth_anything/vit_s", data={"root": str(megadepth_tree), **TINY}, epochs=1, batch_size=4,
        model_args={"pretrained": False}, transform_args={"image_size": SIZE},
        analysis_args={"model_analysis": False, "figures": False}, logger_args={"tensorboard": None},
        checkpoint_args={"save_last": False}, optim_args={"warmup_epochs": 0},
    )
    return run


@pytest.fixture(scope="module")
def loaded(trained):
    return load_model(trained)


def read_json(path):
    return json.loads(Path(path).read_text())


def depth_record(**changes):
    record = predictor.build_depth_record(model="depth_anything/vit_s", family="depth_anything", variant="vit_s",
                                          input_size=(SIZE, SIZE), channels=3)
    record.update(changes)
    return record


def rgb(height, width, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


class Spy:
    """The model, recording every input it is called with."""

    def __init__(self, model):
        self.model, self.inputs = model, []

    def __call__(self, batch, training=None):
        self.inputs.append(np.asarray(batch))
        return self.model(batch, training=training)


def test_the_evidence_imports_the_worktree():
    assert dl_techniques.__file__.startswith(WT_SRC) and train.__file__.startswith(WT_SRC)


# ---------------------------------------------------------------------
# The record and the tables
# ---------------------------------------------------------------------


def test_the_record_has_its_own_keys_and_no_class_names_and_no_statistics():
    record = depth_record()
    assert list(record) == list(RECORD_KEYS[TASK]) == [
        "schema_version", "task", "model", "family", "variant", "input_size", "channels", "normalize", "output",
    ]
    assert (record["task"], record["input_size"], record["channels"], record["normalize"], record["output"]) == (
        TASK, [SIZE, SIZE], 3, "imagenet", "relative_disparity")
    assert record_problems(record) == []


def test_a_trained_run_writes_the_depth_record(trained):
    record = read_json(trained / "task.json")
    assert record == depth_record()
    check_record(record, trained / "task.json")


@pytest.mark.parametrize("name", sorted(TASK_TABLES))
def test_every_table_has_a_depth_row(name):
    assert TASK in TASK_TABLES[name], f"{name} has no {TASK} row: record_problems refuses the record naming {name}'s tasks"


def test_predict_lists_images_for_a_depth_run_and_has_its_writer():
    assert cli.DATA_LISTERS[TASK] is cli._image_files
    assert TASK in cli.PREDICT_WRITERS


@pytest.mark.parametrize("name", sorted(TASK_TABLES))
def test_a_depth_record_missing_from_a_table_is_refused_naming_the_tasks_that_table_knows(monkeypatch, name):
    record = depth_record()  # built first: the builder lays the keys out by RECORD_KEYS
    monkeypatch.delitem(TASK_TABLES[name], TASK)
    knows = ", ".join(TASK_TABLES[name])
    assert record_problems(record) == [f"'task' is {TASK!r}; this reader knows {knows}"]


@pytest.mark.parametrize("changes, message", [
    ({"input_size": [SIZE]}, r"'input_size' must be \[height, width\] of positive ints, got \[28\]"),
    ({"input_size": [0, SIZE]}, r"'input_size' must be \[height, width\] of positive ints"),
    ({"input_size": "28"}, r"'input_size' must be \[height, width\]"),
    ({"channels": 1}, r"'channels' must be 3 \(the model takes RGB\), got 1"),
    ({"channels": True}, r"'channels' must be 3"),
    ({"normalize": "standardize"}, r"'normalize' must be 'imagenet'"),
    ({"output": "masks"}, r"'output' must be one of relative_disparity, got 'masks'"),
    ({"class_names": ["a"]}, r"unknown key\(s\) 'class_names'"),
])
def test_a_bad_depth_record_is_refused_naming_the_key(changes, message):
    with pytest.raises(ConfigError, match=message):
        check_record(depth_record(**changes), "task.json")


def test_a_resume_compares_the_size_the_channels_and_the_normalization_and_not_the_output():
    earlier = depth_record()
    assert record_differences(earlier, depth_record()) == []
    assert record_differences(earlier, depth_record(input_size=[56, 56])) == ["input_size: [28, 28] -> [56, 56]"]
    assert predictor.DATA_KEYS[TASK] == ("input_size", "channels", "normalize")


# ---------------------------------------------------------------------
# load_model and DepthPredictor
# ---------------------------------------------------------------------


def test_load_model_returns_a_depth_predictor_for_a_trained_run(trained, loaded):
    assert type(loaded) is dp().DepthPredictor
    assert loaded.record == read_json(trained / "task.json")
    assert input_shape_of(loaded.record) == (SIZE, SIZE, 3)


def test_another_size_gives_float32_disparity_at_the_images_own_size(loaded, tmp_path):
    path = tmp_path / "wide.png"
    Image.fromarray(rgb(30, 50, seed=3)).save(path)
    items = [rgb(40, 61, seed=1), Image.fromarray(rgb(17, 33, seed=2)), path, str(path)]
    answer = loaded.predict(items)
    assert list(answer) == ["disparity"]
    assert [a.shape for a in answer["disparity"]] == [(40, 61), (17, 33), (30, 50), (30, 50)]
    for one in answer["disparity"]:
        assert one.dtype == np.float32 and np.all(np.isfinite(one))
    single = loaded.predict(rgb(40, 61, seed=1))["disparity"]
    assert len(single) == 1, "a single image still gives a list of one map"
    # The same image alone, or at another row of the batch, gives float32 sums that differ in the last places (measured
    # 2.7e-5 and 2.3e-5 on maps of about 20, CPU), so it is compared within 1e-5 of the map's scale. The same batch
    # twice is bit-equal (the command's test below holds that).
    for got, wanted in ((single[0], answer["disparity"][0]), (answer["disparity"][3], answer["disparity"][2])):
        np.testing.assert_allclose(got, wanted, rtol=0, atol=1e-5 * np.abs(wanted).max())


def test_a_grey_and_an_rgba_image_are_read_as_their_rgb(loaded, tmp_path):
    grey = np.random.default_rng(4).integers(0, 256, (37, 45), dtype=np.uint8)
    colour = rgb(26, 19, seed=5)
    rgba = np.concatenate([colour, np.full((26, 19, 1), 77, np.uint8)], axis=-1)
    Image.fromarray(grey).save(tmp_path / "grey.png")
    Image.fromarray(rgba).save(tmp_path / "rgba.png")
    as_rgb = loaded.predict([np.repeat(grey[..., None], 3, axis=-1), colour])["disparity"]
    for given in ([Image.fromarray(grey), Image.fromarray(rgba)], [tmp_path / "grey.png", tmp_path / "rgba.png"],
                  [grey, rgba]):
        answer = loaded.predict(given)["disparity"]
        assert [a.shape for a in answer] == [(37, 45), (26, 19)] and all(a.dtype == np.float32 for a in answer)
        for got, wanted in zip(answer, as_rgb):
            np.testing.assert_array_equal(got, wanted)


def oracle_input(image):
    """The model input of one uint8 RGB image, from numpy alone: bilinear to the trained size, then ImageNet."""
    resized = np.asarray(Image.fromarray(image).resize((SIZE, SIZE), Image.BILINEAR), np.float64)
    unit = (resized / 127.5 - 1.0 + 1.0) * 0.5
    return ((unit - np.asarray(MEAN)) / np.asarray(STD)).astype(np.float32)


def test_the_oracle_holds_the_librarys_imagenet_statistics():
    from dl_techniques.datasets.vision.imagenet2012 import IMAGENET_MEAN, IMAGENET_STD

    assert tuple(IMAGENET_MEAN) == MEAN and tuple(IMAGENET_STD) == STD


def test_the_model_is_fed_the_imagenet_standardization_of_the_image_resized_to_the_trained_size(loaded, monkeypatch):
    images = [rgb(45, 37, seed=6), rgb(SIZE, SIZE, seed=7)]
    spy = Spy(loaded.model)
    monkeypatch.setattr(loaded, "model", spy)
    loaded.predict(images)
    (fed,) = spy.inputs
    assert fed.shape == (2, SIZE, SIZE, 3) and fed.dtype == np.float32
    for got, image in zip(fed, images):
        np.testing.assert_allclose(got, oracle_input(image), rtol=0, atol=2e-6)


def test_an_image_at_the_trained_size_is_fed_exactly_what_the_loader_trains_on(loaded, tmp_path, monkeypatch):
    import h5py

    from train.common.megadepth import load_and_process_pair, standardize_imagenet

    path, depth = tmp_path / "photo.jpg", tmp_path / "photo.h5"
    Image.fromarray(rgb(SIZE, SIZE, seed=8)).save(path, quality=90)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    with h5py.File(depth, "w") as handle:
        handle.create_dataset("depth", data=1.0 + 0.05 * yy + 0.03 * xx)
    # At the trained size the loader's crop is the whole image (one position), so this is a training input exactly.
    crop, _ = load_and_process_pair(str(path), str(depth), SIZE, augment=False, rng=np.random.default_rng(0))
    spy = Spy(loaded.model)
    monkeypatch.setattr(loaded, "model", spy)
    loaded.predict(path)
    np.testing.assert_array_equal(spy.inputs[0][0], standardize_imagenet(crop))


def test_the_answer_is_the_models_own_map_resized_bilinearly_back(loaded):
    # The input is the predictor's own (the oracle above holds it); the same batch, so the maps are bit-equal.
    at_size, other = rgb(SIZE, SIZE, seed=9), rgb(33, 50, seed=10)
    answer = loaded.predict([at_size, other])["disparity"]
    maps = np.asarray(loaded.model(loaded.model_input([at_size, other]), training=False))
    assert maps.shape == (2, SIZE, SIZE, 1)
    np.testing.assert_array_equal(answer[0], maps[0, ..., 0])
    wanted = np.asarray(Image.fromarray(maps[1, ..., 0]).resize((50, 33), Image.BILINEAR), np.float32)
    np.testing.assert_array_equal(answer[1], wanted)


def test_bad_inputs_are_one_readable_refusal_each(loaded, tmp_path):
    (tmp_path / "not.png").write_bytes(b"not an image")
    with pytest.raises(ConfigError, match="the list of images is empty"):
        loaded.predict([])
    with pytest.raises(ConfigError, match="is not a file"):
        loaded.predict(tmp_path / "missing.png")
    with pytest.raises(ConfigError, match="cannot be decoded"):
        loaded.predict(tmp_path / "not.png")
    with pytest.raises(ConfigError, match="must be uint8 or floating point"):
        loaded.predict(np.zeros((8, 8, 3), np.int32))


def test_the_predictor_refuses_a_record_of_another_task(loaded):
    from tests.test_train.test_api.test_task_json_schema_v2 import segmentation

    with pytest.raises(ConfigError, match=r"serves 'depth_estimation' only"):
        dp().DepthPredictor(loaded.model, segmentation())


# ---------------------------------------------------------------------
# What is not one run, refused at load
# ---------------------------------------------------------------------


def place(trained, tmp_path, name, **changes):
    directory = tmp_path / name
    directory.mkdir()
    shutil.copyfile(trained / "best_model.keras", directory / "best_model.keras")
    (directory / "task.json").write_text(json.dumps({**read_json(trained / "task.json"), **changes}))
    return directory


def test_a_moved_keras_file_is_refused_naming_the_records_it_looks_for(trained, tmp_path):
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copyfile(trained / "best_model.keras", moved / "best_model.keras")
    with pytest.raises(ConfigError, match=r"has no record beside it; .* best_model\.keras\.task\.json"):
        load_model(moved / "best_model.keras")


def test_a_record_of_another_family_or_variant_is_refused(trained, tmp_path):
    with pytest.raises(ConfigError, match=r"the family 'convunext' trains image_segmentation, not 'depth_estimation'"):
        load_model(place(trained, tmp_path, "family", family="convunext", variant="tiny", model="convunext/tiny"))
    with pytest.raises(ConfigError, match=r"names the variant 'vit_b' but .* is not that variant \(encoder_type"):
        load_model(place(trained, tmp_path, "variant", variant="vit_b", model="depth_anything/vit_b"))


@pytest.mark.parametrize("size", [[56, 56], [30, 30], [SIZE, 42]])
def test_a_record_whose_size_is_not_the_models_is_refused_before_the_model_is_called(trained, tmp_path, size):
    # [30, 30] is not a multiple of the patch: the model would take a zero image of it (2 x 2 patches) and answer at
    # 30 x 30, so only the check of the BUILT size catches it.
    with pytest.raises(ConfigError, match=rf"names input_size \[{size[0]}, {size[1]}\] and 3 channels but .* is built at "
                                          rf"\[{SIZE}, {SIZE}, 3\]"):
        load_model(place(trained, tmp_path, "size", input_size=size))


# ---------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------


def folder_of(tmp_path, shapes):
    """``{relative name: (height, width) or "grey"}`` written as images under ``tmp_path/photos``."""
    root = tmp_path / "photos"
    paths = []
    for index, (name, shape) in enumerate(shapes.items()):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(rgb(*shape, seed=20 + index)).convert("L" if index == 2 else "RGB")
        image.save(path)
        paths.append(path)
    return root, sorted(paths)


def test_the_command_writes_an_npy_per_image_at_its_size_and_the_index(trained, loaded, tmp_path, capsys):
    root, paths = folder_of(tmp_path, {"b.png": (30, 40), "sub/a.jpg": (SIZE, SIZE), "c.png": (20, 50)})
    out = tmp_path / "answers"
    assert cli.main(["predict", f"model={trained}", f"data={root}", f"out={out}"]) == 0
    assert capsys.readouterr().out.strip() == str(out)
    assert sorted(p.name for p in out.iterdir()) == ["a.npy", "b.npy", "c.npy", "index.json"]
    index = read_json(out / "index.json")
    assert index["output"] == "relative_disparity" and "scale and shift" in index["note"]
    assert [e["path"] for e in index["disparity"]] == [str(p) for p in paths]
    expected = loaded.predict(paths)["disparity"]
    for entry, wanted in zip(index["disparity"], expected):
        written = np.load(out / entry["file"])
        assert written.dtype == np.float32 and list(written.shape) == entry["size"]
        np.testing.assert_array_equal(written, wanted)


def test_an_existing_out_is_refused_and_left_alone(trained, tmp_path, capsys):
    root, _ = folder_of(tmp_path, {"a.png": (20, 20)})
    out = tmp_path / "answers"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    assert cli.main(["predict", f"model={trained}", f"data={root}", f"out={out}"]) == 2
    assert "already exists" in capsys.readouterr().err
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_two_images_with_one_stem_are_refused_before_the_model_is_loaded(trained, tmp_path, monkeypatch, capsys):
    root, _ = folder_of(tmp_path, {"a/cat.png": (20, 20), "b/cat.jpg": (20, 20)})
    monkeypatch.setattr(predictor, "load_model", lambda *_: (_ for _ in ()).throw(AssertionError("the model was loaded")))
    out = tmp_path / "answers"
    assert cli.main(["predict", f"model={trained}", f"data={root}", f"out={out}"]) == 2
    said = capsys.readouterr().err
    assert "have the same name without its extension" in said and "cat.npy" in said
    assert not out.exists()


def test_a_run_that_fails_half_way_leaves_its_arrays_and_no_index(loaded, tmp_path):
    chunk = dp().CHUNK
    files = []
    for index in range(chunk):
        files.append(tmp_path / f"g{index}.png")
        Image.fromarray(rgb(20, 24, seed=40 + index)).save(files[-1])
    files.append(tmp_path / "z.png")
    files[-1].write_bytes(b"not an image")
    out = tmp_path / "answers"
    with pytest.raises(ConfigError, match="cannot be decoded"):
        dp().write_predictions(loaded, files, out)
    assert sorted(p.name for p in out.iterdir()) == sorted(f"g{i}.npy" for i in range(chunk))
    assert not (out / "index.json").exists(), "the index is the mark of a complete answer"


# ---------------------------------------------------------------------
# Export: every cell refused until the cells are measured (Step 10)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("format", ["keras", "saved_model", "tflite", "onnx"])
def test_every_export_cell_is_refused_until_its_cells_are_measured(format):
    rule = predictor.EXPORT_RULES[TASK]
    record = depth_record()
    assert rule.shape_of(record) == (SIZE, SIZE, 3) and rule.served is None
    with pytest.raises(ConfigError, match=rf"{TASK}.*{format}"):
        rule.check(record, format, (SIZE, SIZE, 3))
    note = rule.note(record)
    assert "relative disparity" in note and "scale and shift" in note and "[28, 28]" in note
